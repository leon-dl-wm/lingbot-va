# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LingBot-VA training entry point: autoregressive video-action joint training with
Diffusion Forcing + flow matching.

Position in the architecture: this file is the training-side "conductor" — it pulls
hyperparameters from ``configs``, reads offline VAE latents and actions via
``dataset.MultiLatentLeRobotDataset``, runs one forward pass through
``modules.model.WanTransformer3DModel.forward_train``, and trains on multiple GPUs via
``distributed`` (FSDP + mixed precision).

Core mechanisms (mapping to the paper):
1. **Training sequence layout**: inside the model, ``[noisy_video | clean_video |
   noisy_action | clean_action]`` are concatenated into one large sequence (the batch is
   flattened into the sequence, samples are isolated by seq_ids, padded to a multiple of
   128); combined with the block-causal mask this realizes "causal AR + bidirectional
   denoising within a chunk".
2. **Diffusion Forcing noise injection** (``Trainer._add_noise``): every latent frame
   samples its own timestep independently, with sigma broadcast along the frame dimension;
   ``noisy_cond_prob=0.5`` also noises the clean condition segment for half of the steps,
   simulating the fact that at inference time the KV cache holds error-carrying predicted
   frames (the key trick for long-horizon AR stability).
3. **Randomized chunk/window during training** (``_prepare_input_dict``): chunk_size~U{1..4}
   and window_size~U{4..64} are resampled every step ⇒ a single training run supports any
   chunk/window configuration at inference (deployment uses frame_chunk_size=2,
   attn_window=72).
4. **Loss** (``compute_loss``): video/action velocity (ε-x₀) MSE summed with equal weights,
   each timestep multiplied by a bell-shaped training weight, normalized per frame; the
   action loss is multiplied by actions_mask so only valid channels count.

Engineering: FSDP sharding (bf16) + gradient accumulation + grad clip 2.0 + periodic
save_checkpoint; attn_mode is ``flex`` for training (FlexAttention + torch.compile).

Launch: ``torchrun ... train.py --config-name robotwin_train`` (see
``script/run_va_posttrain.sh``); entry chain ``main()`` → ``run()`` → ``Trainer.train()``.
"""
import argparse
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc


class Trainer:
    """LingBot-VA trainer: encapsulates model loading / FSDP sharding, noise injection,
    loss computation, the training loop, and checkpointing.

    Responsibilities and key members:
    - ``transformer``: WanTransformer3DModel (FSDP-sharded + activation checkpointing,
      attn_mode='flex');
    - ``train_scheduler_latent`` / ``train_scheduler_action``: independent FlowMatchScheduler
      instances for video and action (decoupled SNR shifts: video shift=5.0 / action shift=1.0);
      ``set_timesteps(1000, training=True)`` also builds the bell-shaped training weights;
    - ``_add_noise``: Diffusion Forcing per-frame independent noise injection (the core);
    - ``_prepare_input_dict``: assembles the forward_train input and randomizes chunk/window;
    - ``compute_loss``: video + action velocity MSE;
    - ``train``: step-driven (not epoch-driven) main loop with gradient accumulation and logging.

    Args:
        config: task configuration object (from ``VA_CONFIGS[config_name]``, with
            rank/local_rank/world_size injected in run()); contains all hyperparameters
            such as learning rate, batch_size, snr_shift, save_root, num_steps.
    """
    def __init__(self, config):
        # rank 0 handles wandb reporting (other ranks only train, no logging)
        if config.enable_wandb and config.rank == 0:
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name='test_lln'
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            # Resume training: restore weights from the checkpoint directory
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            # Cold start: initialize from the Wan2.2 pretrained weights
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        # Load to CPU in fp32 (FSDP casts to bf16 for compute per MixedPrecisionPolicy when
        # sharding); attn_mode='flex': training uses FlexAttention + torch.compile, which
        # supports arbitrary block-causal masks
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex"
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        # fused AdamW: single-kernel parameter update, faster for a large model with long sequences
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        # Linear warmup followed by a constant learning rate
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        # Aggregated latent dataset over multiple LeRobot repos (videos pre-encoded offline by the Wan2.2 VAE)
        train_dataset = MultiLatentLeRobotDataset(config=config)
        # Multi-GPU: DistributedSampler splits samples across ranks; single GPU: plain shuffle
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        # Video and action use two independent flow-matching schedulers (decoupled SNR shifts:
        # snr_shift=5.0 pushes the video noise schedule toward the high-noise end, while
        # action_snr_shift=1.0 applies no shift); with training=True, set_timesteps also
        # generates bell-shaped (Gaussian bsmntw) training weights used by compute_loss to
        # weight each timestep (mid noise levels get the largest weight).
        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        # Fetch the next batch from the DataLoader iterator; when the epoch is exhausted
        # (StopIteration), automatically bump the sampler's epoch (so multi-GPU shuffling
        # differs every epoch) and rebuild the iterator, letting the main loop train by
        # step count (instead of epochs).
        # Returns: a batch dict with latents [B,C,F,H,W], actions [B,30,F,16,1],
        # actions_mask [B,30,F,16,1], text_emb, etc. (see the dataset's __getitem__).
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        """Diffusion Forcing core: noise a video/action tensor with per-frame independent
        timesteps, and build the condition segment plus the RoPE grid ids.

        Unlike standard diffusion training (the whole sample shares one timestep), here
        **every latent frame samples its own timestep independently**
        (``sample_timestep_id(batch_size=F)``), and ``add_noise(t_dim=2)`` broadcasts sigma
        along the frame dimension: x_t = (1-sigma_f)*x0 + sigma_f*eps. This way different
        frames in the same sequence can sit at different noise levels; combined with the
        model's block-causal mask (noisy tokens may only attend to earlier clean frames),
        this trains the autoregressive ability of "denoising the current chunk conditioned
        on noisy history" — the essence of Diffusion Forcing.

        With probability ``noisy_cond_prob`` the clean condition segment is also noised
        (t sampled from the high-noise band [0.5, 1]): at inference time the KV cache holds
        the model's own error-carrying "imagined" frames, so making the condition segment
        dirty during training teaches the model to work on dirty history and suppresses
        error accumulation over long AR rollouts (only video uses 0.5; the action condition
        segment is always clean because actions come from real robot execution feedback).

        Args:
            latent: tensor to noise. Video: VAE latents [B,48,F,H,W]; action:
                [B,30,F,16,1] (30 = unified action-space dim, F = number of latent frames,
                16 = action sub-steps per frame, i.e. action_per_frame).
            train_scheduler: FlowMatchScheduler instance (one each for video/action, with
                different SNR shifts).
            action_mask: valid-channel mask for actions [B,30,F,16,1] (bool); None/False
                means no masking. Invalid channels (slots of the 30-dim unified space that
                this robot does not use) are zeroed after noising.
            action_mode: True when processing the action tensor — the patch size is treated
                as 1 (actions are not spatio-temporally patchified) and RoPE uses fractional
                time positions (the 16 sub-steps are inserted between adjacent video frames).
            noisy_cond_prob: probability that the condition segment gets noised
                (video=0.5, action=0.0).

        Returns:
            dict containing:
            - timesteps [B,F]: per-frame timestep of the noisy segment (repeated over batch);
            - noisy_latents: noised result x_t, same shape as latent;
            - targets: flow-matching regression target eps - x0 (velocity);
            - latent: condition-segment tensor (noised with probability noisy_cond_prob,
              otherwise kept clean);
            - cond_timesteps [B,F]: timesteps of the condition segment (all zeros when clean);
            - grid_id [B,4,seq_len]: (f,h,w,t) grid coordinates for RoPE; in action mode f
              holds fractional positions and h/w = -1 (no spatial position).
        """
        B, C, F, H, W = latent.shape

        # Sample one timestep per frame independently: [F] ids — the key line of diffusion forcing
        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        # t_dim=2: sigma is broadcast along the frame dim (F); each frame applies its own
        # sigma_f via x_t = (1-sigma_f)*x0 + sigma_f*eps
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        # Flow-matching regression target: velocity = eps - x0
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            # Action tokens are not patchified (each sub-step is one token); the grid is
            # expanded over the raw dimensions
            patch_f = patch_h = patch_w = 1
        
        # Build the RoPE position grid: with action=True the 16 sub-steps take fractional
        # time positions (f+1/17..16/17), precisely inserted between the two neighboring
        # video frames — the literal implementation of the interleaved v/a sequence;
        # h/w are set to -1 because actions have no spatial position
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        # With probability noisy_cond_prob, also noise the clean condition segment: simulates
        # the inference-time KV cache holding error-carrying predicted frames (dirty history),
        # teaching the model to work on dirty conditions.
        # t is sampled only from the high-noise band [0.5, 1]: the condition segment is either
        # clean or "very dirty", matching the reality that imagined frames carry sizable errors
        # at inference time.
        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            # Condition segment stays clean: timesteps are all 0 (sigma=0), and the model
            # side treats these tokens as clean
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            # In the 30-dim unified action space, force invalid channels (slots this robot
            # does not use) to zero: they neither contribute to the loss (see compute_loss)
            # nor leak meaningless noise as conditioning
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Assemble one data batch into the input required by transformer.forward_train:
        # apply Diffusion Forcing noise to the video latents and the actions separately
        # (_add_noise), and sample chunk_size / window_size **randomly at every step**.
        #
        # Motivation for randomization: forward_train splits the sequence into AR chunks by
        # chunk_size, applies sliding-window attention by window_size, and builds the
        # corresponding block-causal mask. Resampling chunk_size~U{1..4} and
        # window_size~U{4..64} every step ⇒ a single training run covers all chunk/window
        # combinations, so inference can use any configuration (deployment uses
        # frame_chunk_size=2, attn_window=72) without per-configuration training.
        #
        # Args: batch_dict — DataLoader output with latents [B,48,F,H,W],
        #       actions [B,30,F,16,1], actions_mask [B,30,F,16,1], text_emb.
        # Returns: dict {'latent_dict', 'action_dict', 'chunk_size', 'window_size'}, where
        #       latent_dict/action_dict are _add_noise outputs (with text_emb etc. attached).
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        # Video path: the condition segment is noised with 50% probability (dirty-history
        # training — the key trick for long-horizon AR stability)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        # Action path: independent scheduler (shift=1.0); the condition segment is always
        # clean (noisy_cond_prob=0), and actions_mask zeroes out the invalid channels of the
        # 30-dim unified space
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            # Resample chunk_size∈{1,2,3,4} and window_size∈[4,64] every step:
            # one training run supports any chunk/window configuration at inference
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        # Move every tensor of the batch to the current rank's GPU; precision casting is left
        # to FSDP's MixedPrecisionPolicy (bf16), so no dtype cast happens here (see the
        # commented-out .to at the line end).
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        """Compute the video + action two-way flow-matching velocity MSE loss (summed with equal weights).

        The model's forward_train outputs flattened token sequences; they are first restored
        to the same tensor layout as the targets, then compared element-wise with MSE.
        Key details:
        - every timestep is multiplied by a bell-shaped training weight (the Gaussian
          bsmntw_weighing from the scheduler's training mode: mid noise levels get the
          largest weight, both ends are down-weighted);
        - **per-frame normalization**: each frame's loss is summed, divided by that frame's
          number of valid elements, and finally averaged over all frames — this prevents
          element-rich frames (video frames are far larger than action frames) from
          dominating the gradient;
        - the action loss is multiplied by actions_mask so only valid channels count
          (robotwin uses only 16 of the 30 unified action-space dims: dual-arm EEF 7 +
          gripper 1); invalid slots produce no loss.

        Args:
            input_dict: output of _prepare_input_dict (holds both paths'
                targets/timesteps/actions_mask).
            pred: (latent_pred, action_pred) model outputs —
                latent_pred [B, seq_len, C_patch] is the patch-token sequence,
                action_pred [B, F*16, 30] is the action-token sequence.

        Returns:
            (latent_loss, action_loss): two scalars, both already divided by
            gradient_accumulation_steps (so accumulating G backwards is equivalent to the
            mean loss of a large batch); the caller sums them directly and calls backward.
        """
        latent_pred, action_pred = pred
        # Restore the action prediction sequence to [B,30,F,16,1] (same layout as targets,
        # F = number of latent frames)
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        # Restore the video prediction patch sequence to [B,48,F,H,W]: data_seq_to_patch is
        # the inverse of the model's patchify (reorders tokens by patch_size=(1,2,2))
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        # Look up the bell-shaped training weight [Bn,Fn] by each frame's timestep: frames
        # at different noise levels contribute differently; mid noise levels (the most
        # informative) get the highest weight
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        # velocity MSE: compare pred against target = eps - x0 element-wise (computed in fp32
        # for numerical stability)
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        # Broadcast the per-frame weight over (B,C,F,H,W): weight[:, None, :, None, None]
        # varies only along the F dimension
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        # Per-frame normalization: each frame's sum / that frame's element count, then mean
        # over all frames (video frames are fully valid, so the denominator is H*W*C; this
        # mirrors the mask-based normalization of the action path)
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        # Count only valid action channels: slots with mask=False (unused dims of the unified
        # space) get their loss zeroed
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        # Per-frame normalization: the denominator is the frame's number of valid elements
        # (mask sum), keeping the action loss scale consistent across robots with different
        # numbers of valid channels
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        # Divide by the number of gradient-accumulation steps: after G accumulated backwards
        # the total equals the mean loss of the effective large batch
        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        # Train a single micro-batch: move data to GPU → Diffusion Forcing noising with
        # randomized chunk/window → forward_train (internally concatenates the
        # [noisy_video|clean_video|noisy_action|clean_action] large sequence + block-causal
        # mask) → sum the two losses and backward.
        # Gradient accumulation: only at the accumulation boundary ((batch_idx+1) divisible
        # by accumulation_steps) enable FSDP gradient sync, do grad clip (2.0), and run
        # optimizer.step.
        #
        # Args:
        #     batch: batch dict from the DataLoader.
        #     batch_idx: index within the current gradient-accumulation cycle (0-based);
        #         decides whether to sync/update.
        # Returns:
        #     dict: {'latent_loss', 'action_loss'} detached scalars plus should_log
        #     (whether the accumulation boundary was reached); at the boundary it also
        #     contains 'total_norm' (the gradient norm before clipping).
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        # Disable FSDP gradient reduce-scatter when not at the accumulation boundary, saving
        # communication (gradients stay local and accumulate)
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        # Sum the video/action velocity losses with equal weights
        loss = latent_loss + action_loss

        loss.backward()

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        
        # Only update weights after accumulating gradients
        if should_sync:
            # Clip gradients (max_norm=2.0), then update the weights and zero the gradients
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        # Gather the FSDP-sharded weights into a full state dict, cast to bf16, and let rank 0
        # save it in diffusers format (checkpoint_step_N/transformer/ contains
        # diffusion_pytorch_model.safetensors + config.json), identical to the pretrained-model
        # directory layout ⇒ the inference server (wan_va_server.py) can load it directly with
        # no format conversion.
        #
        # Note: get_model_state_dict is a collective operation — **all ranks must call it**
        # (even though only rank 0 writes to disk); the error path must also reach the final
        # barrier to avoid deadlocks.
        try:
            # On unified-memory platforms (e.g. DGX Spark), cached GPU blocks
            # count against system RAM; release them before the CPU-side
            # full-state-dict gather.
            gc.collect()
            torch.cuda.empty_cache()
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            # Convert incrementally, freeing each fp32 tensor as it is cast:
            # holding both full dicts at once (~30GB) can OOM the host on
            # unified-memory machines (e.g. DGX Spark).
            state_dict_bf16 = {}
            for k in list(state_dict.keys()):
                state_dict_bf16[k] = state_dict.pop(k).to(torch.bfloat16)
            del state_dict
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        # Resume training: restore the optimizer state and the global step from
        # training_state.pt. Must be called after FSDP sharding and optimizer construction
        # (set_optimizer_state_dict needs the sharded model/optimizer to map the state);
        # all ranks must load it (FSDP requires each shard to hold its own optimizer state).
        # If the file is missing, only warn and start from step 0.
        #
        # Args: checkpoint_path — the checkpoint directory (containing training_state.pt).
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        # Step-driven (not epoch-driven) main loop: each step takes one micro-batch and runs
        # _train_step; at the gradient-accumulation boundary (should_log=True) it aggregates
        # losses across ranks (dist_mean/dist_max), updates the progress bar and wandb,
        # increments self.step, and periodically saves checkpoints per save_interval.
        # Epoch exhaustion is handled automatically by _get_next_batch; the loop only ends
        # when num_steps is reached. A barrier at the end of each step keeps ranks in sync.
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        # The progress bar is displayed on rank 0 only
        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                # Cross-rank aggregation: sum first combines the losses of the micro-batches
                # within the accumulation cycle, then all-reduce computes the global mean/max
                # (the max helps spot anomalous ranks/samples)
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                # Periodically release cached VRAM + GC to mitigate memory fragmentation in long runs
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                
                self.step += 1
                
                # Periodic checkpointing (all ranks must enter save_checkpoint: it contains collective ops)
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    # Training entry point: fetch the config from VA_CONFIGS by config_name, read the
    # environment variables injected by torchrun (RANK/LOCAL_RANK/WORLD_SIZE) to initialize
    # the NCCL process group, write the distributed info back into the config, then build
    # the Trainer and start the training loop.
    #
    # Args: args — command-line arguments (--config-name selects the task config,
    #     --save-root overrides the checkpoint root directory).
    config = VA_CONFIGS[args.config_name]

    # torchrun injects distributed info via environment variables (single-node falls back to rank0/world1)
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    # Parse command-line arguments and call run:
    # --config-name selects the task config in VA_CONFIGS (default robotwin_train; also
    # libero_train etc.); --save-root optionally overrides the checkpoint root directory.
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()