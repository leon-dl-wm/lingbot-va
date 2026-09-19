# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LingBot-VA inference server core: autoregressive (AR) rollout of world modeling + action generation on top of the Wan2.2 video DiT.

This file is the "brain" of the deployment side, implementing the paper's asynchronous
execution protocol (three-way dispatch, see ``infer``):

    reset(prompt) initializes caches / encodes text -> compute_kv_cache(real obs) commits
    real history -> infer() returns 32 action sub-steps -> robot executes them
    -> client sends back the new obs -> loop

Key mechanisms:
- ``_infer``: the AR main loop. Each chunk (frame_chunk_size=2 latent frames = 32 action
  sub-steps) runs in two phases -- phase 1 denoises video for 25 steps (+1 padding step at
  t=0) and commits the "fully clean imagined future frames" into the KV cache with
  update_cache=1; phase 2 denoises actions for 50 steps while attention reads the just
  committed imagined-video KV. Since frame_id(action)=2k+1 > 2k(video), actions are
  conditioned on the imagined future frames, i.e. "imagine first, then act".
- KV cache three-state protocol (the update_cache argument): 0 = write then roll back
  (intermediate denoising steps, cache stays unpolluted); 1 = commit imagined frames
  (is_pred=True); 2 = commit real observations (is_pred=False, called by
  ``_compute_kv_cache``). ``clear_pred_cache`` drops all imagined frames whenever a new
  real observation arrives, providing closed-loop error correction.
- Multi-camera T-shape mosaic (env_type='robotwin_tshape', see ``_encode_obs``): cam_high
  at full resolution, the two wrist cameras at half resolution concatenated along width,
  then stacked with cam_high along height into a single (3h/2, w) image fed to the VAE,
  so all three cameras are processed in one stream.
- Action post-processing: the 30-dim unified action space is reduced to the effective
  channels via used_action_channel_ids and de-normalized with q01/q99 quantiles.

The entry point ``main`` supports two modes: i2va (offline local generation of a demo
video) and server (a websocket policy server; the evaluation client and real-robot
deployment share this same interface, see utils/Simple_Remote_Infer/deploy/).
"""
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class VA_Server:
    """Joint video-action inference server.

    Wraps the three Wan2.2 components -- the VAE (video encode/decode), the UMT5 text
    encoder, and the video/action dual-stream DiT transformer -- and implements a
    KV-cached autoregressive rollout on top of them: each ``_infer`` call generates one
    chunk of "imagined future video frames + 32 action sub-steps"; ``_compute_kv_cache``
    discards the imagined frames once a real observation arrives and commits the real
    history into the cache, forming the asynchronous closed-loop execution protocol.

    Attributes:
        transformer: the video/action dual-stream DiT, which internally maintains the KV
            cache pool (a write / roll-back / commit state machine).
        scheduler / action_scheduler: two **decoupled** FlowMatchScheduler instances for
            video and action (video snr_shift=5.0, action snr_shift=1.0), so their noise
            schedules never interfere.
        streaming_vae / streaming_vae_half: streaming VAE wrappers (with causal caches),
            used for full-resolution (cam_high) and half-resolution (wrist cameras)
            encoding respectively.
        frame_st_id: number of latent frames already committed to the KV cache, i.e. the
            temporal start of the next chunk to generate (the frame offset of the RoPE
            grid_id).
    """

    def __init__(self, job_config):
        """Load the VAE / tokenizer / text encoder / transformer and initialize schedulers.

        Args:
            job_config: global config (from configs.VA_CONFIGS) containing model paths,
                snr_shift, action_dim, frame_chunk_size, attn_window, norm_stat, etc.
        """
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        # Video and action use two decoupled flow-matching scheduler instances (different
        # SNR shifts): video shift=5.0 (a high shift pushes sampling density toward the
        # high-noise region, which suits video), action shift=1.0 (no shift).
        # extra_one_step=True: generate one extra step so the last sigma lands exactly on sigma_min.
        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        # First lay out the full 1000 training timesteps; at inference time
        # set_timesteps(N) re-discretizes the schedule
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch"
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            # In T-shape mosaic mode the wrist cameras are encoded at half resolution,
            # which requires a second, independent streaming VAE instance (their causal
            # caches must not be mixed with the full-resolution stream)
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        """Encode prompts into fixed-length embeddings with the UMT5 text encoder.

        Args:
            prompt: a string or a list of strings.
            num_videos_per_prompt: number of copies per prompt (for CFG / multi-sample
                generation).
            max_sequence_length: sequence-length cap for tokenization and output
                embeddings.
            device / dtype: device and precision of the output tensors; default to
                self.device / self.dtype.

        Returns:
            torch.Tensor of shape [B*num_videos_per_prompt, max_sequence_length, D_text];
            positions beyond the real token length are zero-padded.
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        # The text encoder may be offloaded to CPU, so inputs must first be moved to
        # the device it currently lives on
        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        # Strip padding, then re-pad with zeros to max_sequence_length so the batch is rectangular
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO

        Encode the positive / negative prompts into the pair of text embeddings used by CFG.

        Args:
            prompt: the positive prompt (string or list).
            negative_prompt: the negative prompt; when None the empty string "" is used
                (consistent with the unconditional branch trained via cfg_prob, which
                randomly replaces the embedding with an empty one).
            do_classifier_free_guidance: whether to encode the negative branch (True when
                guidance_scale > 1).
            prompt_embeds / negative_prompt_embeds: pre-computed embeddings can be passed
                in to skip encoding.
            max_sequence_length: text sequence-length cap.

        Returns:
            A tuple (prompt_embeds, negative_prompt_embeds), both of shape
            [B, max_sequence_length, D_text]; the latter is None when CFG is disabled.
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize latents with the Wan VAE's per-channel statistics.

        Args:
            latents: VAE output latents, [B, C, F, H, W].
            latents_mean: per-channel means, [C].
            latents_std: per-channel **inverse** std (callers pass 1.0/std), [C].

        Returns:
            Normalized latents with the same shape as the input (computed in float32).
        """
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        """Preprocess environment-side real actions/states into the model's 30-dim unified action tensor.

        Args:
            action: np.ndarray of shape [C_env, F, H] (C_env = raw environment action
                dim, F = number of action sub-steps, H = 1 or a history length).

        Returns:
            torch.Tensor of shape [1, 30, F, H, 1], scattered into the unified action
            space slots and quantile-normalized (q01/q99) to [-1, 1].
        """
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        # Append one all-zero channel at the end of the channel dim: invalid slots in
        # inverse_used_action_channel_ids point at this padding channel, so scattering
        # naturally yields zeros there
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        # Scatter environment action channels into the correct slots of the 30-dim
        # unified action space via the inverse mapping
        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            # q01/q99 quantile normalization to [-1, 1] (identical to the training-time
            # _action_post_process)
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        """Convert normalized model-output actions back to raw environment-executable actions.

        Args:
            action: torch.Tensor of shape [B=1, 30, F, H, W=1] (in normalized space).

        Returns:
            np.ndarray of shape [C_env, F*H], keeping only the effective channels given
            by used_action_channel_ids and de-normalized back to physical units via
            q01/q99.
        """
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            # Inverse of the quantiles normalization applied in preprocess_action
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        # Select the channels this environment actually uses (e.g. 16 dims for robotwin
        # dual arms) out of the 30-dim unified action space
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        """Duplicate inputs along the batch dim into [conditional; unconditional] for classifier-free guidance.

        Batch layout convention: index 0 = positive prompt (conditional branch),
        index 1 = negative prompt (unconditional branch), matching the CFG formula
        pred[1:]+s*(pred[:1]-pred[1:]) used in _infer.

        Args:
            input_dict: a single-branch input dict produced by ``_prepare_latent_input``,
                containing noisy_latents [1,C,F,H,W], grid_id [4,N], timesteps [F].

        Returns:
            The same dict (modified in place): with CFG enabled, noisy_latents becomes
            [2,C,F,H,W] and text_emb becomes [2,L,D]; grid_id/timesteps gain a batch dim.
        """
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        """Assemble the video/action input dicts for the transformer forward pass (with RoPE grid_id and per-frame timesteps).

        Args:
            latent_model_input: video latents (may be None), [1, 48, F, H, W].
            action_model_input: action tensor (may be None),
                [1, 30, F, action_per_frame, 1].
            latent_t / action_t: current denoising timestep (scalar), broadcast into a
                per-frame timestep vector; passing 0 means "clean" input (used by
                compute_kv_cache when committing real observations).
            latent_cond: first-frame conditioning latent (passed only for the first
                chunk, i.e. init_latent encoded from the real image).
            action_cond: first-frame conditioning action (passed only for the first
                chunk; an all-zero placeholder).
            frame_st_id: start frame index of the current chunk within the whole
                sequence; determines the RoPE temporal offset.
            patch_size: the transformer's patchify size (p_t, p_h, p_w).

        Returns:
            dict containing 'latent_res_lst' and/or 'action_res_lst' sub-dicts, each with
            noisy_latents, timesteps [F], grid_id [4, N_token], text_emb.
        """
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                # Per-frame timestep vector: under diffusion forcing each frame may
                # carry a different noise level
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                # RoPE grid id for video tokens: t=0 (modality tag), frames start at frame_st_id
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                # i2va conditioning: replace the first frame with the real-image latent
                # and set its timestep to 0 (treated as a clean frame), aligning with the
                # training mask semantics where "noisy tokens may only attend to clean history"
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                # RoPE grid id for action tokens: t=1 (modality tag); action=True places
                # the 16 sub-steps of a frame at fractional time positions f+k/17,
                # precisely interleaved between the surrounding video frames
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                # The first-frame action of the first chunk is a "history action"
                # placeholder (all zeros); its timestep is set to 0 as a clean condition
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            # Invalid channels of the unified action space are always zeroed
            # (consistent with the training-time actions_mask)
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        """Encode multi-camera RGB observations into video latents (streaming VAE, causal cache kept across chunks).

        Args:
            obs: dict; obs['obs'] is a list of images, each element being
                {cam_key: np.ndarray [H,W,3]} (uint8, 0~255).

        Returns:
            torch.Tensor of shape [1, 48, F_latent, latent_H, latent_W] (normalized
            latents); None when there are no images. In robotwin_tshape mode the three
            cameras are mosaicked into a single (3h/2, w) T-shape image and encoded in
            one stream.
        """
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    # T-shape mosaic: the top cam_high keeps full resolution
                    height_i, width_i = self.height, self.width
                else:
                    # Each wrist camera is downscaled to half resolution; concatenated,
                    # their width exactly matches cam_high's
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            # Stack history frames and resize to the target resolution: [F,H,W,3] -> [1,3,F,H,W]
            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            # Normalize pixels to [-1,1], then encode in two streams: cam_high with the
            # full-resolution VAE, both wrist cameras with the half-resolution VAE
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            # Concatenate the wrist latents along width (-1), then stack with cam_high
            # along height (-2) -> equivalent to the latent of the pixel-domain T-shape
            # mosaic (3h/2, w), so three cameras are handled in a single stream
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            # Plain mode: encode cameras along the batch dim, then concatenate along
            # width into one side-by-side wide image
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        # The VAE outputs (mu, logvar) halves; inference keeps only the mean
        # (deterministic encoding)
        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        """Reset server state: clear KV/VAE caches, rebuild the cache pool per config, encode the prompt.

        Corresponds to the reset(prompt) request of the asynchronous protocol. Afterwards
        either compute_kv_cache(initial obs) or the first _infer(frame_st_id=0) must write
        the initial frames before the rollout can start.

        Args:
            prompt: task text description; when None no text encoding is done (pure
                unconditional generation).
        """
        logger.info('Reset.')
        # CFG is enabled if either guidance_scale > 1 (video or action branch)
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            # T-shape mosaic (3h/2, w): latent height = 3h/2 divided by the VAE
            # downsampling factor 16, latent width = w/16
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            # Side-by-side mosaic: height unchanged, width = per-camera width x num cameras
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        # Token counts of one video/action chunk determine the slot size of the KV cache
        # pool; total pool capacity = (attn_window//2) chunks, i.e. a hardware-level
        # sliding window
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        # With CFG, the conditional/unconditional pair shares one pool, hence batch_size=2
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        # Boolean mask of effective channels in the 30-dim unified action space
        # (e.g. 16 dims for robotwin dual arms)
        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        # q01/q99 quantile statistics, reshaped to [C,1,1] to broadcast over [C,F,H] actions
        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            # Text is encoded once here and reused for the whole rollout (with CFG the
            # negative embedding is obtained at the same time)
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0):
        """One step of the AR main loop: generate one chunk of imagined future video frames + action sub-steps.

        Two-phase flow (frame_chunk_size=2 latent frames = 2x16 = 32 action sub-steps):
          Phase 1 video: num_inference_steps (25) denoising steps + 1 padding step at
            t=0. Intermediate steps use update_cache=0 (temporary write, then roll back,
            so the cache stays unpolluted); after each step the first-frame latent is
            clamped back to the real-image latent (i2va conditioning, first chunk only);
            the last step uses update_cache=1 to commit the fully clean "imagined future
            frames" into the KV cache (is_pred=True).
          Phase 2 action: action_num_inference_steps (50) denoising steps + 1 padding
            step, while attention reads the imagined-video KV committed in phase 1.
            Since frame_id(action)=2k+1 > 2k(video), actions are conditioned on the
            imagined future frames = "imagine first, then act".

        Args:
            obs: environment observation dict (used to encode the initial frame only
                when frame_st_id == 0).
            frame_st_id: starting latent frame index of the current chunk (RoPE temporal
                offset).

        Returns:
            (actions, latents): actions is an np.ndarray [C_env, F*H] (de-normalized,
            effective channels only); latents are the imagined future frames
            [1,48,F,h,w] (for decoding / debugging).
        """
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            # First chunk: encode the initial observation as the i2va conditioning frame
            # (used by the clamp below)
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        # Start from pure Gaussian noise and denoise via flow matching:
        # video latents [1,48,F,h,w]
        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        # Action noise [1,30,F,action_per_frame,1]: the 30-dim unified action space
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        # Re-discretize the sigma schedule for each of the decoupled video/action
        # schedulers at its own inference step count
        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        # Append one padding step at t=0: this step no longer updates the sample; its
        # sole purpose is to commit the KV of the "fully clean imagined frames" into the
        # cache with update_cache=1 -- strictly aligned with the training mask semantics
        # where "noisy tokens may only attend to clean history"
        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            # video_exec_step: truncate the video denoising steps for speed (trading
            # imagined-frame quality for latency); after truncation the last step has
            # t != 0 but still runs with update_cache=1, committing the current
            # (slightly noisy) frames
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                # Only the first chunk has the i2va condition: its first frame is pinned
                # to the real-image latent
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                # Intermediate steps use update_cache=0: KV is written temporarily for
                # this attention call and rolled back afterwards, so half-denoised noisy
                # frames never pollute the cache; the last step uses update_cache=1 to
                # commit the clean imagined frames
                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    # The model output is a patch sequence; restore it to the
                    # [B,48,F,h,w] latent layout
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        # CFG formula: pred_uncond + s * (pred_cond - pred_uncond),
                        # with batch layout [0]=conditional (positive prompt),
                        # [1]=unconditional (negative prompt)
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    # One flow-matching step: x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                # After every step, clamp the first-frame latent back to the real-image
                # latent: the i2va conditioning frame must not be rewritten by the model
                # during denoising (identity assignment for chunks after the first)
                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            # 2. Action Generation Loop: by now the KV cache holds the imagined video
            # frames committed in phase 1; action tokens (frame_id=2k+1) attend to video
            # tokens (frame_id=2k), realizing "infer actions conditioned on the imagined
            # future"
            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                # The first-frame action of the first chunk is a history placeholder
                # (all zeros), clamped as a clean condition
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                # Same as video: intermediate steps roll back, the last step (the t=0
                # padding step) commits the clean action KV
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    # Restore the action token sequence to [B,30,F,action_per_frame,1]
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        # Independent CFG for the action branch (decoupled from the
                        # video guidance_scale)
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                # Clamp the first-frame action condition (same rationale as the video
                # first-frame clamp)
                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        # Zero out invalid action channels so de-normalization cannot produce nonzero garbage
        actions[:, ~self.action_mask] *= 0

        # Asynchronously persist the imagined frames and actions for offline
        # debugging/visualization (does not block inference)
        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        """Update the KV cache with a real observation (the closed-loop correction step of the async protocol).

        First clear_pred_cache drops all imagined frames (is_pred=True), then the real
        observation's video/action are committed with update_cache=2 (is_pred=False) --
        imagined frames only serve action generation and never enter the long-term
        history; they are replaced as soon as real observations arrive.

        Args:
            obs: dict containing obs['obs'] (list of multi-camera images) and obs['state']
                (np.ndarray [C_env, F, H] of executed actions / proprioceptive state).
        """
        ### optional async save obs for debug
        # Discard every imagined frame in the cache; only real history remains
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            # First commit: frame_st_id is still 0, so the initial-frame latent encoded
            # during the first _infer must be concatenated with this real observation to
            # form one complete chunk (2 frames) of real history before entering the cache
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        # The real actions go through preprocess (scattered into the 30-dim unified
        # space and normalized) and are committed together with the video
        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        # latent_t/action_t default to 0: real observations are "clean" inputs, so all
        # per-frame timesteps are 0
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            # update_cache=2: commit real observations (is_pred=False); one forward each
            # for video and action
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        # Advance the temporal cursor (usually += frame_chunk_size=2); the next chunk's
        # RoPE starts here
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        """Unified entry point for websocket requests: three-way dispatch by flag.

        Args:
            obs: dict. obs['reset']=True -> reset the server (expects obs['prompt']);
                obs['compute_kv_cache']=True -> update the KV cache with the real
                observation; neither -> run one chunk of AR inference and return actions.

        Returns:
            dict: reset/compute_kv_cache return an empty dict; inference returns
            dict(action=np.ndarray [C_env, 32]) where
            32 = frame_chunk_size x action_per_frame.
        """
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        """De-normalize latents and decode them into a pixel video with the VAE.

        Args:
            latents: normalized latents [B, 48, F, H, W].
            output_type: output type (e.g. 'np'), passed through to
                VideoProcessor.postprocess_video.

        Returns:
            The decoded video (np.ndarray or torch.Tensor depending on output_type).
        """
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        """Read each camera's initial-frame PNG from input_img_path and assemble the observation dict.

        Returns:
            dict {'obs': [{cam_key: np.ndarray [H,W,3]}]}, encoded by the first _infer
            in i2va mode.
        """
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        """i2va offline mode: open-loop rollout of several chunks from the initial image, then decode and export demo.mp4.

        Unlike server mode, no real observations are fed back (compute_kv_cache is never
        called); the rollout advances purely on imagined frames, visualizing the world
        model's long-horizon prediction capability.
        """
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        # After generation, free transformer/text-encoder VRAM to make room for VAE decoding
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    """Build the VA_Server from config and enter i2va offline generation or websocket server mode.

    Args:
        args: command-line arguments (config_name / port / save_root), see ``main``.
    """
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    # Supports multi-GPU launch via torchrun: rank 0 serves external requests while the
    # other ranks follow synchronously inside worker_loop
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO

    Command-line entry point: parse --config-name / --port / --save_root, then call ``run``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
