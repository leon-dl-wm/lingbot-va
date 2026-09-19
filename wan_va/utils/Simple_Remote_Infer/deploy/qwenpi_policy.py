"""Utils for evaluating the OpenVLA policy."""
# Notes (added): deployment/inference components for the QwenPI0 policy (a pi0-style VLA
# built on the Qwen2.5-VL backbone). This module provides:
# - observation preprocessing: image resize+pad, state zero-padding, PaliGemma-style text
#   tokenization (PolicyPreprocessMixin);
# - AdaptiveEnsembler: cosine-similarity-based adaptive action ensembling (smooths repeated predictions);
# - QwenPiServer: websocket deployment policy wrapper supporting two execution modes —
#   "action chunk execution" and "adaptive ensembling"; it can be passed directly as the
#   policy argument of WebsocketPolicyServer (see the __main__ debug entry at the bottom).
# For the dataset action-replay variant (no model inference, replays recorded actions),
# see replay_policy.py in the same directory.

import json
import os
import time
from collections import deque
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from lerobot.configs.policies import PreTrainedConfig
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor, nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from veomni.models.vla.pi0 import PI0Policy, QwenPI0Policy

# Fixed camera observation key order: base (head) camera + left wrist camera + right wrist camera;
# prepare_images iterates in this order and stacks the images; missing cameras get zero placeholders with mask=False
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


class AdaptiveEnsembler:
    """Adaptive action ensembler: similarity-weighted average of the current and recent predicted actions to suppress jitter.

    Idea: keeps a history queue of the last pred_action_horizon predictions; after each
    new prediction is appended, it computes the cosine similarity between every queued
    prediction and the current one, converts similarities to weights via
    exp(alpha * cos) (normalized), and takes the weighted average. Larger alpha gives
    more weight to predictions that agree with the current one; alpha=0 degenerates to a
    plain mean.
    """

    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        """Initialize the ensembler.

        Args:
            pred_action_horizon (int): length of the prediction history queue (ensemble window; how many past predictions to keep).
            adaptive_ensemble_alpha (float): similarity weighting coefficient; 0 means equal weights.
        """
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        """Clear the prediction history queue (call at episode start to avoid cross-task interference)."""
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        """Adaptively ensemble the current prediction with the historical predictions.

        Args:
            cur_action (np.ndarray): current predicted action. Either 1-D with shape
                [action_dim] (single action), or 2-D with shape [horizon, action_dim]
                (action chunk; in that case the k-th most recent historical prediction
                contributes its step-k action, aligned with the current time step).

        Returns:
            np.ndarray: the ensembled action (weighted average over historical predictions),
            shaped like a single-step action.
        """
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack([
                pred_actions[i] for (
                    i,
                    pred_actions) in zip(range(num_actions -
                                               1, -1, -1), self.action_history)
            ])

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()

        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """Center-crop the image and resize it to 224x224 (matching training-time image augmentation).

    Args:
        image (np.ndarray | Image.Image): input image. ndarray supports float ([0,1] or
            [0,255], auto-detected), uint16 (divided by 257 to map onto 8-bit), uint8, etc.

    Returns:
        Image.Image: RGB PIL image, center-cropped by area ratio crop_scale=0.9 and
        bilinearly resized to (224, 224).

    Note: crop_scale is an area ratio, so the side-length scale is sqrt(0.9), not 0.9.
    """
    crop_scale = 0.9
    side_scale = float(np.sqrt(np.clip(crop_scale, 0.0,
                                       1.0)))  # side length scale
    out_size = (224, 224)

    # Convert input to PIL Image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype.kind == "f":
            # If floats likely in [0,1], map to [0,255]
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        elif arr.dtype == np.uint16:
            # Map 16-bit to 8-bit
            arr = (arr / 257).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        pil = Image.fromarray(arr)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        raise TypeError("image must be a numpy array or PIL.Image.Image")

    # Force RGB for consistent output
    pil = pil.convert("RGB")
    W, H = pil.size

    # Compute centered crop box (integer pixels)
    crop_w = max(1, int(round(W * side_scale)))
    crop_h = max(1, int(round(H * side_scale)))
    left = (W - crop_w) // 2
    top = (H - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = pil.crop((left, top, right, bottom))
    resized = cropped.resize(out_size, resample=Image.BILINEAR)
    return resized


def resize_with_pad(img, width, height, pad_value=-1):
    """Aspect-preserving resize of a (B, C, H, W) image tensor to the target size, padding on the left/top with pad_value.

    Args:
        img (torch.Tensor): input image batch of shape (B, C, H, W); if given as
            (B, H, W, C) (channels last with C in {1,3}) it is permuted to channels-first.
        width (int): target width in pixels.
        height (int): target height in pixels.
        pad_value (float): padding value, default -1.

    Returns:
        torch.Tensor: image of shape (B, C, height, width); first scaled bilinearly by the
        long-side ratio, then the missing area is padded on the left/top sides (unlike
        image_tools.resize_with_pad, which pads symmetrically/centered).
    """
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    # channel last to channel first if necessary
    if img.shape[1] not in (1, 3) and img.shape[-1] in (1, 3):
        img = img.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img,
                                size=(resized_height, resized_width),
                                mode="bilinear",
                                align_corners=False)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0),
                       value=pad_value)
    return padded_img


class PolicyPreprocessMixin:
    """
    A mixin class that provides preprocessing utilities for observations.
    Can be mixed into any policy class to add image, state, action, language handling.

    Notes (added): observation-preprocessing mixin — provides the three preparation
    functions for images (prepare_images), state (prepare_state) and language
    (prepare_language), plus the full select_action inference entry. Combined via
    multiple inheritance with veomni's PI0Policy / QwenPI0Policy to form the inference
    policy classes below; relies on the mixed-in class providing self.config,
    self.image_processor, self.language_tokenizer and self.model.
    """

    def prepare_images(self, observation: dict[str, Tensor]):
        """Normalize, resize, and pad images and stack them into a tensor.

        Args:
            observation (dict[str, Tensor])

        Returns:
            images (torch.Tensor): (*b, n, c, h, w) images in range [-1.0, 1.0]
            img_masks (torch.Tensor): (*b, n) masks for images, True if image is present, False if missing

        Notes (added): iterates over the three camera keys in the fixed IMAGE_KEYS order
        (base / left wrist / right wrist): each present image is resize_with_pad-ed to
        config.resize_imgs_with_padding (pad value 0) and normalized to [-1,1] by
        image_processor; missing cameras get all-zero placeholders with mask=False.
        Finally stacked into an (n, c, h, w) tensor (n=3) and moved, together with the
        mask, to the device of state.
        """
        dtype = observation["state"].dtype
        bsize = observation["state"].shape[0]
        device = observation["state"].device
        images, img_masks = [], []
        for key in IMAGE_KEYS:
            if key in observation["image"]:
                # resize, pad, and normalize
                img = observation["image"][key]  # torch.Size([1, 3, 224, 224])

                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)

                img = resize_with_pad(img,
                                      *self.config.resize_imgs_with_padding,
                                      pad_value=0)
                img = self.image_processor(img)['pixel_values']
                images.append(img)
                img_masks.append(True)
            else:
                img = np.zeros_like(img)
                images.append(img)
                img_masks.append(False)
        # import ipdb; ipdb.set_trace()
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0).to(device=device)
        elif isinstance(images[0], np.ndarray):
            images = torch.from_numpy(np.stack(images, axis=0)).to(
                device=device)  # torch.Size([3, 256, 1176])
        img_masks = torch.tensor(img_masks,
                                 dtype=torch.bool).to(device=device)  # (*b, n)

        return images, img_masks

    def prepare_state(self, observation):
        """Convert the observation's robot state to a tensor and right-pad it to max_state_dim.

        Args:
            observation (dict): observation dict; observation["state"] is an np.ndarray
                of shape (B, state_dim).

        Returns:
            torch.Tensor: state tensor of shape (B, max_state_dim) (missing dims padded
            with 0, so robots with different DoF share the same model input width).
        """
        state = torch.from_numpy(observation["state"])
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[1]))
        return state

    def prepare_language(self, observation: dict[str, Tensor]):
        """If `prompt` is provided, modify it to PaliGemma format and tokenize it.
        If `lang_tokens` and `lang_masks` are provided, use them directly.

        PaliGemma expects prefix prompts to be formatted as:
        <images> .... <images> <bos> prompt <sep>, where <sep> uses `\\n`.
        So here we format the prompt to start with `<bos>` and end with `\\n`.
        Later, we will concatenate the images and language tokens into a single sequence.

        Args:
            observation (dict[str, Tensor])

        Returns:
            lang_tokens (torch.Tensor): (*b, l) language tokens
            lang_masks (torch.Tensor): (*b, l) masks for language tokens, True if token is present, False if missing

        Notes (added): provide either raw prompt strings (this function prepends <bos>,
        appends the newline separator, tokenizes and right-pads to
        config.tokenizer_max_length), or pre-tokenized lang_tokens/lang_masks (moved
        directly to the device of state).
        """
        lang_tokens = observation.get("lang_tokens", None)
        lang_masks = observation.get("lang_masks", None)
        prompt = observation.get("prompt", None)

        # either provide `prompt` or (`lang_tokens`, `lang_masks`)
        if prompt is None and (lang_tokens is None or lang_masks is None):
            raise ValueError(
                "Either 'prompt' or ('lang_tokens', 'lang_masks') must be provided in the observation."
            )

        device = observation["state"].device
        if prompt is not None and (lang_tokens is None or lang_masks is None):
            prompt = [
                p if p.startswith("<bos>") else f"<bos>{p}" for p in prompt
            ]
            prompt = [p if p.endswith("\n") else f"{p}\n" for p in prompt]
            tokenized_prompt = self.language_tokenizer.__call__(
                prompt,
                padding="max_length",
                padding_side="right",
                max_length=self.config.tokenizer_max_length,
                return_tensors="pt",
            )
            lang_tokens = tokenized_prompt["input_ids"].to(device=device)
            lang_masks = tokenized_prompt["attention_mask"].to(
                device=device, dtype=torch.bool)
        else:
            lang_tokens = observation["lang_tokens"].to(device=device)
            lang_masks = observation["lang_masks"].to(device=device,
                                                      dtype=torch.bool)

        return lang_tokens, lang_masks

    @torch.no_grad
    def select_action(self,
                      observation: dict[str, Tensor],
                      noise: Tensor | None = None):
        """Full single-step inference entry: preprocess the observation, then sample an action chunk with the underlying PI0 model.

        Args:
            observation (dict[str, Tensor]): observation dict (image / state / prompt,
                or lang_tokens / lang_masks).
            noise (Tensor | None): reserved argument, currently unused (could specify the
                initial noise for flow-matching sampling).

        Returns:
            torch.Tensor: sampled action chunk (in normalized space), shape approximately
            (B, action_horizon, action_dim); runs under no_grad + eval mode with bf16.
        """
        self.eval()
        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)
        lang_tokens, lang_masks = self.prepare_language(observation)
        device = 'cuda'
        dtype = torch.bfloat16

        actions = self.model.sample_actions(
            images.to(dtype=dtype, device=device),
            img_masks.to(device=device),
            lang_tokens.to(device=device),
            lang_masks.to(device=device),
            state.to(dtype=dtype, device=device),
        )
        return actions


class QwenPI0InferencePolicy(PolicyPreprocessMixin, QwenPI0Policy):
    """QwenPI0 inference policy: combines the preprocessing mixin with veomni's QwenPI0Policy (no extra logic)."""
    pass  # Only combine necessary functions


class PI0InfernecePolicy(PolicyPreprocessMixin, PI0Policy):
    """PI0 (PaliGemma backbone) inference policy: combines the preprocessing mixin with veomni's PI0Policy (no extra logic)."""
    pass  # Only combine necessary functions


def merge_qwen_config(policy_config, qwen_config):
    """Merge Qwen2.5-VL backbone config entries into the lerobot-style policy config.

    Args:
        policy_config: PI0 policy config object (PreTrainedConfig etc., supports setattr), modified in place.
        qwen_config: Qwen2.5-VL AutoConfig object or its dict form.

    Returns:
        The merged policy_config (same object).

    Notes: a policy checkpoint's config only stores the PI0 head hyperparameters; building
    the full model also needs the LLM backbone structure — the text_keys set lists the keys
    to sync (hidden size / num layers / num attention heads / RoPE theta / vocab size /
    activation, etc.); vision_config (the ViT vision-tower config) is copied wholesale when
    present, otherwise a warning is printed.
    """
    if hasattr(qwen_config, 'to_dict'):
        config_dict = qwen_config.to_dict()
    else:
        config_dict = qwen_config

    text_keys = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
        "tokenizer_path",
    }

    for key in text_keys:
        if key in config_dict:
            setattr(policy_config, key, config_dict[key])
            print(f"✅ Merged LLM: {key} = {config_dict[key]}")

    if "vision_config" in config_dict:
        policy_config.vision_config = qwen_config.vision_config
    else:
        print("⚠️ Warning: 'vision_config' not found in qwen_config!")

    return policy_config


class QwenPiServer:
    '''
    policy wrapper to support action ensemble or chunk execution

    Notes (added): websocket-deployment wrapper around the QwenPI0 policy (can be passed
    directly as the policy argument of WebsocketPolicyServer). Two action-execution modes
    (controlled by use_length):
    - use_length > 0: chunk execution — a real inference runs only every use_length steps
      to produce an action chunk, whose actions are replayed in order in between
      (lower inference frequency);
    - use_length == -1: adaptive ensembling — inference runs every step and
      AdaptiveEnsembler weights it against historical predictions (smoother actions).
    Also handles state normalization / action unnormalization (mean/std scheme, statistics
    loaded from norm_stats.json).
    '''

    def __init__(
        self,
        path_to_pi_model="",
        adaptive_ensemble_alpha=0.1,
        action_ensemble_horizon=8,
        use_length=1,  # to control the execution length of the action chunk, -1 denotes using action ensemble
        use_bf16=True,
    ) -> None:
        """Initialize the wrapper: build the ensembler, load the VLA model and move it to GPU.

        Args:
            path_to_pi_model (str): PI0 policy weights directory (hf_ckpt, with config and *.safetensors).
            adaptive_ensemble_alpha (float): similarity exponent coefficient for adaptive ensembling (only used when use_length=-1).
            action_ensemble_horizon (int): ensemble sliding-window length (how many recent predictions to keep).
            use_length (int): chunk execution length; a real inference runs every use_length steps; -1 switches to action-ensembling mode.
            use_bf16 (bool): whether to cast the model to bfloat16 for inference (halves GPU memory).
        """

        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.use_length = use_length

        self.task_description = None

        # Adaptive action ensembler (used in use_length=-1 mode)
        self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon,
                                                  self.adaptive_ensemble_alpha)

        # Load the VLA model (weights, tokenizer/processor, normalization statistics) and set eval mode
        self.vla = self.load_vla(path_to_pi_model)
        self.vla = self.vla.to("cuda").eval()
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        # Server-side global step counter: decides when to re-run inference and which action to take from the chunk
        self.global_step = 0
        self.last_action_chunk = None

    def init_norm(
            self,
            states_path='/home/yangshuai/yangshuai_ssd0/checkpoint/qwen_pi0/norm_stats.json',
            state_dim=14,
            action_dim=14):
        '''
        TODO: show be rewritten as a dict

        Notes (added): loads state/action mean/std normalization statistics from
        norm_stats.json (taking the stack_bowls_three-aloha-agilex_clean_50_rep task
        entry), truncated to the first state_dim / action_dim dims (default 14 = dual
        arm 7+7), stored as instance attributes for state_normalizer / action_unnormalizer.
        '''
        with open(states_path) as f:
            norm_stats = json.load(
                f)['stack_bowls_three-aloha-agilex_clean_50_rep']
        self.state_mean = np.array(
            norm_stats["norm_stats"]["state"]["mean"][:state_dim],
            dtype=np.float32)
        self.state_std = np.array(
            norm_stats["norm_stats"]["state"]["std"][:state_dim],
            dtype=np.float32)
        self.action_mean = np.array(
            norm_stats["norm_stats"]["actions"]["mean"][:action_dim],
            dtype=np.float32)
        self.action_std = np.array(
            norm_stats["norm_stats"]["actions"]["std"][:action_dim],
            dtype=np.float32)

    def state_normalizer(self, unnorm_state):
        """Normalize the state: (state - mean) / (std + 1e-6); input/output are np.ndarray."""
        state = (unnorm_state - self.state_mean) / (self.state_std + 1e-6)
        return state

    def action_unnormalizer(self, norm_action):
        """Unnormalize the action: action * (std + 1e-6) + mean, restoring physical units from model output."""
        action = norm_action * (self.action_std + 1e-6) + self.action_mean
        return action

    def load_vla(self, path_to_pi_model) -> QwenPI0Policy:
        """Load the QwenPI0/PI0 policy model with its tokenizer/processor.

        Steps: read the policy config -> merge the Qwen2.5-VL backbone config
        (merge_qwen_config) -> pick QwenPI0InferencePolicy / PI0InfernecePolicy by path
        name -> merge and load all *.safetensors shards in the directory (strict=True) ->
        attach tokenizer/processor/image_processor -> initialize normalization statistics
        (init_norm).

        Args:
            path_to_pi_model (str): policy weights directory (hf_ckpt).

        Returns:
            QwenPI0Policy: inference policy object ready for select_action (built on CPU;
            __init__ moves it to GPU).
        """
        # load model
        print(f"loading model from: {path_to_pi_model}")
        config = PreTrainedConfig.from_pretrained(path_to_pi_model)

        # Qwen2.5-VL-3B backbone path (provides the LLM/ViT structure config and tokenizer/processor)
        base_model_path = '/home/yangshuai/yangshuai_ssd0/rep/VLA_pretraining/checkpoints/Qwen2.5-VL-3B-Instruct'

        qwen_config = AutoConfig.from_pretrained(base_model_path)
        qwen_config.tokenizer_path = base_model_path
        # merge transformers extra config to lerobot
        config = merge_qwen_config(config, qwen_config)

        # load processors
        language_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        processor = AutoProcessor.from_pretrained(base_model_path)

        print('Initializing model ... ')

        # Pick the backbone by weights path name: contains 'qwen' -> QwenPI0, otherwise the PaliGemma-based PI0
        if 'qwen' in path_to_pi_model:
            policy = QwenPI0InferencePolicy(config)
        else:
            # Shuai: Not tested yet
            policy = PI0InfernecePolicy(config)

        # Merge multiple safetensor weights
        # FSDP-saved weights are usually multiple safetensors shards; read them all and merge into one state_dict
        all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
        merged_weights = {}

        for file_path in tqdm(all_safetensors):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_weights[key] = f.get_tensor(key)

        policy.load_state_dict(merged_weights, strict=True)

        # Load data processors
        # Attach tokenizer/processor to the policy for the PolicyPreprocessMixin preprocessing functions
        policy.language_tokenizer = language_tokenizer
        policy.processor = processor
        policy.image_processor = processor.image_processor

        print('Model initialized ... ')

        self.init_norm()
        return policy

    def reset(self) -> None:
        """Reset inference state: clear ensembler history, zero the step counter, drop the cached action chunk.

        Triggered at the start of each episode when the client sends ``dict(reset=True)`` (see infer).
        """
        if self.use_length == -1:
            self.action_ensembler.reset()

        self.global_step = 0
        self.last_action_chunk = None

    def infer(self, observation, center_crop=True):
        """Generates an action with the VLA policy."""
        # Notes (added): single-step inference entry of the websocket server (observation is the dict sent by the client). Flow:
        # 1) observation contains reset=True -> reset internal state and return dict(action=None);
        # 2) normalize state with mean/std (raw value kept for debugging);
        # 3) chunk mode (use_length>0): call vla.select_action only every use_length steps to get
        #    a normalized action chunk and cache it; ensemble mode (use_length==-1): infer every step and ensemble adaptively;
        # 4) take the current step's action -> unnormalize -> truncate to the first 14 dims (dual arm 7+7, hard-coded, to be removed) -> return dict(action=...).

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if 'reset' in observation and observation['reset']:
            self.reset()
            return dict(action=None)

        # Keep the unnormalized state (for debugging) and replace the observation's state with its normalized version
        unnorm_state = observation['state']
        observation['state'] = self.state_normalizer(observation['state'])

        # Chunk mode: infer once every use_length steps; ensemble mode (-1): infer every step
        if self.use_length == -1 or self.global_step % self.use_length == 0:
            normalized_actions = self.vla.select_action(observation).squeeze(0)
            self.last_action_chunk = normalized_actions.float().cpu().numpy()
            self.last_state = unnorm_state

        if self.use_length > 0:
            # Take one action from the cached chunk at the current step (replay within the chunk)
            action = self.last_action_chunk[self.global_step % self.use_length]
        elif self.use_length == -1:  # do ensemble
            # Ensemble mode: similarity-weighted average with historical predictions
            action = self.action_ensembler.ensemble_action(normalized_actions)

        # Unnormalize back to physical units and truncate to the first 14 dims (dual arm 7+7)
        action = self.action_unnormalizer(
            action[:14])  # TODO: remove the hard code dim!
        action = action[:14]  # + self.last_state[0,:14]
        print(
            f"on server step: {self.global_step} state {observation['state'][:7]} action {action[:7]}"
        )

        self.global_step += 1

        return dict(action=action)


# Manual debug entry: load QwenPI0 weights (use_length=50, i.e. infer once every 50 steps and replay within the chunk),
# then serve it with WebsocketPolicyServer on port 8002
if __name__ == "__main__":

    from .websocket_policy_server import WebsocketPolicyServer
    PATH_TO_PI_MODEL = '/home/yangshuai/yangshuai_ssd0/checkpoint/qwen_pi0/stack_bowls_three-aloha-agilex_clean_50_rep/checkpoints/global_step_30000/hf_ckpt/'

    model = QwenPiServer(PATH_TO_PI_MODEL, use_length=50)

    # To debug model with server
    model_server = WebsocketPolicyServer(model, port=8002)
    model_server.serve_forever()

    # # To debug model only
    # import torch
    # import numpy as np
    # from PIL import Image
    # from .image_tools import convert_to_uint8
    # device = torch.device("cuda")

    # base_0_rgb = np.random.randint(0, 256, size=(1, 3, 224, 224), dtype=np.uint8)
    # left_wrist_0_rgb = np.random.randint(0, 256, size=(1, 3, 224, 224), dtype=np.uint8)
    # state = np.random.rand(1,8).astype(np.float32)
    # prompt = ["do something"]

    # observation = {
    #     "image": {
    #         "base_0_rgb": convert_to_uint8(base_0_rgb),
    #         "left_wrist_0_rgb": convert_to_uint8(left_wrist_0_rgb),
    #         "right_wrist_0_rgb": convert_to_uint8(left_wrist_0_rgb),
    #     },
    #     "state": state,
    #     "prompt": prompt,
    # }

    # model.infer(observation)
