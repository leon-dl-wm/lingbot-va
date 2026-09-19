"""Utils for evaluating the OpenVLA policy."""
# Notes (added): action-replay deployment policy — loads/runs no neural network model at
# all; instead it replays the recorded expert actions from a LeRobot dataset to the
# robot / simulation environment, step by step.
# Use cases: 1) verify that the websocket deployment pipeline (server/client/serialization)
#               is wired correctly;
#            2) serve as an "expert trajectory baseline" to compare policy success rates against.
# The first half of this file (AdaptiveEnsembler, center_crop_image, resize_with_pad,
# PolicyPreprocessMixin, merge_qwen_config, etc.) is identical to qwenpi_policy.py and is
# kept for interface consistency; what actually takes effect here is QwenPiServer.load_vla
# (builds the replay dataset) and QwenPiServer.infer (returns recorded actions in
# global_step order).

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
    (Note: replay mode never runs model inference; this class is kept only for interface
    consistency with qwenpi_policy.py.)
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
    (Note: the replay-mode QwenPiServer does not use this mixin; it is kept for
    structural consistency with qwenpi_policy.py.)
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

    Notes: the text_keys set lists the LLM structure hyperparameters to sync (hidden
    size / num layers / num attention heads / RoPE theta / vocab size / activation, etc.);
    vision_config (the ViT vision-tower config) is copied wholesale when present,
    otherwise a warning is printed. (Replay mode builds no model; this function is kept
    only for consistency with qwenpi_policy.py.)
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

    Notes (added): replay-version deployment wrapper (same name and interface as
    qwenpi_policy.QwenPiServer so it can be swapped in). The difference: self.vla is not
    a neural network model but the LeRobot dataset object built by build_vla_dataset;
    infer() performs no inference at all and simply replays the recorded expert actions
    from the dataset in global_step order. Used to verify the deployment pipeline or as
    an expert-trajectory baseline for comparison evals.
    '''

    def __init__(
        self,
        path_to_pi_model="",
        adaptive_ensemble_alpha=0.1,
        action_ensemble_horizon=8,
        use_length=1,  # to control the execution length of the action chunk, -1 denotes using action ensemble
        use_bf16=True,
    ) -> None:
        """Initialize the replay wrapper.

        Args:
            path_to_pi_model (str): compatibility argument, unused in replay mode (no model weights are loaded).
            adaptive_ensemble_alpha (float): compatibility argument, replay mode does no ensembling.
            action_ensemble_horizon (int): compatibility argument, ensemble window length.
            use_length (int): compatibility argument, chunk execution length; replay mode takes a new action every step.
            use_bf16 (bool): compatibility argument, replay mode has no model precision to speak of.
        """

        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.use_length = use_length

        self.task_description = None

        # Ensembler (never actually used in replay mode; kept for interface consistency)
        self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon,
                                                  self.adaptive_ensemble_alpha)

        # Note: self.vla here is actually the replay dataset (see load_vla), not a model; no GPU needed
        self.vla = self.load_vla(path_to_pi_model)
        self.vla = self.vla
        # Replay cursor: index of the dataset sample to replay next
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
        norm_stats.json (taking the hanging_mug-aloha-agilex_clean_50_rep task entry),
        truncated to the first state_dim / action_dim dims (default 14 = dual arm 7+7).
        (Replay mode currently never calls this function; kept for interface consistency.)
        '''
        with open(states_path) as f:
            norm_stats = json.load(f)['hanging_mug-aloha-agilex_clean_50_rep']
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
        """Unnormalize the action: action * (std + 1e-6) + mean, restoring physical units."""
        action = norm_action * (self.action_std + 1e-6) + self.action_mean
        return action

    def load_vla(self, path_to_pi_model) -> QwenPI0Policy:
        """Build the replay dataset (note: does NOT load a model; the return type annotation exists only for compatibility with qwenpi_policy).

        Steps: load the Qwen2.5-VL tokenizer -> hand-assemble a minimal dataset config with
        SimpleNamespace (max_state_dim/max_action_dim=14, tokenizer_max_length=128, images
        resized to 224x224) -> call veomni's build_vla_dataset to read the agilex-format
        hanging_mug LeRobot dataset (chunk_size=50).

        Args:
            path_to_pi_model (str): compatibility argument, unused; the dataset path is hard-coded inside.

        Returns:
            Dataset object: supports ``dataset[i]['actions']`` to fetch recorded samples by
            index, consumed by infer for replay.
        """
        # load model
        from types import SimpleNamespace

        from transformers import AutoTokenizer
        from veomni.data.dataset import build_vla_dataset

        # Dataset construction needs the tokenizer (for tokenizing task instruction text)
        tokenizer = AutoTokenizer.from_pretrained(
            '/home/yangshuai/yangshuai_ssd0/rep/VLA_pretraining/checkpoints/Qwen2.5-VL-3B-Instruct'
        )

        # Hand-assembled minimal config: state/action dim 14 (dual arm 7+7), max 128 text tokens, images 224x224
        config = SimpleNamespace()
        config.max_state_dim = 14
        config.max_action_dim = 14
        config.tokenizer_max_length = 128
        config.resize_imgs_with_padding = (224, 224)

        # Read the agilex-format hanging_mug demo dataset; each sample contains a recorded actions chunk
        dataset = build_vla_dataset(
            datasets_type='agilex',
            repo_id=
            '/home/yangshuai/yangshuai_ssd0/cache/huggingface/lerobot/hanging_mug-aloha-agilex_clean_50_rep',
            config=config,
            chunk_size=50,
            tokenizer=tokenizer,
        )
        return dataset

    def reset(self, task_description: str) -> None:
        """Reset replay state: record the task description, clear the ensembler, zero the replay cursor.

        Args:
            task_description (str): task text description (only recorded; replay does not depend on it).
        """
        self.task_description = task_description
        if self.use_length == -1:
            self.action_ensembler.reset()

        self.global_step = 0
        self.last_action_chunk = None

    def infer(self, observation, center_crop=True):
        """Generates an action with the VLA policy."""
        # Notes (added): replay-version single-step "inference" — completely ignores the incoming
        # observation; takes the recorded action chunk of the i-th dataset sample at the global_step
        # cursor, returns its step-0 action (as numpy); the cursor advances so the next call replays
        # the next sample.
        action = self.vla[self.global_step]['actions'][0]
        self.global_step += 1

        return dict(action=action.numpy())


# Manual debug entry: build the replay dataset and serve it with WebsocketPolicyServer on port 8000
# (PATH_TO_PI_MODEL is only a compatibility argument; replay mode loads no model)
if __name__ == "__main__":

    from .websocket_policy_server import WebsocketPolicyServer

    # PATH_TO_PI_MODEL = "/home/yangshuai/yangshuai_ssd0/checkpoint/qwen_pi0/qwenpi0_libero_48token_6bs_4node_bf16vlm_fp32_fsdp2_compile_tuneV/checkpoints/global_step_30260/hf_ckpt"
    PATH_TO_PI_MODEL = "/home/yangshuai/yangshuai_ssd0/checkpoint/qwen_pi0/8GPU_cotraining/checkpoints/global_step_35000/hf_ckpt"

    model = QwenPiServer(PATH_TO_PI_MODEL, use_length=50)

    # To debug model with server
    model_server = WebsocketPolicyServer(model, port=8000)
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
