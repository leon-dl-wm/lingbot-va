# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Base config for the LIBERO simulation benchmark (websocket inference server mode).

LIBERO uses a single-arm Franka robot: 7 action dims (6 end-effector pose deltas +
1 gripper), occupying slots 0~6 of the unified 30-dim action space; observations are
128x128 agentview + eye-in-hand cameras. This config is inherited by
va_libero_train_cfg (training) and va_libero_i2va (image-to-video-action inference).
"""
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

va_libero_cfg = EasyDict(__name__='Config: VA libero')
# Merge shared config (host/port, param_dtype, patch_size and other common items)
va_libero_cfg.update(va_shared_cfg)
# Default inference mode: websocket remote inference server (eval clients / real robots use the same interface)
va_shared_cfg.infer_mode = 'server'

# Path to the Wan2.2 base model weights (directory contains transformer / vae / text_encoder / tokenizer subdirs)
va_libero_cfg.wan22_pretrained_model_name_or_path = os.path.expanduser("~/works/dataset/lingbot-va-base")

# Sliding attention window size (unit: frame_id, i.e. half-chunks); limits how far back each token can look
va_libero_cfg.attn_window = 30
# Number of latent frames generated per autoregressive step (chunk size)
va_libero_cfg.frame_chunk_size = 4
# Observation stitching mode: 'none' means no multi-camera T-shape stitching
va_libero_cfg.env_type = 'none'

# Height of video frames fed into the VAE (pixels); LIBERO uses low resolution to save memory / speed up
va_libero_cfg.height = 128
# Width of video frames fed into the VAE (pixels)
va_libero_cfg.width = 128
# Unified action space dimension (fixed at 30)
va_libero_cfg.action_dim = 30
# Control sub-steps per latent frame, i.e. each frame inference outputs 4 actions
va_libero_cfg.action_per_frame = 4
# Observation camera keys: LIBERO's third-person agentview camera + eye-in-hand camera
va_libero_cfg.obs_cam_keys = [
    'observation.images.agentview_rgb', 'observation.images.eye_in_hand_rgb'
]
# Video-branch CFG (classifier-free guidance) scale; enabled when >1
va_libero_cfg.guidance_scale = 5
# Action-branch CFG scale; 1 means no CFG for actions
va_libero_cfg.action_guidance_scale = 1

# Number of flow-matching denoising steps for video latents
va_libero_cfg.num_inference_steps = 20
# Denoising steps actually executed by the video branch; -1 means run all (can be truncated for speed)
va_libero_cfg.video_exec_step = -1
# Number of denoising steps for the action branch (decoupled from video)
va_libero_cfg.action_num_inference_steps = 50

# SNR shift of the video noise schedule: sigma' = shift*sigma/(1+(shift-1)*sigma); larger shifts mass toward high noise
va_libero_cfg.snr_shift = 5.0
# SNR shift of the action noise schedule; <1 biases the schedule toward the low-noise end (decoupled from the video schedule)
va_libero_cfg.action_snr_shift = 0.05

# Channel indices actually used in the unified 30-dim action space: 0~6, 7 dims total (6 pose + 1 gripper); other channels are zeroed and excluded from the loss
va_libero_cfg.used_action_channel_ids = list(range(0, 7))
# Inverse mapping: unified-space channel j -> index i in the dataset action vector; unused channels are filled with len(used) (an out-of-range value marking invalid slots)
inverse_used_action_channel_ids = [len(va_libero_cfg.used_action_channel_ids)
                                   ] * va_libero_cfg.action_dim
for i, j in enumerate(va_libero_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_libero_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Action normalization method: linear normalization to [-1,1] by q01/q99 quantiles (post-processed with clip to +/-1.5 at inference)
va_libero_cfg.action_norm_method = 'quantiles'
# q01/q99 quantile statistics (30-dim, computed offline from LIBERO dataset actions);
# the first 7 entries are quantiles of the effective channels, the trailing [0.]*23 are zero placeholders for unused channels
va_libero_cfg.norm_stat = {
    "q01": [
        -0.6589285731315613,
        -0.84375,
        -0.9375,
        -0.12107142806053162,
        -0.15964286029338837,
        -0.26571428775787354,
        -1.0
    ] + [0.] * 23,
    "q99": [
        0.8999999761581421,
        0.8544642925262451,
        0.9375,
        0.17142857611179352,
        0.1842857152223587,
        0.34392857551574707,
        1.0
    ] + [0.] * 23,
}
