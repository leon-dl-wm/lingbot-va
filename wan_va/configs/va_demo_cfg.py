# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Base config for the demo task (websocket inference server mode).

demo is a lightweight demonstration task: a single 256x256 image, two-camera observation,
6 effective action dims (5 arm joint/pose channels + 1 gripper channel, occupying slots
0~4 and 28 of the unified 30-dim action space). This config is inherited by both
va_demo_train_cfg (training) and va_demo_i2va (image-to-video-action inference).
"""
import torch
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_demo_cfg = EasyDict(__name__='Config: VA demo')
# Merge shared config (host/port, param_dtype, patch_size and other common items)
va_demo_cfg.update(va_shared_cfg)
# Default inference mode: websocket remote inference server (eval clients / real robots use the same interface)
va_shared_cfg.infer_mode = 'server'

# Path to the Wan2.2 base model weights (directory contains transformer / vae / text_encoder / tokenizer subdirs)
va_demo_cfg.wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"

# Sliding attention window size (unit: frame_id, i.e. half-chunks); limits how far back each token can look
va_demo_cfg.attn_window = 30
# Number of latent frames generated per autoregressive step (chunk size); larger chunks produce more per step but accumulate more error
va_demo_cfg.frame_chunk_size = 4
# Observation stitching mode: 'none' means no multi-camera T-shape stitching (single-camera image used directly)
va_demo_cfg.env_type = 'none'

# Height of video frames fed into the VAE (pixels)
va_demo_cfg.height = 256
# Width of video frames fed into the VAE (pixels)
va_demo_cfg.width = 256
# Unified action space dimension (fixed at 30, covering channel slots for all tasks: single arm / dual arm / grippers)
va_demo_cfg.action_dim = 30
# Control sub-steps per latent frame, i.e. each frame inference outputs 8 actions
va_demo_cfg.action_per_frame = 8
# Observation camera keys (must match the image dict keys sent by the dataset/client): top camera + wrist camera
va_demo_cfg.obs_cam_keys = [
    'observation.images.top', 'observation.images.wrist'
]
# Video-branch CFG (classifier-free guidance) scale; enabled when >1, larger follows the prompt more closely
va_demo_cfg.guidance_scale = 5
# Action-branch CFG scale; 1 means no CFG for actions
va_demo_cfg.action_guidance_scale = 1

# Number of flow-matching denoising steps for video latents
va_demo_cfg.num_inference_steps = 5
# Denoising steps actually executed by the video branch; -1 means run all (can be truncated for speed)
va_demo_cfg.video_exec_step = -1
# Number of denoising steps for the action branch (decoupled from video)
va_demo_cfg.action_num_inference_steps = 10

# SNR shift of the video noise schedule: sigma' = shift*sigma/(1+(shift-1)*sigma); larger shifts mass toward high noise
va_demo_cfg.snr_shift = 5.0
# SNR shift of the action noise schedule; 1.0 means no shift (two schedulers decoupled from the video one)
va_demo_cfg.action_snr_shift = 1.0

# Channel indices actually used in the unified 30-dim action space: 0~4 (arm) + 28 (gripper), 6 dims total; other channels are zeroed and excluded from the loss
va_demo_cfg.used_action_channel_ids = list(range(0, 5)) + list(range(28, 29))
# Inverse mapping: unified-space channel j -> index i in the dataset action vector; unused channels are filled with len(used) (an out-of-range value marking invalid slots)
inverse_used_action_channel_ids = [len(va_demo_cfg.used_action_channel_ids)
                                   ] * va_demo_cfg.action_dim
for i, j in enumerate(va_demo_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_demo_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Action normalization method: linear normalization to [-1,1] by q01/q99 quantiles (post-processed with clip to +/-1.5 at inference)
va_demo_cfg.action_norm_method = 'quantiles'
# q01/q99 quantile statistics (30-dim, computed offline from dataset actions, e.g. evaluation/robotwin/calc_stat.py);
# the first 5 and the 29th entries are quantiles of the effective channels, the middle [0.]*23 are zero placeholders for unused channels
va_demo_cfg.norm_stat = {
    "q01": [
        -90.60303497314453,
        -98.73043060302734,
        -79.9008560180664,
        48.95470428466797,
        -32.794578552246094,
    ] + [0.] * 23 + [0.8250824809074402, 0],
    "q99": [
        71.735107421875,
        65.89081573486328,
        92.87967681884766,
        100.0,
        22.784151077270508,
    ] + [0.] * 23 + [100.0, 0],
}
