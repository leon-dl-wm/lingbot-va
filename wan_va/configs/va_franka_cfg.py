# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Base config for the Franka real-robot task (websocket inference server mode).

Targets Franka dual/single-arm real-robot deployment: three-camera observation
(cam_high + both wrist cameras) at 224x320 resolution, 16 effective action dims
(7 end-effector pose dims + 1 gripper per arm), occupying slots 0~6 and 28
(left arm + left gripper) and 7~13 and 29 (right arm + right gripper) of the
unified 30-dim action space. This config is inherited by va_franka_i2va
(image-to-video-action inference).
"""
import torch
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_franka_cfg = EasyDict(__name__='Config: VA franka')
# Merge shared config (host/port, param_dtype, patch_size and other common items)
va_franka_cfg.update(va_shared_cfg)
# Default inference mode: websocket remote inference server (real-robot deployment uses the same interface)
va_shared_cfg.infer_mode = 'server'

# Path to the Wan2.2 base model weights (directory contains transformer / vae / text_encoder / tokenizer subdirs)
va_franka_cfg.wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"

# Sliding attention window size (unit: frame_id, i.e. half-chunks); limits how far back each token can look
va_franka_cfg.attn_window = 30
# Number of latent frames generated per autoregressive step (chunk size)
va_franka_cfg.frame_chunk_size = 4
# Observation stitching mode: 'none' means no RoboTwin-style T-shape stitching
va_franka_cfg.env_type = 'none'

# Height of video frames fed into the VAE (pixels)
va_franka_cfg.height = 224
# Width of video frames fed into the VAE (pixels)
va_franka_cfg.width = 320
# Unified action space dimension (fixed at 30)
va_franka_cfg.action_dim = 30
# Control sub-steps per latent frame, i.e. each frame inference outputs 20 actions (higher real-robot control frequency)
va_franka_cfg.action_per_frame = 20
# Observation camera keys: top camera + left/right wrist cameras (openpi-style real-robot data naming)
va_franka_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
# Video-branch CFG (classifier-free guidance) scale; enabled when >1
va_franka_cfg.guidance_scale = 5
# Action-branch CFG scale; 1 means no CFG for actions
va_franka_cfg.action_guidance_scale = 1

# Number of flow-matching denoising steps for video latents (few steps to keep real-robot latency low)
va_franka_cfg.num_inference_steps = 5
# Denoising steps actually executed by the video branch; -1 means run all (can be truncated for speed)
va_franka_cfg.video_exec_step = -1
# Number of denoising steps for the action branch (decoupled from video)
va_franka_cfg.action_num_inference_steps = 10

# SNR shift of the video noise schedule: sigma' = shift*sigma/(1+(shift-1)*sigma); larger shifts mass toward high noise
va_franka_cfg.snr_shift = 5.0
# SNR shift of the action noise schedule; 1.0 means no shift (decoupled from the video schedule)
va_franka_cfg.action_snr_shift = 1.0

# Channel indices actually used in the unified 30-dim action space: 0~6 (left arm) + 28 (left gripper)
# + 7~13 (right arm) + 29 (right gripper), 16 dims total; other channels are zeroed and excluded from the loss
va_franka_cfg.used_action_channel_ids = list(range(0, 7)) + list(range(
    28, 29)) + list(range(7, 14)) + list(range(29, 30))
# Inverse mapping: unified-space channel j -> index i in the dataset action vector; unused channels are filled with len(used) (an out-of-range value marking invalid slots)
inverse_used_action_channel_ids = [len(va_franka_cfg.used_action_channel_ids)
                                   ] * va_franka_cfg.action_dim
for i, j in enumerate(va_franka_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_franka_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Action normalization method: linear normalization to [-1,1] by q01/q99 quantiles (post-processed with clip to +/-1.5 at inference)
va_franka_cfg.action_norm_method = 'quantiles'
# q01/q99 quantile statistics (30-dim, computed offline from Franka real-robot dataset actions);
# the first 14 entries are quantiles of the dual-arm pose channels, the middle [0.]*16 are placeholders for unused channels, the last two 1.0/0 correspond to the gripper channels
va_franka_cfg.norm_stat = {
    "q01": [
        0.3051295876502991, -0.22647984325885773, 0.19957000017166138,
        -0.022680532187223434, -0.05553057789802551, -0.2693849802017212,
        -0.29341773986816405, 0.2935442328453064, -0.4431332051753998,
        0.21256473660469055, -0.7962440848350525, -0.40816226601600647,
        -0.28359392285346985, -0.44507765769958496
    ] + [0.] * 16,
    "q99": [
        0.7572150230407715, 0.47736290097236633, 0.6428080797195435,
        0.9835678935050964, 0.9927203059196472, 0.28041139245033264,
        0.47529348731040877, 0.7564866304397571, 0.04082797020673729,
        0.5355993628501885, 0.9976375699043274, 0.8973174452781656,
        0.6016915678977965, 0.5027598619461056
    ] + [0.] * 14 + [1.0, 1.0],
}
