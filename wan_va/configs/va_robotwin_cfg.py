# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Base config for the RoboTwin 2.0 dual-arm simulation task (websocket inference server mode).

RoboTwin is the main evaluation benchmark of this project: three-camera observation
(cam_high at 256x320 full resolution + both wrist cameras at half resolution) stitched
into one T-shape image (env_type='robotwin_tshape') before entering the VAE; 16 effective
action dims (7 end-effector pose dims + 1 gripper per arm). Recommended deployment
hyperparameters: frame_chunk_size=2, attn_window=72. This config is inherited by
va_robotwin_train_cfg (training) and va_robotwin_i2va / va_robotwin_i2va_eval
(inference/eval).
"""
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
# Merge shared config (host/port, param_dtype, patch_size and other common items)
va_robotwin_cfg.update(va_shared_cfg)

# Path to the Wan2.2 base model weights (directory contains transformer / vae / text_encoder / tokenizer subdirs)
va_robotwin_cfg.wan22_pretrained_model_name_or_path = "/home/tione/notebook/model/lingbot-va-base"

# Sliding attention window size (unit: frame_id, i.e. half-chunks); limits how far back each token can look
va_robotwin_cfg.attn_window = 72
# Number of latent frames generated per autoregressive step; each chunk yields 2*16=32 action sub-steps
va_robotwin_cfg.frame_chunk_size = 2
# Observation stitching mode: three cameras stitched into a T-shape image (high at full resolution, both wrists at half resolution; concatenated along width first, then along height)
va_robotwin_cfg.env_type = 'robotwin_tshape'

# Height of the T-shape stitched image fed into the VAE (pixels)
va_robotwin_cfg.height = 256
# Width of the T-shape stitched image fed into the VAE (pixels)
va_robotwin_cfg.width = 320
# Unified action space dimension (fixed at 30)
va_robotwin_cfg.action_dim = 30
# Control sub-steps per latent frame, i.e. each frame inference outputs 16 actions
va_robotwin_cfg.action_per_frame = 16
# Observation camera keys: top camera + left/right wrist cameras (matching RoboTwin LeRobot dataset naming)
va_robotwin_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
# Video-branch CFG (classifier-free guidance) scale; enabled when >1
va_robotwin_cfg.guidance_scale = 5
# Action-branch CFG scale; 1 means no CFG for actions
va_robotwin_cfg.action_guidance_scale = 1

# Number of flow-matching denoising steps for video latents
va_robotwin_cfg.num_inference_steps = 25
# Denoising steps actually executed by the video branch; -1 means run all (can be truncated for speed)
va_robotwin_cfg.video_exec_step = -1
# Number of denoising steps for the action branch (decoupled from video)
va_robotwin_cfg.action_num_inference_steps = 50

# SNR shift of the video noise schedule: sigma' = shift*sigma/(1+(shift-1)*sigma); larger shifts mass toward high noise
va_robotwin_cfg.snr_shift = 5.0
# SNR shift of the action noise schedule; 1.0 means no shift (decoupled from the video schedule)
va_robotwin_cfg.action_snr_shift = 1.0

# Channel indices actually used in the unified 30-dim action space: 0~6 (left arm) + 28 (left gripper)
# + 7~13 (right arm) + 29 (right gripper), 16 dims total; other channels are zeroed and excluded from the loss (actions_mask=False)
va_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
# Inverse mapping: unified-space channel j -> index i in the dataset action vector; unused channels are filled with len(used) (an out-of-range value marking invalid slots)
inverse_used_action_channel_ids = [
    len(va_robotwin_cfg.used_action_channel_ids)
] * va_robotwin_cfg.action_dim
for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Action normalization method: linear normalization to [-1,1] by q01/q99 quantiles (post-processed with clip to +/-1.5 at inference)
va_robotwin_cfg.action_norm_method = 'quantiles'
# q01/q99 quantile statistics (30-dim, computed offline from RoboTwin dataset actions by evaluation/robotwin/calc_stat.py);
# the first 14 entries are quantiles of the dual-arm pose channels (actions converted to relative poses), the middle [0.]*16 are placeholders for unused channels, the last two 1.0 correspond to the gripper channels
va_robotwin_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}
