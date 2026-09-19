# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Image-to-video-action (i2va) offline inference config for the Franka task.

Inherits va_franka_cfg; given a directory of first-frame images and a text prompt, it
autoregressively generates several chunks of "future video + action sequence"
(outputs demo.mp4). Launch with ``--config-name franka_i2av`` (see the VA_CONFIGS registry).
"""
from easydict import EasyDict
from .va_franka_cfg import va_franka_cfg

va_franka_i2va_cfg = EasyDict(__name__='Config: VA franka i2va')
# Inherit the Franka base config (model path, chunk/window, action space, norm_stat, etc.)
va_franka_i2va_cfg.update(va_franka_cfg)

# First-frame image directory: one initial observation image per camera in obs_cam_keys, used as the i2va image condition
va_franka_i2va_cfg.input_img_path = 'example/franka'
# Number of chunks to generate autoregressively; total latent frames = num_chunks_to_infer * frame_chunk_size
va_franka_i2va_cfg.num_chunks_to_infer = 10
# Task text instruction (prompt), encoded by the text encoder as conditioning
va_franka_i2va_cfg.prompt = 'pick bunk'
# Inference mode: 'i2va' = offline image-to-video-action inference (produces demo.mp4), as opposed to 'server' (websocket online service)
va_franka_i2va_cfg.infer_mode = 'i2va'