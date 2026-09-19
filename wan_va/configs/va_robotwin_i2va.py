# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Image-to-video-action (i2va) offline inference config for the RoboTwin task.

Inherits va_robotwin_cfg; given a directory of first-frame images and the "hang the mug"
task prompt, it autoregressively generates several chunks of "future video + action
sequence" (outputs demo.mp4). Launch with ``--config-name robotwin_i2av``
(see the VA_CONFIGS registry).
"""
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_i2va_cfg = EasyDict(__name__='Config: VA robotwin i2va')
# Inherit the RoboTwin base config (model path, chunk/window, action space, norm_stat, etc.)
va_robotwin_i2va_cfg.update(va_robotwin_cfg)

# First-frame image directory: one initial observation image per camera in obs_cam_keys, used as the i2va image condition
va_robotwin_i2va_cfg.input_img_path = 'example/robotwin'
# Number of chunks to generate autoregressively; total latent frames = num_chunks_to_infer * frame_chunk_size (2)
va_robotwin_i2va_cfg.num_chunks_to_infer = 10
# Task text instruction (prompt): grab the white mug -> rotate it -> hang it on the dark gray rack
va_robotwin_i2va_cfg.prompt = 'Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack.'
# Inference mode: 'i2va' = offline image-to-video-action inference (produces demo.mp4), as opposed to 'server' (websocket online service)
va_robotwin_i2va_cfg.infer_mode = 'i2va'