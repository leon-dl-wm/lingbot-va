# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Image-to-video-action (i2va) offline inference config for the demo task.

Inherits va_demo_cfg; given a directory of first-frame images and a text prompt, it
autoregressively generates several chunks of "future video + action sequence", saved as
demo.mp4 (and related files) under save_root. Launch with ``--config-name demo_i2av``
(see the VA_CONFIGS registry).
"""
from easydict import EasyDict
from .va_demo_cfg import va_demo_cfg

va_demo_i2va_cfg = EasyDict(__name__='Config: VA demo i2va')
# Inherit the demo base config (model path, chunk/window, action space, norm_stat, etc.)
va_demo_i2va_cfg.update(va_demo_cfg)

# First-frame image directory: one initial observation image per camera in obs_cam_keys, used as the i2va image condition
va_demo_i2va_cfg.input_img_path = 'example/demo'
# Number of chunks to generate autoregressively; total latent frames = num_chunks_to_infer * frame_chunk_size
va_demo_i2va_cfg.num_chunks_to_infer = 10
# Task text instruction (prompt), encoded by the text encoder as conditioning; an empty prompt is used for the CFG unconditional branch
va_demo_i2va_cfg.prompt = 'Pick the green cube and place it inside the blue box'
# Inference mode: 'i2va' = offline image-to-video-action inference (produces demo.mp4), as opposed to 'server' (websocket online service)
va_demo_i2va_cfg.infer_mode = 'i2va'