# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Image-to-video-action (i2va) offline inference config for the LIBERO task.

Inherits va_libero_cfg and points the model path to the 200-step post-trained checkpoint,
used to verify that post-trained weights can run autoregressive "video + action"
generation correctly (outputs demo.mp4). Launch with ``--config-name libero_i2av``
(see the VA_CONFIGS registry).
"""
from easydict import EasyDict
import os

from .va_libero_cfg import va_libero_cfg

va_libero_i2va_cfg = EasyDict(__name__='Config: VA libero i2va')
# Inherit the LIBERO base config (chunk/window, action space, norm_stat, etc.)
va_libero_i2va_cfg.update(va_libero_cfg)

# First-frame image directory: one initial observation image per camera in obs_cam_keys, used as the i2va image condition
va_libero_i2va_cfg.input_img_path = 'example/libero'
# Infer from the 200-step post-trained checkpoint (server forces attn_mode="torch")
# Override the base model path: use the checkpoint obtained after 200 training steps (transformer weights)
va_libero_i2va_cfg.wan22_pretrained_model_name_or_path = os.path.expanduser(
    '~/works/codeworks/wolrd_models/lingbot-va/train_out/checkpoints/checkpoint_step_200')
# Number of chunks to generate autoregressively; total latent frames = num_chunks_to_infer * frame_chunk_size
va_libero_i2va_cfg.num_chunks_to_infer = 10
# Task text instruction (prompt): the LIBERO-10 "put both the alphabet soup and the tomato sauce in the basket" task
va_libero_i2va_cfg.prompt = "put both the alphabet soup and the tomato sauce in the basket"
# Inference mode: 'i2va' = offline image-to-video-action inference (produces demo.mp4), as opposed to 'server' (websocket online service)
va_libero_i2va_cfg.infer_mode = 'i2va'