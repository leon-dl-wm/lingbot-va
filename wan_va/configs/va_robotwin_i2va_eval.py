# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# i2va config for evaluating a post-training checkpoint (model path from env).
"""RoboTwin i2va eval config: used for automatic evaluation of a given checkpoint during training.

Differences from va_robotwin_i2va:
1. The model path is read from the ``EVAL_MODEL_PATH`` environment variable first
   (injected by script/eval_checkpoint.sh, pointing to the assembled eval model
   directory); falls back to the base model path in the base config when unset;
2. enable_offload is turned on by default so VAE/text encoder live on CPU, allowing
   the eval to coexist with training on the same machine (~18G GPU memory).
Launch with ``--config-name robotwin_i2av_eval`` (see the VA_CONFIGS registry).
"""
import os
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_i2va_eval_cfg = EasyDict(__name__='Config: VA robotwin i2va eval')
# Inherit the RoboTwin base config (chunk/window, action space, norm_stat, etc.)
va_robotwin_i2va_eval_cfg.update(va_robotwin_cfg)

# Model path to evaluate: prefer the EVAL_MODEL_PATH env var (passed by eval_checkpoint.sh),
# fall back to the base model path from va_robotwin_cfg when unset
va_robotwin_i2va_eval_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'EVAL_MODEL_PATH',
    va_robotwin_cfg.wan22_pretrained_model_name_or_path,
)
va_robotwin_i2va_eval_cfg.enable_offload = True  # fit alongside training (~18G)
# First-frame image directory: one initial observation image per camera in obs_cam_keys, used as the i2va image condition
va_robotwin_i2va_eval_cfg.input_img_path = 'example/robotwin'
# Number of chunks to generate autoregressively; total latent frames = num_chunks_to_infer * frame_chunk_size (2)
va_robotwin_i2va_eval_cfg.num_chunks_to_infer = 10
# Task text instruction (prompt): grab the white mug -> rotate it -> hang it on the dark gray rack
va_robotwin_i2va_eval_cfg.prompt = 'Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack.'
# Inference mode: 'i2va' = offline image-to-video-action inference (produces demo.mp4), as opposed to 'server' (websocket online service)
va_robotwin_i2va_eval_cfg.infer_mode = 'i2va'
