# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Long-horizon i2va: 40 chunks exceeds attn_window/2=36 cached chunks, exercising eviction.
import os
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_i2va_long_cfg = EasyDict(__name__='Config: VA robotwin i2va long-horizon')
va_robotwin_i2va_long_cfg.update(va_robotwin_cfg)

va_robotwin_i2va_long_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'EVAL_MODEL_PATH', va_robotwin_cfg.wan22_pretrained_model_name_or_path)
va_robotwin_i2va_long_cfg.enable_offload = False
va_robotwin_i2va_long_cfg.num_chunks_to_infer = 40
va_robotwin_i2va_long_cfg.input_img_path = 'example/robotwin'
va_robotwin_i2va_long_cfg.prompt = 'Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack.'
va_robotwin_i2va_long_cfg.infer_mode = 'i2va'
