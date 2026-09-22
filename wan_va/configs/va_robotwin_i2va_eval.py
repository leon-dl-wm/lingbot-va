# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# i2va config for evaluating a post-training checkpoint (model path from env).
import os
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_i2va_eval_cfg = EasyDict(__name__='Config: VA robotwin i2va eval')
va_robotwin_i2va_eval_cfg.update(va_robotwin_cfg)

va_robotwin_i2va_eval_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'EVAL_MODEL_PATH',
    va_robotwin_cfg.wan22_pretrained_model_name_or_path,
)
# Offload keeps VAE/text_encoder on CPU; UMT5-5B CPU encode takes ~25 min with
# default threads, so only enable when sharing GPU with training. GPU-exclusive
# eval (training stopped) should leave this False.
va_robotwin_i2va_eval_cfg.enable_offload = False
# Reduced attention window so eval can share a GPU with training (~14G free):
# KV cache 7.2G->2.4G with window 24 (12 chunks of history, enough for a 10-chunk demo).
va_robotwin_i2va_eval_cfg.attn_window = 24
va_robotwin_i2va_eval_cfg.num_chunks_to_infer = 10
va_robotwin_i2va_eval_cfg.input_img_path = 'example/robotwin'
va_robotwin_i2va_eval_cfg.num_chunks_to_infer = 10
va_robotwin_i2va_eval_cfg.prompt = 'Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack.'
va_robotwin_i2va_eval_cfg.infer_mode = 'i2va'
