# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .va_libero_cfg import va_libero_cfg

va_libero_i2va_cfg = EasyDict(__name__='Config: VA libero i2va')
va_libero_i2va_cfg.update(va_libero_cfg)

va_libero_i2va_cfg.input_img_path = 'example/libero'
# Infer from the 200-step post-trained checkpoint (server forces attn_mode="torch")
va_libero_i2va_cfg.wan22_pretrained_model_name_or_path = os.path.expanduser(
    '~/works/codeworks/wolrd_models/lingbot-va/train_out/checkpoints/checkpoint_step_200')
va_libero_i2va_cfg.num_chunks_to_infer = 10
va_libero_i2va_cfg.prompt = "put both the alphabet soup and the tomato sauce in the basket"
va_libero_i2va_cfg.infer_mode = 'i2va'