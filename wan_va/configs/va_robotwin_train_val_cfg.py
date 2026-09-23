# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Short validation run of robotwin post-training (default 500 steps).
# Inherits everything from robotwin_train; only num_steps/save_interval differ.
# MODEL_PATH / DATASET_PATH / VAL_STEPS env vars override inherited values.
import os
from easydict import EasyDict
from .va_robotwin_train_cfg import va_robotwin_train_cfg

va_robotwin_train_val_cfg = EasyDict(__name__='Config: VA robotwin train validation')
va_robotwin_train_val_cfg.update(va_robotwin_train_cfg)

va_robotwin_train_val_cfg.num_steps = int(os.environ.get('VAL_STEPS', 500))
va_robotwin_train_val_cfg.save_interval = min(500, va_robotwin_train_val_cfg.num_steps)

va_robotwin_train_val_cfg.dataset_path = os.environ.get(
    'DATASET_PATH', va_robotwin_train_cfg.dataset_path)
va_robotwin_train_val_cfg.empty_emb_path = os.path.join(
    va_robotwin_train_val_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train_val_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'MODEL_PATH', va_robotwin_train_cfg.wan22_pretrained_model_name_or_path)
