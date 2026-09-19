# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Config package entry: aggregates all task configs and registers them in the VA_CONFIGS dict.

Training/inference entry points (``wan_va/train.py``, ``wan_va/wan_va_server.py``) select a
config by name via the hydra-style ``--config-name <key>``. Naming conventions:

- ``*_train``: post-training (posttrain) entry configs, with dataset paths and optimizer hyperparameters;
- ``*_i2av`` / ``*_i2av_eval``: image-to-video-action (i2va) offline inference/eval configs;
- others (robotwin / franka / demo / libero): websocket inference server (server mode) configs;
  both simulation eval clients and real-robot deployment connect to this service.
"""
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_i2va_eval import va_robotwin_i2va_eval_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg

# Registry of config name -> EasyDict config object; --config-name takes one of these keys
VA_CONFIGS = {
    # RoboTwin dual-arm simulation task: websocket inference server config
    'robotwin': va_robotwin_cfg,
    # Franka real-robot task: websocket inference server config
    'franka': va_franka_cfg,
    # RoboTwin image-to-video-action (i2va) offline inference config
    'robotwin_i2av': va_robotwin_i2va_cfg,
    # RoboTwin i2va eval config (model path from env var; used by auto-eval during training)
    'robotwin_i2av_eval': va_robotwin_i2va_eval_cfg,
    # Franka i2va offline inference config
    'franka_i2av': va_franka_i2va_cfg,
    # RoboTwin post-training entry config
    'robotwin_train': va_robotwin_train_cfg,
    # Demo task (5+1 action dims) inference server config
    'demo': va_demo_cfg,
    # Demo task post-training entry config
    'demo_train': va_demo_train_cfg,
    # Demo task i2va offline inference config
    'demo_i2av': va_demo_i2va_cfg,
    # LIBERO simulation benchmark (single arm, 7 action dims) inference server config
    'libero': va_libero_cfg,
    # LIBERO post-training entry config
    'libero_train': va_libero_train_cfg,
    # LIBERO i2va offline inference config
    'libero_i2av': va_libero_i2va_cfg,
}