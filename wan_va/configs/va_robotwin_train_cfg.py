# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Post-training (posttrain) entry config for the RoboTwin task.

Inherits all inference hyperparameters from va_robotwin_cfg, then adds dataset paths and
optimizer hyperparameters. Launch with ``bash script/run_va_posttrain.sh`` (defaults to
``CONFIG_NAME=robotwin_train``; hydra style, supports command-line overrides such as
``--learning_rate 5e-5``).
"""
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
# Inherit the RoboTwin base config (model path, chunk/window, action space, norm_stat, etc.)
va_robotwin_train_cfg.update(va_robotwin_cfg)

# LeRobot latent dataset path (videos pre-encoded offline by the Wan2.2 VAE into latents/*.pth; text embeddings pre-computed as well)
va_robotwin_train_cfg.dataset_path = '/home/tione/notebook/data/Robbyant/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500'
# Empty text embedding file (empty_emb.pt): replaces the real prompt embedding when training the CFG unconditional branch
va_robotwin_train_cfg.empty_emb_path = os.path.join(va_robotwin_train_cfg.dataset_path, 'empty_emb.pt')
# Whether to log training curves (loss, grad_norm, etc.) to wandb
va_robotwin_train_cfg.enable_wandb = False
# Number of DataLoader worker processes; more workers load faster but use more RAM
va_robotwin_train_cfg.load_worker = 16
# Save a checkpoint every this many steps (written to save_root/checkpoints/)
va_robotwin_train_cfg.save_interval = 1000
# Trigger manual garbage collection every this many steps (mitigates GPU/host memory fragmentation)
va_robotwin_train_cfg.gc_interval = 50
# Probability of replacing the text embedding with the empty embedding during training, to learn the CFG unconditional branch
va_robotwin_train_cfg.cfg_prob = 0.1

# Training parameters
# AdamW learning rate (small lr for post-training to avoid damaging the base model)
va_robotwin_train_cfg.learning_rate = 1e-5
# AdamW first-moment coefficient
va_robotwin_train_cfg.beta1 = 0.9
# AdamW second-moment coefficient (0.95 is more robust to gradient spikes than the default 0.999)
va_robotwin_train_cfg.beta2 = 0.95
# AdamW weight decay
va_robotwin_train_cfg.weight_decay = 0.1
# Learning-rate warmup steps (linear ramp up to learning_rate)
va_robotwin_train_cfg.warmup_steps = 10
# Per-GPU batch size (sequences are long, so usually 1; gradient accumulation makes up the effective batch)
va_robotwin_train_cfg.batch_size = 1 
# Gradient accumulation steps; effective batch = batch_size * num_gpus * this value
va_robotwin_train_cfg.gradient_accumulation_steps = 4
# Total training steps (optimizer steps)
va_robotwin_train_cfg.num_steps = 50000 
