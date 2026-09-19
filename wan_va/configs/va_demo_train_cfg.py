# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Post-training (posttrain) entry config for the demo task.

Inherits all inference hyperparameters from va_demo_cfg, then adds dataset paths and
optimizer hyperparameters. Launch with ``bash script/run_va_posttrain.sh`` and
``CONFIG_NAME=demo_train`` (hydra style; supports command-line overrides such as
``--learning_rate 5e-5``).
"""
from easydict import EasyDict
from .va_demo_cfg import va_demo_cfg
import os

va_demo_train_cfg = EasyDict(__name__='Config: VA demo train')
# Inherit the demo base config (model path, chunk/window, action space, norm_stat, etc.)
va_demo_train_cfg.update(va_demo_cfg)

# LeRobot latent dataset path (videos pre-encoded offline by the Wan2.2 VAE into latents/*.pth; text embeddings pre-computed as well)
va_demo_train_cfg.dataset_path = '/path/to/your/dataset'
# Empty text embedding file (empty_emb.pt): replaces the real prompt embedding when training the CFG unconditional branch
va_demo_train_cfg.empty_emb_path = os.path.join(va_demo_train_cfg.dataset_path, 'empty_emb.pt')
# Whether to log training curves (loss, grad_norm, etc.) to wandb
va_demo_train_cfg.enable_wandb = True
# Number of DataLoader worker processes; more workers load faster but use more RAM
va_demo_train_cfg.load_worker = 16
# Save a checkpoint every this many steps (written to save_root/checkpoints/)
va_demo_train_cfg.save_interval = 50
# Trigger manual garbage collection every this many steps (mitigates GPU/host memory fragmentation)
va_demo_train_cfg.gc_interval = 50
# Probability of replacing the text embedding with the empty embedding during training, to learn the CFG unconditional branch
va_demo_train_cfg.cfg_prob = 0.1

# Training parameters
# AdamW learning rate (larger lr for the demo post-training)
va_demo_train_cfg.learning_rate = 1e-4
# AdamW first-moment coefficient
va_demo_train_cfg.beta1 = 0.9
# AdamW second-moment coefficient (0.95 is more robust to gradient spikes than the default 0.999)
va_demo_train_cfg.beta2 = 0.95
# AdamW weight decay
va_demo_train_cfg.weight_decay = 1e-1
# Learning-rate warmup steps (linear ramp up to learning_rate)
va_demo_train_cfg.warmup_steps = 10
# Per-GPU batch size (sequences are long, so usually 1; gradient accumulation makes up the effective batch)
va_demo_train_cfg.batch_size = 1 
# Gradient accumulation steps; effective batch = batch_size * num_gpus * this value
va_demo_train_cfg.gradient_accumulation_steps = 8
# Total training steps (optimizer steps)
va_demo_train_cfg.num_steps = 2000 
