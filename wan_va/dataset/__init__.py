# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Dataset subpackage: exports the LeRobot v2.1 latent dataset reader.

``MultiLatentLeRobotDataset`` aggregates all LeRobot repos discovered under ``dataset_path``
into a single torch Dataset (global idx -> (sub-dataset, local idx) mapping), consumed by the
DataLoader in ``wan_va/train.py``. Videos are read as offline pre-extracted VAE latents (.pth),
and text embeddings (UMT5) are likewise pre-computed offline.
"""
from .lerobot_latent_dataset import MultiLatentLeRobotDataset

__all__ = [
    'MultiLatentLeRobotDataset'
]