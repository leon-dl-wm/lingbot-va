# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""LingBot-VA top-level package: an autoregressive diffusion "world model + action generation"
framework built on the Wan2.2 video DiT.

Package layout (position in the overall architecture):
- ``configs``: global / task-level hyperparameters (frame_chunk_size, attn_window, action_dim=30, etc.);
- ``modules``: algorithm core; ``model.py`` defines WanTransformer3DModel (interleaved video-action
  sequence, block-causal masking, KV cache) and the FlexAttention implementation;
- ``distributed``: FSDP sharding and distributed utilities (training-side infrastructure);
- ``dataset``: LeRobot v2.1 latent dataset reader (videos are pre-encoded offline by the Wan2.2 VAE);
- ``train.py``: training entry point (Diffusion Forcing noise injection + flow-matching loss);
- ``wan_va_server.py``: inference server (AR rollout + KV-cache state machine + asynchronous
  execution protocol).
"""
from . import configs, distributed, modules