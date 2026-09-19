# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Distributed training infrastructure subpackage.

Contents:
- ``fsdp.py``: FSDP2 (``fully_shard``) model sharding + activation checkpointing;
- ``util.py``: process-group initialization, model configuration (sharding / device transfer),
  and cross-rank aggregation helpers (dist_mean / dist_max).

Used by ``wan_va/train.py`` as the training-side engineering foundation; the inference server
does not depend on the sharding logic in this subpackage.
"""
