# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""FSDP sharding and activation-checkpointing utilities (training-side infrastructure).

Responsibilities:
- ``apply_ac``: wraps every transformer block with activation checkpointing, trading compute
  for memory (intermediate activations are not stored in the forward pass and are recomputed
  in the backward pass);
- ``shard_model``: shards parameters / gradients / optimizer states with the PyTorch FSDP2
  ``fully_shard`` API and configures the mixed-precision policy (bf16 parameters for compute,
  fp32 for gradient reduction to preserve precision);
- ``free_model``: drops the model reference and reclaims GPU memory.

Sharding granularity: each block's attn1/attn2/ffn submodules are fully_shard'ed first
(finer-grained communication-overlap units), then the whole block, then the root model.
Called by ``wan_va/train.py`` and ``distributed/util.py:_configure_model``.
"""
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    # Apply activation checkpointing to every transformer block: intermediate activations are
    # not kept in the forward pass and are recomputed in backward, trading compute for a large
    # reduction in peak memory for long-sequence training.
    # preserve_rng_state=False skips saving/restoring the RNG state (no random ops inside the
    # block need RNG reproduction here), removing that extra overhead.
    #
    # Args:
    #     model: WanTransformer3DModel; its ``blocks`` ModuleList entries are replaced in place
    #         with the wrapped blocks.
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.blocks[layer_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    """Shard the model with FSDP2 (``fully_shard``) and configure mixed precision.

    Sharding strategy: per-block attn1/attn2/ffn submodules -> whole block -> root model,
    registered bottom-up so that communication (all-gather / reduce-scatter) can overlap
    with computation. ``reshard_after_forward=True``: release the all-gathered full
    parameters right after the forward pass to save more memory (they are gathered again
    in the backward pass).

    Args:
        model: model to shard (WanTransformer3DModel).
        param_dtype: forward/backward compute precision, default bf16 (the master copy of
            the weights remains fp32).
        reduce_dtype: gradient reduce-scatter precision, default fp32 to avoid precision
            loss when aggregating bf16 gradients.

    Returns:
        The sharded model (the same object, now managed by FSDP).
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    """Drop the model reference and reclaim memory: trigger GC + empty the CUDA cache (used to free VRAM before swapping models or saving checkpoints)."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
