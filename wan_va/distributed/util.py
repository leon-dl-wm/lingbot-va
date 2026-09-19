# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""General distributed-training utilities: process-group initialization, model device/sharding
configuration, and cross-rank tensor aggregation.

Position in the architecture: called by ``wan_va/train.py`` — ``init_distributed`` sets up the
NCCL process group in the ``run()`` entry point; ``_configure_model`` decides whether the model
goes through FSDP sharding (multi-GPU) or is simply moved to a single device;
``dist_mean`` / ``dist_max`` aggregate per-rank losses into global statistics for
wandb / progress-bar reporting.
"""
import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO

    Configure the model for the current runtime: when distributed is initialized, shard it with
    ``shard_fn`` (usually ``distributed.fsdp.shard_model``) — FSDP then manages parameter dtype
    and device placement itself; in single-process mode, simply cast the model to ``param_dtype``
    and move it to ``device``.

    Args:
        model: model to configure (e.g. WanTransformer3DModel).
        shard_fn: FSDP sharding function with signature ``model -> sharded model``.
        param_dtype: parameter precision for the single-GPU path (e.g. bf16).
        device: target device for the single-GPU path (e.g. ``cuda:0``).
        eval_mode: if True, switch to eval mode and freeze gradients (inference); pass False
            for training.

    Returns:
        The configured model.
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    # Synchronize all ranks before sharding so every process enters fully_shard together
    # (it is a collective operation).
    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    """Initialize the NCCL process group (torchrun injects RANK/WORLD_SIZE etc. via environment variables).

    Args:
        world_size: total number of processes.
        local_rank: process index on this machine; determines the bound GPU (``cuda:local_rank``).
        rank: global process rank.
    """
    # if world_size > 1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size)

def dist_mean(local_tensor):
    """All-reduce (average) a tensor across ranks (returned unchanged in single-process mode); used to aggregate the global mean loss."""
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    """All-reduce (max) a tensor across ranks (returned unchanged in single-process mode); used to monitor the worst-rank loss."""
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
