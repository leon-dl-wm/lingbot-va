# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference deployment utilities: multi-GPU coordination + websocket policy server wrapper.

With multi-GPU inference the model is FSDP-sharded across ranks, so every forward pass
must be executed by all ranks in sync. This module implements a master/worker
coordination via torch.distributed broadcasts: "rank 0 receives the request -> all ranks
execute it synchronously". Rank 0 runs the WebsocketPolicyServer exposing the
reset/compute_kv_cache/infer interface (shared by the evaluation client and real-robot
deployment), while the other ranks block in ``worker_loop`` waiting for broadcast
commands.
"""
import torch
import torch.distributed as dist

from .logging import logger
from .Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer


class DistributedModelWrapper:
    """
    TODO

    Wraps VA_Server into a "distributed-consistent" inference model: the
    WebsocketPolicyServer only talks to rank 0, and every infer call goes through
    ``distributed_infer`` which broadcasts to all ranks for synchronous execution,
    ensuring the collective communications of the FSDP-sharded model never deadlock.
    """

    def __init__(self, model, local_rank):
        """Args: model is a VA_Server instance; local_rank is this process's local rank."""
        self.model = model
        self.local_rank = local_rank

    def infer(self, obs):
        """Websocket request entry point: forwards to ``distributed_infer`` for multi-GPU synchronized inference."""
        return distributed_infer(self.model, obs, self.local_rank)


def distributed_infer(model, obs, local_rank):
    """
    TODO

    One distributed inference on the rank-0 side: first broadcast the "execute" command
    (cmd=1) and the request object obs, then call model.infer locally -- each worker rank
    receives the same broadcasts and synchronously runs the identical forward pass.

    Args:
        model: the VA_Server instance.
        obs: client request dict (reset/compute_kv_cache/infer three-way dispatch, see
            VA_Server.infer).
        local_rank: current local rank (must be 0).

    Returns:
        The result dict of model.infer(obs) on rank 0.
    """
    rank = dist.get_rank()
    assert rank == local_rank, "distributed_infer can only run at（rank 0)"

    # Two-stage broadcast: first a fixed-size cmd tensor (to wake the workers), then the
    # obs content pickled via broadcast_object_list
    cmd = torch.tensor(1,
                       dtype=torch.int64,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
    dist.broadcast(cmd, src=0)

    obj_list = [obs]
    dist.broadcast_object_list(obj_list, src=0)

    result = model.infer(obs)

    return result


def worker_loop(model, local_rank):
    """
    TODO

    Main loop for non-zero ranks: block waiting for rank 0's broadcast command -- on
    cmd=1 receive obs and synchronously execute model.infer (the result is discarded;
    only rank 0's return value goes back to the client); on cmd=-1 exit the loop.

    Args:
        model: the VA_Server instance (this rank's FSDP shard).
        local_rank: current local rank (used for logging only).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rank = dist.get_rank()

    while True:
        cmd = torch.zeros(1, dtype=torch.int64, device=device)
        dist.broadcast(cmd, src=0)
        cmd_val = cmd.item()

        if cmd_val == -1:
            # Received the exit command (broadcast after the server stops serving)
            break
        elif cmd_val == 1:
            # Received an inference command: take obs and run the same forward as rank 0
            # (keeping collective communications in sync)
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            obs = obj_list[0]
            _ = model.infer(obs)
        else:
            pass

    logger.info(f"[worker_loop] Rank {rank} exiting.")


def run_async_server_mode(model, local_rank, host, port):
    """Start the async websocket policy server (the top-level entry for multi-GPU coordination).

    Rank 0: wraps DistributedModelWrapper and serves forever; on exit it broadcasts
    cmd=-1 to tell all workers to finish. Other ranks: enter ``worker_loop`` and wait
    for commands.

    Args:
        model: the VA_Server instance.
        local_rank: current local rank.
        host / port: address and port the websocket service listens on.
    """
    logger.info("Running in ASYNC SERVER mode")
    if local_rank == 0:
        dist_model = DistributedModelWrapper(model, local_rank=local_rank)
        model_server = WebsocketPolicyServer(dist_model, host=host, port=port)
        model_server.serve_forever()

        cmd = torch.tensor(
            -1,
            dtype=torch.int64,
            device='cuda' if torch.cuda.is_available() else 'cpu')
        dist.broadcast(cmd, src=0)
    else:
        try:
            worker_loop(model, local_rank)
        except KeyboardInterrupt:
            logger.info(f"Rank {local_rank}: Shutting down")
