#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"8"}
# Multi-node (TI-ONE 任务式建模): NNODES = worker count, NODE_RANK = this worker's
# index, MASTER_ADDR = worker-0 address. Platform injects these automatically.
NNODES=${NNODES:-"1"}
NODE_RANK=${NODE_RANK:-"0"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# WandB: set WANDB_MODE=offline for credential-free local logging (sync later with
# `wandb sync`), or fill real values below for online logging.
export WANDB_MODE=${WANDB_MODE:-"offline"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your key"}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"your url"}
export WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-"your team name"}
export WANDB_PROJECT=${WANDB_PROJECT:-"va_robotwin"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"robotwin_train"}

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## torchrun rendezvous setting (multi-node when NNODES > 1)
dist_args="--nproc_per_node=${num_gpu} --master_port ${master_port}"
if [ "${NNODES}" -gt 1 ]; then
    dist_args="${dist_args} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR}"
fi

## cmd setting
export TOKENIZERS_PARALLELISM=false
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# TI-ONE task containers may start in an arbitrary cwd - make wan_va importable
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${REPO_ROOT}/va_env/bin:$PATH"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    ${dist_args} \
    --local-ranks-filter=${log_rank} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
