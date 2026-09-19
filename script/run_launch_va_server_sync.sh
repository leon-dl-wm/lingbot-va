#!/usr/bin/bash
# =============================================================================
# Inference server launch script (synchronous/blocking): starts wan_va/wan_va_server.py
# via torchrun, serving websocket remote inference (eval clients / real robots connect
# via host:port); when CONFIG_NAME is one of the *_i2av* configs it instead runs offline
# image-to-video-action (i2va) inference.
#
# Usage:
#   bash script/run_launch_va_server_sync.sh                 # default robotwin server config, 8 GPUs
#   NGPU=1 CONFIG_NAME=robotwin_i2av_eval \
#       EVAL_MODEL_PATH=<model dir> bash script/run_launch_va_server_sync.sh
#                                                            # single-GPU checkpoint eval (this is how eval_checkpoint.sh calls it)
#   bash script/run_launch_va_server_sync.sh --attn_window 30 # append hydra-style command-line overrides
#
# Key environment variables (all overridable from outside):
#   NGPU                  GPUs per node (default 8; with multiple GPUs, sever_utils coordinates sharded inference)
#   MASTER_PORT           torchrun distributed master port (default 29501)
#   PORT                  reserved port variable (not used directly; the service port comes from cfg.port in the config)
#   LOG_RANK              only show logs of this rank (default 0)
#   TORCHFT_LIGHTHOUSE    torchft lighthouse address (generally unused for inference; keep the default)
#   CONFIG_NAME           hydra config name, one of the inference keys in VA_CONFIGS
#                         (robotwin / franka / demo / libero / *_i2av / *_i2av_eval)
#   EVAL_MODEL_PATH       only used by the robotwin_i2av_eval config: model directory to evaluate
# =============================================================================

set -x

umask 007
 
# GPUs per node (override with the NGPU env var, default 8)
NGPU=${NGPU:-"8"}
# torchrun distributed master port (pick a different one when colocated with a training job)
MASTER_PORT=${MASTER_PORT:-"29501"}
# Reserved port variable (not used directly; the websocket service port comes from cfg.port in the config, default 29536)
PORT=${PORT:-"1106"}
# Only echo logs of this rank (default rank0)
LOG_RANK=${LOG_RANK:-"0"}
# torchft lighthouse service address (generally unused for inference; keep the default)
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
# Hydra config name: one of the inference/eval keys of VA_CONFIGS in configs/__init__.py (default robotwin server mode)
CONFIG_NAME=${CONFIG_NAME:-"robotwin"}

# All positional arguments of this script are passed through to the inference entry as hydra-style config overrides
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
# Disable HF tokenizer multiprocessing (avoids fork-related warnings/deadlocks)
export TOKENIZERS_PARALLELISM=false
# Locate the repo root and prepend the bundled virtualenv va_env's bin to PATH
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${REPO_ROOT}/va_env/bin:$PATH"
# Launch the inference process:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  use expandable segments to mitigate fragmentation OOM
#   --nproc_per_node      processes per node = number of GPUs (multi-GPU inference is sharded/coordinated)
#   --local-ranks-filter  only output logs of the given rank
#   --master_port         distributed master port
#   --tee 3               forward worker stdout/stderr to the launcher terminal
#   -m wan_va.wan_va_server  inference entry (server mode stays resident; i2va mode exits when done)
#   --config-name         select the inference config from VA_CONFIGS; command-line overrides appended at the end
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.wan_va_server --config-name ${config_name} $overrides
