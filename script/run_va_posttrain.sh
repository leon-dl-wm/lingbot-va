#!/usr/bin/bash
# =============================================================================
# Training launch script: starts wan_va/train.py post-training in distributed
# mode via torchrun.
#
# Usage:
#   bash script/run_va_posttrain.sh                          # default robotwin_train config, 8 GPUs
#   NGPU=1 bash script/run_va_posttrain.sh                   # single GPU (combine with gradient accumulation)
#   CONFIG_NAME=libero_train bash script/run_va_posttrain.sh # switch training task config
#   bash script/run_va_posttrain.sh --learning_rate 5e-5     # append hydra-style command-line overrides
#
# Key environment variables (all overridable from outside):
#   NGPU                  GPUs per node (default 8)
#   MASTER_PORT           torchrun distributed master port (default 29501)
#   PORT                  reserved port variable (not used directly by this script)
#   LOG_RANK              only show logs of this rank (default 0)
#   TORCHFT_LIGHTHOUSE    lighthouse service address for torchft fault-tolerant training (default localhost:29510)
#   CONFIG_NAME           hydra config name, one of the *_train keys in VA_CONFIGS
#                         (robotwin_train / libero_train / demo_train)
#   WANDB_*               wandb logging service settings (API key / base URL / team / project)
# =============================================================================

set -x

umask 007
 
# GPUs per node (override with the NGPU env var, default 8)
NGPU=${NGPU:-"8"}
# torchrun distributed master port (pick a different one when running multiple jobs on the same machine)
MASTER_PORT=${MASTER_PORT:-"29501"}
# Reserved port variable (not used directly by this script)
PORT=${PORT:-"1106"}
# Only echo logs of this rank (default rank0, avoids interleaved multi-GPU logs)
LOG_RANK=${LOG_RANK:-"0"}
# Lighthouse service address for torchft fault-tolerant training (node failure recovery)
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
# Hydra config name: one of the *_train keys of VA_CONFIGS in configs/__init__.py
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train

# All positional arguments of this script are passed through to the training entry as hydra-style config overrides (e.g. --learning_rate 5e-5)
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# wandb training-curve logging settings (replace with your own key/URL/team/project; if wandb is not needed, disable enable_wandb in the config)
export WANDB_API_KEY="your key"
export WANDB_BASE_URL="your url"
export WANDB_TEAM_NAME="your team name"
export WANDB_PROJECT="your project"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
# Disable HF tokenizer multiprocessing (avoids deadlock warnings when DataLoader forks workers)
export TOKENIZERS_PARALLELISM=false
# Locate the repo root and prepend the bundled virtualenv va_env's bin to PATH
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${REPO_ROOT}/va_env/bin:$PATH"
# Launch distributed training:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  use expandable segments to mitigate fragmentation OOM
#   TORCHFT_LIGHTHOUSE                               torchft fault-tolerance service address
#   --nproc_per_node      processes per node = number of GPUs
#   --local-ranks-filter  only output logs of the given rank
#   --master_port         distributed master port
#   --tee 3               forward worker stdout/stderr to the launcher terminal
#   -m wan_va.train       run the training entry as a module
#   --config-name         select the training config from VA_CONFIGS; command-line overrides appended at the end
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
