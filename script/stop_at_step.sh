#!/bin/bash
# Stop training cleanly once checkpoint_step_<TARGET> is fully saved.
# Usage: nohup bash script/stop_at_step.sh 10000 &
# =============================================================================
# Stop-training-at-step script (background daemon): once the target step's checkpoint
# is fully saved, gracefully terminate the training processes — avoids "overshooting"
# (wasting machine time) and avoids losing the latest weights by killing manually.
#
# Usage: nohup bash script/stop_at_step.sh 10000 &   (10000 is the target step, default 10000)
#
# Key environment variables/paths:
#   TARGET     1st positional argument; stop training after this step's checkpoint is fully saved
#   REPO       repository root (derived automatically)
#   CKPT_ROOT  checkpoint root directory (train_out/checkpoints)
#   MARK       marker file used to decide "save complete" (transformer/config.json is written last)
# =============================================================================
set -u
# Target step (1st positional argument, default 10000)
TARGET=${1:-10000}
# Repository root: one level above the script's directory
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Checkpoint root directory
CKPT_ROOT="${REPO}/train_out/checkpoints"
# "Save complete" marker: transformer/config.json existing means the checkpoint is fully flushed to disk
MARK="checkpoint_step_${TARGET}/transformer/config.json"

# Poll every 5 minutes; exit directly if the training process has already terminated on its own
echo "[stop_at_step] waiting for ${CKPT_ROOT}/${MARK}"
while true; do
    if [ -f "${CKPT_ROOT}/${MARK}" ]; then
        echo "[stop_at_step] checkpoint ${TARGET} saved at $(date), stopping training"
        # graceful: SIGTERM to torchrun launcher; workers follow
        # Graceful stop: send SIGTERM (default signal) to the torchrun launcher first; workers follow it down
        LAUNCHER=$(pgrep -f "torch.distributed.run" | head -1)
        if [ -n "${LAUNCHER}" ]; then
            kill "${LAUNCHER}"
            sleep 30
        fi
        # ensure all training procs are gone
        # Fallback cleanup: if training processes remain, SIGTERM them first, wait 10 s, then SIGKILL
        pgrep -f "wan_va.train" | xargs -r kill 2>/dev/null
        sleep 10
        pgrep -f "wan_va.train" | xargs -r kill -9 2>/dev/null
        echo "[stop_at_step] training stopped at $(date)"
        break
    fi
    # Training process no longer running (finished normally or crashed): nothing to stop, exit the loop
    if ! pgrep -f "wan_va.train" > /dev/null; then
        echo "[stop_at_step] training already exited at $(date)"
        break
    fi
    sleep 300
done
