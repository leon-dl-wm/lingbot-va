#!/bin/bash
# Stop training cleanly once checkpoint_step_<TARGET> is fully saved.
# Usage: nohup bash script/stop_at_step.sh 10000 &
set -u
TARGET=${1:-10000}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT_ROOT="${REPO}/train_out/checkpoints"
MARK="checkpoint_step_${TARGET}/transformer/config.json"

echo "[stop_at_step] waiting for ${CKPT_ROOT}/${MARK}"
while true; do
    if [ -f "${CKPT_ROOT}/${MARK}" ]; then
        echo "[stop_at_step] checkpoint ${TARGET} saved at $(date), stopping training"
        # graceful: SIGTERM to torchrun launcher; workers follow
        LAUNCHER=$(pgrep -f "torch.distributed.run" | head -1)
        if [ -n "${LAUNCHER}" ]; then
            kill "${LAUNCHER}"
            sleep 30
        fi
        # ensure all training procs are gone
        pgrep -f "wan_va.train" | xargs -r kill 2>/dev/null
        sleep 10
        pgrep -f "wan_va.train" | xargs -r kill -9 2>/dev/null
        echo "[stop_at_step] training stopped at $(date)"
        break
    fi
    if ! pgrep -f "wan_va.train" > /dev/null; then
        echo "[stop_at_step] training already exited at $(date)"
        break
    fi
    sleep 300
done
