#!/bin/bash
# Append training snapshot to report.md every 30 min
# =============================================================================
# Training monitor script (background daemon): every 30 minutes, scrape the latest
# metric line from the training log (latent_loss / action_loss / step / grad_norm),
# the iteration speed (s/it) and GPU memory usage, and append them as a markdown table
# row to report.md; when the training process exits, one final row is recorded and the
# loop ends.
#
# Usage: nohup bash script/monitor_train.sh &
#
# Key environment variables/paths:
#   REPORT  markdown report file the snapshots are appended to
#   LOG     training log file (run_va_posttrain.sh output redirected here)
# =============================================================================
# Report file the monitoring snapshots are appended to
REPORT=/home/tione/notebook/code/lingbot-va/report.md
# Training log file (training output must be redirected here first, e.g. ... 2>&1 | tee /tmp/train.log)
LOG=/tmp/train.log
while true; do
    # Training process exited: record a final "exited" snapshot row and end the monitor loop
    if ! pgrep -f "wan_va.train" > /dev/null; then
        echo "| $(date '+%m-%d %H:%M') | 训练进程已退出 | - | - | - | - | - |" >> "$REPORT"
        break
    fi
    # Scrape the latest training metric line from the log: latent_loss (video-path loss), action_loss (action-path loss),
    # step (current step), grad_norm (gradient norm)
    STATS=$(grep -oE "latent_loss=[0-9.]+, action_loss=[0-9.]+, step=[0-9]+, grad_norm=[0-9.]+" "$LOG" | tail -1)
    # Scrape the iteration speed (s/it) from the tqdm progress bar; total 50000 steps matches the robotwin_train config
    SPEED=$(grep -oE "[0-9]+/50000 \[[0-9:]+<[0-9:]+, +([0-9.]+)s/it" "$LOG" | tail -1 | grep -oE "[0-9.]+s/it" | head -1)
    # Read GPU 0 memory usage (MiB) via hy-smi (Hygon GPU management tool); left empty when the command fails on non-Hygon environments
    MEM=$(hy-smi --showmeminfo vram 2>/dev/null | grep "HCU\[0\]" | grep -oE "Used Memory \(MiB\): [0-9]+" | grep -oE "[0-9]+$")
    if [ -n "$STATS" ]; then
        # Split the metric line into the four fields step / latent_loss / action_loss / grad_norm
        STEP=$(echo "$STATS" | grep -oE "step=[0-9]+" | cut -d= -f2)
        LL=$(echo "$STATS" | grep -oE "latent_loss=[0-9.]+" | cut -d= -f2)
        AL=$(echo "$STATS" | grep -oE "action_loss=[0-9.]+" | cut -d= -f2)
        GN=$(echo "$STATS" | grep -oE "grad_norm=[0-9.]+" | cut -d= -f2)
        # Append one markdown table row: time | step | video loss | action loss | grad norm | speed | memory
        echo "| $(date '+%m-%d %H:%M') | ${STEP} | ${LL} | ${AL} | ${GN} | ${SPEED:-?} | ${MEM:-?}MiB |" >> "$REPORT"
    fi
    # Sample once every 30 minutes
    sleep 1800
done
