#!/bin/bash
# Append training snapshot to report.md every 30 min
REPORT=/home/tione/notebook/code/lingbot-va/report.md
LOG=/tmp/train.log
while true; do
    if ! pgrep -f "wan_va.train" > /dev/null; then
        echo "| $(date '+%m-%d %H:%M') | 训练进程已退出 | - | - | - | - | - |" >> "$REPORT"
        break
    fi
    STATS=$(grep -oE "latent_loss=[0-9.]+, action_loss=[0-9.]+, step=[0-9]+, grad_norm=[0-9.]+" "$LOG" | tail -1)
    SPEED=$(grep -oE "[0-9]+/50000 \[[0-9:]+<[0-9:]+, +([0-9.]+)s/it" "$LOG" | tail -1 | grep -oE "[0-9.]+s/it" | head -1)
    MEM=$(hy-smi --showmeminfo vram 2>/dev/null | grep "HCU\[0\]" | grep -oE "Used Memory \(MiB\): [0-9]+" | grep -oE "[0-9]+$")
    if [ -n "$STATS" ]; then
        STEP=$(echo "$STATS" | grep -oE "step=[0-9]+" | cut -d= -f2)
        LL=$(echo "$STATS" | grep -oE "latent_loss=[0-9.]+" | cut -d= -f2)
        AL=$(echo "$STATS" | grep -oE "action_loss=[0-9.]+" | cut -d= -f2)
        GN=$(echo "$STATS" | grep -oE "grad_norm=[0-9.]+" | cut -d= -f2)
        echo "| $(date '+%m-%d %H:%M') | ${STEP} | ${LL} | ${AL} | ${GN} | ${SPEED:-?} | ${MEM:-?}MiB |" >> "$REPORT"
    fi
    sleep 1800
done
