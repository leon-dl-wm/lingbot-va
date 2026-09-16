#!/bin/bash
# Wait for checkpoint_step_<STEP>, then run i2va eval and append results to report.md.
# Usage: nohup bash script/auto_eval_at_step.sh 5000 &
set -u
STEP=${1:-5000}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT="${REPO}/train_out/checkpoints/checkpoint_step_${STEP}"
REPORT="${REPO}/report.md"
LOG="/tmp/eval_step_${STEP}.log"

echo "[auto_eval] waiting for ${CKPT}"
while true; do
    if [ -f "${CKPT}/transformer/config.json" ]; then break; fi
    if ! pgrep -f "wan_va.train" > /dev/null; then
        echo "[auto_eval] training exited before step ${STEP}"
        exit 1
    fi
    sleep 300
done
# wait a bit for safetensors to finish writing
sleep 120
echo "[auto_eval] checkpoint ${STEP} ready, starting i2va eval at $(date)"

cd "${REPO}"
source /opt/dtk/env.sh
bash script/eval_checkpoint.sh "${STEP}" > "${LOG}" 2>&1
RC=$?

{
    echo ""
    echo "### i2va 评测 @ checkpoint_step_${STEP} ($(date '+%m-%d %H:%M'))"
    echo ""
    echo '```'
    grep -E "latent_loss|action_loss|patched|demo.mp4|Error|error|Traceback" "${LOG}" | tail -15
    echo '```'
    if [ ${RC} -eq 0 ] && ls "${REPO}"/train_out/demo.mp4 >/dev/null 2>&1; then
        echo "- 输出: \`train_out/demo.mp4\`(10 chunks 自回归视频-动作生成)"
        echo "- 判读: 视频应展示抓取白色马克杯→旋转→挂到深灰色架子;动作曲线平滑无抖动"
    else
        echo "- ⚠️ 评测失败(exit ${RC}),完整日志: \`${LOG}\`"
    fi
} >> "${REPORT}"
echo "[auto_eval] done, rc=${RC}"
