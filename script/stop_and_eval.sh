#!/bin/bash
# Stop training cleanly once checkpoint_step_<TARGET> is fully saved,
# then run i2va eval on the given checkpoints sequentially and append
# results to report.md.
# Usage: setsid bash script/stop_and_eval.sh 10000 5000 10000 &
set -u
TARGET=${1:-10000}
shift || true
EVAL_STEPS=${@:-${TARGET}}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT_ROOT="${REPO}/train_out/checkpoints"
REPORT="${REPO}/report.md"
MARK="checkpoint_step_${TARGET}/transformer/config.json"

echo "[stop_and_eval] waiting for ${CKPT_ROOT}/${MARK}"
while true; do
    if [ -f "${CKPT_ROOT}/${MARK}" ]; then
        echo "[stop_and_eval] checkpoint ${TARGET} saved at $(date), stopping training"
        LAUNCHER=$(pgrep -f "torch.distributed.run" | head -1)
        if [ -n "${LAUNCHER}" ]; then
            kill "${LAUNCHER}"
            sleep 60
        fi
        for p in $(pgrep -f "wan_va.train"); do kill "$p" 2>/dev/null || true; done
        sleep 15
        for p in $(pgrep -f "wan_va.train"); do kill -9 "$p" 2>/dev/null || true; done
        echo "[stop_and_eval] training stopped at $(date)"
        break
    fi
    if ! pgrep -f "wan_va.train" > /dev/null; then
        echo "[stop_and_eval] training already exited at $(date)"
        break
    fi
    sleep 300
done

# wait for GPU memory to be released
sleep 60

source /opt/dtk/env.sh
cd "${REPO}"

for STEP in ${EVAL_STEPS}; do
    CKPT="${CKPT_ROOT}/checkpoint_step_${STEP}"
    if [ ! -f "${CKPT}/transformer/config.json" ]; then
        echo "[stop_and_eval] checkpoint ${STEP} missing, skip"
        continue
    fi
    LOG="/tmp/eval_step_${STEP}.log"
    echo "[stop_and_eval] evaluating checkpoint_step_${STEP} at $(date)"
    bash script/eval_checkpoint.sh "${STEP}" > "${LOG}" 2>&1
    RC=$?
    {
        echo ""
        echo "### i2va 评测 @ checkpoint_step_${STEP} ($(date '+%m-%d %H:%M'))"
        echo ""
        echo '```'
        grep -E "patched|USE I2AV|demo.mp4|OutOfMemory|Error|Traceback" "${LOG}" | head -10
        echo '```'
        if [ ${RC} -eq 0 ] && ls "${REPO}"/train_out/demo.mp4 >/dev/null 2>&1; then
            echo "- ✅ 输出: \`train_out/demo.mp4\` → 已另存为 \`train_out/eval/demo_step_${STEP}.mp4\`"
            cp "${REPO}/train_out/demo.mp4" "${REPO}/train_out/eval/demo_step_${STEP}.mp4"
            echo "- 判读: 视频应展示抓取白色马克杯→旋转→挂到深灰色架子;动作平滑无抖动"
        else
            echo "- ⚠️ 评测失败(exit ${RC}),完整日志: \`${LOG}\`"
        fi
    } >> "${REPORT}"
    echo "[stop_and_eval] checkpoint ${STEP} eval done rc=${RC}"
done
echo "[stop_and_eval] ALL DONE at $(date)"
