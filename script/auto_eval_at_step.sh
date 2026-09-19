#!/bin/bash
# Wait for checkpoint_step_<STEP>, then run i2va eval and append results to report.md.
# Usage: nohup bash script/auto_eval_at_step.sh 5000 &
# =============================================================================
# Auto-eval-at-step script (background daemon).
#
# Function: polls until train_out/checkpoints/checkpoint_step_<STEP> is fully saved,
#       then runs eval_checkpoint.sh for one i2va eval (image -> video-action generation)
#       and appends a key-log summary to report.md, enabling unattended checkpoint
#       quality checks during training.
#
# Usage: nohup bash script/auto_eval_at_step.sh 5000 &   (5000 is the target step; defaults to 5000 when omitted)
#
# Key environment variables/paths:
#   STEP    1st positional argument, the checkpoint step number to evaluate
#   REPO    repository root (derived automatically from the script location)
#   CKPT    the checkpoint directory to wait for
#   REPORT  markdown report file the eval results are appended to
#   LOG     full eval log (/tmp/eval_step_<STEP>.log)
# =============================================================================
set -u
# Target step (1st positional argument, default 5000)
STEP=${1:-5000}
# Repository root: one level above the script's directory
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Checkpoint directory to wait for (training writes it periodically per save_interval)
CKPT="${REPO}/train_out/checkpoints/checkpoint_step_${STEP}"
# Report file the eval summary is appended to
REPORT="${REPO}/report.md"
# Full eval log path
LOG="/tmp/eval_step_${STEP}.log"

# Poll for the checkpoint: transformer/config.json is among the last files written by the
# save flow, so its existence means the weights for this step are saved; if the training
# process has already exited, fail out
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
# Wait another 2 minutes to make sure the safetensors weight shards are fully flushed to disk (config.json may be written before the weights)
sleep 120
echo "[auto_eval] checkpoint ${STEP} ready, starting i2va eval at $(date)"

# Switch to the repo root, load the DTK (Hygon GPU) runtime environment, then run the single-checkpoint eval script;
# full output is redirected to the log file and RC records the eval exit code
cd "${REPO}"
source /opt/dtk/env.sh
bash script/eval_checkpoint.sh "${STEP}" > "${LOG}" 2>&1
RC=$?

# Append the eval summary to report.md: title (with step and timestamp) + key loss/error lines from the log (up to 15)
# + result interpretation (on success: demo.mp4 path and expected footage; on failure: exit code and log path)
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
