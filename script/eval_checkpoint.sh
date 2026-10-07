#!/bin/bash
# Evaluate a training checkpoint via i2va demo (image -> video-action generation).
# Usage: bash script/eval_checkpoint.sh <step>   e.g. bash script/eval_checkpoint.sh 5000
set -eu
STEP=${1:?usage: eval_checkpoint.sh <step>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=${BASE:-${MODEL_PATH:-${STORAGE_MOUNT_PATH:?'STORAGE_MOUNT_PATH is empty'}/model/lingbot-va-base}}
CKPT_ROOT=${CKPT_ROOT:-${REPO}/train_out/checkpoints}
EVAL_ROOT=${EVAL_ROOT:-${REPO}/train_out/eval}
CKPT="${CKPT_ROOT}/checkpoint_step_${STEP}"
EVAL_DIR="${EVAL_ROOT}/checkpoint_step_${STEP}"

[ -d "${CKPT}/transformer" ] || { echo "checkpoint ${CKPT} not found"; exit 1; }

# Build eval model dir: symlink frozen components from base, copy transformer with attn_mode=torch
mkdir -p "${EVAL_DIR}"
for d in vae text_encoder tokenizer; do
    # Always refresh the symlinks: absolute links created under another mount
    # point (notebook /home/tione/notebook vs task /opt/ml/input/data) dangle
    # here, so [ -e ] is false yet plain ln -s fails with "File exists".
    if [ -d "${EVAL_DIR}/${d}" ] && [ ! -L "${EVAL_DIR}/${d}" ]; then continue; fi
    ln -sfn "${BASE}/${d}" "${EVAL_DIR}/${d}"
done
rm -rf "${EVAL_DIR}/transformer"
cp -r "${CKPT}/transformer" "${EVAL_DIR}/transformer"
python - "${EVAL_DIR}/transformer/config.json" <<'EOF'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
cfg["attn_mode"] = "torch"   # inference mode (flex is train-only)
json.dump(cfg, open(p, "w"), indent=2)
print("patched attn_mode ->", cfg["attn_mode"])
EOF

cd "${REPO}"
source /opt/dtk/env.sh
echo "=== running i2va with checkpoint_step_${STEP} ==="
# --save_root redirects server outputs (demo.mp4, real/) into EVAL_DIR; the
# config default is ./train_out, which must stay reserved for real training.
RC=0
NGPU=1 CONFIG_NAME='robotwin_i2av_eval' \
    EVAL_MODEL_PATH="${EVAL_DIR}" \
    MASTER_PORT=${MASTER_PORT:-29699} \
    bash script/run_launch_va_server_sync.sh --save_root "${EVAL_DIR}" > /tmp/i2va_server.log 2>&1 || RC=$?
tail -40 /tmp/i2va_server.log
echo "=== output ==="
if [ "${RC}" -ne 0 ]; then
    echo "i2va server failed (rc=${RC}), full log: /tmp/i2va_server.log"
    exit "${RC}"
fi
if [ -f "${EVAL_DIR}/demo.mp4" ]; then
    cp "${EVAL_DIR}/demo.mp4" "${EVAL_ROOT}/demo_step_${STEP}.mp4"
    echo "demo archived: ${EVAL_ROOT}/demo_step_${STEP}.mp4"
else
    echo "demo video missing: expected ${EVAL_DIR}/demo.mp4 (see /tmp/i2va_server.log)"
    exit 1
fi
