#!/bin/bash
# Evaluate a training checkpoint via i2va demo (image -> video-action generation).
# Usage: bash script/eval_checkpoint.sh <step>   e.g. bash script/eval_checkpoint.sh 5000
set -eu
STEP=${1:?usage: eval_checkpoint.sh <step>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=${BASE:-/home/tione/notebook/model/lingbot-va-base}
CKPT_ROOT=${CKPT_ROOT:-${REPO}/train_out/checkpoints}
EVAL_ROOT=${EVAL_ROOT:-${REPO}/train_out/eval}
CKPT="${CKPT_ROOT}/checkpoint_step_${STEP}"
EVAL_DIR="${EVAL_ROOT}/checkpoint_step_${STEP}"

[ -d "${CKPT}/transformer" ] || { echo "checkpoint ${CKPT} not found"; exit 1; }

# Build eval model dir: symlink frozen components from base, copy transformer with attn_mode=torch
mkdir -p "${EVAL_DIR}"
for d in vae text_encoder tokenizer; do
    [ -e "${EVAL_DIR}/${d}" ] || ln -s "${BASE}/${d}" "${EVAL_DIR}/${d}"
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
NGPU=1 CONFIG_NAME='robotwin_i2av_eval' \
    EVAL_MODEL_PATH="${EVAL_DIR}" \
    MASTER_PORT=${MASTER_PORT:-29699} \
    bash script/run_launch_va_server_sync.sh > /tmp/i2va_server.log 2>&1
RC=$?
tail -40 /tmp/i2va_server.log
echo "=== output ==="
if [ -f "${REPO}/train_out/demo.mp4" ]; then
    mkdir -p "${EVAL_ROOT}"
    cp "${REPO}/train_out/demo.mp4" "${EVAL_ROOT}/demo_step_${STEP}.mp4"
    echo "demo archived: ${EVAL_ROOT}/demo_step_${STEP}.mp4"
else
    find "${REPO}" -maxdepth 2 -name "demo.mp4" -mmin -60 2>/dev/null || echo "check train_out/demo.mp4"
fi
