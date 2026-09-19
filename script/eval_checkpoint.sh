#!/bin/bash
# Evaluate a training checkpoint via i2va demo (image -> video-action generation).
# Usage: bash script/eval_checkpoint.sh <step>   e.g. bash script/eval_checkpoint.sh 5000
set -eu
STEP=${1:?usage: eval_checkpoint.sh <step>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=/home/tione/notebook/model/lingbot-va-base
CKPT="${REPO}/train_out/checkpoints/checkpoint_step_${STEP}"
EVAL_DIR="${REPO}/train_out/eval/checkpoint_step_${STEP}"

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
    bash script/run_launch_va_server_sync.sh 2>&1 | tail -30
echo "=== output ==="
ls -la "${REPO}/train_out/demo.mp4" 2>/dev/null || find "${REPO}" -maxdepth 2 -name "demo.mp4" -mmin -60 2>/dev/null || echo "check train_out/demo.mp4"
