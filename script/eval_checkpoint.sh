#!/bin/bash
# Evaluate a training checkpoint via i2va demo (image -> video-action generation).
# Usage: bash script/eval_checkpoint.sh <step>   e.g. bash script/eval_checkpoint.sh 5000
# =============================================================================
# Single-checkpoint i2va eval script: runs one "image-to-video-action" (i2va) offline
# inference with the transformer weights saved by training, producing
# train_out/demo.mp4 for manual inspection.
#
# Usage: bash script/eval_checkpoint.sh <step>   (e.g. bash script/eval_checkpoint.sh 5000)
#
# Key environment variables/paths:
#   STEP      1st positional argument (required), checkpoint step number
#   REPO      repository root (derived automatically)
#   BASE      base model directory (frozen components vae/text_encoder/tokenizer are symlinked from here)
#   CKPT      training checkpoint directory (train_out/checkpoints/checkpoint_step_<STEP>)
#   EVAL_DIR  assembled eval model dir: symlinks + a copied transformer with attn_mode patched
#   NGPU=1 / CONFIG_NAME=robotwin_i2av_eval / EVAL_MODEL_PATH
#             passed to run_launch_va_server_sync.sh: single GPU, eval config, model path
# =============================================================================
set -eu
# 1st positional argument: checkpoint step number; prints usage and exits when missing
STEP=${1:?usage: eval_checkpoint.sh <step>}
# Repository root: one level above the script's directory
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Base model directory (provides the frozen vae / text_encoder / tokenizer components needed for eval)
BASE=/home/tione/notebook/model/lingbot-va-base
# Training checkpoint directory to evaluate
CKPT="${REPO}/train_out/checkpoints/checkpoint_step_${STEP}"
# Assembled eval model directory (avoids polluting the original checkpoint)
EVAL_DIR="${REPO}/train_out/eval/checkpoint_step_${STEP}"

# Precondition check: the checkpoint's transformer subdirectory must exist
[ -d "${CKPT}/transformer" ] || { echo "checkpoint ${CKPT} not found"; exit 1; }

# Build eval model dir: symlink frozen components from base, copy transformer with attn_mode=torch
# Build the eval model dir: symlink vae/text_encoder/tokenizer from the base model (saves disk/time),
# while the transformer is fully copied from the checkpoint (the next step patches its config in place)
mkdir -p "${EVAL_DIR}"
for d in vae text_encoder tokenizer; do
    [ -e "${EVAL_DIR}/${d}" ] || ln -s "${BASE}/${d}" "${EVAL_DIR}/${d}"
done
rm -rf "${EVAL_DIR}/transformer"
cp -r "${CKPT}/transformer" "${EVAL_DIR}/transformer"
# Inline Python patches attn_mode in transformer/config.json to "torch":
# training uses flex (FlexAttention+compile, train-only), inference must switch to the SDPA implementation
python - "${EVAL_DIR}/transformer/config.json" <<'EOF'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
cfg["attn_mode"] = "torch"   # inference mode (flex is train-only)
json.dump(cfg, open(p, "w"), indent=2)
print("patched attn_mode ->", cfg["attn_mode"])
EOF

# Switch to the repo root, load the DTK (Hygon GPU) runtime environment, then launch the single-GPU i2va eval:
# CONFIG_NAME=robotwin_i2av_eval selects the eval config (offload enabled to save memory),
# EVAL_MODEL_PATH injects the assembled model directory; only the last 30 log lines are echoed
cd "${REPO}"
source /opt/dtk/env.sh
echo "=== running i2va with checkpoint_step_${STEP} ==="
NGPU=1 CONFIG_NAME='robotwin_i2av_eval' \
    EVAL_MODEL_PATH="${EVAL_DIR}" \
    bash script/run_launch_va_server_sync.sh 2>&1 | tail -30
# Result check: prefer listing train_out/demo.mp4; if missing, search for any demo.mp4 generated within the last 60 minutes
echo "=== output ==="
ls -la "${REPO}/train_out/demo.mp4" 2>/dev/null || find "${REPO}" -maxdepth 2 -name "demo.mp4" -mmin -60 2>/dev/null || echo "check train_out/demo.mp4"
