#!/bin/bash
# End-to-end validation pipeline: robotwin post-training (short) + checkpoint eval.
#
# What it does:
#   1. robotwin_train post-training for VAL_STEPS (default 1000) steps using
#      config 'robotwin_train_val' (identical to real training, just shorter)
#   2. waits for checkpoint_step_<VAL_STEPS> to be saved
#   3. runs i2va evaluation on that checkpoint and archives the demo video
#
# Usage:
#   bash script/run_validation.sh                         # 8 GPUs, detached
#   NGPU=2 bash script/run_validation.sh                  # fewer GPUs
#   VALIDATION_DETACHED=0 bash script/run_validation.sh   # foreground
#   PREFLIGHT_ONLY=1 bash script/run_validation.sh       # preflight checks only
#   FORCE=1 bash script/run_validation.sh                 # retrain even if ckpt exists
#
# Env overrides:
#   NGPU (8)  VAL_STEPS (500)  MASTER_PORT (29505)  SAVE_ROOT (train_out_val)
#   MODEL_PATH  DATASET_PATH    LOG (/tmp/validation.log)
#
# Artifacts:
#   ${SAVE_ROOT}/checkpoints/checkpoint_step_{500,1000}/   trained checkpoints
#   ${SAVE_ROOT}/eval/checkpoint_step_1000/                eval model dir
#   ${SAVE_ROOT}/eval/demo_step_1000.mp4                   generated demo video
#   /tmp/validation.log, /tmp/validation_train.log         logs
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NGPU=${NGPU:-8}
VAL_STEPS=${VAL_STEPS:-500}
MASTER_PORT=${MASTER_PORT:-29505}
SAVE_ROOT=${SAVE_ROOT:-${REPO}/train_out_val}
export MODEL_PATH=${MODEL_PATH:-/home/tione/notebook/model/lingbot-va-base}
export DATASET_PATH=${DATASET_PATH:-/home/tione/notebook/data/Robbyant/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}
export VAL_STEPS
FORCE=${FORCE:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
LOG=${LOG:-/tmp/validation.log}
TRAIN_LOG=/tmp/validation_train.log

# --- self-detach so the run survives shell session cleanup ---
if [ "${VALIDATION_DETACHED:-1}" = "1" ] && [ "${VALIDATION_CHILD:-0}" != "1" ]; then
    VALIDATION_CHILD=1 setsid bash "$0" "$@" < /dev/null > "${LOG}" 2>&1 &
    echo "[validation] launched detached (pid $!)"
    echo "[validation] log: ${LOG}   monitor: tail -f ${LOG}"
    exit 0
fi

log() { echo "[validation][$(date '+%m-%d %H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

# ================= Phase 0: preflight =================
log "Phase 0: preflight (NGPU=${NGPU}, VAL_STEPS=${VAL_STEPS}, SAVE_ROOT=${SAVE_ROOT})"
source /opt/dtk/env.sh || die "cannot source /opt/dtk/env.sh"

GPUS=$("${REPO}/va_env/bin/python" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
[ "${GPUS}" -ge "${NGPU}" ] || die "need ${NGPU} GPUs, visible: ${GPUS}"
log "GPUs visible: ${GPUS}"

[ -f "${MODEL_PATH}/transformer/config.json" ] || die "base model not found: ${MODEL_PATH}"
[ -f "${DATASET_PATH}/empty_emb.pt" ] || die "empty_emb.pt missing under ${DATASET_PATH}"
N_DSETS=$(find "${DATASET_PATH}" -name info.json 2>/dev/null | wc -l)
[ "${N_DSETS}" -gt 0 ] || die "no lerobot datasets (meta/info.json) under ${DATASET_PATH}"
log "model OK, lerobot datasets found: ${N_DSETS}"

USED=$(hy-smi --showmeminfo vram 2>/dev/null | grep "HCU\[0\]" | grep -oE "Used Memory \(MiB\): [0-9]+" | grep -oE "[0-9]+$" || echo 0)
[ "${USED}" -lt 10000 ] || log "WARN: GPU0 already using ${USED} MiB, training may OOM"
log "preflight passed"

if [ "${PREFLIGHT_ONLY}" = "1" ]; then log "PREFLIGHT_ONLY=1, exiting"; exit 0; fi

# ================= Phase 1: training =================
CKPT="${SAVE_ROOT}/checkpoints/checkpoint_step_${VAL_STEPS}"
LAST_LOSS="skipped (checkpoint existed)"
if [ -f "${CKPT}/transformer/config.json" ] && [ "${FORCE}" != "1" ]; then
    log "checkpoint ${CKPT} already exists, skip training (FORCE=1 to retrain)"
else
    log "Phase 1: robotwin post-training ${VAL_STEPS} steps on ${NGPU} GPUs"
    T0=$(date +%s)
    NGPU=${NGPU} CONFIG_NAME=robotwin_train_val MASTER_PORT=${MASTER_PORT} \
        bash "${REPO}/script/run_va_posttrain.sh" --save-root "${SAVE_ROOT}" 2>&1 | tee "${TRAIN_LOG}"
    RC=${PIPESTATUS[0]}
    T1=$(date +%s)
    [ "${RC}" -eq 0 ] || die "training exited rc=${RC} (see ${TRAIN_LOG})"
    [ -f "${CKPT}/transformer/config.json" ] || die "training finished but ${CKPT} missing"
    LAST_LOSS=$(grep -oE "latent_loss=[0-9.]+, action_loss=[0-9.]+, step=[0-9]+" "${TRAIN_LOG}" | tail -1 || true)
    log "training done in $(( (T1-T0)/60 )) min, checkpoint: ${CKPT}"
fi
log "last training loss: ${LAST_LOSS:-n/a}"

# ================= Phase 2: checkpoint eval =================
log "Phase 2: i2va eval on checkpoint_step_${VAL_STEPS} (waiting for GPU memory release)"
sleep 30
CKPT_ROOT="${SAVE_ROOT}/checkpoints" EVAL_ROOT="${SAVE_ROOT}/eval" \
    bash "${REPO}/script/eval_checkpoint.sh" "${VAL_STEPS}" 2>&1 | tee /tmp/validation_eval.log
RC=${PIPESTATUS[0]}
DEMO="${SAVE_ROOT}/eval/demo_step_${VAL_STEPS}.mp4"
if [ "${RC}" -eq 0 ] && [ -f "${DEMO}" ]; then
    log "eval OK, demo: ${DEMO}"
else
    die "eval failed (rc=${RC}), see /tmp/validation_eval.log and /tmp/i2va_server.log"
fi

# ================= Phase 3: summary =================
log "================ VALIDATION SUMMARY ================"
log "checkpoint : ${CKPT}"
log "last loss  : ${LAST_LOSS:-n/a}"
log "demo video : ${DEMO}"
log "train log  : ${TRAIN_LOG}"
log "eval log   : /tmp/validation_eval.log"
log "===================================================="
log "VALIDATION PASSED"
