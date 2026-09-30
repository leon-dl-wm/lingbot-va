#!/bin/bash
# End-to-end validation pipeline: robotwin post-training (short) + checkpoint eval.
#
# What it does:
#   1. robotwin_train post-training for VAL_STEPS (default 1000) steps using
#      config 'robotwin_train_val' (identical to real training, just shorter)
#   2. waits for checkpoint_step_<VAL_STEPS> to be saved
#   3. runs i2va evaluation on that checkpoint and archives the demo video
#   4. auto-generates final_report.md (script/gen_final_report.py) into
#      ${SAVE_ROOT}/ and the repo root
#
# Usage:
#   bash script/run_validation.sh                         # single node, detached
#   NGPU=2 bash script/run_validation.sh                  # fewer GPUs
#   VALIDATION_DETACHED=0 bash script/run_validation.sh   # foreground
#   PREFLIGHT_ONLY=1 bash script/run_validation.sh       # preflight checks only
#   FORCE=1 bash script/run_validation.sh                 # retrain even if ckpt exists
#
# Tencent TI-ONE 任务式建模 (task-mode training, e.g. HCC-BW1000 x3 nodes,
# 8 GPUs/node = 24 GPUs total):
#   Use this script as the task's 启动命令 (start command) on every worker:
#       bash ${STORAGE_MOUNT_PATH}/code/lingbot-va/script/run_validation.sh
#   (STORAGE_MOUNT_PATH defaults to /home/tione/notebook)
#   The platform injects MASTER_ADDR / MASTER_PORT / WORLD_SIZE (node count) /
#   RANK (node index); the script detects them automatically, runs foreground,
#   trains multi-node via torchrun, and only worker-0 runs the eval phase.
#   Code/data/checkpoints must live on the shared CFS mount.
#
# STORAGE_MOUNT_PATH (the CFS Turbo mount root) is injected by TI-ONE in
# notebooks (/home/tione/notebook); task-mode jobs (/opt/ml/input/data) may
# not export it, so the script auto-detects it from the repo path
# (<mount>/code/lingbot-va) or known mount points, and exits if none found.
# All resolved env vars are printed in an "ENV" block at startup.
#
# Env overrides:
#   STORAGE_MOUNT_PATH (auto-detected if unset)  shared CFS Turbo mount root
#   Hardware (GPU count/name/VRAM, CPU cores, memory) is auto-detected on each
#   node (cgroup-aware) and logged; NGPU defaults to the detected GPU count.
#   NGPU (auto)  NNODES (auto)  NODE_RANK (auto)  MASTER_ADDR (auto)
#   VAL_STEPS (500)  MASTER_PORT (29505)  SAVE_ROOT (train_out_val)
#   MODEL_PATH  DATASET_PATH    LOG (/tmp/validation.log)
#
# Artifacts:
#   ${SAVE_ROOT}/checkpoints/checkpoint_step_{500,1000}/   trained checkpoints
#   ${SAVE_ROOT}/eval/checkpoint_step_1000/                eval model dir
#   ${SAVE_ROOT}/eval/demo_step_1000.mp4                   generated demo video
#   ${SAVE_ROOT}/loss_curves.png                           loss curves (report)
#   ${SAVE_ROOT}/final_report.md + <repo>/final_report.md  auto-generated report
#   /tmp/validation.log, /tmp/validation_train.log         logs
set -uo pipefail

SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO}"   # TI-ONE task containers may start in an arbitrary cwd

# Shared CFS Turbo mount root. TI-ONE injects STORAGE_MOUNT_PATH in notebooks,
# but task-mode containers may not export it even though the CFS is mounted
# (e.g. at /opt/ml/input/data). Auto-detect: the repo lives at
# <mount>/code/lingbot-va in both environments; fall back to known mounts.
if [ -z "${STORAGE_MOUNT_PATH:-}" ] \
    && [ "$(basename "${REPO}")" = "lingbot-va" ] \
    && [ "$(basename "$(dirname "${REPO}")")" = "code" ]; then
    STORAGE_MOUNT_PATH=$(dirname "$(dirname "${REPO}")")
    echo "[validation] STORAGE_MOUNT_PATH not set; derived from repo path: ${STORAGE_MOUNT_PATH}"
fi
if [ -z "${STORAGE_MOUNT_PATH:-}" ]; then
    for cand in /opt/ml/input/data /home/tione/notebook; do
        if [ -d "${cand}" ]; then
            STORAGE_MOUNT_PATH="${cand}"
            echo "[validation] STORAGE_MOUNT_PATH not set; using ${cand}"
            break
        fi
    done
fi
if [ -z "${STORAGE_MOUNT_PATH:-}" ]; then
    echo "[validation] FATAL: STORAGE_MOUNT_PATH is empty - attach the CFS Turbo storage to the TI-ONE task" >&2
    exit 1
fi
export STORAGE_MOUNT_PATH

# --- TI-ONE 任务式建模 detection ---
# The platform injects MASTER_ADDR / MASTER_PORT / WORLD_SIZE (node count) /
# RANK (node index). Outside a TI-ONE task these are unset -> single node.
# Only ever *read* them here: the detach decision below depends on it, and
# synthesising a MASTER_ADDR default before the re-exec would make the detached
# child misdetect a TI-ONE task.
TIONE_TASK=0
[ -n "${MASTER_ADDR:-}" ] && TIONE_TASK=1

NGPU=${NGPU:-auto}   # resolved to visible GPU count in preflight
# optional positional arg: VAL_STEPS, e.g. `bash run_validation.sh 10000`
if [ $# -ge 1 ] && [[ "${1}" =~ ^[0-9]+$ ]]; then VAL_STEPS=${1}; shift; fi
VAL_STEPS=${VAL_STEPS:-500}
# checkpoints are only written at multiples of save_interval (500), so any other
# VAL_STEPS would never produce checkpoint_step_<VAL_STEPS>
if [ "${VAL_STEPS}" -ge 500 ] && [ $(( VAL_STEPS % 500 )) -ne 0 ]; then
    VAL_STEPS=$(( (VAL_STEPS + 499) / 500 * 500 ))
    echo "[validation] VAL_STEPS rounded up to ${VAL_STEPS} (checkpoint save interval is 500)" >&2
fi
MASTER_PORT=${MASTER_PORT:-29505}
export MASTER_PORT VAL_STEPS
SAVE_ROOT=${SAVE_ROOT:-${REPO}/train_out_val}
export MODEL_PATH=${MODEL_PATH:-${STORAGE_MOUNT_PATH}/model/lingbot-va-base}
export DATASET_PATH=${DATASET_PATH:-${STORAGE_MOUNT_PATH}/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}
FORCE=${FORCE:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
LOG=${LOG:-/tmp/validation.log}
TRAIN_LOG=/tmp/validation_train.log

# --- self-detach so the run survives shell session cleanup ---
# Inside a TI-ONE task the platform tracks the start command, so run foreground.
if [ "${VALIDATION_DETACHED:-$([ "${TIONE_TASK}" = "1" ] && echo 0 || echo 1)}" = "1" ] \
    && [ "${VALIDATION_CHILD:-0}" != "1" ]; then
    VALIDATION_CHILD=1 setsid bash "${SELF}" "$@" < /dev/null > "${LOG}" 2>&1 &
    echo "[validation] launched detached (pid $!)"
    echo "[validation] log: ${LOG}   monitor: tail -f ${LOG}"
    exit 0
fi

# --- multi-node resolution (after the detach: see comment above) ---
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NNODES NODE_RANK MASTER_ADDR

log() { echo "[validation][$(date '+%m-%d %H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

# ================= env dump =================
# Print every env var this script reads or forwards, up front, so a failed
# TI-ONE task log shows the full context without needing a second run.
print_var() {
    local name=$1 val
    val=${!name-}
    [ -n "${val}" ] || val="<unset>"
    printf '  %-26s = %s\n' "${name}" "${val}"
}
log "===================== ENV ====================="
log "[resolved by this script]"
for v in REPO TIONE_TASK NNODES NODE_RANK NGPU VAL_STEPS SAVE_ROOT \
         MODEL_PATH DATASET_PATH STORAGE_MOUNT_PATH FORCE PREFLIGHT_ONLY \
         LOG TRAIN_LOG VALIDATION_CHILD; do print_var "${v}"; done
log "[TI-ONE / distributed platform]"
for v in MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK \
         K8S_JOB_TYPE TI_JOB_SOURCE TI_CLOUD_NOTEBOOK_NAME TI_NODE_ID; do print_var "${v}"; done
log "[storage / paths]"
for v in STORAGE_MOUNT_PATH WORK_DIR USER_STORAGE_MOUNT_INFO HOME PWD; do print_var "${v}"; done
log "[hardware / accelerator]"
for v in GPU_NAME TI_GPU_PROVIDER AMDGPU_TARGETS \
         CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do print_var "${v}"; done
log "[network / runtime]"
for v in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE \
         CONFIG_NAME TORCHFT_LIGHTHOUSE TOKENIZERS_PARALLELISM \
         PYTORCH_CUDA_ALLOC_CONF PYTHONPATH; do print_var "${v}"; done
log "[wandb]"
for v in WANDB_MODE WANDB_PROJECT WANDB_RUN_NAME WANDB_BASE_URL; do print_var "${v}"; done
log "==============================================="

# ================= Phase 0: preflight =================
source /opt/dtk/env.sh || die "cannot source /opt/dtk/env.sh"

# --- hardware auto-detection: GPU / CPU / MEM (cgroup-aware) ---
detect_cpu_cores() {
    local q p
    if [ -f /sys/fs/cgroup/cpu.max ]; then                     # cgroup v2
        read -r q p < /sys/fs/cgroup/cpu.max
        if [ "${q}" != "max" ] && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
            echo $(( (q + p - 1) / p )); return
        fi
    fi
    if [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then        # cgroup v1
        q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [ "${q}" -gt 0 ] 2>/dev/null && [ "${p}" -gt 0 ] 2>/dev/null; then
            echo $(( (q + p - 1) / p )); return
        fi
    fi
    nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN
}
detect_mem_gb() {
    local lim="" host_kb host_gb
    [ -f /sys/fs/cgroup/memory.max ] && lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
    if [ -z "${lim}" ] || [ "${lim}" = "max" ]; then
        [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ] && lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
    fi
    host_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    host_gb=$(( host_kb / 1024 / 1024 ))
    if [[ "${lim}" =~ ^[0-9]+$ ]] && [ $(( lim / 1024 / 1024 / 1024 )) -lt "${host_gb}" ]; then
        echo $(( lim / 1024 / 1024 / 1024 ))
    else
        echo "${host_gb}"
    fi
}

GPUS=$("${REPO}/va_env/bin/python" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if ! [[ "${GPUS}" =~ ^[0-9]+$ ]] || [ "${GPUS}" = "0" ]; then  # fallback: hy-smi
    GPUS=$(hy-smi --showmeminfo vram 2>/dev/null | grep -c "vram Total Memory")
    GPUS=${GPUS:-0}
fi
GPU_NAME=$(hy-smi --showproductname 2>/dev/null | awk -F': *' '/Card Series/{gsub(/[ \t]/,"",$NF); print $NF; exit}')
GPU_MEM_GB=$(hy-smi --showmeminfo vram 2>/dev/null | awk '/vram Total Memory/{printf "%d", $NF/1024; exit}')
HW_CPU_CORES=$(detect_cpu_cores)
HW_MEM_GB=$(detect_mem_gb)
export HW_GPU_COUNT=${GPUS} HW_GPU_NAME=${GPU_NAME:-unknown} HW_GPU_MEM_GB=${GPU_MEM_GB:-0}
export HW_CPU_CORES HW_MEM_GB
log "hardware: GPU ${GPUS}x ${HW_GPU_NAME} (${HW_GPU_MEM_GB:-?}G vram each) | CPU ${HW_CPU_CORES}C | MEM ${HW_MEM_GB}G"

if [ "${NGPU}" = "auto" ]; then NGPU=${GPUS}; fi
[ "${NGPU}" -ge 1 ] || die "no GPU visible on this node"
[ "${GPUS}" -ge "${NGPU}" ] || die "need ${NGPU} GPUs, visible: ${GPUS}"
export NGPU
log "Phase 0: preflight (node ${NODE_RANK}/${NNODES}, NGPU=${NGPU}, total GPUs=$((NGPU * NNODES)), VAL_STEPS=${VAL_STEPS})"
if [ "${TIONE_TASK}" = "1" ]; then
    log "TI-ONE env: MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE:-unset} RANK=${RANK:-unset} -> NNODES=${NNODES} NODE_RANK=${NODE_RANK}"
fi
if [ "${NNODES}" -gt 1 ]; then
    log "multi-node: checkpoints must be on storage visible to node 0 (eval runs there)"
fi

# --- shared storage root ---
[ -d "${STORAGE_MOUNT_PATH}" ] || die "STORAGE_MOUNT_PATH not mounted: ${STORAGE_MOUNT_PATH}"
log "STORAGE_MOUNT_PATH=${STORAGE_MOUNT_PATH}"

mkdir -p "${SAVE_ROOT}" 2>/dev/null
[ -w "${SAVE_ROOT}" ] || die "SAVE_ROOT not writable: ${SAVE_ROOT} (override with SAVE_ROOT=/writable/path)"
log "SAVE_ROOT=${SAVE_ROOT} (writable)"

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
TRAIN_START_TS=$(date '+%Y-%m-%d %H:%M:%S')
TRAIN_SECS=""
GPU_MEM_PEAK_FILE="${SAVE_ROOT}/gpu_mem_peak_mib_node${NODE_RANK}.txt"
if [ -f "${CKPT}/transformer/config.json" ] && [ "${FORCE}" != "1" ]; then
    log "checkpoint ${CKPT} already exists, skip training (FORCE=1 to retrain)"
    TRAIN_END_TS=$(date '+%Y-%m-%d %H:%M:%S')
else
    log "Phase 1: robotwin post-training ${VAL_STEPS} steps on $((NGPU * NNODES)) GPUs (${NNODES} node(s) x ${NGPU})"
    # sample peak GPU memory (max across cards) every 60s while training
    (
        peak=0
        while true; do
            m=$(hy-smi --showmeminfo vram 2>/dev/null \
                | grep -oE "Used Memory \(MiB\): [0-9]+" | grep -oE "[0-9]+$" \
                | sort -n | tail -1)
            if [ -n "${m:-}" ] && [ "${m}" -gt "${peak}" ]; then
                peak=${m}; echo "${peak}" > "${GPU_MEM_PEAK_FILE}"
            fi
            sleep 60
        done
    ) &
    MEM_SAMPLER_PID=$!
    trap '[ -n "${MEM_SAMPLER_PID:-}" ] && kill "${MEM_SAMPLER_PID}" 2>/dev/null' EXIT
    T0=$(date +%s)
    NGPU=${NGPU} NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} \
        CONFIG_NAME=robotwin_train_val MASTER_PORT=${MASTER_PORT} \
        bash "${REPO}/script/run_va_posttrain.sh" --save-root "${SAVE_ROOT}" 2>&1 | tee "${TRAIN_LOG}"
    RC=${PIPESTATUS[0]}
    T1=$(date +%s)
    TRAIN_SECS=$((T1 - T0))
    TRAIN_END_TS=$(date '+%Y-%m-%d %H:%M:%S')
    kill "${MEM_SAMPLER_PID}" 2>/dev/null; wait "${MEM_SAMPLER_PID}" 2>/dev/null
    MEM_SAMPLER_PID=""
    [ "${RC}" -eq 0 ] || die "training exited rc=${RC} (see ${TRAIN_LOG})"
    [ -f "${CKPT}/transformer/config.json" ] || die "training finished but ${CKPT} missing"
    LAST_LOSS=$(grep -oE "latent_loss=[0-9.]+, action_loss=[0-9.]+, step=[0-9]+" "${TRAIN_LOG}" | tail -1 || true)
    log "training done in $(( (T1-T0)/60 )) min, checkpoint: ${CKPT}"
fi
log "last training loss: ${LAST_LOSS:-n/a}"

if [ "${NODE_RANK}" != "0" ]; then
    log "node ${NODE_RANK}: training finished, eval runs on node 0 only - exiting"
    exit 0
fi

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

# ================= Phase 4: final report =================
# Auto-generate final_report.md (format mirrors the hand-written one).
# Written to both ${SAVE_ROOT}/final_report.md and the repo root.
log "Phase 4: generating final_report.md"
REPORT_TRAIN_LOG="${TRAIN_LOG}" REPORT_EVAL_LOG=/tmp/validation_eval.log \
REPORT_SAVE_ROOT="${SAVE_ROOT}" REPORT_VAL_STEPS="${VAL_STEPS}" \
REPORT_NGPU="${NGPU}" REPORT_NNODES="${NNODES}" \
REPORT_MODEL_PATH="${MODEL_PATH}" REPORT_DATASET_PATH="${DATASET_PATH}" \
REPORT_TRAIN_START="${TRAIN_START_TS}" REPORT_TRAIN_END="${TRAIN_END_TS}" \
REPORT_TRAIN_SECS="${TRAIN_SECS}" \
REPORT_OUTPUTS="${SAVE_ROOT}/final_report.md,${REPO}/final_report.md" \
    "${REPO}/va_env/bin/python" "${REPO}/script/gen_final_report.py" \
    && log "final report: ${SAVE_ROOT}/final_report.md , ${REPO}/final_report.md" \
    || log "WARN: final report generation failed (non-fatal)"
