#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[krea2-matrix] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="${WORK_DIR:-/tmp/freefuse-krea2}"
FREEFUSE_REPO="${FREEFUSE_REPO:-${DEFAULT_REPO_DIR}}"
COMFYUI_DIR="${COMFYUI_DIR:-${WORK_DIR}/ComfyUI}"
VENV_DIR="${VENV_DIR:-${WORK_DIR}/.venv}"
LORA_SOURCE_DIR="${LORA_SOURCE_DIR:-${WORK_DIR}/input_loras}"
KIM_LORA_FILE="${KIM_LORA_FILE:-${LORA_SOURCE_DIR}/Krea 2 - Kim Possible.safetensors}"
VIOLET_LORA_FILE="${VIOLET_LORA_FILE:-${LORA_SOURCE_DIR}/Krea 2 - Violet Parr.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/results/krea2_matrix}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

SEEDS="${SEEDS:-354347915735006,354347915735007,354347915735008,354347915735009,354347915735010,354347915735011,354347915735012,354347915735013}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
STEPS="${STEPS:-8}"
PHASE1_STEPS="${PHASE1_STEPS:-${STEPS}}"
COLLECT_STEP="${COLLECT_STEP:-3}"
COLLECT_BLOCK="${COLLECT_BLOCK:-10}"
COLLECT_BLOCK_END="${COLLECT_BLOCK_END:-10}"
BALANCE_ITERATIONS="${BALANCE_ITERATIONS:-15}"
BIAS_SCALE="${BIAS_SCALE:-4.0}"
POSITIVE_BIAS_SCALE="${POSITIVE_BIAS_SCALE:-2.0}"
LORA_STRENGTH="${LORA_STRENGTH:-0.85}"
MAX_RSS_GB="${MAX_RSS_GB:-96}"
MAX_MEMORY_PERCENT="${MAX_MEMORY_PERCENT:-95}"
KREA2_UNET_NAME="${KREA2_UNET_NAME:-krea2_turbo_fp8_scaled.safetensors}"
KREA2_CLIP_NAME="${KREA2_CLIP_NAME:-qwen3vl_4b_fp8_scaled.safetensors}"
KREA2_DEVICE="${KREA2_DEVICE:-auto}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '[krea2-matrix] nvidia-smi is required for CUDA matrix testing\n' >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf '[krea2-matrix] missing test venv: %s\nRun setup_krea2_cuda_test.sh first.\n' "${VENV_DIR}" >&2
  exit 1
fi

if [[ ! -d "${COMFYUI_DIR}" ]]; then
  printf '[krea2-matrix] missing ComfyUI dir: %s\nRun setup_krea2_cuda_test.sh first.\n' "${COMFYUI_DIR}" >&2
  exit 1
fi

if [[ ! -f "${KIM_LORA_FILE}" || ! -f "${VIOLET_LORA_FILE}" ]]; then
  printf '[krea2-matrix] missing LoRA files; check KIM_LORA_FILE and VIOLET_LORA_FILE\n' >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  printf '[krea2-matrix] no CUDA GPUs visible\n' >&2
  exit 1
fi
MAX_PARALLEL="${MAX_PARALLEL:-${GPU_COUNT}}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
log "running ${#SEED_ARRAY[@]} seeds across ${GPU_COUNT} visible GPUs (max_parallel=${MAX_PARALLEL})"

running=0
failures=0

run_one_seed() {
  local seed="$1"
  local gpu="$2"
  local seed_dir="${OUTPUT_DIR}/seed_${seed}"
  local log_file="${LOG_DIR}/seed_${seed}.log"
  mkdir -p "${seed_dir}"
  log "launch seed=${seed} gpu=${gpu} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${VENV_DIR}/bin/python" \
    "${FREEFUSE_REPO}/freefuse_comfyui/scripts/run_krea2_compare.py" \
    --comfyui-dir "${COMFYUI_DIR}" \
    --plugin-dir "${FREEFUSE_REPO}/freefuse_comfyui" \
    --kim-lora-file "${KIM_LORA_FILE}" \
    --violet-lora-file "${VIOLET_LORA_FILE}" \
    --output-dir "${seed_dir}" \
    --mode both \
    --device "${KREA2_DEVICE}" \
    --unet-name "${KREA2_UNET_NAME}" \
    --clip-name "${KREA2_CLIP_NAME}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --steps "${STEPS}" \
    --phase1-steps "${PHASE1_STEPS}" \
    --collect-step "${COLLECT_STEP}" \
    --collect-block "${COLLECT_BLOCK}" \
    --collect-block-end "${COLLECT_BLOCK_END}" \
    --balance-iterations "${BALANCE_ITERATIONS}" \
    --bias-scale "${BIAS_SCALE}" \
    --positive-bias-scale "${POSITIVE_BIAS_SCALE}" \
    --lora-strength "${LORA_STRENGTH}" \
    --max-rss-gb "${MAX_RSS_GB}" \
    --max-memory-percent "${MAX_MEMORY_PERCENT}" \
    >"${log_file}" 2>&1
}

for index in "${!SEED_ARRAY[@]}"; do
  seed="${SEED_ARRAY[$index]}"
  gpu="$((index % GPU_COUNT))"
  run_one_seed "${seed}" "${gpu}" &
  running="$((running + 1))"
  if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
    if ! wait -n; then
      failures="$((failures + 1))"
    fi
    running="$((running - 1))"
  fi
done

while [[ "${running}" -gt 0 ]]; do
  if ! wait -n; then
    failures="$((failures + 1))"
  fi
  running="$((running - 1))"
done

if [[ "${failures}" -ne 0 ]]; then
  printf '[krea2-matrix] %s seed job(s) failed; inspect logs in %s\n' "${failures}" "${LOG_DIR}" >&2
  exit 1
fi

log "all seed jobs passed"
find "${OUTPUT_DIR}" -name 'krea2_baseline_vs_freefuse.png' -print | sort
