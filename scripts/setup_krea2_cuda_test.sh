#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[krea2-setup] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[krea2-setup] missing required command: %s\n' "$1" >&2
    exit 1
  fi
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
COMFYUI_REF="${COMFYUI_REF:-master}"
KREA2_REPO="${KREA2_REPO:-Comfy-Org/Krea-2}"

export HF_HOME="${HF_HOME:-${WORK_DIR}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${WORK_DIR}/hf_hub_cache}"
export HF_XET_CACHE="${HF_XET_CACHE:-${WORK_DIR}/hf_xet_cache}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

require_cmd git
require_cmd python3
require_cmd nvidia-smi

mkdir -p "${WORK_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${LORA_SOURCE_DIR}"

if [[ -d "${COMFYUI_DIR}/.git" ]]; then
  if ! git -C "${COMFYUI_DIR}" diff --quiet || ! git -C "${COMFYUI_DIR}" diff --cached --quiet; then
    printf '[krea2-setup] ComfyUI checkout has local tracked changes: %s\n' "${COMFYUI_DIR}" >&2
    exit 1
  fi
  log "updating ComfyUI at ${COMFYUI_DIR} (${COMFYUI_REF})"
  git -C "${COMFYUI_DIR}" fetch --depth 1 origin "${COMFYUI_REF}"
  git -C "${COMFYUI_DIR}" checkout --detach FETCH_HEAD
elif [[ -e "${COMFYUI_DIR}" ]]; then
  printf '[krea2-setup] COMFYUI_DIR exists but is not a git checkout: %s\n' "${COMFYUI_DIR}" >&2
  exit 1
else
  log "cloning ComfyUI into ${COMFYUI_DIR}"
  git clone --depth 1 --branch "${COMFYUI_REF}" https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
HF="${VENV_DIR}/bin/hf"

log "installing Python dependencies"
"${PYTHON}" -m pip install -U pip wheel
"${PYTHON}" -m pip install -r "${COMFYUI_DIR}/requirements.txt"
"${PYTHON}" -m pip install -U "huggingface_hub[hf_xet]" hf_xet psutil pillow
"${PYTHON}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the test venv")
print(f"[krea2-setup] torch={torch.__version__}, cuda_devices={torch.cuda.device_count()}")
PY

download_model_file() {
  local repo_path="$1"
  local target="${COMFYUI_DIR}/models/${repo_path}"
  if [[ -f "${target}" ]]; then
    log "model file exists: ${target}"
    return
  fi
  log "downloading ${KREA2_REPO}/${repo_path}"
  "${HF}" download "${KREA2_REPO}" "${repo_path}" --local-dir "${COMFYUI_DIR}/models"
}

download_model_file "diffusion_models/krea2_turbo_fp8_scaled.safetensors"
download_model_file "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
download_model_file "vae/qwen_image_vae.safetensors"

if [[ ! -f "${KIM_LORA_FILE}" || ! -f "${VIOLET_LORA_FILE}" ]]; then
  cat >&2 <<EOF
[krea2-setup] Missing required character LoRA files.

Expected:
  KIM_LORA_FILE=${KIM_LORA_FILE}
  VIOLET_LORA_FILE=${VIOLET_LORA_FILE}

Copy the two local LoRAs to ${LORA_SOURCE_DIR}, or set KIM_LORA_FILE and VIOLET_LORA_FILE explicitly.
EOF
  exit 2
fi

CUSTOM_NODE_LINK="${COMFYUI_DIR}/custom_nodes/freefuse_comfyui"
if [[ -e "${CUSTOM_NODE_LINK}" || -L "${CUSTOM_NODE_LINK}" ]]; then
  if [[ "$(readlink -f "${CUSTOM_NODE_LINK}")" != "$(readlink -f "${FREEFUSE_REPO}/freefuse_comfyui")" ]]; then
    printf '[krea2-setup] custom node path points somewhere else: %s -> %s\n' \
      "${CUSTOM_NODE_LINK}" "$(readlink -f "${CUSTOM_NODE_LINK}")" >&2
    exit 1
  fi
else
  mkdir -p "${COMFYUI_DIR}/custom_nodes"
  ln -s "${FREEFUSE_REPO}/freefuse_comfyui" "${CUSTOM_NODE_LINK}"
fi

mkdir -p "${COMFYUI_DIR}/models/loras/FreeFuse/Krea2"
ln -sfn "${KIM_LORA_FILE}" "${COMFYUI_DIR}/models/loras/FreeFuse/Krea2/Krea 2 - Kim Possible.safetensors"
ln -sfn "${VIOLET_LORA_FILE}" "${COMFYUI_DIR}/models/loras/FreeFuse/Krea2/Krea 2 - Violet Parr.safetensors"

log "LoRA files found"
log "ComfyUI: ${COMFYUI_DIR}"
log "venv: ${VENV_DIR}"
log "HF cache: ${HF_HUB_CACHE}"
log "ready; run:"
cat <<EOF
${VENV_DIR}/bin/python ${FREEFUSE_REPO}/freefuse_comfyui/scripts/run_krea2_compare.py \\
  --comfyui-dir ${COMFYUI_DIR} \\
  --plugin-dir ${FREEFUSE_REPO}/freefuse_comfyui \\
  --kim-lora-file "${KIM_LORA_FILE}" \\
  --violet-lora-file "${VIOLET_LORA_FILE}" \\
  --output-dir ${WORK_DIR}/results/krea2_compare \\
  --mode both --width 1024 --height 1024 --steps 8 --phase1-steps 8 --collect-step 3
EOF
