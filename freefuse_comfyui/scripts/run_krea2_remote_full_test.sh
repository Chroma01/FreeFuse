#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[krea2-full] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="${WORK_DIR:-/tmp/freefuse-krea2}"
FREEFUSE_REPO="${FREEFUSE_REPO:-${DEFAULT_REPO_DIR}}"
LORA_BUNDLE="${LORA_BUNDLE:-${WORK_DIR}/krea2_character_loras.tar}"
LORA_SOURCE_DIR="${LORA_SOURCE_DIR:-${WORK_DIR}/input_loras}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/results/krea2_matrix}"
RESULT_BUNDLE="${RESULT_BUNDLE:-${WORK_DIR}/krea2_cuda_results.tar.gz}"
PULL_REPO="${PULL_REPO:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '[krea2-full] nvidia-smi is required; run this on a CUDA host\n' >&2
  exit 1
fi

if [[ ! -d "${FREEFUSE_REPO}/.git" ]]; then
  printf '[krea2-full] FREEFUSE_REPO is not a git checkout: %s\n' "${FREEFUSE_REPO}" >&2
  exit 1
fi

if [[ "${PULL_REPO}" == "1" ]]; then
  if ! git -C "${FREEFUSE_REPO}" diff --quiet || ! git -C "${FREEFUSE_REPO}" diff --cached --quiet; then
    printf '[krea2-full] refusing to pull because the repo has local tracked changes: %s\n' "${FREEFUSE_REPO}" >&2
    exit 1
  fi
  log "pulling latest repo changes"
  git -C "${FREEFUSE_REPO}" pull --ff-only
fi

log "repo commit: $(git -C "${FREEFUSE_REPO}" rev-parse --short HEAD)"

if [[ ! -f "${LORA_BUNDLE}" ]]; then
  cat >&2 <<EOF
[krea2-full] missing LoRA bundle: ${LORA_BUNDLE}

Create it on the local Mac with:
  WORK_DIR=/tmp/freefuse-krea2 bash freefuse_comfyui/scripts/pack_krea2_loras.sh

Then copy it to this CUDA host:
  scp /tmp/freefuse-krea2/krea2_character_loras.tar <host>:${WORK_DIR}/
EOF
  exit 1
fi

mkdir -p "${LORA_SOURCE_DIR}"
log "extracting LoRA bundle"
tar -C "${LORA_SOURCE_DIR}" -xvf "${LORA_BUNDLE}"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${LORA_SOURCE_DIR}" && sha256sum -c SHA256SUMS)
elif command -v shasum >/dev/null 2>&1; then
  (cd "${LORA_SOURCE_DIR}" && shasum -a 256 -c SHA256SUMS)
else
  printf '[krea2-full] missing sha256sum or shasum for LoRA verification\n' >&2
  exit 1
fi

log "setting up ComfyUI and Krea2 weights"
WORK_DIR="${WORK_DIR}" \
FREEFUSE_REPO="${FREEFUSE_REPO}" \
LORA_SOURCE_DIR="${LORA_SOURCE_DIR}" \
bash "${FREEFUSE_REPO}/freefuse_comfyui/scripts/setup_krea2_cuda_test.sh"

log "running CUDA seed matrix"
WORK_DIR="${WORK_DIR}" \
FREEFUSE_REPO="${FREEFUSE_REPO}" \
LORA_SOURCE_DIR="${LORA_SOURCE_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${FREEFUSE_REPO}/freefuse_comfyui/scripts/run_krea2_cuda_matrix.sh"

log "packing result images and logs"
tar -C "${WORK_DIR}" -czvf "${RESULT_BUNDLE}" \
  "results/$(basename "${OUTPUT_DIR}")"

log "result bundle: ${RESULT_BUNDLE}"
find "${OUTPUT_DIR}" -name 'krea2_baseline_vs_freefuse.png' -print | sort
