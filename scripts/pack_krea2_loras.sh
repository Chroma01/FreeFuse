#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[krea2-loras] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="${WORK_DIR:-/tmp/freefuse-krea2}"
FREEFUSE_REPO="${FREEFUSE_REPO:-${DEFAULT_REPO_DIR}}"
LORA_SOURCE_DIR="${LORA_SOURCE_DIR:-${WORK_DIR}/input_loras}"
LOCAL_DOWNLOADS="${LOCAL_DOWNLOADS:-${HOME}/Downloads}"
KIM_LORA_FILE="${KIM_LORA_FILE:-${LOCAL_DOWNLOADS}/Krea 2 - Kim Possible.safetensors}"
VIOLET_LORA_FILE="${VIOLET_LORA_FILE:-${LOCAL_DOWNLOADS}/Krea 2 - Violet Parr.safetensors}"
BUNDLE_PATH="${BUNDLE_PATH:-${WORK_DIR}/krea2_character_loras.tar}"

for file in "${KIM_LORA_FILE}" "${VIOLET_LORA_FILE}"; do
  if [[ ! -f "${file}" ]]; then
    printf '[krea2-loras] missing LoRA file: %s\n' "${file}" >&2
    exit 1
  fi
done

mkdir -p "${WORK_DIR}" "${LORA_SOURCE_DIR}"

log "staging LoRAs under ${LORA_SOURCE_DIR}"
cp -f "${KIM_LORA_FILE}" "${LORA_SOURCE_DIR}/Krea 2 - Kim Possible.safetensors"
cp -f "${VIOLET_LORA_FILE}" "${LORA_SOURCE_DIR}/Krea 2 - Violet Parr.safetensors"

log "writing checksums"
(
  cd "${LORA_SOURCE_DIR}"
  shasum -a 256 "Krea 2 - Kim Possible.safetensors" "Krea 2 - Violet Parr.safetensors" > SHA256SUMS
)

log "creating bundle ${BUNDLE_PATH}"
tar -C "${LORA_SOURCE_DIR}" -cvf "${BUNDLE_PATH}" \
  "Krea 2 - Kim Possible.safetensors" \
  "Krea 2 - Violet Parr.safetensors" \
  SHA256SUMS

ls -lh "${BUNDLE_PATH}"
log "bundle checksum:"
shasum -a 256 "${BUNDLE_PATH}"

if [[ -n "${REMOTE:-}" ]]; then
  if ! command -v rsync >/dev/null 2>&1; then
    printf '[krea2-loras] REMOTE was set but rsync is not available\n' >&2
    exit 1
  fi
  log "rsyncing staged LoRAs to ${REMOTE}"
  rsync -avh --progress "${LORA_SOURCE_DIR}/" "${REMOTE%/}/"
else
  cat <<EOF
[krea2-loras] To copy manually:

scp "${BUNDLE_PATH}" <user>@<host>:/tmp/freefuse-krea2/
ssh <user>@<host> 'mkdir -p /tmp/freefuse-krea2/input_loras && tar -C /tmp/freefuse-krea2/input_loras -xvf /tmp/freefuse-krea2/krea2_character_loras.tar && cd /tmp/freefuse-krea2/input_loras && sha256sum -c SHA256SUMS'

Then in the remote repo:
WORK_DIR=/tmp/freefuse-krea2 bash freefuse_comfyui/scripts/setup_krea2_cuda_test.sh
WORK_DIR=/tmp/freefuse-krea2 bash freefuse_comfyui/scripts/run_krea2_cuda_matrix.sh
EOF
fi
