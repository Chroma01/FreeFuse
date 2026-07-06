#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[krea2-import] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="${WORK_DIR:-/tmp/freefuse-krea2}"
FREEFUSE_REPO="${FREEFUSE_REPO:-${DEFAULT_REPO_DIR}}"
RESULT_BUNDLE="${RESULT_BUNDLE:-${1:-${WORK_DIR}/krea2_cuda_results.tar.gz}}"
IMPORT_DIR="${IMPORT_DIR:-${WORK_DIR}/imported_results}"
CONTACT_SHEET="${CONTACT_SHEET:-}"
UPDATE_README="${UPDATE_README:-1}"

if [[ ! -f "${RESULT_BUNDLE}" ]]; then
  printf '[krea2-import] missing result bundle: %s\n' "${RESULT_BUNDLE}" >&2
  exit 1
fi

if [[ ! -d "${FREEFUSE_REPO}/.git" ]]; then
  printf '[krea2-import] FREEFUSE_REPO is not a git checkout: %s\n' "${FREEFUSE_REPO}" >&2
  exit 1
fi

mkdir -p "${IMPORT_DIR}"
log "extracting ${RESULT_BUNDLE} -> ${IMPORT_DIR}"
tar -C "${IMPORT_DIR}" -xzvf "${RESULT_BUNDLE}" >/dev/null

sheets=()
while IFS= read -r sheet; do
  sheets+=("${sheet}")
done < <(find "${IMPORT_DIR}" -name 'krea2_baseline_vs_freefuse.png' -type f | sort)

if [[ "${#sheets[@]}" -eq 0 ]]; then
  printf '[krea2-import] no krea2_baseline_vs_freefuse.png found under %s\n' "${IMPORT_DIR}" >&2
  exit 1
fi

log "available comparison sheets:"
printf '%s\n' "${sheets[@]}"

if [[ -z "${CONTACT_SHEET}" ]]; then
  if [[ "${#sheets[@]}" -eq 1 ]]; then
    CONTACT_SHEET="${sheets[0]}"
  else
    cat >&2 <<EOF
[krea2-import] Multiple comparison sheets were found. Pick the best one explicitly:

CONTACT_SHEET="/path/to/krea2_baseline_vs_freefuse.png" \\
  RESULT_BUNDLE="${RESULT_BUNDLE}" \\
  bash freefuse_comfyui/scripts/import_krea2_cuda_results.sh
EOF
    exit 2
  fi
fi

if [[ ! -f "${CONTACT_SHEET}" ]]; then
  printf '[krea2-import] CONTACT_SHEET does not exist: %s\n' "${CONTACT_SHEET}" >&2
  exit 1
fi

if [[ "${UPDATE_README}" == "1" ]]; then
  log "updating README with ${CONTACT_SHEET}"
  python3 "${FREEFUSE_REPO}/freefuse_comfyui/scripts/update_krea2_readme_result.py" \
    --repo-root "${FREEFUSE_REPO}" \
    --contact-sheet "${CONTACT_SHEET}"
else
  log "dry-run README update for ${CONTACT_SHEET}"
  python3 "${FREEFUSE_REPO}/freefuse_comfyui/scripts/update_krea2_readme_result.py" \
    --repo-root "${FREEFUSE_REPO}" \
    --contact-sheet "${CONTACT_SHEET}" \
    --dry-run
fi
