#!/usr/bin/env bash
# Opt-in Nsight profiling. Prefer submit_session.sh --with-nsight.
# No site-specific nodelist — inject partition/gres via submit_session.
#SBATCH --job-name=colide_nsight
#SBATCH --time=00:45:00
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

if [[ -z "${COLIDE_ROOT:-}" ]]; then
  COLIDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  export COLIDE_ROOT
fi
require_colide_root
cd "${COLIDE_ROOT}"

# Default to a100 profile if not already set by submit_session
if [[ -z "${COLIDE_KERNELS_SUBDIR:-}" ]]; then
  # shellcheck source=profiles/a100.env
  source "${COLIDE_ROOT}/dicc_scripts/profiles/a100.env"
fi

export COLIDE_CAMPAIGN="${COLIDE_CAMPAIGN:-nsight}"
export COLIDE_DATE_LABEL="${COLIDE_DATE_LABEL:-$(date -u +%Y%m%d)}"
export COLIDE_GPU_LABEL="${COLIDE_GPU_LABEL:-a100}"
export COLIDE_GPU_NAME_RE="${COLIDE_GPU_NAME_RE:-A100}"
export COLIDE_GPU_CC="${COLIDE_GPU_CC:-8.0}"
export COLIDE_GPU_MIN_MEM="${COLIDE_GPU_MIN_MEM:-30000}"
export COLIDE_KERNELS_DIR="${COLIDE_ROOT}/inference/kernels/${COLIDE_KERNELS_SUBDIR:-a100}"

JOB_ID="${SLURM_JOB_ID:-local}"
RUN_DIR="$(make_run_dir "${COLIDE_CAMPAIGN}" "${COLIDE_GPU_LABEL}" "${COLIDE_DATE_LABEL}" "${JOB_ID}")"
export COLIDE_RUN_DIR="${RUN_DIR}"

exec > >(tee -a "${RUN_DIR}/logs/nsight.log") 2>&1

EXIT_CODE=0
on_exit() {
  local ec=$?
  if [[ ${ec} -ne 0 ]]; then
    EXIT_CODE=${ec}
  fi
  mark_success "${RUN_DIR}" "${EXIT_CODE}"
}
trap on_exit EXIT

log "=== COLIDE Nsight Compute Profiling ==="
log "Node=$(hostname) RUN_DIR=${RUN_DIR}"

assert_gpu "${COLIDE_GPU_LABEL}" "${COLIDE_GPU_NAME_RE}" "${COLIDE_GPU_CC}" "${COLIDE_GPU_MIN_MEM}"
command -v ncu >/dev/null 2>&1 || die "ncu (Nsight Compute) not found on PATH"

write_manifest "${RUN_DIR}"
write_environment "${RUN_DIR}"
copy_kernel_checksums "${RUN_DIR}" "${COLIDE_KERNELS_DIR}"

mkdir -p "${RUN_DIR}/ncu"
NSIGHT_DIR="${RUN_DIR}/ncu"

profile_one() {
  local label="$1"
  local bin="$2"
  local path="${COLIDE_KERNELS_DIR}/${bin}"
  [[ -x "${path}" ]] || die "missing binary ${path}"
  log "Profiling ${label}..."
  (
    cd "${COLIDE_KERNELS_DIR}"
    ncu --set full -o "${NSIGHT_DIR}/${label}" "./${bin}"
  )
}

profile_one "block1_${COLIDE_GPU_LABEL}" fused_block1
profile_one "block2_${COLIDE_GPU_LABEL}" fused_block2
profile_one "block3_fp32_${COLIDE_GPU_LABEL}" fused_block3
profile_one "block3_fp16_${COLIDE_GPU_LABEL}" fused_block3_fp16
profile_one "block4_${COLIDE_GPU_LABEL}" fused_block4

log "=== Nsight Profiling Complete ==="
log "Reports under ${NSIGHT_DIR}"
EXIT_CODE=0
