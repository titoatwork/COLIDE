#!/usr/bin/env bash
#SBATCH --job-name=colide_nsight
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
# Opt-in Nsight profiling on A100. Prefer submit_session.sh --with-nsight
# which sets COLIDE_ROOT, chdir, log paths, and afterok dependency on A100.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_colide_root
cd "${COLIDE_ROOT}"

export COLIDE_CAMPAIGN="${COLIDE_CAMPAIGN:-nsight}"
export COLIDE_DATE_LABEL="${COLIDE_DATE_LABEL:-$(date -u +%Y%m%d)}"
export COLIDE_GPU_LABEL="${COLIDE_GPU_LABEL:-a100}"
export COLIDE_KERNELS_DIR="${COLIDE_ROOT}/inference/kernels/a100"

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

assert_gpu a100 'A100' 8.0 30000
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

profile_one block1_a100 fused_block1
profile_one block2_a100 fused_block2
profile_one block3_fp32_a100 fused_block3
profile_one block3_fp16_a100 fused_block3_fp16
profile_one block4_a100 fused_block4

log "=== Nsight Profiling Complete ==="
log "Reports under ${NSIGHT_DIR}"
EXIT_CODE=0
