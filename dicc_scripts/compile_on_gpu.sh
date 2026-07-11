#!/usr/bin/env bash
# Submit (or run under srun) kernel compilation on a GPU partition where nvcc
# usually lives when the login node has no CUDA module (Rostam pattern).
#
# Usage:
#   bash dicc_scripts/compile_on_gpu.sh v100
#   bash dicc_scripts/compile_on_gpu.sh a100
#   bash dicc_scripts/compile_on_gpu.sh both
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

TARGET="${1:-both}"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export COLIDE_ROOT="${COLIDE_ROOT:-${ROOT}}"

submit_compile() {
  local profile="$1"
  local partition="$2"
  local nvcc_arch="$3"
  local subdir="$4"
  local out="${COLIDE_ROOT}/benchmarks/results/dicc/compile_${profile}_%j.out"
  local err="${COLIDE_ROOT}/benchmarks/results/dicc/compile_${profile}_%j.err"
  mkdir -p "${COLIDE_ROOT}/benchmarks/results/dicc"

  log "Submitting compile job: profile=${profile} partition=${partition} arch=${nvcc_arch}"
  sbatch --parsable \
    --job-name="colide_compile_${profile}" \
    --partition="${partition}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:00:00 \
    --chdir="${COLIDE_ROOT}" \
    --output="${out}" \
    --error="${err}" \
    --wrap="set -Eeuo pipefail; cd '${COLIDE_ROOT}'; export COLIDE_ROOT='${COLIDE_ROOT}'; bash dicc_scripts/01_setup.sh --kernels-only --allow-dirty --targets ${nvcc_arch}:${subdir}"
}

case "${TARGET}" in
  v100)
    submit_compile v100 "${COLIDE_V100_PARTITION:-cuda-V100}" sm_70 v100
    ;;
  a100)
    submit_compile a100 "${COLIDE_A100_PARTITION:-cuda-A100}" sm_80 a100
    ;;
  both)
    submit_compile v100 "${COLIDE_V100_PARTITION:-cuda-V100}" sm_70 v100
    submit_compile a100 "${COLIDE_A100_PARTITION:-cuda-A100}" sm_80 a100
    ;;
  *)
    die "Usage: $0 v100|a100|both"
    ;;
esac

log "Watch with: squeue -u ${USER}"
log "When done: ls -la ${COLIDE_ROOT}/inference/kernels/v100 ${COLIDE_ROOT}/inference/kernels/a100 2>/dev/null || true"
