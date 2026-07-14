#!/usr/bin/env bash
# Submit kernel compilation on a GPU partition (login nodes often lack nvcc).
#
# Rostam note: sinfo often shows GRES=(null). Requesting --gres=gpu:1 can then
# fail with "Requested node configuration is not available". Override with:
#   COLIDE_SBATCH_GRES=           # omit gres
#   COLIDE_SBATCH_GRES=gpu:1      # classic
#   COLIDE_SBATCH_GPUS=1          # use --gpus=1 instead of --gres
#
# Usage:
#   bash dicc_scripts/compile_on_gpu.sh v100
#   bash dicc_scripts/compile_on_gpu.sh a100
#   bash dicc_scripts/compile_on_gpu.sh both
#   COLIDE_SBATCH_GRES= bash dicc_scripts/compile_on_gpu.sh both
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

TARGET="${1:-both}"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export COLIDE_ROOT="${COLIDE_ROOT:-${ROOT}}"

# Defaults tuned for Rostam-like sites: no forced gres unless user sets it.
# Empty COLIDE_SBATCH_GRES => do not pass --gres.
USE_GRES="${COLIDE_SBATCH_GRES-}"
USE_GPUS="${COLIDE_SBATCH_GPUS-}"
MEM="${COLIDE_SBATCH_MEM:-8G}"
CPUS="${COLIDE_SBATCH_CPUS:-2}"
TIME="${COLIDE_SBATCH_TIME:-01:00:00}"

submit_compile() {
  local profile="$1"
  local partition="$2"
  local nvcc_arch="$3"
  local subdir="$4"
  mkdir -p "${COLIDE_ROOT}/benchmarks/results/dicc"

  local -a cmd=(
    sbatch
    --parsable
    --job-name="colide_compile_${profile}"
    --partition="${partition}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${TIME}"
    --chdir="${COLIDE_ROOT}"
    --output="${COLIDE_ROOT}/benchmarks/results/dicc/compile_${profile}_%j.out"
    --error="${COLIDE_ROOT}/benchmarks/results/dicc/compile_${profile}_%j.err"
  )

  if [[ -n "${USE_GPUS}" ]]; then
    cmd+=(--gpus="${USE_GPUS}")
  elif [[ -n "${USE_GRES}" ]]; then
    cmd+=(--gres="${USE_GRES}")
  fi

  cmd+=(
    --wrap="set -Eeuo pipefail; cd '${COLIDE_ROOT}'; export COLIDE_ROOT='${COLIDE_ROOT}'; hostname; command -v nvcc || true; find /usr/local /opt -name nvcc 2>/dev/null | head -5; bash dicc_scripts/01_setup.sh --kernels-only --allow-dirty --targets ${nvcc_arch}:${subdir}"
  )

  log "Submitting: profile=${profile} partition=${partition} arch=${nvcc_arch} gres='${USE_GRES}' gpus='${USE_GPUS}'"
  {
    printf 'CMD:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } >&2

  local jid
  jid="$("${cmd[@]}")"
  jid="${jid%%;*}"
  log "Submitted compile job ${jid} (${profile})"
  echo "${jid}"
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

log "Watch: squeue -u ${USER}"
log "When done: ls inference/kernels/v100 inference/kernels/a100"
