#!/usr/bin/env bash
#SBATCH --job-name=colide_a100
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Thin SBATCH resource wrapper. All logic lives in lib/run_benchmark.sh.
# Output/error paths and --chdir are injected by submit_session.sh so
# relative log paths cannot resolve against a wrong CWD.
set -Eeuo pipefail

if [[ -z "${COLIDE_ROOT:-}" ]]; then
  echo "ERROR: COLIDE_ROOT must be exported (use submit_session.sh)" >&2
  exit 1
fi

export COLIDE_GPU_LABEL="${COLIDE_GPU_LABEL:-a100}"
export COLIDE_GPU_NAME_RE="${COLIDE_GPU_NAME_RE:-A100}"
export COLIDE_GPU_CC="${COLIDE_GPU_CC:-8.0}"
export COLIDE_GPU_MIN_MEM="${COLIDE_GPU_MIN_MEM:-30000}"
export COLIDE_KERNELS_SUBDIR="${COLIDE_KERNELS_SUBDIR:-a100}"

exec bash "${COLIDE_ROOT}/dicc_scripts/lib/run_benchmark.sh"
