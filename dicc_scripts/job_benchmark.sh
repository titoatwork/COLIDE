#!/usr/bin/env bash
# Generic SLURM entrypoint for one GPU profile.
# Site-specific resources (partition, account, nodelist, gres, …) are injected
# by submit_session.sh on the sbatch command line — do NOT hardcode cluster
# paths or hostnames here.
#SBATCH --job-name=colide_bench
#SBATCH --time=02:00:00
set -Eeuo pipefail

if [[ -z "${COLIDE_ROOT:-}" ]]; then
  echo "ERROR: COLIDE_ROOT must be set (use submit_session.sh)" >&2
  exit 1
fi
if [[ -z "${COLIDE_GPU_LABEL:-}" || -z "${COLIDE_KERNELS_SUBDIR:-}" ]]; then
  echo "ERROR: profile env incomplete (COLIDE_GPU_LABEL / COLIDE_KERNELS_SUBDIR)" >&2
  exit 1
fi

# Defaults only if profile forgot them (should not happen with profiles/*.env)
export COLIDE_GPU_NAME_RE="${COLIDE_GPU_NAME_RE:-.}"
export COLIDE_GPU_CC="${COLIDE_GPU_CC:-0.0}"
export COLIDE_GPU_MIN_MEM="${COLIDE_GPU_MIN_MEM:-0}"

exec bash "${COLIDE_ROOT}/dicc_scripts/lib/run_benchmark.sh"
