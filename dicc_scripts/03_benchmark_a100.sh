#!/usr/bin/env bash
# Back-compat thin wrapper. Prefer: submit_session.sh --targets a100
# No site-specific nodelist/partition — resources come from sbatch CLI.
#SBATCH --job-name=colide_a100
#SBATCH --time=02:00:00
set -Eeuo pipefail

if [[ -z "${COLIDE_ROOT:-}" ]]; then
  echo "ERROR: COLIDE_ROOT must be exported (use submit_session.sh)" >&2
  exit 1
fi

# shellcheck source=profiles/a100.env
source "${COLIDE_ROOT}/dicc_scripts/profiles/a100.env"
export COLIDE_GPU_LABEL COLIDE_GPU_NAME_RE COLIDE_GPU_CC COLIDE_GPU_MIN_MEM COLIDE_KERNELS_SUBDIR

exec bash "${COLIDE_ROOT}/dicc_scripts/lib/run_benchmark.sh"
