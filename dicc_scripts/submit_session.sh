#!/usr/bin/env bash
# =============================================================================
# Submit a DICC measurement session (V100 + A100 core benchmarks by default).
# Nsight is opt-in and depends on successful A100 completion.
#
# Usage:
#   export COLIDE_ROOT=/scr/$USER/colide   # or let this script resolve it
#   bash dicc_scripts/submit_session.sh
#   bash dicc_scripts/submit_session.sh --with-nsight
#   bash dicc_scripts/submit_session.sh --campaign day1 --date 20260711
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

WITH_NSIGHT=0
CAMPAIGN="${COLIDE_CAMPAIGN:-core}"
DATE_LABEL="${COLIDE_DATE_LABEL:-$(date -u +%Y%m%d)}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_session.sh [options]

  --with-nsight     Also submit 04_nsight_profile.sh after A100 succeeds
  --campaign NAME   Campaign label (default: core)
  --date YYYYMMDD   Date label for result paths (default: UTC today)
  --dry-run         Print sbatch commands without submitting
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-nsight) WITH_NSIGHT=1; shift ;;
    --campaign) CAMPAIGN="${2:?}"; shift 2 ;;
    --date) DATE_LABEL="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# Resolve repository root: explicit env > sibling of this script > /scr/$USER/colide
if [[ -z "${COLIDE_ROOT:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../inference/kernels" ]]; then
    COLIDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  elif [[ -d "/scr/${USER}/colide/inference/kernels" ]]; then
    COLIDE_ROOT="/scr/${USER}/colide"
  else
    die "Cannot resolve COLIDE_ROOT. Export COLIDE_ROOT to the cluster checkout."
  fi
fi
export COLIDE_ROOT
require_colide_root

GIT_SHA="$(git -C "${COLIDE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
if git -C "${COLIDE_ROOT}" status --porcelain 2>/dev/null | grep -q .; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "WARN: working tree dirty under ${COLIDE_ROOT} (allowed for --dry-run only)"
  else
    die "Working tree dirty under ${COLIDE_ROOT}; clean/commit before submitting so provenance is tight"
  fi
fi

# Pre-create absolute log directories so Slurm never has to invent relative paths
# against an unknown spool CWD (this was a real failure mode for 05_run_all.sh).
SESSION_DIR="${COLIDE_ROOT}/benchmarks/results/dicc/${CAMPAIGN}/_sessions/${DATE_LABEL}"
LOG_DIR="${SESSION_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

export COLIDE_CAMPAIGN="${CAMPAIGN}"
export COLIDE_DATE_LABEL="${DATE_LABEL}"
export COLIDE_CHECKPOINT="${COLIDE_CHECKPOINT:-model/best_model_botiot_twostage.pth}"
export COLIDE_N_TRIALS_CUDA="${COLIDE_N_TRIALS_CUDA:-100}"
export COLIDE_N_TRIALS_PYTORCH="${COLIDE_N_TRIALS_PYTORCH:-20}"
export COLIDE_PYTORCH_INNER="${COLIDE_PYTORCH_INNER:-1000}"

[[ -f "${COLIDE_ROOT}/${COLIDE_CHECKPOINT}" ]] \
  || die "Checkpoint missing: ${COLIDE_ROOT}/${COLIDE_CHECKPOINT}"
[[ -x "${COLIDE_ROOT}/inference/kernels/v100/fused_block3" ]] \
  || die "V100 kernels missing; run 01_setup.sh first"
[[ -x "${COLIDE_ROOT}/inference/kernels/a100/fused_block3" ]] \
  || die "A100 kernels missing; run 01_setup.sh first"

submit_one() {
  local script_name="$1"
  shift
  local out_base="$1"
  shift
  local -a extra=("$@")
  local script_path="${COLIDE_ROOT}/dicc_scripts/${script_name}"
  [[ -f "${script_path}" ]] || die "Missing ${script_path}"

  # Quote the export list so commas are not treated as shell array separators.
  local export_list="ALL,COLIDE_ROOT=${COLIDE_ROOT},COLIDE_CAMPAIGN=${CAMPAIGN},COLIDE_DATE_LABEL=${DATE_LABEL},COLIDE_CHECKPOINT=${COLIDE_CHECKPOINT},COLIDE_N_TRIALS_CUDA=${COLIDE_N_TRIALS_CUDA},COLIDE_N_TRIALS_PYTORCH=${COLIDE_N_TRIALS_PYTORCH},COLIDE_PYTORCH_INNER=${COLIDE_PYTORCH_INNER}"
  local -a cmd=(
    sbatch
    --parsable
    --chdir="${COLIDE_ROOT}"
    --export="${export_list}"
    --output="${LOG_DIR}/${out_base}_%j.out"
    --error="${LOG_DIR}/${out_base}_%j.err"
  )
  if [[ ${#extra[@]} -gt 0 ]]; then
    cmd+=("${extra[@]}")
  fi
  cmd+=("${script_path}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    # Command line goes to stderr so only the fake job id is captured from stdout.
    {
      printf 'DRY-RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    } >&2
    echo "dryrun_${out_base}"
    return 0
  fi

  local job_id
  job_id="$("${cmd[@]}")"
  # --parsable may return "jobid" or "jobid;cluster"
  job_id="${job_id%%;*}"
  echo "${job_id}"
}

log "=== Submitting COLIDE DICC session ==="
log "COLIDE_ROOT=${COLIDE_ROOT}"
log "git_sha=${GIT_SHA}"
log "campaign=${CAMPAIGN} date=${DATE_LABEL}"
log "slurm_logs=${LOG_DIR}"

V100_JOB="$(submit_one 02_benchmark_v100.sh v100)"
log "Submitted V100 job: ${V100_JOB}"

A100_JOB="$(submit_one 03_benchmark_a100.sh a100)"
log "Submitted A100 job: ${A100_JOB}"

NSIGHT_JOB=""
if [[ "${WITH_NSIGHT}" == "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    NSIGHT_JOB="$(submit_one 04_nsight_profile.sh nsight --dependency=afterok:DRYRUN_A100)"
  else
    NSIGHT_JOB="$(submit_one 04_nsight_profile.sh nsight --dependency="afterok:${A100_JOB}")"
  fi
  log "Submitted Nsight job (afterok:${A100_JOB}): ${NSIGHT_JOB}"
else
  log "Nsight not requested (pass --with-nsight to enable, depends on A100)"
fi

SESSION_JSON="${SESSION_DIR}/session.json"
cat > "${SESSION_JSON}" <<EOF
{
  "campaign": "${CAMPAIGN}",
  "date_label": "${DATE_LABEL}",
  "git_sha": "${GIT_SHA}",
  "colide_root": "${COLIDE_ROOT}",
  "checkpoint": "${COLIDE_CHECKPOINT}",
  "n_trials_cuda": ${COLIDE_N_TRIALS_CUDA},
  "n_trials_pytorch": ${COLIDE_N_TRIALS_PYTORCH},
  "pytorch_inner_forwards": ${COLIDE_PYTORCH_INNER},
  "jobs": {
    "v100": "${V100_JOB}",
    "a100": "${A100_JOB}",
    "nsight": "${NSIGHT_JOB}"
  },
  "slurm_log_dir": "${LOG_DIR}",
  "submitted_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "with_nsight": $([[ "${WITH_NSIGHT}" == "1" ]] && echo true || echo false)
}
EOF

log "Session record: ${SESSION_JSON}"
log "Check queue: squeue -u ${USER}"
log "After both days complete, compare with:"
log "  PYTHONPATH=. python scripts/compare_dicc_sessions.py --gpu v100s --date-a YYYYMMDD --date-b YYYYMMDD"
log "  PYTHONPATH=. python scripts/compare_dicc_sessions.py --gpu a100  --date-a YYYYMMDD --date-b YYYYMMDD"
