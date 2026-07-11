#!/usr/bin/env bash
# =============================================================================
# Submit a multi-GPU measurement session (portable SLURM).
#
# No hardcoded cluster paths (/scr/...) or hostnames (gpu05/gpu06).
# COLIDE_ROOT defaults to the repo that contains this script.
#
# Usage:
#   cd /path/to/COLIDE
#   bash dicc_scripts/01_setup.sh
#   bash dicc_scripts/submit_session.sh --targets v100,a100
#   bash dicc_scripts/submit_session.sh --targets v100 --partition gpu --account MYACCT
#   bash dicc_scripts/submit_session.sh --with-nsight --targets a100
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Optional site defaults (copy site.env.example -> site.env)
if [[ -f "${SCRIPT_DIR}/site.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/site.env"
fi

WITH_NSIGHT=0
CAMPAIGN="${COLIDE_CAMPAIGN:-core}"
DATE_LABEL="${COLIDE_DATE_LABEL:-$(date -u +%Y%m%d)}"
DRY_RUN=0
TARGETS="${COLIDE_TARGETS:-v100,a100}"

# SLURM overrides (env or flags)
SBATCH_PARTITION="${COLIDE_SBATCH_PARTITION:-}"
SBATCH_ACCOUNT="${COLIDE_SBATCH_ACCOUNT:-}"
SBATCH_QOS="${COLIDE_SBATCH_QOS:-}"
# Use ${VAR-default} (not :-) so an explicit empty COLIDE_SBATCH_GRES= omits --gres
# (required on sites like Rostam where sinfo reports GRES=(null)).
SBATCH_GRES="${COLIDE_SBATCH_GRES-gpu:1}"
SBATCH_CONSTRAINT="${COLIDE_SBATCH_CONSTRAINT:-}"
SBATCH_NODELIST="${COLIDE_SBATCH_NODELIST:-}"
SBATCH_TIME="${COLIDE_SBATCH_TIME:-02:00:00}"
SBATCH_CPUS="${COLIDE_SBATCH_CPUS:-4}"
SBATCH_MEM="${COLIDE_SBATCH_MEM:-32G}"
SBATCH_EXTRA="${COLIDE_SBATCH_EXTRA:-}"
SBATCH_GPUS="${COLIDE_SBATCH_GPUS-}"

usage() {
  cat <<'EOF'
Usage: submit_session.sh [options]

  --targets LIST       comma-separated profiles under profiles/ (default: v100,a100)
  --campaign NAME      campaign label (default: core)
  --date YYYYMMDD      date label for result paths (default: UTC today)
  --with-nsight        after last target that is a100 (or final target), submit Nsight
  --partition NAME     sbatch -p
  --account NAME       sbatch -A
  --qos NAME           sbatch -q
  --gres SPEC          sbatch --gres (default: gpu:1; use --gres none to omit)
  --gpus N             sbatch --gpus N (alternative to --gres on some sites)
  --constraint EXPR    sbatch -C
  --nodelist NODES     sbatch -w (optional; avoid on portable sites)
  --time TIME          sbatch -t (default: 02:00:00)
  --cpus N             sbatch -c (default: 4)
  --mem SIZE           sbatch --mem (default: 32G)
  --extra "FLAGS..."   extra sbatch flags (quoted)
  --allow-dirty        allow submit with a dirty git tree (site tweaks)
  --dry-run            print sbatch commands without submitting
  -h, --help           show help

COLIDE_ROOT defaults to the parent of dicc_scripts/ (this checkout). Export it
only if you intentionally run against another tree.
EOF
}

ALLOW_DIRTY="${COLIDE_ALLOW_DIRTY:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-nsight) WITH_NSIGHT=1; shift ;;
    --campaign) CAMPAIGN="${2:?}"; shift 2 ;;
    --date) DATE_LABEL="${2:?}"; shift 2 ;;
    --targets) TARGETS="${2:?}"; shift 2 ;;
    --partition) SBATCH_PARTITION="${2:?}"; shift 2 ;;
    --account) SBATCH_ACCOUNT="${2:?}"; shift 2 ;;
    --qos) SBATCH_QOS="${2:?}"; shift 2 ;;
    --gres)
      if [[ "${2:?}" == "none" || "${2}" == "omit" || "${2}" == "-" ]]; then
        SBATCH_GRES=""
      else
        SBATCH_GRES="$2"
      fi
      shift 2
      ;;
    --gpus) SBATCH_GPUS="${2:?}"; shift 2 ;;
    --constraint) SBATCH_CONSTRAINT="${2:?}"; shift 2 ;;
    --nodelist) SBATCH_NODELIST="${2:?}"; shift 2 ;;
    --time) SBATCH_TIME="${2:?}"; shift 2 ;;
    --cpus) SBATCH_CPUS="${2:?}"; shift 2 ;;
    --mem) SBATCH_MEM="${2:?}"; shift 2 ;;
    --extra) SBATCH_EXTRA="${2:?}"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# Resolve repository root: explicit env > parent of this script (never /scr hardcode)
if [[ -z "${COLIDE_ROOT:-}" ]]; then
  COLIDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
export COLIDE_ROOT
require_colide_root

GIT_SHA="$(git -C "${COLIDE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
if git -C "${COLIDE_ROOT}" status --porcelain 2>/dev/null | grep -q .; then
  if [[ "${DRY_RUN}" == "1" || "${ALLOW_DIRTY}" == "1" ]]; then
    log "WARN: working tree dirty under ${COLIDE_ROOT} (allowed via --dry-run/--allow-dirty)"
    log "WARN: git status:"; git -C "${COLIDE_ROOT}" status --porcelain | head -20 >&2 || true
  else
    die "Working tree dirty under ${COLIDE_ROOT}. git restore site edits, commit them, or pass --allow-dirty"
  fi
fi

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

# Print resource flags one-per-line (portable; no namerefs / assoc arrays).
# Optional args override globals for this call:
#   sbatch_resource_flags [partition] [gres] [constraint]
sbatch_resource_flags() {
  local part="${1:-${SBATCH_PARTITION}}"
  local gres="${2-${SBATCH_GRES}}"
  local constraint="${3:-${SBATCH_CONSTRAINT}}"
  [[ -n "${part}" ]] && printf '%s\n' "--partition=${part}"
  [[ -n "${SBATCH_ACCOUNT}" ]] && printf '%s\n' "--account=${SBATCH_ACCOUNT}"
  [[ -n "${SBATCH_QOS}" ]] && printf '%s\n' "--qos=${SBATCH_QOS}"
  # Prefer --gpus when set (some sites); else --gres if non-empty.
  if [[ -n "${SBATCH_GPUS}" ]]; then
    printf '%s\n' "--gpus=${SBATCH_GPUS}"
  elif [[ -n "${gres}" ]]; then
    printf '%s\n' "--gres=${gres}"
  fi
  [[ -n "${constraint}" ]] && printf '%s\n' "--constraint=${constraint}"
  [[ -n "${SBATCH_NODELIST}" ]] && printf '%s\n' "--nodelist=${SBATCH_NODELIST}"
  [[ -n "${SBATCH_TIME}" ]] && printf '%s\n' "--time=${SBATCH_TIME}"
  [[ -n "${SBATCH_CPUS}" ]] && printf '%s\n' "--cpus-per-task=${SBATCH_CPUS}"
  [[ -n "${SBATCH_MEM}" ]] && printf '%s\n' "--mem=${SBATCH_MEM}"
  # Always pin single-node for GPU benches unless user overrode via --extra.
  printf '%s\n' "--nodes=1"
  printf '%s\n' "--ntasks=1"
  if [[ -n "${SBATCH_EXTRA}" ]]; then
    # shellcheck disable=SC2086
    printf '%s\n' ${SBATCH_EXTRA}
  fi
}

submit_profile() {
  local profile_name="$1"
  shift
  local -a dep_args=("$@")

  local profile_file="${COLIDE_ROOT}/dicc_scripts/profiles/${profile_name}.env"
  [[ -f "${profile_file}" ]] || die "Unknown profile '${profile_name}' (expected ${profile_file})"

  # Load profile in a subshell-safe way into current shell
  # shellcheck source=/dev/null
  source "${profile_file}"

  local kernels_dir="${COLIDE_ROOT}/inference/kernels/${COLIDE_KERNELS_SUBDIR}"
  if [[ ! -x "${kernels_dir}/fused_block3" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "WARN: kernels missing at ${kernels_dir} (ok for --dry-run)"
    else
      die "Kernels missing for profile ${profile_name} at ${kernels_dir} — run 01_setup.sh --targets … first"
    fi
  fi

  local job_script="${COLIDE_ROOT}/dicc_scripts/${COLIDE_JOB_SCRIPT:-job_benchmark.sh}"
  [[ -f "${job_script}" ]] || die "Missing job script ${job_script}"

  local export_list="ALL"
  export_list+=",COLIDE_ROOT=${COLIDE_ROOT}"
  export_list+=",COLIDE_CAMPAIGN=${CAMPAIGN}"
  export_list+=",COLIDE_DATE_LABEL=${DATE_LABEL}"
  export_list+=",COLIDE_CHECKPOINT=${COLIDE_CHECKPOINT}"
  export_list+=",COLIDE_N_TRIALS_CUDA=${COLIDE_N_TRIALS_CUDA}"
  export_list+=",COLIDE_N_TRIALS_PYTORCH=${COLIDE_N_TRIALS_PYTORCH}"
  export_list+=",COLIDE_PYTORCH_INNER=${COLIDE_PYTORCH_INNER}"
  export_list+=",COLIDE_GPU_LABEL=${COLIDE_GPU_LABEL}"
  export_list+=",COLIDE_GPU_NAME_RE=${COLIDE_GPU_NAME_RE}"
  export_list+=",COLIDE_GPU_CC=${COLIDE_GPU_CC}"
  export_list+=",COLIDE_GPU_MIN_MEM=${COLIDE_GPU_MIN_MEM}"
  export_list+=",COLIDE_KERNELS_SUBDIR=${COLIDE_KERNELS_SUBDIR}"

  # Profile may override partition/gres/constraint (needed when V100/A100 are
  # different queues, e.g. Rostam cuda-V100 vs cuda-A100).
  local use_part="${COLIDE_PROFILE_PARTITION:-}"
  local use_gres="${COLIDE_PROFILE_GRES:-}"
  local use_constraint="${COLIDE_PROFILE_CONSTRAINT:-}"
  [[ -n "${use_part}" ]] || use_part="${SBATCH_PARTITION}"
  [[ -n "${use_gres}" ]] || use_gres="${SBATCH_GRES}"
  [[ -n "${use_constraint}" ]] || use_constraint="${SBATCH_CONSTRAINT}"

  local -a cmd=(
    sbatch
    --parsable
    --chdir="${COLIDE_ROOT}"
    --job-name="colide_${profile_name}"
    --export="${export_list}"
    --output="${LOG_DIR}/${profile_name}_%j.out"
    --error="${LOG_DIR}/${profile_name}_%j.err"
  )
  while IFS= read -r flag; do
    [[ -n "${flag}" ]] && cmd+=("${flag}")
  done < <(sbatch_resource_flags "${use_part}" "${use_gres}" "${use_constraint}")
  if [[ ${#dep_args[@]} -gt 0 ]]; then
    cmd+=("${dep_args[@]}")
  fi
  cmd+=("${job_script}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf 'DRY-RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    } >&2
    echo "dryrun_${profile_name}"
    return 0
  fi

  command -v sbatch >/dev/null 2>&1 || die "sbatch not found — are you on a SLURM login node?"

  local job_id
  job_id="$("${cmd[@]}")"
  job_id="${job_id%%;*}"
  echo "${job_id}"
}

log "=== Submitting COLIDE session (portable) ==="
log "COLIDE_ROOT=${COLIDE_ROOT}"
log "git_sha=${GIT_SHA}"
log "campaign=${CAMPAIGN} date=${DATE_LABEL} targets=${TARGETS}"
log "slurm_logs=${LOG_DIR}"
log "sbatch gres=${SBATCH_GRES} partition=${SBATCH_PARTITION:-<default>} account=${SBATCH_ACCOUNT:-<default>}"

IFS=',' read -r -a target_arr <<< "${TARGETS}"
# Parallel arrays (bash 3.2+ portable; no associative arrays).
JOB_NAMES=()
JOB_IDS=()
LAST_JOB=""
A100_JOB=""

for raw in "${target_arr[@]}"; do
  t="$(echo "${raw}" | tr -d ' ')"
  [[ -n "${t}" ]] || continue
  jid="$(submit_profile "${t}")"
  JOB_NAMES+=("${t}")
  JOB_IDS+=("${jid}")
  LAST_JOB="${jid}"
  log "Submitted profile=${t} job=${jid}"
  if [[ "${t}" == "a100" ]]; then
    A100_JOB="${jid}"
  fi
done

NSIGHT_JOB=""
if [[ "${WITH_NSIGHT}" == "1" ]]; then
  dep_job="${A100_JOB:-${LAST_JOB}}"
  [[ -n "${dep_job}" ]] || die "--with-nsight requires at least one submitted target"
  if [[ "${DRY_RUN}" == "1" ]]; then
    NSIGHT_JOB="dryrun_nsight"
    {
      printf 'DRY-RUN: sbatch … 04_nsight_profile.sh --dependency=afterok:%s\n' "${dep_job}"
    } >&2
  else
    # shellcheck source=profiles/a100.env
    source "${COLIDE_ROOT}/dicc_scripts/profiles/a100.env"
    export_list="ALL,COLIDE_ROOT=${COLIDE_ROOT},COLIDE_CAMPAIGN=${CAMPAIGN},COLIDE_DATE_LABEL=${DATE_LABEL},COLIDE_CHECKPOINT=${COLIDE_CHECKPOINT},COLIDE_GPU_LABEL=${COLIDE_GPU_LABEL},COLIDE_GPU_NAME_RE=${COLIDE_GPU_NAME_RE},COLIDE_GPU_CC=${COLIDE_GPU_CC},COLIDE_GPU_MIN_MEM=${COLIDE_GPU_MIN_MEM},COLIDE_KERNELS_SUBDIR=${COLIDE_KERNELS_SUBDIR}"
    ns_cmd=(
      sbatch --parsable --chdir="${COLIDE_ROOT}"
      --job-name=colide_nsight
      --export="${export_list}"
      --output="${LOG_DIR}/nsight_%j.out"
      --error="${LOG_DIR}/nsight_%j.err"
      --dependency="afterok:${dep_job}"
    )
    while IFS= read -r flag; do
      [[ -n "${flag}" ]] && ns_cmd+=("${flag}")
    done < <(sbatch_resource_flags)
    ns_cmd+=("${COLIDE_ROOT}/dicc_scripts/04_nsight_profile.sh")
    NSIGHT_JOB="$("${ns_cmd[@]}")"
    NSIGHT_JOB="${NSIGHT_JOB%%;*}"
  fi
  log "Submitted Nsight job (afterok:${dep_job}): ${NSIGHT_JOB}"
else
  log "Nsight not requested (pass --with-nsight to enable)"
fi

# Build jobs JSON object
jobs_json="{"
first=1
idx=0
while [[ ${idx} -lt ${#JOB_NAMES[@]} ]]; do
  if [[ ${first} -eq 1 ]]; then first=0; else jobs_json+=","; fi
  jobs_json+=" \"${JOB_NAMES[$idx]}\": \"${JOB_IDS[$idx]}\""
  idx=$((idx + 1))
done
jobs_json+=", \"nsight\": \"${NSIGHT_JOB}\" }"

SESSION_JSON="${SESSION_DIR}/session.json"
cat > "${SESSION_JSON}" <<EOF
{
  "campaign": "${CAMPAIGN}",
  "date_label": "${DATE_LABEL}",
  "git_sha": "${GIT_SHA}",
  "colide_root": "${COLIDE_ROOT}",
  "checkpoint": "${COLIDE_CHECKPOINT}",
  "targets": "${TARGETS}",
  "n_trials_cuda": ${COLIDE_N_TRIALS_CUDA},
  "n_trials_pytorch": ${COLIDE_N_TRIALS_PYTORCH},
  "pytorch_inner_forwards": ${COLIDE_PYTORCH_INNER},
  "sbatch": {
    "partition": "${SBATCH_PARTITION}",
    "account": "${SBATCH_ACCOUNT}",
    "gres": "${SBATCH_GRES}",
    "constraint": "${SBATCH_CONSTRAINT}",
    "nodelist": "${SBATCH_NODELIST}",
    "time": "${SBATCH_TIME}",
    "cpus": "${SBATCH_CPUS}",
    "mem": "${SBATCH_MEM}"
  },
  "jobs": ${jobs_json},
  "slurm_log_dir": "${LOG_DIR}",
  "submitted_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "with_nsight": $([[ "${WITH_NSIGHT}" == "1" ]] && echo true || echo false)
}
EOF

log "Session record: ${SESSION_JSON}"
log "Check queue: squeue -u ${USER}"
log "After two distinct days complete, compare with:"
log "  PYTHONPATH=. python scripts/compare_dicc_sessions.py --gpu <label> --date-a D1 --date-b D2"
