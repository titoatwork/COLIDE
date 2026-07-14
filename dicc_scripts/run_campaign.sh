#!/usr/bin/env bash
# =============================================================================
# COLIDE — one-command cluster campaign
#
# Usage (any machine / any SLURM site):
#   bash dicc_scripts/run_campaign.sh --full     # Day1 + wait + Day2 + wait + compare
#   bash dicc_scripts/run_campaign.sh            # Day 1 only
#   bash dicc_scripts/run_campaign.sh --day 2
#
# Auto-detects: repo root, Python venv, nvcc, SLURM partitions, GRES, compile path.
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

export COLIDE_ROOT="${COLIDE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${COLIDE_ROOT}"
require_colide_root

DAY="${COLIDE_DAY:-1}"
TARGETS=""
DRY_RUN=0
FORCE_LOCAL=0
SKIP_SETUP=0
WAIT=0
FULL=0
N_TRIALS_CUDA="${COLIDE_N_TRIALS_CUDA:-100}"
N_TRIALS_PT="${COLIDE_N_TRIALS_PYTORCH:-20}"

usage() {
  cat <<'EOF'
COLIDE one-command campaign runner

  bash dicc_scripts/run_campaign.sh --full       # ★ Day1 + Day2 + compare (one shot)
  bash dicc_scripts/run_campaign.sh              # Day 1 only
  bash dicc_scripts/run_campaign.sh --day 2      # Day 2 only
  bash dicc_scripts/run_campaign.sh --local
  bash dicc_scripts/run_campaign.sh --targets v100,a100
  bash dicc_scripts/run_campaign.sh --wait
  bash dicc_scripts/run_campaign.sh --dry-run

--full always waits for SLURM jobs between days and runs compare at the end.
Day-2 date label is <UTCDAY>_d2 so it works even on the same calendar day.

Environment overrides (optional):
  COLIDE_V100_PARTITION  COLIDE_A100_PARTITION
  COLIDE_SBATCH_GRES     (empty to omit; default auto)
  COLIDE_SBATCH_ACCOUNT  COLIDE_SBATCH_PARTITION
  TORCH_INDEX_URL        (default cu121 for V100+A100 support)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=1; WAIT=1; DAY=1; shift ;;
    --day) DAY="${2:?}"; shift 2 ;;
    --targets) TARGETS="${2:?}"; shift 2 ;;
    --local) FORCE_LOCAL=1; shift ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --wait) WAIT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --n-trials-cuda) N_TRIALS_CUDA="${2:?}"; shift 2 ;;
    --n-trials-pytorch) N_TRIALS_PT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (try --help)" ;;
  esac
done

log "=== COLIDE run_campaign ==="
log "COLIDE_ROOT=${COLIDE_ROOT}"
log "host=$(hostname) day=${DAY} full=${FULL}"

# ---------------------------------------------------------------------------
# Site auto-detection
# ---------------------------------------------------------------------------
have_slurm=0
command -v sbatch >/dev/null 2>&1 && command -v sinfo >/dev/null 2>&1 && have_slurm=1

detect_partition() {
  # args: regex patterns to match partition names (first match wins)
  local pat part
  if [[ "${have_slurm}" != "1" ]]; then
    echo ""
    return 0
  fi
  while [[ $# -gt 0 ]]; do
    pat="$1"; shift
    part="$(sinfo -h -o '%P' 2>/dev/null | sed 's/\*//g' | grep -E -i "${pat}" | head -1 || true)"
    if [[ -n "${part}" ]]; then
      echo "${part}"
      return 0
    fi
  done
  echo ""
}

partition_gres_is_null() {
  local part="$1"
  [[ -z "${part}" ]] && return 0
  local g
  g="$(sinfo -h -p "${part}" -o '%G' 2>/dev/null | head -1 | tr -d ' ' || true)"
  [[ -z "${g}" || "${g}" == "(null)" || "${g}" == "N/A" ]]
}

V100_PART="${COLIDE_V100_PARTITION:-$(detect_partition 'cuda-?v100' 'v100' 'gpu.*v100')}"
A100_PART="${COLIDE_A100_PARTITION:-$(detect_partition 'cuda-?a100' 'a100' 'dgx' 'gpu.*a100')}"
# Generic fallbacks
if [[ -z "${V100_PART}" && -z "${A100_PART}" ]]; then
  GEN_PART="${COLIDE_SBATCH_PARTITION:-$(detect_partition '^cuda$' 'gpu' 'gpu-')}"
  V100_PART="${V100_PART:-${GEN_PART}}"
  A100_PART="${A100_PART:-${GEN_PART}}"
fi

log "detected partitions: v100='${V100_PART:-none}' a100='${A100_PART:-none}' slurm=${have_slurm}"

# Auto targets if not specified
if [[ -z "${TARGETS}" ]]; then
  tlist=()
  [[ -n "${V100_PART}" ]] && tlist+=("v100")
  [[ -n "${A100_PART}" ]] && tlist+=("a100")
  if [[ ${#tlist[@]} -eq 0 ]]; then
    # Local GPU guess
    if command -v nvidia-smi >/dev/null 2>&1; then
      name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
      if echo "${name}" | grep -qi 'A100'; then tlist+=("a100")
      elif echo "${name}" | grep -qi 'V100'; then tlist+=("v100")
      else
        # Compile for this machine's arch if possible later; default both profiles if unknown
        tlist+=("v100")
      fi
    else
      tlist+=("v100" "a100")
    fi
  fi
  TARGETS=$(IFS=,; echo "${tlist[*]}")
fi
log "targets=${TARGETS}"

# GRES: omit when partition reports null (Rostam pattern)
if [[ -z "${COLIDE_SBATCH_GRES+x}" ]]; then
  # unset by user → auto
  if partition_gres_is_null "${V100_PART}" && partition_gres_is_null "${A100_PART}"; then
    export COLIDE_SBATCH_GRES=""
    GRES_FLAG=(--gres none)
    log "auto GRES: none (sinfo reports null/empty)"
  else
    export COLIDE_SBATCH_GRES=gpu:1
    GRES_FLAG=(--gres gpu:1)
    log "auto GRES: gpu:1"
  fi
else
  if [[ -z "${COLIDE_SBATCH_GRES}" ]]; then
    GRES_FLAG=(--gres none)
  else
    GRES_FLAG=(--gres "${COLIDE_SBATCH_GRES}")
  fi
  log "GRES from env: '${COLIDE_SBATCH_GRES}'"
fi

export COLIDE_N_TRIALS_CUDA="${N_TRIALS_CUDA}"
export COLIDE_N_TRIALS_PYTORCH="${N_TRIALS_PT}"
export COLIDE_ALLOW_DIRTY=1
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

UTC_DAY="$(date -u +%Y%m%d)"
mkdir -p benchmarks/results/dicc
if [[ -n "${COLIDE_DATE_LABEL:-}" ]]; then
  DATE_LABEL="${COLIDE_DATE_LABEL}"
elif [[ "${DAY}" == "2" ]]; then
  # Distinct from Day 1 even on the same calendar day (required by compare_dicc_sessions).
  d1="$(awk -F= '/^DAY1=/{print $2; exit}' benchmarks/results/dicc/DAY1_LABEL.txt 2>/dev/null || true)"
  if [[ -n "${d1}" ]]; then
    DATE_LABEL="${d1}_d2"
  else
    DATE_LABEL="${UTC_DAY}_d2"
  fi
  echo "DAY2=${DATE_LABEL}" | tee benchmarks/results/dicc/DAY2_LABEL.txt >/dev/null
  echo "GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)" | tee -a benchmarks/results/dicc/DAY2_LABEL.txt >/dev/null
else
  DATE_LABEL="${UTC_DAY}"
  echo "DAY1=${DATE_LABEL}" | tee benchmarks/results/dicc/DAY1_LABEL.txt >/dev/null
  echo "GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)" | tee -a benchmarks/results/dicc/DAY1_LABEL.txt >/dev/null
fi
export COLIDE_DATE_LABEL="${DATE_LABEL}"
export COLIDE_CAMPAIGN="${COLIDE_CAMPAIGN:-core}"

# ---------------------------------------------------------------------------
# Setup Python (login-safe)
# ---------------------------------------------------------------------------
ensure_python() {
  if [[ "${SKIP_SETUP}" == "1" ]]; then
    log "skip setup"
    return 0
  fi
  if [[ -x "${COLIDE_ROOT}/.venv-cluster/bin/python" ]]; then
    if "${COLIDE_ROOT}/.venv-cluster/bin/python" -c "import torch,numpy,yaml" 2>/dev/null; then
      log "python env OK: .venv-cluster"
      return 0
    fi
  fi
  log "Setting up Python env (venv + torch cu121)…"
  if command -v module >/dev/null 2>&1; then
    module load python/3.12.3 2>/dev/null || module load python 2>/dev/null || true
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY-RUN: would run 01_setup.sh --python-only"
    return 0
  fi
  bash "${SCRIPT_DIR}/01_setup.sh" --python-only --allow-dirty
}

# ---------------------------------------------------------------------------
# Ensure kernels for each target
# ---------------------------------------------------------------------------
arch_for_target() {
  case "$1" in
    v100) echo "sm_70:v100" ;;
    a100) echo "sm_80:a100" ;;
    *) die "unknown target profile: $1" ;;
  esac
}

part_for_target() {
  case "$1" in
    v100) echo "${V100_PART}" ;;
    a100) echo "${A100_PART}" ;;
    *) echo "" ;;
  esac
}

kernel_ok() {
  local t="$1"
  local sub
  sub="$(arch_for_target "${t}")"
  sub="${sub#*:}"
  [[ -x "${COLIDE_ROOT}/inference/kernels/${sub}/fused_block3" ]]
}

ensure_nvcc_on_path() {
  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  local n
  if n="$(find_nvcc)"; then
    export PATH="$(dirname "${n}"):${PATH}"
    log "found nvcc: ${n}"
    return 0
  fi
  return 1
}

compile_target() {
  local t="$1"
  local pair part
  pair="$(arch_for_target "${t}")"
  part="$(part_for_target "${t}")"

  if kernel_ok "${t}"; then
    log "kernels already present for ${t}"
    return 0
  fi

  log "compiling kernels for ${t} (${pair})…"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY-RUN: would compile ${pair}"
    return 0
  fi

  # Prefer local compile when nvcc is available (GPU node interactive or workstation)
  if ensure_nvcc_on_path; then
    bash "${SCRIPT_DIR}/01_setup.sh" --kernels-only --allow-dirty --targets "${pair}"
    kernel_ok "${t}" || die "local compile failed for ${t}"
    return 0
  fi

  # Login node without nvcc: submit a one-shot compile job on the GPU partition
  [[ "${have_slurm}" == "1" ]] || die "nvcc not found and no SLURM — compile on a CUDA node"
  [[ -n "${part}" ]] || die "no partition detected for ${t}; set COLIDE_V100_PARTITION / COLIDE_A100_PARTITION"

  local wrap jid out err
  out="${COLIDE_ROOT}/benchmarks/results/dicc/compile_${t}_%j.out"
  err="${COLIDE_ROOT}/benchmarks/results/dicc/compile_${t}_%j.err"
  mkdir -p "${COLIDE_ROOT}/benchmarks/results/dicc"
  wrap="set -Eeuo pipefail; cd '${COLIDE_ROOT}'; export COLIDE_ROOT='${COLIDE_ROOT}'; export PATH=/usr/local/cuda/bin:\$PATH; command -v nvcc || true; bash dicc_scripts/01_setup.sh --kernels-only --allow-dirty --targets ${pair}"

  local -a scmd=(
    sbatch --parsable --job-name="colide_compile_${t}"
    --partition="${part}" --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=01:00:00
    --chdir="${COLIDE_ROOT}" --output="${out}" --error="${err}"
  )
  if [[ -n "${COLIDE_SBATCH_GRES}" ]]; then
    scmd+=(--gres="${COLIDE_SBATCH_GRES}")
  fi
  scmd+=(--wrap="${wrap}")

  log "sbatch compile on partition=${part}"
  jid="$("${scmd[@]}")"
  jid="${jid%%;*}"
  log "compile job ${t}: ${jid} — waiting…"
  # Wait for job
  while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do
    sleep 15
  done
  kernel_ok "${t}" || {
    log "compile failed — last err:"
    ls -t "${COLIDE_ROOT}/benchmarks/results/dicc/compile_${t}_"*.err 2>/dev/null | head -1 | xargs -I{} tail -40 {} >&2 || true
    die "kernels still missing for ${t} after compile job ${jid}"
  }
  log "kernels ready for ${t}"
}

# ---------------------------------------------------------------------------
# Run or submit benchmarks
# ---------------------------------------------------------------------------
run_local_target() {
  local t="$1"
  local pair sub part
  pair="$(arch_for_target "${t}")"
  sub="${pair#*:}"
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/profiles/${t}.env"
  export COLIDE_KERNELS_SUBDIR="${sub}"
  export COLIDE_GPU_LABEL COLIDE_GPU_NAME_RE COLIDE_GPU_CC COLIDE_GPU_MIN_MEM
  export COLIDE_CAMPAIGN COLIDE_DATE_LABEL
  export COLIDE_N_TRIALS_CUDA COLIDE_N_TRIALS_PYTORCH
  export PYTHONPATH="${COLIDE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ -f "${COLIDE_ROOT}/.venv-cluster/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${COLIDE_ROOT}/.venv-cluster/bin/activate"
  fi
  log "LOCAL run profile=${t}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY-RUN: would exec lib/run_benchmark.sh for ${t}"
    return 0
  fi
  bash "${SCRIPT_DIR}/lib/run_benchmark.sh"
}

submit_target() {
  local t="$1"
  local part
  part="$(part_for_target "${t}")"
  [[ -n "${part}" ]] || die "no partition for ${t}"

  local -a cmd=(
    bash "${SCRIPT_DIR}/submit_session.sh"
    --targets "${t}"
    --campaign "${COLIDE_CAMPAIGN}"
    --date "${COLIDE_DATE_LABEL}"
    --partition "${part}"
    --allow-dirty
  )
  cmd+=("${GRES_FLAG[@]}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  log "submit: ${t} -> ${part}"
  "${cmd[@]}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ensure_python

IFS=',' read -r -a target_arr <<< "${TARGETS}"
for raw in "${target_arr[@]}"; do
  t="$(echo "${raw}" | tr -d ' ')"
  [[ -n "${t}" ]] || continue
  compile_target "${t}"
done

JOB_IDS=()
if [[ "${FORCE_LOCAL}" == "1" || "${have_slurm}" != "1" ]]; then
  log "mode=local"
  ensure_nvcc_on_path || log "WARN: nvcc missing (ok if kernels prebuilt)"
  if [[ "${DRY_RUN}" != "1" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "local mode needs a GPU (nvidia-smi)"
  fi
  for raw in "${target_arr[@]}"; do
    t="$(echo "${raw}" | tr -d ' ')"
    [[ -n "${t}" ]] || continue
    run_local_target "${t}"
  done
else
  log "mode=slurm"
  for raw in "${target_arr[@]}"; do
    t="$(echo "${raw}" | tr -d ' ')"
    [[ -n "${t}" ]] || continue
    submit_target "${t}"
  done
fi

log "=== campaign submit/run finished (day=${DAY} date=${COLIDE_DATE_LABEL}) ==="
log "Results root: ${COLIDE_ROOT}/benchmarks/results/dicc/${COLIDE_CAMPAIGN}/"
log "Check SUCCESS: find ${COLIDE_ROOT}/benchmarks/results/dicc -name SUCCESS | sort"

wait_for_colide_jobs() {
  if [[ "${have_slurm}" != "1" || "${FORCE_LOCAL}" == "1" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  log "waiting for colide_* jobs to leave the queue…"
  while squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -qE '^colide_'; do
    squeue -u "${USER}" | head -20 || true
    sleep 60
  done
  log "queue clear for colide_* jobs"
  find "${COLIDE_ROOT}/benchmarks/results/dicc" -name SUCCESS | sort || true
}

if [[ "${WAIT}" == "1" ]]; then
  wait_for_colide_jobs
fi

# --full: after Day 1 completes, immediately run Day 2 (distinct date label) + compare
if [[ "${FULL}" == "1" && "${DAY}" == "1" ]]; then
  log "=== FULL mode: starting Day 2 ==="
  d1_label="${COLIDE_DATE_LABEL}"
  export COLIDE_DATE_LABEL="${d1_label}_d2"
  echo "DAY2=${COLIDE_DATE_LABEL}" | tee benchmarks/results/dicc/DAY2_LABEL.txt >/dev/null
  echo "GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)" | tee -a benchmarks/results/dicc/DAY2_LABEL.txt >/dev/null
  SKIP_SETUP=1
  # Re-submit only (kernels/python already ready)
  if [[ "${FORCE_LOCAL}" == "1" || "${have_slurm}" != "1" ]]; then
    for raw in "${target_arr[@]}"; do
      t="$(echo "${raw}" | tr -d ' ')"
      [[ -n "${t}" ]] || continue
      run_local_target "${t}"
    done
  else
    for raw in "${target_arr[@]}"; do
      t="$(echo "${raw}" | tr -d ' ')"
      [[ -n "${t}" ]] || continue
      submit_target "${t}"
    done
    wait_for_colide_jobs
  fi

  log "=== FULL mode: cross-day compare ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY-RUN: would compare ${d1_label} vs ${COLIDE_DATE_LABEL}"
  else
    if [[ -f "${COLIDE_ROOT}/.venv-cluster/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "${COLIDE_ROOT}/.venv-cluster/bin/activate"
    fi
    export PYTHONPATH="${COLIDE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    for g in v100s a100; do
      if compgen -G "${COLIDE_ROOT}/benchmarks/results/dicc/${COLIDE_CAMPAIGN}/${g}/${d1_label}_job*/SUCCESS" >/dev/null \
        && compgen -G "${COLIDE_ROOT}/benchmarks/results/dicc/${COLIDE_CAMPAIGN}/${g}/${COLIDE_DATE_LABEL}_job*/SUCCESS" >/dev/null; then
        log "compare gpu=${g} ${d1_label} vs ${COLIDE_DATE_LABEL}"
        python "${COLIDE_ROOT}/scripts/compare_dicc_sessions.py" \
          --campaign "${COLIDE_CAMPAIGN}" \
          --gpu "${g}" \
          --date-a "${d1_label}" \
          --date-b "${COLIDE_DATE_LABEL}" \
          || log "WARN: compare failed for ${g} (check SUCCESS + provenance)"
      else
        log "WARN: skip compare for ${g} — missing SUCCESS on one or both days"
      fi
    done
  fi
  log "=== FULL campaign complete ==="
fi
