#!/usr/bin/env bash
# Shared helpers for COLIDE DICC scripts.
# shellcheck shell=bash

# Resolve absolute path of this file's directory, then repo-relative anchors.
_DICC_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DICC_SCRIPTS_DIR="$(cd "${_DICC_LIB_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
# Prefer an explicitly exported COLIDE_ROOT. Otherwise derive from this
# file's location (repo/dicc_scripts/lib/common.sh -> repo/).
resolve_colide_root() {
  if [[ -n "${COLIDE_ROOT:-}" ]]; then
    (cd "${COLIDE_ROOT}" && pwd)
    return
  fi
  (cd "${_DICC_SCRIPTS_DIR}/.." && pwd)
}

# Require COLIDE_ROOT to already be set and to look like the COLIDE repo.
require_colide_root() {
  if [[ -z "${COLIDE_ROOT:-}" ]]; then
    echo "ERROR: COLIDE_ROOT is not set. Export it before running DICC jobs." >&2
    return 1
  fi
  if [[ ! -d "${COLIDE_ROOT}/dicc_scripts" ]] || [[ ! -d "${COLIDE_ROOT}/inference/kernels" ]]; then
    echo "ERROR: COLIDE_ROOT=${COLIDE_ROOT} does not look like the COLIDE repository." >&2
    return 1
  fi
  COLIDE_ROOT="$(cd "${COLIDE_ROOT}" && pwd)"
  export COLIDE_ROOT
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

# ---------------------------------------------------------------------------
# Conda (non-interactive safe)
# ---------------------------------------------------------------------------
init_conda() {
  # Fail hard if conda cannot be initialized — non-interactive Slurm jobs
  # do not source ~/.bashrc, so `conda activate` alone is a silent no-op
  # or hard failure depending on how the module was installed.
  #
  # COLIDE_CONDA_DISABLE_AUTODETECT=1: only accept conda already on PATH
  # (used by local validation mocks).
  if ! command -v conda >/dev/null 2>&1; then
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
      :
    elif [[ "${COLIDE_CONDA_DISABLE_AUTODETECT:-0}" != "1" ]]; then
      # Try common module-provided locations after `module load miniconda`.
      local candidate
      for candidate in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        /opt/conda/etc/profile.d/conda.sh \
        /usr/local/miniconda3/etc/profile.d/conda.sh; do
        if [[ -f "${candidate}" ]]; then
          # shellcheck source=/dev/null
          source "${candidate}"
          break
        fi
      done
    fi
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null)" || die "conda info --base failed"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "${conda_base}/etc/profile.d/conda.sh"
    else
      eval "$(conda shell.bash hook 2>/dev/null)" \
        || die "Unable to initialize conda shell hook from ${conda_base}"
    fi
  else
    die "conda not found on PATH. Load the miniconda module and re-run."
  fi
}

activate_colide_env() {
  init_conda
  conda activate colide || die "Failed to activate conda env 'colide'. Run 01_setup.sh first."
  command -v python >/dev/null 2>&1 || die "python not available after conda activate colide"
}

# ---------------------------------------------------------------------------
# GPU assertions (compute node only)
# ---------------------------------------------------------------------------
# Expected identity is passed as:
#   assert_gpu <label> <name_regex> <compute_capability> <min_mem_mib>
# Example: assert_gpu v100s 'V100' 7.0 30000
assert_gpu() {
  local label="$1"
  local name_regex="$2"
  local expect_cc="$3"
  local min_mem_mib="$4"

  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi not found (expected on compute nodes, not login nodes)"

  local gpu_count
  gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${gpu_count}" != "1" ]]; then
    die "Expected exactly 1 visible GPU for ${label}, found ${gpu_count}. Refusing multi-GPU/MIG-shared runs."
  fi

  local name
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -n "${name}" ]] || die "Unable to query GPU name"
  if ! [[ "${name}" =~ ${name_regex} ]]; then
    die "GPU name '${name}' does not match expected pattern /${name_regex}/ for ${label}"
  fi

  # MIG: if MIG mode is enabled or nvidia-smi reports MIG devices, refuse.
  local mig_mode
  mig_mode="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ "${mig_mode}" == "Enabled" ]]; then
    die "GPU is in MIG mode (current=${mig_mode}); full-device allocation required for ${label}"
  fi
  if nvidia-smi -L 2>/dev/null | grep -qi 'MIG'; then
    die "Visible device list includes MIG instances; full-device allocation required for ${label}"
  fi

  local mem_total
  mem_total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  [[ -n "${mem_total}" ]] || die "Unable to query GPU memory"
  if (( mem_total < min_mem_mib )); then
    die "GPU memory ${mem_total} MiB is below minimum ${min_mem_mib} MiB for ${label}"
  fi

  # Compute capability via nvidia-smi compute_cap if available, else python/torch later.
  local cc
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ -n "${cc}" && "${cc}" != "[N/A]" ]]; then
    if [[ "${cc}" != "${expect_cc}" ]]; then
      die "Compute capability ${cc} != expected ${expect_cc} for ${label}"
    fi
  else
    log "WARN: compute_cap not reported by nvidia-smi; will re-check via torch if available"
  fi

  export COLIDE_GPU_LABEL="${label}"
  export COLIDE_GPU_NAME="${name}"
  export COLIDE_GPU_CC="${cc:-unknown}"
  export COLIDE_GPU_MEM_MIB="${mem_total}"
  log "GPU OK: label=${label} name='${name}' cc=${COLIDE_GPU_CC} mem=${mem_total}MiB"
}

# ---------------------------------------------------------------------------
# Result directory isolation
# ---------------------------------------------------------------------------
# Layout: benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/
make_run_dir() {
  local campaign="$1"
  local gpu_label="$2"
  local date_label="${3:-}"
  local job_id="${4:-local}"

  if [[ -z "${date_label}" ]]; then
    date_label="$(date -u +%Y%m%d)"
  fi

  require_colide_root
  local run_dir="${COLIDE_ROOT}/benchmarks/results/dicc/${campaign}/${gpu_label}/${date_label}_job${job_id}"
  mkdir -p "${run_dir}/logs" "${run_dir}/raw"
  echo "${run_dir}"
}

write_manifest() {
  local run_dir="$1"
  require_colide_root
  local sha
  sha="$(git -C "${COLIDE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  local dirty
  if git -C "${COLIDE_ROOT}" status --porcelain 2>/dev/null | grep -q .; then
    dirty=true
  else
    dirty=false
  fi

  cat > "${run_dir}/manifest.json" <<EOF
{
  "campaign": "${COLIDE_CAMPAIGN:-}",
  "gpu_label": "${COLIDE_GPU_LABEL:-}",
  "gpu_name": "${COLIDE_GPU_NAME:-}",
  "gpu_compute_capability": "${COLIDE_GPU_CC:-}",
  "gpu_memory_mib": ${COLIDE_GPU_MEM_MIB:-0},
  "date_label": "${COLIDE_DATE_LABEL:-}",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_job_name": "${SLURM_JOB_NAME:-}",
  "slurm_nodelist": "${SLURM_NODELIST:-}",
  "hostname": "$(hostname)",
  "git_sha": "${sha}",
  "git_dirty": ${dirty},
  "colide_root": "${COLIDE_ROOT}",
  "kernels_dir": "${COLIDE_KERNELS_DIR:-}",
  "checkpoint": "${COLIDE_CHECKPOINT:-}",
  "n_trials_cuda": ${COLIDE_N_TRIALS_CUDA:-100},
  "n_trials_pytorch": ${COLIDE_N_TRIALS_PYTORCH:-20},
  "pytorch_inner_forwards": ${COLIDE_PYTORCH_INNER:-1000},
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

write_environment() {
  local run_dir="$1"
  {
    echo "=== date ==="
    date -u
    echo "=== hostname ==="
    hostname
    echo "=== uname ==="
    uname -a
    echo "=== COLIDE_ROOT ==="
    echo "${COLIDE_ROOT:-}"
    echo "=== git ==="
    git -C "${COLIDE_ROOT}" rev-parse HEAD 2>/dev/null || true
    git -C "${COLIDE_ROOT}" status --porcelain 2>/dev/null || true
    echo "=== modules ==="
    module list 2>&1 || true
    echo "=== which ==="
    command -v python || true
    command -v nvcc || true
    command -v nvidia-smi || true
    echo "=== python ==="
    python - <<'PY' 2>&1 || true
import sys
print("executable", sys.executable)
print("version", sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        p = torch.cuda.get_device_properties(0)
        print("props", p.major, p.minor, p.total_memory)
except Exception as e:
    print("torch_error", e)
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as e:
    print("numpy_error", e)
PY
    echo "=== nvidia-smi ==="
    nvidia-smi 2>&1 || true
    echo "=== env (selected) ==="
    env | grep -E '^(CUDA|SLURM|COLIDE|PATH|LD_|CONDA|PYTHON)' | sort || true
  } > "${run_dir}/environment.txt"
}

copy_kernel_checksums() {
  local run_dir="$1"
  local kernels_dir="$2"
  if [[ -f "${kernels_dir}/SHA256SUMS" ]]; then
    cp -f "${kernels_dir}/SHA256SUMS" "${run_dir}/kernel_SHA256SUMS"
  else
    (cd "${kernels_dir}" && sha256sum ./* 2>/dev/null || true) > "${run_dir}/kernel_SHA256SUMS"
  fi
}

mark_success() {
  local run_dir="$1"
  local exit_code="${2:-0}"
  printf '%s\n' "${exit_code}" > "${run_dir}/exit_status"
  if [[ "${exit_code}" == "0" ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/SUCCESS"
    log "Run marked SUCCESS: ${run_dir}"
  else
    rm -f "${run_dir}/SUCCESS"
    log "Run FAILED (exit=${exit_code}): ${run_dir}"
  fi
}
