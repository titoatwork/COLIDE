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
# If unset, resolve from this file's location (…/dicc_scripts/lib → repo root).
require_colide_root() {
  if [[ -z "${COLIDE_ROOT:-}" ]]; then
    COLIDE_ROOT="$(resolve_colide_root)"
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
  # Always stderr so command substitutions (e.g. job_id="$(submit_profile …)")
  # never capture log lines as fake output.
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2
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

# Activate Python for benchmarks. Prefer (1) repo venv python binary,
# (2) conda env colide, (3) already-working system/module python.
# On heterogeneous clusters, `source activate` + bare `python` can still hit
# a login/module interpreter without site-packages — always pin venv/bin.
activate_colide_env() {
  local root="${COLIDE_ROOT:-$(resolve_colide_root)}"
  local venv_dir="${COLIDE_VENV:-${root}/.venv-cluster}"
  local py=""

  # Stale PYTHONHOME from sbatch --export=ALL breaks venv site-packages.
  unset PYTHONHOME || true

  if [[ -x "${venv_dir}/bin/python" ]]; then
    export VIRTUAL_ENV="${venv_dir}"
    export PATH="${venv_dir}/bin:${PATH}"
    hash -r 2>/dev/null || true
    # shellcheck source=/dev/null
    [[ -f "${venv_dir}/bin/activate" ]] && source "${venv_dir}/bin/activate"
    py="${venv_dir}/bin/python"
    log "Using venv python: ${py}"
  elif command -v conda >/dev/null 2>&1 || [[ "${COLIDE_FORCE_CONDA:-0}" == "1" ]]; then
    init_conda
    conda activate colide || die "Failed to activate conda env 'colide'. Run 01_setup.sh first."
    py="$(command -v python)"
    log "Using conda env: colide (${py})"
  elif [[ -x "${root}/.venv/bin/python" ]]; then
    export VIRTUAL_ENV="${root}/.venv"
    export PATH="${root}/.venv/bin:${PATH}"
    hash -r 2>/dev/null || true
    # shellcheck source=/dev/null
    [[ -f "${root}/.venv/bin/activate" ]] && source "${root}/.venv/bin/activate"
    py="${root}/.venv/bin/python"
    log "Using venv python: ${py}"
  else
    log "WARN: no colide conda env or .venv-cluster; using python on PATH"
    py="$(command -v python || true)"
  fi

  [[ -n "${py}" && -x "${py}" ]] || die "python not available after env activation"
  export COLIDE_PYTHON="${py}"
  # Ensure subsequent bare `python` hits the same interpreter.
  if [[ "$(command -v python 2>/dev/null || true)" != "${py}" ]]; then
    export PATH="$(dirname "${py}"):${PATH}"
    hash -r 2>/dev/null || true
  fi

  "${py}" - <<'PY' || die "python missing numpy/torch — re-run: .venv-cluster/bin/pip install numpy scipy pyyaml torch (cu121)"
import sys
print("python_ok", sys.executable)
import numpy, yaml
print("numpy", numpy.__version__, "at", getattr(numpy, "__file__", "?"))
try:
    import torch
except Exception as e:
    raise SystemExit(f"torch import failed: {e}")
print("torch", torch.__version__)
PY
}

# Locate nvcc without requiring a module named "cuda" (many sites only ship
# the toolkit on GPU images or under /usr/local/cuda*).
find_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return 0
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/bin/nvcc \
    /usr/local/cuda-12.4/bin/nvcc \
    /usr/local/cuda-12.3/bin/nvcc \
    /usr/local/cuda-12.2/bin/nvcc \
    /usr/local/cuda-12.1/bin/nvcc \
    /usr/local/cuda-12.0/bin/nvcc \
    /usr/local/cuda-11.8/bin/nvcc \
    /usr/local/cuda-11.7/bin/nvcc \
    /opt/cuda/bin/nvcc \
    /opt/nvidia/hpc_sdk/*/compilers/bin/nvcc; do
    # shellcheck disable=SC2086
    for c in ${candidate}; do
      if [[ -x "${c}" ]]; then
        echo "${c}"
        return 0
      fi
    done
  done
  # Last resort: filesystem search limited to common roots (fast-fail if missing)
  if command -v find >/dev/null 2>&1; then
    candidate="$(find /usr/local /opt -name nvcc -type f 2>/dev/null | head -1 || true)"
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi
  return 1
}

# ---------------------------------------------------------------------------
# GPU assertions (compute node only)
# ---------------------------------------------------------------------------
# Expected identity is passed as:
#   assert_gpu <label> <name_regex> <compute_capability> <min_mem_mib>
# Example: assert_gpu v100s 'V100' 7.0 30000
#
# On sites without GRES isolation (e.g. Rostam), a job may see every GPU on
# the node (2x V100, 4x A100). We pin to CUDA_VISIBLE_DEVICES (default 0)
# so benchmarks use exactly one device and name/CC checks apply to that card.
assert_gpu() {
  local label="$1"
  local name_regex="$2"
  local expect_cc="$3"
  local min_mem_mib="$4"

  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi not found (expected on compute nodes, not login nodes)"

  local gpu_count
  gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -z "${gpu_count}" || "${gpu_count}" -lt 1 ]]; then
    die "No GPUs visible via nvidia-smi for ${label}"
  fi

  # Pin to one physical index before queries (so multi-GPU nodes are usable).
  local dev_index="${COLIDE_CUDA_DEVICE:-0}"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${dev_index}"
    log "Pinned CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${gpu_count} GPUs visible on node)"
  else
    log "Using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${gpu_count} GPUs on node before mask)"
  fi

  # After masking, nvidia-smi -L may still show all GPUs depending on driver;
  # query by index from the original list using CUDA_VISIBLE_DEVICES first entry.
  local phys_idx
  phys_idx="${CUDA_VISIBLE_DEVICES%%,*}"
  # Strip any MIG uuid form — if non-numeric, fall back to device 0 fields.
  if ! [[ "${phys_idx}" =~ ^[0-9]+$ ]]; then
    phys_idx=0
  fi

  local name
  name="$(nvidia-smi --id="${phys_idx}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  if [[ -z "${name}" ]]; then
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n "$((phys_idx + 1))p" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  fi
  [[ -n "${name}" ]] || die "Unable to query GPU name for index ${phys_idx}"

  # name_regex of "." or ".*" matches anything (portable / unknown product strings).
  if [[ "${name_regex}" != "." && "${name_regex}" != ".*" ]]; then
    if ! [[ "${name}" =~ ${name_regex} ]]; then
      die "GPU name '${name}' does not match expected pattern /${name_regex}/ for ${label}"
    fi
  fi

  # MIG: refuse MIG instances for these benches (need full device).
  local mig_mode
  mig_mode="$(nvidia-smi --id="${phys_idx}" --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ "${mig_mode}" == "Enabled" ]]; then
    die "GPU is in MIG mode (current=${mig_mode}); full-device allocation required for ${label}"
  fi
  if echo "${name}" | grep -qi 'MIG'; then
    die "Selected device looks like a MIG instance ('${name}'); full-device required for ${label}"
  fi

  local mem_total
  mem_total="$(nvidia-smi --id="${phys_idx}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -z "${mem_total}" ]]; then
    mem_total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n "$((phys_idx + 1))p" | tr -d ' ')"
  fi
  [[ -n "${mem_total}" ]] || die "Unable to query GPU memory"
  if (( mem_total < min_mem_mib )); then
    die "GPU memory ${mem_total} MiB is below minimum ${min_mem_mib} MiB for ${label}"
  fi

  local cc
  cc="$(nvidia-smi --id="${phys_idx}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ -z "${cc}" || "${cc}" == "[N/A]" ]]; then
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n "$((phys_idx + 1))p" | tr -d ' ' || true)"
  fi
  if [[ -n "${cc}" && "${cc}" != "[N/A]" ]]; then
    if [[ "${expect_cc}" != "0.0" && "${expect_cc}" != "any" && "${cc}" != "${expect_cc}" ]]; then
      die "Compute capability ${cc} != expected ${expect_cc} for ${label}"
    fi
  else
    log "WARN: compute_cap not reported by nvidia-smi; will re-check via torch if available"
  fi

  export COLIDE_GPU_LABEL="${label}"
  export COLIDE_GPU_NAME="${name}"
  export COLIDE_GPU_CC="${cc:-unknown}"
  export COLIDE_GPU_MEM_MIB="${mem_total}"
  export COLIDE_GPU_NODE_COUNT="${gpu_count}"
  log "GPU OK: label=${label} name='${name}' cc=${COLIDE_GPU_CC} mem=${mem_total}MiB visible_on_node=${gpu_count} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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
