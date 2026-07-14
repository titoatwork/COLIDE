#!/usr/bin/env bash
# =============================================================================
# COLIDE cluster setup — portable across any login/build *or* GPU compute node.
#
# Defaults to the repository that contains this script (no hardcoded site paths).
# On sites where the login node has no CUDA toolkit (e.g. Rostam), run this
# *inside* a GPU job / interactive srun where nvcc exists.
#
# Usage:
#   cd /path/to/COLIDE
#   bash dicc_scripts/01_setup.sh
#   bash dicc_scripts/01_setup.sh --targets sm_70:v100
#   bash dicc_scripts/01_setup.sh --kernels-only --targets sm_70:v100
#   bash dicc_scripts/01_setup.sh --python-only
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

if [[ -f "${SCRIPT_DIR}/site.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/site.env"
fi

REPO_URL="${COLIDE_REPO_URL:-https://github.com/titoatwork/COLIDE.git}"
DO_PULL=0
CLONE_TO=""
COMPILE_TARGETS="${COLIDE_COMPILE_TARGETS:-sm_70:v100,sm_80:a100}"
ALLOW_DIRTY="${COLIDE_SETUP_ALLOW_DIRTY:-0}"
KERNELS_ONLY=0
PYTHON_ONLY=0

KERNELS=(
  fused_block1
  fused_block2
  fused_block3
  fused_block3_fp16
  fused_block3_naive
  fused_block4
  fused_pipeline
)

usage() {
  cat <<'EOF'
Usage: 01_setup.sh [options]

  --pull              git pull --ff-only
  --clone-to DIR      clone REPO_URL into DIR if missing
  --targets LIST      arch:subdir pairs (default: sm_70:v100,sm_80:a100)
  --kernels-only      only compile CUDA kernels (skip Python env)
  --python-only       only create Python env (skip nvcc/kernels)
  --allow-dirty       do not abort if working tree is dirty
  -h, --help

If nvcc is missing on the login node, run this on a GPU node, e.g.:
  srun -p cuda-V100 --gres=gpu:1 -t 01:00:00 --pty bash -l
  cd $COLIDE_ROOT && bash dicc_scripts/01_setup.sh --targets sm_70:v100
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull) DO_PULL=1; shift ;;
    --clone-to) CLONE_TO="${2:?}"; shift 2 ;;
    --targets) COMPILE_TARGETS="${2:?}"; shift 2 ;;
    --kernels-only) KERNELS_ONLY=1; shift ;;
    --python-only) PYTHON_ONLY=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

if [[ -n "${CLONE_TO}" ]]; then
  if [[ -d "${CLONE_TO}/.git" ]]; then
    COLIDE_ROOT="$(cd "${CLONE_TO}" && pwd)"
  else
    parent="$(dirname "${CLONE_TO}")"
    mkdir -p "${parent}" || die "Cannot create parent directory for clone: ${parent}"
    log "Cloning ${REPO_URL} -> ${CLONE_TO}"
    git clone "${REPO_URL}" "${CLONE_TO}"
    COLIDE_ROOT="$(cd "${CLONE_TO}" && pwd)"
  fi
elif [[ -n "${COLIDE_ROOT:-}" ]]; then
  COLIDE_ROOT="$(cd "${COLIDE_ROOT}" && pwd)"
else
  COLIDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
export COLIDE_ROOT

if [[ ! -d "${COLIDE_ROOT}/dicc_scripts" || ! -d "${COLIDE_ROOT}/inference/kernels" ]]; then
  die "COLIDE_ROOT=${COLIDE_ROOT} does not look like the COLIDE repo"
fi

cd "${COLIDE_ROOT}"
log "=== COLIDE CLUSTER SETUP (portable) ==="
log "COLIDE_ROOT=${COLIDE_ROOT}"
log "host=$(hostname) kernels_only=${KERNELS_ONLY} python_only=${PYTHON_ONLY}"

if [[ "${DO_PULL}" == "1" ]]; then
  if git status --porcelain | grep -q . && [[ "${ALLOW_DIRTY}" != "1" ]]; then
    die "Working tree dirty. Commit/stash, or pass --allow-dirty."
  fi
  git pull --ff-only || die "git pull --ff-only failed"
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
log "git_sha=${GIT_SHA}"
if git status --porcelain 2>/dev/null | grep -q .; then
  if [[ "${ALLOW_DIRTY}" == "1" ]]; then
    log "WARN: working tree is dirty (allowed)"
  else
    log "WARN: working tree is dirty (setup continues; clean before *submit* for provenance)"
  fi
fi

mkdir -p "${COLIDE_ROOT}/benchmarks/results/dicc" "${COLIDE_ROOT}/inference/kernels"

# ---------------------------------------------------------------------------
# Python env (conda if available, else venv + module python)
# ---------------------------------------------------------------------------
setup_python() {
  log "=== Python environment ==="
  if ! command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC2206
    conda_mods=(${COLIDE_CONDA_MODULES:-miniconda anaconda conda})
    for m in "${conda_mods[@]}"; do
      module load "${m}" 2>/dev/null && break || true
    done
  fi

  if command -v conda >/dev/null 2>&1; then
    init_conda
    if conda env list | awk '{print $1}' | grep -qx 'colide'; then
      log "conda env 'colide' already exists"
    else
      log "Creating conda env 'colide' (python=3.12)"
      conda create -n colide python=3.12 -y
    fi
    conda activate colide || die "conda activate colide failed"
  else
    log "conda not found — using venv at .venv-cluster"
    if ! command -v python3 >/dev/null 2>&1; then
      module load python/3.12.3 2>/dev/null || module load python 2>/dev/null || true
    fi
    command -v python3 >/dev/null 2>&1 || die "python3 not found. module load python/… first."
    local venv_dir="${COLIDE_VENV:-${COLIDE_ROOT}/.venv-cluster}"
    if [[ ! -x "${venv_dir}/bin/python" ]]; then
      python3 -m venv "${venv_dir}"
    fi
    # shellcheck source=/dev/null
    source "${venv_dir}/bin/activate"
    log "venv: ${venv_dir}"
  fi

  log "Installing minimal benchmark dependencies"
  python -m pip install --upgrade pip
  # shellcheck disable=SC2086
  python -m pip install \
    ${COLIDE_PIP_EXTRA:-} \
    'numpy>=2.0.0' \
    'scipy>=1.13.0' \
    'pyyaml>=6.0' \
    'scikit-learn>=1.5.0'
  # Default to cu121 wheels: support V100 (SM 7.0) + A100. PyPI's newest torch
  # (e.g. 2.13+cu130) ships cuDNN that refuses SM < 7.5 and breaks V100.
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  log "Installing torch from ${TORCH_INDEX_URL}"
  python -m pip install --upgrade 'torch>=2.5.0,<2.7' --index-url "${TORCH_INDEX_URL}"

  python - <<'PY'
import numpy, scipy, yaml, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
PY
}

# ---------------------------------------------------------------------------
# CUDA kernels
# ---------------------------------------------------------------------------
setup_kernels() {
  log "=== CUDA kernel compile ==="
  if ! command -v nvcc >/dev/null 2>&1; then
    # shellcheck disable=SC2206
    cuda_mods=(${COLIDE_CUDA_MODULES:-cuda/12.1 cuda cuda/12.0 cuda/11.8 cudatoolkit})
    for m in "${cuda_mods[@]}"; do
      module load "${m}" 2>/dev/null && break || true
    done
  fi

  local nvcc_bin=""
  if nvcc_bin="$(find_nvcc)"; then
    export PATH="$(dirname "${nvcc_bin}"):${PATH}"
    log "nvcc: ${nvcc_bin}"
  else
    log "HINT: Rostam login nodes often have no CUDA toolkit."
    log "HINT: compile on a GPU node, e.g.:"
    log "  srun -p cuda-V100 -N 1 -t 01:00:00 --gres=gpu:1 --pty bash -l"
    log "  cd ${COLIDE_ROOT} && bash dicc_scripts/01_setup.sh --kernels-only --targets sm_70:v100"
    log "HINT: or:  find /usr/local /opt -name nvcc 2>/dev/null | head"
    die "nvcc not found on $(hostname). Run setup on a node that has the CUDA toolkit."
  fi

  nvcc --version | tee "${COLIDE_ROOT}/benchmarks/results/dicc/setup_nvcc_version.txt" || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi:"
    nvidia-smi -L || true
  fi

  compile_arch() {
    local arch="$1"
    local subdir="$2"
    local out_final="${COLIDE_ROOT}/inference/kernels/${subdir}"
    local tmp
    tmp="$(mktemp -d "${COLIDE_ROOT}/inference/kernels/.build_${subdir}.XXXXXX")"

    log "=== Compiling ${subdir} (${arch}) into ${tmp} ==="
    local name
    for name in "${KERNELS[@]}"; do
      log "nvcc -arch=${arch} -o ${tmp}/${name} ${name}.cu"
      nvcc -arch="${arch}" \
        -o "${tmp}/${name}" \
        "${COLIDE_ROOT}/inference/kernels/${name}.cu"
      [[ -x "${tmp}/${name}" ]] || die "compile produced non-executable ${tmp}/${name}"
    done

    (
      cd "${tmp}"
      sha256sum "${KERNELS[@]}" > SHA256SUMS
    )
    log "Checksums for ${subdir}:"
    cat "${tmp}/SHA256SUMS"

    local backup=""
    if [[ -d "${out_final}" ]]; then
      backup="${out_final}.prev.$$"
      mv "${out_final}" "${backup}"
    fi
    mv "${tmp}" "${out_final}"
    if [[ -n "${backup}" ]]; then
      rm -rf "${backup}"
    fi
    log "Installed kernels -> ${out_final}"
  }

  IFS=',' read -r -a _targets <<< "${COMPILE_TARGETS}"
  declare -a compiled_paths=()
  for pair in "${_targets[@]}"; do
    pair="$(echo "${pair}" | tr -d ' ')"
    [[ -n "${pair}" ]] || continue
    local arch="${pair%%:*}"
    local subdir="${pair#*:}"
    if [[ -z "${arch}" || -z "${subdir}" || "${arch}" == "${subdir}" ]]; then
      die "Bad --targets entry '${pair}' (expected arch:subdir, e.g. sm_70:v100)"
    fi
    compile_arch "${arch}" "${subdir}"
    compiled_paths+=("\"${subdir}\": \"${COLIDE_ROOT}/inference/kernels/${subdir}/SHA256SUMS\"")
  done

  local kernels_json
  kernels_json=$(printf '%s,' "${compiled_paths[@]}")
  kernels_json="{ ${kernels_json%,} }"

  local stamp="${COLIDE_ROOT}/benchmarks/results/dicc/setup_kernels_$(date -u +%Y%m%dT%H%M%SZ).json"
  cat > "${stamp}" <<EOF
{
  "git_sha": "${GIT_SHA}",
  "colide_root": "${COLIDE_ROOT}",
  "host": "$(hostname)",
  "nvcc": "$(command -v nvcc)",
  "compile_targets": "${COMPILE_TARGETS}",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "kernels": ${kernels_json}
}
EOF
  log "Kernel setup stamp: ${stamp}"
}

if [[ "${PYTHON_ONLY}" == "1" ]]; then
  setup_python
elif [[ "${KERNELS_ONLY}" == "1" ]]; then
  setup_kernels
else
  setup_python
  setup_kernels
fi

log "=== Setup Complete ==="
log "  export COLIDE_ROOT=${COLIDE_ROOT}"
log "Submit example (Rostam):"
log "  bash dicc_scripts/submit_session.sh --targets v100 --partition cuda-V100 --gres gpu:1 --allow-dirty"
log "  bash dicc_scripts/submit_session.sh --targets a100 --partition cuda-A100 --gres gpu:1 --allow-dirty"
