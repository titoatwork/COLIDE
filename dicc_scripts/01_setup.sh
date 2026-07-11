#!/usr/bin/env bash
# =============================================================================
# COLIDE DICC Setup — run once (or re-run to recompile) on a login/build node.
# Does NOT require a GPU or nvidia-smi (login nodes are often GPU-less).
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

SCRATCH_ROOT="${COLIDE_SCRATCH_ROOT:-/scr/${USER}}"
REPO_URL="${COLIDE_REPO_URL:-https://github.com/titoatwork/COLIDE.git}"
REPO_DIR="${COLIDE_ROOT:-${SCRATCH_ROOT}/colide}"
KERNELS=(
  fused_block1
  fused_block2
  fused_block3
  fused_block3_fp16
  fused_block3_naive
  fused_block4
  fused_pipeline
)

log "=== COLIDE DICC SETUP ==="
log "REPO_DIR=${REPO_DIR}"

mkdir -p "${SCRATCH_ROOT}"

# ---------------------------------------------------------------------------
# Clone or fast-forward pull
# ---------------------------------------------------------------------------
if [[ -d "${REPO_DIR}/.git" ]]; then
  log "Repository exists; fast-forward pull only"
  cd "${REPO_DIR}"
  if git status --porcelain | grep -q .; then
    die "Working tree is dirty under ${REPO_DIR}. Commit/stash before setup so SHA provenance is meaningful."
  fi
  git pull --ff-only || die "git pull --ff-only failed (refusing merge commits on cluster checkout)"
else
  log "Cloning ${REPO_URL} -> ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
  cd "${REPO_DIR}"
fi

COLIDE_ROOT="$(cd "${REPO_DIR}" && pwd)"
export COLIDE_ROOT
GIT_SHA="$(git rev-parse HEAD)"
log "COLIDE_ROOT=${COLIDE_ROOT}"
log "git_sha=${GIT_SHA}"
if git status --porcelain | grep -q .; then
  die "Working tree became dirty after pull; aborting"
fi

# ---------------------------------------------------------------------------
# Toolchain (CUDA compiler only — no nvidia-smi on login nodes)
# ---------------------------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
  module load cuda/12.1 2>/dev/null \
    || module load cuda 2>/dev/null \
    || true
fi
command -v nvcc >/dev/null 2>&1 || die "nvcc not found. Run 'module avail' and load a CUDA toolkit module."
log "nvcc: $(command -v nvcc)"
nvcc --version | tee "${COLIDE_ROOT}/benchmarks/results/dicc/setup_nvcc_version.txt" >/dev/null \
  || nvcc --version

# Optional: record GPU presence if available, but never require it.
if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi present (optional on login node):"
  nvidia-smi -L || true
else
  log "nvidia-smi not available (OK for GPU-less login/build node)"
fi

# ---------------------------------------------------------------------------
# Conda env + minimal benchmark dependencies
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  module load miniconda 2>/dev/null || module load anaconda 2>/dev/null || true
fi
init_conda

if conda env list | awk '{print $1}' | grep -qx 'colide'; then
  log "conda env 'colide' already exists"
else
  log "Creating conda env 'colide' (python=3.12)"
  conda create -n colide python=3.12 -y
fi
conda activate colide || die "conda activate colide failed after create/list"

# Minimal set for DICC core benchmarks (CUDA stats + PyTorch GPU harness).
# Avoid pulling the full requirements.txt (LLM stack, etc.) on the cluster.
log "Installing minimal benchmark dependencies"
python -m pip install --upgrade pip
python -m pip install \
  'torch>=2.5.0' \
  'numpy>=2.0.0' \
  'scipy>=1.13.0' \
  'pyyaml>=6.0' \
  'scikit-learn>=1.5.0'

python - <<'PY'
import numpy, scipy, yaml, torch
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
PY

# ---------------------------------------------------------------------------
# Atomic compile helpers
# ---------------------------------------------------------------------------
# Compile into a temp directory, verify all binaries exist and are executable,
# write SHA256SUMS, then replace the target directory atomically via rename.
compile_arch() {
  local arch="$1"   # sm_70 / sm_80
  local subdir="$2" # v100 / a100
  local out_final="${COLIDE_ROOT}/inference/kernels/${subdir}"
  local tmp
  tmp="$(mktemp -d "${COLIDE_ROOT}/inference/kernels/.build_${subdir}.XXXXXX")"
  # shellcheck disable=SC2064
  trap "rm -rf '${tmp}'" RETURN

  log "=== Compiling ${subdir} (${arch}) into ${tmp} ==="
  # NOTE: no extra -O flags — must match Dockerfile/build command exactly.
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

  # Atomic replace: move old aside, move new into place, remove old.
  local backup=""
  if [[ -d "${out_final}" ]]; then
    backup="${out_final}.prev.$$"
    mv "${out_final}" "${backup}"
  fi
  mv "${tmp}" "${out_final}"
  # Disarm RETURN trap cleanup for tmp (already moved)
  trap - RETURN
  if [[ -n "${backup}" ]]; then
    rm -rf "${backup}"
  fi
  log "Installed kernels -> ${out_final}"
}

mkdir -p "${COLIDE_ROOT}/inference/kernels" "${COLIDE_ROOT}/benchmarks/results/dicc"

compile_arch sm_70 v100
compile_arch sm_80 a100

# Persist setup provenance
SETUP_STAMP="${COLIDE_ROOT}/benchmarks/results/dicc/setup_$(date -u +%Y%m%dT%H%M%SZ).json"
cat > "${SETUP_STAMP}" <<EOF
{
  "git_sha": "${GIT_SHA}",
  "colide_root": "${COLIDE_ROOT}",
  "nvcc": "$(command -v nvcc)",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "kernels": {
    "v100": "${COLIDE_ROOT}/inference/kernels/v100/SHA256SUMS",
    "a100": "${COLIDE_ROOT}/inference/kernels/a100/SHA256SUMS"
  }
}
EOF

log "=== Setup Complete ==="
log "Export before submitting jobs:"
log "  export COLIDE_ROOT=${COLIDE_ROOT}"
log "Then:"
log "  bash ${COLIDE_ROOT}/dicc_scripts/submit_session.sh"
log "Setup stamp: ${SETUP_STAMP}"
