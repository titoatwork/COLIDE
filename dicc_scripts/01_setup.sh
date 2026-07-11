#!/usr/bin/env bash
# =============================================================================
# COLIDE cluster setup — portable across any login/build node.
#
# Defaults to the repository that contains this script (no hardcoded /scr paths).
# Does NOT require a GPU or nvidia-smi.
#
# Usage:
#   cd /path/to/COLIDE
#   bash dicc_scripts/01_setup.sh
#   bash dicc_scripts/01_setup.sh --pull
#   bash dicc_scripts/01_setup.sh --targets sm_70:v100
#   bash dicc_scripts/01_setup.sh --targets sm_70:v100,sm_80:a100
#   bash dicc_scripts/01_setup.sh --clone-to /some/writable/path   # optional
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Optional site knobs (partition is unused here; modules/targets may be set).
if [[ -f "${SCRIPT_DIR}/site.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/site.env"
fi

REPO_URL="${COLIDE_REPO_URL:-https://github.com/titoatwork/COLIDE.git}"
DO_PULL=0
CLONE_TO=""
# arch:subdir pairs — override for single-GPU clusters
COMPILE_TARGETS="${COLIDE_COMPILE_TARGETS:-sm_70:v100,sm_80:a100}"
ALLOW_DIRTY="${COLIDE_SETUP_ALLOW_DIRTY:-0}"

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

  --pull              git pull --ff-only in the resolved repo (default: no network pull)
  --clone-to DIR      clone REPO_URL into DIR if missing, then use that as COLIDE_ROOT
  --targets LIST      comma-separated arch:subdir pairs (default: sm_70:v100,sm_80:a100)
  --allow-dirty       do not abort if the working tree is dirty
  -h, --help          show help

Environment:
  COLIDE_ROOT            absolute repo path (default: parent of dicc_scripts/)
  COLIDE_COMPILE_TARGETS same as --targets
  COLIDE_CUDA_MODULES    space-separated module names to try for nvcc
  COLIDE_CONDA_MODULES   space-separated module names to try for conda
  COLIDE_REPO_URL        clone URL when using --clone-to
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull) DO_PULL=1; shift ;;
    --clone-to) CLONE_TO="${2:?}"; shift 2 ;;
    --targets) COMPILE_TARGETS="${2:?}"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ---------------------------------------------------------------------------
# Resolve repository root — never assume /scr or any site path
# ---------------------------------------------------------------------------
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
  # This script lives at <repo>/dicc_scripts/01_setup.sh
  COLIDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
export COLIDE_ROOT

if [[ ! -d "${COLIDE_ROOT}/dicc_scripts" || ! -d "${COLIDE_ROOT}/inference/kernels" ]]; then
  die "COLIDE_ROOT=${COLIDE_ROOT} does not look like the COLIDE repo"
fi

cd "${COLIDE_ROOT}"
log "=== COLIDE CLUSTER SETUP (portable) ==="
log "COLIDE_ROOT=${COLIDE_ROOT}"

if [[ "${DO_PULL}" == "1" ]]; then
  if git status --porcelain | grep -q .; then
    if [[ "${ALLOW_DIRTY}" == "1" ]]; then
      log "WARN: dirty tree; pulling may fail"
    else
      die "Working tree dirty under ${COLIDE_ROOT}. Commit/stash, or pass --allow-dirty / --pull carefully."
    fi
  fi
  git pull --ff-only || die "git pull --ff-only failed"
fi

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
log "git_sha=${GIT_SHA}"
if git status --porcelain 2>/dev/null | grep -q .; then
  if [[ "${ALLOW_DIRTY}" == "1" ]]; then
    log "WARN: working tree is dirty (allowed)"
  else
    die "Working tree is dirty under ${COLIDE_ROOT}. Commit/stash for tight provenance, or pass --allow-dirty."
  fi
fi

# ---------------------------------------------------------------------------
# Toolchain (CUDA compiler only — no nvidia-smi required)
# ---------------------------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
  # shellcheck disable=SC2206
  cuda_mods=(${COLIDE_CUDA_MODULES:-cuda/12.1 cuda cuda/12.0 cuda/11.8})
  for m in "${cuda_mods[@]}"; do
    module load "${m}" 2>/dev/null && break || true
  done
fi
command -v nvcc >/dev/null 2>&1 || die "nvcc not found. Load a CUDA toolkit module (module avail | grep -i cuda)."
log "nvcc: $(command -v nvcc)"
mkdir -p "${COLIDE_ROOT}/benchmarks/results/dicc"
nvcc --version | tee "${COLIDE_ROOT}/benchmarks/results/dicc/setup_nvcc_version.txt" >/dev/null || true
nvcc --version || true

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
  # shellcheck disable=SC2206
  conda_mods=(${COLIDE_CONDA_MODULES:-miniconda anaconda conda})
  for m in "${conda_mods[@]}"; do
    module load "${m}" 2>/dev/null && break || true
  done
fi
init_conda

if conda env list | awk '{print $1}' | grep -qx 'colide'; then
  log "conda env 'colide' already exists"
else
  log "Creating conda env 'colide' (python=3.12)"
  conda create -n colide python=3.12 -y
fi
conda activate colide || die "conda activate colide failed after create/list"

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
# Atomic compile per arch:subdir target
# ---------------------------------------------------------------------------
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

mkdir -p "${COLIDE_ROOT}/inference/kernels" "${COLIDE_ROOT}/benchmarks/results/dicc"

IFS=',' read -r -a _targets <<< "${COMPILE_TARGETS}"
declare -a compiled_paths=()
for pair in "${_targets[@]}"; do
  pair="$(echo "${pair}" | tr -d ' ')"
  [[ -n "${pair}" ]] || continue
  arch="${pair%%:*}"
  subdir="${pair#*:}"
  if [[ -z "${arch}" || -z "${subdir}" || "${arch}" == "${subdir}" ]]; then
    die "Bad --targets entry '${pair}' (expected arch:subdir, e.g. sm_70:v100)"
  fi
  compile_arch "${arch}" "${subdir}"
  compiled_paths+=("\"${subdir}\": \"${COLIDE_ROOT}/inference/kernels/${subdir}/SHA256SUMS\"")
done

kernels_json=$(printf '%s,' "${compiled_paths[@]}")
kernels_json="{ ${kernels_json%,} }"

SETUP_STAMP="${COLIDE_ROOT}/benchmarks/results/dicc/setup_$(date -u +%Y%m%dT%H%M%SZ).json"
cat > "${SETUP_STAMP}" <<EOF
{
  "git_sha": "${GIT_SHA}",
  "colide_root": "${COLIDE_ROOT}",
  "nvcc": "$(command -v nvcc)",
  "compile_targets": "${COMPILE_TARGETS}",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "kernels": ${kernels_json}
}
EOF

log "=== Setup Complete ==="
log "COLIDE_ROOT is already set to this tree:"
log "  export COLIDE_ROOT=${COLIDE_ROOT}"
log "Submit (portable; set site env as needed):"
log "  bash ${COLIDE_ROOT}/dicc_scripts/submit_session.sh --targets v100,a100"
log "  # or single GPU:  --targets v100"
log "Setup stamp: ${SETUP_STAMP}"
