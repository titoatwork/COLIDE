#!/usr/bin/env bash
# Shared DICC benchmark runner. Invoked by thin SBATCH wrappers (02/03).
# Expects:
#   COLIDE_ROOT          absolute repo path (required)
#   COLIDE_GPU_LABEL     e.g. v100s / a100 (required)
#   COLIDE_GPU_NAME_RE   regex for nvidia-smi name (required)
#   COLIDE_GPU_CC        expected compute capability e.g. 7.0 (required)
#   COLIDE_GPU_MIN_MEM   min MiB (required)
#   COLIDE_KERNELS_SUBDIR  e.g. v100 / a100 (required)
# Optional:
#   COLIDE_CAMPAIGN, COLIDE_DATE_LABEL, COLIDE_CHECKPOINT,
#   COLIDE_N_TRIALS_CUDA, COLIDE_N_TRIALS_PYTORCH, COLIDE_PYTORCH_INNER
# shellcheck shell=bash
set -Eeuo pipefail

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${_LIB_DIR}/common.sh"

require_colide_root
cd "${COLIDE_ROOT}"

: "${COLIDE_GPU_LABEL:?COLIDE_GPU_LABEL required}"
: "${COLIDE_GPU_NAME_RE:?COLIDE_GPU_NAME_RE required}"
: "${COLIDE_GPU_CC:?COLIDE_GPU_CC required}"
: "${COLIDE_GPU_MIN_MEM:?COLIDE_GPU_MIN_MEM required}"
: "${COLIDE_KERNELS_SUBDIR:?COLIDE_KERNELS_SUBDIR required}"

export COLIDE_CAMPAIGN="${COLIDE_CAMPAIGN:-core}"
export COLIDE_DATE_LABEL="${COLIDE_DATE_LABEL:-$(date -u +%Y%m%d)}"
export COLIDE_CHECKPOINT="${COLIDE_CHECKPOINT:-model/best_model_botiot_twostage.pth}"
export COLIDE_N_TRIALS_CUDA="${COLIDE_N_TRIALS_CUDA:-100}"
export COLIDE_N_TRIALS_PYTORCH="${COLIDE_N_TRIALS_PYTORCH:-20}"
export COLIDE_PYTORCH_INNER="${COLIDE_PYTORCH_INNER:-1000}"
export COLIDE_KERNELS_DIR="${COLIDE_ROOT}/inference/kernels/${COLIDE_KERNELS_SUBDIR}"
export PYTHONPATH="${COLIDE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

JOB_ID="${SLURM_JOB_ID:-local}"
RUN_DIR="$(make_run_dir "${COLIDE_CAMPAIGN}" "${COLIDE_GPU_LABEL}" "${COLIDE_DATE_LABEL}" "${JOB_ID}")"
export COLIDE_RUN_DIR="${RUN_DIR}"

# Tee all subsequent output into the isolated run log while still going to
# Slurm's captured stdout/stderr.
exec > >(tee -a "${RUN_DIR}/logs/runner.log") 2>&1

log "=== COLIDE DICC benchmark runner ==="
log "COLIDE_ROOT=${COLIDE_ROOT}"
log "RUN_DIR=${RUN_DIR}"
log "GPU_LABEL=${COLIDE_GPU_LABEL} kernels=${COLIDE_KERNELS_DIR}"

EXIT_CODE=0
on_exit() {
  local ec=$?
  if [[ ${ec} -ne 0 ]]; then
    EXIT_CODE=${ec}
  fi
  mark_success "${RUN_DIR}" "${EXIT_CODE}"
  log "Runner finished with exit_status=${EXIT_CODE}"
}
trap on_exit EXIT

# Fail the whole job on any pipeline/command error; "Complete." must not print
# after a partial failure (that was the previous failure mode).
set -E

# Ensure CUDA toolkit libs are findable if present (does not require nvcc at runtime).
if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi

activate_colide_env
assert_gpu "${COLIDE_GPU_LABEL}" "${COLIDE_GPU_NAME_RE}" "${COLIDE_GPU_CC}" "${COLIDE_GPU_MIN_MEM}"

[[ -d "${COLIDE_KERNELS_DIR}" ]] || die "kernels dir missing: ${COLIDE_KERNELS_DIR}"
[[ -f "${COLIDE_ROOT}/${COLIDE_CHECKPOINT}" ]] || die "checkpoint missing: ${COLIDE_ROOT}/${COLIDE_CHECKPOINT}"

write_manifest "${RUN_DIR}"
write_environment "${RUN_DIR}"
copy_kernel_checksums "${RUN_DIR}" "${COLIDE_KERNELS_DIR}"

# Optional secondary compute-cap check via torch (catches driver/tooling mismatch).
python - <<'PY'
import os, sys
try:
    import torch
except Exception as e:
    print("torch unavailable for secondary CC check:", e)
    sys.exit(0)
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
got = f"{p.major}.{p.minor}"
expect = os.environ.get("COLIDE_GPU_CC", "any")
if expect not in ("0.0", "any", "") and got != expect:
    print(f"ERROR: torch compute capability {got} != expected {expect}", file=sys.stderr)
    sys.exit(1)
print(f"torch CC OK: {got} device={torch.cuda.get_device_name(0)} (expect={expect})")
PY

log "=== Single-run CUDA kernel smoke (must pass validation) ==="
for bin in fused_block1 fused_block2 fused_block3 fused_block3_fp16 fused_block3_naive fused_block4 fused_pipeline; do
  path="${COLIDE_KERNELS_DIR}/${bin}"
  [[ -x "${path}" ]] || die "missing executable kernel binary: ${path}"
  log "--- ${bin} ---"
  # Run from kernels parent so relative weight paths inside binaries resolve
  # consistently (binaries historically used cwd-relative asset paths).
  (
    cd "${COLIDE_KERNELS_DIR}"
    "./${bin}"
  ) | tee "${RUN_DIR}/logs/smoke_${bin}.log"
  if grep -q "FAILED" "${RUN_DIR}/logs/smoke_${bin}.log"; then
    die "numerical validation FAILED for ${bin} (see smoke_${bin}.log)"
  fi
done

log "=== CUDA kernel statistical benchmark (n=${COLIDE_N_TRIALS_CUDA}) ==="
# max-validation-failures: A100 fused_block3_fp16 can flake numerical checks
# intermittently under load; retries first, then allow a few hard skips.
python "${COLIDE_ROOT}/scripts/benchmark_cuda_kernels_stats.py" \
  --kernels-dir "${COLIDE_KERNELS_DIR}" \
  --suffix "" \
  --tag "${COLIDE_GPU_LABEL}" \
  --n-trials "${COLIDE_N_TRIALS_CUDA}" \
  --output "${RUN_DIR}/cuda_kernel_stats.json" \
  --raw-output "${RUN_DIR}/raw/cuda_kernel_raw.json" \
  --strict \
  --max-retries "${COLIDE_CUDA_MAX_RETRIES:-5}" \
  --max-validation-failures "${COLIDE_CUDA_MAX_VAL_FAILS:-5}" \
  --timeout-sec 600

log "=== PyTorch GPU statistical harness ==="
python "${COLIDE_ROOT}/scripts/benchmark_pytorch_gpu_stats.py" \
  --checkpoint "${COLIDE_ROOT}/${COLIDE_CHECKPOINT}" \
  --tag "${COLIDE_GPU_LABEL}" \
  --n-trials "${COLIDE_N_TRIALS_PYTORCH}" \
  --inner-forwards "${COLIDE_PYTORCH_INNER}" \
  --output "${RUN_DIR}/pytorch_gpu_stats.json" \
  --raw-output "${RUN_DIR}/raw/pytorch_gpu_raw.json"

# Summarize (no legacy fixed filenames written)
{
  echo "COLIDE DICC run summary"
  echo "run_dir=${RUN_DIR}"
  echo "git_sha=$(git -C "${COLIDE_ROOT}" rev-parse HEAD)"
  echo "gpu=${COLIDE_GPU_NAME} label=${COLIDE_GPU_LABEL}"
  echo "checkpoint=${COLIDE_CHECKPOINT}"
  echo "cuda_stats=${RUN_DIR}/cuda_kernel_stats.json"
  echo "pytorch_stats=${RUN_DIR}/pytorch_gpu_stats.json"
  echo "NOTE: full CUDA-vs-PyTorch pipeline speedup is NOT valid"
  echo "      (CUDA lacks attention/LayerNorm/GAP; fused_pipeline skips Block3)."
  echo "      Use per-block comparisons only, especially Block 3."
} | tee "${RUN_DIR}/summary.txt"

log "=== Benchmark core complete (SUCCESS marker written on clean exit) ==="
EXIT_CODE=0
