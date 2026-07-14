#!/usr/bin/env bash
# Local validation suite for DICC hardening (no cluster / no GPU required).
# Run from anywhere:
#   bash dicc_scripts/validate/local_validate.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DICC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DICC_DIR}/.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/colide_dicc_validate.XXXXXX")"
# shellcheck disable=SC2064
trap "rm -rf '${TMP_ROOT}'" EXIT

pass=0
fail=0

ok() { echo "  PASS: $*"; pass=$((pass + 1)); }
bad() { echo "  FAIL: $*"; fail=$((fail + 1)); }

# Prefer project venv, then a reusable cache venv, else a throwaway venv.
PYTHON="python3"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]] \
    && "${REPO_ROOT}/.venv/bin/python" -c "import numpy" 2>/dev/null; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
elif [[ -x /tmp/colide_dicc_validate_py/bin/python ]] \
    && /tmp/colide_dicc_validate_py/bin/python -c "import numpy" 2>/dev/null; then
  PYTHON="/tmp/colide_dicc_validate_py/bin/python"
elif python3 -c "import numpy" 2>/dev/null; then
  PYTHON="python3"
else
  python3 -m venv "${TMP_ROOT}/pyenv"
  # shellcheck disable=SC1091
  source "${TMP_ROOT}/pyenv/bin/activate"
  pip install -q numpy scipy
  PYTHON="${TMP_ROOT}/pyenv/bin/python"
fi

echo "=== COLIDE DICC local validation ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "TMP_ROOT=${TMP_ROOT}"
echo "PYTHON=${PYTHON}"

# ---------------------------------------------------------------------------
# 1. bash -n
# ---------------------------------------------------------------------------
echo ""
echo "--- bash -n ---"
SHELL_SCRIPTS=(
  "${DICC_DIR}/01_setup.sh"
  "${DICC_DIR}/02_benchmark_v100.sh"
  "${DICC_DIR}/03_benchmark_a100.sh"
  "${DICC_DIR}/04_nsight_profile.sh"
  "${DICC_DIR}/05_run_all.sh"
  "${DICC_DIR}/submit_session.sh"
  "${DICC_DIR}/lib/common.sh"
  "${DICC_DIR}/lib/run_benchmark.sh"
  "${DICC_DIR}/validate/local_validate.sh"
)
for s in "${SHELL_SCRIPTS[@]}"; do
  if bash -n "${s}"; then
    ok "bash -n $(basename "${s}")"
  else
    bad "bash -n $(basename "${s}")"
  fi
done

# ---------------------------------------------------------------------------
# 2. shellcheck (zero warnings required)
# ---------------------------------------------------------------------------
echo ""
echo "--- shellcheck ---"
if command -v shellcheck >/dev/null 2>&1; then
  # SC1091: dynamic SCRIPT_DIR sources; paths are validated by bash -n + runtime.
  if shellcheck -x \
      -e SC1091 \
      --source-path="${DICC_DIR}:${DICC_DIR}/lib" \
      "${DICC_DIR}/01_setup.sh" \
      "${DICC_DIR}/02_benchmark_v100.sh" \
      "${DICC_DIR}/03_benchmark_a100.sh" \
      "${DICC_DIR}/04_nsight_profile.sh" \
      "${DICC_DIR}/05_run_all.sh" \
      "${DICC_DIR}/submit_session.sh" \
      "${DICC_DIR}/lib/common.sh" \
      "${DICC_DIR}/lib/run_benchmark.sh" \
      "${DICC_DIR}/validate/local_validate.sh"; then
    ok "shellcheck zero warnings"
  else
    bad "shellcheck reported issues"
  fi
else
  bad "shellcheck not installed"
fi

# ---------------------------------------------------------------------------
# 3. Python compile
# ---------------------------------------------------------------------------
echo ""
echo "--- python -m py_compile ---"
PY_SCRIPTS=(
  "${REPO_ROOT}/scripts/benchmark_cuda_kernels_stats.py"
  "${REPO_ROOT}/scripts/benchmark_pytorch_gpu_stats.py"
  "${REPO_ROOT}/scripts/compare_dicc_sessions.py"
  "${SCRIPT_DIR}/mock_fixtures/make_fake_kernel.py"
)
for p in "${PY_SCRIPTS[@]}"; do
  if "${PYTHON}" -m py_compile "${p}"; then
    ok "py_compile $(basename "${p}")"
  else
    bad "py_compile $(basename "${p}")"
  fi
done

# ---------------------------------------------------------------------------
# 4. Mock CUDA stats: success path
# ---------------------------------------------------------------------------
echo ""
echo "--- mock CUDA stats success ---"
GOOD_DIR="${TMP_ROOT}/kernels_good"
mkdir -p "${GOOD_DIR}"
for name in fused_block1 fused_block2 fused_block3 fused_block3_fp16 fused_block3_naive fused_block4 fused_pipeline; do
  "${PYTHON}" "${SCRIPT_DIR}/mock_fixtures/make_fake_kernel.py" \
    --out-dir "${GOOD_DIR}" --name "${name}" --latency 50 --latency-no 55
done
GOOD_OUT="${TMP_ROOT}/cuda_good.json"
GOOD_RAW="${TMP_ROOT}/cuda_good_raw.json"
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/benchmark_cuda_kernels_stats.py" \
    --kernels-dir "${GOOD_DIR}" --suffix "" --tag mock \
    --n-trials 3 --strict --timeout-sec 10 \
    --output "${GOOD_OUT}" --raw-output "${GOOD_RAW}"; then
  if "${PYTHON}" -c "import json; d=json.load(open('${GOOD_OUT}')); assert d['fused_block1']['latency_us']['n']==3"; then
    ok "strict success path n=3"
  else
    bad "success JSON missing expected samples"
  fi
else
  bad "strict success path should exit 0"
fi

# ---------------------------------------------------------------------------
# 5. Mock CUDA stats: failure fixtures
# ---------------------------------------------------------------------------
echo ""
echo "--- mock CUDA stats failures ---"

# missing binary
MISS_DIR="${TMP_ROOT}/kernels_miss"
mkdir -p "${MISS_DIR}"
"${PYTHON}" "${SCRIPT_DIR}/mock_fixtures/make_fake_kernel.py" --out-dir "${MISS_DIR}" --name fused_block1
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/benchmark_cuda_kernels_stats.py" \
    --kernels-dir "${MISS_DIR}" --only fused_block2 --n-trials 1 --strict \
    --output "${TMP_ROOT}/miss.json" 2>/dev/null; then
  bad "missing binary should be fatal in strict mode"
else
  ok "missing binary fatal"
fi

# validation failure
FAIL_DIR="${TMP_ROOT}/kernels_failval"
mkdir -p "${FAIL_DIR}"
"${PYTHON}" "${SCRIPT_DIR}/mock_fixtures/make_fake_kernel.py" \
  --out-dir "${FAIL_DIR}" --name fused_block1 --fail-validation
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/benchmark_cuda_kernels_stats.py" \
    --kernels-dir "${FAIL_DIR}" --only fused_block1 --n-trials 1 --strict \
    --output "${TMP_ROOT}/failval.json" 2>/dev/null; then
  bad "validation FAILED should be fatal"
else
  ok "validation FAILED fatal"
fi

# nonzero exit
EXIT_DIR="${TMP_ROOT}/kernels_exit"
mkdir -p "${EXIT_DIR}"
"${PYTHON}" "${SCRIPT_DIR}/mock_fixtures/make_fake_kernel.py" \
  --out-dir "${EXIT_DIR}" --name fused_block1 --exit-code 7
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/benchmark_cuda_kernels_stats.py" \
    --kernels-dir "${EXIT_DIR}" --only fused_block1 --n-trials 1 --strict \
    --output "${TMP_ROOT}/exit.json" 2>/dev/null; then
  bad "nonzero exit should be fatal"
else
  ok "nonzero exit fatal"
fi

# missing metric
OMIT_DIR="${TMP_ROOT}/kernels_omit"
mkdir -p "${OMIT_DIR}"
"${PYTHON}" "${SCRIPT_DIR}/mock_fixtures/make_fake_kernel.py" \
  --out-dir "${OMIT_DIR}" --name fused_block1 --omit-metric
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/benchmark_cuda_kernels_stats.py" \
    --kernels-dir "${OMIT_DIR}" --only fused_block1 --n-trials 1 --strict \
    --output "${TMP_ROOT}/omit.json" 2>/dev/null; then
  bad "missing metric should be fatal"
else
  ok "missing metric fatal"
fi

# ---------------------------------------------------------------------------
# 6. Comparator: same-day, incomplete, provenance mismatch, happy path
# ---------------------------------------------------------------------------
echo ""
echo "--- compare_dicc_sessions mocks ---"

make_fake_run() {
  local dir="$1"
  local date_label="$2"
  local sha="$3"
  local cksum_extra="${4:-}"
  mkdir -p "${dir}/raw" "${dir}/logs"
  cat > "${dir}/manifest.json" <<EOF
{
  "campaign": "core",
  "gpu_label": "v100s",
  "gpu_name": "Tesla V100-SXM2-32GB",
  "gpu_compute_capability": "7.0",
  "gpu_memory_mib": 32510,
  "date_label": "${date_label}",
  "slurm_job_id": "1",
  "git_sha": "${sha}",
  "git_dirty": false,
  "checkpoint": "model/best_model_botiot_twostage.pth",
  "n_trials_cuda": 3,
  "n_trials_pytorch": 2,
  "pytorch_inner_forwards": 10
}
EOF
  echo "deadbeef  fused_block1${cksum_extra}" > "${dir}/kernel_SHA256SUMS"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${dir}/SUCCESS"
  echo 0 > "${dir}/exit_status"

  cat > "${dir}/cuda_kernel_stats.json" <<EOF
{
  "hardware_tag": "v100s",
  "n_trials": 3,
  "kernels": {
    "fused_block3_fp16": {
      "latency_us": {
        "n": 3, "mean": 100.0, "std": 1.0, "std_sample": 1.0,
        "p50": 100.0, "p95": 101.0, "min": 99.0, "max": 101.0,
        "cv_pct": 1.0, "ci95_low": 97.0, "ci95_high": 103.0, "ci95_halfwidth": 3.0
      }
    }
  },
  "fused_block3_fp16": {
    "latency_us": {
      "n": 3, "mean": 100.0, "std": 1.0, "std_sample": 1.0,
      "p50": 100.0, "p95": 101.0, "min": 99.0, "max": 101.0,
      "cv_pct": 1.0, "ci95_low": 97.0, "ci95_high": 103.0, "ci95_halfwidth": 3.0
    }
  }
}
EOF
  cat > "${dir}/raw/cuda_kernel_raw.json" <<EOF
{
  "kernels": {
    "fused_block3_fp16": {
      "samples": { "latency_us": [99.0, 100.0, 101.0] }
    }
  }
}
EOF
  cat > "${dir}/pytorch_gpu_stats.json" <<EOF
{
  "hardware_tag": "v100s",
  "checkpoint_sha256": "abc123",
  "protocol": {
    "n_trials": 2,
    "inner_forwards": 10,
    "warmup": 5,
    "model": "CNNBiLSTMAttention (V3)"
  },
  "metrics": {
    "block3_us": {
      "n": 2, "mean": 200.0, "std": 2.0, "std_sample": 2.0,
      "p50": 200.0, "p95": 201.0, "min": 198.0, "max": 202.0,
      "cv_pct": 1.0, "ci95_low": 180.0, "ci95_high": 220.0, "ci95_halfwidth": 20.0,
      "values": [198.0, 202.0]
    },
    "full_model_us": {
      "n": 2, "mean": 500.0, "std": 5.0, "std_sample": 5.0,
      "p50": 500.0, "p95": 505.0, "min": 495.0, "max": 505.0,
      "cv_pct": 1.0, "ci95_low": 450.0, "ci95_high": 550.0, "ci95_halfwidth": 50.0,
      "values": [495.0, 505.0]
    }
  },
  "block3_us": {
    "n": 2, "mean": 200.0, "std": 2.0, "values": [198.0, 202.0],
    "cv_pct": 1.0, "ci95_low": 180.0, "ci95_high": 220.0
  },
  "full_model_us": {
    "n": 2, "mean": 500.0, "std": 5.0, "values": [495.0, 505.0],
    "cv_pct": 1.0, "ci95_low": 450.0, "ci95_high": 550.0
  }
}
EOF
}

RUN_A="${TMP_ROOT}/runs/20260711_job1"
RUN_B="${TMP_ROOT}/runs/20260712_job2"
make_fake_run "${RUN_A}" "20260711" "deadbeefsha"
make_fake_run "${RUN_B}" "20260712" "deadbeefsha"

if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/compare_dicc_sessions.py" \
    --run-a "${RUN_A}" --run-b "${RUN_B}" --output "${TMP_ROOT}/compare_ok.json"; then
  ok "happy-path cross-day compare"
else
  bad "happy-path cross-day compare"
fi

# same-day reject
RUN_SAME="${TMP_ROOT}/runs/20260711_job9"
make_fake_run "${RUN_SAME}" "20260711" "deadbeefsha"
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/compare_dicc_sessions.py" \
    --run-a "${RUN_A}" --run-b "${RUN_SAME}" --output "${TMP_ROOT}/compare_same.json" 2>/dev/null; then
  bad "same-day compare should be rejected"
else
  ok "same-day rejected"
fi

# incomplete run (no SUCCESS)
RUN_INC="${TMP_ROOT}/runs/20260713_job3"
make_fake_run "${RUN_INC}" "20260713" "deadbeefsha"
rm -f "${RUN_INC}/SUCCESS"
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/compare_dicc_sessions.py" \
    --run-a "${RUN_A}" --run-b "${RUN_INC}" --output "${TMP_ROOT}/compare_inc.json" 2>/dev/null; then
  bad "incomplete run should be rejected"
else
  ok "incomplete run rejected"
fi

# provenance: different git sha
RUN_SHA="${TMP_ROOT}/runs/20260714_job4"
make_fake_run "${RUN_SHA}" "20260714" "other_sha_xxx"
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/compare_dicc_sessions.py" \
    --run-a "${RUN_A}" --run-b "${RUN_SHA}" --output "${TMP_ROOT}/compare_sha.json" 2>/dev/null; then
  bad "git sha mismatch should be rejected"
else
  ok "git sha mismatch rejected"
fi

# provenance: different binary checksums
RUN_BIN="${TMP_ROOT}/runs/20260715_job5"
make_fake_run "${RUN_BIN}" "20260715" "deadbeefsha" "_different"
if PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/compare_dicc_sessions.py" \
    --run-a "${RUN_A}" --run-b "${RUN_BIN}" --output "${TMP_ROOT}/compare_bin.json" 2>/dev/null; then
  bad "binary checksum mismatch should be rejected"
else
  ok "binary checksum mismatch rejected"
fi

# ---------------------------------------------------------------------------
# 7. GPU assertion helpers (mock nvidia-smi)
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU assertion / missing conda mocks ---"
# shellcheck source=../lib/common.sh
source "${DICC_DIR}/lib/common.sh"

MOCK_BIN="${TMP_ROOT}/mockbin"
mkdir -p "${MOCK_BIN}"
export PATH="${MOCK_BIN}:${PATH}"

# wrong GPU name
cat > "${MOCK_BIN}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"-L"* ]]; then
  echo "GPU 0: NVIDIA GeForce RTX 3050 (UUID: fake)"
  exit 0
fi
if [[ "$*" == *"query-gpu=name"* ]]; then
  echo "NVIDIA GeForce RTX 3050"
  exit 0
fi
if [[ "$*" == *"mig.mode"* ]]; then
  echo "Disabled"
  exit 0
fi
if [[ "$*" == *"memory.total"* ]]; then
  echo "6144"
  exit 0
fi
if [[ "$*" == *"compute_cap"* ]]; then
  echo "8.6"
  exit 0
fi
exit 0
EOF
chmod +x "${MOCK_BIN}/nvidia-smi"

# assert_gpu/die call exit — run in subshell so the suite continues.
if ( assert_gpu v100s 'V100' 7.0 30000 ) 2>/dev/null; then
  bad "wrong GPU name should fail assert_gpu"
else
  ok "wrong GPU rejected by assert_gpu"
fi

# PATH with only mock nvidia and system essentials — no conda.
# Disable filesystem autodetection so a developer laptop's miniconda does not
# mask the failure path that non-interactive cluster jobs hit.
if (
  export PATH="${MOCK_BIN}:/usr/bin:/bin"
  export COLIDE_CONDA_DISABLE_AUTODETECT=1
  unset CONDA_EXE CONDA_PREFIX CONDA_DEFAULT_ENV
  unset -f conda 2>/dev/null || true
  init_conda
) 2>/dev/null; then
  bad "missing conda should fail init_conda"
else
  ok "missing conda rejected by init_conda"
fi

# ---------------------------------------------------------------------------
# 8. Invocation from simulated Slurm spool directory (relative path trap)
# ---------------------------------------------------------------------------
echo ""
echo "--- simulated Slurm spool invocation ---"
SPOOL="${TMP_ROOT}/slurm_spool"
FAKE_ROOT="${TMP_ROOT}/fake_colide"
mkdir -p "${SPOOL}" \
  "${FAKE_ROOT}/dicc_scripts" \
  "${FAKE_ROOT}/inference/kernels/v100" \
  "${FAKE_ROOT}/inference/kernels/a100" \
  "${FAKE_ROOT}/model" \
  "${FAKE_ROOT}/benchmarks/results/dicc"

# Minimal tree so submit_session preflight passes without writing into the
# real repository checkout.
cp "${DICC_DIR}/02_benchmark_v100.sh" "${FAKE_ROOT}/dicc_scripts/"
cp "${DICC_DIR}/03_benchmark_a100.sh" "${FAKE_ROOT}/dicc_scripts/"
cp "${DICC_DIR}/04_nsight_profile.sh" "${FAKE_ROOT}/dicc_scripts/"
cp "${DICC_DIR}/submit_session.sh" "${FAKE_ROOT}/dicc_scripts/"
cp -R "${DICC_DIR}/lib" "${FAKE_ROOT}/dicc_scripts/lib"
for k in fused_block1 fused_block2 fused_block3 fused_block3_fp16 fused_block3_naive fused_block4 fused_pipeline; do
  echo '#!/bin/true' > "${FAKE_ROOT}/inference/kernels/v100/${k}"
  echo '#!/bin/true' > "${FAKE_ROOT}/inference/kernels/a100/${k}"
  chmod +x "${FAKE_ROOT}/inference/kernels/v100/${k}" "${FAKE_ROOT}/inference/kernels/a100/${k}"
done
: > "${FAKE_ROOT}/model/best_model_botiot_twostage.pth"
(
  cd "${FAKE_ROOT}"
  git init -q
  git config user.email "validate@example.com"
  git config user.name "validate"
  git add -A
  git commit -q -m "validate fixture"
)

if (
  cd "${SPOOL}"
  export COLIDE_ROOT="${FAKE_ROOT}"
  bash "${FAKE_ROOT}/dicc_scripts/submit_session.sh" --dry-run --date 20990101 --campaign validate_spool
); then
  ok "submit_session dry-run from spool CWD with COLIDE_ROOT"
else
  bad "submit_session dry-run from spool CWD"
fi

# Also ensure common.sh resolve works when sourced from non-repo CWD
if (
  cd "${SPOOL}"
  # shellcheck source=/dev/null
  source "${DICC_DIR}/lib/common.sh"
  root="$(resolve_colide_root)"
  [[ -d "${root}/dicc_scripts" ]]
); then
  ok "resolve_colide_root from spool CWD"
else
  bad "resolve_colide_root from spool CWD"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Validation summary: ${pass} passed, ${fail} failed ==="
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi
exit 0
