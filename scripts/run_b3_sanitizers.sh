#!/usr/bin/env bash
# Archive compute-sanitizer results for corrected Block-3 binaries (Phase 4).
# Usage: bash scripts/run_b3_sanitizers.sh [sm_86]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KDIR="$ROOT/inference/kernels"
OUT="$ROOT/benchmarks/results/sanitizer_b3"
ARCH="${1:-sm_86}"
mkdir -p "$OUT"
export PATH="${CUDA_HOME:-/usr/local/cuda-12.6}/bin:${PATH:-}"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
meta="$OUT/meta_${ts}.txt"
{
  echo "timestamp_utc=$ts"
  echo "pwd=$ROOT"
  echo "git_sha=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty=$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l)"
  echo "nvcc=$(nvcc --version 2>/dev/null | tail -1 || true)"
  echo "gpu=$(nvidia-smi -L 2>/dev/null || true)"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  echo "arch=$ARCH"
} | tee "$meta"

compile_one() {
  local src="$1" out="$2"
  echo "=== nvcc $src -> $out ($ARCH) ==="
  nvcc -arch="$ARCH" -O3 -o "$out" "$src" 2>&1 | tee "$OUT/compile_$(basename "$out")_${ts}.log"
  sha256sum "$out" | tee -a "$meta"
}

run_tool() {
  local bin="$1" tool="$2"
  local base
  base="$(basename "$bin")"
  local log="$OUT/${base}_${tool}_${ts}.log"
  echo "=== compute-sanitizer --tool $tool $bin ==="
  # memcheck is the default tool name in newer sanitizer
  if compute-sanitizer --tool "$tool" "$bin" >"$log" 2>&1; then
    echo "PASS $base $tool" | tee -a "$meta"
  else
    echo "FAIL $base $tool exit=$?" | tee -a "$meta"
  fi
  # extract summary lines
  grep -E 'RACECHECK|ERROR SUMMARY|ERROR SUMMARY|errors|hazards|=========' "$log" | tail -20 | tee -a "$meta" || true
}

cd "$KDIR"
compile_one fused_block3.cu fused_block3
compile_one fused_block3_fp16.cu fused_block3_fp16

for bin in fused_block3 fused_block3_fp16; do
  # plain self-check first
  "./$bin" >"$OUT/${bin}_selfcheck_${ts}.log" 2>&1 || echo "selfcheck_exit=$?" | tee -a "$meta"
  tail -20 "$OUT/${bin}_selfcheck_${ts}.log" | tee -a "$meta"
  for tool in racecheck synccheck initcheck memcheck; do
    run_tool "./$bin" "$tool"
  done
done

echo "Artifacts under $OUT"
ls -la "$OUT" | tee -a "$meta"
