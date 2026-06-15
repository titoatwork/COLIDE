#!/bin/bash
# =============================================================================
# COLIDE - Complete Benchmark Suite
# Runs all benchmarks and saves results to benchmarks/results/
#
# Usage:
#   bash benchmark.sh          # Run all benchmarks
#   bash benchmark.sh --quick  # Skip slow benchmarks (streaming, LLM)
# =============================================================================
set -e

cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=.

QUICK_MODE=false
if [ "$1" == "--quick" ]; then
    QUICK_MODE=true
fi

TOTAL_START=$(date +%s)

echo "============================================================"
echo "COLIDE BENCHMARK SUITE"
echo "============================================================"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'N/A')"
echo "Python: $(python3 --version 2>/dev/null || echo 'N/A')"
echo "Mode: $(if $QUICK_MODE; then echo 'QUICK'; else echo 'FULL'; fi)"
echo "============================================================"

run_benchmark() {
    local name=$1
    local cmd=$2
    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] $name"
    echo "------------------------------------------------------------"
    local start=$(date +%s)
    eval "$cmd"
    local end=$(date +%s)
    echo "[Completed in $((end - start)) seconds]"
}

# CUDA kernel benchmarks
run_benchmark "Block 1 (Proj+Conv+BN+ReLU)" "./inference/kernels/fused_block1"
run_benchmark "Block 2 (Conv+BN+ReLU+Pool)" "./inference/kernels/fused_block2"
run_benchmark "Block 3 FP32 (BiLSTM)" "./inference/kernels/fused_block3"
run_benchmark "Block 3 FP16 (BiLSTM half2)" "./inference/kernels/fused_block3_fp16"
run_benchmark "Block 4 (Dense Head)" "./inference/kernels/fused_block4"

# Python benchmarks
run_benchmark "Full Pipeline Comparison" "python scripts/benchmark_pipeline.py"
run_benchmark "Batch Size Scaling" "python scripts/benchmark_batch.py"
run_benchmark "ONNX Runtime Comparison" "python scripts/benchmark_ort.py"
run_benchmark "Statistical Confidence" "python scripts/benchmark_stats.py"
run_benchmark "Energy Efficiency" "python scripts/benchmark_energy.py"
run_benchmark "Ablation Study" "python scripts/ablation_study.py"
run_benchmark "V2 vs V3 Comparison" "python scripts/compare_v2_v3.py"

if ! $QUICK_MODE; then
    run_benchmark "Streaming Throughput" "python scripts/benchmark_streaming.py"
    run_benchmark "LLM Explainability" "python scripts/llm_explainability.py"
fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "============================================================"
echo "Total time: $((TOTAL_ELAPSED / 60)) min $((TOTAL_ELAPSED % 60)) sec"
echo "Results saved to: benchmarks/results/"
echo "============================================================"
