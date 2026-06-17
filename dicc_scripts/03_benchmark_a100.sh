#!/bin/bash
#SBATCH --job-name=colide_a100
#SBATCH --output=benchmarks/results/dicc_a100_%j.log
#SBATCH --error=benchmarks/results/dicc_a100_%j.err
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

echo "=== COLIDE A100 Benchmark ==="
echo "Node: $(hostname)"
nvidia-smi
date

cd /scr/$USER/colide
conda activate colide
export PYTHONPATH=.

echo ""
echo "=== CUDA Kernel Benchmarks (A100) ==="
./inference/kernels/a100/fused_block1
echo "---"
./inference/kernels/a100/fused_block2
echo "---"
./inference/kernels/a100/fused_block3
echo "---"
./inference/kernels/a100/fused_block3_fp16
echo "---"
./inference/kernels/a100/fused_block4

echo ""
echo "=== Python Benchmarks ==="
python scripts/benchmark_pipeline.py
echo "---"
python scripts/benchmark_batch.py
echo "---"
python scripts/benchmark_ort.py
echo "---"
python scripts/benchmark_energy.py
echo "---"
python scripts/benchmark_stats.py

echo ""
echo "=== A100 Benchmark Complete ==="
date
