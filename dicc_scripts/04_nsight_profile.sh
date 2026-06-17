#!/bin/bash
#SBATCH --job-name=colide_nsight
#SBATCH --output=benchmarks/results/dicc_nsight_%j.log
#SBATCH --error=benchmarks/results/dicc_nsight_%j.err
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

echo "=== COLIDE Nsight Compute Profiling ==="
echo "Node: $(hostname)"
nvidia-smi
date

cd /scr/$USER/colide
mkdir -p benchmarks/results/nsight

# Profile each kernel
echo "Profiling Block 1..."
ncu --set full -o benchmarks/results/nsight/block1_a100 ./inference/kernels/a100/fused_block1

echo "Profiling Block 2..."
ncu --set full -o benchmarks/results/nsight/block2_a100 ./inference/kernels/a100/fused_block2

echo "Profiling Block 3 FP32..."
ncu --set full -o benchmarks/results/nsight/block3_fp32_a100 ./inference/kernels/a100/fused_block3

echo "Profiling Block 3 FP16..."
ncu --set full -o benchmarks/results/nsight/block3_fp16_a100 ./inference/kernels/a100/fused_block3_fp16

echo "Profiling Block 4..."
ncu --set full -o benchmarks/results/nsight/block4_a100 ./inference/kernels/a100/fused_block4

echo ""
echo "=== Nsight Profiling Complete ==="
echo "Download .ncu-rep files and open in Nsight Compute GUI"
date
