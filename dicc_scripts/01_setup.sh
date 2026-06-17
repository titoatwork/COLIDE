#!/bin/bash
# =============================================================================
# COLIDE DICC Setup - Run this ONCE after first SSH login
# =============================================================================
set -e

echo "=== COLIDE DICC SETUP ==="

# Clone repo to scratch (fast storage)
cd /scr/$USER
git clone https://github.com/titoatwork/COLIDE.git colide
cd colide

# Load modules (adjust based on 'module avail' output)
module load cuda/12.1 2>/dev/null || echo "CUDA module not found, checking nvcc..."
which nvcc || echo "ERROR: nvcc not found. Run 'module avail' and load CUDA manually."

# Setup Python environment
module load miniconda 2>/dev/null || echo "Miniconda module not found"
conda create -n colide python=3.12 -y
conda activate colide
pip install torch numpy pyyaml scikit-learn scipy onnx onnxruntime-gpu

# Check GPU
nvidia-smi
nvcc --version

# Compile kernels for V100 (sm_70) and A100 (sm_80)
echo "=== Compiling for V100 (sm_70) ==="
mkdir -p inference/kernels/v100
nvcc -arch=sm_70 -o inference/kernels/v100/fused_block1 inference/kernels/fused_block1.cu
nvcc -arch=sm_70 -o inference/kernels/v100/fused_block2 inference/kernels/fused_block2.cu
nvcc -arch=sm_70 -o inference/kernels/v100/fused_block3 inference/kernels/fused_block3.cu
nvcc -arch=sm_70 -o inference/kernels/v100/fused_block3_fp16 inference/kernels/fused_block3_fp16.cu
nvcc -arch=sm_70 -o inference/kernels/v100/fused_block4 inference/kernels/fused_block4.cu

echo "=== Compiling for A100 (sm_80) ==="
mkdir -p inference/kernels/a100
nvcc -arch=sm_80 -o inference/kernels/a100/fused_block1 inference/kernels/fused_block1.cu
nvcc -arch=sm_80 -o inference/kernels/a100/fused_block2 inference/kernels/fused_block2.cu
nvcc -arch=sm_80 -o inference/kernels/a100/fused_block3 inference/kernels/fused_block3.cu
nvcc -arch=sm_80 -o inference/kernels/a100/fused_block3_fp16 inference/kernels/fused_block3_fp16.cu
nvcc -arch=sm_80 -o inference/kernels/a100/fused_block4 inference/kernels/fused_block4.cu

echo "=== Setup Complete ==="
echo "Run: sbatch 02_benchmark_v100.sh"
echo "Run: sbatch 03_benchmark_a100.sh"
echo "Run: sbatch 04_nsight_profile.sh"
