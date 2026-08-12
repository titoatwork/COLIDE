#!/bin/bash
#SBATCH --job-name=colide_mcomp_a100
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/framework/multi_compiler_a100_%j.out
#SBATCH --error=/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/framework/multi_compiler_a100_%j.err
set -Eeuo pipefail
export HOME=/home/user/ibteshamulhaque
export MAMBA_ROOT_PREFIX=$HOME/micromamba
export PATH=/app/cuda/cuda-12.4/bin:$PATH
export COLIDE_TRT8_LIBS=$HOME/colide/third_party/trt8_libs
echo "host=$(hostname)"
nvidia-smi -L 2>&1 | sed -n '1p' || true
echo "trt8_libs=$(ls $COLIDE_TRT8_LIBS/libnvinfer.so* 2>/dev/null | sed -n '1p')"
cd $HOME/colide
$HOME/micromamba/bin/micromamba run -n colide env \
  COLIDE_TRT8_LIBS="$COLIDE_TRT8_LIBS" \
  python scripts/benchmark_multi_compiler_dicc.py \
  --checkpoint model/best_model_botiot_twostage.pth \
  --onnx benchmarks/results/dicc/framework/colide_champion.onnx \
  --n-trials 20 --inner 200 --warmup 50 --tag a100 \
  --output benchmarks/results/dicc/framework/multi_compiler_a100.json
