#!/bin/bash
#SBATCH --job-name=colide_mcomp_v100s
#SBATCH --partition=gpu-v100s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/framework/multi_compiler_v100s_%j.out
#SBATCH --error=/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/framework/multi_compiler_v100s_%j.err
set -Eeuo pipefail
export HOME=/home/user/ibteshamulhaque
export MAMBA_ROOT_PREFIX=$HOME/micromamba
export PATH=/app/cuda/cuda-12.4/bin:$PATH
# Do NOT force nvidia/cudnn into LD_LIBRARY_PATH on V100 — mixed cuDNN8/9
# fragments break torch with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
# TRT8 libs are preloaded later inside Python via COLIDE_TRT8_LIBS only.
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
  --n-trials 20 --inner 200 --warmup 50 --tag v100s \
  --output benchmarks/results/dicc/framework/multi_compiler_v100s.json
