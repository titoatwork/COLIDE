#!/bin/bash
#SBATCH --job-name=colide_tcomp_v100s
#SBATCH --partition=gpu-v100s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/framework/torch_compile_v100s_%j.out
#SBATCH --error=/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/framework/torch_compile_v100s_%j.err
set -Eeuo pipefail
export HOME=/home/user/ibteshamulhaque
export MAMBA_ROOT_PREFIX=$HOME/micromamba
export PATH=/app/cuda/cuda-12.4/bin:$PATH
echo host=$(hostname) gpu=$(nvidia-smi -L | head -1)
cd /home/user/ibteshamulhaque/colide
$HOME/micromamba/bin/micromamba run -n colide python scripts/benchmark_torch_compile_dicc.py \
  --checkpoint model/best_model_botiot_twostage.pth \
  --n-trials 20 --inner 200 --warmup 50 --tag v100s \
  --output benchmarks/results/dicc/framework/torch_compile_v100s.json
