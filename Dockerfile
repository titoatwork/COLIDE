# =============================================================================
# COLIDE Reproducibility Container
# CUDA-Optimized CNN-BiLSTM with LLM Explainability for IoT IDS
#
# Build:  docker build -t colide .
# Run:    docker run --gpus all colide
# Shell:  docker run --gpus all -it colide bash
# =============================================================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL maintainer="Ibteshamul Haque"
LABEL description="COLIDE: Custom CUDA inference kernels for IoT IDS"
LABEL version="1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/colide

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /colide
COPY requirements.txt .

# Python dependencies
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    -r requirements.txt

# Copy project
COPY . .

# Compile CUDA kernels
# Default sm_86 (Ampere: RTX 3050-3090, A100)
# Override with: docker build --build-arg CUDA_ARCH=sm_89 -t colide .
ARG CUDA_ARCH=sm_86
RUN nvcc -arch=${CUDA_ARCH} -o inference/kernels/fused_block1 inference/kernels/fused_block1.cu && \
    nvcc -arch=${CUDA_ARCH} -o inference/kernels/fused_block2 inference/kernels/fused_block2.cu && \
    nvcc -arch=${CUDA_ARCH} -o inference/kernels/fused_block3 inference/kernels/fused_block3.cu && \
    nvcc -arch=${CUDA_ARCH} -o inference/kernels/fused_block3_fp16 inference/kernels/fused_block3_fp16.cu && \
    nvcc -arch=${CUDA_ARCH} -o inference/kernels/fused_block4 inference/kernels/fused_block4.cu

RUN chmod +x benchmark.sh

CMD ["bash", "benchmark.sh"]
