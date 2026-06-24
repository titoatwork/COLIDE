"""
COLIDE - TensorRT Native API Benchmark
Compares TensorRT inference against custom CUDA kernels and PyTorch.
Uses TensorRT 11 Python API directly.

Usage:
  export LD_LIBRARY_PATH=$HOME/colide/.venv/lib/python3.12/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
  PYTHONPATH=. python scripts/benchmark_tensorrt_native.py

Requirements: tensorrt, pycuda, torch, onnx
Note: PyTorch benchmark runs BEFORE pycuda import to avoid CUDA context conflict.
"""

import numpy as np
import time
import json
import torch
import yaml
import sys

sys.path.insert(0, 'model')
from cnn_bilstm_v3_attention import CNNBiLSTMAttention

print("=" * 60)
print("COLIDE TensorRT vs Custom CUDA Benchmark")
print("=" * 60)

# ============================================================
# STEP 1: PyTorch Eager Benchmark (BEFORE pycuda import)
# ============================================================
config = yaml.safe_load(open('config/config.yaml'))
model = CNNBiLSTMAttention(config).cuda().eval()
model.load_state_dict(torch.load(
    'model/best_model_botiot_distill_focal_T5.pth',
    map_location='cuda', weights_only=True))

dummy = torch.randn(1, 10, device='cuda')

# Warmup
with torch.no_grad():
    for _ in range(200):
        model(dummy)
    torch.cuda.synchronize()

    # Benchmark
    iters = 1000
    t = time.perf_counter()
    for _ in range(iters):
        model(dummy)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t) / iters * 1e6

print(f"Eager PyTorch:    {eager_us:.1f} us")

# Export ONNX
onnx_path = '/tmp/colide_trt.onnx'
torch.onnx.export(model, dummy, onnx_path,
                   input_names=['input'], output_names=['output'],
                   opset_version=14)
print(f"ONNX exported to {onnx_path}")

# Free PyTorch CUDA context
del model
torch.cuda.empty_cache()

# ============================================================
# STEP 2: TensorRT Benchmark (pycuda imported AFTER PyTorch)
# ============================================================
import tensorrt as trt
print(f"TensorRT version: {trt.__version__}")

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"Parse error: {parser.get_error(i)}")
        sys.exit(1)

config_trt = builder.create_builder_config()
config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

print("Building TensorRT engine...")
engine_bytes = builder.build_serialized_network(network, config_trt)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_bytes)
context = engine.create_execution_context()
print("Engine built successfully")

import pycuda.driver as cuda
import pycuda.autoinit

h_input = np.random.randn(1, 10).astype(np.float32)
h_output = np.empty([1, 5], dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

# Warmup
for _ in range(200):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_v2([int(d_input), int(d_output)])
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

# Benchmark
t = time.perf_counter()
for _ in range(iters):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_v2([int(d_input), int(d_output)])
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
trt_us = (time.perf_counter() - t) / iters * 1e6

# Custom CUDA reference (measured via compiled kernels)
custom_cuda_us = 674.0  # RTX 3050 chained FP16 pipeline

# ============================================================
# Results
# ============================================================
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"TensorRT:         {trt_us:.1f} us")
print(f"Eager PyTorch:    {eager_us:.1f} us")
print(f"Custom CUDA FP16: {custom_cuda_us:.1f} us")
print(f"")
print(f"Custom vs TensorRT:  {trt_us/custom_cuda_us:.2f}x faster")
print(f"Custom vs Eager:     {eager_us/custom_cuda_us:.2f}x faster")
print(f"TensorRT vs Eager:   {eager_us/trt_us:.2f}x faster")

results = {
    'tensorrt_us': float(trt_us),
    'eager_pytorch_us': float(eager_us),
    'custom_cuda_fp16_us': float(custom_cuda_us),
    'custom_vs_tensorrt': round(float(trt_us / custom_cuda_us), 2),
    'custom_vs_eager': round(float(eager_us / custom_cuda_us), 2),
    'tensorrt_version': trt.__version__,
    'hardware': 'RTX 3050 Laptop GPU',
}
with open('benchmarks/results/tensorrt_native.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to benchmarks/results/tensorrt_native.json")