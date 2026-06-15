"""
COLIDE - ONNX Runtime GPU Benchmark
Exports PyTorch model to ONNX, runs inference with ONNX Runtime GPU.
Provides framework comparison numbers for the paper.
"""

import sys
import time
import numpy as np
import yaml
import torch

sys.path.insert(0, '.')

from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

import onnxruntime as ort

# Load config and model
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

model = CNNBiLSTM(config)
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu', weights_only=True))
model.eval()

print("=== COLIDE ONNX Runtime GPU Benchmark ===\n")

# ---------------------------------------------------------------
# Step 1: Export to ONNX
# ---------------------------------------------------------------
dummy = torch.randn(1, 10)
onnx_path = 'model/colide_model.onnx'

torch.onnx.export(
    model, dummy, onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17
)
print(f"Exported ONNX model to {onnx_path}")

# ---------------------------------------------------------------
# Step 2: Create ONNX Runtime sessions (CPU and GPU)
# ---------------------------------------------------------------
# CPU session
cpu_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# GPU session
gpu_providers = ort.get_available_providers()
print(f"Available providers: {gpu_providers}")

if 'CUDAExecutionProvider' in gpu_providers:
    gpu_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    has_gpu = True
    print("Using CUDAExecutionProvider")
else:
    has_gpu = False
    print("WARNING: CUDAExecutionProvider not available, GPU benchmark skipped")

# ---------------------------------------------------------------
# Step 3: PyTorch baselines (for comparison in same script)
# ---------------------------------------------------------------
input_np = np.random.randn(1, 10).astype(np.float32)
input_torch = torch.from_numpy(input_np)

# PyTorch CPU
runs = 200
times = []
for _ in range(runs):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_torch)
    times.append((time.perf_counter() - start) * 1e6)
times.sort()
pytorch_cpu_us = times[runs // 2]

# PyTorch GPU
model_gpu = model.cuda()
input_gpu = input_torch.cuda()
for _ in range(20):
    with torch.no_grad():
        _ = model_gpu(input_gpu)
torch.cuda.synchronize()

times = []
for _ in range(runs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model_gpu(input_gpu)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1e6)
times.sort()
pytorch_gpu_us = times[runs // 2]

print(f"\n--- PyTorch Baselines ---")
print(f"PyTorch CPU p50: {pytorch_cpu_us:.1f} us")
print(f"PyTorch GPU p50: {pytorch_gpu_us:.1f} us")

# ---------------------------------------------------------------
# Step 4: ONNX Runtime CPU benchmark
# ---------------------------------------------------------------
# Warmup
for _ in range(20):
    cpu_session.run(None, {'input': input_np})

times = []
for _ in range(runs):
    start = time.perf_counter()
    cpu_session.run(None, {'input': input_np})
    times.append((time.perf_counter() - start) * 1e6)
times.sort()
ort_cpu_us = times[runs // 2]

print(f"\n--- ONNX Runtime ---")
print(f"ORT CPU p50:     {ort_cpu_us:.1f} us")

# ---------------------------------------------------------------
# Step 5: ONNX Runtime GPU benchmark
# ---------------------------------------------------------------
if has_gpu:
    # Warmup
    for _ in range(20):
        gpu_session.run(None, {'input': input_np})

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        gpu_session.run(None, {'input': input_np})
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    ort_gpu_us = times[runs // 2]
    print(f"ORT GPU p50:     {ort_gpu_us:.1f} us")

# ---------------------------------------------------------------
# Step 5b: TensorRT via ONNX Runtime
# ---------------------------------------------------------------
if 'TensorrtExecutionProvider' in gpu_providers:
    trt_session = ort.InferenceSession(onnx_path, providers=['TensorrtExecutionProvider'])
    for _ in range(20):
        trt_session.run(None, {'input': input_np})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        trt_session.run(None, {'input': input_np})
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    trt_us = times[runs // 2]
    print(f"TensorRT p50:    {trt_us:.1f} us")
    
# ---------------------------------------------------------------
# Step 6: Summary table
# ---------------------------------------------------------------
print(f"\n{'='*55}")
print(f"{'Method':<25} {'Latency (us)':>12} {'vs PyTorch GPU':>15}")
print(f"{'='*55}")
print(f"{'PyTorch CPU':<25} {pytorch_cpu_us:>12.1f} {pytorch_gpu_us/pytorch_cpu_us:>14.2f}x")
print(f"{'PyTorch GPU':<25} {pytorch_gpu_us:>12.1f} {'1.00x':>15}")
print(f"{'ORT CPU':<25} {ort_cpu_us:>12.1f} {pytorch_gpu_us/ort_cpu_us:>14.2f}x")
if has_gpu:
    print(f"{'ORT GPU':<25} {ort_gpu_us:>12.1f} {pytorch_gpu_us/ort_gpu_us:>14.2f}x")
if 'TensorrtExecutionProvider' in gpu_providers:
    print(f"{'TensorRT':<25} {trt_us:>12.1f} {pytorch_gpu_us/trt_us:>14.2f}x")
print(f"{'Custom CUDA FP32':<25} {'1143':>12} {pytorch_gpu_us/1143:>14.2f}x")
print(f"{'Custom CUDA FP16':<25} {'770':>12} {pytorch_gpu_us/770:>14.2f}x")
print(f"{'='*55}")