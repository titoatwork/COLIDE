"""
COLIDE - Batch Size Inference Benchmark
Compares PyTorch CPU, PyTorch GPU, ORT CPU, ORT GPU at batch sizes 1, 32, 128, 256.
Shows the crossover point where GPU starts dominating CPU.
"""

import sys
import time
import copy
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

# ONNX export (with dynamic batch)
dummy = torch.randn(1, 10)
onnx_path = 'model/colide_model.onnx'
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17
)

model_cpu = copy.deepcopy(model)
model_gpu = model.cuda()

cpu_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
gpu_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

print("=== COLIDE Batch Inference Benchmark ===\n")

batch_sizes = [1, 32, 128, 256]
runs = 100

results = {}

for bs in batch_sizes:
    input_np = np.random.randn(bs, 10).astype(np.float32)
    input_torch = torch.from_numpy(input_np)
    input_gpu_t = input_torch.cuda()

    # --- PyTorch CPU ---
    times = []
    for _ in range(10):
        with torch.no_grad():
            _ = model_cpu(input_torch)
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu(input_torch)
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    pt_cpu = times[runs // 2]

    # --- PyTorch GPU ---
    for _ in range(10):
        with torch.no_grad():
            _ = model_gpu(input_gpu_t)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_gpu(input_gpu_t)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    pt_gpu = times[runs // 2]

    # --- ORT CPU ---
    for _ in range(10):
        cpu_session.run(None, {'input': input_np})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        cpu_session.run(None, {'input': input_np})
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    ort_cpu = times[runs // 2]

    # --- ORT GPU ---
    for _ in range(10):
        gpu_session.run(None, {'input': input_np})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        gpu_session.run(None, {'input': input_np})
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    ort_gpu = times[runs // 2]

    results[bs] = {
        'pt_cpu': pt_cpu,
        'pt_gpu': pt_gpu,
        'ort_cpu': ort_cpu,
        'ort_gpu': ort_gpu
    }

# Print latency table
print(f"{'='*75}")
print(f"{'LATENCY (us)':<15} {'PyTorch CPU':>12} {'PyTorch GPU':>12} {'ORT CPU':>12} {'ORT GPU':>12}")
print(f"{'='*75}")
for bs in batch_sizes:
    r = results[bs]
    print(f"Batch {bs:<9} {r['pt_cpu']:>12.1f} {r['pt_gpu']:>12.1f} {r['ort_cpu']:>12.1f} {r['ort_gpu']:>12.1f}")

# Print throughput table (flows/sec)
print(f"\n{'='*75}")
print(f"{'THROUGHPUT (flows/sec)':<15} {'PyTorch CPU':>12} {'PyTorch GPU':>12} {'ORT CPU':>12} {'ORT GPU':>12}")
print(f"{'='*75}")
for bs in batch_sizes:
    r = results[bs]
    pt_cpu_tput = bs / (r['pt_cpu'] / 1e6)
    pt_gpu_tput = bs / (r['pt_gpu'] / 1e6)
    ort_cpu_tput = bs / (r['ort_cpu'] / 1e6)
    ort_gpu_tput = bs / (r['ort_gpu'] / 1e6)
    print(f"Batch {bs:<9} {pt_cpu_tput:>12,.0f} {pt_gpu_tput:>12,.0f} {ort_cpu_tput:>12,.0f} {ort_gpu_tput:>12,.0f}")

# Print per-sample latency
print(f"\n{'='*75}")
print(f"{'PER-SAMPLE (us)':<15} {'PyTorch CPU':>12} {'PyTorch GPU':>12} {'ORT CPU':>12} {'ORT GPU':>12}")
print(f"{'='*75}")
for bs in batch_sizes:
    r = results[bs]
    print(f"Batch {bs:<9} {r['pt_cpu']/bs:>12.1f} {r['pt_gpu']/bs:>12.1f} {r['ort_cpu']/bs:>12.1f} {r['ort_gpu']/bs:>12.1f}")

# Find crossover point
print(f"\n--- Key Findings ---")
for bs in batch_sizes:
    r = results[bs]
    fastest = min(r, key=r.get)
    print(f"Batch {bs}: fastest = {fastest} ({r[fastest]:.1f} us)")