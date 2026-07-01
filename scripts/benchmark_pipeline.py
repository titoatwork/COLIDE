"""
COLIDE - Full Pipeline Benchmark Summary
Runs PyTorch baselines and collects custom CUDA block times.
Produces paper-ready comparison tables and saves to JSON.
"""

import re
import sys
import time
import json
import copy
import numpy as np
import yaml
import torch

sys.path.insert(0, '.')

# Hardware tag so this script's output survives being run on multiple GPUs
# (RTX 3050 locally, V100S/A100 via DICC) without one run silently
# overwriting another's results at the same fixed path -- this is why no
# per-hardware PyTorch GPU baseline existed for V100S/A100 previously.
_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
GPU_TAG = re.sub(r'[^A-Za-z0-9]+', '_', _gpu_name).strip('_').lower()

from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

# Load config and model
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

model = CNNBiLSTM(config)
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu', weights_only=True))
model.eval()

model_cpu = copy.deepcopy(model)
model_gpu = model.cuda()

print("=" * 70)
print("COLIDE FULL PIPELINE BENCHMARK")
print("=" * 70)

# ================================================================
# Custom CUDA block times (measured from individual benchmarks)
# ================================================================
cuda_blocks = {
    'block1_proj_conv_bn_relu': 61.7,
    'block2_conv_bn_relu_pool': 87.2,
    'block3_bilstm_transposed': 973.8,   # with CUDA Graphs
    'block3_bilstm_fp16_half2': 601.4,
    'block3_bilstm_no_graphs':  1007.3,   # without CUDA Graphs
    'block4_dense_head':        20.1,
}
cuda_total_graphs = cuda_blocks['block1_proj_conv_bn_relu'] + cuda_blocks['block2_conv_bn_relu_pool'] + cuda_blocks['block3_bilstm_transposed'] + cuda_blocks['block4_dense_head']
cuda_total_fp16 = cuda_blocks['block1_proj_conv_bn_relu'] + cuda_blocks['block2_conv_bn_relu_pool'] + cuda_blocks['block3_bilstm_fp16_half2'] + cuda_blocks['block4_dense_head']
cuda_total_no_graphs = cuda_blocks['block1_proj_conv_bn_relu'] + cuda_blocks['block2_conv_bn_relu_pool'] + cuda_blocks['block3_bilstm_no_graphs'] + cuda_blocks['block4_dense_head']

print(f"\n--- Custom CUDA Block Times (single sample) ---")
for name, t in cuda_blocks.items():
    print(f"  {name:<35} {t:>8.1f} us")
print(f"  {'TOTAL (with CUDA Graphs)':<35} {cuda_total_graphs:>8.1f} us")
print(f"  {'TOTAL (without CUDA Graphs)':<35} {cuda_total_no_graphs:>8.1f} us")

# ================================================================
# PyTorch block-level benchmarks
# ================================================================
print(f"\n--- PyTorch Block-Level Benchmarks ---")

# Block definitions matching custom CUDA blocks
class Block1(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.proj = m.input_projection
        self.conv1 = m.conv1; self.bn1 = m.bn1; self.relu = m.relu
    def forward(self, x):
        x = self.proj(x)
        x = x.view(x.size(0), 2, 32)
        x = self.relu(self.bn1(self.conv1(x)))
        return x

class Block2(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv2 = m.conv2; self.bn2 = m.bn2; self.relu = m.relu; self.pool = m.pool
    def forward(self, x):
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

class Block3(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.bilstm1 = m.bilstm1; self.bilstm2 = m.bilstm2
        self.dropout = m.dropout
    def forward(self, x):
        x, _ = self.bilstm1(x); x = self.dropout(x)
        x, _ = self.bilstm2(x); x = self.dropout(x)
        x = x[:, -1, :]  # last timestep, matching CUDA block 3
        return x

class Block4(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.fc1 = m.fc1; self.fc2 = m.fc2; self.relu = m.relu; self.dropout = m.dropout
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def bench_block(name, block, inp, runs=200):
    block_cpu = copy.deepcopy(block).eval()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = block_cpu(inp)
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    cpu_p50 = times[runs // 2]

    block_gpu = copy.deepcopy(block).cuda()
    inp_gpu = inp.cuda()
    for _ in range(20):
        with torch.no_grad():
            _ = block_gpu(inp_gpu)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = block_gpu(inp_gpu)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    gpu_p50 = times[runs // 2]
    return cpu_p50, gpu_p50

b1_cpu, b1_gpu = bench_block("Block1", Block1(model_cpu), torch.randn(1, 10))
b2_cpu, b2_gpu = bench_block("Block2", Block2(model_cpu), torch.randn(1, 64, 32))
b3_cpu, b3_gpu = bench_block("Block3", Block3(model_cpu), torch.randn(1, 16, 128))
b4_cpu, b4_gpu = bench_block("Block4", Block4(model_cpu), torch.randn(1, 128))

pytorch_blocks = {
    'block1': {'cpu': b1_cpu, 'gpu': b1_gpu},
    'block2': {'cpu': b2_cpu, 'gpu': b2_gpu},
    'block3': {'cpu': b3_cpu, 'gpu': b3_gpu},
    'block4': {'cpu': b4_cpu, 'gpu': b4_gpu},
}

pt_gpu_total = b1_gpu + b2_gpu + b3_gpu + b4_gpu
pt_cpu_total = b1_cpu + b2_cpu + b3_cpu + b4_cpu

# ================================================================
# Full model PyTorch benchmarks
# ================================================================
print(f"\n--- Full Model PyTorch Benchmarks ---")
runs = 200

# Reload CPU model (model was moved to GPU)
model_cpu2 = copy.deepcopy(model_cpu)

# CPU
times = []
inp_cpu = torch.randn(1, 10)
for _ in range(runs):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model_cpu2(inp_cpu)
    times.append((time.perf_counter() - start) * 1e6)
times.sort()
full_cpu = times[runs // 2]

# GPU
inp_gpu = torch.randn(1, 10).cuda()
for _ in range(20):
    with torch.no_grad():
        _ = model_gpu(inp_gpu)
torch.cuda.synchronize()
times = []
for _ in range(runs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model_gpu(inp_gpu)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1e6)
times.sort()
full_gpu = times[runs // 2]

# ================================================================
# PAPER TABLE 1: Block-Level Comparison
# ================================================================
print(f"\n{'='*80}")
print(f"TABLE 1: Block-Level Inference Latency (single sample, us)")
print(f"{'='*80}")
print(f"{'Block':<30} {'PyTorch CPU':>12} {'PyTorch GPU':>12} {'Custom CUDA':>12} {'Speedup':>10}")
print(f"{'-'*80}")

blocks_info = [
    ("Proj+Conv1+BN+ReLU", b1_cpu, b1_gpu, 61.7),
    ("Conv2+BN+ReLU+Pool", b2_cpu, b2_gpu, 87.2),
    ("BiLSTM (2 layers)", b3_cpu, b3_gpu, 973.8),
    ("Dense Head (128->5)", b4_cpu, b4_gpu, 20.1),
]

for name, cpu, gpu, cuda in blocks_info:
    speedup = gpu / cuda
    marker = "faster" if speedup > 1 else "slower"
    print(f"{name:<30} {cpu:>12.1f} {gpu:>12.1f} {cuda:>12.1f} {speedup:>8.2f}x")

print(f"{'-'*80}")
print(f"{'TOTAL (sum of blocks)':<30} {pt_cpu_total:>12.1f} {pt_gpu_total:>12.1f} {cuda_total_graphs:>12.1f} {pt_gpu_total/cuda_total_graphs:>8.2f}x")
print(f"{'Full model (end-to-end)':<30} {full_cpu:>12.1f} {full_gpu:>12.1f} {'---':>12} {'---':>10}")

# ================================================================
# PAPER TABLE 2: Framework Comparison (single sample)
# ================================================================
print(f"\n{'='*80}")
print(f"TABLE 2: Framework Comparison (single sample, us)")
print(f"{'='*80}")
print(f"{'Method':<30} {'Latency (us)':>15} {'vs PyTorch GPU':>15}")
print(f"{'-'*80}")
methods = [
    ("PyTorch CPU", full_cpu),
    ("PyTorch GPU", full_gpu),
    ("Custom CUDA FP32", cuda_total_graphs),
    ("Custom CUDA FP16", cuda_total_fp16),
    ("Custom CUDA (no graphs)", cuda_total_no_graphs),
]
for name, lat in methods:
    speedup = full_gpu / lat
    print(f"{name:<30} {lat:>15.1f} {speedup:>14.2f}x")

# ================================================================
# Save results to JSON
# ================================================================
results = {
    'hardware': _gpu_name,
    'cuda_blocks': cuda_blocks,
    'cuda_total_graphs': cuda_total_graphs,
    'cuda_total_no_graphs': cuda_total_no_graphs,
    'pytorch_blocks': {k: v for k, v in pytorch_blocks.items()},
    'pytorch_full': {'cpu': full_cpu, 'gpu': full_gpu},
    'pytorch_block_totals': {'cpu': pt_cpu_total, 'gpu': pt_gpu_total},
}

# Hardware-tagged path: permanent per-GPU record, never clobbered by a run
# on different hardware.
tagged_path = f'benchmarks/results/pipeline_benchmark_{GPU_TAG}.json'
with open(tagged_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {tagged_path}")

# Legacy fixed path: kept for backward compatibility with anything still
# reading the untagged filename -- always reflects the MOST RECENT run,
# so don't treat it as hardware-specific.
out_path = 'benchmarks/results/pipeline_benchmark.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_path}")
print("Done.")