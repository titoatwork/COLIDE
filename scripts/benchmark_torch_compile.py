#!/usr/bin/env python3
"""
Benchmark PyTorch 2 torch.compile + CUDA graphs vs. eager mode.
Compares single‑sample latency with custom CUDA kernel.

Usage:
  conda activate colide
  python benchmark_torch_compile.py
"""

import time, json, os, sys
import torch
import numpy as np
sys.path.insert(0, "model")
from cnn_bilstm_v3_attention import CNNBiLSTMAttention
import yaml

config = yaml.safe_load(open("config/config.yaml"))
model = CNNBiLSTMAttention(config).cuda()
model.load_state_dict(torch.load("model/best_model_botiot_distill_focal_T5.pth", map_location="cuda"))
model.eval()

# Eager baseline
dummy = torch.randn(1, 10, device="cuda")
with torch.no_grad():
    # warmup
    for _ in range(10): model(dummy)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        model(dummy)
    torch.cuda.synchronize()
    eager_single_us = (time.perf_counter() - start) / 1000 * 1e6

print(f"Eager PyTorch single: {eager_single_us:.1f} µs")

# torch.compile + CUDA graph
compiled = torch.compile(model, mode="reduce-overhead")
with torch.no_grad():
    # warmup compile
    for _ in range(10): compiled(dummy)
    torch.cuda.synchronize()
    # capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = compiled(dummy)
    # benchmark
    start = time.perf_counter()
    for _ in range(1000):
        g.replay()
    torch.cuda.synchronize()
    compile_single_us = (time.perf_counter() - start) / 1000 * 1e6

print(f"torch.compile + CUDA graph single: {compile_single_us:.1f} µs")
print("Custom CUDA (ref): 674 µs single (chained FP16)")

results = {
    "pytorch_eager_single_us": eager_single_us,
    "torch_compile_cuda_graph_single_us": compile_single_us,
    "custom_cuda_reference_us": 674
}
os.makedirs("benchmarks/results", exist_ok=True)
with open("benchmarks/results/torch_compile.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to benchmarks/results/torch_compile.json")