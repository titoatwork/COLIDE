"""
COLIDE - Block-Level PyTorch Benchmark
Measures latency for each block separately.
Used to set targets for custom CUDA kernels.
"""

import torch
import numpy as np
import time
import yaml
from model.cnn_bilstm import CNNBiLSTM

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

model = CNNBiLSTM(config)
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu', weights_only=True))
model.eval()

def benchmark(name, module, input_tensor, runs=200):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = module(input_tensor)
        times.append((time.perf_counter() - start) * 1000)
    times = sorted(times)
    cpu_p50 = times[runs//2]

    module_gpu = module.cuda()
    input_gpu = input_tensor.cuda()
    for _ in range(20):
        with torch.no_grad():
            _ = module_gpu(input_gpu)
    torch.cuda.synchronize()

    times_gpu = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = module_gpu(input_gpu)
        torch.cuda.synchronize()
        times_gpu.append((time.perf_counter() - start) * 1000)
    times_gpu = sorted(times_gpu)
    gpu_p50 = times_gpu[runs//2]

    print(f"{name}")
    print(f"  CPU p50: {cpu_p50:.4f} ms ({cpu_p50*1000:.1f} us)")
    print(f"  GPU p50: {gpu_p50:.4f} ms ({gpu_p50*1000:.1f} us)")
    print()

print("=== COLIDE Block-Level Benchmark ===\n")

# Block 1: Projection + Conv1 + BN1 + ReLU
class Block1(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.proj = m.input_projection
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
    def forward(self, x):
        x = self.proj(x)
        x = x.view(x.size(0), 2, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

# Block 2: Conv2 + BN2 + ReLU + MaxPool
class Block2(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv2 = m.conv2
        self.bn2 = m.bn2
        self.relu = m.relu
        self.pool = m.pool
    def forward(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Block 3: BiLSTM layers
class Block3(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.bilstm1 = m.bilstm1
        self.bilstm2 = m.bilstm2
        self.dropout = m.dropout
    def forward(self, x):
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        return x

# Block 4: Dense head
class Block4(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.fc1 = m.fc1
        self.fc2 = m.fc2
        self.relu = m.relu
        self.dropout = m.dropout
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

benchmark("BLOCK 1 (Proj+Conv1+BN+ReLU)", Block1(model), torch.randn(1, 10))
benchmark("BLOCK 2 (Conv2+BN+ReLU+Pool)", Block2(model), torch.randn(1, 64, 32))
benchmark("BLOCK 3 (BiLSTM x2)", Block3(model), torch.randn(1, 16, 128))
benchmark("BLOCK 4 (Dense head)", Block4(model), torch.randn(1, 128))
