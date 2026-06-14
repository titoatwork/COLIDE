"""
COLIDE - Energy Efficiency Benchmark
Measures GPU power draw during inference vs idle.
Reports energy per inference and throughput per watt.
"""

import sys
import os
import time
import copy
import json
import subprocess
import threading
import numpy as np
import yaml
import torch

sys.path.insert(0, '.')
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

def sample_power():
    """Read GPU power draw in watts."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        return float(result.stdout.strip())
    except:
        return 0.0

def power_monitor(readings, stop_event, interval=0.1):
    """Background thread that samples power at regular intervals."""
    while not stop_event.is_set():
        readings.append((time.time(), sample_power()))
        time.sleep(interval)

# Load model
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

model = CNNBiLSTM(config)
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu', weights_only=True))
model.eval()

model_cpu = copy.deepcopy(model)
model_gpu = model.cuda()

print("=" * 60)
print("COLIDE ENERGY EFFICIENCY BENCHMARK")
print("=" * 60)

# ================================================================
# 1. Idle power
# ================================================================
print("\n--- Idle Power ---")
time.sleep(2)
idle_readings = [sample_power() for _ in range(20)]
idle_power = np.mean(idle_readings)
print(f"Idle power: {idle_power:.2f} W")

# ================================================================
# 2. GPU inference power (single sample)
# ================================================================
print("\n--- GPU Inference (batch=1) ---")
inp = torch.randn(1, 10).cuda()

# Warmup
for _ in range(50):
    with torch.no_grad():
        _ = model_gpu(inp)
torch.cuda.synchronize()

# Monitor power during sustained inference
readings = []
stop_event = threading.Event()
monitor = threading.Thread(target=power_monitor, args=(readings, stop_event, 0.05))
monitor.start()

iters = 2000
start = time.time()
for _ in range(iters):
    with torch.no_grad():
        _ = model_gpu(inp)
torch.cuda.synchronize()
elapsed = time.time() - start

stop_event.set()
monitor.join()

powers = [r[1] for r in readings]
gpu_b1_power = np.mean(powers)
gpu_b1_latency = elapsed / iters
gpu_b1_throughput = iters / elapsed
gpu_b1_energy_per_inf = gpu_b1_power * gpu_b1_latency * 1000  # millijoules

print(f"Avg power:         {gpu_b1_power:.2f} W")
print(f"Throughput:        {gpu_b1_throughput:.0f} inf/sec")
print(f"Energy/inference:  {gpu_b1_energy_per_inf:.3f} mJ")

# ================================================================
# 3. GPU inference power (batch=128)
# ================================================================
print("\n--- GPU Inference (batch=128) ---")
inp_batch = torch.randn(128, 10).cuda()

for _ in range(50):
    with torch.no_grad():
        _ = model_gpu(inp_batch)
torch.cuda.synchronize()

readings = []
stop_event = threading.Event()
monitor = threading.Thread(target=power_monitor, args=(readings, stop_event, 0.05))
monitor.start()

iters = 500
start = time.time()
for _ in range(iters):
    with torch.no_grad():
        _ = model_gpu(inp_batch)
torch.cuda.synchronize()
elapsed = time.time() - start

stop_event.set()
monitor.join()

powers = [r[1] for r in readings]
gpu_b128_power = np.mean(powers)
gpu_b128_latency = elapsed / iters
gpu_b128_throughput = (iters * 128) / elapsed
gpu_b128_energy_per_flow = gpu_b128_power * gpu_b128_latency / 128 * 1000  # mJ per flow

print(f"Avg power:         {gpu_b128_power:.2f} W")
print(f"Throughput:        {gpu_b128_throughput:.0f} flows/sec")
print(f"Energy/flow:       {gpu_b128_energy_per_flow:.4f} mJ")

# ================================================================
# 4. CPU inference power (batch=1)
# ================================================================
print("\n--- CPU Inference (batch=1) ---")
inp_cpu = torch.randn(1, 10)

readings = []
stop_event = threading.Event()
monitor = threading.Thread(target=power_monitor, args=(readings, stop_event, 0.05))
monitor.start()

iters = 500
start = time.time()
for _ in range(iters):
    with torch.no_grad():
        _ = model_cpu(inp_cpu)
elapsed = time.time() - start

stop_event.set()
monitor.join()

powers = [r[1] for r in readings]
cpu_power = np.mean(powers)
cpu_latency = elapsed / iters
cpu_throughput = iters / elapsed
cpu_energy_per_inf = cpu_power * cpu_latency * 1000

print(f"Avg power (GPU):   {cpu_power:.2f} W (GPU idle during CPU inference)")
print(f"Throughput:        {cpu_throughput:.0f} inf/sec")

# ================================================================
# Summary table
# ================================================================
print(f"\n{'='*60}")
print(f"ENERGY EFFICIENCY SUMMARY")
print(f"{'='*60}")
print(f"{'Config':<25} {'Power(W)':>10} {'Tput(f/s)':>12} {'mJ/flow':>10}")
print(f"{'-'*60}")
print(f"{'Idle':<25} {idle_power:>10.2f} {'---':>12} {'---':>10}")
print(f"{'GPU batch=1':<25} {gpu_b1_power:>10.2f} {gpu_b1_throughput:>12.0f} {gpu_b1_energy_per_inf:>10.3f}")
print(f"{'GPU batch=128':<25} {gpu_b128_power:>10.2f} {gpu_b128_throughput:>12.0f} {gpu_b128_energy_per_flow:>10.4f}")
print(f"{'CPU batch=1':<25} {cpu_power:>10.2f} {cpu_throughput:>12.0f} {cpu_energy_per_inf:>10.3f}")

# Save
out = {
    'idle_power_w': idle_power,
    'gpu_batch1': {'power_w': gpu_b1_power, 'throughput': gpu_b1_throughput, 'mj_per_inf': gpu_b1_energy_per_inf},
    'gpu_batch128': {'power_w': gpu_b128_power, 'throughput': gpu_b128_throughput, 'mj_per_flow': gpu_b128_energy_per_flow},
    'cpu_batch1': {'power_w': cpu_power, 'throughput': cpu_throughput, 'mj_per_inf': cpu_energy_per_inf},
}
with open('benchmarks/results/energy_efficiency.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to benchmarks/results/energy_efficiency.json")