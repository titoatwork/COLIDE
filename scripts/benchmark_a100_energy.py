#!/home/user/ibteshamulhaque/.conda/envs/colide/bin/python
"""
Measure CNN‑BiLSTM inference energy (mJ/flow) on the current GPU (intended for A100).
Uses pynvml for real‑time power draw and computes energy per flow.
Saves results to benchmarks/results/a100_energy.json.
"""

import time, json, os, sys
import numpy as np
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown

# Add model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM
import yaml

# ---------------------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------------------
config_path = "config/config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

device = torch.device("cuda")
model = CNNBiLSTM(config).to(device)
ckpt = torch.load("model/best_model_botiot_distill_focal_T5.pth", map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.eval()

# Dummy input (batch size 128 for stable power measurement)
batch_size = 128
dummy = torch.randn(batch_size, 10, device=device)

# ---------------------------------------------------------------------------
# Warm‑up
# ---------------------------------------------------------------------------
print("Warming up...")
for _ in range(100):
    with torch.no_grad():
        _ = model(dummy)
torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# Power measurement loop
# ---------------------------------------------------------------------------
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

NUM_ITER = 500
total_energy_mj = 0.0
total_time_s = 0.0

print(f"Running {NUM_ITER} inference batches...")
for _ in range(NUM_ITER):
    # Sample power before
    power_before_mW = nvmlDeviceGetPowerUsage(handle)  # milliwatts

    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    end = time.perf_counter()

    # Sample power after
    power_after_mW = nvmlDeviceGetPowerUsage(handle)

    # Use average power during the interval (approximate)
    avg_power_mW = (power_before_mW + power_after_mW) / 2.0
    elapsed_s = end - start
    energy_mj = avg_power_mW * elapsed_s   # mW * s = mJ

    total_energy_mj += energy_mj
    total_time_s += elapsed_s

nvmlShutdown()

# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
total_flows = NUM_ITER * batch_size
energy_per_flow_mj = total_energy_mj / total_flows
avg_batch_time_ms = (total_time_s / NUM_ITER) * 1000
avg_power_w = (total_energy_mj / total_time_s) / 1000  # mJ/s -> W

results = {
    "hardware": "A100 (DICC)",
    "model": "CNN-BiLSTM (distilled)",
    "batch_size": batch_size,
    "iterations": NUM_ITER,
    "energy_per_flow_mj": round(energy_per_flow_mj, 6),
    "avg_batch_time_ms": round(avg_batch_time_ms, 3),
    "avg_power_watts": round(avg_power_w, 2)
}

os.makedirs("benchmarks/results", exist_ok=True)
with open("benchmarks/results/a100_energy.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ A100 Energy Measurement Complete")
print(f"   Energy per flow: {energy_per_flow_mj:.6f} mJ")
print(f"   Avg batch time:  {avg_batch_time_ms:.3f} ms")
print(f"   Avg power:       {avg_power_w:.2f} W")
print(f"   Reference: cuML RF on A100 = 0.048 mJ/flow")