#!/home/user/ibteshamulhaque/.conda/envs/colide/bin/python
"""
Measure single‑sample GPU inference latency for the distilled MLP.
Saves result to benchmarks/results/mlp_latency.json.
"""

import time, json, os, sys
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Define MLP (same architecture as in ablation training)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=10, num_classes=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
device = torch.device("cuda")
model = MLP().to(device)
ckpt = torch.load("model/best_model_botiot_mlp_mlp.pth", map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.eval()

# ---------------------------------------------------------------------------
# Benchmark single‑sample latency
# ---------------------------------------------------------------------------
dummy = torch.randn(1, 10, device=device)
WARMUP = 100
ITERS = 5000

# Warm‑up
for _ in range(WARMUP):
    with torch.no_grad():
        _ = model(dummy)
torch.cuda.synchronize()

# Timing with CUDA events for accuracy
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings_us = []

with torch.no_grad():
    for _ in range(ITERS):
        starter.record()
        _ = model(dummy)
        ender.record()
        torch.cuda.synchronize()
        timings_us.append(starter.elapsed_time(ender) * 1000)  # ms -> µs

avg_us = sum(timings_us) / len(timings_us)
p50_us = sorted(timings_us)[len(timings_us)//2]
p95_us = sorted(timings_us)[int(len(timings_us)*0.95)]
p99_us = sorted(timings_us)[int(len(timings_us)*0.99)]

results = {
    "model": "MLP (distilled, 400K params)",
    "hardware": "A100 (DICC) — update if different",
    "avg_latency_us": round(avg_us, 2),
    "p50_us": round(p50_us, 2),
    "p95_us": round(p95_us, 2),
    "p99_us": round(p99_us, 2),
    "comparison_cnn_bilstm_us": 674,  # RTX 3050; update to A100 pipeline latency if available
    "note": "Compare with custom CUDA CNN-BiLSTM latency on same GPU (e.g., A100 chained: 592 us)"
}

os.makedirs("benchmarks/results", exist_ok=True)
with open("benchmarks/results/mlp_latency.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ MLP Latency Measurement Complete")
print(f"   Avg: {avg_us:.2f} µs  |  p50: {p50_us:.2f} µs  |  p95: {p95_us:.2f} µs  |  p99: {p99_us:.2f} µs")
print(f"   CNN-BiLSTM reference: 674 µs (RTX 3050) / 592 µs (A100 chained)")
