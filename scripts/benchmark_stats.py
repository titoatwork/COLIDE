"""
COLIDE - Statistical Confidence Intervals
Runs PyTorch benchmarks 10 times each and reports mean, std, 95% CI.
"""

import sys
import time
import copy
import json
import numpy as np
import yaml
import torch
from scipy import stats

sys.path.insert(0, '.')
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

model = CNNBiLSTM(config)
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu', weights_only=True))
model.eval()
model_cpu = copy.deepcopy(model)
model_gpu = model.cuda()

print("=" * 70)
print("COLIDE STATISTICAL CONFIDENCE INTERVALS")
print("=" * 70)

NUM_TRIALS = 10
RUNS_PER_TRIAL = 200

def run_trial_cpu(model, inp, runs=RUNS_PER_TRIAL):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(inp)
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    return times[runs // 2]

def run_trial_gpu(model, inp, runs=RUNS_PER_TRIAL):
    for _ in range(20):
        with torch.no_grad():
            _ = model(inp)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(inp)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    return times[runs // 2]

def compute_stats(values, name):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=std/np.sqrt(len(values)))
    print(f"{name:<30} {mean:>10.1f} +/- {(ci[1]-mean):>6.1f} us  (std={std:.1f})")
    return {'mean': mean, 'std': std, 'ci_low': ci[0], 'ci_high': ci[1], 'values': list(values)}

# Full model benchmarks
print(f"\n--- Running {NUM_TRIALS} trials x {RUNS_PER_TRIAL} runs each ---\n")

inp_cpu = torch.randn(1, 10)
inp_gpu = torch.randn(1, 10).cuda()

cpu_trials = []
gpu_trials = []

for trial in range(NUM_TRIALS):
    cpu_p50 = run_trial_cpu(model_cpu, inp_cpu)
    gpu_p50 = run_trial_gpu(model_gpu, inp_gpu)
    cpu_trials.append(cpu_p50)
    gpu_trials.append(gpu_p50)
    print(f"  Trial {trial+1:>2}/{NUM_TRIALS}: CPU={cpu_p50:.1f} us, GPU={gpu_p50:.1f} us")

print(f"\n{'='*70}")
print("FULL MODEL RESULTS (95% Confidence Intervals)")
print(f"{'='*70}")
cpu_stats = compute_stats(cpu_trials, "PyTorch CPU (single sample)")
gpu_stats = compute_stats(gpu_trials, "PyTorch GPU (single sample)")

# Batch benchmarks
print(f"\n{'='*70}")
print("BATCH BENCHMARKS (95% Confidence Intervals)")
print(f"{'='*70}")

batch_results = {}
for bs in [1, 32, 128, 256]:
    inp_b = torch.randn(bs, 10).cuda()
    trials = []
    for _ in range(NUM_TRIALS):
        trials.append(run_trial_gpu(model_gpu, inp_b))
    batch_results[bs] = compute_stats(trials, f"GPU batch={bs}")

# Custom CUDA block numbers (from individual benchmarks, report as-is)
print(f"\n{'='*70}")
print("CUSTOM CUDA BLOCKS (from individual benchmark runs)")
print(f"{'='*70}")
print(f"{'Block':<30} {'Reported (us)':>15} {'Note':>25}")
print(f"{'-'*70}")
print(f"{'Block 1 (Conv)':<30} {'61.7':>15} {'100 iters, p50':>25}")
print(f"{'Block 2 (Conv)':<30} {'87.2':>15} {'100 iters, p50':>25}")
print(f"{'Block 3 (BiLSTM)':<30} {'973.8':>15} {'100 iters, CUDA Graphs':>25}")
print(f"{'Block 4 (Dense)':<30} {'20.1':>15} {'10000 iters, p50':>25}")
print(f"{'TOTAL':<30} {'1142.8':>15} {'sum of blocks':>25}")

# Save
results = {
    'num_trials': NUM_TRIALS,
    'runs_per_trial': RUNS_PER_TRIAL,
    'pytorch_cpu_single': cpu_stats,
    'pytorch_gpu_single': gpu_stats,
    'batch_results': {str(k): v for k, v in batch_results.items()},
}
with open('benchmarks/results/statistical_confidence.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nSaved to benchmarks/results/statistical_confidence.json")
print("Done.")