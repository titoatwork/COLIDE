"""
COLIDE - Streaming Throughput Benchmark
Simulates continuous network traffic at increasing rates.
Measures latency vs throughput and finds saturation point.
Tests both single-sample and batched inference modes.
"""

import sys
import time
import copy
import json
import threading
import queue
import numpy as np
import yaml
import torch

sys.path.insert(0, '.')
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

model = CNNBiLSTM(config)
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu', weights_only=True))
model.eval()

# Load real test data for realistic inference
X_test = np.load('data/processed/X_test.npy')
print(f"Loaded {len(X_test)} test flows")

print("=" * 70)
print("COLIDE STREAMING THROUGHPUT BENCHMARK")
print("=" * 70)

# ================================================================
# Mode 1: GPU Batched Inference (throughput-optimized)
# ================================================================
print("\n--- Mode 1: GPU Batched Inference ---")
model_gpu = model.cuda()

def benchmark_gpu_batched(offered_rate, duration_sec=5, batch_size=128):
    """Simulate traffic at given rate, process in batches."""
    interval = 1.0 / offered_rate  # seconds between flows
    total_flows = int(offered_rate * duration_sec)
    
    # Pre-generate all inputs
    indices = np.random.randint(0, len(X_test), total_flows)
    all_inputs = torch.tensor(X_test[indices], dtype=torch.float32)
    
    latencies = []
    processed = 0
    batch_buffer = []
    
    start_time = time.perf_counter()
    
    for i in range(total_flows):
        arrival_time = time.perf_counter()
        batch_buffer.append(all_inputs[i])
        
        # Process when batch is full or last flow
        if len(batch_buffer) >= batch_size or i == total_flows - 1:
            batch = torch.stack(batch_buffer).cuda()
            
            with torch.no_grad():
                _ = model_gpu(batch)
            torch.cuda.synchronize()
            
            done_time = time.perf_counter()
            batch_latency = (done_time - arrival_time) * 1e6  # us
            per_flow_latency = batch_latency / len(batch_buffer)
            
            for _ in batch_buffer:
                latencies.append(per_flow_latency)
            
            processed += len(batch_buffer)
            batch_buffer = []
    
    elapsed = time.perf_counter() - start_time
    actual_throughput = processed / elapsed
    
    latencies.sort()
    n = len(latencies)
    return {
        'offered_rate': offered_rate,
        'actual_throughput': actual_throughput,
        'processed': processed,
        'elapsed_sec': elapsed,
        'p50_us': latencies[n // 2] if n > 0 else 0,
        'p95_us': latencies[int(n * 0.95)] if n > 0 else 0,
        'p99_us': latencies[int(n * 0.99)] if n > 0 else 0,
        'dropped': total_flows - processed,
    }

# ================================================================
# Mode 2: CPU Single-Sample Inference (latency-optimized)
# ================================================================
print("--- Mode 2: CPU Single-Sample Inference ---")
model_cpu = copy.deepcopy(model).cpu()

def benchmark_cpu_single(offered_rate, duration_sec=5):
    """Simulate traffic at given rate, process one at a time on CPU."""
    total_flows = int(offered_rate * duration_sec)
    indices = np.random.randint(0, len(X_test), total_flows)
    all_inputs = torch.tensor(X_test[indices], dtype=torch.float32)
    
    latencies = []
    processed = 0
    
    start_time = time.perf_counter()
    
    for i in range(total_flows):
        flow_start = time.perf_counter()
        
        with torch.no_grad():
            _ = model_cpu(all_inputs[i:i+1])
        
        flow_end = time.perf_counter()
        latencies.append((flow_end - flow_start) * 1e6)
        processed += 1
        
        # Check if we're falling behind
        expected_time = (i + 1) / offered_rate
        actual_time = flow_end - start_time
        if actual_time > expected_time + 1.0:  # more than 1 sec behind
            break  # saturated
    
    elapsed = time.perf_counter() - start_time
    actual_throughput = processed / elapsed
    
    latencies.sort()
    n = len(latencies)
    return {
        'offered_rate': offered_rate,
        'actual_throughput': actual_throughput,
        'processed': processed,
        'elapsed_sec': elapsed,
        'p50_us': latencies[n // 2] if n > 0 else 0,
        'p95_us': latencies[int(n * 0.95)] if n > 0 else 0,
        'p99_us': latencies[int(n * 0.99)] if n > 0 else 0,
        'dropped': total_flows - processed,
    }

# ================================================================
# Mode 3: GPU Single-Sample Inference
# ================================================================
def benchmark_gpu_single(offered_rate, duration_sec=5):
    """Simulate traffic at given rate, process one at a time on GPU."""
    total_flows = int(offered_rate * duration_sec)
    indices = np.random.randint(0, len(X_test), total_flows)
    all_inputs = torch.tensor(X_test[indices], dtype=torch.float32).cuda()
    
    latencies = []
    processed = 0
    
    start_time = time.perf_counter()
    
    for i in range(total_flows):
        torch.cuda.synchronize()
        flow_start = time.perf_counter()
        
        with torch.no_grad():
            _ = model_gpu(all_inputs[i:i+1])
        torch.cuda.synchronize()
        
        flow_end = time.perf_counter()
        latencies.append((flow_end - flow_start) * 1e6)
        processed += 1
        
        expected_time = (i + 1) / offered_rate
        actual_time = flow_end - start_time
        if actual_time > expected_time + 1.0:
            break
    
    elapsed = time.perf_counter() - start_time
    actual_throughput = processed / elapsed
    
    latencies.sort()
    n = len(latencies)
    return {
        'offered_rate': offered_rate,
        'actual_throughput': actual_throughput,
        'processed': processed,
        'elapsed_sec': elapsed,
        'p50_us': latencies[n // 2] if n > 0 else 0,
        'p95_us': latencies[int(n * 0.95)] if n > 0 else 0,
        'p99_us': latencies[int(n * 0.99)] if n > 0 else 0,
        'dropped': total_flows - processed,
    }

# ================================================================
# Run benchmarks at increasing rates
# ================================================================
test_rates = [100, 500, 1000, 5000, 10000, 25000, 50000]

# Warmup GPU
dummy = torch.randn(128, 10).cuda()
for _ in range(20):
    with torch.no_grad():
        _ = model_gpu(dummy)
torch.cuda.synchronize()

print(f"\nTesting at rates: {test_rates} flows/sec")
print(f"Duration per test: 5 seconds\n")

gpu_batched_results = []
gpu_single_results = []
cpu_single_results = []

for rate in test_rates:
    print(f"Rate: {rate:>6} flows/sec ... ", end='', flush=True)
    
    # GPU batched
    r = benchmark_gpu_batched(rate, duration_sec=5, batch_size=128)
    gpu_batched_results.append(r)
    
    # GPU single (skip very high rates, too slow)
    if rate <= 5000:
        r2 = benchmark_gpu_single(rate, duration_sec=3)
        gpu_single_results.append(r2)
    
    # CPU single (skip very high rates)
    if rate <= 5000:
        r3 = benchmark_cpu_single(rate, duration_sec=3)
        cpu_single_results.append(r3)
    
    print("done")

# ================================================================
# Results Table 1: GPU Batched (throughput-optimized)
# ================================================================
print(f"\n{'='*80}")
print("GPU BATCHED INFERENCE (batch=128, throughput-optimized)")
print(f"{'='*80}")
print(f"{'Offered':>10} {'Achieved':>12} {'Utilization':>12} {'p50(us)':>10} {'p95(us)':>10} {'p99(us)':>10}")
print(f"{'-'*80}")
for r in gpu_batched_results:
    util = min(r['actual_throughput'] / r['offered_rate'] * 100, 100)
    print(f"{r['offered_rate']:>10,} {r['actual_throughput']:>12,.0f} {util:>11.1f}% {r['p50_us']:>10.0f} {r['p95_us']:>10.0f} {r['p99_us']:>10.0f}")

# Find saturation point
for i, r in enumerate(gpu_batched_results):
    util = r['actual_throughput'] / r['offered_rate']
    if util < 0.95:
        print(f"\nSaturation point: ~{gpu_batched_results[i-1]['offered_rate']:,} flows/sec (GPU batched)")
        break
else:
    print(f"\nNo saturation detected up to {test_rates[-1]:,} flows/sec")

# ================================================================
# Results Table 2: GPU Single (latency-optimized)
# ================================================================
if gpu_single_results:
    print(f"\n{'='*80}")
    print("GPU SINGLE-SAMPLE INFERENCE (latency-optimized)")
    print(f"{'='*80}")
    print(f"{'Offered':>10} {'Achieved':>12} {'Utilization':>12} {'p50(us)':>10} {'p95(us)':>10}")
    print(f"{'-'*70}")
    for r in gpu_single_results:
        util = min(r['actual_throughput'] / r['offered_rate'] * 100, 100)
        print(f"{r['offered_rate']:>10,} {r['actual_throughput']:>12,.0f} {util:>11.1f}% {r['p50_us']:>10.0f} {r['p95_us']:>10.0f}")

# ================================================================
# Results Table 3: CPU Single
# ================================================================
if cpu_single_results:
    print(f"\n{'='*80}")
    print("CPU SINGLE-SAMPLE INFERENCE")
    print(f"{'='*80}")
    print(f"{'Offered':>10} {'Achieved':>12} {'Utilization':>12} {'p50(us)':>10} {'p95(us)':>10}")
    print(f"{'-'*70}")
    for r in cpu_single_results:
        util = min(r['actual_throughput'] / r['offered_rate'] * 100, 100)
        print(f"{r['offered_rate']:>10,} {r['actual_throughput']:>12,.0f} {util:>11.1f}% {r['p50_us']:>10.0f} {r['p95_us']:>10.0f}")

# ================================================================
# Summary
# ================================================================
print(f"\n{'='*80}")
print("DEPLOYMENT SUMMARY")
print(f"{'='*80}")

# Max sustained throughput for each mode
gpu_batch_max = max(r['actual_throughput'] for r in gpu_batched_results)
gpu_single_max = max(r['actual_throughput'] for r in gpu_single_results) if gpu_single_results else 0
cpu_single_max = max(r['actual_throughput'] for r in cpu_single_results) if cpu_single_results else 0

print(f"Max sustained throughput:")
print(f"  GPU batched (batch=128): {gpu_batch_max:>10,.0f} flows/sec")
if gpu_single_max > 0:
    print(f"  GPU single-sample:       {gpu_single_max:>10,.0f} flows/sec")
if cpu_single_max > 0:
    print(f"  CPU single-sample:       {cpu_single_max:>10,.0f} flows/sec")
print(f"\nGPU batched is {gpu_batch_max/cpu_single_max:.1f}x faster than CPU single")

# Save results
results = {
    'gpu_batched': gpu_batched_results,
    'gpu_single': gpu_single_results,
    'cpu_single': cpu_single_results,
    'max_throughput': {
        'gpu_batched': gpu_batch_max,
        'gpu_single': gpu_single_max,
        'cpu_single': cpu_single_max,
    }
}
out_path = 'benchmarks/results/streaming_throughput.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")