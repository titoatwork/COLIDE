"""
COLIDE — Statistical Significance Testing
20 independent trials x 1000 iterations for each framework.
Reports mean, std, 95% CI, and paired t-test p-values.

Usage:
  export LD_LIBRARY_PATH=$HOME/colide/.venv/lib/python3.12/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
  PYTHONPATH=. python scripts/benchmark_stats_v2.py
"""
import numpy as np
import time
import torch
import yaml
import sys
import json
from scipy import stats

sys.path.insert(0, 'model')
from cnn_bilstm_v3_attention import CNNBiLSTMAttention

config = yaml.safe_load(open('config/config.yaml'))
model = CNNBiLSTMAttention(config).cuda().eval()
model.load_state_dict(torch.load(
    'model/best_model_botiot_distill_focal_T5.pth',
    map_location='cuda', weights_only=True))

dummy = torch.randn(1, 10, device='cuda')
N_TRIALS = 20
N_ITERS = 1000

results = {}

# 1. Eager PyTorch GPU
print("Benchmarking Eager PyTorch GPU...")
trials = []
with torch.no_grad():
    for _ in range(500):
        model(dummy)
    torch.cuda.synchronize()
    for t_idx in range(N_TRIALS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(N_ITERS):
            model(dummy)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t) / N_ITERS * 1e6
        trials.append(us)
        print(f"  Trial {t_idx+1}: {us:.1f} us")
results['Eager PyTorch'] = np.array(trials)

# 2. torch.compile
print("\nBenchmarking torch.compile...")
compiled = torch.compile(model, mode='reduce-overhead')
trials = []
with torch.no_grad():
    for _ in range(500):
        compiled(dummy)
    torch.cuda.synchronize()
    for t_idx in range(N_TRIALS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(N_ITERS):
            compiled(dummy)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t) / N_ITERS * 1e6
        trials.append(us)
        print(f"  Trial {t_idx+1}: {us:.1f} us")
results['torch.compile'] = np.array(trials)

# 3. ONNX export for ORT benchmarks
print("\nExporting ONNX...")
torch.onnx.export(model, dummy, '/tmp/colide_stat.onnx',
                   input_names=['input'], output_names=['output'],
                   opset_version=14)
x_np = np.random.randn(1, 10).astype(np.float32)

# 4. ORT CPU
print("Benchmarking ORT CPU...")
import onnxruntime as ort
sess_cpu = ort.InferenceSession('/tmp/colide_stat.onnx',
                                 providers=['CPUExecutionProvider'])
trials = []
for _ in range(500):
    sess_cpu.run(None, {'input': x_np})
for t_idx in range(N_TRIALS):
    t = time.perf_counter()
    for _ in range(N_ITERS):
        sess_cpu.run(None, {'input': x_np})
    us = (time.perf_counter() - t) / N_ITERS * 1e6
    trials.append(us)
    print(f"  Trial {t_idx+1}: {us:.1f} us")
results['ORT CPU'] = np.array(trials)

# 5. ORT GPU
print("\nBenchmarking ORT GPU...")
sess_gpu = ort.InferenceSession('/tmp/colide_stat.onnx',
                                 providers=['CUDAExecutionProvider'])
trials = []
for _ in range(500):
    sess_gpu.run(None, {'input': x_np})
for t_idx in range(N_TRIALS):
    t = time.perf_counter()
    for _ in range(N_ITERS):
        sess_gpu.run(None, {'input': x_np})
    us = (time.perf_counter() - t) / N_ITERS * 1e6
    trials.append(us)
    print(f"  Trial {t_idx+1}: {us:.1f} us")
results['ORT GPU'] = np.array(trials)

# 6. TensorRT (may fail due to pycuda/pytorch conflict)
print("\nBenchmarking TensorRT...")
del model, compiled
torch.cuda.empty_cache()

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open('/tmp/colide_stat.onnx', 'rb') as f:
        parser.parse(f.read())
    config_trt = builder.create_builder_config()
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    print("  Building TensorRT engine...")
    engine_bytes = builder.build_serialized_network(network, config_trt)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    h_in = np.random.randn(1, 10).astype(np.float32)
    h_out = np.empty([1, 5], dtype=np.float32)
    d_in = cuda.mem_alloc(h_in.nbytes)
    d_out = cuda.mem_alloc(h_out.nbytes)
    stream = cuda.Stream()

    trials = []
    for _ in range(500):
        cuda.memcpy_htod_async(d_in, h_in, stream)
        context.execute_v2([int(d_in), int(d_out)])
        stream.synchronize()
    for t_idx in range(N_TRIALS):
        t_start = time.perf_counter()
        for _ in range(N_ITERS):
            cuda.memcpy_htod_async(d_in, h_in, stream)
            context.execute_v2([int(d_in), int(d_out)])
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
            stream.synchronize()
        us = (time.perf_counter() - t_start) / N_ITERS * 1e6
        trials.append(us)
        print(f"  Trial {t_idx+1}: {us:.1f} us")
    results['TensorRT'] = np.array(trials)
except Exception as e:
    print(f"  TensorRT failed: {e}")
    results['TensorRT'] = None

# Custom CUDA reference -- previously a bare constant (674.0) with no
# variance, forcing every significance test below to be a one-sample test
# against a fixed point rather than a real two-sample comparison. Fixed
# 2026-07-01: load the actual n=100-trial distribution for the chained
# pipeline (benchmarks/results/cuda_kernel_stats_rtx3050.json), derived as
# fused_pipeline's b124_chained + fused_block3_fp16's latency (independent
# binaries, summed the same additive way as the headline pipeline total).
# For two independent random variables, mean(A+B) = mean(A) + mean(B) and
# std(A+B) = sqrt(std(A)^2 + std(B)^2); n = min(n_A, n_B) is used as a
# conservative degrees-of-freedom estimate for the two-sample test below.
with open('benchmarks/results/cuda_kernel_stats_rtx3050.json') as f:
    cuda_stats = json.load(f)
_b124 = cuda_stats['fused_pipeline']['b124_chained_us']
_b3fp16 = cuda_stats['fused_block3_fp16']['latency_us']
custom_cuda_us = _b124['mean'] + _b3fp16['mean']
custom_cuda_std = float(np.sqrt(_b124['std'] ** 2 + _b3fp16['std'] ** 2))
custom_cuda_n = min(cuda_stats['fused_pipeline']['n_trials'],
                     cuda_stats['fused_block3_fp16']['n_trials'])

# Report
print(f"\n{'='*70}")
print(f"STATISTICAL SIGNIFICANCE REPORT ({N_TRIALS} trials x {N_ITERS} iters)")
print(f"{'='*70}")
print(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'95% CI':>20} {'vs CUDA':>10}")
print(f"{'-'*70}")

for name, trials in results.items():
    if trials is None:
        print(f"{name:<20} {'FAILED':>8}")
        continue
    mean = trials.mean()
    std = trials.std()
    ci_low = mean - 1.96 * std / np.sqrt(N_TRIALS)
    ci_high = mean + 1.96 * std / np.sqrt(N_TRIALS)
    speedup = mean / custom_cuda_us
    print(f"{name:<20} {mean:>7.1f}  {std:>7.1f}  [{ci_low:>7.1f}, {ci_high:>7.1f}]  {speedup:>7.2f}x")

print(f"{'Custom CUDA FP16':<20} {custom_cuda_us:>7.1f}  {custom_cuda_std:>7.1f}  "
      f"{'(n=' + str(custom_cuda_n) + ' trials)':>20}  {'1.00x':>10}")

print(f"\n{'='*70}")
print("TWO-SAMPLE WELCH'S T-TESTS (H0: framework mean = Custom CUDA mean)")
print("Previously a one-sample test against a fixed constant with no")
print("variance -- fixed 2026-07-01 now that Custom CUDA has its own")
print("real n=100-trial distribution instead of a bare point value.")
print(f"{'='*70}")
for name, trials in results.items():
    if trials is None:
        continue
    t_stat, p_val = stats.ttest_ind_from_stats(
        mean1=trials.mean(), std1=trials.std(ddof=1), nobs1=len(trials),
        mean2=custom_cuda_us, std2=custom_cuda_std, nobs2=custom_cuda_n,
        equal_var=False,
    )
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"{name:<20} t={t_stat:>8.2f}  p={p_val:.2e}  {sig}")

# Save
output = {}
for name, trials in results.items():
    if trials is None:
        continue
    output[name] = {
        'mean_us': float(trials.mean()),
        'std_us': float(trials.std()),
        'ci_95_low': float(trials.mean() - 1.96 * trials.std() / np.sqrt(N_TRIALS)),
        'ci_95_high': float(trials.mean() + 1.96 * trials.std() / np.sqrt(N_TRIALS)),
        'n_trials': N_TRIALS,
        'n_iters': N_ITERS,
        'trials': [float(x) for x in trials],
    }
output['Custom CUDA FP16'] = {
    'mean_us': custom_cuda_us,
    'std_us': custom_cuda_std,
    'n_trials': custom_cuda_n,
    'note': ('Derived from benchmarks/results/cuda_kernel_stats_rtx3050.json: '
             'fused_pipeline b124_chained_us + fused_block3_fp16 latency_us '
             '(independent binaries, summed mean/variance). Was a bare '
             'constant with no variance until 2026-07-01.'),
}
json.dump(output, open('benchmarks/results/statistical_significance_v2.json', 'w'), indent=2)
print(f"\nSaved to benchmarks/results/statistical_significance_v2.json")