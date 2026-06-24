"""
COLIDE - torch.compile Benchmark
Compares eager PyTorch vs torch.compile vs custom CUDA.

Usage:
  PYTHONPATH=. python scripts/benchmark_torch_compile_native.py
"""
import time, json, torch, yaml, sys

sys.path.insert(0, 'model')
from cnn_bilstm_v3_attention import CNNBiLSTMAttention

print("=" * 60)
print("COLIDE torch.compile vs Custom CUDA Benchmark")
print("=" * 60)

config = yaml.safe_load(open('config/config.yaml'))
model = CNNBiLSTMAttention(config).cuda().eval()
model.load_state_dict(torch.load(
    'model/best_model_botiot_distill_focal_T5.pth',
    map_location='cuda', weights_only=True))

dummy = torch.randn(1, 10, device='cuda')
iters = 1000

# Eager baseline
with torch.no_grad():
    for _ in range(200): model(dummy)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): model(dummy)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t) / iters * 1e6

# torch.compile
compiled = torch.compile(model, mode='reduce-overhead')
with torch.no_grad():
    for _ in range(200): compiled(dummy)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): compiled(dummy)
    torch.cuda.synchronize()
    compile_us = (time.perf_counter() - t) / iters * 1e6

custom_cuda_us = 674.0  # RTX 3050, or 551.0 for V100S

print(f"\nEager PyTorch:    {eager_us:.1f} us")
print(f"torch.compile:    {compile_us:.1f} us")
print(f"Custom CUDA FP16: {custom_cuda_us:.1f} us")
print(f"")
print(f"Compile vs Eager:  {eager_us/compile_us:.2f}x")
print(f"Custom vs Compile: {compile_us/custom_cuda_us:.2f}x")
print(f"Custom vs Eager:   {eager_us/custom_cuda_us:.2f}x")

results = {
    'eager_us': float(eager_us),
    'compile_us': float(compile_us),
    'custom_cuda_us': float(custom_cuda_us),
    'compile_vs_eager': round(float(eager_us/compile_us), 2),
    'custom_vs_compile': round(float(compile_us/custom_cuda_us), 2),
    'custom_vs_eager': round(float(eager_us/custom_cuda_us), 2),
}
with open('benchmarks/results/torch_compile_native.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to benchmarks/results/torch_compile_native.json")
