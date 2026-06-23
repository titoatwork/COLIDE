#!/usr/bin/env python3
"""
Export CNN-BiLSTM to ONNX → TensorRT engine, benchmark against custom CUDA.
Compares single‑sample and batch(128) latency.

Usage (DICC):
  conda activate colide
  python benchmark_tensorrt.py

Requires: tensorrt, onnx, onnxruntime (optional), torch
"""

import time, json, os, sys, warnings
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Load model & weights
# ---------------------------------------------------------------------------
sys.path.insert(0, "model")
from cnn_bilstm_v3_attention import CNNBiLSTMAttention
import yaml
config = yaml.safe_load(open("config/config.yaml"))
model = CNNBiLSTMAttention(config).cuda()
model.load_state_dict(torch.load("model/best_model_botiot_distill_focal_T5.pth", map_location="cuda"))
model.eval()

# ---------------------------------------------------------------------------
# Export to ONNX
# ---------------------------------------------------------------------------
dummy_input = torch.randn(1, 10, device="cuda")
onnx_path = "model/model.onnx"
torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
print(f"ONNX exported to {onnx_path}")

# ---------------------------------------------------------------------------
# Build TensorRT engine
# ---------------------------------------------------------------------------
try:
    import tensorrt as trt
except ImportError:
    print("TensorRT not installed. Try: pip install tensorrt")
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(onnx_path, "rb") as f:
    parser.parse(f.read())

config_trt = builder.create_builder_config()
config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
config_trt.set_flag(trt.BuilderFlag.FP16)  # FP16 mode; remove for FP32

profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 10), (1, 10), (128, 10))
config_trt.add_optimization_profile(profile)

engine = builder.build_serialized_network(network, config_trt)
if engine is None:
    print("Failed to build TensorRT engine.")
    sys.exit(1)
with open("model/model_fp16.engine", "wb") as f:
    f.write(engine)
print("TensorRT engine saved as model/model_fp16.engine")

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
import pycuda.driver as cuda
import pycuda.autoinit
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(open("model/model_fp16.engine", "rb").read())
context = engine.create_execution_context()

# Allocate buffers
d_input = cuda.mem_alloc(128 * 10 * 4)  # max batch 128, float32
d_output = cuda.mem_alloc(128 * 5 * 4)
bindings = [int(d_input), int(d_output)]

def infer_trt(batch_size):
    context.set_binding_shape(0, (batch_size, 10))
    input_data = np.random.randn(batch_size, 10).astype(np.float32)
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(d_output, np.empty((batch_size, 5), dtype=np.float32))

# Warmup
for _ in range(10):
    infer_trt(1)
    infer_trt(128)

# Single sample
start = time.perf_counter()
for _ in range(1000):
    infer_trt(1)
single_ms = (time.perf_counter() - start) / 1000 * 1000

# Batch 128
start = time.perf_counter()
for _ in range(100):
    infer_trt(128)
batch_ms = (time.perf_counter() - start) / 100 * 1000

print(f"TensorRT FP16 single: {single_ms:.3f} µs, batch(128): {batch_ms:.3f} ms")
print("Custom CUDA (ref): 674 µs single (chained FP16)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
results = {
    "tensorrt_fp16": {
        "single_us": single_ms * 1000,
        "batch128_ms": batch_ms
    },
    "custom_cuda_fp16_reference_us": 674
}
os.makedirs("benchmarks/results", exist_ok=True)
with open("benchmarks/results/tensorrt.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to benchmarks/results/tensorrt.json")