#!/usr/bin/env python3
"""
Benchmark GPU-accelerated Random Forest on BoT-IoT using RAPIDS cuML FIL.
Compares throughput (flows/sec) and latency against CPU sklearn RF.
Measures GPU power draw to compute mJ/flow.

Usage (DICC):
  conda activate colide
  python benchmark_cuml_rf.py

Expected output:
  - Latency & throughput for CPU RF vs. GPU RF (single + batch)
  - Energy comparison (mJ/flow) if power measurement is available
  - Saved JSON: benchmarks/results/cuml_rf.json
"""

import time, json, os, sys, warnings
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
X_test = np.load("data/processed/X_test.npy")   # (733705, 10) float32
y_test = np.load("data/processed/y_test.npy")   # (733705,) int
print(f"Loaded test set: {X_test.shape}, {y_test.shape}")

# ---------------------------------------------------------------------------
# Train (or load) a 200‑tree Random Forest – same as in distillation
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

print("Training 200‑tree Random Forest on GPU (cuML) …")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
# We need training data. Load from CSV or use the same pipeline.
# For a fair comparison we must use the EXACT same training data as distillation.
# Since we don't have the pre‑SMOTE training set readily saved, we'll retrain quickly.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df_train = pd.read_csv("data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv", sep=',')
feature_cols = ['N_IN_Conn_P_DstIP','N_IN_Conn_P_SrcIP','drate','max','mean','min','seq','srate','state_number','stddev']
X_tr = df_train[feature_cols].values.astype(np.float32)
le = LabelEncoder()
y_tr = le.fit_transform(df_train['category'])
rf.fit(X_tr, y_tr)
print(f"CPU RF trained. Macro-F1 on test: {f1_score(y_test, rf.predict(X_test), average='macro'):.4f}")

# ---------------------------------------------------------------------------
# CPU baseline
# ---------------------------------------------------------------------------
def benchmark_cpu_rf(X, y, batch_size=1):
    """Measure single‑sample and batch inference on CPU."""
    # single
    start = time.perf_counter()
    for i in range(100):
        rf.predict(X[:batch_size])
    single_ms = (time.perf_counter() - start) / 100 * 1000

    # batch
    batch = X[:batch_size] if batch_size > 1 else X[:1]
    start = time.perf_counter()
    for _ in range(100):
        rf.predict(batch)
    batch_ms = (time.perf_counter() - start) / 100 * 1000

    return single_ms, batch_ms

cpu_single, cpu_batch = benchmark_cpu_rf(X_test, y_test, 1)
print(f"CPU RF single: {cpu_single:.3f} ms, batch(128): {benchmark_cpu_rf(X_test, y_test, 128)[1]:.3f} ms")

# ---------------------------------------------------------------------------
# GPU Random Forest via RAPIDS cuML FIL
# ---------------------------------------------------------------------------
try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.fil import ForestInference
except ImportError:
    print("cuML not installed. Install with: conda install -c rapidsai -c conda-forge cuml")
    sys.exit(1)

# Train cuML RF on GPU (use same parameters)
print("Training cuML RandomForest on GPU …")
cu_rf = cuRF(n_estimators=200, random_state=42)
cu_rf.fit(X_tr.astype(np.float32), y_tr.astype(np.int32))

# Convert to FIL model
print("Converting to FIL model …")
fil_model = ForestInference.load_from_sklearn(cu_rf, output_class=True)

# Benchmark GPU inference
def benchmark_gpu_rf(fil_model, X, batch_size=1):
    # single
    start = time.perf_counter()
    for i in range(100):
        fil_model.predict(X[:batch_size].astype(np.float32))
    single_ms = (time.perf_counter() - start) / 100 * 1000

    # batch
    batch = X[:batch_size].astype(np.float32)
    start = time.perf_counter()
    for _ in range(100):
        fil_model.predict(batch)
    batch_ms = (time.perf_counter() - start) / 100 * 1000
    return single_ms, batch_ms

gpu_single, gpu_batch128 = benchmark_gpu_rf(fil_model, X_test, 1)
gpu_batch128_time, _ = benchmark_gpu_rf(fil_model, X_test, 128)
print(f"GPU RF single: {gpu_single:.3f} ms, batch(128): {gpu_batch128_time:.3f} ms")

# ---------------------------------------------------------------------------
# Throughput (flows/sec)
# ---------------------------------------------------------------------------
def throughput(batch_time_ms, batch_size):
    return batch_size / (batch_time_ms / 1000) if batch_time_ms > 0 else float('inf')

cpu_throughput = throughput(cpu_batch, 128)
gpu_throughput = throughput(gpu_batch128_time, 128)
print(f"CPU RF throughput: {cpu_throughput:.0f} flows/sec")
print(f"GPU RF throughput: {gpu_throughput:.0f} flows/sec")

# ---------------------------------------------------------------------------
# Power measurement (optional)
# ---------------------------------------------------------------------------
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    power_watts = nvmlDeviceGetPowerUsage(handle) / 1000.0  # milliwatts to watts
    nvmlShutdown()
    energy_per_flow = (power_watts * (gpu_batch128_time / 1000)) / 128 * 1000  # mJ
    print(f"GPU power: {power_watts:.1f} W, energy per flow: {energy_per_flow:.3f} mJ")
except:
    energy_per_flow = None
    print("Power measurement not available (pynvml missing or no GPU access).")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "cpu_rf": {
        "single_ms": cpu_single,
        "batch128_ms": cpu_batch,
        "throughput_flows_sec": cpu_throughput
    },
    "gpu_rf": {
        "single_ms": gpu_single,
        "batch128_ms": gpu_batch128_time,
        "throughput_flows_sec": gpu_throughput,
        "energy_mj_per_flow": energy_per_flow
    }
}
os.makedirs("benchmarks/results", exist_ok=True)
with open("benchmarks/results/cuml_rf.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to benchmarks/results/cuml_rf.json")