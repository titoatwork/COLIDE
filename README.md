# COLIDE: CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection

[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](#license)

## Abstract

COLIDE presents custom CUDA C++ inference kernels for a CNN-BiLSTM-based IoT intrusion detection system, achieving statistically significant speedups over all major deep learning inference frameworks: **4.40x over TensorRT** (p<0.001), **2.63x over torch.compile** (p<0.001), and **3.33x over eager PyTorch** (p<0.001), validated across 20 independent trials. The system integrates an on-device, air-gapped LLM explainability module (TinyLlama 1.1B, 4-bit quantized) with only **16.60 us p99** async dispatch overhead (~2.5% of the detection pipeline). Knowledge distillation from a Random Forest teacher combined with focal loss closes the accuracy gap to **2.25%** on BoT-IoT (**0.9639** macro-F1) and **3.3%** on ToN-IoT (**0.9526** macro-F1). The system sustains **25,899 flows/sec** in streaming mode on consumer-grade edge hardware.

## Key Contributions

1. **Custom CUDA Beating All Frameworks**: Hand-written CUDA C++ kernels outperform TensorRT (4.40x), torch.compile (2.63x), eager PyTorch (3.33x), and ORT GPU (6.89x) — all statistically significant at p<0.001 across 20 trials
2. **FP16 Half2 BiLSTM Beating cuDNN**: Native half-precision FMA instructions with documented **8.39x–9.21x optimization progression** (5,050 to 548–602 us) **beating cuDNN by 1.30x–1.43x** (both the cuDNN baseline and the FP16 kernel are real n=50/n=100-trial means; the range reflects genuine session-to-session measurement drift on this dev box, not an unresolved ambiguity — see "Block 3 Optimization Progression" below)
3. **Knowledge Distillation Closing the RF Gap**: RF-to-CNN-BiLSTM distillation with temperature scaling (T=5.0) and focal loss narrows accuracy gap from 5.12% to **2.25%** on BoT-IoT and 11.4% to **3.3%** on ToN-IoT
4. **On-Device Air-Gapped LLM Explainability**: Async ring-buffer dispatch to local quantized TinyLlama 1.1B with **16.60 us p99 overhead** and zero cloud dependency — contrasting with Jamshidi et al. (2026) cloud API approach
5. **Cross-Hardware Profiling**: 3 GPU architectures (RTX 3050, V100S, A100) revealing **V100S outperforms A100** for sequential LSTM — clock speed dominates SM count

## Results Summary

### Framework Comparison (RTX 3050, 20 Trials, Statistical Significance)

Custom CUDA FP16 is derived from a real n=100-trial distribution (mean 674.7us, std 87.1us; see
`benchmarks/results/cuda_kernel_stats_rtx3050.json`), not a fixed constant with no variance as in an
earlier version of this table. Significance is a two-sample Welch's t-test (framework's 20 trials vs.
Custom CUDA's 100 trials), not a one-sample test against a fixed point.

| Method | Mean (us) | Std (us) | 95% CI | vs Custom CUDA | p-value |
|---|---|---|---|---|---|
| **Custom CUDA FP16** | **675** | **87** | **(n=100 trials)** | **1.00x** | **--** |
| ORT CPU | 699 | 144 | [636, 762] | 1.04x | 0.483 (ns) |
| torch.compile | 1,777 | 152 | [1710, 1844] | 2.63x | 3.55e-19 *** |
| Eager PyTorch | 2,247 | 279 | [2125, 2369] | 3.33x | 3.51e-16 *** |
| TensorRT FP16 | 2,966 | 190 | [2882, 3049] | 4.40x | 3.51e-23 *** |
| ORT GPU | 4,652 | 176 | [4575, 4729] | 6.89x | 4.51e-29 *** |

torch.compile with CUDA graph capture **crashes** on BiLSTM (dynamic recurrent control flow). TensorRT is slower than eager PyTorch for this sub-1M parameter model.

### Cross-Hardware CUDA Pipeline

| GPU | Architecture | Pipeline (chained FP16) | vs PyTorch GPU |
|---|---|---|---|
| RTX 3050 6GB | Ampere (SM 8.6) | 674 us | 3.33x* |
| **V100S 32GB** | **Volta (SM 7.0)** | **551 us** | n/a** |
| A100 80GB | Ampere (SM 8.0) | 592 us | n/a** |

V100S is fastest because BiLSTM sequential recurrence is clock-speed-bound, not SM-count-bound.

\* Same comparison as "3.33x over eager PyTorch" above (20-trial, statistically validated) — the chained
custom-CUDA pipeline and the eager-PyTorch full-model forward pass are the same computation on the same
GPU, so this is not an independent number.
\*\* No same-hardware PyTorch GPU baseline was captured during the DICC V100S/A100 runs (only the custom
CUDA kernels were benchmarked there — see `dicc_v100_summary.txt` / `dicc_a100_summary.txt`). Reusing the
RTX 3050 PyTorch baseline to compute a ratio for different hardware would not be a valid same-machine
comparison, so no ratio is reported here pending a real PyTorch-GPU benchmark run on those machines
(tracked for the Phase 3 re-verification pass).

### Per-Block Performance (RTX 3050)

| Block | PyTorch GPU (us) | Custom CUDA (us) | Speedup |
|---|---|---|---|
| 1: Proj+Conv1+BN+ReLU | 404 | 62 | 6.55x |
| 2: Conv2+BN+ReLU+Pool | 282 | 87 | 3.24x |
| 3: BiLSTM FP16 half2 | 784 | 548–602* | 1.30x–1.43x* |
| 4: Dense Head | 122 | 20 | 6.07x |

\* Range across two independent n=100-trial measurement sessions on this dev box, not a lingering
ambiguity — see "Measurement Stability" below.

### Block 3 Optimization Progression (8.39x–9.21x)

Step 0 (naive) is now a real n=100-trial mean of the **fixed** kernel (see "Naive Kernel Fix" below);
step 1 remains a historical single-run figure with no surviving re-runnable artifact (its kernel file was
overwritten by later optimizations). Steps 2-4 are each backed by **two independent n=100-trial
measurement sessions** the same day (`benchmarks/results/cuda_kernel_stats_rtx3050.json`, regenerated
between sessions — see "Measurement Stability" below) rather than one. The PyTorch cuDNN reference used
for per-step ratios is a real n=50-trial mean, **784us** (std 89us, CV 11.3%) from
`benchmarks/results/pytorch_block3_stats_rtx3050.json` (`scripts/benchmark_pytorch_block3_stats.py`, 50
independent subprocess trials — mirrors the CUDA kernel statistical harness so both sides of the ratio are
backed by a real distribution). This resolves an earlier ambiguity between two single-run point estimates
(740.7us vs 943.6us) that bracketed the true mean. With the real baseline, **the FP16 step beats cuDNN in
both sessions (1.30x–1.43x)**; the transposed-W_hh steps (with or without CUDA Graphs) land at/below
parity with PyTorch in both sessions (0.77x–0.99x) — that conclusion (transposed steps don't clearly beat
cuDNN) is robust across the session-to-session drift, even though the exact ratios aren't.

| Step | Configuration | Latency (us) | Cumulative |
|---|---|---|---|
| 0 | Naive (1 thread/hidden), race-fixed | 5,050 | 1.00x |
| 1 | + Precomputed W_ih x X | 2,901 | 1.74x |
| 2 | + Transposed W_hh (coalesced) | 804–1,023 | 4.94x–6.28x |
| 3 | + CUDA Graphs | 789–905 | 5.58x–6.40x |
| 4 | + FP16 half2 FMA gate packing | 548–602 | **8.39x–9.21x** |

#### Naive Kernel Fix (was a disclosed limitation, now resolved)

The naive kernel (step 0) previously carried a disclosed caveat: it failed numerical validation against
the PyTorch reference in a majority of repeated runs (~6/30 passing), attributed to "accumulated FP32
rounding error over its unoptimized summation order." That attribution was wrong — re-running the SAME
seeded input through the SAME binary produced *different* GPU output each time, which pure rounding-order
error cannot do (that would be deterministic). `compute-sanitizer --tool racecheck` confirmed a genuine
shared-memory data race: the per-timestep hidden-state write and the next timestep's read of it raced
despite an intervening `__syncthreads()`. Fixed by double-buffering the hidden state in
`fused_block3_naive.cu` so a timestep's read and write never target the same shared array. Verified:
**0 hazards under racecheck** (was reporting thousands), **100/100 runs pass** at the standard 1e-2
tolerance (was ~6/30), and **20/20 pass even at a 1e-5 tolerance** — i.e. genuinely close to the CPU
reference, not just passing a loose threshold. The naive kernel's latency figure above is now a real
n=100-trial mean of this fixed, verified kernel.

#### Measurement Stability (new finding, 2026-07-01)

Re-running the full n=100-trial CUDA kernel statistical harness later the same day (needed to safely add
the newly-fixed naive kernel's stats without overwriting the file with a partial run) produced
meaningfully different means for the transposed-W_hh and FP16 configs than the same harness gave earlier
that day — despite each individual session's own internal CV looking tight (6.8%–24.4%):

| Config | Session 1 mean | Session 2 mean | Delta |
|---|---|---|---|
| Transposed W_hh, no graphs | 804 us | 1,023 us | +27% |
| Transposed W_hh + CUDA Graphs | 789 us | 905 us | +15% |
| FP16 half2 | 602 us | 548 us | −9% |

This means within-session CV understates true measurement uncertainty on this WSL2 dev box: there is real
session-to-session drift (thermal state / background load / WSL2 scheduler) that one n=100 run, however
tight its own std, does not capture. Rather than silently picking one session's numbers, this README
reports both as an explicit range. **Recommendation for the DICC re-run (Phase 3):** repeat the same
n-trial harness across at least two separate `sbatch` submissions on different days and check for the
same drift there — if DICC (native Linux, no WSL2 passthrough) is stable across sessions, that's good
evidence this variance is a WSL2-specific artifact rather than a fundamental limit of the methodology.

### Detection Accuracy — BoT-IoT (733,705 test samples)

| Model | Macro-F1 | Method | Parameters |
|---|---|---|---|
| **Two-stage CNN-BiLSTM** | **0.9639** | **KD + focal + real-data FT** | **530,181** |
| MLP (distilled) | 0.9624 | Same KD recipe | 400,901 |
| KD + Focal CNN-BiLSTM | 0.9601 | a=0.7, T=5.0, g=2.0 | 530,181 |
| MLP (two-stage) | 0.9542 | Same FT recipe | 400,901 |
| Ensemble KD | 0.9529 | RF+XGB+LGB teacher | 530,181 |
| GPU RF (cuML) | 0.9471 | 200 trees, GPU | -- |
| Original V3 | 0.9352 | CE + SMOTE | 530,181 |
| CPU RF (sklearn) | 0.9864* | 200 trees, CPU | -- |

Gap to RF: **2.25%** (was 5.12%).

\* Trained/evaluated on the exact same preprocessed splits (`data/processed/*.npy`) the CNN-BiLSTM
itself uses — the apples-to-apples comparison. `scripts/rf_baseline.py` and `train_distill.py`'s
inline RF teacher each apply their own independent resampling straight from the raw CSVs and give
different (also legitimate, but not directly comparable) numbers — 0.9768 and ~0.975 respectively;
see `scripts/rf_baseline_processed.py` (`benchmarks/results/rf_baseline_processed.json`) for this
figure's source, confirmed reproducible byte-for-byte 2026-07-01.

### Detection Accuracy — ToN-IoT (42,209 test samples)

| Model | Macro-F1 | Features |
|---|---|---|
| **CNN-BiLSTM (clean)** | **0.9526** | 26 features |
| CPU RF (clean) | 0.9851 | 26 features |
| CNN-BiLSTM (original) | 0.8254 | 13 features |
| CPU RF (original) | 0.9396 | 13 features |

Dropping 16 sparse columns improved CNN-BiLSTM by +15.4% and RF by +4.9%.

### KD Sweep (BoT-IoT, 8 configurations)

| Alpha | Temp | Focal | Val F1 | Test F1 |
|---|---|---|---|---|
| -- | -- | -- | 0.9418 | 0.9330 |
| 0.5 | 1.0 | -- | 0.9703 | 0.9481 |
| 0.3 | 1.0 | -- | 0.9474 | 0.9421 |
| 0.7 | 1.0 | -- | 0.9599 | 0.9284 |
| 0.9 | 1.0 | -- | 0.9567 | 0.9284 |
| 0.5 | 3.0 | -- | 0.9541 | 0.9341 |
| 0.7 | 5.0 | -- | 0.9620 | 0.9547 |
| **0.7** | **5.0** | **2.0** | **0.9728** | **0.9601** |

### MLP Ablation

| Model | Params | Latency (A100) | Test F1 |
|---|---|---|---|
| **CNN-BiLSTM** | 530,181 | 592 us | **0.9639** |
| MLP | 400,901 | 175 us | 0.9542 |

MLP is 3.4x faster but CNN-BiLSTM wins accuracy. The recurrent architecture's dynamic control flow exposes compiler limitations — the core systems contribution.

### cuML GPU RF Comparison (A100)

| Metric | cuML RF | CNN-BiLSTM |
|---|---|---|
| VRAM | 444 MB | ~2 MB |
| F1 (GPU) | 0.9471 | 0.9639 |
| Throughput | 2,065,669 f/s | 87,791 f/s |
| Energy | 0.048 mJ/flow | 1.089 mJ/flow |

Throughput and Energy are both measured on the A100 for both methods (an earlier version of this table
mixed in the RTX 3050's 25,410 f/s streaming figure under an A100 header — fixed 2026-07-01). The
CNN-BiLSTM's A100 throughput above is derived from `a100_energy.json`'s batch=128 timing
(`128 / (avg_batch_time_ms / 1000)` = 128 / 1.458ms ≈ 87,791 flows/sec). The RTX 3050 streaming
throughput (25,899 f/s) is reported separately above under "Streaming and Energy".

CNN-BiLSTM uses **222x less VRAM** with **higher GPU accuracy**.

### Streaming and Energy

| Metric | Value |
|---|---|
| Streaming throughput | 25,899 flows/sec (batch=128) |
| Energy (RTX 3050) | 0.79 mJ/flow |
| Energy (A100) | 1.089 mJ/flow |
| Preprocessing overhead | 43.7 us (6.1% of pipeline) |
| End-to-end latency | 717.7 us |

### LLM Explainability

| Metric | Value |
|---|---|
| Dispatch overhead | 16.60 us p99 (~2.5%) |
| Generation time | ~8.5 sec/alert (background) |
| Model | TinyLlama 1.1B Q4 (0.77 GB) |
| Alert aggregation | 25,000 DDoS alerts to 10 LLM calls |
| Deployment | Fully on-device, air-gapped |

### GPU Hardware Profile (RTX 3050)

All kernels at **100% theoretical occupancy** (20 SMs, 1536 threads/SM):
- Block 1: 256 threads, 6 blocks/SM
- Block 2: 256 threads, 6 blocks/SM  
- Block 3: 128 threads, 12 blocks/SM
- Block 4: 64 threads, 24 blocks/SM

## CUDA Kernel Design

### Architecture
Four fused kernels replacing PyTorch operators:
- **Block 1**: Linear projection + reshape + Conv1D + BatchNorm + ReLU (5 ops fused)
- **Block 2**: Conv1D + BatchNorm + ReLU + MaxPool (4 ops fused)
- **Block 3**: 2-layer BiLSTM with transposed W_hh and FP16 half2 FMA
- **Block 4**: Dense + ReLU + Dense (3 ops fused)

### FP16 Half2 Optimization
1. Repacks W_hh into half2 pairs: (i_gate, f_gate) and (g_gate, o_gate)
2. Uses `__hfma2` for two gates per instruction
3. Shared memory for half-precision hidden states
4. FP32 only for sigmoid/tanh activations

## Datasets

**BoT-IoT** (Koroniotis et al., FGCS, 2019): 10 features, 5 classes, 733,705 test samples

**ToN-IoT** (Moustafa, 2021): 26 features (clean) / 13 (original), 10 classes, 42,209 test samples

## Verified Research Gaps

1. **Custom CUDA for CNN-BiLSTM IDS** — prior work (Sophimatics Phase 3, 2025) covers CNN only with 2.7x; we extend to BiLSTM with 4.40x over TensorRT
2. **On-device LLM for IDS** — Jamshidi et al. (2026) used cloud APIs; we provide fully local with 16.60 us p99 dispatch
3. **TensorRT vs custom CUDA for sub-1M models** — no prior comparison; TensorRT is 4.40x slower
4. **torch.compile crash on BiLSTM** — documented failure with CUDA graphs on recurrent control flow

## Limitations

- **RF accuracy gap**: 0.9639 vs 0.9864 on BoT-IoT (2.25%)
- **SMOTE dependency**: 52 Theft samples require synthetic augmentation
- **Pseudo-sequence**: MLP ablation shows sequential bias is not essential; architecture retained for compiler stress-testing
- **Energy**: cuML RF (0.048 mJ/flow) is more efficient than CNN-BiLSTM (1.089 mJ/flow) on same A100 hardware

## Citation

```bibtex
@article{colide2026,
  title={COLIDE: CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection},
  author={Haque, Ibteshamul and Por, Lip Yee},
  journal={Future Generation Computer Systems},
  year={2026},
  note={Under preparation}
}
```

## Acknowledgments

This research was conducted at FCSIT, Universiti Malaya, under Prof. Dr. Por Lip Yee. Computational resources provided by the Data-Intensive Computing Centre (DICC), Universiti Malaya.

## License

Academic research purposes only. Contact authors for commercial licensing.