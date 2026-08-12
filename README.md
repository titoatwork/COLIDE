# COLIDE: CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection

[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](#license)

## Abstract

COLIDE is a multi-objective IoT IDS systems project: a sealed-protocol CAD-CBA detection package (competitive but not pure-F1 SOTA vs RF/LGBM) plus **Option A** Custom CUDA kernels (per-block vs matching ops) and measured multi-GPU multi-session latency on UM DICC (V100S + A100). On a **laptop RTX 3050 (WSL2)**, Custom CUDA pipeline ranges beat full-model eager / torch.compile / TensorRT / ORT-GPU when reported as multi-session ranges (see below; incomplete CUDA scope caveats apply). On **DICC**, three sessions show stable means: matching **PyTorch Block 3 is faster than CUDA Block 3 FP16** (~363 vs ~513 µs V100S; ~385–391 vs ~667–671 µs A100), while CUDA remains much faster on Blocks 1/2/4. Full Custom CUDA vs full V3 speedup is **not** claimed. Async on-device LLM dispatch is **16.60 µs p99** (not full generation). See `docs/DICC_EXTRACTION_TABLES.md` and `docs/PRE_MANUSCRIPT_CLOSURE.md`.

## Key Contributions

1. **Multi-objective CAD-CBA package** under sealed protocol (HPO, focal + ensemble KD, ablations, dual bars vs classical baselines) — accuracy–efficiency story, not F1 supremacy.
2. **Option A Custom CUDA** for CNN-BiLSTM blocks: large wins on B1/B2/B4 vs matching PT; **honest B3 result on DICC** (PT B3 faster than CUDA FP16 B3 on V100S/A100).
3. **Multi-session multi-GPU measurement** (3 sessions × V100S + A100 SUCCESS trees) with formal compares — not RTX-only portability claims.
4. **Laptop multi-compiler ranges** (eager / torch.compile / TensorRT / ORT-GPU vs Custom pipeline) with measurement-stability ranges; **not** re-proven as portable to DICC without server-side remeasure.
5. **On-device LLM dispatch** micro-benchmark (**16.60 µs p99**); full free-form LLM explainability is **not** a title-level claim.

## Results Summary

### Framework Comparison (RTX 3050, 20 Trials, Statistical Significance)

**MEASUREMENT STABILITY (2026-07-02, session 3):** re-running this benchmark suite — even twice,
back-to-back, minutes apart in the same sitting — gave meaningfully different framework latencies
(torch.compile and TensorRT both swung 14-17% run to run). This is the same phenomenon already
documented for Block 3 alone, now confirmed to affect the headline framework-comparison numbers too.
The table below reports the **range across every independent session measured so far**: 3 for the
framework side (an original measurement plus two fresh 2026-07-02 re-runs, 20 trials each) and 5 for
the Custom CUDA side (the Custom CUDA FP16 figure is re-derived every time the Block 1-4 kernels are
re-checked, not just when the framework side is re-run, so it has more independent data points).
Custom CUDA FP16 is derived from a real n=100-trial distribution each session (see
`benchmarks/results/cuda_kernel_stats_rtx3050.json`), not a fixed constant with no variance as in an
earlier version of this table. Significance is a two-sample Welch's t-test (framework's 20 trials vs.
Custom CUDA's 100 trials), not a one-sample test against a fixed point; the exact CI/p-value shown is
from the most recent session — see below for which comparisons are robust across sessions and which
aren't.

| Method | Mean (us) range | vs Custom CUDA (range) |
|---|---|---|
| **Custom CUDA FP16** | **594–675** | **1.00x** |
| ORT CPU | 487–699 | 0.72x–1.18x |
| torch.compile | 1,519–1,777 | 2.25x–2.99x |
| Eager PyTorch | 2,050–2,247 | 3.04x–3.78x |
| TensorRT FP16 | 2,427–2,966 | 3.60x–4.99x |
| ORT GPU | 3,862–4,652 | 5.72x–7.83x |

**Significance robustness differs by comparison.** Eager PyTorch, torch.compile, TensorRT, and ORT GPU
are all p<0.001 significant in **all three** independently measured framework-side sessions — the
"Custom CUDA is faster" conclusion is robust even though the exact ratio isn't. **ORT CPU is not consistently
significant**: not significantly different from Custom CUDA in the original session (p=0.483, ns), but
significantly *faster* than Custom CUDA in both fresh 2026-07-02 sessions (p<0.001) — its ratio range
(0.72x–1.18x) genuinely straddles parity, unlike the other four frameworks.

torch.compile with CUDA graph capture **crashes** on BiLSTM (dynamic recurrent control flow). TensorRT is slower than eager PyTorch for this sub-1M parameter model.

### Cross-Hardware (UM DICC multi-session campaign, Option A)

**Source:** `docs/DICC_EXTRACTION_TABLES.md` · three sessions (S1, S2, Day2) · SUCCESS on laptop.

| GPU | CUDA B3 FP16 mean | PT B3 mean | PT full V3 mean | Note |
|-----|------------------:|-----------:|----------------:|------|
| **V100S** | ~**513 µs** | ~**363 µs** | ~**964–973 µs** | PT B3 **faster** than CUDA B3 FP16 |
| **A100** | ~**667–671 µs** | ~**384–391 µs** | ~**945–962 µs** | same direction |
| RTX 3050 (laptop) | framework table above | local ranges | local | multi-compiler matrix is **laptop** |

Legacy June single-shot pipeline (~551 / ~592 µs CUDA-only) remains **legacy** (no same-GPU PT that day).  
**Do not** claim full Custom CUDA pipeline vs full V3 PT. Prefer per-block tables (esp. B3 honesty + B1/B2/B4 CUDA wins).  
Details: `docs/DICC_EXTRACTION_TABLES.md`, `docs/DICC_B3_CUDA_VS_PT_REPORT.md`.

### DICC torch.compile stretch (full-model absolute, not Option A)

| GPU | Eager full V3 mean | torch.compile mean | Job / source |
|-----|-------------------:|-------------------:|--------------|
| **V100S** | ~**1033 µs** | ~**818 µs** | job 395338 · `framework/torch_compile_v100s.json` |
| **A100** | ~**957 µs** | ~**761 µs** | job 395339 · `framework/torch_compile_a100.json` |

Protocol: n=20 trial medians, inner=200, warmup=50, `reduce-overhead`. Compile ~**1.26×** faster than eager on both GPUs. Absolute framework latencies only — **not** Custom CUDA parity. TRT/ORT not run on DICC. Details: `docs/DICC_TORCH_COMPILE_STRETCH.md`.

### Per-Block Performance (RTX 3050)

| Block | PyTorch GPU (us) | Custom CUDA (us) | Speedup |
|---|---|---|---|
| 1: Proj+Conv1+BN+ReLU | 404 | 62 | 6.55x |
| 2: Conv2+BN+ReLU+Pool | 282 | 87 | 3.24x |
| 3: BiLSTM FP16 half2 | 784 | 532–602* | 1.30x–1.47x* |
| 4: Dense Head | 122 | 20 | 6.07x |

\* Range across five independent n=100-trial measurement sessions on this dev box, not a lingering
ambiguity — see "Measurement Stability" below.

### Block 3 Optimization Progression (7.55x–9.50x)

Step 0 (naive) is now backed by **three independent n=100-trial sessions** of the **fixed** kernel (see
"Naive Kernel Fix" below) — the race-condition fix landed mid-session-2, so no session-1 measurement of
the fixed kernel exists. Step 1 remains a historical single-run figure with no surviving re-runnable
artifact (its kernel file was overwritten by later optimizations). Steps 2-4 are each backed by **five
independent n=100-trial measurement sessions** (`benchmarks/results/cuda_kernel_stats_rtx3050.json`,
regenerated across sessions — see "Measurement Stability" below) rather than one. The PyTorch cuDNN
reference used for per-step ratios is a real n=50-trial mean, **784us** (std 89us, CV 11.3%) from
`benchmarks/results/pytorch_block3_stats_rtx3050.json` (`scripts/benchmark_pytorch_block3_stats.py`, 50
independent subprocess trials — mirrors the CUDA kernel statistical harness so both sides of the ratio are
backed by a real distribution). This resolves an earlier ambiguity between two single-run point estimates
(740.7us vs 943.6us) that bracketed the true mean. With the real baseline, **the FP16 step beats cuDNN in
all five sessions (1.30x–1.47x)**; the transposed-W_hh steps (with or without CUDA Graphs) land at/around
parity with PyTorch across all five sessions (0.77x–1.08x, occasionally edging past parity) — that
conclusion (transposed steps don't clearly beat cuDNN) is robust across the session-to-session drift, even
though the exact ratios aren't.

| Step | Configuration | Latency (us) | Cumulative |
|---|---|---|---|
| 0 | Naive (1 thread/hidden), race-fixed | 4,544–5,050 | 1.00x |
| 1 | + Precomputed W_ih x X | 2,901 | 1.57x–1.74x |
| 2 | + Transposed W_hh (coalesced) | 732–1,023 | 4.44x–6.90x |
| 3 | + CUDA Graphs | 724–905 | 5.02x–6.97x |
| 4 | + FP16 half2 FMA gate packing | 532–602 | **7.55x–9.50x** |

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
reports both as an explicit range. **Phase 3 DICC re-run (tooling ready on `final-polish`):** use
`dicc_scripts/submit_session.sh` for Day 1 and Day 2 on the **same git SHA and same compiled binaries**,
then `scripts/compare_dicc_sessions.py` (rejects provenance mismatches; requires two distinct dates and
`SUCCESS` markers). If DICC session means are stable, describe that as *consistent with WSL2-specific
drift* — not as proof that WSL2 is the sole cause. Operator guide: `dicc_scripts/README.md`.

### Detection Accuracy — BoT-IoT (733,705 test samples)

| Model | Macro-F1 | Method | Parameters |
|---|---|---|---|
| **Two-stage CNN-BiLSTM** | **0.9790** | **KD (a=0.6,T=10.0) + focal + real-data FT** | **530,181** |
| KD + Focal CNN-BiLSTM | 0.9763 | a=0.6, T=10.0, g=2.0 | 530,181 |
| MLP (distilled) | 0.9624 | Same KD recipe | 400,901 |
| MLP (two-stage) | 0.9542 | Same FT recipe | 400,901 |
| Ensemble KD | 0.9529 | RF+XGB+LGB teacher | 530,181 |
| GPU RF (cuML) | 0.9471 | 200 trees, GPU | -- |
| Original V3 | 0.9352 | CE + SMOTE | 530,181 |
| CPU RF (sklearn) | 0.9864* | 200 trees, CPU | -- |

Gap to RF: **0.74%** (was 5.12%).

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

### KD Sweep (BoT-IoT, 14 configurations)

Round 1 (no focal loss, or a single focal point) established alpha=0.7/T=5.0 as the best region found
at the time. Round 2 (2026-07-01, session 2) extended temperature past 5.0 with focal_gamma=2.0 fixed
across a 3x2 alpha/temperature grid, motivated by Round 1's still-rising T=1->3->5 trend. One config
(a=0.7, T=10.0) is a genuine outlier — collapsed on the Normal/Theft minority classes (precision
0.75/0.67) despite excellent majority-class performance, dragging macro-F1 down; kept in the table as a
real, useful negative result rather than dropped.

| Alpha | Temp | Focal | Val F1 | Test F1 |
|---|---|---|---|---|
| -- | -- | -- | 0.9418 | 0.9330 |
| 0.5 | 1.0 | -- | 0.9703 | 0.9481 |
| 0.3 | 1.0 | -- | 0.9474 | 0.9421 |
| 0.7 | 1.0 | -- | 0.9599 | 0.9284 |
| 0.9 | 1.0 | -- | 0.9567 | 0.9284 |
| 0.5 | 3.0 | -- | 0.9541 | 0.9341 |
| 0.7 | 5.0 | -- | 0.9620 | 0.9547 |
| 0.7 | 5.0 | 2.0 | 0.9728 | 0.9601 |
| 0.6 | 7.0 | 2.0 | 0.9780 | 0.9702 |
| 0.7 | 7.0 | 2.0 | 0.9728 | 0.9687 |
| 0.8 | 7.0 | 2.0 | 0.9751 | 0.9757 |
| 0.7 | 10.0 | 2.0 | 0.9482 | 0.9033 |
| 0.8 | 10.0 | 2.0 | 0.9672 | 0.9745 |
| **0.6** | **10.0** | **2.0** | **0.9757** | **0.9763** |

### MLP Ablation

| Model | Params | Latency (A100) | Test F1 |
|---|---|---|---|
| **CNN-BiLSTM** | 530,181 | 592 us | **0.9790** |
| MLP | 400,901 | 175 us | 0.9542 |

MLP is 3.4x faster but CNN-BiLSTM wins accuracy. The recurrent architecture's dynamic control flow exposes compiler limitations — the core systems contribution.

### cuML GPU RF Comparison (A100)

| Metric | cuML RF | CNN-BiLSTM |
|---|---|---|
| VRAM | 444 MB | ~2 MB |
| F1 (GPU) | 0.9471 | 0.9790 |
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

1. **Custom CUDA for CNN-BiLSTM IDS** — closest prior work (Ibrahim et al., *Computer Networks*, 2026) applies custom CUDA kernels to a GNN-based IDS vs. a CPU baseline only (1.22x-1.48x); we target a recurrent CNN-BiLSTM benchmarked against production ML frameworks. On **laptop RTX 3050**, Custom CUDA pipeline ranges show **3.60x–4.99x** over TensorRT (multi-session ranges; incomplete CUDA scope caveats apply). On **DICC**, Option A shows CUDA wins on B1/B2/B4 and **PT wins B3**.
2. **On-device LLM for IDS** — Jamshidi et al. (2026) used cloud APIs; we provide fully local with 16.60 us p99 dispatch
3. **TensorRT vs custom CUDA for sub-1M models** — no prior comparison on laptop; TensorRT is 3.60x–4.99x slower than Custom pipeline **on RTX 3050** (not re-proven on DICC)
4. **torch.compile on recurrent models** — laptop CUDA-graph path can crash on BiLSTM; DICC stretch full-model `reduce-overhead` ran on V100S (~818 vs ~1033 µs) and A100 (~761 vs ~957 µs)

## Limitations

- **RF accuracy gap**: 0.9790 vs 0.9864 on BoT-IoT (0.74%)
- **SMOTE dependency**: 52 Theft samples require synthetic augmentation
- **Pseudo-sequence**: MLP ablation shows sequential bias is not essential; architecture retained for compiler stress-testing
- **Energy**: cuML RF (0.048 mJ/flow) is more efficient than CNN-BiLSTM (1.089 mJ/flow) on same A100 hardware
- **Measurement environment**: WSL2/RTX 3050 session drift → framework ratios as ranges. DICC multi-session SUCCESS (V100S+A100) with same-GPU PT is available under `benchmarks/results/dicc/`; Block 3 CUDA does **not** beat matching PT on servers (see `docs/DICC_B3_CUDA_VS_PT_REPORT.md`)
- **Numerical fidelity**: export path is bit-identical on n=10 reference samples; CUDA block self-checks PASS at disclosed tolerances (FP16 Block 3: 5e-2) — see `docs/paper_text_blocks.md` §15–§16 and `scripts/numerical_fidelity.py`

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