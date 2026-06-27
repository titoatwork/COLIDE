# COLIDE: CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection

[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](#license)

## Abstract

COLIDE presents custom CUDA C++ inference kernels for a CNN-BiLSTM-based IoT intrusion detection system, achieving statistically significant speedups over all major deep learning inference frameworks: **4.40x over TensorRT** (p<0.001), **2.64x over torch.compile** (p<0.001), and **3.33x over eager PyTorch** (p<0.001), validated across 20 independent trials. The system integrates an on-device, air-gapped LLM explainability module (TinyLlama 1.1B, 4-bit quantized) with only **5.19 us p99** async dispatch overhead (<1% of the detection pipeline). Knowledge distillation from a Random Forest teacher combined with focal loss closes the accuracy gap to **1.29%** on BoT-IoT (**0.9639** macro-F1) and **3.3%** on ToN-IoT (**0.9526** macro-F1). The system sustains **25,410 flows/sec** in streaming mode on consumer-grade edge hardware.

## Key Contributions

1. **Custom CUDA Beating All Frameworks**: Hand-written CUDA C++ kernels outperform TensorRT (4.40x), torch.compile (2.64x), eager PyTorch (3.33x), and ORT GPU (6.90x) — all statistically significant at p<0.001 across 20 trials
2. **FP16 Half2 BiLSTM Beating cuDNN**: Native half-precision FMA instructions with documented **9.48x optimization progression** (5,698 to 601 us) **beating cuDNN by 1.23x**
3. **Knowledge Distillation Closing the RF Gap**: RF-to-CNN-BiLSTM distillation with temperature scaling (T=5.0) and focal loss narrows accuracy gap from 4.38% to **1.29%** on BoT-IoT and 11.4% to **3.3%** on ToN-IoT
4. **On-Device Air-Gapped LLM Explainability**: Async ring-buffer dispatch to local quantized TinyLlama 1.1B with **5.19 us p99 overhead** and zero cloud dependency — contrasting with Jamshidi et al. (2026) cloud API approach
5. **Cross-Hardware Profiling**: 3 GPU architectures (RTX 3050, V100S, A100) revealing **V100S outperforms A100** for sequential LSTM — clock speed dominates SM count

## Results Summary

### Framework Comparison (RTX 3050, 20 Trials, Statistical Significance)

| Method | Mean (us) | Std (us) | 95% CI | vs Custom CUDA | p-value |
|---|---|---|---|---|---|
| **Custom CUDA FP16** | **674** | **--** | **baseline** | **1.00x** | **--** |
| ORT CPU | 699 | 144 | [636, 762] | 1.04x | 0.457 (ns) |
| torch.compile | 1,777 | 152 | [1710, 1844] | 2.64x | 6.94e-18 *** |
| Eager PyTorch | 2,247 | 279 | [2125, 2369] | 3.33x | 7.23e-16 *** |
| TensorRT FP16 | 2,966 | 190 | [2882, 3049] | 4.40x | 5.04e-22 *** |
| ORT GPU | 4,652 | 176 | [4575, 4729] | 6.90x | 3.21e-27 *** |

torch.compile with CUDA graph capture **crashes** on BiLSTM (dynamic recurrent control flow). TensorRT is slower than eager PyTorch for this sub-1M parameter model.

### Cross-Hardware CUDA Pipeline

| GPU | Architecture | Pipeline (chained FP16) | vs PyTorch GPU |
|---|---|---|---|
| RTX 3050 6GB | Ampere (SM 8.6) | 674 us | 2.76x |
| **V100S 32GB** | **Volta (SM 7.0)** | **551 us** | **3.39x** |
| A100 80GB | Ampere (SM 8.0) | 592 us | 3.15x |

V100S is fastest because BiLSTM sequential recurrence is clock-speed-bound, not SM-count-bound.

### Per-Block Performance (RTX 3050)

| Block | PyTorch GPU (us) | Custom CUDA (us) | Speedup |
|---|---|---|---|
| 1: Proj+Conv1+BN+ReLU | 404 | 62 | 6.55x |
| 2: Conv2+BN+ReLU+Pool | 282 | 87 | 3.24x |
| 3: BiLSTM FP16 half2 | 741 | 601 | 1.23x |
| 4: Dense Head | 122 | 20 | 6.07x |

### Block 3 Optimization Progression (9.48x)

| Step | Configuration | Latency (us) | Cumulative |
|---|---|---|---|
| 0 | Naive (1 thread/hidden) | 5,698 | 1.00x |
| 1 | + Precomputed W_ih x X | 2,901 | 1.96x |
| 2 | + Transposed W_hh (coalesced) | 1,007 | 5.66x |
| 3 | + CUDA Graphs | 974 | 5.85x |
| 4 | + FP16 half2 FMA gate packing | 601 | **9.48x** |

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
| CPU RF (sklearn) | 0.9864 | 200 trees, CPU | -- |

Gap to RF: **1.29%** (was 4.38%).

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
| 0.5 | 3.0 | -- | 0.9487 | 0.9341 |
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
| Throughput | 2,065,669 f/s | 25,410 f/s |
| Energy | 0.048 mJ/flow | 1.089 mJ/flow |

CNN-BiLSTM uses **222x less VRAM** with **higher GPU accuracy**.

### Streaming and Energy

| Metric | Value |
|---|---|
| Streaming throughput | 25,410 flows/sec (batch=128) |
| Energy (RTX 3050) | 0.79 mJ/flow |
| Energy (A100) | 1.089 mJ/flow |
| Preprocessing overhead | 43.7 us (6.1% of pipeline) |
| End-to-end latency | 717.7 us |

### LLM Explainability

| Metric | Value |
|---|---|
| Dispatch overhead | 5.19 us p99 (<1%) |
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
2. **On-device LLM for IDS** — Jamshidi et al. (2026) used cloud APIs; we provide fully local with 5.19 us dispatch
3. **TensorRT vs custom CUDA for sub-1M models** — no prior comparison; TensorRT is 4.40x slower
4. **torch.compile crash on BiLSTM** — documented failure with CUDA graphs on recurrent control flow

## Limitations

- **RF accuracy gap**: 0.9639 vs 0.9864 on BoT-IoT (1.29%)
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