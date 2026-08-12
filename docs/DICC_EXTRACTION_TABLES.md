# DICC extraction tables (laptop JSON)
**Generated:** 2026-08-12 · **Option A** · means from SUCCESS runs only
## 1. Block 3 CUDA FP16 vs PT B3 (primary)
| GPU | Session | CUDA B3 FP16 mean | PT B3 mean | ratio CUDA/PT | CUDA B1 | PT B1 | CUDA B2 | PT B2 | CUDA B4 | PT B4 | PT full |
|-----|---------|------------------:|-----------:|--------------:|--------:|------:|--------:|------:|--------:|------:|---------:|
| V100S | S1 | 513.3 | 363.5 | 1.41 | 9.4 | 168.1 | 29.4 | 155.7 | 8.2 | 93.9 | 963.5 |
| V100S | S2 | 513.0 | 363.6 | 1.41 | 9.6 | 169.2 | 29.7 | 157.2 | 8.2 | 95.6 | 966.7 |
| V100S | Day2 | 513.1 | 363.3 | 1.41 | 8.4 | 169.7 | 29.3 | 156.6 | 8.2 | 94.5 | 973.4 |
| A100 | S1 | 668.0 | 383.7 | 1.74 | 12.1 | 166.3 | 35.1 | 164.7 | 10.3 | 83.9 | 945.2 |
| A100 | S2 | 667.4 | 389.0 | 1.72 | 12.0 | 169.3 | 34.9 | 167.4 | 10.4 | 85.4 | 961.8 |
| A100 | Day2 | 671.2 | 390.9 | 1.72 | 12.4 | 168.9 | 35.1 | 167.2 | 10.4 | 85.3 | 956.6 |

## 2. Session stability (relative spread of means)
| GPU | Metric | S1 | S2 | Day2 | max spread % |
|-----|--------|---:|---:|-----:|-------------:|
| V100 | CUDA B3 FP16 | 513.3 | 513.0 | 513.1 | 0.05 |
| V100 | PT B3 | 363.5 | 363.6 | 363.3 | 0.08 |
| V100 | PT full | 963.5 | 966.7 | 973.4 | 1.03 |
| V100 | CUDA B1 | 9.4 | 9.6 | 8.4 | 13.30 |
| A100 | CUDA B3 FP16 | 668.0 | 667.4 | 671.2 | 0.57 |
| A100 | PT B3 | 383.7 | 389.0 | 390.9 | 1.86 |
| A100 | PT full | 945.2 | 961.8 | 956.6 | 1.74 |
| A100 | CUDA B1 | 12.1 | 12.0 | 12.4 | 3.26 |

## 3. Stretch S1a — torch.compile full-model (absolute µs)

| GPU | Job | Eager mean | compile mean | ratio compile/eager | JSON |
|-----|-----|----------:|-------------:|--------------------:|------|
| V100S | 395338 | 1033.0 | 818.0 | 0.79 | `framework/torch_compile_v100s.json` |
| A100 | 395339 | 956.9 | 760.6 | 0.79 | `framework/torch_compile_a100.json` |

Protocol differs from campaign PT full (inner=200 here). Use within-JSON pairs only.
