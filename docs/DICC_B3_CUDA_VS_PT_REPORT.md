# Block 3: Custom CUDA FP16 vs matching PyTorch (DICC)

**Date:** 2026-08-12  
**Option A:** same block, same GPU, same campaign protocol  
**Sources:** `benchmarks/results/dicc/core/{v100s,a100}/*_job*/{cuda_kernel_stats,pytorch_gpu_stats}.json`  
**Compare:** `core/v100s/compare_*.json`, `core/a100/compare_*_allow_dirty.json`

---

## 1. Protocol

| Side | Protocol |
|------|----------|
| CUDA B3 FP16 | `fused_block3_fp16` binary · n=100 trial medians · mean/std/CI in JSON |
| PyTorch B3 | Isolated BiLSTM block forward · n=20 trial medians · 1000 inners · warmup 50 |
| Checkpoint | `model/best_model_botiot_twostage.pth` |
| Not compared | Full Custom pipeline vs full V3 (invalid) |

---

## 2. Results (means, µs)

| GPU | Session | CUDA B3 FP16 | PT B3 | Ratio CUDA/PT | Winner |
|-----|---------|-------------:|------:|--------------:|--------|
| V100S | S1 | 513.3 | 363.5 | 1.41 | **PT** |
| V100S | S2 | 513.0 | 363.6 | 1.41 | **PT** |
| V100S | Day2 | 513.1 | 363.3 | 1.41 | **PT** |
| A100 | S1 | 668.0 | 383.7 | 1.74 | **PT** |
| A100 | S2 | 667.4 | 389.0 | 1.72 | **PT** |
| A100 | Day2 | 671.2 | 390.9 | 1.72 | **PT** |

**Interpretation:** On both server GPUs, matching PyTorch Block 3 is **faster** than our FP16 Custom CUDA Block 3. Effect is **stable across three sessions** (not measurement noise).

### Session stability (B3)

| GPU | Metric | Max relative spread of means (S1/S2/Day2) |
|-----|--------|------------------------------------------:|
| V100S | CUDA B3 FP16 | **0.05%** |
| V100S | PT B3 | **0.08%** |
| A100 | CUDA B3 FP16 | **0.57%** |
| A100 | PT B3 | **1.86%** |

Formal V100 S1–S2: `stable_cross_day=True` (max 2.49%).  
Formal V100 S1–Day2: max spread 11% driven by **B1**, not B3.  
A100 compares: require `--allow-dirty` (git_dirty); with that, stable ≤2.93%.

---

## 3. Context: other blocks (same runs)

CUDA B1/B2/B4 remain much faster than PT counterparts (e.g. V100 B1 ~8–10 µs vs PT ~168 µs).  
B3 dominates the BiLSTM path, so overall “recurrent systems” narrative must not claim CUDA B3 leadership on DICC.

---

## 4. Relation to laptop / multi-compiler claims

| Claim | Status |
|-------|--------|
| Laptop Custom pipeline vs eager/compile/TRT/ORT | Separate; **local ranges only** |
| Local B3 FP16 vs cuDNN ~784 µs (1.30–1.47×) | **Laptop only** — **does not port** to V100/A100 vs PT B3 |
| DICC B3 CUDA vs PT | **PT wins** |

---

## 5. Paper-facing statement (approved wording)

> On UM DICC, three multi-session campaigns on V100S and A100 show that matching PyTorch Block-3 (BiLSTM) latency is lower than our FP16 Custom CUDA Block-3 kernel (V100S: ~363 µs vs ~513 µs; A100: ~385–391 µs vs ~667–671 µs). Session-to-session means for Block 3 are highly consistent. Custom CUDA remains substantially faster on CNN and dense blocks (Blocks 1, 2, and 4). Full Custom CUDA pipeline versus full V3 PyTorch speedup is not claimed (architecture parity incomplete).

---

## 6. Optional next science (not pre-manuscript required)

- Profile B3 CUDA on A100 vs V100 (Nsight)  
- Kernel optim + re-measure  
- Do **not** rewrite this table until new JSON exists  

*End.*
