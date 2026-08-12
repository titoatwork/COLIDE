# Stretch S1a: torch.compile on DICC (full-model absolute)

> **SUPERSEDED for multi-compiler claims (2026-08-12).**  
> Authoritative full matrix (eager/compile/ORT/TRT): `docs/DICC_MULTI_COMPILER_MATRIX.md`  
> Jobs 395433 (V100S) / 395417 (A100). This file remains as early S1a history only.

**Date (UTC):** 2026-08-12  
**Scope:** Full-model **eager** vs **torch.compile** absolute latencies on UM DICC.  
**Not Option A:** does **not** compare Custom CUDA blocks; full-model PT frameworks only.  
**Script:** `scripts/benchmark_torch_compile_dicc.py`  
**Champion:** `model/best_model_botiot_twostage.pth` md5 `80a90f7…`  
**Protocol:** n_trials=20 · inner=200 · warmup=50 · batch=1 · `mode=reduce-overhead`

---

## Jobs

| GPU | Job | State | Partition | Node | JSON |
|-----|-----|-------|-----------|------|------|
| V100S | **395338** | **COMPLETED** (exit 0, ~48 s) | gpu-v100s | gpu05 | `benchmarks/results/dicc/framework/torch_compile_v100s.json` |
| A100 | **395339** | **COMPLETED** (exit 0) | gpu-a100 | gpu06 | `benchmarks/results/dicc/framework/torch_compile_a100.json` |

---

## Results (means, µs)

| GPU | Eager mean | Eager CV% | compile mean | compile CV% | compile/eager |
|-----|----------:|----------:|-------------:|------------:|--------------:|
| **V100S** | **1033.0** | 1.38 | **818.0** | 1.19 | **0.79** (~1.26× faster) |
| **A100** | **956.9** | 0.31 | **760.6** | 0.22 | **0.79** (~1.26× faster) |

| GPU | Eager p50 | compile p50 | Eager min–max | compile min–max |
|-----|----------:|------------:|---------------|-----------------|
| V100S | 1040.2 | 813.5 | 1004–1045 | 807–833 |
| A100 | 956.0 | 760.0 | 954–966 | 759–765 |

**Hardware:** Tesla V100S-PCIE-32GB · NVIDIA A100-SXM4-80GB  
**compile error:** null on both GPUs

### Relation to campaign PT full V3

Campaign SUCCESS harness PT full means: V100S ~**964–973 µs**, A100 ~**945–962 µs** (different inner protocol).  
This stretch harness uses inner=200 trial medians.  
**Do not** mix ratios across harnesses. Prefer absolute pairs **within each JSON**.

---

## Paper-facing wording (approved)

> On UM DICC, full-model torch.compile (`reduce-overhead`) yields mean latencies of **~818 µs** (V100S) and **~761 µs** (A100) versus eager full V3 **~1033 µs** / **~957 µs** under a fixed multi-trial protocol (n=20 trial medians, inner=200). Compile is ~1.26× faster than eager on both server GPUs. These are absolute framework results, not Option A Custom CUDA parity. TensorRT / ORT-GPU were **not** re-run on DICC.

---

## Files

```text
benchmarks/results/dicc/framework/torch_compile_v100s.json
benchmarks/results/dicc/framework/torch_compile_a100.json
benchmarks/results/dicc/framework/torch_compile_*_{395338,395339}.{out,err}
scripts/benchmark_torch_compile_dicc.py
logs/job_torch_compile_v100s.sh
logs/job_torch_compile_a100.sh
```

*End S1a pack — COMPLETE both GPUs.*
