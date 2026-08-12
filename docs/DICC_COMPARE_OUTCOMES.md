# DICC formal + manual compare outcomes

**Date (UTC):** 2026-08-12  
**Laptop tree:** `benchmarks/results/dicc/` rsynced (~1.9 MB, 6 SUCCESS)

---

## Formal `compare_dicc_sessions.py`

| Pair | Result | Notes |
|------|--------|-------|
| V100S S1 (`20260807`) vs Day2 (`20260808`) | **RAN** | `stable_cross_day=False` (max spread **11.0%** on CUDA B1). B3 FP16 spread **0.04%**. Output: `core/v100s/compare_20260807_vs_20260808.json` |
| V100S S1 vs S2 (`20260807_s2`) | **RAN + stable** | max spread **2.49%** &lt; 5%. Output: `core/v100s/compare_20260807_vs_20260807_s2.json` |
| A100 S1 vs Day2 | **REJECTED** | `git_dirty=true` on both |
| A100 S1 vs S2 | **REJECTED** | `git_dirty=true` on both |

**FLAG:** V100 manifests have `git_dirty=false` but `git_sha=unknown`. A100 all `git_dirty=true`, `git_sha=ac9ed1b658…`.

---

## Manual means (all 6 SUCCESS) — B3 Option A

| GPU | Session | CUDA B3 FP16 | PT B3 | CUDA/PT |
|-----|---------|-------------:|------:|--------:|
| V100S | S1 | 513.3 | 363.4 | 1.41× CUDA slower |
| V100S | S2 | 513.0 | 363.6 | 1.41× |
| V100S | Day2 | 513.1 | 363.3 | 1.41× |
| A100 | S1 | 668.0 | 383.6 | 1.74× |
| A100 | S2 | 667.4 | 389.0 | 1.72× |
| A100 | Day2 | 671.2 | 390.9 | 1.72× |

Session-to-session B3 CUDA/PT: **very stable**. Primary Option A B3 head-to-head: **PT wins** on both GPUs.

---

## Stability judgment

| Metric | V100 | A100 (manual) |
|--------|------|----------------|
| B3 FP16 CUDA across sessions | ~513 µs all three | ~667–671 µs |
| B3 PT across sessions | ~363 µs | ~384–391 µs |
| Formal “stable_cross_day” S1–Day2 | **False** (B1 CUDA 11% spread) | N/A (dirty reject) |
| Formal S1–S2 | **True** (2.49%) | N/A |

**Paper wording:** B3 multi-session means are consistent; do **not** claim all metrics session-stable without caveats (B1 CUDA showed larger S1–Day2 spread on V100).

---

## Files on laptop

```text
benchmarks/results/dicc/core/{v100s,a100}/…_SUCCESS runs
benchmarks/results/dicc/core/v100s/compare_20260807_vs_20260808.json
benchmarks/results/dicc/core/v100s/compare_20260807_vs_20260807_s2.json
```
