# Claim map — pre-write authority (pre-manuscript closed)

**Date:** 2026-08-12  
**Use:** manuscript drafting only. Every number must trace to JSON or a locked doc table.  
**Champion:** md5 `80a90f7…`

---

## A. Detection / multi-objective (local sealed protocol)

| Claim | Status | Source |
|-------|--------|--------|
| Sealed multi-seed test macro-F1 **0.9780 ± 0.0033** | **OK** | `sealed_test/summary.json` |
| Not pure-F1 SOTA vs protocol LGBM **0.9818** | **OK** | dual-bar framing |
| Multi-obj efficiency path (local WP6b ranges) | **OK** | energy / PT@256 / peak mem |
| Full free-form LLM explainability as title claim | **FORBIDDEN** | dispatch-only **16.60 µs p99** |

---

## B. Option A Custom CUDA (DICC multi-session)

| Claim | Status | Source |
|-------|--------|--------|
| B1/B2/B4 CUDA much faster than matching PT on V100S/A100 | **OK** | extraction tables |
| B3 CUDA FP16 **slower** than matching PT B3 (both GPUs, 3 sessions) | **OK — required honesty** | B3 report + extraction |
| Session-stable B3 means | **OK** | max spread ≪ 2% B3 |
| Full Custom CUDA pipeline vs full V3 PT speedup | **FORBIDDEN** | Option A / harness `valid=false` |
| Portable “CUDA BiLSTM beats PT/cuDNN on servers” | **FORBIDDEN** | PT wins B3 |
| All metrics session-stable S1–Day2 without caveat | **FORBIDDEN** | V100 B1 11% formal spread |

### B3 Welch-style summary (S1/S2/Day2, trial distributions)

Protocol caveat: CUDA n=100 kernel trials vs PT n=20 subprocess trials — different harnesses; direction is unambiguous.

| GPU | Session | CUDA mean±std | PT mean±std | Welch t | Cohen d | Winner |
|-----|---------|---------------|-------------|---------|---------|--------|
| V100S | S1 | 513.3±0.8 | 363.5±12.6 | 53 | ~29 | **PT** |
| V100S | S2 | 513.0±0.7 | 363.6±4.8 | 139 | ~74 | **PT** |
| V100S | Day2 | 513.1±0.7 | 363.3±6.2 | 108 | ~58 | **PT** |
| A100 | S1 | 668.0±18.3 | 383.7±5.0 | 133 | ~17 | **PT** |
| A100 | S2 | 667.4±18.7 | 389.0±6.9 | 115 | ~16 | **PT** |
| A100 | Day2 | 671.2±20.0 | 390.9±6.7 | 112 | ~15 | **PT** |

---

## C. Multi-compiler

### C1 Laptop (RTX 3050 / WSL) — multi-session **ranges**

| Claim | Status |
|-------|--------|
| Custom pipeline ranges vs eager/compile/TRT/ORT-GPU as ranges | **OK** (local) |
| Same ratios portable to DICC without remeasure | **FORBIDDEN** |

### C2 DICC multi-compiler matrix (batch-1 absolutes, n=20)

| Method | V100S mean µs | A100 mean µs | Status |
|--------|--------------:|-------------:|--------|
| Eager | 1041 | 932 | **OK** |
| torch.compile | 865 | 770 | **OK** |
| ORT CUDA | 895 | 865 | **OK** |
| ORT CPU | 500 | 461 | **OK** (not a GPU deploy claim) |
| ORT TRT EP | 766 | 2033 | **OK** (active TRT EP; A100 slow) |
| TRT native FP16 | **528** | **588** | **OK** (fastest GPU path here) |

Source: `docs/DICC_MULTI_COMPILER_MATRIX.md` · jobs 395433 / 395417.

---

## D. Explicitly do **not** write

1. Full-pipeline Custom CUDA × vs full V3.  
2. “CUDA Block 3 beats PyTorch on V100/A100.”  
3. Mixing laptop framework ratios with DICC absolute µs in one headline.  
4. Treating ORT CPU win as the main GPU systems result.  
5. Claiming multi-day “all metrics stable” without B1 caveat.  
6. Inventing Nsight / clean-A100 / energy-on-DICC numbers (not run).

---

## E. Preferred paper tables (pre-write)

1. Detection dual bars + sealed test.  
2. Option A per-block table (B1–B4) with B3 PT win highlighted.  
3. DICC multi-session stability for B3 / full PT.  
4. DICC multi-compiler absolute table (both GPUs).  
5. Laptop multi-compiler as **separate** subsection with environment label.

*End claim map.*
