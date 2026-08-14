# Claim map — pre-write authority

**Date:** 2026-08-14 (Phase 1 claim hygiene + Phase 2 corrected ToN)  
**Use:** manuscript drafting only. Every number must trace to JSON or a locked doc table.  
**Champion:** md5 `80a90f7…`  
**Also read:** `docs/ISSUE_REGISTER.md` · `docs/KNOWN_LIMITATIONS.md` · `docs/PRE_MANUSCRIPT_CLOSURE.md` · `docs/REMEDIATION_STATUS.md`  
**External readiness review:** `COLIDE_Remediation_Update_Review.md` (Phases 1–8; project **not** submission-ready while CUDA evidence + publication sync pending)

**Closure posture:** DATA remediation closed; CUDA evidence and publication synchronization **pending**. Do not draft as if remediation is fully closed.

---

## A. Detection / multi-objective (local sealed protocol)

| Claim | Status | Source |
|-------|--------|--------|
| Sealed multi-seed test macro-F1 **0.9780 ± 0.0033** | **OK** (principal) | `sealed_test/summary.json` |
| Historical single-run **0.9790** as principal / headline accuracy | **FORBIDDEN** | Use only as historical/legacy development result |
| Historical single-run **0.9790** labeled development/legacy | **OK** | `twostage_botiot.json` (legacy path) |
| Not pure-F1 SOTA vs protocol LGBM **0.9818** | **OK** | dual-bar framing |
| Multi-obj efficiency path (local WP6b ranges) | **OK** | energy / PT@256 / peak mem |
| Full free-form LLM explainability as title claim | **FORBIDDEN** | dispatch-only **16.60 µs p99** |
| ToN corrected leakage-safe CNN **0.8075** / RF **0.9626** (`toniot_leakage_safe_v1`, 13 allowlist, no SMOTE/KD) | **OK** | `toniot_corrected/summary.json` (`valid: true`) |
| ToN “clean” CNN **0.9526** / RF **0.9851** / **+15.4%** as active evidence | **FORBIDDEN** | **INVALID** — DATA-TON-001; tombstoned JSON |
| ToN 13-feat neural ~**0.811** vs same-split RF ~**0.939** (older package path, labeled) | **OK** | `toniot_final/summary.json` (comparable prior; prefer corrected for multi-dataset) |

---

## B. Option A Custom CUDA (DICC multi-session)

| Claim | Status | Source |
|-------|--------|--------|
| B1/B2/B4 CUDA much faster than matching PT on V100S/A100 | **OK** | extraction tables (matched operator-vs-operator) |
| B3 CUDA FP16 **slower** than matching PT B3 (both GPUs, 3 sessions) as wall-clock of **pre_fix** historical binaries | **OK — required honesty** | B3 report + extraction; label **pre_fix** / not post_fix parity |
| B3 CUDA vs PT as closed matching-op / post_fix parity result | **FORBIDDEN** until CUDA-B3-001/002/003 rebench + production-weight parity green | Issue register |
| Session-stable B3 means (**pre_fix** trees) | **OK** | max spread ≪ 2% B3 |
| Full Custom CUDA pipeline vs full V3 PT speedup | **FORBIDDEN** | Option A / CLAIM-PIPE-001 |
| Any ratio of **partial** custom pipeline (Blocks 1–4 sum) vs full V3 / eager / compile / TRT / ORT | **FORBIDDEN** | CLAIM-PIPE-001; review Phase 1 |
| Portable “CUDA BiLSTM beats PT/cuDNN on servers” | **FORBIDDEN** | PT wins B3 wall-clock **pre_fix** |
| All metrics session-stable S1–Day2 without caveat | **FORBIDDEN** | V100 B1 11% formal spread |

### B3 Welch-style summary (S1/S2/Day2, trial distributions) — **pre_fix** wall-clock only

Protocol caveat: CUDA n=100 kernel trials vs PT n=20 subprocess trials — different harnesses; direction is unambiguous for wall-clock. Source race+align fixed; **production-weight parity open**. Numbers describe **pre_fix** historical binaries until rebench.

| GPU | Session | CUDA mean±std | PT mean±std | Welch t | Cohen d | Winner |
|-----|---------|---------------|-------------|---------|---------|--------|
| V100S | S1 | 513.3±0.8 | 363.5±12.6 | 53 | ~29 | **PT** |
| V100S | S2 | 513.0±0.7 | 363.6±4.8 | 139 | ~74 | **PT** |
| V100S | Day2 | 513.1±0.7 | 363.3±6.2 | 108 | ~58 | **PT** |
| A100 | S1 | 668.0±18.3 | 383.7±5.0 | 133 | ~17 | **PT** |
| A100 | S2 | 667.4±18.7 | 389.0±6.9 | 115 | ~16 | **PT** |
| A100 | Day2 | 671.2±20.0 | 390.9±6.7 | 112 | ~15 | **PT** |

---

## C. Multi-compiler and laptop tables (strict separation)

### Rule (CLAIM-PIPE-001 / review Phase 1)

| Table class | What is allowed |
|-------------|-----------------|
| **Matched operator-vs-operator** | Per-block Custom CUDA vs matching PyTorch op (B1–B4). Ratios **OK within** this class only. |
| **Full-model-vs-full-model** | Eager / compile / ORT / TRT (and ORT CPU) absolute latencies. Ratios **OK within** this class only. |
| **Partial custom pipeline sum** | Absolute ranges only (e.g. Blocks 1–4 sum). **No** ratios vs full-model rows. |

**FORBIDDEN:** any speedup or ratio computed **across** partial custom pipeline and full V3 / eager / compile / TRT / ORT tables.

### C1 Laptop (RTX 3050 / WSL)

| Claim | Status |
|-------|--------|
| Table A: Custom CUDA Blocks 1–4 sum absolute multi-session range **594–675 µs** (incomplete scope labeled) | **OK** (absolute only) |
| Table B: Full-model eager / compile / TRT / ORT multi-session absolute ranges | **OK** (full-model only) |
| Ratios between Table A (partial) and Table B (full model), e.g. “3.60×–4.99× over TensorRT” | **FORBIDDEN** |
| Same laptop numbers portable to DICC without remeasure | **FORBIDDEN** |

### C2 DICC multi-compiler matrix (batch-1 absolutes, n=20)

| Method | V100S mean µs | A100 mean µs | Status |
|--------|--------------:|-------------:|--------|
| Eager | 1041 | 931.5 (~**932**) | **OK** |
| torch.compile | 865 | 770 | **OK** |
| ORT CUDA | 895 | 865 | **OK** |
| ORT CPU | 500 | 461 | **OK** (not a GPU deploy claim) |
| ORT TRT EP | 766 | 2033 | **OK** (active TRT EP; A100 slow) |
| TRT native FP16 | **528** | **588** | **OK** (fastest GPU path here) |

Source: `docs/DICC_MULTI_COMPILER_MATRIX.md` · jobs 395433 / 395417.  
**OK** as full-model-vs-full-model only. **FORBIDDEN** to ratio against incomplete Custom CUDA pipeline sums.

---

## D. Streaming / energy / bulk throughput

| Claim | Status | Source / note |
|-------|--------|----------------|
| ~**25,899** f/s as **bulk batched** processing throughput (batch=128, RTX 3050) | **OK** | `streaming_throughput.json`; must say bulk, not stream arrivals |
| Same figure as controlled streaming / offered-load / saturation / sustainable stream rate | **FORBIDDEN** | BENCH-STREAM-001 |
| WP6b energy **0.920–0.943** mJ/flow as multi-session laptop range | **OK** | `wp6b_local_ranges/summary.json` |
| A100 **1.089** mJ/flow or RTX **0.79** as exploratory board-power estimate (caveated) | **OK** if labeled exploratory | ENERGY-001 |
| **1.089** as controlled efficiency win/loss vs cuML without caveats | **FORBIDDEN** | ENERGY-001 |
| “CPU energy” inferred only from GPU-board NVML samples | **FORBIDDEN** | |

---

## E. Explicitly do **not** write

1. Full-pipeline Custom CUDA × vs full V3.  
2. **Any ratio of partial custom pipeline (Blocks 1–4) vs full V3 / eager / compile / TRT / ORT.**  
3. “CUDA Block 3 beats PyTorch on V100/A100.”  
4. Mixing laptop framework numbers with DICC absolute µs in one headline.  
5. Treating ORT CPU win as the main GPU systems result.  
6. Claiming multi-day “all metrics stable” without B1 caveat.  
7. Inventing Nsight / clean-A100 / energy-on-DICC numbers (not run).  
8. ToN clean **0.9526** / **0.9851** / **+15.4%** as valid accuracy evidence (**INVALID** / tombstone).  
9. Principal BoT accuracy as bare **0.9790** without historical/legacy label (use sealed **0.9780 ± 0.0033**).  
10. “Streaming latency” or paced-arrival claims from the bulk harness.  
11. Title-level validated LLM explainability.  
12. Post_fix B3 matching-op claims before production-weight parity + rebench green.  
13. “Remediation fully closed / submission-ready” while CUDA evidence + publication sync pending.

---

## F. Preferred paper tables (pre-write)

1. Detection dual bars + sealed test (**0.9780 ± 0.0033**).  
2. Option A **per-block** table (B1–B4) with B3 PT win highlighted (**pre_fix** label).  
3. DICC multi-session stability for B3 / full PT (**pre_fix** until rebench).  
4. DICC multi-compiler absolute table (both GPUs, batch-1) — full-model only.  
5. Laptop: **separate** Table A (incomplete Custom CUDA absolute) and Table B (full-model absolute); no cross ratios.  
6. ToN corrected leakage-safe (**0.8075** CNN / **0.9626** RF); optional older 13-feat ~0.811 vs ~0.939 as comparable prior.  
7. Bulk throughput + exploratory energy appendix (not streaming chapter).

*End claim map.*
