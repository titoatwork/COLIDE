> **Status: HISTORICAL / NOT CURRENT AUTHORITY (frozen ~2026-08-12).**  
> Do not cite this file as the live claim surface. Current authority: `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/RESULTS_INDEX.md`, `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
> The header **PRE-MANUSCRIPT FULLY CLOSED** below is **not** live truth (superseded 2026-08-14/15). CUDA evidence / publication sync remain open; see `docs/PRE_MANUSCRIPT_CLOSURE.md`.  
> Kept for audit trail. Stale numbers here may be superseded (ToN clean 0.9526 INVALID; principal BoT is 0.9780±0.0033; DICC B3 latency is pre_fix / Option B).  
> Public GitHub visitors: treat this as a frozen session log, not manuscript evidence.

# COLIDE — Session Handoff

**MODE:** ✅ **PRE-MANUSCRIPT FULLY CLOSED** · **next = manuscript writing only**  
**Date:** 2026-08-12  
**Git tip:** see `git log -1 --oneline` after push  
**Champion:** md5 **`80a90f7cc210276300eaa90173a5a385`** (never clobber)  
**Authority:** `docs/PRE_MANUSCRIPT_CLOSURE.md` · `docs/CLAIM_MAP_PREWRITE.md` · Option A  

---

## Pre-manuscript deliverables (complete)

| Item | Status | Where |
|------|--------|-------|
| Local science playlist | **DONE** | tracker / sealed_test / WP6b |
| DICC 6× SUCCESS S1/S2/Day2 | **DONE** | `benchmarks/results/dicc/core/` |
| Formal compares | **DONE** | `DICC_COMPARE_OUTCOMES.md` |
| B3 PT-win + Welch/d | **DONE** | `DICC_B3_CUDA_VS_PT_REPORT.md` |
| Full multi-compiler DICC | **DONE** | `DICC_MULTI_COMPILER_MATRIX.md` · jobs 395433/395417 |
| README / claim hygiene | **DONE** | `README.md` |
| Claim map for writing | **DONE** | `CLAIM_MAP_PREWRITE.md` |
| Tracker DICC stale notes | **DONE** | `PROF_FEEDBACK_TRACKER.md` |
| S1b clean A100 / Nsight / B3 optim | **DEFERRED** optional | not blocking |

### Multi-compiler means (µs)

| Method | V100S | A100 |
|--------|------:|-----:|
| Eager | 1041 | 932 |
| torch.compile | 865 | 770 |
| ORT CUDA | 895 | 865 |
| ORT CPU | 500 | 461 |
| ORT TRT EP | 766 | 2033 |
| **TRT native** | **528** | **588** |

---

## Locked science

- Option A: CUDA B1/B2/B4 wins; **PT wins B3** on servers; no full CUDA vs full V3.  
- DICC multi-compiler complete; native TRT fastest **GPU** framework path in matrix.  
- Laptop multi-compiler = separate environment.  

---

## Next phase (manuscript only)

1. Write multi-GPU + multi-compiler sections from `CLAIM_MAP_PREWRITE.md` tables.  
2. Rebuild PDF when prose lands.  
3. PI venue class file / BibTeX after journal choice.  
4. Optional science only if PI asks (S1b / Nsight / B3 optim).

---

## Paste-ready prompt for next chat

```text
Resume COLIDE — manuscript writing only (pre-manuscript CLOSED).

Do not re-run DICC campaign or invent numbers.
Champion md5 80a90f7cc210276300eaa90173a5a385 — never clobber.
Option A only.

Read first:
1. HANDOFF.md
2. docs/PRE_MANUSCRIPT_CLOSURE.md
3. docs/CLAIM_MAP_PREWRITE.md
4. docs/DICC_EXTRACTION_TABLES.md
5. docs/DICC_B3_CUDA_VS_PT_REPORT.md
6. docs/DICC_MULTI_COMPILER_MATRIX.md
7. docs/execution_plan/WP9b_MANUSCRIPT_SPINE.md
8. docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md

Your job:
1) Insert multi-GPU Option A tables (honest B3 PT win) and DICC multi-compiler table into manuscript.
2) Keep laptop multi-compiler as a separate local subsection.
3) No full CUDA vs full V3; no portable B3 CUDA win; no mixing laptop ratios with DICC absolutes.
4) Rebuild PDF; commit on master when coherent.

SSH dicc only if re-checking cluster artifacts.
```

*End handoff — pre-manuscript closed.*
