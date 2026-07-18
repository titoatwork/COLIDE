# COLIDE — Interim Status (through current blocker only)

**Audience:** Prof. Dr. Por Lip Yee (or internal PI status)  
**Draft date:** 2026-07-18  
**Scope of this report:** Everything **done and frozen on the laptop**, up to the **current blocker**.  
**Out of scope for this report:** Multi-day UM DICC numbers, final “all cluster cells filled” pack, manuscript (P4).  

**Authority:** `docs/DESIGN_PLAN.md` **Option A** + `docs/FINAL_PLAN.md`  
**Evidence feedstock:** `docs/audit/` only (not freehand; not `docs/PROF_POR_STATUS_REPORT.md`)  

| This document is | This document is **not** |
|------------------|--------------------------|
| Interim status **through the DICC blocker** | A final email with multi-day cluster means |
| Local frozen numbers + Option A caveats | P2 complete for multi-day DICC |
| Honest empty multi-day cells + **leftover** work list | Invented cluster numbers |

---

## 1. Goal and claim scope (Option A — locked)

COLIDE is a **systems / measurement** effort: custom CUDA inference blocks for a CNN–BiLSTM IDS classifier, multi-framework latency protocol, and low-overhead LLM dispatch — **not** accuracy-SOTA.

**Option A (approved):** valid claims are **per-block** Custom CUDA vs matching PyTorch modules (flagship: **Block 3** BiLSTM, last-timestep contract).

**We do not claim:**

- Full-pipeline Custom CUDA vs **full** production PyTorch V3 as apples-to-apples speedup.  
- That custom CUDA and eager V3 are “**the same computation**” (V3 has attention + LayerNorm + GAP; CUDA path is incomplete + last-timestep; pipeline totals are derived/additive).  
- Cross-hardware “vs PyTorch” ratios from June 2026 legacy DICC (CUDA-only that day).

---

## 2. What is done (frozen local evidence)

### 2.1 Accuracy

| Metric | Value | Source |
|--------|------:|--------|
| BoT-IoT two-stage CNN-BiLSTM test macro-F1 | **0.9790** | `twostage_botiot.json` |
| Champion checkpoint md5 | **`80a90f7cc210276300eaa90173a5a385`** | `model/best_model_botiot_twostage.pth` |
| sklearn RF (apples-to-apples) test macro-F1 | **0.9864** | `rf_baseline_processed.json` |
| Gap (RF − CNN) | **0.74%** | computed |
| Stage-1 KD (α=0.6, T=10, γ=2) test F1 | **0.9763** | `distill_botiot_a0.6_T10.0_focal2.json` (+ bit-identical repro) |
| ToN-IoT clean CNN / RF | **0.9526** / **0.9851** (gap ~**3.25–3.3%**) | `toniot_clean_*` |

**RF honesty:** We **do not** claim beating RF. Published RF bar stays **0.9864** (a diagnostic strengthen sweep ~0.9885 is **not** the published bar).

### 2.2 Laptop latency (WSL2 RTX 3050) — ranges

Session-to-session drift is real; primary claims use **ranges**, not single lucky points.

| Path | Range (µs) | Note |
|------|-----------:|------|
| Custom CUDA FP16 **derived** pipeline | **594–675** | Implemented-block composition, not full V3 |
| Eager / compile / TRT / ORT-GPU / ORT-CPU | **2050–2247** / **1519–1777** / **2427–2966** / **3862–4652** / **487–699** | Full V3 framework absolutes; ORT CPU significance **not robust** |
| Block 3 FP16 | **532–602** | vs cuDNN baseline **784** µs (n=50) → local **1.30×–1.47×** (RTX 3050 / this protocol only) |

Framework numbers = production V3 tax. Custom CUDA = implemented blocks only (**Option A**).

### 2.3 LLM / streaming / energy (local)

| Metric | Value | Source |
|--------|------:|--------|
| LLM dispatch p99 | **16.60 µs** (n=5000) | `llm_explainability.json` (generation quality illustrative) |
| Streaming | **~25,899** flows/s (batch=128) | `streaming_throughput.json` |
| Energy | ~**0.79** mJ/flow (RTX); ~**1.089** mJ/flow (A100) | energy JSONs |
| cuML RF (A100) | ~**0.048** mJ/flow; ~**444 MB** VRAM vs CNN ~**2 MB** | `cuml_rf_resources.json` |

### 2.4 Numerical fidelity

| Check | Result |
|-------|--------|
| Export path | bit-identical (max abs **0**, n=10) |
| CUDA self-checks | **6/6 PASS** (FP16 tol **5e-2**) |

Contract validated = **CUDA last-timestep path**, not full V3 attention/GAP.

### 2.5 Claim tooling (local)

`scripts/verify_claims.py` (audit): **66 pass / 0 fail / 0 regressions**.  
Note: green verifier checks number presence, not Option A wording safety.

### 2.6 Legacy cluster note only (not multi-day)

June 2026 single jobs (CUDA-only, not multi-day, no same-GPU PyTorch that day):

| GPU | Pipeline total | Label |
|-----|---------------:|-------|
| V100S | **~551 µs** (550.664), job 363046 | **LEGACY single-shot** |
| A100 | **~592 µs** (592.044), job 363047 | **LEGACY single-shot** |

Sources: `dicc_v100_summary.txt`, `dicc_a100_summary.txt`. **Not** paper multi-day; **not** vs-PyTorch ratios.

---

## 3. Current blocker (where this report stops)

| Blocker | Detail |
|---------|--------|
| **What** | Official **multi-day UM DICC** campaign (Day1 + Day2 + compare) not completed; artifacts not on laptop |
| **Checked** | `benchmarks/results/dicc/` → **ABSENT** |
| **Why blocked** | Remote SSH / ops path unstable; **Prof decision** on Cheran (or other campus runner) pending; no invented numbers allowed |
| **Official site** | **UM DICC only** (Rostam = tooling trial only, not paper-final) |
| **Campaign scripts** | Ready (`dicc_scripts/run_campaign.sh`); **not a substitute for SUCCESS results** |

**Multi-day cells (intentionally empty in this interim report):**

| Cell | Status |
|------|--------|
| Day1 / Day2 V100 SUCCESS | **EMPTY** |
| Day1 / Day2 A100 SUCCESS | **EMPTY** |
| compare accept/reject | **NOT RUN** |
| Block 3 CUDA vs PT same-GPU (cluster) | **EMPTY** |
| Full V3 PT absolute (cluster) | **EMPTY** |

---

## 4. Leftover work (blocked on DICC / after unblock)

These items are **planned remaining work**, held as **leftover due to the DICC blocker**. They are **not** claimed as done in this report.

| # | Leftover (after unblock) | Depends on |
|---|--------------------------|------------|
| L1 | **P0** — Unblock runner (Prof/Cheran/user on UM DICC) | Human decision / access |
| L2 | **P1** — Day1 + Day2 SUCCESS; `compare_dicc_sessions.py`; scp `benchmarks/results/dicc/` home | L1 |
| L3 | **P2a** — Extract Block 3 CUDA vs PT + full V3 PT absolutes from JSON only | L2 |
| L4 | **P2b–c** — Codebase-wide numbers match + `verify_claims` green | L3 |
| L5 | **P2d** — Final Prof update with multi-day numbers | L4 |
| L6 | **P3 / P4 / P5** — residual hygiene, manuscript, stretch | After L5 |

**Hard gate before any final multi-day numbers email:** L4 complete (numbers match + verify_claims).  
**This interim report does not claim L2–L5.**

Optional polish while waiting (not required for this interim status): fix residual “same computation” wording; package claim-source JSONs for clone repro — see `docs/audit/08_UNPLANNED_IMPROVEMENTS.md`.

---

## 5. Short ask (if shared with Prof)

1. Local accuracy, latency ranges, fidelity, and Option A scope are **frozen and ready**.  
2. Multi-day DICC is the **remaining critical path**; currently **leftover / blocked**.  
3. Guidance on whether Cheran (or campus-stable runner) may run the campaign scripts would **unblock** L1–L5.  
4. No credential sharing; scripts and tarball path already documented in `docs/FINAL_PLAN.md` / `dicc_scripts/README.md`.

---

## Appendix — source map

| Topic | Primary sources |
|-------|-----------------|
| Accuracy / RF / KD | `twostage_botiot.json`, `rf_baseline_processed.json`, distill JSONs |
| Latency ranges / B3 | multi-session composition; `cuda_kernel_stats_rtx3050.json`; `pytorch_block3_stats_rtx3050.json` |
| LLM / stream / energy | `llm_explainability.json`, `streaming_throughput.json`, energy JSONs |
| Fidelity | `numerical_fidelity.json` |
| Legacy DICC | `dicc_*_summary.txt` |
| Option A / leftover plan | `DESIGN_PLAN.md` §5; `FINAL_PLAN.md` P0–P2; `docs/audit/07_PLANNED_WORK_LEFT.md` |

---

*End of interim report. Scope ends at the DICC blocker; multi-day campaign work remains leftover until unblocked.*
