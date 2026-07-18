# COLIDE — Status Report Draft (local evidence only)

**Audience:** Prof. Dr. Por Lip Yee (or internal PI status)  
**Draft date:** 2026-07-18  
**Repo HEAD at audit:** `803c157ddc1ed4103452f08044a7e0cd74cc728b` (audit pack); draft session on `master`  
**Authority:** `docs/DESIGN_PLAN.md` **Option A** + `docs/FINAL_PLAN.md`  
**Feedstock only:** `docs/audit/` (especially `11_REPORT_WRITER_BRIEF.md`, `04_CLAIMS_REGISTER.md`, `09_RAW_NUMBER_TABLES.md`)  

### Status of this document

| Item | Statement |
|------|-----------|
| What this is | Honest **local** status + **legacy** cluster notes + **empty** multi-day DICC cells |
| What this is **not** | Final “all numbers including multi-day DICC” email; not P2 complete for cluster |
| Non-authoritative prior draft | `docs/PROF_POR_STATUS_REPORT.md` — **do not treat as truth** |
| Hard gate before any final send | Codebase-wide numbers match + `verify_claims.py` green after any claim edits (`FINAL_PLAN` P2) |
| Multi-day DICC | **`benchmarks/results/dicc/` ABSENT** — cells left empty below |

---

## 1. Goal and claim scope (Option A — locked)

COLIDE is positioned as a **systems / measurement** contribution: custom CUDA inference blocks for a CNN–BiLSTM IDS classifier, multi-framework latency protocol, and low-overhead LLM dispatch for explanations — **not** as an accuracy-SOTA paper.

**Option A (approved):** valid claims are **per-block** Custom CUDA vs matching PyTorch modules (flagship: Block 3 BiLSTM, last-timestep contract).  

**Forbidden under Option A:**

- Full-pipeline Custom CUDA vs **full** production PyTorch V3 as apples-to-apples speedup.  
- Language that custom CUDA and eager V3 are “**the same computation**” — they are **not** (V3 has MultiheadAttention + residual LayerNorm + global average pool; CUDA path is incomplete and uses last-timestep reduce; fused pipeline times B3 as a separate addend, not a true device chain into B4).  
- Cross-hardware “vs PyTorch” ratios from June 2026 legacy DICC (CUDA-only; no same-GPU PyTorch that day).

Framework absolute latencies (eager / compile / TRT / ORT) may be reported as **what the production V3 classifier pays** on each stack; custom CUDA must be scoped to **implemented blocks**, with pipeline totals labeled as **derived/additive** where applicable.

---

## 2. Accuracy (frozen local numbers)

Sources: on-disk JSON under `benchmarks/results/` as inventoried in the audit pack. Champion checkpoint md5 confirmed on disk this audit.

| Metric | Value | Source | Label |
|--------|------:|--------|-------|
| BoT-IoT two-stage CNN-BiLSTM test macro-F1 | **0.9790** | `twostage_botiot.json` | CURRENT, HIGH |
| Champion checkpoint md5 | **`80a90f7cc210276300eaa90173a5a385`** | `model/best_model_botiot_twostage.pth` | CURRENT, confirmed |
| sklearn RF (apples-to-apples processed features) test macro-F1 | **0.9864** | `rf_baseline_processed.json` | CURRENT published bar |
| Accuracy gap (RF − CNN) | **0.74%** | computed from above | CURRENT |
| Stage-1 KD winner (α=0.6, T=10, γ=2) test F1 | **0.9763** | `distill_botiot_a0.6_T10.0_focal2.json` | CURRENT; bit-identical repro matches |
| MLP two-stage | **0.9542** | `mlp_twostage.json` | CURRENT (not champion) |
| Ensemble KD | **0.9529** | `ensemble_distill.json` | CURRENT (not champion) |
| ToN-IoT clean CNN-BiLSTM | **0.9526** | `toniot_clean_*` | CURRENT |
| ToN-IoT clean RF | **0.9851** | `toniot_clean_comparison` | CURRENT |
| ToN-IoT gap | **~3.25–3.3%** | comparison JSON | CURRENT |

**Honesty on RF:** We **do not** claim beating RF or SOTA accuracy. RF remains higher on BoT-IoT hard labels (**0.9864 vs 0.9790**). A diagnostic strengthen sweep reached ~**0.9885** (`rf_teacher_strengthen.json`); the **published RF bar stays 0.9864**.

Superseded champion path **0.9639** is historical only and must not be stated as current.

---

## 3. Laptop latency (WSL2 RTX 3050) — ranges, not lucky points

Measurement stability is a first-class result: **session-to-session drift on WSL2** is documented. Within-session CV understates uncertainty. All primary laptop latency claims use **multi-session ranges**.

### 3.1 Framework / custom-block composition (batch=1 protocol)

| Path | Latency range (µs) | Notes |
|------|-------------------:|-------|
| Custom CUDA FP16 **derived** pipeline | **594–675** | Multi-session composition; not a single lucky point |
| Eager PyTorch (full V3) | **2,050–2,247** | Framework absolute (production model) |
| torch.compile | **1,519–1,777** | Framework absolute |
| TensorRT | **2,427–2,966** | Framework absolute |
| ONNX Runtime GPU | **3,862–4,652** | Framework absolute |
| ONNX Runtime CPU | **487–699** | Straddles parity; significance **not robust** across sessions |

**Option A framing:** Framework columns time **full V3**. Custom CUDA totals are **implemented-block composition** (derived/additive: B1+B2+B4 chain + B3 timed separately). They are **not** the same computation as full V3 eager/TRT/ORT. Do **not** lead status or abstract with “full-model Custom CUDA × vs full V3” speedups.

If ranges vs frameworks are cited at all, they must carry the Option A caveat above (audit: vs TRT ~3.60×–4.99×, vs compile ~2.25×–2.99×, vs eager ~3.04×–3.78× are **range math under incomplete CUDA scope**, not parity-safe full-model claims).

### 3.2 Block 3 (flagship per-block result)

| Metric | Value | Source |
|--------|------:|--------|
| cuDNN / PyTorch Block 3 baseline | **784 µs** (n=50) | `pytorch_block3_stats_rtx3050.json` |
| Custom CUDA Block 3 FP16 | **532–602 µs** | multi-session |
| Local “beats cuDNN” | **1.30×–1.47×** | 784 / FP16 range |

**Caveat:** Local beats-cuDNN is an **RTX 3050 / this protocol** finding. Portability to server GPUs is **unconfirmed** on official UM DICC (see §5). ORT CPU is **not** a clean “beats all frameworks” win.

### 3.3 Other systems metrics (local)

| Metric | Value | Source |
|--------|------:|--------|
| LLM dispatch overhead p99 | **16.60 µs** (n=5000) | `llm_explainability.json` |
| LLM generation | multi-second async; quality **illustrative** | same |
| Streaming throughput | **~25,899** flows/s (batch=128) | `streaming_throughput.json` |
| Energy (RTX) | **~0.79 mJ/flow** | `energy_efficiency.json` |
| Energy (A100, separate run) | **~1.089 mJ/flow** | `a100_energy.json` |
| cuML RF energy (A100) | **~0.048 mJ/flow**; much higher VRAM (**~444 MB** vs CNN **~2 MB**) | `cuml_rf_resources.json` |

---

## 4. Numerical fidelity

| Check | Result | Source |
|-------|--------|--------|
| Weight export path | bit-identical, max abs error **0** (n=10) | `numerical_fidelity.json` |
| CUDA self-checks | **6/6 PASS** at disclosed tolerances (FP16 **5e-2**) | `numerical_fidelity.json` |

**Contract note:** Fidelity validates the **implemented CUDA contract** (V2 last-timestep path), **not** full V3 attention + GAP forward. This reinforces Option A scoping.

---

## 5. Cluster (UM DICC) — multi-day empty; June legacy only

**Official campaign site:** UM DICC only. Rostam = tooling trial only (not paper-final, not UM official).

### 5.1 Multi-day campaign cells (REQUIRED EMPTY)

Path checked this audit: `benchmarks/results/dicc/` → **ABSENT**. Campaign is **scripted** (`dicc_scripts/run_campaign.sh`) but **not completed** with SUCCESS artifacts on the laptop.

| Cell | Value |
|------|-------|
| Day1 V100 SUCCESS | **EMPTY / ABSENT** |
| Day2 V100 SUCCESS | **EMPTY / ABSENT** |
| Day1 A100 SUCCESS | **EMPTY / ABSENT** |
| Day2 A100 SUCCESS | **EMPTY / ABSENT** |
| `compare_dicc_sessions.py` accept | **NOT RUN** |
| Block 3 CUDA vs PT same-GPU (cluster) | **EMPTY** |
| Full V3 PT absolute (cluster) | **EMPTY** |

**We do not invent multi-day means, CVs, or Day1/Day2 compare results.**

### 5.2 June 2026 — LEGACY single-shot only (if cited at all)

| GPU | Pipeline total (µs) | Jobs | Label |
|-----|--------------------:|------|-------|
| V100S | **~551** (550.664) | 363046 | **LEGACY single-shot**, CUDA kernels only |
| A100 | **~592** (592.044) | 363047 | **LEGACY single-shot**, CUDA kernels only |

Sources: `dicc_v100_summary.txt`, `dicc_a100_summary.txt`. Validation PASSED in those summaries.  

**Not multi-day. No same-GPU PyTorch that day. Not compare-gated. Not a vs-PyTorch ratio.**

### 5.3 Portability risk (tooling-only peek — not UM official)

A Rostam Day 1 tooling trial (not paper-final) provisionally suggested Block 3 custom CUDA **slower** than PT on V100/A100 (~581 vs ~512 µs V100; ~706 vs ~353 µs A100 per design-plan notes). **Must be confirmed or refuted on UM DICC** before any portable “beats cuDNN” claim.

---

## 6. Evidence hygiene and tooling

| Item | Status |
|------|--------|
| `scripts/verify_claims.py` (this audit) | **66 claims PASSED; 0 FAILED; 0 regressions; EXIT 0** |
| Bold numbers in README not covered by claim manifest | 0.6, 1.00x, 10.0, 2.0, 3.3% (KD hypers / ToN gap formatting) |
| Provenance risk | `.gitignore` ignores `benchmarks/results/`; several load-bearing claim JSONs exist **on disk only** (e.g. `twostage_botiot.json`, `rf_baseline_processed.json`, CUDA/B3 stats, fidelity, round-2 KD). Clean clone may not regenerate README numbers without those files or re-runs. |
| Historical re-verification | Fabricated/unsourced figures (e.g. LLM 5.19 µs, 2.76× pipeline, cross-HW DICC ratios) were **found and fixed** in July 2026 commits; current SoT is JSON-backed where claimed. |
| Verifier limitation | Green `verify_claims` checks **number presence**, not Option A construct validity. “Same computation” language can still be invalid while claims pass. |

---

## 7. Threats to validity (summary for status)

1. **Accuracy:** RF still higher (0.9864 vs 0.9790); systems story is the contribution.  
2. **WSL2 drift:** laptop latency must be **ranges**.  
3. **Architecture parity:** custom CUDA ≠ full V3; no “same computation.”  
4. **Derived pipeline totals:** additive composition, not necessarily true full-device pipeline.  
5. **Cluster multi-day:** **pending** (ops / Prof decision).  
6. **Portable beats-cuDNN:** local only until UM confirms; Rostam provisional opposite direction.  
7. **ORT CPU:** non-robust significance.  
8. **LLM:** dispatch is the systems result; generation quality illustrative.  
9. **Packaging:** gitignored claim sources → repro risk for coauthors.  
10. **June 551/592:** legacy single-shot only.

---

## 8. Planned work left (from `docs/audit/07_PLANNED_WORK_LEFT.md` only)

Sources: `FINAL_PLAN.md`, `PROF_POR_3DAY.md`, `DESIGN_PLAN.md`, `HANDOFF.md`. No invented deadlines beyond those docs.

| Phase | Status | Content |
|-------|--------|---------|
| **Strategy freeze** | **DONE / LOCKED** | Option A; champion freeze; UM DICC only; numbers-match gate before final email |
| **P0 — Unblock DICC** | **BLOCKED / pending human** | Prof decision on Cheran/cluster help; no credential sharing; exit = ≥1 GPU class Day1 SUCCESS |
| **P1 — Multi-day + artifacts home** | **NOT DONE** | Day2 SUCCESS; compare accept; scp `benchmarks/results/dicc/` to laptop (**ABSENT** today) |
| **P2 — Extract + numbers match + Prof report** | **BLOCKED on P1** for full pack | P2a extract from DICC JSON; P2b codebase-wide match; P2c verify_claims; P2d send — **final send forbidden until P2a–c** |
| **P2c only (local)** | **PASS this audit** | Necessary, not sufficient for multi-day-inclusive final email |
| **P3 — Residual hygiene** | After P2 | Residual only |
| **P4 — Manuscript spine** | **EXPLICITLY DEFERRED** until P2 complete | |
| **P5 — Stretch** | **DEFERRED / OPTIONAL** | Nsight, full frameworks on DICC, Option B full V3 CUDA, retrain to close RF gap — out of freeze path |

**Next concrete step while multi-day cells are empty:** await Prof decision on Cheran/DICC access; if agreed, run campaign card (P0→P1); when SUCCESS tree lands on laptop, run P2a–P2d with empty-cell discipline replaced by JSON-only fills.

**This draft does not claim P2 complete for multi-day DICC.**

---

## 9. Optional / unplanned improvements (auditor suggestions only)

From `docs/audit/08_UNPLANNED_IMPROVEMENTS.md` — **not** approved gates unless absorbed into FINAL_PLAN later:

| ID | Optional idea |
|----|----------------|
| U3 | Lint / fix residual “same computation” full-pipeline wording (truthfulness) |
| U1 | Force-add curated claim-source JSONs or publish checksum manifest (repro packaging) |
| U13 | Split abstract tables: (a) per-block CUDA (b) full V3 framework absolutes |
| U2 | Persist multi-session latency files instead of hardcoding HIST/S3A/EXTRA in verify_claims |
| U4 | Multi-session ranges for B1/B2/B4 (README point values drift vs live stats) |
| U15 | Banner/supersede non-authoritative `PROF_POR_STATUS_REPORT.md` |

Priority if only three optional items: **U3 → U1 → U13**. Multi-day DICC remains **planned** critical path, not optional.

---

## 10. What we need from Prof / ops (questions)

1. Decision on Cheran (or other campus-stable runner) helping with **UM DICC** campaign scripts (manuscript role vs ops)?  
2. Timeline expectation for Day1+Day2 SUCCESS + compare deliverable to laptop?  
3. Preferred status cadence: **local-only honest updates** (this draft) vs wait for full multi-day pack?  
4. Whether to disclose Rostam provisional B3 direction now as risk, or wait for UM confirm/refute?

---

## Appendix A — Multi-day table template (leave empty until artifacts)

| GPU | Day1 CUDA B3 FP16 | Day1 PT B3 | Day2 CUDA B3 FP16 | Day2 PT B3 | Compare | Full V3 PT abs |
|-----|-------------------|------------|-------------------|------------|---------|----------------|
| V100S | — | — | — | — | NOT RUN | — |
| A100 | — | — | — | — | NOT RUN | — |

## Appendix B — Source map (report topics → audit feedstock)

| Topic | Primary sources (via audit) |
|-------|-----------------------------|
| Accuracy champion | `twostage_botiot.json`; md5sum; `04` / `09` |
| RF / gap | `rf_baseline_processed.json` |
| KD path | `distill_botiot_a0.6_T10.0_focal2*.json` |
| Framework ranges | multi-session composition / `verify_claims` HIST+S3A+live |
| Block 3 | `cuda_kernel_stats_rtx3050.json`; `pytorch_block3_stats_rtx3050.json` |
| LLM / stream / energy | `llm_explainability.json`; `streaming_throughput.json`; energy JSONs |
| Fidelity | `numerical_fidelity.json` |
| Legacy DICC | `dicc_v100_summary.txt`; `dicc_a100_summary.txt` |
| Option A / invalid | `DESIGN_PLAN.md` §5; `FINAL_PLAN.md` §1; `06_CONTRADICTIONS.md` |
| Planned work | `07_PLANNED_WORK_LEFT.md` |
| Gaps | `10_EVIDENCE_GAPS.md` |

---

*End of draft. Not for final send until P1 multi-day artifacts exist (if email claims cluster multi-day numbers) and P2 numbers-match + verify_claims gates pass on the locked table.*
