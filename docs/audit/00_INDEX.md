# COLIDE Evidence Audit Pack — Master Index

**Audit date:** 2026-07-18  
**Repo HEAD inspected:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`  
**Branch:** `master` (tracking `origin/master`)  
**Auditor role:** forensic feedstock only — NOT a professor report  
**Authority:** Option A (`docs/DESIGN_PLAN.md`) + `docs/FINAL_PLAN.md`  

## Confidence legend

| Label | Meaning |
|-------|---------|
| **HIGH** | Number matches on-disk JSON/txt; producer script known; re-verification history exists OR bit-repro shown |
| **MED** | On-disk source exists but single-shot / multi-session partially hardcoded / gitignored source |
| **LOW** | Doc-only, historical single-run, cross-hardware contaminated (fixed), or construct-invalid if claimed |
| **CURRENT** | Present production claim surface uses this |
| **LEGACY** | Historical; may still be cited only with label |
| **SUPERSEDED** | Replaced by later re-verification |
| **INVALID** | Must not be stated as claimed (Option A or known fabrication fixed) |
| **ABSENT** | Path checked; artifact missing |
| **PLANNED** | Scripted/planned; not run |
| **OPTIONAL / UNPLANNED** | Auditor suggestion; not approved work |

## For report-writer LLM: start here

**Reading order (do not skip):**

1. **This file** — constraints and non-claims  
2. **`11_REPORT_WRITER_BRIEF.md`** — Facts you may / must NOT state  
3. **`04_CLAIMS_REGISTER.md`** — claim → source → confidence  
4. **`09_RAW_NUMBER_TABLES.md`** — dense copy-paste metrics  
5. **`05_GIT_REVERIFICATION.md`** — what was wrong before and what fixed it  
6. **`06_CONTRADICTIONS.md`** — Option A / README risks  
7. **`02_RESULTS_INVENTORY.md`** — every results file  
8. **`10_EVIDENCE_GAPS.md`** — multi-day DICC ABSENT; gitignore holes  
9. **`07_PLANNED_WORK_LEFT.md`** — only from FINAL_PLAN/HANDOFF  
10. **`08_UNPLANNED_IMPROVEMENTS.md`** — optional only  
11. **`01_REPO_MAP.md`**, **`03_PRODUCER_GRAPH.md`** — architecture / producers  

## Absolute non-negotiables for any downstream report

1. **Do NOT invent multi-day DICC numbers.** Path checked: `benchmarks/results/dicc/` → **ABSENT**.  
2. June 2026 **551 µs (V100S) / 592 µs (A100)** = **LEGACY single-shot** CUDA-only (`dicc_*_summary.txt`). No same-GPU PyTorch that day.  
3. **Option A:** valid claims are **per-block** Custom CUDA vs matching PyTorch modules. **FORBIDDEN:** full-pipeline Custom CUDA vs full PyTorch V3 as apples-to-apples speedup.  
4. Champion: `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`** — **CONFIRMED on disk this audit**.  
5. `docs/PROF_POR_STATUS_REPORT.md` = mistaken prior draft → **NON-AUTHORITATIVE**.  
6. RF published bar stays **0.9864** (not strengthen-sweep 0.9885).  
7. `scripts/verify_claims.py` this audit: **66 claim(s) PASSED; 0 FAILED; 5 bolded numbers not covered (0.6, 1.00x, 10.0, 2.0, 3.3%); RESULT: all tracked claims verified, no regressions; EXIT 0**

## Critical meta-finding (read before citing)

`.gitignore` line 23: `benchmarks/results/` — entire results directory is **ignored by default**.  
37 files were force-tracked historically; **17 load-bearing files exist only on disk (untracked/gitignored)**, including:

- `cuda_kernel_stats_rtx3050.json` (multi-session CUDA means)
- `twostage_botiot.json` (0.9790 source)
- `rf_baseline_processed.json` (0.9864 source)
- `pytorch_block3_stats_rtx3050.json` (784 µs cuDNN baseline)
- `numerical_fidelity.json`
- Round-2 KD sweep JSONs (`a0.6_T10.0_focal2`, repro, T=7/10 grid, focal γ sweeps)
- `rf_teacher_strengthen.json`

Verify_claims **can** read them locally; a clean clone from GitHub **cannot** regenerate README numbers without these files or re-runs. Flag as **provenance / packaging risk**.

## Audit completion checklist

| Gate | Status |
|------|--------|
| Every benchmarks/results file listed/parsed | YES (54 files) |
| verify_claims.py run + gap analysis | YES (66 pass, 5 bold gaps) |
| Claim register README + paper_text_blocks | YES (see 04) |
| Git archaeology with real hashes | YES (see 05) |
| Champion md5 on disk | YES `80a90f7cc210276300eaa90173a5a385` |
| Multi-day DICC path checked | YES **ABSENT** |
| Planned work cites FINAL_PLAN/HANDOFF | YES (07) |
| Unplanned labeled optional | YES (08) |
| No invented multi-day numbers | YES |
| HANDOFF + commit audit docs | (session close) |

## Pack file list

| File | Role |
|------|------|
| `00_INDEX.md` | This file |
| `01_REPO_MAP.md` | Structure, models, kernels, configs |
| `02_RESULTS_INVENTORY.md` | Every results artifact |
| `03_PRODUCER_GRAPH.md` | Script → JSON → claim |
| `04_CLAIMS_REGISTER.md` | Claim register |
| `05_GIT_REVERIFICATION.md` | History archaeology |
| `06_CONTRADICTIONS.md` | Cross-surface mismatches |
| `07_PLANNED_WORK_LEFT.md` | From plans only |
| `08_UNPLANNED_IMPROVEMENTS.md` | Optional |
| `09_RAW_NUMBER_TABLES.md` | Dense metrics |
| `10_EVIDENCE_GAPS.md` | Missing evidence |
| `11_REPORT_WRITER_BRIEF.md` | Machine brief for report LLM |
