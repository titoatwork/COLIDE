# COLIDE Execution Plan Pack — Master Index

**Created:** 2026-07-19  
**Purpose:** Exceptional, end-to-end plan to meet Prof. Por’s WoS-level expectations (`docs/feedback1.docx`) while respecting Option A CUDA claim discipline.  
**Standard:** Not “just enough.” Every Prof requirement mapped to codebase reality, gaps, work packages, acceptance criteria, and quality bar.  
**Folder:** `docs/execution_plan/`

---

## Authority stack (read in this order)

| Priority | Document |
|----------|----------|
| 1 | `docs/feedback1.docx` — Prof technical plan |
| 2 | **This pack** (`docs/execution_plan/`) — operational plan |
| 3 | `docs/PROF_FEEDBACK_ROADMAP.md` — short phase summary |
| 4 | `docs/MOD_DECISION_TABLE.md` — modification IDs |
| 5 | `docs/DESIGN_PLAN.md` Option A — CUDA claim rules (still binding) |
| 6 | `docs/FINAL_PLAN.md` — DICC ops / numbers-match gates |
| 7 | `docs/audit/` — evidence feedstock for frozen local numbers |

---

## Pack contents

| File | Contents |
|------|----------|
| `00_INDEX.md` | This file |
| **`PROF_FEEDBACK_TRACKER.md`** | **Live checklist of every Prof requirement (update always)** |
| `01_CODEBASE_INVENTORY.md` | Every major area of the repo; what each file family does |
| `02_FEEDBACK_DEEP_ANALYSIS.md` | Line-by-line interpretation of Prof feedback + implications |
| `03_GAP_MATRIX.md` | Prof requirement × current code/results × gap × severity |
| `04_PHASE0_DICC.md` | Multi-day UM DICC — full protocol (HARD GATE) |
| `05_PHASE1_BASELINE.md` | Reproducible baseline freeze, multi-run protocol |
| `06_PHASE2_METHOD.md` | One clear methodological contribution |
| `07_PHASE3_HPO.md` | Systematic hyperparameter optimisation |
| `08_PHASE4_IMBALANCE_TEACHER.md` | Imbalance strategies + distillation teachers |
| `09_PHASE5_ABLATIONS_BASELINES.md` | Ablations + fair ML/DL baselines + multi-objective |
| `10_PHASE6_DEPLOY_CUDA.md` | Export, parity, kernels, frameworks, precision |
| `11_PHASE7_XAI.md` | Beyond 16.60 µs — faithfulness and comparisons |
| `12_PHASE8_MULTIDATASET.md` | ToN-IoT / second dataset |
| `13_PHASE9_MANUSCRIPT.md` | Paper structure, tables, RQs, abstract rules |
| `14_EXCEPTIONAL_STANDARDS.md` | Quality bar: stats, repro, anti-sloppiness |
| `15_WORK_PACKAGES.md` | Chat-sized packages, dependencies, exit criteria |
| `16_SAFETY_AND_RULES.md` | Champion backup, no invented numbers, Option A |

---

## Golden path (do not reorder casually)

```text
0 DICC multi-day UM          ← CURRENT BLOCKER (artifacts ABSENT)
1 Baseline freeze + multi-run
2 Choose ONE method package (decision table)
3 HPO on validation only
4 Imbalance + teacher under that method
5 Ablations + baselines + Pareto
6 Deploy / CUDA re-bench (Option A)
7 XAI quality (if title claims explainable)
8 Second dataset (ToN)
9 Numbers-match + manuscript
```

**Parallel only while 0 waits:** inventory, scripts scaffolding, claim-JSON packaging, decision-table refinement — **not** bulk retrain, **not** manuscript freeze of cluster numbers.

---

## Current frozen reference (local — interim)

| Item | Value |
|------|-------|
| Champion | `model/best_model_botiot_twostage.pth` |
| md5 | `80a90f7cc210276300eaa90173a5a385` |
| Test macro-F1 | 0.9790 |
| RF bar | 0.9864 (gap 0.74%) |
| Multi-day DICC | **ABSENT** |

These are **baseline references**, not the final paper champion until Phases 1–5 complete under Prof protocol.

---

## How to use this pack in every session

1. Open `00_INDEX.md` + active phase file.  
2. Do **one** work package from `15_WORK_PACKAGES.md`.  
3. Update phase file status checkboxes.  
4. Never skip acceptance criteria for speed.  
5. End session: verify → commit → HANDOFF next prompt.

---

## Success definition (Prof bar, not “done-ish”)

A submission-ready package when **all** hold:

1. Clear quantitative advantage on ≥1 major dimension (or strong Pareto).  
2. Method justified; val-only selection; untouched test.  
3. Ablations + fair baselines complete.  
4. DICC multi-day + same-GPU stats complete.  
5. Option A CUDA claims only.  
6. Minority-class reporting first-class.  
7. XAI depth matches title.  
8. Reproducible artifacts (not only gitignored laptop files).  
9. Paper answers all five RQs Prof listed.

---

*This pack is the operational spine for exceptional execution.*
