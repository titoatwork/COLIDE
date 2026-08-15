# Pre-manuscript index (closed pack)

**Date:** 2026-08-15  
**Status:** **PRE-MANUSCRIPT CLOSED** for evidence, claims, tables, figures, and writer handoff.  
**Next phase:** manuscript *writing* (Cheran leads) — not more experiments.

This file is the single start page for “are we done before the paper?”

---

## What “pre-manuscript” means here

| In scope (this pack) | Out of scope (manuscript phase) |
|----------------------|----------------------------------|
| Frozen champion + locked numbers | Venue-ready PDF / Word / LaTeX |
| Claim map OK / FORBIDDEN | Abstract polish, related-work rewrite |
| Result → artifact index | Author list, BibTeX, journal class |
| Corrected ToN + tombstones | New experiments (DICC rebench optional later) |
| Local CUDA parity + sanitizers | Camera-ready figure design |
| DICC historical tables + B3 Option B | Full paper narrative |
| Figures/tables generated from JSON | |
| Cheran handoff + reading pack | |

**We did produce manuscript *fragments*** so the writer is not starting from zero. They are **feedstock**, not a finished paper.

---

## Manuscript fragments (working drafts — keep, labeled)

| File | What it is | How to treat it |
|------|------------|-----------------|
| `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` | Long draft; numbers synced Aug 2026 | **Working draft** for Cheran to rewrite |
| `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf` | PDF of an **older** (22 Jul 2026) draft | **STALE** vs the MD — do not send as current |
| `docs/paper_text_blocks.md` | Reusable paragraphs | Feedstock; check claim map before paste |
| `docs/manuscript/TABLES_FROM_ARTIFACTS.md` | Tables copied from locked JSON | **Use** as table source of truth |
| `docs/manuscript/figures/` | CURRENT figures from artifacts | **Use** |
| `docs/COLIDE_Current_State_Reading_Pack.pdf` | 4-page human reading pack | For you / Cheran orientation |

---

## Pre-manuscript deliverables (complete)

| # | Deliverable | Path | Status |
|---|-------------|------|--------|
| 1 | Champion freeze | `model/best_model_botiot_twostage.pth` md5 `80a90f7…` | **DONE** |
| 2 | Claim map | `docs/CLAIM_MAP_PREWRITE.md` | **DONE** |
| 3 | Results index | `docs/RESULTS_INDEX.md` | **DONE** |
| 4 | Issue register | `docs/ISSUE_REGISTER.md` | **DONE** |
| 5 | Known limitations | `docs/KNOWN_LIMITATIONS.md` | **DONE** |
| 6 | Corrected ToN | `benchmarks/results/toniot_corrected/` CNN 0.8075 / RF 0.9626 | **DONE** |
| 7 | Invalid ToN tombstone | `toniot_clean_comparison.json` | **DONE** |
| 8 | Local B3 parity gate | `block3_parity_gate.json` `valid=true` | **DONE** |
| 9 | Local sanitizers | `sanitizer_b3/` | **DONE** |
| 10 | Framework logit gate | `framework_parity_gate.json` (TRT skipped) | **DONE** |
| 11 | DICC historical + multi-compiler | `docs/DICC_*` + SUCCESS trees | **DONE** |
| 12 | B3 server latency decision | `docs/B3_SERVER_LATENCY_DECISION.md` **Option B** | **DONE** (rebench deferred, claim dropped) |
| 13 | Figure inventory + regen | `docs/FIGURE_STATUS.md` + `figures/` | **DONE** |
| 14 | Tables from artifacts | `docs/manuscript/TABLES_FROM_ARTIFACTS.md` | **DONE** |
| 15 | Stale-claim sweep | `docs/STALE_CLAIM_SWEEP_2026-08-15.md` | **DONE** (active surfaces) |
| 16 | Writer handoff | `docs/CHERAN_MANUSCRIPT_HANDOFF.md` | **DONE** |
| 17 | Public repo map | `docs/README.md` + README start-here | **DONE** |

---

## Explicitly not required to close pre-manuscript

- New DICC B3 post_fix campaign (Option B already taken)  
- Native TensorRT engine  
- Formal 100-run determinism campaign  
- Rebuilding the Jul 22 manuscript PDF  
- Venue formatting  

Those belong to manuscript / optional later work.

---

## Handoff sentence for Cheran

> Pre-manuscript evidence is frozen. Draft MD + tables + figures are feedstock. Please write the paper from `docs/CHERAN_MANUSCRIPT_HANDOFF.md`; do not treat `CAD_CBA_v1_MANUSCRIPT.pdf` as current.
