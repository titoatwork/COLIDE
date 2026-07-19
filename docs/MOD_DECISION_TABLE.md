# COLIDE — Method modification decision table

**Purpose:** Prof request — compare possible modifications **before** choosing the final model package.  
**Authority:** `docs/feedback1.docx` + `docs/PROF_FEEDBACK_ROADMAP.md`  
**Rule:** Do not change many parts at once. Prefer **one core novelty** after Phase 0 DICC.

**Status:** Pre-DICC draft recommendations (2026-07-19). Revisit after cluster B3 results.

---

## Decision legend

| Tag | Meaning |
|-----|---------|
| **MUST** | Required for WoS path Prof described |
| **SHOULD** | High value / low regret |
| **CANDIDATE** | In the “one method” shortlist |
| **CONDITIONAL** | Only if DICC or other evidence forces it |
| **LATER / OUT** | Stretch or out of default path |

---

## Master table

| ID | Modification | Weakness addressed | Expected benefit | GPU / effort | Risk | After DICC? | Decision now |
|----|--------------|--------------------|------------------|--------------|------|-------------|--------------|
| M0 | Multi-day UM DICC (B3 same-GPU, full PT abs, 2 days, stats) | Portability unproven | Systems validity | Queue / ops | Project fails review without it | **Is Phase 0** | **MUST — NOW** |
| M1 | Freeze baseline protocol + ≥5-run mean±std where feasible | Irreproducible single run | Trust | Med train | Time | No | **MUST after M0** |
| M2 | Systematic Optuna/Bayesian HPO on **val only** | Under-tuned model | Macro-F1, minority | Med–High | Test leakage if sloppy | No | **SHOULD** |
| M3 | Class-balanced focal / logit-adjusted loss | Extreme imbalance | Minority F1 | Low–Med | — | No | **CANDIDATE (in package)** |
| M4 | Class-specific decision thresholds (calibrate on val) | Rare-class errors | Minority recall | Low | Must not tune on test | No | **CANDIDATE (in package)** |
| M5 | Lightweight temporal / class-aware attention | Weak temporal focus | Macro / minority | Med | Breaks CUDA parity until re-export | No | **CANDIDATE (Prof lean)** |
| M6 | Multi-scale 1D convolution | Multi-scale patterns | Macro-F1 | Med | Arch change | No | **CANDIDATE** |
| M7 | Ensemble teacher (RF+XGB+LGBM) distilled | Teacher ceiling | Student F1 | Med | May still lag RF | No | **CANDIDATE** |
| M8 | Stronger single RF teacher (already ~0.9885 diagnostic) | Teacher quality | Student F1 | Low | Published RF bar protocol | No | **CANDIDATE** |
| M9 | Fair baseline suite (LR…Transformer) same split | Weak comparison | Reviewer defense | Med | Time | No | **MUST before paper** |
| M10 | Full ablation ladder | Black-box story | Reviewer defense | Med | Time | No | **MUST before paper** |
| M11 | Pareto F1–latency–memory (+ energy/thr) | Separate narratives | Multi-objective win | Low analysis | Empty if no edge | Partial DICC | **MUST** |
| M12 | Block 3 kernel optim if CUDA loses on V100S/A100 | Server loss | Portability | High | Only if needed | **Yes** | **CONDITIONAL** |
| M13 | Full V3 CUDA (attn/LN/GAP) | Invalid full-pipeline claim | Option B | Very high | Delays everything | No | **OUT default** |
| M14 | XAI quality (faithfulness, vs SHAP/LIME, structured evidence) | Dispatch-only “explain” | Real XAI claim | High | Subjective | No | **If title says explainable** |
| M15 | ToN-IoT / second dataset on **final** method | Single-dataset | Generalisation | Med | — | No | **MUST final** |
| M16 | Force-track claim-source JSONs / manifest | Gitignore repro hole | Repro | Low | — | No | **SHOULD soon** |
| M17 | Train only to beat RF F1 | Gap 0.74% | Accuracy headline | High | May fail; not sole goal | No | Not sole goal |
| M18 | SupCon / asymmetric loss | Minority | Rare-class | High | Scope creep | No | **LATER** |
| M19 | Uncertainty-aware detection | Calibration | Ops value | High | Scope | No | **LATER** |
| M20 | INT8 / prune after arch fixed | Deploy cost | Latency/memory | Med | Accuracy drop | After arch | Stretch |

---

## Recommended default package (confirm after M0)

**Core method package (one story):**  
**Class-aware distilled CNN–BiLSTM** = baseline CNN–BiLSTM + **M3 + M4 + (M7 or M8)** + optional **M5** (attention only if ablations justify).

**Systems package:** M0 + op-matched CUDA (Option A) + M11 + M12 if needed.  
**Paper hygiene:** M9 + M10 + M15 + M16.  
**XAI:** M14 only if abstract/title claim explainability beyond dispatch.

**Novelty sentence (draft, only if M3–M5 evaluated):**  
*Under severe class imbalance, prior CUDA-accelerated IDS work optimises inference but under-reports minority detection; we jointly improve minority-aware training/distillation and deployment metrics under operation-matched CUDA evaluation.*

---

## Explicit non-goals (default)

- Changing all of architecture + loss + teacher + CUDA + XAI in one week  
- Claiming full-pipeline CUDA vs full V3 without M13  
- Using test set for HPO or threshold search  
- Clobbering `best_model_botiot_twostage.pth` without `BACKUP_*`  
- Treating Rostam as UM official  

---

## Next fill-in after DICC (user/agent)

| Field | Value after M0 |
|-------|----------------|
| V100S B3 CUDA vs PT | |
| A100 B3 CUDA vs PT | |
| Multi-day compare | accept / reject |
| Primary contribution dimension | latency / memory / trade-off / detection |
| Final M-package IDs | |
| Attention (M5) in or out | |

---

*Update this file when a decision is made; do not implement CANDIDATEs in bulk without a signed package.*
