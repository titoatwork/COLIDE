# 03 — Gap Matrix: Prof Requirements × Codebase Reality

> **SUPERSEDED as live gap list (2026-08-12).**  
> Pre-manuscript closed; DICC multi-session + multi-compiler measured.  
> Use `docs/CLAIM_MAP_PREWRITE.md` + tracker for current gaps (optional S1b/Nsight/B3 optim / manuscript insert only).

**Legend:** DONE = usable as-is · PARTIAL = exists but incomplete · MISSING = must build · BLOCKED = waiting on ops/data

---

## A. Detection science

| Prof requirement | Codebase today | Status | Severity | Plan phase |
|------------------|----------------|--------|----------|------------|
| Controlled HPO (Optuna/Bayesian) | Manual CLI sweeps only | **MISSING** | CRITICAL | 3 |
| Val objectives: macro-F1, bal-acc, minority recall | Macro-F1 primary; minority not optim objective | **PARTIAL** | HIGH | 1,3 |
| Test set sealed until final config | Test often evaluated during sweeps | **PARTIAL / RISK** | CRITICAL | 1,3 |
| ≥5-run mean±std training | Single seed 42 | **MISSING** | HIGH | 1 |
| Novel method component with named weakness | V3 MHA exists; not class-aware package | **PARTIAL** | CRITICAL | 2 |
| Imbalance strategy comparison table | SMOTE+focal only systematic | **PARTIAL** | HIGH | 4 |
| Class weights / logit-adj / CB-focal / thresholds | weights.npy unused; no logit-adj/thresholds | **MISSING** | HIGH | 4 |
| Improved teacher / ensemble | RF + strengthen + ensemble script | **PARTIAL** | MED | 4 |
| Student deployment trade-off vs RF | VRAM/energy numbers exist, not integrated story | **PARTIAL** | HIGH | 5,11 |
| Full ablation ladder + systems metrics | Fragmented | **MISSING** | CRITICAL | 5 |
| Fair baselines (LR…Transformer) same split | RF/MLP/ensemble only | **MISSING** | CRITICAL | 5 |
| Per-class / minority first-class reporting | classification_report exists ad hoc | **PARTIAL** | HIGH | 1,5 |
| Beat or near-RF with justification | 0.9790 vs 0.9864 | **PARTIAL** | CRITICAL | 3–5 |
| Multi-dataset final method | ToN exists separately | **PARTIAL** | HIGH | 8 |

---

## B. Systems / CUDA

| Prof requirement | Codebase today | Status | Severity | Plan phase |
|------------------|----------------|--------|----------|------------|
| B3 CUDA vs matching PT same GPU (cluster) | Local RTX only | **BLOCKED** | CRITICAL | 0 |
| Full-model PT absolute same GPU cluster | Tooling yes | **BLOCKED** | CRITICAL | 0 |
| V100S + A100 multi-day | Tooling yes | **BLOCKED** | CRITICAL | 0 |
| mean/median/std/CV/CI | Harness supports | **BLOCKED** (no tree) | CRITICAL | 0 |
| Batch-size sensitivity cluster | Config lists batch sizes | **MISSING** on cluster | MED | 0,6 |
| Significance + effect size multi-day | compare_dicc has Welch/d | **BLOCKED** | HIGH | 0 |
| Operation-matched claims only | Option A + harness flag | **DONE** (discipline) | — | 6 |
| Avoid full-pipeline invalid claims | README still risky language | **PARTIAL** | HIGH | 6,9 |
| Profile → optimise dominant kernels | Nsight script exists | **PARTIAL** | MED | 6 |
| FP16/INT8 accuracy delta | FP16 kernels; INT8 no | **PARTIAL** | MED | 6 |
| TRT/ORT/compile matrix | Local ranges; cluster optional | **PARTIAL** | MED | 6 |
| Portability conclusion | Unknown until DICC | **BLOCKED** | CRITICAL | 0 |

---

## C. Multi-objective & presentation

| Prof requirement | Status | Severity | Phase |
|------------------|--------|----------|-------|
| Pareto F1–latency–memory | **MISSING** | CRITICAL | 5 |
| Near-RF + low VRAM narrative quantified | **PARTIAL** (numbers exist) | HIGH | 5 |
| Energy + throughput in same comparison | **PARTIAL** | MED | 5,6 |
| Composite objective defined a priori | **MISSING** | HIGH | 1,5 |

---

## D. Explainability

| Prof requirement | Status | Severity | Phase |
|------------------|--------|----------|-------|
| Dispatch overhead measured | **DONE** (16.60 µs) | — | 7 |
| Faithfulness / consistency / hallucination | **MISSING** | HIGH if title claims XAI | 7 |
| vs SHAP/LIME/attention/rules | **MISSING** | HIGH if XAI | 7 |
| Structured evidence into LLM | **PARTIAL** (features in prompt) | MED | 7 |
| Analyst usefulness | **MISSING** | MED | 7 |

**Decision:** Either complete Phase 7 or **remove “explainable” from title/abstract claims**.

---

## E. Reproducibility & engineering hygiene

| Prof requirement | Status | Severity | Phase |
|------------------|--------|----------|-------|
| Unified train/val/test protocol | Duplicated loaders | **PARTIAL** | 1 |
| Seeds, configs, git SHA on results | Incomplete | **PARTIAL** | 1,14 |
| Claim JSON tracked / manifest | gitignored results | **MISSING** | 1,16 |
| verify_claims | **DONE** for number presence | MED (not semantic) | 9 |
| Checkpoint backup discipline | BACKUP files exist | **PARTIAL** | 16 |

---

## F. Priority ranking for exceptional delivery

### P0 — Without these, journal path fails
1. Phase 0 DICC complete  
2. Unified sealed-split protocol  
3. Clear multi-objective or detection win with tables  
4. Ablations + fair baselines  
5. Option A CUDA honesty on cluster  

### P1 — Strongly expected by Prof
6. Optuna HPO  
7. Imbalance comparison + thresholds  
8. Method novelty with ablation  
9. Pareto figure  
10. ToN final method  

### P2 — Title-dependent / stretch
11. Full XAI suite  
12. INT8  
13. Full V3 CUDA parity (Option B)  
14. Nsight deep dive  

---

## G. Assets we can reuse (do not rebuild from zero)

| Asset | Reuse as |
|-------|----------|
| `train_distill.py` / `train_twostage.py` | Basis for unified trainer |
| `train_ensemble_distill.py` | Ensemble teacher branch |
| `rf_baseline_processed.py` / strengthen | Teacher + RF bar |
| `benchmark_cuda_kernels_stats.py` | Cluster + local latency |
| `benchmark_pytorch_block3_stats.py` | Matching PT B3 |
| `compare_dicc_sessions.py` | Multi-day gate |
| `dicc_scripts/run_campaign.sh` | Phase 0 entry |
| `numerical_fidelity.py` | After re-export |
| `verify_claims.py` | Extend after new numbers |
| `llm_explainability.py` | Dispatch baseline for Phase 7 |
| Existing distill JSON grid | Partial HPO prior / warm start |

---

*Next: phase files `04`–`13`*
