# 07 — Phase 3: Systematic Hyperparameter Optimisation

**Status:** DONE (WP3 2026-07-21) — train HPs INCORPORATED; arch search deferred; sealed multi-seed test of winner still open  
**Depends on:** Phase 1 protocol; preferably method skeleton from Phase 2  
**Prof:** §1 + staged Phase 2 search table  
**Evidence:** `benchmarks/results/hpo/summary.json` · `config/hpo_best.yaml` · `scripts/hpo_optuna_botiot.py`  

---

## 1. Goal

Replace manual sweeps with **reproducible** HPO (Optuna/Bayesian) on **validation** objectives only.

---

## 2. Search space (from Prof; refine if compute limited)

| Hyperparameter | Suggested range |
|----------------|-----------------|
| CNN filters | 32, 64, 128, 256 |
| Kernel size | 3, 5, 7 |
| CNN blocks | 1–4 (if architecture allows) |
| BiLSTM hidden | 32–256 |
| BiLSTM layers | 1–3 |
| Dropout | 0.1–0.6 |
| Learning rate | 1e-5 – 1e-2 |
| Batch size | 64–1024 |
| Focal γ | 0–5 |
| Distill T | 1–20 |
| Distill α | 0.1–0.9 |
| Sequence/reshape length | dataset-dependent / config reshape |
| Class threshold | 0.1–0.9 per class (post-hoc on val) |

**Budget (exceptional default):**  
- Stage A: 40–80 trials architecture/train (reduced epochs + patience)  
- Stage B: refine top-5 full epochs  
- Stage C: threshold calibration on val for top-1  
- **Then** sealed test once (+ multi-seed confirm)

---

## 3. Objectives

Primary multi-objective (Optuna):

```text
maximize  val_macro_f1
maximize  val_balanced_accuracy
maximize  val_minority_score   # e.g. min per-class F1 or Theft F1
```

Optional constraint: param count or train-time budget.

**Forbidden:** optimising test macro-F1.

---

## 4. Implementation

| Deliverable | Notes |
|-------------|-------|
| `scripts/hpo_optuna_botiot.py` | Study + SQLite storage |
| `benchmarks/results/hpo/study.db` + trials JSON | Track or manifest |
| `config/hpo_best.yaml` | Winner hyperparams |
| Report table | Top-10 trials val metrics |

Integrate with Phase 1 data protocol.

---

## 5. Acceptance criteria

- [x] Study fully logged (trial id, params, val metrics, seed) — trial JSONs + `study.db` + `summary.json`  
- [ ] Test evaluated **once** for winner (plus multi-seed if Phase 1 standard) — **deferred** (test sealed until final lock)  
- [x] No silent use of test for early stopping model selection beyond val  
- [x] Winner config committed — `config/hpo_best.yaml`  

### WP3 outcome (2026-07-21)
- Stage A: 20 trials, max_train=400k explore, full val; 11 complete / 9 pruned  
- Stage B: full-train refine top-3; winner trial **8** val macro-F1 **0.9791** (**INCORPORATE**, Δ+0.0010 vs multirun seed42 0.9780)  
- Arch HPs (filters/kernel/BiLSTM) **not** searched — CAD-CBA-v1 freeze for KD init  

---

## 6. Exit

`hpo_best.yaml` is the CAD-CBA-v1 train-HP source.  
Next: multi-seed confirm and/or FT from ensemble KD with these HPs; sealed test only after final config lock.
