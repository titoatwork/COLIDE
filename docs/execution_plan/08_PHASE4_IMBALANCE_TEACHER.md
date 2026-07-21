# 08 — Phase 4: Class Imbalance + Distillation Teachers

**Status:** PARTIAL — imbalance losses DONE; WP4b teachers DONE (ensemble INCORPORATE); stratified/SupCon still open  
**Depends on:** Phase 1–3 (can interleave carefully with HPO)  
**Prof:** §3 imbalance; §4 teachers  

---

## 1. Goals

1. Compare imbalance-handling strategies under **identical** protocol.  
2. Compare teachers; pick based on **student val metrics + deployment story**.  
3. Avoid “student only copies RF” without trade-off justification.

---

## 2. Imbalance strategies to compare

| ID | Method | Implementation notes |
|----|--------|----------------------|
| I0 | CE only (control) | |
| I1 | Weighted CE (class_weights.npy) | Enable config flag |
| I2 | Focal (γ sweep) | Exists |
| I3 | Class-balanced focal | Implement |
| I4 | Logit adjustment | Implement |
| I5 | Stratified batch sampling | WeightedRandomSampler |
| I6 | Val-tuned class thresholds | Phase thresholds module |
| I7 | Controlled SMOTE (current) | Document exactly |
| I8 | SupCon (optional later) | High cost — LATER unless time |

**Selection rule:** best **val** macro-F1 with constraint on minority (e.g. Theft F1 ≥ baseline).

---

## 3. Teachers to compare

| ID | Teacher | Notes |
|----|---------|-------|
| T0 | sklearn RF (published 0.9864 protocol) | Bar |
| T1 | Strengthened RF (~0.9885 diagnostic) | `rf_teacher_strengthen.py` |
| T2 | XGBoost | Add |
| T3 | LightGBM | Add |
| T4 | Ensemble RF+XGB+LGBM | `train_ensemble_distill.py` base |
| T5 | Neural teacher | Optional |

Student always same architecture family; vary soft labels only.

---

## 4. Deployment justification (must write into results)

For final student vs RF:

| Metric | RF | Student | Winner |
|--------|----|---------|--------|
| macro-F1 | | | |
| minority F1 | | | |
| GPU VRAM | | | |
| CPU-only possible? | | | |
| Latency neural batch-1 | | | |
| Throughput | | | |
| Energy | | | |
| Temporal representation | RF no / CNN-BiLSTM yes | | |

---

## 5. Acceptance criteria

- [ ] Imbalance comparison table (val + sealed test for winner only)  
- [ ] Teacher comparison table  
- [ ] Final KD recipe documented (α, T, γ, teacher type)  
- [ ] Trade-off paragraph supported by numbers  

---

## 6. Exit

Frozen teacher + loss + threshold policy for final method package.
