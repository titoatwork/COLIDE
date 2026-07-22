# 12 — Phase 8: Multi-Dataset Validation

**Status:** DONE 2026-07-22 final-method pass RUN_DOCUMENTED (`toniot_final/`)  
**Prof:** §10  

---

## 1. Goal

BoT-IoT remains primary. Final method must also be evaluated on **ToN-IoT** (clean protocol) under the same scientific standards (sealed test, metrics, optional light tuning rules documented).

---

## 2. Existing assets

- `scripts/preprocess_toniot.py`, `train_toniot_clean.py`, `train_distill_toniot*.py`  
- Results: ~0.9526 CNN / 0.9851 RF (historical)  
- Must **not** treat historical as final-method result without re-run  

---

## 3. Protocol

1. Freeze BoT final method hyperparameters **first**.  
2. Map method to ToN feature space (26-feat clean as current).  
3. Train under ToN split protocol (document).  
4. Report same metric suite + RF baseline same split.  
5. Optional: transfer / train-on-BoT eval-on-ToN only if scientifically meaningful and labeled.

---

## 4. Acceptance criteria

- [x] Final method ToN table (`toniot_final/summary.json`)  
- [x] RF same-split comparison (val/test); XGB/LGBM teacher vals logged  
- [x] Honest gap: neural test **0.8110** vs RF **0.9393**; ≠ historical 26-feat 0.9526  
- [x] One sealed test pass after val selection (no iterative test fishing)  

---

## 5. Exit

Multi-dataset evidence on disk. Abstract may claim recipe evaluated on ToN under documented protocol — **not** parity with historical clean 0.9526.
