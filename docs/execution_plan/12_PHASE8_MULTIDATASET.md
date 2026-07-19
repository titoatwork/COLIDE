# 12 — Phase 8: Multi-Dataset Validation

**Status:** PARTIAL historical ToN exists; final-method pass NOT DONE  
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

- [ ] Final method ToN table  
- [ ] RF/XGB comparison on ToN  
- [ ] Honest gap discussion  
- [ ] No silent hyperparameter fishing on ToN test  

---

## 5. Exit

Results support multi-dataset claims in abstract.
