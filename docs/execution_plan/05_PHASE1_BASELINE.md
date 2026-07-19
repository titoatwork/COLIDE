# 05 — Phase 1: Strong Reproducible Baseline

**Status:** NOT STARTED (prep allowed during DICC wait)  
**Depends on:** Prefer after Phase 0; prep can start earlier  
**Prof:** “Establish a strong reproducible baseline”

---

## 1. Goals

1. **One** canonical data + split + metric protocol.  
2. Freeze **baseline CNN–BiLSTM** (current champion path as reference).  
3. Report **mean ± std** over ≥5 training runs where feasible.  
4. Seal **test set** from model selection forever under this protocol.

---

## 2. Problems in current code (must fix)

| Issue | Evidence |
|-------|----------|
| Split logic duplicated | `train_distill.py`, `train_twostage.py`, preprocess_v2, ensemble each reload CSV |
| Stage-1 uses SMOTE; stage-2 real-only | Intentional but must be **documented** as two-stage protocol |
| `use_class_weights: false` while weights saved | Inconsistent imbalance story |
| Single seed 42 | No multi-run variance |
| Test metrics during sweeps | Leakage risk for HPO era |

---

## 3. Deliverables

| Artifact | Description |
|----------|-------------|
| `colide/data_protocol.md` or section in this pack | Written freeze card |
| `scripts/data/botiot_protocol.py` (new) | Single load: train/val/test arrays, class names, scaler policy |
| `benchmarks/results/baseline_multirun.json` | 5-run stats |
| Config freeze file | `config/baseline_freeze.yaml` copy of locked hyperparams |
| Checkpoint policy | Reference md5 for baseline champion |

---

## 4. Frozen protocol (draft — finalize in implementation)

### 4.1 Data

- Dataset: BoT-IoT 10-best features (list in `config.yaml`)  
- Labels: 5-class `category`  
- Split: stratified train/val from official train CSV; official test CSV as **test only**  
- Stage-A (KD): document SMOTE strategy exactly (targets, k)  
- Stage-B (FT): **no SMOTE**, real counts only (as `train_twostage.py`)  
- Scaler: MinMax fit on stage train only; apply val/test  

### 4.2 Metrics (always log)

- macro-F1, weighted-F1, balanced accuracy, accuracy  
- per-class precision/recall/F1  
- minority: min per-class F1, Theft/Normal recall (explicit)  
- confusion matrix  

### 4.3 Seeds

- Report runs with seeds `{42, 43, 44, 45, 46}` or equivalent 5 seeds  
- cudnn deterministic as current  

### 4.4 Baseline model

- Architecture: `CNNBiLSTMAttention` + `config.yaml` freeze  
- Weights reference: two-stage 0.9790 path (single-run historical) + new multi-run of **same recipe** if compute allows  

---

## 5. Work steps

1. Implement unified `botiot_protocol.py` used by all new train/eval scripts.  
2. Write `eval_checkpoint.py` — load ckpt, eval val+test, dump JSON (test flag: only if `allow_test=true`).  
3. Multi-run driver: train baseline recipe × 5 seeds → aggregate.  
4. Document any deviation from historical 0.9790.  
5. Freeze YAML + commit.

---

## 6. Acceptance criteria

- [x] One import path for data in new experiments (`scripts/protocol/botiot.py`)  
- [x] Written freeze card reviewed (`BASELINE_FREEZE_CARD.md`)  
- [x] Val metrics available without touching test (`eval_checkpoint.py`)  
- [ ] Multi-run JSON exists OR documented compute blocker with single-run + justification  
- [x] Baseline champion md5 recorded (freeze card)  
- [ ] Legacy train scripts fully migrated to protocol (still ad hoc loaders)  

**2026-07-19 progress:** smoke OK; champion val macro-F1 **0.9780** under `stage_b_ft` (test sealed).  

---

## 7. Exit

Phase 1 complete when any new experiment **must** call the frozen protocol or is rejected in review of PR/session.
