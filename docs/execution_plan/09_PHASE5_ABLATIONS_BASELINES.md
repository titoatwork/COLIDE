# 09 — Phase 5: Ablations, Fair Baselines, Multi-Objective

**Status:** NOT STARTED  
**Depends on:** Final method candidate from Phases 2–4  
**Prof:** §5 ablations; §6 baselines; §7 multi-objective  

---

## 1. Ablation ladder (minimum)

| Row | Model |
|-----|--------|
| A1 | CNN only |
| A2 | BiLSTM only |
| A3 | CNN–BiLSTM (no KD, no extras) |
| A4 | + knowledge distillation |
| A5 | + imbalance method |
| A6 | + attention/fusion (if in package) |
| A7 | Full proposed method |

**Metrics per row:** macro-F1, weighted-F1, balanced acc, precision, recall, per-class F1, latency, param count, memory, energy (where measurable).

Same split, same seeds policy.

---

## 2. Baseline suite (same protocol)

| Family | Models |
|--------|--------|
| Classical | Logistic Regression, SVM, RF, XGBoost, LightGBM |
| Neural | MLP, 1D-CNN, LSTM, BiLSTM, CNN–LSTM, CNN–BiLSTM, Transformer or temporal-attention baseline |
| External | Reproducible lightweight IDS if code available |

**Rules:**

- Same train/val/test  
- Comparable tuning effort (document budget per model class)  
- JSON schema unified  

---

## 3. Multi-objective / Pareto

### 3.1 Axes (primary figure)

- X: latency (µs) or throughput  
- Y: macro-F1  
- Size/color: GPU memory or energy  

Include RF, cuML RF, final student, ablations, frameworks as relevant.

### 3.2 Composite score (define **before** final selection)

Example (document exact weights):

```text
score = w1 * macro_f1_norm + w2 * (1 - latency_norm) + w3 * (1 - mem_norm) + w4 * minority_norm
```

Do not change weights after seeing test ranking.

---

## 4. Implementation

| Script | Role |
|--------|------|
| `scripts/run_baselines.py` | Classical + simple neural |
| `scripts/run_ablations.py` | Ladder A1–A7 |
| `scripts/make_pareto.py` | Plot + CSV |
| `benchmarks/results/baselines/*.json` | Outputs |
| `benchmarks/results/ablations/*.json` | Outputs |

---

## 5. Acceptance criteria

- [ ] Full ablation table  
- [ ] Full baseline table BoT-IoT  
- [ ] Pareto figure  
- [ ] Answers: where we win/lose vs RF and vs best neural  
- [ ] All numbers path-traceable to JSON  

---

## 6. Exit

Phase 5 tables are manuscript-ready (Results 5.1–5.5).
