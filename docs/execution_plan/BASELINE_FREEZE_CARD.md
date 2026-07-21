# Baseline Freeze Card (Phase 1) — BoT-IoT

**Status:** ACTIVE foundation (2026-07-19); multirun baseline **DONE** 2026-07-19 (mean val macro-F1 0.9714±0.0109)  
**Protocol ID:** `botiot_v1`  
**Code:** `scripts/protocol/botiot.py`, `scripts/protocol/metrics.py`, `scripts/eval_checkpoint.py`  
**Handoff numbers:** see `RESULTS_DISK_MANIFEST.md`

---

## 1. Data

| Item | Freeze |
|------|--------|
| Dataset | UNSW BoT-IoT 10-best features |
| Train CSV | `data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv` |
| Test CSV | `data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv` |
| Features | 10 columns in `FEATURE_COLUMNS` / `config.yaml` |
| Label | `category` → 5 classes via `LabelEncoder` |
| Val split | 10% stratified from official Training CSV (`val_size=0.1`) |
| Seed (default) | 42 (split + SMOTE) |
| Scaler | `MinMaxScaler` fit on **train fold only** after optional SMOTE |

---

## 2. Stages

| Stage | Name | SMOTE | Used for |
|-------|------|-------|----------|
| `stage_a_kd` | KD / distill | Yes (historical targets; k=3) | Stage-1 distillation |
| `stage_b_ft` | Fine-tune | **No** | Two-stage real FT (champion path) |

SMOTE targets (stage_a, scale=1.0): DDoS/DoS 100k, Recon 50k, Normal 2k, Theft 1k (only if target &gt; natural count).

---

## 3. Metrics (always)

- accuracy, balanced_accuracy  
- macro_f1, weighted_f1  
- min_per_class_f1  
- per-class P/R/F1/support  
- confusion_matrix  
- Theft / Normal F1 and recall (explicit keys)

**Selection uses validation only.**  
**Test:** only with `eval_checkpoint.py --allow-test` after config freeze.

---

## 4. Reference champion (baseline, not necessarily final paper model)

| Item | Value |
|------|--------|
| Path | `model/best_model_botiot_twostage.pth` |
| md5 | `80a90f7cc210276300eaa90173a5a385` |
| Historical test macro-F1 | 0.9790 |
| Architecture | `CNNBiLSTMAttention` + `config/config.yaml` |
| RF bar | 0.9864 (`rf_baseline_processed.json`) |

**Rule:** never overwrite champion without `BACKUP_*` + explicit OK.

---

## 5. How to run

```bash
cd /home/titoisalive/colide
source .venv/bin/activate   # or .venv path
PYTHONPATH=. python scripts/smoke_botiot_protocol.py

PYTHONPATH=. python scripts/eval_checkpoint.py \
  --checkpoint model/best_model_botiot_twostage.pth \
  --stage stage_b_ft

# Sealed test (final only):
PYTHONPATH=. python scripts/eval_checkpoint.py \
  --checkpoint model/best_model_botiot_twostage.pth \
  --stage stage_b_ft --allow-test
```

---

## 6. Next non-DICC work (after this card)

1. Multi-run baseline driver (5 seeds)  
2. Sign method package (MOD table)  
3. Losses + thresholds modules  
4. Optuna HPO on val  
5. Baselines + ablations  

DICC remains parallel hard gate for cluster claims.
