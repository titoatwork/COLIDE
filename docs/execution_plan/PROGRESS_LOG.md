# Progress log (results as they land)

**Policy:** document everything; label pilot vs full; test sealed unless noted.  
**Handoff snapshot:** 2026-07-21 (WP4b teacher/KD DONE)

---

## 2026-07-21 — WP4b teacher/KD under protocol (science)

Scripts: `scripts/train_protocol_kd.py`, `scripts/run_teacher_kd_compare.py`  
Protocol: `botiot_v1` / **stage_a_kd** / seed 42 / full train / **val only** (test sealed)  
Recipe: α=0.6, T=10.0, focal γ=2.0, epochs≤10, patience=4, batch=512, lr=1e-3  
Summary: `benchmarks/results/teachers_kd/summary.json` md5 `63ab4bd3f40e24adc6788fa1ca255bd8`  
Wall ~6879 s (~1.9 h). Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.

| Rank | Teacher | Student val macro-F1 | Min-cls | Theft | Teacher val | Decision |
|------|---------|----------------------|---------|-------|-------------|----------|
| 1 | **ensemble** (RF+XGB+LGBM mean) | **0.9401** | 0.8434 | 0.9231 | 0.9803 | **INCORPORATE** |
| 2 | rf | 0.9346 | 0.8000 | **1.0000** | 0.9750 | RUN_DOCUMENTED fallback |
| 3 | none (hard-label focal) | 0.9326 | 0.8409 | 0.9231 | — | RUN_DOCUMENTED control |
| 4 | xgb | 0.9270 | 0.8434 | 0.8571 | **0.9918** | RUN_DOCUMENTED |
| 5 | lgbm | 0.8829 | 0.7059 | 0.7059 | 0.5928 | RUN_DOCUMENTED (weak) |

**Interpretation**
- Best **student** is ensemble soft labels — not solo XGB despite XGB’s highest teacher hard-label F1.
- RF KD still strong and simpler; Theft F1=1.0 on best RF student ckpt.
- Hard-label `none` nearly matches RF KD (Δmacro ≈ +0.002 for RF) — KD lift modest under this budget.
- LGBM alone is a poor teacher on stage_a_kd (mirrors classical LGBM weakness).
- Numbers are **stage_a from-scratch KD**, not stage_b FT (do not mix with multirun mean 0.9714).

Ckpts: `model/teachers_kd/kd_{none,rf,xgb,lgbm,ensemble}_a0.6_T10.0_g2.0_seed42.pth`  
Tracker: E1–E5/E7/C9/WP4b updated. CAD-CBA-v1 KD teacher → **ensemble**.

**Next science:** WP3 Optuna HPO **or** WP5 ablations/neural baselines **or** stage_b FT from ensemble KD init.

---

## 2026-07-21 — WP2d val thresholds (science)

Script: `scripts/run_val_thresholds.py` (+ hardened `scripts/protocol/thresholds.py`)  
Checkpoint: `model/imbalance_loss/ft_focal_seed42.pth` md5 `170eaccc584ba12cce2a34ca52ebfbf2`  
Protocol: `botiot_v1` / `stage_b_ft` / seed 42 / **val only** (test sealed)  
Result: `benchmarks/results/imbalance_loss/thresholds_focal_seed42.json` md5 `2a6bc98d967883efc53d326535cf9d5b`

| Variant | Val macro-F1 | Min cls F1 | Theft F1 | Normal F1 |
|---------|--------------|------------|----------|-----------|
| argmax (baseline) | **0.9780** | 0.9315 | 1.0000 | 0.9315 |
| fixed t=0.5 | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| search macro_f1 | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| search min_per_class_f1 | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| joint macro→min | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| search class_f1 Theft | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| search class_f1 Normal | 0.9780 | 0.9315 | 1.0000 | 0.9315 |

**Decision: RUN_DOCUMENTED** — Δmacro=0, Δmin=0 vs argmax. Keep default **argmax** decode for CAD-CBA-v1.  
Interpretation: focal FT probabilities are already decisive on val; per-class thresholds add no selection signal for this checkpoint.  
Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
Tracker: B12 / C11 / D7 → RUN_DOCUMENTED. WP2d DONE.

**Next science:** WP4b teachers/KD under protocol **or** WP3 Optuna **or** WP5 ablations/neural baselines.

---

## 2026-07-21 — Handoff closure (docs only)

- **No training / no new science runs** (user request).
- Aligned tracker G3–G5, D*, C6, L5 with completed runs.
- Wrote / refreshed: `SESSION_CONTINUITY.md`, `HANDOFF.md`, `RESULTS_DISK_MANIFEST.md`, `METHOD_PACKAGE_DECISION.md`, `15_WORK_PACKAGES.md`.
- Jobs: train/multirun/imbalance **idle** (completed 2026-07-19).
- DICC: still **ABSENT**.
- Next chat: verify disk → continue tracker from continuity §5.

---

## 2026-07-19 — Foundation + first experiment wave

### Protocol foundation
- `scripts/protocol/*` live (`botiot_v1`, metrics, losses, thresholds, result_schema)
- Champion sealed eval val macro-F1 **0.9780** (`stage_b_ft`, seed 42)
- Production champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`

### Fine-tune multirun (WP1b) — COMPLETE
Init checkpoint: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth`  
Scripts: `train_protocol_ft.py`, `run_baseline_multirun.py`  
Summary: `benchmarks/results/multirun/summary.json`

| Seed | Best val macro-F1 | Min per-class F1 | Elapsed (approx) |
|------|-------------------|------------------|------------------|
| 42 | 0.9780 | 0.9315 | 28 min |
| 43 | 0.9578 | 0.9091 | 25 min |
| 44 | **0.9840** | **0.9589** | 36 min |
| 45 | 0.9746 | 0.9143 | 16 min |
| 46 | 0.9624 | 0.9091 | 28 min |

**Mean 0.9714 ± 0.0109** (n=5) · min 0.9578 · max 0.9840  
Test sealed. Champion not overwritten.  
Ckpts: `model/multirun/ft_seed{42..46}.pth`

### Classical baselines (protocol `stage_b_ft`, val only) — DOCUMENTED
Script: `run_classical_baselines.py`  
**Authoritative table:** `benchmarks/results/baselines_classical/summary_handoff.json`  
(Note: `summary.json` may only hold the last single-model run — do not use alone.)

| Model | Train | Val macro-F1 | Min cls F1 | Theft F1 | Decision |
|-------|-------|--------------|------------|----------|----------|
| LR | full | 0.5231 | 0.0 | 0.0 | weak linear baseline |
| RF | full | **0.9778** | 0.9231 | 0.9231 | strong; protocol-fair |
| XGB | full | **0.9762** | 0.9231 | 0.9231 | strong |
| LGBM | full | 0.5512 | 0.0 | 0.0 | RUN_DOCUMENTED weak |
| SVM pilot 150k | subsample | FAILED | — | — | &lt;3 samples/class |

**Note:** Published README RF **0.9864** uses `rf_baseline_processed` — **different** from protocol-fair RF **0.9778**.

Pilot 100k earlier: LR 0.463 / RF 0.686 — RUN_DOCUMENTED pilot only.

### Imbalance loss compare — COMPLETE
Script: `run_imbalance_loss_compare.py` (5 ep, seed 42)  
Summary: `benchmarks/results/imbalance_loss/summary.json`

| Loss | Best val macro-F1 | Min cls F1 | Bal acc | Decision |
|------|-------------------|------------|---------|----------|
| ce | 0.9755 | 0.9189 | 0.975 | RUN_DOCUMENTED |
| **focal** | **0.9780** | **0.9315** | 0.975 | **INCORPORATE (default)** |
| focal_cb | 0.9121 | 0.8000 | 0.984 | RUN_DOCUMENTED (hurts macro) |
| logit_adj | 0.9225 | 0.8000 | 0.986 | RUN_DOCUMENTED (hurts macro) |

Ckpts: `model/imbalance_loss/ft_{ce,focal,focal_cb,logit_adj}_seed42.pth`

### Method package
Signed: **CAD-CBA-v1** in `METHOD_PACKAGE_DECISION.md` (keep V3 arch + **focal** + KD path; thresholds/HPO next).

### Explicitly NOT started this arc
- Optuna HPO  
- Ablation ladder / neural baselines suite  
- Teacher/KD under protocol (WP4b)  
- DICC multi-day  
- XAI quality suite  
- ToN final method  
- Manuscript  

*(Val thresholds: completed later same calendar day — see WP2d section above.)*

---

## Desirability
Multirun + protocol-fair RF/XGB put neural FT in the **same val band** as strong trees under one protocol — desirable baseline before HPO/method push. Focal remains best loss in the four-way compare.
