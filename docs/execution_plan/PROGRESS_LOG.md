# Progress log (results as they land)

**Policy:** document everything; label pilot vs full; test sealed unless noted.  
**Handoff snapshot:** 2026-07-21 (documentation only — no new experiments this day)

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
- Val threshold JSON on best ckpt (util ready)  
- DICC multi-day  
- XAI quality suite  
- ToN final method  
- Manuscript  

---

## Desirability
Multirun + protocol-fair RF/XGB put neural FT in the **same val band** as strong trees under one protocol — desirable baseline before HPO/method push. Focal remains best loss in the four-way compare.
