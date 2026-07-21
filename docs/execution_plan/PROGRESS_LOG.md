# Progress log (results as they land)

**Policy:** document everything; label pilot vs full; test sealed unless noted.  
**Handoff snapshot:** 2026-07-21 (multi-seed HPO confirm DONE + full-playlist context hygiene)

---

## 2026-07-21 — Full-playlist / context hygiene (docs only)

User lock reaffirmed: **complete every tracker playlist item** (not only the critical path).  
Updated `PROF_FEEDBACK_TRACKER.md` from **existing disk/docs evidence** (no new training):

- Closed / upgraded where evidence already existed: B10 (historical α/T sweeps + recipe), B11 (seq len locked), C13, D8, D10, F3/F4/F8, G6 historical, G11, G14, H1/H5, J1, K1, L2–L4/L7/L9, etc.
- Open rows explicitly tagged **Playlist required** (SupCon, stratified batch, arch HPO B2–B4, neural baselines, XAI, ToN, Pareto, sealed test, …).
- Continuity + HANDOFF policy text aligned.

**No invented numbers. No science re-runs this micro-pass.**

---

## 2026-07-21 — Multi-seed HPO confirm (original distill + hpo_best) (science)

Scripts: `scripts/run_hpo_multiseed_confirm.py`, `scripts/train_protocol_ft.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seeds 42–46 / epochs≤10 patience=3 / **val only** (test sealed)  
Init: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` (same as WP3 Optuna study)  
HPs: `config/hpo_best.yaml` (WP3 winner train recipe)  
Tag: `multirun_hpo_confirm/` (does **not** clobber WP1b `multirun/` or package `multirun_ensemble_hpo/`)  
Summary: `benchmarks/results/multirun_hpo_confirm/summary.json` md5 `ef6c92a592474c321a4c1300e19a8065`  
Wall ~2981 s (~50 min). Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
Thermal: soft 85 / hard 90; peak ~81°C; no hard trip.

| Seed | Best val macro-F1 | Min-cls | Theft | Elapsed |
|------|-------------------|---------|-------|---------|
| 42 | **0.9791** | 0.9351 | 1.0000 | ~8.1 min |
| 43 | 0.9587 | 0.9091 | 0.9091 | ~14.1 min |
| 44 | **0.9797** | 0.9367 | 1.0000 | ~8.1 min |
| 45 | 0.9787 | 0.9333 | 1.0000 | ~5.7 min |
| 46 | 0.9483 | 0.8333 | 0.8333 | ~12.0 min |

**Mean 0.9689 ± 0.0145** (n=5) · min 0.9483 · max 0.9797  
min-cls mean **0.9095** · Theft mean **0.9485**

**Comparators**
- WP3 HPO full-train seed42: **0.9791** — seed42 **reproduces** (0.979055 ≈ 0.979064)
- WP1b multirun (old distill + default HPs): **0.9714 ± 0.0109**
- Package ensemble KD + HPO: **0.9639 ± 0.0185**

**Interpretation**
- Fair multi-seed confirm of Optuna train HPs on the **same init** as the study.
- Seed42 is a clean repro of the WP3 winner; seed44 slightly higher (0.9797).
- Aggregate mean is **slightly below** WP1b default-HP multirun and **higher variance** (seed46 0.9483 and seed43 0.9587 drag).
- HPO HPs remain **INCORPORATED** as CAD-CBA-v1 train defaults (credible seed42 lift vs default 0.9780); multi-seed aggregate is **RUN_DOCUMENTED** evidence of seed sensitivity, not a mean-win claim over WP1b.
- Do not promote HPO multirun mean as “beats baseline multirun.”

**Decision: RUN_DOCUMENTED** (multi-seed aggregate) · train HPs stay **INCORPORATED** from WP3.

**Next science:** WP5a ablation ladder **or** neural baselines / Pareto. DICC only when user opens session.

---

## 2026-07-21 — Package FT multirun: ensemble KD init + HPO HPs (science)

Scripts: `scripts/train_protocol_ft.py` (HPO-aware AdamW/cosine/dropout), `scripts/run_package_ft_multirun.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seeds 42–46 / epochs≤10 patience=3 / **val only** (test sealed)  
Init: `model/teachers_kd/kd_ensemble_a0.6_T10.0_g2.0_seed42.pth`  
HPs: `config/hpo_best.yaml` (lr≈5.89e-5, batch=1024, γ≈1.92, drop≈0.148, att≈0.214, wd≈1.92e-4, cosine, AdamW)  
Summary: `benchmarks/results/multirun_ensemble_hpo/summary.json` md5 `1fa206e34c50e799d531f5eee70629e8`  
Wall ~84 min. Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
WP1b `multirun/` tree **not clobbered**.

| Seed | Best val macro-F1 | Min-cls | Theft | Elapsed |
|------|-------------------|---------|-------|---------|
| 42 | 0.9741 | 0.9333 | 1.0000 | ~14 min |
| 43 | 0.9328 | 0.8000 | 0.8000 | ~29 min |
| 44 | 0.9699 | 0.8947 | 1.0000 | ~13 min |
| 45 | **0.9803** | **0.9474** | **1.0000** | ~12 min |
| 46 | 0.9623 | 0.9091 | 0.9091 | ~14 min |

**Mean 0.9639 ± 0.0185** (n=5) · min 0.9328 · max 0.9803  

**Comparators**
- WP1b multirun (old distill + default HPs): **0.9714 ± 0.0109**
- WP3 HPO full-train seed42: **0.9791**
- Package seed45 max **0.9803** exceeds HPO seed42 point estimate

**Interpretation**
- Full CAD-CBA-v1 train path (ensemble KD → FT with Optuna HPs) is **run and documented**.
- Aggregate mean is **slightly below** WP1b and **higher variance** (seed 43 drags).
- Honest finding: HPO HPs optimized on old distill init do not transfer into a better multi-seed mean on ensemble KD init.
- Early epochs often collapse Theft then recover by ep3–4 (documented dynamics, not a bug; zero-train KD eval still 0.9401).
- **Decision: RUN_DOCUMENTED** for package multirun aggregate (not a mean win over WP1b). Component decisions unchanged (ensemble teacher, HPO HPs, focal, argmax).

**Also this session (tooling, not yet run to completion)**
- `model/ablation_variants.py` + `scripts/run_ablation_ladder.py` (WP5a ready)
- `scripts/run_hpo_multiseed_confirm.py` (multi-seed HPO on original distill init)
- Thermal guard `logs/thermal_guard.sh` (soft 85°C / hard pause 90°C)

**Next science:** multi-seed HPO confirm (n≥5, original init) **or** WP5 ablations (A1–A7) **or** neural baselines. DICC only when user opens session.

---

## 2026-07-21 — WP3 Optuna HPO under protocol (science)

Script: `scripts/hpo_optuna_botiot.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seed 42 / full val / **test sealed**  
Init: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth`  
Arch: fixed `cnn_bilstm_v3_attention` (CAD-CBA-v1; no arch search this WP)  
Study: `botiot_stage_b_ft_hpo_v1` · SQLite `benchmarks/results/hpo/study.db`  
Summary: `benchmarks/results/hpo/summary.json` md5 `5ba39a920706100b13975e89c3b20924`  
Winner config: `config/hpo_best.yaml` md5 `598d87c9ea5d0f26847ce7b860a0eb68`  
Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
Wall ~69 min (Stage A ~38.6 min + refine ~29.7 min).

### Stage A (explore)
- n_trials=20 · epochs≤4 · patience=2 · **max_train=400_000** stratified (val full)  
- COMPLETE **11** / PRUNED **9**  
- Best Stage A: trial **11** val macro-F1 **0.9787** (min-cls 0.9351, Theft 1.0)

### Stage B (full-train refine top-3)

| Rank | Source trial | Full-train val macro-F1 | Min-cls | Theft | Bal-acc | Decision |
|------|--------------|-------------------------|---------|-------|---------|----------|
| **1 (winner)** | **8** | **0.9791** | **0.9351** | **1.0000** | **0.9863** | **INCORPORATE** |
| 2 | 11 | 0.9721 | 0.9014 | 1.0000 | 0.9645 | RUN_DOCUMENTED (Stage-A best collapsed) |
| 3 | 13 | 0.8656 | 0.5000 | 0.5000 | 0.8203 | RUN_DOCUMENTED (unstable) |

Baseline ref (multirun seed42 default HPs): **0.9780** · Δwinner **+0.0010**

### Winner train HPs (CAD-CBA-v1)
| HP | Value |
|----|-------|
| lr | 5.893e-5 |
| batch_size | 1024 |
| focal_gamma | 1.917 |
| dropout_rate | 0.148 |
| attention_dropout | 0.214 |
| weight_decay | 1.916e-4 |
| scheduler | cosine |

Ckpt: `model/hpo/refine_rank2_trial008_seed42.pth` md5 `f9360aec2c003815140823cfe9b2a386`

**Interpretation**
- Controlled Optuna search beats default multirun seed42 HPs slightly but **credibly** under full protocol val.  
- Cosine + lower lr + larger batch cluster on Stage A; Stage B refine is required (trial 11 did not transfer).  
- Arch dims not searched (KD init / CAD-CBA-v1 freeze) — WP2c only if plateaus.  
- Test still sealed; multi-seed confirm + FT from ensemble KD remain open.

Tracker: B1/B6–B8/L6/WP3 updated. CAD-CBA-v1 train HPs → `hpo_best.yaml`.

**Next science:** FT multirun from ensemble KD + HPO HPs **or** WP5 ablations **or** multi-seed HPO confirm.

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
