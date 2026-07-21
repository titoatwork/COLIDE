# Method package decision (Phase 2) — signed default

**Date:** 2026-07-19  
**Last status update:** 2026-07-21 (WP3 Optuna HPO — train HPs INCORPORATE 0.9791)  
**Rule:** one package first; other ideas still RUN_DOCUMENTED later.

## Chosen package (v1): **Class-aware distilled CNN–BiLSTM (CAD-CBA-v1)**

| Component | Decision | Tracker |
|-----------|----------|---------|
| Base arch | Keep V3 CNN–BiLSTM–Attention (`cnn_bilstm_v3_attention`) | F3 |
| KD teacher | **Ensemble** (mean RF+XGB+LGBM soft labels); RF fallback | E1–E5, E7 |
| KD recipe | α=0.6, T=10 (stage_a_kd); FT focal γ from HPO ≈1.92 | B8, B10, E* |
| Loss | **Focal** wins 4-way val compare (CE / focal / focal_cb / logit_adj) | D2–D5 |
| Train HPs | **Optuna WP3 winner** → `config/hpo_best.yaml` (val 0.9791) | B1, B6–B8, L6 |
| Thresholds | Val search on focal seed42: **no gain** → keep **argmax** (RUN_DOCUMENTED) | D7, C11 |
| Multi-scale CNN / new attention | **Later** only if CAD-CBA-v1 plateaus | C3–C4 |
| SupCon / asymmetric / uncertainty | Bounded run later (skip-nothing) | C7–C10 |

## Named weakness
Extreme class imbalance + minority (Theft) under neural deploy path; RF still stronger on published path — student must win on **trade-off** and/or close minority gap under **same protocol**.

## Success criteria (val first)
1. val macro-F1 ≥ baseline multirun mean  
2. val min per-class F1 / Theft F1 improved vs pure focal FT  
3. Then sealed multi-seed test + multi-obj vs RF/XGB  

## Explicit non-goals for v1
- Full V3 CUDA parity (Option B)  
- Beating RF on every metric before multi-obj tables  

## Status (as of 2026-07-21 WP3)

| Item | Status | Evidence |
|------|--------|----------|
| Multirun baseline FT (5 seeds) | **DONE** | mean **0.9714 ± 0.0109**; `benchmarks/results/multirun/summary.json` |
| Loss compare CE / focal / focal_cb / logit_adj | **DONE** | focal **INCORPORATED** 0.9780; CB/logit_adj **RUN_DOCUMENTED** worse |
| Default loss for CAD-CBA-v1 | **focal** | keep (γ from HPO ≈1.92) |
| Val thresholds on best focal ckpt | **RUN_DOCUMENTED** | all variants = argmax 0.9780; keep argmax (`thresholds_focal_seed42.json`) |
| Default decode for CAD-CBA-v1 | **argmax** | thresholds not incorporated |
| Teacher/KD under protocol (WP4b) | **DONE** | ensemble student **0.9401** INCORPORATE; `teachers_kd/summary.json` |
| Default KD teacher for CAD-CBA-v1 | **ensemble** | RF fallback; none/xgb/lgbm RUN_DOCUMENTED |
| Optuna HPO (WP3) | **DONE** | full-train refine winner **0.9791 INCORPORATE**; `hpo/summary.json` + `config/hpo_best.yaml` |
| Default train HPs for CAD-CBA-v1 | **hpo_best.yaml** | lr 5.89e-5, batch 1024, γ≈1.92, cosine, … |
| Arch deltas (attention/multi-scale) | deferred | only if plateaus |
| Multi-seed / sealed test of HPO winner | **TODO** | next confirm track |

## Negative results locked in (do not re-litigate without new protocol)
- `focal_cb` val macro-F1 0.9121 — hurts macro  
- `logit_adj` val macro-F1 0.9225 — hurts macro  
- LGBM solo teacher val 0.5928 → student 0.8829 — weak KD path  
- XGB solo teacher 0.9918 but student 0.9270 — does **not** beat ensemble/RF student  
- HPO Stage-A best trial 11 → full refine **0.9721** — Stage A rank ≠ Stage B rank  
- HPO refine rank3 trial 13 → **0.8656** — unstable under full data  

See `RESULTS_DISK_MANIFEST.md` for md5s and paths.
