# Method package decision (Phase 2) — signed default

**Date:** 2026-07-19  
**Last status update:** 2026-07-22 (C*/E6/B2–B4 DONE — all novelty probes RUN_DOCUMENTED reject; CAD-CBA-v1 package locked; train HPs stay INCORPORATED)  
**Rule:** one package first; **every other playlist idea still gets a bounded run → RUN_DOCUMENTED** (skip-nothing).

## Chosen package (v1): **Class-aware distilled CNN–BiLSTM (CAD-CBA-v1)**

| Component | Decision | Tracker |
|-----------|----------|---------|
| Base arch | Keep V3 CNN–BiLSTM–Attention (`cnn_bilstm_v3_attention`) | F3 |
| KD teacher | **Ensemble** (mean RF+XGB+LGBM soft labels); RF fallback | E1–E5, E7 |
| KD recipe | α=0.6, T=10 (stage_a_kd); FT focal γ from HPO ≈1.92 | B8, B10, E* |
| Loss | **Focal** wins 4-way val compare (CE / focal / focal_cb / logit_adj) | D2–D5 |
| Train HPs | **Optuna WP3 winner** → `config/hpo_best.yaml` (val 0.9791) | B1, B6–B8, L6 |
| Thresholds | Val search on focal seed42: **no gain** → keep **argmax** (RUN_DOCUMENTED) | D7, C11 |
| Multi-scale CNN / new attention | **Rejected** (C4 0.9167 / A4 hurts; keep V3) | C3–C4 |
| SupCon / asymmetric / uncertainty | **Rejected** bounded (C7 0.7732 / C8 0.8012 / C10 no lift) | C7–C10 |
| Gated fusion | **Rejected** (C5 0.9132 ≪ CTRL) | C5 |
| Neural teacher KD | **Rejected** (E6 student 0.8513 ≪ ensemble 0.9401) | E6 |
| Arch HPO B2–B4 | **Plateau reject** — freeze V3 dims | B2–B4 |

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
| Multi-seed confirm of HPO winner (val) | **RUN_DOCUMENTED** | mean **0.9689 ± 0.0145** n=5; seed42 **0.9791** repro; does **not** beat WP1b mean |
| Sealed multi-seed **test** of final lock | **TODO** | only after final config freeze |
| Package FT multirun (ensemble KD + HPO HPs) | **RUN_DOCUMENTED** | mean **0.9639 ± 0.0185** n=5; max 0.9803; does **not** beat WP1b mean 0.9714±0.0109 |
| Ablation ladder A1–A7 (seed42) | **RUN_DOCUMENTED** | A7 **0.9699** tops; A3 0.9493; A4 attn+CE **0.7378** underperforms A3 — package composition credit |
| Protocol-fair neural baselines G6–G12 (seed42 CE) | **RUN_DOCUMENTED** | G11 cnn_bilstm **0.9493** tops suite; G6 MLP 0.9285; G12 transformer **0.5808** weak; equal fixed HPs (G15) |
| Classical G2 LinearSVC full | **RUN_DOCUMENTED** | val **0.4268** weak; pilot CV error fixed |
| Classical G5 LGBM fix | **DONE (val)** | val **0.9818** (balanced multiclass) tops protocol classical |
| D6 stratified batch vs shuffle | **RUN_DOCUMENTED** | stratified **0.9209** ≪ shuffle **0.9791** (Δ−0.058); **keep shuffle** |
| Default train_sampler for CAD-CBA-v1 | **shuffle** | inv-freq stratified hurts under hpo_best |

## Negative results locked in (do not re-litigate without new protocol)
- `focal_cb` val macro-F1 0.9121 — hurts macro  
- `logit_adj` val macro-F1 0.9225 — hurts macro  
- LGBM solo teacher val 0.5928 → student 0.8829 — weak KD path  
- XGB solo teacher 0.9918 but student 0.9270 — does **not** beat ensemble/RF student  
- HPO Stage-A best trial 11 → full refine **0.9721** — Stage A rank ≠ Stage B rank  
- HPO refine rank3 trial 13 → **0.8656** — unstable under full data  
- Package multirun ensemble+HPO mean **0.9639 ± 0.0185** < WP1b **0.9714 ± 0.0109** — HPO HPs (tuned on old distill) do not lift multi-seed mean on ensemble KD init; seed 43 weak (0.9328)
- HPO multi-seed confirm mean **0.9689 ± 0.0145** < WP1b **0.9714 ± 0.0109** — seed42 reproduces 0.9791 but seed46 (0.9483) / seed43 (0.9587) drag; keep train HPs, no mean-win claim
- Ablation A4 attn+CE **0.7378** < A3 cnn_bilstm CE **0.9493** under seed42/8-ep — attention alone is not a free gain; A7 package path still wins ladder (**0.9699**)
- WP5b G12 lightweight temporal transformer CE scratch **0.5808** < G11 **0.9493** under equal budget — transformer not a free architecture win on BoT-IoT tabular protocol
- WP5b G7 1D-CNN **0.6221** weak (matches A1); pure CNN insufficient under this budget
- D6 inv-freq stratified batch **0.9209** ≪ shuffle **0.9791** (Δ−0.058) — do not add class-balanced batch sampling to CAD-CBA-v1
- G2 LinearSVC full **0.4268** — linear SVM not competitive under protocol imbalance
- G5 LGBM legacy **0.5512** superseded by balanced fix **0.9818** (document both; paper uses fixed)
- C4 multi-scale **0.9167** ≪ CTRL **0.9787** — not free arch win
- C5 gated fusion **0.9132** ≪ CTRL — not free fusion win
- C7 SupCon+focal **0.7732** (Theft=0) — hurts under budget
- C8 asymmetric **0.8012** — hurts vs focal
- C10 MC-dropout selective — no robust high-coverage lift over argmax 0.9791
- E6 G11 neural teacher → student **0.8513** ≪ ensemble KD student **0.9401** — keep tree ensemble teacher
- B2–B4 full arch Optuna — plateau reject (`B2B4_ARCH_HPO_PLATEAU_REJECT.md`)

See `RESULTS_DISK_MANIFEST.md` for md5s and paths.
