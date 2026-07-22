# Protocol Claims Package (WP9a)

**Generated (UTC):** 2026-07-22T15:04:00.342965+00:00
**Git:** `d058030052824bd17f58dcae254d6ad3d0f75941`
**Champion md5:** `80a90f7cc210276300eaa90173a5a385` (unchanged=True; expected `80a90f7cc210276300eaa90173a5a385`)
**Method:** CAD-CBA-v1

> Source of truth for **protocol-era** numbers. Every row traces to a local JSON under `benchmarks/results/` (gitignored) with md5 recorded here and in `RESULTS_DISK_MANIFEST.md`. Never invent multi-day DICC numbers.

## Claim inventory

| ID | Value | Status | Source | Notes |
|----|-------|--------|--------|-------|
| `wp1b_multirun_mean` | **0.9714** | LOCKED_VAL | `multirun/summary.json` | WP1b n=5 val-only |
| `wp1b_multirun_std` | **0.0109** | LOCKED_VAL | `multirun/summary.json` |  |
| `wp3_hpo_winner_val` | **0.9791** | LOCKED_VAL | `hpo/summary.json` | INCORPORATE train HPs |
| `package_ensemble_hpo_mean` | **0.9639** | LOCKED_VAL | `multirun_ensemble_hpo/summary.json` | RUN_DOCUMENTED; not mean-win vs WP1b |
| `package_ensemble_hpo_std` | **0.0185** | LOCKED_VAL | `multirun_ensemble_hpo/summary.json` |  |
| `hpo_confirm_mean` | **0.9689** | LOCKED_VAL | `multirun_hpo_confirm/summary.json` |  |
| `hpo_confirm_std` | **0.0145** | LOCKED_VAL | `multirun_hpo_confirm/summary.json` |  |
| `wp4b_ensemble_student_val` | **0.9401** | LOCKED_VAL | `teachers_kd/kd_ensemble_seed42.json` | INCORPORATE KD teacher (stage_a_kd) |
| `g5_lgbm_val` | **0.9818** | LOCKED_VAL | `baselines_classical/lgbm_seed42.json` | protocol classical top |
| `g3_rf_val` | **0.9778** | LOCKED_VAL | `baselines_classical/rf_seed42.json` |  |
| `g4_xgb_val` | **0.9762** | LOCKED_VAL | `baselines_classical/xgb_seed42.json` |  |
| `g2_svm_val` | **0.4268** | LOCKED_VAL | `baselines_classical/svm_seed42.json` | RUN_DOCUMENTED weak |
| `published_rf_test` | **0.9864** | HISTORICAL | `rf_baseline_processed.json` | NOT protocol-fair; dual bar only |
| `wp5a_a7_val` | **0.9699** | LOCKED_VAL | `ablation_ladder/A7_full_cad_cba_v1_seed42.json` | ladder top package path |
| `wp5a_a3_val` | **0.9493** | LOCKED_VAL | `ablation_ladder/A3_cnn_bilstm_seed42.json` |  |
| `wp5a_a4_val` | **0.7378** | LOCKED_VAL | `ablation_ladder/A4_cnn_bilstm_attn_ce_seed42.json` | attn+CE underperforms A3 |
| `g11_cnn_bilstm_val` | **0.9493** | LOCKED_VAL | `baselines_neural/G11_cnn_bilstm_seed42.json` |  |
| `g6_mlp_val` | **0.9285** | LOCKED_VAL | `baselines_neural/G6_mlp_seed42.json` |  |
| `g12_transformer_val` | **0.5808** | LOCKED_VAL | `baselines_neural/G12_transformer_seed42.json` | weak under equal budget |
| `cstar_ctrl_val` | **0.9787** | LOCKED_VAL | `cstar_bounded/CTRL_control_v3_focal_seed42.json` |  |
| `cstar_c4_val` | **0.9167** | LOCKED_VAL | `cstar_bounded/C4_multi_scale_seed42.json` | no package incorporate |
| `e6_neural_teacher_student_val` | **0.8513** | LOCKED_VAL | `teachers_kd_neural/summary.json` | ≪ ensemble 0.9401; keep ensemble |
| `xai_rank_corr` | **0.9636** | LOCKED_VAL | `xai/summary.json` |  |
| `xai_faith_mass` | **0.5109** | LOCKED_VAL | `xai/summary.json` |  |
| `xai_dispatch_p99_us` | **16.60** | LOCKED_VAL | `xai/summary.json` | dispatch only — not gen |
| `xai_llm_feature_mention` | **0.333** | LOCKED_VAL | `xai/summary.json` | free-form weak; not class_mention_rate |
| `xai_llm_top3_agree` | **0.167** | LOCKED_VAL | `xai/summary.json` | weak top-3 agreement |
| `xai_llm_gen_mean_ms` | **7400** | LOCKED_VAL | `xai/summary.json` | never conflate with dispatch µs |
| `xai_structured_usefulness` | **1.0** | LOCKED_VAL | `xai/summary.json` |  |
| `xai_j10` | **DROP_FULL_EXPLAINABLE_CLAIM_KEEP_STRUCTURED** | LOCKED_VAL | `xai/summary.json` | DROP full claim keep structured+dispatch |
| `f9_rtx_mj_per_flow` | **0.786** | HISTORICAL | `energy_table/summary.json` | HISTORICAL single-shot batch128; prefer WP6b multi-session range |
| `pareto_h8_composite_g6` | **0.9056** | LOCKED_VAL | `pareto_h8/summary.json` | a priori composite #1 G6 |
| `wp6b_energy_mj_per_flow_mean` | **0.933** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | multi-session n=5; supersedes single-shot for ranges |
| `wp6b_energy_mj_per_flow_std` | **0.010** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` |  |
| `wp6b_energy_mj_per_flow_range_low` | **0.920** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | session-mean min |
| `wp6b_energy_mj_per_flow_range_high` | **0.943** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | session-mean max |
| `wp6b_pt_batch256_us_mean` | **24.90** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | full V3 PT absolute (Option A allowed) |
| `wp6b_pt_batch256_us_range_low` | **24.15** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` |  |
| `wp6b_pt_batch256_us_range_high` | **25.68** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` |  |
| `wp6b_cuda_pipeline_us_range_low` | **565.2** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | Option A derived sum; not full V3 parity |
| `wp6b_cuda_pipeline_us_range_high` | **570.3** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | Option A derived sum; not full V3 parity |
| `wp6b_cuda_block3_fp16_us_range_low` | **503.2** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | per-block Option A |
| `wp6b_cuda_block3_fp16_us_range_high` | **508.5** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | per-block Option A |
| `wp6b_peak_alloc_mb` | **322.2** | LOCKED_SYSTEMS | `wp6b_local_ranges/summary.json` | H3 peak alloc across batches/sessions |
| `wp6b_i8_batch256_us_mean` | **24.49** | LOCKED_SYSTEMS | `systems_i8_h3/summary.json` | I8 multi-session |
| `ton_val` | **0.8080** | LOCKED_TEST | `toniot_final/summary.json` | 13-feat processed_toniot |
| `ton_test` | **0.8110** | LOCKED_TEST | `toniot_final/summary.json` | WP8 ToN test allowed |
| `ton_rf_test` | **0.9393** | LOCKED_TEST | `toniot_final/summary.json` |  |
| `d6_shuffle_val` | **0.9791** | LOCKED_VAL | `stratified_batch/ft_shuffle_seed42.json` | keep shuffle default |
| `d6_stratified_val` | **0.9209** | LOCKED_VAL | `stratified_batch/ft_stratified_seed42.json` | hurts under hpo_best |
| `d3_focal_val` | **0.9780** | LOCKED_VAL | `imbalance_loss/ft_focal_seed42.json` | INCORPORATED CAD-CBA loss |
| `fidelity_bit_identical` | **true** | LOCKED_VAL | `numerical_fidelity.json` | WP6a re-export |
| `champion_md5` | **80a90f7cc210276300eaa90173a5a385** | LOCKED_VAL | `model/best_model_botiot_twostage.pth` | never clobber without BACKUP + OK |
| `bot_sealed_test_mean` | **0.9780** | LOCKED_TEST | `sealed_test/summary.json` | B14 init path A; n=5 |
| `bot_sealed_test_std` | **0.0033** | LOCKED_TEST | `sealed_test/summary.json` | B14 multi-seed std |
| `bot_sealed_test_min_cls_mean` | **0.9292** | LOCKED_TEST | `sealed_test/summary.json` | B14 test min-cls mean |
| `bot_sealed_test_theft_mean` | **1.0000** | LOCKED_TEST | `sealed_test/summary.json` | B14 test Theft mean |
| `bot_sealed_test_multiseed` | **0.9780±0.0033** | LOCKED_TEST | `sealed_test/summary.json` | B14 path A; champion_unchanged=True |
| `dicc_multiday_stats` | **PENDING** | BLOCKED_DICC | `PENDING` | WP0 / I1–I6 user-scheduled DICC session |

## Minority / per-class F1 (val only)

| Model | macro-F1 | min-cls | DDoS | DoS | Normal | Reconnaissance | Theft | source |
|-------|----------|---------|------|------|------|------|------|--------|
| HPO winner seed42 | 0.9791 | 0.9351 | 0.9836 | 0.9812 | 0.9351 | 0.9955 | 1.0000 | `hpo/refine_rank2_trial008_seed42.json` |
| WP1b multirun seed42 | 0.9780 | 0.9315 | 0.9833 | 0.9807 | 0.9315 | 0.9948 | 1.0000 | `multirun/ft_seed42.json` |
| Focal FT seed42 | 0.9780 | 0.9315 | 0.9833 | 0.9807 | 0.9315 | 0.9948 | 1.0000 | `imbalance_loss/ft_focal_seed42.json` |
| LGBM protocol | 0.9818 | 0.9231 | 0.9998 | 0.9998 | 0.9867 | 0.9998 | 0.9231 | `baselines_classical/lgbm_seed42.json` |
| RF protocol | 0.9778 | 0.9231 | 0.9914 | 0.9900 | 0.9867 | 0.9977 | 0.9231 | `baselines_classical/rf_seed42.json` |
| XGB protocol | 0.9762 | 0.9231 | 0.9997 | 0.9997 | 0.9589 | 0.9997 | 0.9231 | `baselines_classical/xgb_seed42.json` |
| A7 CAD-CBA ladder | 0.9699 | 0.8974 | 0.9828 | 0.9803 | 0.8974 | 0.9888 | 1.0000 | `ablation_ladder/A7_full_cad_cba_v1_seed42.json` |
| CTRL cstar | 0.9787 | 0.9333 | 0.9836 | 0.9812 | 0.9333 | 0.9952 | 1.0000 | `cstar_bounded/CTRL_control_v3_focal_seed42.json` |
| Package seed45 (max) | 0.9803 | 0.9474 | 0.9827 | 0.9801 | 0.9474 | 0.9912 | 1.0000 | `multirun_ensemble_hpo/ft_seed45.json` |
| HPO confirm seed42 | 0.9791 | 0.9351 | 0.9835 | 0.9811 | 0.9351 | 0.9956 | 1.0000 | `multirun_hpo_confirm/ft_seed42.json` |
| Ensemble KD stage_a | 0.9401 | 0.8434 | 0.9813 | 0.9794 | 0.8434 | 0.9734 | 0.9231 | `teachers_kd/kd_ensemble_seed42.json` |

## Quantitative advantage snapshot (A4 / multi-obj)

| Dimension | Evidence | Source |
|-----------|----------|--------|
| Protocol classical F1 top | LGBM **0.9818** | `lgbm_seed42.json` |
| Protocol neural HPO | **0.9791** | `hpo/summary.json` |
| Protocol neural multirun | **0.9714±0.0109** | `multirun/summary.json` |
| A priori composite #1 | G6 score **0.9056** @ 4.33 µs F1 0.9285 | `pareto_h8` / energy headline |
| Energy (RTX batch128) | **0.786 mJ/flow** | `energy_table/summary.json` |
| LLM dispatch p99 | **16.60 µs** (dispatch only) | `xai` / energy |
| XAI policy | DROP full explainable claim; keep structured+dispatch | `xai/summary.json` J10 |
| ToN honesty | test 0.8110 ≪ RF 0.9393 (13-feat) | `toniot_final` |
| Published RF dual bar | 0.9864 ≠ protocol RF 0.9778 | historical vs protocol |

## Open gates (do not claim as done)

- WP0 DICC multi-day (user-scheduled) — I1–I5, H7, K7, I11
- Final journal class file / BibTeX after PI venue choice (local-complete draft + PI venue polish DONE: `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf`)

## Usage

```bash
PYTHONPATH=. python3 scripts/build_claims_package.py
PYTHONPATH=. python3 scripts/verify_claims.py
# protocol prose block lives in docs/paper_text_blocks.md §Protocol-era
```

## Status legend

| Status | Meaning |
|--------|---------|
| LOCKED_VAL | Val-only protocol number; safe to quote with source |
| LOCKED_TEST | Explicitly allowed test number (ToN WP8) |
| HISTORICAL | Prior pipeline / dual bar — label carefully |
| PENDING_SEALED_TEST | Wait for user final-config lock + B14 run |
| BLOCKED_DICC | Wait for dedicated DICC session |
