# 09 — Raw Number Tables (copy feedstock)

**HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`

| metric | value | source path | platform | n | date/commit | label |
|--------|-------|-------------|----------|--:|-------------|-------|
| champion_macro_f1 | 0.9790 (report 0.9790) | twostage_botiot.json | train local | 1 FT run | mtime 2026-07-02; f98bf33 path | CURRENT |
| champion_md5 | 80a90f7cc210276300eaa90173a5a385 | md5sum model/best_model_botiot_twostage.pth | disk | — | audit 2026-07-18 | CURRENT |
| rf_test_macro_f1 | 0.9864 (report 0.9864) | rf_baseline_processed.json | CPU sklearn | 200 trees | acdcba5 | CURRENT |
| rf_gap_pct | 0.74 | computed | — | — | f98bf33 | CURRENT |
| kd_stage1_f1 | 0.9763 | distill_botiot_a0.6_T10.0_focal2.json | local | 1 | f98bf33 | CURRENT |
| kd_stage1_repro_identical | yes same macros | distill_botiot_a0.6_T10.0_focal2_repro.json | local | 1 | 2560348 | CURRENT |
| ton_clean_cnn_f1 | 0.9526 | toniot_clean_comparison.json | local | 1 | 5dda060 | CURRENT |
| ton_clean_gap_pct | 3.25 | toniot_clean_comparison.json | local | 1 | 5dda060 | CURRENT |
| llm_overhead_p99_us | 16.6 | llm_explainability.json | local | 5000 | 09b509b | CURRENT |
| streaming_gpu_batched_max | 25899 | streaming_throughput.json | RTX3050 | sweep | a11196b era | CURRENT |
| energy_rtx_mj_per_flow | 0.79 | energy_efficiency.json | RTX3050 | — | tracked | CURRENT |
| energy_a100_mj_per_flow | 1.089 | a100_energy.json | A100 | 500 | ca24ccf | CURRENT |
| a100_throughput_derived | 87791 | a100_energy avg_batch_time_ms=1.458 batch=128 | A100 | 500 | e928d8e fix | CURRENT |
| cuml_vram_mb | 444 | cuml_rf_resources.json | A100 | 2 | b194e8f | CURRENT |
| cuml_throughput | 2065669 | cuml_rf_resources.json | A100 | 2 | b194e8f | CURRENT |
| pytorch_b3_mean_us | 784.1 | pytorch_block3_stats_rtx3050.json | RTX3050 | 50 | d85271c | CURRENT |
| cuda_b3_fp16_live_mean_us | 531.59 | cuda_kernel_stats_rtx3050.json | RTX3050 | 100 | live file | CURRENT live session |
| cuda_derived_pipeline_live_us | 593.89 | cuda_kernel_stats derived | RTX3050 | 100 | live | CURRENT live |
| cuda_pipeline_range_us | 594–675 | verify_claims multi-session composition | RTX3050 | 5 sess claimed | d9e1f79 | CURRENT range |
| fw_eager_range_us | 2050–2247 | HIST+S3A+live | RTX3050 | 3 sess | bd3777e/d9e1f79 | CURRENT range |
| fw_trt_range_us | 2427–2966 | HIST+S3A+live | RTX3050 | 3 | d9e1f79 | CURRENT range |
| fw_compile_range_us | 1519–1777 | HIST+S3A+live | RTX3050 | 3 | d9e1f79 | CURRENT range |
| speedup_vs_eager_range | 3.04x–3.78x | range math | RTX3050 | 3×5 | d9e1f79 | CURRENT; Option A risk |
| speedup_vs_trt_range | 3.60x–4.99x | range math | RTX3050 | 3×5 | d9e1f79 | CURRENT; Option A risk |
| dicc_v100_pipeline_us | 550.664 | dicc_v100_summary.txt | V100S | 1 job 363046 | dd923c6 2026-06-21 | LEGACY single-shot |
| dicc_a100_pipeline_us | 592.044 | dicc_a100_summary.txt | A100 | 1 job 363047 | dd923c6 2026-06-21 | LEGACY single-shot |
| dicc_multiday_any | ABSENT | benchmarks/results/dicc/ | — | 0 | audit check | ABSENT |
| fidelity_export_max_abs | 0.0 | numerical_fidelity.json | local | 10 | 3f99243 | CURRENT |
| fidelity_cuda_selfcheck | 6/6 PASS | numerical_fidelity.json | local | 6 | 3f99243 | CURRENT |
| mlp_f1_twostage | 0.9542 | mlp_twostage.json | local | 1 | 0553542 | CURRENT |
| mlp_latency_us | 175.48 | mlp_latency.json | A100 claimed | — | ca24ccf | CURRENT |
| ensemble_f1 | 0.9529 | ensemble_distill.json | local | 1 | 0553542 | CURRENT not champion |
| rf_strengthen_best | 0.9885 | rf_teacher_strengthen.json | local | sweep | 31940dd | DIAGNOSTIC not published bar |
| superseded_twostage_0.9639 | 0.9639 | twostage.json | local | 1 | 0553542 | SUPERSEDED |
| fabricated_llm_5.19 | 5.19 us | removed | — | — | 09b509b fixed | SUPERSEDED/INVALID |
| fabricated_2.76x | 2.76x | unsourced 1864 | — | — | a0ff1a8 fixed | SUPERSEDED/INVALID |
| rostam_b3_v100_approx | CUDA~581 PT~512 | DESIGN_PLAN §5.3 only | Rostam | Day1 only | tooling trial | TOOLING-ONLY not official |

## KD sweep full (from on-disk JSON)

| file | alpha | T | gamma | val_f1 | test_f1 | git |
|------|------:|--:|------:|-------:|--------:|-----|
| distill_botiot.json | 0.5 | None | None | 0.9702866084457449 | 0.9481128622066717 | T |
| distill_botiot_a0.3_T1.0.json | 0.3 | 1.0 | None | 0.9473907163166718 | 0.9421075839256872 | T |
| distill_botiot_a0.5_T3.0.json | 0.5 | 3.0 | None | 0.9540667878645603 | 0.934063440132222 | T |
| distill_botiot_a0.6_T10.0_focal1.json | 0.6 | 10.0 | 1.0 | 0.9703814963668179 | 0.942578059126754 | GITIGNORED |
| distill_botiot_a0.6_T10.0_focal2.json | 0.6 | 10.0 | 2.0 | 0.9756881295269905 | 0.9763373489452476 | GITIGNORED |
| distill_botiot_a0.6_T10.0_focal2_repro.json | 0.6 | 10.0 | 2.0 | 0.9756881295269905 | 0.9763373489452476 | GITIGNORED |
| distill_botiot_a0.6_T10.0_focal3.json | 0.6 | 10.0 | 3.0 | 0.945698777188386 | 0.9503874727326516 | GITIGNORED |
| distill_botiot_a0.6_T10.0_focal4.json | 0.6 | 10.0 | 4.0 | 0.9768441320656844 | 0.9693064117615542 | GITIGNORED |
| distill_botiot_a0.6_T7.0_focal2.json | 0.6 | 7.0 | 2.0 | 0.9780220777564075 | 0.9702018195307816 | GITIGNORED |
| distill_botiot_a0.7_T1.0.json | 0.7 | 1.0 | None | 0.9598678022159038 | 0.9284490277061336 | T |
| distill_botiot_a0.7_T10.0_focal2.json | 0.7 | 10.0 | 2.0 | 0.9481823349291488 | 0.9033099357294712 | GITIGNORED |
| distill_botiot_a0.7_T5.0.json | 0.7 | 5.0 | None | 0.9620104745951614 | 0.9547140187639174 | T |
| distill_botiot_a0.7_T7.0_focal2.json | 0.7 | 7.0 | 2.0 | 0.9728410347569689 | 0.9686595408048941 | GITIGNORED |
| distill_botiot_a0.8_T10.0_focal2.json | 0.8 | 10.0 | 2.0 | 0.9672109909231121 | 0.9745145863799365 | GITIGNORED |
| distill_botiot_a0.8_T7.0_focal2.json | 0.8 | 7.0 | 2.0 | 0.9750766989005457 | 0.9757331814921264 | GITIGNORED |
| distill_botiot_a0.9_T1.0.json | 0.9 | 1.0 | None | 0.9566828503670044 | 0.9283570977027317 | T |
| distill_botiot_focal_T5.json | 0.7 | 5.0 | 2.0 | 0.9728271894149234 | 0.960094250319179 | T |

## Live cuda_kernel_stats means (one session on disk)

| block | mean_us | p50 | cv_pct |
|-------|--------:|----:|-------:|
| B1 | 57.334 | 51.612 | 26.71 |
| B2 | 117.793 | 90.144 | 251.06 |
| B3 no_graphs | 1020.012 | 728.387 | 284.83 |
| B3 graphs | 732.713 | 724.677 | 5.07 |
| B3 fp16 | 531.585 | 533.135 | 5.79 |
| B3 naive | 4783.022 | 4462.715 | 62.31 |
| B4 | 18.723 | 15.454 | 159.42 |
| pipeline B1+B2+B4 chained | 62.304 | | |
| derived total (+B3 fp16) | 593.889 | | |

## Live statistical_significance_v2 means (latest session only)

| method | mean_us | std | n |
|--------|--------:|----:|--:|
| Eager PyTorch | 2077.205 | 153.92111841756406 | 20 |
| torch.compile | 1776.264 | 192.61333203656395 | 20 |
| ORT CPU | 522.307 | 35.46520682175754 | 20 |
| ORT GPU | 4059.135 | 271.4513652418115 | 20 |
| TensorRT | 2777.387 | 74.94349144822948 | 20 |
| Custom CUDA FP16 | 652.422 | 50.0771707282943 | 100 |