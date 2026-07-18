# 02 — Results Inventory (every file under benchmarks/results/)

**Audit HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`  
**Total files:** 54  
**Multi-day tree:** `benchmarks/results/dicc/` → **ABSENT**

## Master table

| path | size | mtime | git | status label | headline metrics | producer (confirmed/heuristic) | platform | n | multi-session? |
|------|-----:|-------|-----|--------------|------------------|-------------------------------|----------|--:|----------------|
| `a100_energy.json` | 202 | 2026-06-27T00:14:58 | TRACKED | LEGACY/SINGLE-SHOT CURRENT for energy table | energy_per_flow_mj=1.088789 | `scripts/benchmark_a100_energy.py` | A100 DICC |  | no/single-file |
| `ablation_mlp.json` | 174 | 2026-06-25T22:52:58 | TRACKED | CURRENT ablation | macro_f1=0.9623604184972688; best_val_f1=0.9588704157121125 | `scripts/train_mlp_distill.py` | local train |  | no/single-file |
| `baseline_latency.json` | 391 | 2026-06-07T06:31:14 | TRACKED | LEGACY point estimates |  | `early baseline (benchmark_batch/ort era)` | RTX 3050 |  | no/single-file |
| `cuda_kernel_stats_rtx3050.json` | 2411 | 2026-07-02T15:13:37 | GITIGNORED_ON_DISK | CURRENT live file; multi-session ranges LIVE+hardcoded in verify_claims | derived_total=593.8892259999999; b3fp16=531.5854899999999; b3naive=4783.022000000001 | `scripts/benchmark_cuda_kernels_stats.py` | RTX 3050 WSL2 | 100 | see verify_claims HIST+sessions |
| `cuml_rf_native.json` | 412 | 2026-06-24T01:26:58 | TRACKED | LEGACY; F1 labels include stale 0.9601 |  | `scripts/benchmark_cuml_rf_native.py` | V100S+laptop |  | no/single-file |
| `cuml_rf_resources.json` | 414 | 2026-06-26T19:18:42 | TRACKED | CURRENT VRAM/throughput table; F1 0.9639 STALE vs champion | cuml_rf_vram_mb=444 | `A100 resource script (b194e8f)` | A100 |  | no/single-file |
| `dicc_a100_summary.txt` | 498 | 2026-06-20T20:34:47 | TRACKED | LEGACY single-shot CUDA-only | Pipeline total (with B3 FP16): 592.044 us | `DICC job 363047 Jun 2026` | A100 |  | no/single-file |
| `dicc_v100_summary.txt` | 499 | 2026-06-20T20:34:47 | TRACKED | LEGACY single-shot CUDA-only | Pipeline total (with B3 FP16): 550.664 us | `DICC job 363046 Jun 2026` | V100S |  | no/single-file |
| `distill_botiot.json` | 220 | 2026-06-23T02:19:20 | TRACKED | CURRENT KD table row a=0.5 T=1 | macro_f1=0.9481128622066717; best_val_f1=0.9702866084457449 | `scripts/train_distill.py / sweep` | local |  | no/single-file |
| `distill_botiot_a0.3_T1.0.json` | 205 | 2026-06-23T13:52:52 | TRACKED | CURRENT KD | macro_f1=0.9421075839256872; best_val_f1=0.9473907163166718 | `sweep_distill / train_distill` | local |  | no/single-file |
| `distill_botiot_a0.5_T3.0.json` | 204 | 2026-06-23T18:10:40 | TRACKED | CURRENT KD | macro_f1=0.934063440132222; best_val_f1=0.9540667878645603 | `sweep` | local |  | no/single-file |
| `distill_botiot_a0.6_T10.0_focal1.json` | 249 | 2026-07-02T08:53:27 | GITIGNORED_ON_DISK | CURRENT negative/γ result; GITIGNORED | macro_f1=0.942578059126754; best_val_f1=0.9703814963668179 | `focal-γ sweep` | local |  | no/single-file |
| `distill_botiot_a0.6_T10.0_focal2.json` | 250 | 2026-07-01T19:48:06 | GITIGNORED_ON_DISK | CURRENT stage-1; GITIGNORED; bit-repro match | macro_f1=0.9763373489452476; best_val_f1=0.9756881295269905 | `sweep Round 2 champion stage-1` | local |  | no/single-file |
| `distill_botiot_a0.6_T10.0_focal2_repro.json` | 250 | 2026-07-02T05:50:00 | GITIGNORED_ON_DISK | CURRENT bit-identical Stage-1; GITIGNORED | macro_f1=0.9763373489452476; best_val_f1=0.9756881295269905 | `repro re-run` | local |  | no/single-file |
| `distill_botiot_a0.6_T10.0_focal3.json` | 249 | 2026-07-02T10:14:22 | GITIGNORED_ON_DISK | CURRENT negative; GITIGNORED | macro_f1=0.9503874727326516; best_val_f1=0.945698777188386 | `focal-γ sweep` | local |  | no/single-file |
| `distill_botiot_a0.6_T10.0_focal4.json` | 250 | 2026-07-02T13:03:24 | GITIGNORED_ON_DISK | CURRENT; GITIGNORED | macro_f1=0.9693064117615542; best_val_f1=0.9768441320656844 | `focal-γ sweep` | local |  | no/single-file |
| `distill_botiot_a0.6_T7.0_focal2.json` | 248 | 2026-07-01T13:15:58 | GITIGNORED_ON_DISK | CURRENT; GITIGNORED | macro_f1=0.9702018195307816; best_val_f1=0.9780220777564075 | `Round 2` | local |  | no/single-file |
| `distill_botiot_a0.7_T1.0.json` | 205 | 2026-06-23T15:08:46 | TRACKED | CURRENT | macro_f1=0.9284490277061336; best_val_f1=0.9598678022159038 | `Round 1` | local |  | no/single-file |
| `distill_botiot_a0.7_T10.0_focal2.json` | 250 | 2026-07-01T20:55:43 | GITIGNORED_ON_DISK | CURRENT negative outlier; GITIGNORED | macro_f1=0.9033099357294712; best_val_f1=0.9481823349291488 | `Round 2 outlier` | local |  | no/single-file |
| `distill_botiot_a0.7_T5.0.json` | 204 | 2026-06-23T20:33:11 | TRACKED | CURRENT | macro_f1=0.9547140187639174; best_val_f1=0.9620104745951614 | `Round 1` | local |  | no/single-file |
| `distill_botiot_a0.7_T7.0_focal2.json` | 249 | 2026-07-01T15:25:44 | GITIGNORED_ON_DISK | CURRENT; GITIGNORED | macro_f1=0.9686595408048941; best_val_f1=0.9728410347569689 | `Round 2` | local |  | no/single-file |
| `distill_botiot_a0.8_T10.0_focal2.json` | 250 | 2026-07-01T23:26:53 | GITIGNORED_ON_DISK | CURRENT; GITIGNORED | macro_f1=0.9745145863799365; best_val_f1=0.9672109909231121 | `Round 2` | local |  | no/single-file |
| `distill_botiot_a0.8_T7.0_focal2.json` | 249 | 2026-07-01T17:32:53 | GITIGNORED_ON_DISK | CURRENT; GITIGNORED | macro_f1=0.9757331814921264; best_val_f1=0.9750766989005457 | `Round 2` | local |  | no/single-file |
| `distill_botiot_a0.9_T1.0.json` | 205 | 2026-06-23T16:45:08 | TRACKED | CURRENT | macro_f1=0.9283570977027317; best_val_f1=0.9566828503670044 | `Round 1` | local |  | no/single-file |
| `distill_botiot_focal_T5.json` | 247 | 2026-06-23T22:44:37 | TRACKED | CURRENT 0.9601 | macro_f1=0.960094250319179; best_val_f1=0.9728271894149234 | `Round 1 best KD+focal` | local |  | no/single-file |
| `distill_toniot.json` | 222 | 2026-06-23T02:30:16 | TRACKED | SUPERSEDED by v2 for original path | macro_f1=0.8061386557326262; best_val_f1=0.8010736614424367 | `train_distill_toniot` | local |  | no/single-file |
| `distill_toniot_v2.json` | 232 | 2026-06-23T02:58:35 | TRACKED | CURRENT original ToN 0.8254 | macro_f1=0.8253809344574631; best_val_f1=0.8233781107224228 | `train_distill_toniot_v2` | local |  | no/single-file |
| `energy_efficiency.json` | 437 | 2026-06-19T12:24:01 | TRACKED | CURRENT 0.79 mJ/flow |  | `benchmark_energy.py` | RTX 3050 |  | no/single-file |
| `ensemble_distill.json` | 73 | 2026-06-26T18:53:27 | TRACKED | CURRENT 0.9529; not champion | macro_f1=0.9528816634630726 | `train_ensemble_distill.py` | local |  | no/single-file |
| `gpu_hardware_profile.json` | 689 | 2026-06-26T19:04:28 | TRACKED | CURRENT occupancy theory |  | `profiling` | RTX 3050 |  | no/single-file |
| `llm_explainability.json` | 4528 | 2026-07-01T03:58:18 | TRACKED | CURRENT 16.60 p99; RE-VERIFIED 09b509b | overhead_p99_us=16.599169557594042; dispatch_trials=5000 | `llm_explainability.py` | local |  | no/single-file |
| `mlp_latency.json` | 322 | 2026-06-27T00:15:08 | TRACKED | CURRENT 175 us; note field says update if different | avg_latency_us=175.48 | `benchmark_mlp_latency.py` | A100 (claimed) |  | no/single-file |
| `mlp_twostage.json` | 74 | 2026-06-26T01:13:22 | TRACKED | CURRENT 0.9542 | macro_f1=0.9541768098521576 | `MLP two-stage FT` | local |  | no/single-file |
| `numerical_fidelity.json` | 2608 | 2026-07-14T09:14:10 | GITIGNORED_ON_DISK | CURRENT; GITIGNORED | export_max_abs=0.0; cuda_selfchecks=6/6 | `numerical_fidelity.py S7` | local |  | no/single-file |
| `optimized_botiot_focal_only.json` | 304 | 2026-06-22T22:54:45 | TRACKED | CURRENT catastrophic collapse row | macro_f1=0.0010218424444893902; best_val_f1=0.0011698529274881831 | `train_optimized focal-only fail` | local |  | no/single-file |
| `pipeline_benchmark.json` | 923 | 2026-07-01T04:17:28 | TRACKED | LEGACY/PARTIAL single-run blocks; fixed a0ff1a8 |  | `benchmark_pipeline.py` | RTX 3050 |  | no/single-file |
| `pipeline_benchmark_nvidia_geforce_rtx_3050_6gb_laptop_gpu.json` | 923 | 2026-07-01T04:17:28 | GITIGNORED_ON_DISK | duplicate of pipeline_benchmark; GITIGNORED |  | `benchmark_pipeline GPU-tagged` | RTX 3050 |  | no/single-file |
| `pytorch_block3_stats_rtx3050.json` | 307 | 2026-07-01T08:03:28 | GITIGNORED_ON_DISK | CURRENT 784 us n=50; GITIGNORED | gpu_p50_us={'mean': 784.0712600227562, 'std': 88.60898085266037, 'p50': 755.7175013062079, 'p95': 946.8688503147858, 'min': 652.0349998027086, 'max': 1040.9690003143623, 'cv_pct':  | `benchmark_pytorch_block3_stats.py` | RTX 3050 | 50 | no/single-file |
| `real_weight_validation.json` | 179 | 2026-07-02T06:56:48 | TRACKED | CURRENT pass; re-export c27ac2a | classification_accuracy=0.98 | `validate_real_weights.py` | local |  | no/single-file |
| `rf_baseline_processed.json` | 475 | 2026-07-01T10:56:10 | GITIGNORED_ON_DISK | CURRENT 0.9864; GITIGNORED | test_macro_f1=0.9863877607840802 | `rf_baseline_processed.py` | local |  | no/single-file |
| `rf_baseline_toniot.json` | 181 | 2026-06-18T08:34:59 | TRACKED | CURRENT original ToN RF 0.9396 | macro_f1=0.9396369689945695 | `rf_baseline_toniot.py` | local |  | no/single-file |
| `rf_teacher_strengthen.json` | 5749 | 2026-07-14T09:11:16 | GITIGNORED_ON_DISK | CURRENT diagnostic 0.9885; published bar NOT raised; GITIGNORED | best={'name': 'trees_200_balanced', 'test_macro_f1': 0.9885334403304447, 'delta_vs_baseline_200': 0.002145679546364465}; baseline=0.9863877607840802 | `rf_teacher_strengthen.py S5` | local |  | no/single-file |
| `statistical_confidence.json` | 2781 | 2026-06-19T12:23:23 | TRACKED | LEGACY early CIs; SUPERSEDED by v2 for framework claims |  | `benchmark_stats.py` | RTX 3050 |  | no/single-file |
| `statistical_significance_v2.json` | 4023 | 2026-07-02T14:35:59 | TRACKED | CURRENT live session; ranges via HIST+3A+live | Eager PyTorch.mean_us=2077.2045994497603; torch.compile.mean_us=1776.263776748965; ORT CPU.mean_us=522.3068096514908; ORT GPU.mean_us=4059.134772951802; TensorRT.mean_us=2777.38681 | `benchmark_stats_v2.py` | RTX 3050 | 20/framework live | see verify_claims HIST+sessions |
| `streaming_throughput.json` | 4389 | 2026-06-19T12:25:05 | TRACKED | CURRENT 25,899 | max_throughput={'gpu_batched': 25898.678852122866, 'gpu_single': 476.9607054701815, 'cpu_single': 513.9733919137194}; gpu_batched=25898.678852122866 | `benchmark_streaming.py` | RTX 3050 |  | no/single-file |
| `tensorrt_native.json` | 246 | 2026-06-24T01:24:49 | TRACKED | LEGACY single-point; ranges supersede | tensorrt_us=3403.3 | `benchmark_tensorrt_native.py` | RTX 3050 |  | no/single-file |
| `toniot_clean_comparison.json` | 207 | 2026-06-27T01:01:58 | TRACKED | CURRENT 0.9526 / gap 3.25 | cnn_bilstm_clean_f1=0.9526; rf_clean_f1=0.9851; gap_percent=3.25 | `compare after clean retrain` | local |  | no/single-file |
| `toniot_clean_retrain.json` | 65 | 2026-06-27T01:01:53 | TRACKED | CURRENT | macro_f1=0.9526283476667281 | `train_toniot_clean.py` | local |  | no/single-file |
| `toniot_multi_eval.json` | 447 | 2026-06-19T12:14:12 | TRACKED | CURRENT multi-granularity |  | `evaluate_toniot_multi.py` | local |  | no/single-file |
| `torch_compile_native.json` | 182 | 2026-06-24T01:22:30 | TRACKED | LEGACY single-point |  | `benchmark_torch_compile_native.py` | RTX 3050 |  | no/single-file |
| `training_history.json` | 6599 | 2026-06-14T00:04:12 | TRACKED | LEGACY training curves |  | `train.py V2/V3 era` | local |  | no/single-file |
| `training_history_toniot.json` | 4525 | 2026-06-17T10:39:29 | TRACKED | LEGACY |  | `train_toniot.py` | local |  | no/single-file |
| `twostage.json` | 70 | 2026-06-26T00:43:42 | TRACKED | SUPERSEDED 0.9639 by twostage_botiot 0.9790 | macro_f1=0.9638892562021631 | `early two-stage` | local |  | no/single-file |
| `twostage_botiot.json` | 280 | 2026-07-02T03:09:03 | GITIGNORED_ON_DISK | CURRENT champion metrics; GITIGNORED | macro_f1=0.9789974453814448; best_val_f1=0.9780220777564075 | `train_twostage.py` | local |  | no/single-file |

## Detailed key files

### twostage_botiot.json (CHAMPION metrics) — GITIGNORED ON DISK
```json
{
  "checkpoint": "model/best_model_botiot_distill_a0.6_T10.0_focal2.pth",
  "focal_gamma": 2.0,
  "epochs_requested": 10,
  "lr": 0.0001,
  "macro_f1": 0.9789974453814448,
  "weighted_f1": 0.982282086905235,
  "accuracy": 0.9822721666064699,
  "best_val_f1": 0.9780220777564075
}
```

### rf_baseline_processed.json — GITIGNORED ON DISK
```json
{
  "n_estimators": 200,
  "random_state": 42,
  "data_source": "data/processed/*.npy (preprocessing/preprocess_v2.py: undersample -> SMOTE -> scale)",
  "train_shape": [
    268627,
    10
  ],
  "val_shape": [
    293482,
    10
  ],
  "test_shape": [
    733705,
    10
  ],
  "val_macro_f1": 0.990308530511674,
  "val_weighted_f1": 0.990401623082919,
  "test_macro_f1": 0.9863877607840802,
  "test_weighted_f1": 0.9906660893970239,
  "test_accuracy": 0.9906597338167247
}
```

### dicc_v100_summary.txt (LEGACY)
```
DICC V100S-PCIE-32GB Benchmark Results (June 21, 2026)
Job ID: 363046

Block 1 FP32:     10.0114 us  (validation PASSED)
Block 2 FP32:     29.1387 us  (validation PASSED)
Block 3 FP32:    878.909 us   (validation PASSED)
Block 3 Graphs:  773.274 us   (validation PASSED)
Block 3 FP16:    511.852 us   (validation PASSED)
Block 4 FP32:      8.298 us   (validation PASSED)
Block 4 FP16:      5.243 us   (validation PASSED)
Pipeline chained (1+2+4): 38.812 us
Pipeline total (with B3 FP16): 550.664 us
```

### dicc_a100_summary.txt (LEGACY)
```
DICC A100-SXM4-80GB Benchmark Results (June 21, 2026)
Job ID: 363047

Block 1 FP32:     12.032 us   (validation PASSED)
Block 2 FP32:     34.734 us   (validation PASSED)
Block 3 FP32:   1130.07 us    (validation PASSED)
Block 3 Graphs: 1117.84 us    (validation PASSED)
Block 3 FP16:    548.368 us   (validation PASSED)
Block 4 FP32:     10.195 us   (validation PASSED)
Block 4 FP16:      6.617 us   (validation PASSED)
Pipeline chained (1+2+4): 43.676 us
Pipeline total (with B3 FP16): 592.044 us
```

### cuda_kernel_stats_rtx3050.json (LIVE session on disk) — GITIGNORED
- n_trials=100
- derived_pipeline_total_us=593.8892259999999
- B1 mean=57.334
- B2 mean=117.793 (high CV — outlier max 3057)
- B3 no_graphs mean=1020.012
- B3 with_graphs mean=732.713
- B3 fp16 mean=531.585
- B3 naive mean=4783.022
- B4 mean=18.723
- pipeline b124_chained mean=62.304

**Note:** File stores **one session**. Multi-session ranges in README come from combining this live file with **hardcoded** HIST/SESSION3A/EXTRA arrays in `scripts/verify_claims.py` (not from a multi-session JSON array on disk).

### statistical_significance_v2.json (LIVE framework session)
- Eager PyTorch: mean=2077.205 std=153.92111841756406 n=20
- torch.compile: mean=1776.264 std=192.61333203656395 n=20
- ORT CPU: mean=522.307 std=35.46520682175754 n=20
- ORT GPU: mean=4059.135 std=271.4513652418115 n=20
- TensorRT: mean=2777.387 std=74.94349144822948 n=20
- Custom CUDA FP16: mean=652.422 std=50.0771707282943 n=100

Custom CUDA FP16 note (from JSON): Derived from benchmarks/results/cuda_kernel_stats_rtx3050.json: fused_pipeline b124_chained_us + fused_block3_fp16 latency_us (independent binaries, summed mean/variance). Was a bare constant with no variance until 2026-07-01.

### llm_explainability.json
- overhead_p99_us = 16.599169557594042
- dispatch_p99_us = 16.717170128686167
- trials = 5000
- llm_generation_mean_ms = 7400.471511333308

### numerical_fidelity.json
- model_version_note: V3 (attention, using V2 last-timestep for CUDA)
- export full max_abs_error: 0.0
- prediction agreement: {'agree': 10, 'total': 10, 'rate': 1.0}
- cuda_selfcheck: [('fused_block1', True, 0.001), ('fused_block2', True, 0.001), ('fused_block3', True, 0.01), ('fused_block3_fp16', True, 0.05), ('fused_block3_naive', True, 0.01), ('fused_block4', True, 0.001)]

### Stale F1 labels inside resource JSONs
- cuml_rf_resources.cnn_bilstm_f1 = 0.9639 (**STALE** vs 0.9790)
- cuml_rf_native.local_laptop.custom_cuda_f1 = 0.9601 (**STALE** 0.9601)