# 03 — Producer / Consumer Graph

**HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`

## Legend

- **→** produces artifact  
- **⇒** consumed by claim surface  
- **DEAD / EXPERIMENTAL / ABLATION** labels are auditor judgments

---

## A. Accuracy / training track

```
data/raw (gitignored)
  → preprocessing/preprocess_v2.py
  → data/processed/*.npy (gitignored)

scripts/train.py
  → model/best_model.pth (early V3 path)
  → benchmarks/results/training_history.json
  ⇒ SUPERSEDED by distill/two-stage path for production accuracy

scripts/train_v1_windowed.py
  → LEGACY V1 windowed; archived approach
  → DEAD for production

scripts/train_distill.py + scripts/sweep_distill_botiot.py
  → model/best_model_botiot_distill_*.pth
  → benchmarks/results/distill_botiot*.json
  ⇒ README KD sweep table; paper_text_blocks KD table
  ⇒ Stage-1 winner: distill_botiot_a0.6_T10.0_focal2.json (0.9763)
  ⇒ bit-repro: distill_botiot_a0.6_T10.0_focal2_repro.json (identical macros)

scripts/train_twostage.py
  ← checkpoint best_model_botiot_distill_a0.6_T10.0_focal2.pth
  → model/best_model_botiot_twostage.pth  [CHAMPION md5 80a90f7c...]
  → benchmarks/results/twostage_botiot.json  [GITIGNORED ON DISK]
  ⇒ README 0.9790; paper_text_blocks; HANDOFF; FINAL_PLAN; verify_claims

scripts/train_twostage.py (round 1 historical)
  → twostage.json (0.9639) SUPERSEDED
  → BACKUP_0.9639.pth

scripts/train_mlp_distill.py / MLP two-stage
  → ablation_mlp.json (0.9624)
  → mlp_twostage.json (0.9542)
  → model/best_model_botiot_mlp_*.pth
  ⇒ MLP ablation table

scripts/train_ensemble_distill.py
  → ensemble_distill.json (0.9529)
  → best_model_botiot_ensemble.pth
  ⇒ not champion (S5 Val-F1 fix still not beat 0.9790)

scripts/train_optimized.py
  → optimized_botiot_focal_only.json (catastrophic collapse ~0.001)
  ⇒ paper KD table "Focal only" row

scripts/rf_baseline_processed.py
  → rf_baseline_processed.json (test_macro_f1 0.986387… → 0.9864)
  ⇒ published RF bar; gap 0.74%
  ⇒ verify_claims rf_baseline_processed_test_f1

scripts/rf_baseline.py
  → independent resampling from raw CSV → ~0.9768 (README footnote)
  ⇒ NOT the published apples-to-apples bar

scripts/rf_teacher_strengthen.py (S5)
  → rf_teacher_strengthen.json (best balanced 0.9885)
  ⇒ decision: keep published bar 0.9864 (HANDOFF/DESIGN_PLAN)

scripts/train_toniot*.py / train_toniot_clean.py
  → distill_toniot.json, distill_toniot_v2.json (0.8254)
  → toniot_clean_retrain.json (0.9526)
  → toniot_clean_comparison.json (gap 3.25%)
  → toniot_multi_eval.json
  ⇒ ToN-IoT tables
```

## B. CUDA / latency track

```
model/best_model_botiot_twostage.pth
  → weight export → model/weights_bin/*.bin + validation_metadata.json
  → re-export fix c27ac2a after stale pre-distill weights

nvcc compile inference/kernels/*.cu
  → fused_block{1,2,3,3_fp16,3_naive,4,4_fp16}, fused_pipeline binaries

scripts/benchmark_cuda_kernels_stats.py
  → cuda_kernel_stats_rtx3050.json  [GITIGNORED; LIVE one session]
  ⇒ Block 3 ranges (with HIST hardcoding in verify_claims)
  ⇒ derived_pipeline_total_us

scripts/benchmark_pytorch_block3_stats.py
  → pytorch_block3_stats_rtx3050.json  [GITIGNORED]
  ⇒ 784 us cuDNN baseline; beats-cuDNN 1.30x–1.47x

scripts/benchmark_stats_v2.py
  → statistical_significance_v2.json  [TRACKED]
  ⇒ framework table means (LIVE session)
  ⇒ Custom CUDA derived from cuda_kernel_stats

scripts/verify_claims.py
  ← HIST_LATENCY, SESSION3A_LATENCY, CUSTOM_CUDA_EXTRA_TOTALS hardcoded
  ← live statistical_significance_v2.json + cuda_kernel_stats
  ⇒ enforces README range strings; REGRESSION_GUARDS

scripts/benchmark_pipeline.py
  → pipeline_benchmark.json (+ GPU-tagged copy)
  ⇒ early per-block points (LEGACY single-run); B3 601.4 historical

scripts/benchmark_tensorrt_native.py → tensorrt_native.json (LEGACY points)
scripts/benchmark_torch_compile_native.py → torch_compile_native.json (LEGACY)
scripts/benchmark_stats.py → statistical_confidence.json (LEGACY early CIs)
scripts/benchmark_streaming.py → streaming_throughput.json (25,899 CURRENT)
scripts/benchmark_energy.py → energy_efficiency.json (0.79 mJ CURRENT)
scripts/benchmark_a100_energy.py → a100_energy.json (1.089 mJ; throughput derived)
scripts/benchmark_mlp_latency.py → mlp_latency.json (175 us)
scripts/benchmark_cuml_rf_native.py → cuml_rf_native.json
(A100 resources) → cuml_rf_resources.json
scripts/llm_explainability.py → llm_explainability.json (16.60 p99; fixed 09b509b)
scripts/numerical_fidelity.py → numerical_fidelity.json (S7)
scripts/validate_real_weights.py → real_weight_validation.json
scripts/ablation_study.py → CONSUMES json (no longer hardcodes most numbers)
```

## C. Cluster campaign track

```
dicc_scripts/run_campaign.sh
  → 01_setup.sh (venv + compile)
  → submit_session.sh / job_benchmark.sh / lib/run_benchmark.sh
  → SHOULD write:
      benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/
        SUCCESS, manifest.json, environment.txt, kernel_SHA256SUMS,
        cuda_kernel_stats.json, pytorch_gpu_stats.json, raw/, logs/
  → PLANNED / NOT RUN on UM hardened stack (this laptop: ABSENT)

scripts/compare_dicc_sessions.py
  → accept/reject gate across Day1 vs Day2
  → PLANNED / no inputs present

Legacy:
  DICC jobs 363046/363047 (2026-06-21)
  → dicc_v100_summary.txt / dicc_a100_summary.txt
  → CUDA-only single-shot; no pytorch_gpu_stats; no multi-day SUCCESS dirs

Rostam Day 1:
  → TOOLING-ONLY (not paper-official); provisional B3 CUDA slower than PT
  → numbers recorded in DESIGN_PLAN §5.3 as approx; not in benchmarks/results/dicc/
```

## D. Claim surfaces (consumers)

| Surface | What it consumes | Gate |
|---------|------------------|------|
| README.md | ranges + accuracy + DICC legacy | verify_claims subset |
| docs/paper_text_blocks.md | same + prose | verify_claims subset |
| HANDOFF.md / FINAL_PLAN.md | frozen anchors | manual |
| docs/PROF_POR_3DAY.md | empty DICC cells until P1 | P2 gate |
| docs/PROF_POR_STATUS_REPORT.md | NON-AUTHORITATIVE draft | ignore for truth |
| scripts/verify_claims.py | 66 claims + regression guards | automated |

## E. Dead / experimental / ablation-only scripts

| Script | Label | Notes |
|--------|-------|-------|
| train_v1_windowed.py | DEAD/LEGACY | Phase 0 abandoned |
| train_optimized.py | ABLATION | focal-only collapse |
| train_ensemble_distill.py | EXPERIMENTAL | not champion |
| rf_teacher_strengthen.py | DIAGNOSTIC | bar not raised |
| benchmark_blocks.py | LEGACY harness | early block targets; loads best_model.pth |
| benchmark_batch.py / benchmark_ort.py | LEGACY | early framework |
| benchmark_stats.py | SUPERSEDED by v2 | early CIs |
| train_distill_toniot.py | SUPERSEDED by v2 | |
| docs/DICC_RUNBOOK.md | SUPERSEDED | by dicc_scripts/README |

## F. Graph summary (load-bearing path)

```
champion .pth (md5 80a90f7c)
   ├─ accuracy claims ← twostage_botiot.json + rf_baseline_processed.json
   ├─ CUDA weights_bin ← validate/fidelity
   ├─ framework latency ranges ← stats_v2 + HIST hardcode + cuda_kernel_stats
   └─ LLM dispatch ← llm_explainability.json
```
