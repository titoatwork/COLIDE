# 01 — Full Codebase Inventory

**Repo root:** `/home/titoisalive/colide`  
**Inventory date:** 2026-07-19  
**Scope:** All project source/docs (excluding `.venv`, raw data blobs, `__pycache__`, `.git`)

---

## 1. Top-level layout

| Path | Role |
|------|------|
| `model/` | Neural architectures + checkpoints + weight export |
| `scripts/` | Train, distill, RF, benchmarks, fidelity, verify_claims |
| `inference/kernels/` | Custom CUDA C++ (blocks 1–4, FP16, naive, fused_pipeline) |
| `inference/include/`, `inference/tests/` | Headers / kernel tests |
| `preprocessing/` | BoT-IoT preprocess v1 (windowed, legacy) / v2 (per-flow) |
| `llm_integration/` | Package stub; main LLM logic in `scripts/llm_explainability.py` |
| `config/config.yaml` | Model + train + eval + benchmark defaults |
| `dicc_scripts/` | UM/Rostam campaign: setup, submit, run_campaign, compare helpers |
| `benchmarks/results/` | JSON/txt result artifacts (**mostly gitignored**) |
| `benchmarks/plots/` | Figures |
| `docs/` | Plans, audit pack, interim report, Prof feedback, **this pack** |
| `data/` | raw / processed / splits (local; gitignored patterns) |
| `notebooks/` | Exploratory |
| `requirements.txt`, `Dockerfile`, `benchmark.sh` | Env / entry |
| `HANDOFF.md`, `CLAUDE.md`, `AGENTS.md`, `README.md` | Ops + claims surface |
| `DAILY_LOG.md`, `environment.md` | Historical notes |

Approx. tracked project sources: **~50+ Python**, **8 CUDA**, **13 shell**, **30 MD**, **50+ results JSON on disk**.

---

## 2. Models (`model/`)

| File | Role | Paper relevance |
|------|------|-----------------|
| `cnn_bilstm_v3_attention.py` | **Production architecture**: projection, CNN, BiLSTM, multi-head attention, residual LN, GAP, dense head | Final student model class |
| `cnn_bilstm.py` | Earlier / simpler variant | Ablation ancestry |
| `cnn_bilstm_v1_windowed.py` | Windowed pipeline model | LEGACY path (abandoned for main) |
| `best_model_botiot_twostage.pth` | **Current champion** 0.9790 | Baseline reference |
| `best_model_botiot_twostage_BACKUP_0.9790_s5.pth` | Backup same era | Safety |
| `best_model_botiot_twostage_BACKUP_0.9639.pth` | Superseded | Do not quote as current |
| `best_model_botiot_distill_*.pth` | KD sweep checkpoints | Stage-1 lineage |
| `best_model_botiot_ensemble.pth` | Ensemble teacher student | Not champion |
| `best_model_botiot_mlp_*.pth` | MLP ablation | Ablation table |
| `best_model_toniot*.pth` | ToN models | Secondary dataset |
| `weights/`, `weights_bin/` | Exported weights for CUDA | Fidelity / kernels |

**Critical architecture fact:** V3 has **attention + LN + GAP**. Custom CUDA implements **subset** (last-timestep BiLSTM contract). Option A forbids full-pipeline speedup claims.

---

## 3. Training & evaluation scripts (`scripts/`)

### 3.1 Detection / KD / RF

| Script | What it does | Prof coverage |
|--------|--------------|---------------|
| `train_distill.py` | RF teacher + CNN-BiLSTM KD; focal; SMOTE train path; α/T CLI | Partial KD/HPO (manual) |
| `train_twostage.py` | Fine-tune distilled ckpt on **real** (no SMOTE) data + focal | Current champion path |
| `sweep_distill_botiot.py` | Small fixed α/T grid via subprocess | **Not** Optuna/Bayesian |
| `train_ensemble_distill.py` | RF+XGB+LGBM teacher distill | Exists; student not champion |
| `train_mlp_distill.py` | MLP student | Ablation |
| `train_optimized.py` | Earlier optimized variant | Historical |
| `train.py` / `train_v1_windowed.py` | Legacy | Do not use for final |
| `rf_baseline.py` | RF (may not match processed path) | Prefer processed |
| `rf_baseline_processed.py` | **Published RF 0.9864** path | MUST keep protocol |
| `rf_teacher_strengthen.py` | RF strengthen → ~0.9885 diagnostic | Teacher research |
| `rf_baseline_toniot.py` | ToN RF | Secondary |
| `train_toniot*.py`, `train_distill_toniot*.py` | ToN training | Secondary |
| `evaluate_toniot_multi.py` | Multi-eval modes | Secondary metrics |
| `preprocess_toniot.py` | ToN preprocess | Secondary |
| `compare_datasets.py` | Dataset comparison | Support |

### 3.2 Systems / CUDA / frameworks

| Script | Role |
|--------|------|
| `benchmark_cuda_kernels_stats.py` | Multi-trial CUDA kernel stats (n, mean, CI) |
| `benchmark_pytorch_block3_stats.py` | Block 3 cuDNN-style baseline |
| `benchmark_pytorch_gpu_stats.py` | Same-GPU PT stats; **full_pipeline valid=false** flag |
| `benchmark_stats_v2.py` | Framework comparison multi-session style |
| `benchmark_stats.py` | Earlier stats |
| `benchmark_pipeline.py`, `benchmark_blocks.py` | Block timing |
| `benchmark_ort.py`, `benchmark_tensorrt_native.py`, `benchmark_torch_compile_native.py` | Frameworks |
| `benchmark_energy.py`, `benchmark_a100_energy.py` | Energy |
| `benchmark_streaming.py` | Throughput |
| `benchmark_batch.py`, `benchmark_mlp_latency.py` | Batch / MLP |
| `benchmark_cuml_rf_native.py` | cuML RF thr/VRAM/energy |
| `compare_dicc_sessions.py` | Day1 vs Day2 compare (Welch, Cohen’s d) |
| `numerical_fidelity.py` | Export + CUDA self-check |
| `validate_weights.py`, `validate_real_weights.py` | Weight export validation |
| `verify_claims.py` | Number presence vs JSON (not Option A semantics) |

### 3.3 LLM / other

| Script | Role | Gap vs Prof |
|--------|------|-------------|
| `llm_explainability.py` | Async ring buffer + TinyLlama; **dispatch p99** | No faithfulness/SHAP/LIME/analyst study |
| `alert_aggregation.py` | Alert grouping | Support |
| `ablation_study.py` | Aggregates results tables (not full scientific ablation ladder) | Insufficient alone |

---

## 4. CUDA kernels (`inference/kernels/`)

| Kernel | Maps to | Notes |
|--------|---------|-------|
| `fused_block1.cu` | Projection + reshape + Conv + BN + ReLU | Implemented |
| `fused_block2.cu` | CNN path continued | Implemented |
| `fused_block3.cu` / `_fp16.cu` / `_naive.cu` | BiLSTM (last-timestep contract) | Flagship; race-fixed naive |
| `fused_block4.cu` / `_fp16.cu` | Dense head path | Implemented |
| `fused_pipeline.cu` | B1+B2+B4 chain; B3 additive | **Not** full V3; not true B3→B4 device chain |

Compiled binaries present for local + some `v100_*` names (historical).

---

## 5. Preprocessing

| File | Role |
|------|------|
| `preprocessing/preprocess_v2.py` | Per-flow BoT-IoT; SMOTE/undersample; class_weights.npy; MinMax | **Main scientific data path narrative** |
| `preprocessing/preprocess_v1_windowed.py` | Windowed LEGACY | Abandoned for main claim |
| `scripts/preprocess_toniot.py` | ToN clean features | Secondary |

**Risk:** `train_distill.py` / `train_twostage.py` re-implement load paths from CSV rather than only reading `data/processed/*.npy` — **protocol unification is required** for Prof “freeze split” Phase 1.

---

## 6. Config

`config/config.yaml`:

- seed 42  
- 10 features, 5 classes  
- model: proj 64, reshape [2,32], CNN 64/128 k=3, BiLSTM 128/64, dense 64, dropout 0.3, attn heads 4  
- train: batch 128, epochs 50, lr 0.001, cosine, patience 10, **use_class_weights: false**  
- benchmark: warmup 3, runs 10, batch sizes 1/32/128/256  

**Gap:** No Optuna search space file; class weights computed in preprocess but training often ignores them.

---

## 7. DICC tooling (`dicc_scripts/`)

| Asset | Role |
|-------|------|
| `run_campaign.sh` | Day1/Day2/full entry |
| `01_setup.sh` … `05_run_all.sh` | Setup / per-GPU / Nsight |
| `submit_session.sh`, `job_benchmark.sh`, `compile_on_gpu.sh` | SLURM |
| `lib/common.sh`, `lib/run_benchmark.sh` | Shared |
| `profiles/`, `site.env*.example` | Site portability |
| `validate/local_validate.sh` | Offline validate |
| `README.md` | Operator docs |

**Gap:** Results tree `benchmarks/results/dicc/` **ABSENT** on laptop — tooling ready, campaign not completed for paper.

---

## 8. Results artifacts (`benchmarks/results/`)

On disk (~52 JSON), including:

- Accuracy: `twostage_botiot.json`, `rf_baseline_processed.json`, distill_*, ensemble, mlp, toniot_*  
- Systems: `cuda_kernel_stats_rtx3050.json`, `pytorch_block3_stats_rtx3050.json`, `statistical_significance_v2.json`, energy, streaming, cuml  
- Fidelity: `numerical_fidelity.json`  
- Legacy DICC: `dicc_v100_summary.txt`, `dicc_a100_summary.txt`  
- **Missing:** multi-day `dicc/` SUCCESS tree  

`.gitignore` ignores `benchmarks/results/` → **repro packaging gap** Prof already noted.

---

## 9. Docs already present (do not duplicate blindly)

| Doc | Use |
|-----|-----|
| `docs/audit/*` | Forensic evidence for interim numbers |
| `docs/STATUS_REPORT_DRAFT.md` / Word interim | Sent to Prof |
| `docs/feedback1.docx` | His requirements |
| `docs/PROF_FEEDBACK_ROADMAP.md` | Short phases |
| `docs/execution_plan/*` | **This deep pack** |

---

## 10. What the codebase is strong at today

- Real CUDA kernels + statistical harness + claim verifier  
- KD + two-stage path to 0.9790 with bit-repro of stage-1 winner  
- Honest RF baseline scripted  
- Multi-session laptop latency discipline (ranges)  
- DICC automation scripts  
- LLM **dispatch** prototype  

## 11. What the codebase is weak at vs Prof bar

- No systematic Optuna/Bayesian HPO  
- No complete fair baseline suite (LR/SVM/XGB/Transformer equal protocol)  
- No full ablation ladder with systems metrics per cell  
- No class-threshold calibration module  
- No logit-adjustment / CB-focal / SupCon library  
- XAI = dispatch only  
- Split/load logic duplicated across scripts (repro risk)  
- No multi-day DICC artifacts  
- No Pareto analysis script  
- Architecture changes (multi-scale CNN, new attention) not implemented as paper method package  
- 5-run training stability not automated  

---

## 12. Dependency map (simplified)

```text
preprocess_v2 / CSV loaders
    → train_distill / ensemble / mlp
        → train_twostage
            → champion .pth
                → validate_weights / numerical_fidelity
                    → CUDA kernels + benchmark_* 
                        → verify_claims / README
llm_explainability (parallel systems story)
dicc_scripts (cluster replication of benchmark_*)
```

---

*Next: `02_FEEDBACK_DEEP_ANALYSIS.md`*
