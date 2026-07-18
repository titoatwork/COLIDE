# 01 — Repository Map

**HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`  
**Top-level (meaningful):** `README.md`, `HANDOFF.md`, `AGENTS.md`, `CLAUDE.md`, `DAILY_LOG.md`, `environment.md`, `Dockerfile`, `benchmark.sh`, `requirements.txt`, `model/`, `inference/`, `scripts/`, `benchmarks/`, `docs/`, `dicc_scripts/`, `config/`, `preprocessing/`, `llm_integration/`, `data/`, `notebooks/`

## Counts (this audit)

| Category | Count |
|----------|------:|
| `benchmarks/results/*` files | 54 |
| `scripts/*.py` | 44 |
| `inference/kernels/*.cu` | 8 source (+ compiled binaries) |
| `dicc_scripts` files | 18 |
| Tracked result files (`git ls-files`) | 37 |
| On-disk untracked/gitignored results | 17 |

## Model architecture map

| Module | Path | Role | Production? |
|--------|------|------|-------------|
| V2 CNN-BiLSTM (last-timestep) | `model/cnn_bilstm.py` | Base per-flow architecture; CUDA contract matches this | Weights used via export path; V3 wraps |
| V1 windowed | `model/cnn_bilstm_v1_windowed.py` | Abandoned Phase 0 approach | **LEGACY** |
| V3 attention | `model/cnn_bilstm_v3_attention.py` | MultiheadAttention + LayerNorm + GAP; production training class | **YES for training/frameworks** |
| ONNX exports | `model/colide_model.onnx`, `model/model.onnx` | Framework export artifacts | Secondary |

**Production checkpoint:** `model/best_model_botiot_twostage.pth`  
- md5 **80a90f7cc210276300eaa90173a5a385** (confirmed this audit; matches BACKUP_0.9790_s5)  
- macro-F1 **0.978997… → reported 0.9790** from `twostage_botiot.json`  
- Stage-1 KD source: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` (md5 `4b7accd4e0a42905ac1b51b80302a85c`)  
- Round-1 backup: `best_model_botiot_twostage_BACKUP_0.9639.pth` (md5 `d9326dd9…`)

**Note:** All `model/*.pth` appear tracked in git (`git ls-files model/*.pth` lists them). `model/weights/` is gitignored; `model/weights_bin/` has 36 files + `validation_metadata.json` (export path).

### Parameter counts (from README / paper tables)

| Model | Params | Source |
|-------|-------:|--------|
| CNN-BiLSTM V3 | 530,181 | README detection table |
| MLP | 400,901 | README MLP ablation |

## CUDA kernels (`inference/kernels/`)

| Source | Binary purpose | Measures |
|--------|----------------|----------|
| `fused_block1.cu` | Block 1: proj+Conv1D+BN+ReLU | Standalone B1 latency + CPU validation |
| `fused_block2.cu` | Block 2: Conv+BN+ReLU+MaxPool | Standalone B2 |
| `fused_block3.cu` | BiLSTM FP32 transposed W_hh; optional CUDA Graphs | B3 no-graphs / with-graphs |
| `fused_block3_fp16.cu` | BiLSTM FP16 half2 FMA | B3 FP16 (headline Block 3) |
| `fused_block3_naive.cu` | Naive 1-thread/hidden; race-fixed double-buffer | Progression step 0 |
| `fused_block4.cu` | Dense 128→64 ReLU→5 | B4 FP32 |
| `fused_block4_fp16.cu` | Dense FP16 half2 | B4 FP16 (launch-overhead study) |
| `fused_pipeline.cu` | Intended full chain; **times B1+B2+B4 chained**; B3 often additive | Derived pipeline total |

**Option A issues (code-level):**

1. Production model is V3 (attention/LN/GAP); CUDA implements V2 last-timestep BiLSTM contract (`numerical_fidelity.json`: *"V3 (attention, using V2 last-timestep for CUDA)"*).  
2. `fused_pipeline.cu` does not truly chain live B3 output into B4 in timed path (DESIGN_PLAN §5.1).  
3. Derived total = `b124_chained_us + fused_block3_fp16` (see `cuda_kernel_stats` + `statistical_significance_v2` Custom CUDA note).

## Export / weights

| Path | Role |
|------|------|
| `model/weights_bin/` | Binary weight + reference tensors for CUDA validation |
| `model/weights_bin/validation_metadata.json` | Export metadata |
| `model/weights/` | npy fp16/fp32 dumps including attention (V3) — gitignored |
| Scripts | `scripts/validate_real_weights.py`, `scripts/validate_weights.py`, re-export fixed in `c27ac2a` |

## Config / data

| Path | Role |
|------|------|
| `config/config.yaml` | Model/training config |
| `data/processed/*.npy` | Preprocessed splits (gitignored; required for RF/train) |
| `preprocessing/` | V2 per-flow pipeline |

## Docs surface (claim-relevant)

| Doc | Authority |
|-----|-----------|
| `docs/DESIGN_PLAN.md` | Option A lock, solid/weak/invalid |
| `docs/FINAL_PLAN.md` | P0–P5 execution + hard numbers-match gate |
| `docs/PROF_POR_3DAY.md` | Prof pack tables (to fill after DICC) |
| `docs/PROF_POR_STATUS_REPORT.md` | **NON-AUTHORITATIVE mistaken draft** |
| `docs/paper_text_blocks.md` | Manuscript text blocks |
| `docs/DICC_RUNBOOK.md` | Superseded by `dicc_scripts/README.md` |
| `HANDOFF.md` | Session state / lifecycle |
| `colide_review_brief.md` (home `~/`) | External review brief; outside repo |

## Environment (from `environment.md`)

- Local: WSL2, RTX 3050 Laptop 6GB, SM 8.6  
- Known: session-to-session latency drift  
- Official multi-day cluster: UM DICC only  
- Rostam: tooling trial only  

## DICC expected SUCCESS layout (from `dicc_scripts/README.md`)

```
benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/
  manifest.json  environment.txt  kernel_SHA256SUMS
  cuda_kernel_stats.json  pytorch_gpu_stats.json
  raw/  logs/  exit_status  SUCCESS
```

**On laptop today:** `benchmarks/results/dicc/` → **ABSENT**.  
**Legacy only:** `dicc_v100_summary.txt`, `dicc_a100_summary.txt` (June 21, 2026; jobs 363046/363047).
