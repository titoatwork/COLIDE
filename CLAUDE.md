# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

COLIDE is an academic research codebase (FGCS journal submission) for an IoT network intrusion detection
system. It has two halves that get benchmarked against each other:

1. A CNN-BiLSTM classifier trained in PyTorch (`model/`, `scripts/train*.py`) on flow-level network
   traffic features from the **BoT-IoT** and **ToN-IoT** datasets.
2. Hand-written CUDA C++ inference kernels (`inference/kernels/*.cu`) that reimplement that same
   trained model's forward pass to beat PyTorch eager, `torch.compile`, TensorRT, and ONNX Runtime on
   latency — plus an async, on-device, air-gapped LLM explainability layer (TinyLlama, 4-bit quantized).

The paper's contribution is systems/performance engineering, not model novelty — see README.md for the
full results tables (framework speedups, cross-GPU profiling, KD accuracy recovery). Read README.md
before changing benchmark methodology or reporting new numbers; the numbers there are treated as
citable, verified results, not scratch notes.

## Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.   # required for all scripts/ invocations, they import model/ and preprocessing/ as top-level packages
```

GPU/CUDA details for the dev machine and DICC cluster are tracked in `environment.md` — check it before
assuming a CUDA arch or driver version. Reproducibility: every script seeds `numpy`/`torch`/`random`
with `seed=42` (also the global default in `config/config.yaml`).

## Common commands

There is no test suite, linter, or build system config (no pytest/flake8/package.json) — this is a
research repo driven by standalone scripts. "Running it" means running the relevant script directly.

**Train the main model (BoT-IoT):**
```bash
PYTHONPATH=. python scripts/train.py
PYTHONPATH=. python scripts/train_toniot_clean.py      # ToN-IoT, 26-feature clean variant
PYTHONPATH=. python scripts/train_distill.py            # + knowledge distillation from RF teacher
PYTHONPATH=. python scripts/train_twostage.py           # KD/focal pretrain + real-data fine-tune (best BoT-IoT result)
PYTHONPATH=. python scripts/train_mlp_distill.py         # MLP student ablation (see README MLP Ablation)
```
Other `scripts/train_*.py` variants (ensemble, optimized, v1_windowed) are earlier/ablation experiments
kept for the paper's ablation section — check README.md's results tables before assuming one is current.

**Preprocess data:**
```bash
PYTHONPATH=. python -m preprocessing.preprocess_v2      # BoT-IoT: per-flow, no windowing (current)
PYTHONPATH=. python scripts/preprocess_toniot.py         # ToN-IoT
```
`preprocessing/preprocess_v1_windowed.py` is deliberately superseded — see "Architecture notes" below
for why windowing was abandoned. Don't resurrect it without understanding that history.

**Compile CUDA kernels** (each `.cu` file is standalone, no Makefile/CMake):
```bash
nvcc -arch=sm_86 -o inference/kernels/fused_block1 inference/kernels/fused_block1.cu   # RTX 30xx/A100 (Ampere)
nvcc -arch=sm_70 -o inference/kernels/fused_block1 inference/kernels/fused_block1.cu   # V100 (Volta)
```
Swap `fused_block1` for `fused_block2`/`fused_block3`/`fused_block3_fp16`/`fused_block4`/`fused_pipeline`
(and `fused_block3_naive` for the pre-optimization baseline). `Dockerfile` builds all of them for
`sm_86` by default (`--build-arg CUDA_ARCH=sm_89` to override). Each compiled binary runs its own
benchmark when executed directly (see `benchmark.sh`).

**Run the full benchmark suite:**
```bash
bash benchmark.sh            # everything, including streaming + LLM benchmarks
bash benchmark.sh --quick    # skips benchmark_streaming.py and llm_explainability.py
```
This assumes the CUDA kernels are already compiled in `inference/kernels/`. Individual benchmark
scripts in `scripts/benchmark_*.py` can be run standalone for iterating on one measurement at a time
(e.g. `python scripts/benchmark_stats.py` for the statistical-significance trial runs).

**DICC cluster (SLURM):** `dicc_scripts/01_setup.sh` clones the repo, compiles kernels for both `sm_70`
(V100) and `sm_80` (A100) into `inference/kernels/v100/` and `inference/kernels/a100/`, then
`02_benchmark_v100.sh`/`03_benchmark_a100.sh`/`04_nsight_profile.sh` are submitted via `sbatch`.

## Architecture notes

**Data flow is per-flow, not sequential**, despite the model containing an LSTM. Early on (see
`DAILY_LOG.md`, Week 5) a windowed/sequence pipeline (`preprocessing/preprocess_v1_windowed.py`,
`model/cnn_bilstm_v1_windowed.py`) was tried and abandoned: windowing across SMOTE-resampled data
produced impossible sequences and collapsed the model to predicting 2 of 5 classes. An RF baseline on
un-windowed per-flow data hit 0.9768 macro-F1, proving the BoT-IoT 10-best features are pre-aggregated
flow statistics with no real inter-flow temporal dependency. The MLP ablation in the current README
reaches this same conclusion quantitatively (MLP without any recurrence nearly matches CNN-BiLSTM
accuracy). **The BiLSTM is kept anyway** — it exists specifically to stress-test the CUDA/compiler work
(dynamic recurrent control flow is what breaks `torch.compile`'s CUDA graph capture; see
`docs/torch_compile_crash_trace.txt`), not because it's needed for accuracy.

**Model input shape is `(batch, input_features)`** — a flat feature vector, reshaped internally into a
pseudo-2-channel sequence (`reshape: [2, 32]` in config) purely so the CNN/BiLSTM stack has something to
operate on. `model/cnn_bilstm.py` is the reference PyTorch implementation; every architectural constant
(filter counts, LSTM hidden sizes, dense sizes) is read from `config/config.yaml`'s `model:` block — the
CUDA kernels hardcode these same dimensions, so if you change `config.yaml`'s model shape you must also
update the kernel source. `model/cnn_bilstm_v3_attention.py` is the actual class used by current training
scripts (imported as `CNNBiLSTM` alias) — it adds a multi-head attention block over the BiLSTM output;
`cnn_bilstm.py`/`cnn_bilstm_v1_windowed.py` are earlier versions kept for the paper's version-history/
ablation narrative, not dead code to delete.

**The four CUDA kernels are 1:1 replacements for named blocks of the PyTorch forward pass**, and the
block numbering is used consistently across code, benchmarks, and README:
- Block 1 = input projection + reshape + Conv1D + BatchNorm + ReLU (`fused_block1.cu`)
- Block 2 = Conv1D + BatchNorm + ReLU + MaxPool (`fused_block2.cu`)
- Block 3 = 2-layer BiLSTM (`fused_block3.cu` FP32, `fused_block3_fp16.cu` uses `__hfma2` half2-packed
  gates — this is the kernel with the documented 9.48x optimization progression from naive to fp16;
  `fused_block3_naive.cu` is that progression's starting point, kept for the paper's ablation table, not
  a discard candidate)
- Block 4 = Dense + ReLU + Dense output head (`fused_block4.cu`)
`fused_pipeline.cu` chains all four blocks. Weights are exported from the trained PyTorch model via
`CNNBiLSTM.export_weights()` in `model/cnn_bilstm.py` (dumps every parameter as both `_fp32.npy` and
`_fp16.npy` into `model/weights/`) and loaded by the `.cu` files at kernel-binary startup — if you retrain
the model, re-export weights before re-benchmarking the kernels, or the kernels will be profiling stale
weights.

**LLM explainability is intentionally decoupled from the detection path.** `scripts/llm_explainability.py`
implements a producer/consumer split: the detection thread pushes `Alert` objects into a bounded
`RingBuffer` (drops oldest on overflow — never blocks detection) and a separate LLM thread consumes from
it to run TinyLlama-1.1B generation. `benchmark_dispatch_overhead()` in that script measures the real
p50/p95/p99 of the classify+construct+push code path over 5,000 trials — this is what the README's
"16.60 us p99 dispatch overhead" figure comes from (an earlier "5.19 us" figure in this codebase was a
hardcoded placeholder never backed by a real percentile computation; fixed 2026-07-01 — see
`benchmarks/results/llm_explainability.json` for the raw numbers). The dispatch cost is measured
separately from LLM generation time (~7.4s/alert), which lives on a completely different timescale.
`llm_integration/` (the package) is currently a
stub (empty `__init__.py`); the working implementation lives in the `scripts/llm_explainability.py`
script — don't assume `llm_integration/` has logic just because the package exists.

**Two independent dataset pipelines** exist side by side and are not interchangeable: BoT-IoT
(`data/raw/`, `data/processed/`, `config/config.yaml`, 10 features, 5 classes) and ToN-IoT
(`data/raw/toniot/`, `data/processed_toniot/`, `data/processed_toniot/config_toniot.yaml`, 13 or 26
features depending on clean/original variant, 10 classes). Scripts named `*_toniot*` target the second
pipeline; everything else defaults to BoT-IoT. Preprocessing for both follows the same
undersample → SMOTE → MinMax-scale → stratified split methodology (see `preprocessing/preprocess_v2.py`
docstring), but resampling targets and feature lists differ per dataset/config file — don't reuse one
dataset's config values for the other.

**Reported numbers are provenance-sensitive.** `README.md` and `docs/paper_text_blocks.md` contain the
finalized, submission-ready results (with p-values, CIs, trial counts) — treat any number there as
already verified against a specific script/config/hardware combination documented in `DAILY_LOG.md`.
When asked to update or regenerate results, reproduce with the same script rather than recomputing by
hand, and update README.md and paper_text_blocks.md together so they don't diverge.
