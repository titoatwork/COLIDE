# COLIDE: CUDA-Optimized CNN-BiLSTM for IoT Intrusion Detection (On-Device Alert Dispatch Prototype)

[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](#license)

**Public academic research repo** (Academic Research License — not MIT).  
**Manuscript lead:** see [`docs/CHERAN_MANUSCRIPT_HANDOFF.md`](docs/CHERAN_MANUSCRIPT_HANDOFF.md).  
**Doc map:** [`docs/README.md`](docs/README.md) (current vs historical — stale files are **kept and labeled**, not deleted).

### Start here

| If you are… | Open |
|-------------|------|
| Checking if pre-manuscript is done | [`docs/PRE_MANUSCRIPT_INDEX.md`](docs/PRE_MANUSCRIPT_INDEX.md) |
| Writing the paper (Cheran / coauthor) | [`docs/CHERAN_MANUSCRIPT_HANDOFF.md`](docs/CHERAN_MANUSCRIPT_HANDOFF.md) |
| Checking what may be claimed | [`docs/CLAIM_MAP_PREWRITE.md`](docs/CLAIM_MAP_PREWRITE.md) + [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md) |
| Reading results only | Abstract + Results Summary below; numbers from `benchmarks/results/` |
| Reviewing remediation | [`COLIDE_Remediation_Update_Review.md`](COLIDE_Remediation_Update_Review.md) |

**Principal BoT:** sealed multi-seed test macro-F1 **0.9780 ± 0.0033** · champion md5 `80a90f7cc210276300eaa90173a5a385`.  
**Corrected ToN:** CNN **0.8075** / RF **0.9626**. Historical “clean” 0.9526 / 0.9851 are **INVALID**.

### Repository layout

| Path | Contents |
|------|----------|
| `docs/` | Documentation index — **current authority, manuscript, DICC, historical packs** |
| `docs/manuscript/` | Draft manuscript + figures |
| `scripts/` | Training, benchmarks, parity gates, stale-claim guard |
| `inference/kernels/` | Option A Custom CUDA sources (`.cu`); build locally — see [Building the CUDA kernels](#building-the-cuda-kernels) |
| `model/` | Weights — see [`model/README.md`](model/README.md) (champion vs historical vs **invalid** ToN-clean) |
| `benchmarks/results/` | Claim-eligible JSON/gates, including the artifacts behind every headline number (many raw benches stay local / gitignored) |
| `data/` | BoT-IoT raw and processed splits are **not** tracked (size). The ToN-IoT processed splits under `data/processed_toniot/` **are** tracked deliberately, so the corrected leakage-safe result is reproducible without redownloading |
| `tests/` | Unit tests (`pytest tests/`) |
| `dicc_scripts/` | UM DICC campaign helpers |
| `HANDOFF.md`, `DAILY_LOG.md`, `AGENTS.md`, `CLAUDE.md` | **Internal / historical** coding-session notes — not paper authority |
| Root `COLIDE Remediation…Checklist.md` | Long checklist; some boxes lag later gate JSONs — prefer review + RESULTS_INDEX |

### Building the CUDA kernels

Compiled binaries are **not** tracked (they are architecture-specific and rebuildable).
Each `.cu` file is standalone — no Makefile — and runs its own benchmark when executed:

```bash
nvcc -arch=sm_86 -o inference/kernels/fused_block1 inference/kernels/fused_block1.cu   # RTX 30xx / A100 (Ampere)
nvcc -arch=sm_70 -o inference/kernels/fused_block3 inference/kernels/fused_block3.cu   # V100 (Volta)
```

Swap in `fused_block2`, `fused_block3_fp16`, `fused_block3_naive`, `fused_block4`, or
`fused_pipeline` as needed. `Dockerfile` builds all of them for `sm_86` by default
(`--build-arg CUDA_ARCH=sm_89` to override); `benchmark.sh` runs the suite once built.
Kernels load weights exported by `CNNBiLSTM.export_weights()` into `model/weights/` —
re-export after retraining or you will be profiling stale weights.

### Verifying a published number

Every headline figure traces to a JSON artifact in `benchmarks/results/`, and the
artifacts behind the principal results are tracked in this repository:

```bash
PYTHONPATH=. python scripts/verify_claims.py       # every claim vs its source JSON
PYTHONPATH=. python scripts/check_stale_claims.py  # withdrawn/forbidden strings
pytest tests/                                      # champion hash, config, schemas
```

For example, the principal sealed multi-seed result is
`benchmarks/results/sealed_test/summary.json` (`test_macro_f1_mean`).

### What is *not* current authority

Older status emails, `docs/PROF_POR_*`, `docs/STATUS_REPORT_DRAFT.md`, `docs/execution_plan/`, `docs/audit/`, and `HANDOFF.md` (Aug 12 “pre-manuscript closed”) are **kept for audit**. If they conflict with `docs/RESULTS_INDEX.md` or the claim map, **the latter wins**.

## Abstract

COLIDE is a multi-objective IoT IDS systems project: a sealed-protocol CAD-CBA detection package (competitive but not pure-F1 SOTA vs RF/LGBM) plus **Option A** Custom CUDA kernels (per-block vs matching ops) and measured multi-GPU multi-session latency on UM DICC (V100S + A100). The principal BoT-IoT detection result is sealed multi-seed test macro-F1 **0.9780 ± 0.0033** (n=5). On a **laptop RTX 3050 (WSL2)**, report incomplete Custom CUDA Blocks 1–4 pipeline sums and full-model framework latencies as **separate absolute tables only** — **do not** compute speedups across those tables (CLAIM-PIPE-001). On **DICC**, historical multi-session trees show matching **PyTorch Block 3 faster than CUDA Block 3 FP16** (~363 vs ~513 µs V100S; ~385–391 vs ~667–671 µs A100) as **pre_fix** wall-clock only (**Option B**, 2026-08-15: not a post_fix server rebench — `docs/B3_SERVER_LATENCY_DECISION.md`), while CUDA remains much faster on Blocks 1/2/4. Full Custom CUDA vs full V3 speedup is **not** claimed. B3 source race+alignment is fixed; **local production-weight CUDA–PyTorch full-sequence parity is closed** (`benchmarks/results/block3_parity_gate.json`, `valid=true`, `kernel_status=post_fix`); local parity is **not** DICC latency. Async on-device LLM **dispatch** is **16.60 µs p99** (not full generation or validated free-form explainability). Authority: `docs/CLAIM_MAP_PREWRITE.md`, `docs/B3_SERVER_LATENCY_DECISION.md`, `docs/PRE_MANUSCRIPT_CLOSURE.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/ISSUE_REGISTER.md`, `COLIDE_Remediation_Update_Review.md`.

## Key Contributions

1. **Multi-objective CAD-CBA package** under sealed protocol (HPO, focal + ensemble KD, ablations, dual bars vs classical baselines) — accuracy–efficiency story, not F1 supremacy. Principal test macro-F1 **0.9780 ± 0.0033**.
2. **Option A Custom CUDA** for CNN-BiLSTM blocks: large wins on B1/B2/B4 vs matching PT; DICC B3 wall-clock is **historical pre_fix only** (not a post_fix server rebench; Option B active path — `docs/B3_SERVER_LATENCY_DECISION.md`).
3. **Multi-session multi-GPU measurement** (3 sessions × V100S + A100 SUCCESS trees) with formal compares — not RTX-only portability claims.
4. **Multi-compiler on laptop + DICC** — laptop full-model multi-session **absolute** ranges (eager/compile/TRT/ORT) **and** full DICC batch-1 matrix (eager/compile/ORT/TRT native on V100S+A100); incomplete Custom CUDA block-sum ranges stay in a **separate** table — no cross-table speedups; do not mix laptop numbers with server absolutes.
5. **On-device alert dispatch prototype** micro-benchmark (**16.60 µs p99**); full free-form LLM explainability is **not** a title-level claim.

## Claim hygiene (read first)

| Doc | Role |
|-----|------|
| [`docs/ISSUE_REGISTER.md`](docs/ISSUE_REGISTER.md) | Stable issue IDs (DATA-TON-001 … LLM-001), severity, status |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | Pseudo-sequence, test access, baselines, CUDA scope, energy, bulk throughput, ToN invalid path |
| [`docs/CLAIM_MAP_PREWRITE.md`](docs/CLAIM_MAP_PREWRITE.md) | OK / FORBIDDEN claim rows for manuscript drafting |
| [`docs/B3_SERVER_LATENCY_DECISION.md`](docs/B3_SERVER_LATENCY_DECISION.md) | **Option B (2026-08-15):** DICC B3 latency is **historical pre_fix only** — not post_fix rebench |
| [`docs/PRE_MANUSCRIPT_CLOSURE.md`](docs/PRE_MANUSCRIPT_CLOSURE.md) | Data + **local** CUDA gates closed; DICC B3 post_fix latency still Option B; manuscript prep underway |
| [`docs/CHERAN_MANUSCRIPT_HANDOFF.md`](docs/CHERAN_MANUSCRIPT_HANDOFF.md) | Pack for manuscript lead (locked numbers + claim rules) |
| [`COLIDE_Remediation_Update_Review.md`](COLIDE_Remediation_Update_Review.md) | Evidence-backed readiness review (local gates vs remaining pub items) |
| [`docs/README.md`](docs/README.md) | Full documentation map (current / historical / correspondence) |

Stale-claim guard: `python scripts/check_stale_claims.py` (fails on forbidden ToN-clean strings and risky post_fix+B3+DICC phrasing in active surfaces).

## Results Summary

### Framework Comparison (RTX 3050) — two tables, no cross-table speedups

**MEASUREMENT STABILITY (2026-07-02, session 3):** re-running this benchmark suite — even twice,
back-to-back, minutes apart in the same sitting — gave meaningfully different framework latencies
(torch.compile and TensorRT both swung 14-17% run to run). This is the same phenomenon already
documented for Block 3 alone, now confirmed to affect headline framework numbers too.
Tables report the **range across every independent session measured so far**: 3 for the
full-model framework side (original + two 2026-07-02 re-runs, 20 trials each) and 5 for the
Custom CUDA Blocks 1–4 side (re-derived whenever Block 1–4 kernels are re-checked; see
`benchmarks/results/cuda_kernel_stats_rtx3050.json`).

**Scope caveat (CLAIM-PIPE-001):** Custom CUDA covers fused Blocks 1–4 only. Full CAD-CBA V3 includes
attention / LN / residual / pooling / classifier paths **not** in the CUDA chain. Table A and Table B
have **different scopes**. **Do NOT compute speedups, ratios, or “Custom CUDA beats TRT/eager/…”
claims across Table A and Table B.**

#### Table A — Custom CUDA Blocks 1–4 pipeline sum (incomplete scope; absolute ranges only)

Sum of fused Blocks 1–4 FP16 latencies on RTX 3050 (WSL2). **Not** a full V3 model execution.

| Method | Mean (µs) multi-session range | Scope |
|---|---|---|
| **Custom CUDA FP16 (Blocks 1–4 sum)** | **594–675** | Incomplete vs full CAD-CBA V3 |

#### Table B — Full-model frameworks (absolute multi-session ranges; RTX 3050)

Full CAD-CBA V3 (or exported full-model path) under each backend. Absolute latencies only.

| Method | Mean (µs) multi-session range | Scope |
|---|---|---|
| ORT CPU | 487–699 | Full model |
| torch.compile | 1,519–1,777 | Full model |
| Eager PyTorch | 2,050–2,247 | Full model |
| TensorRT FP16 | 2,427–2,966 | Full model |
| ORT GPU | 3,862–4,652 | Full model |

**Do not** form ratios such as “Custom CUDA is 3.60×–4.99× faster than TensorRT” from these two tables —
those ratios mix incomplete CUDA block sums with complete model graphs and are **FORBIDDEN**
(CLAIM-PIPE-001; `docs/CLAIM_MAP_PREWRITE.md`).

Within Table B, session-to-session drift means exact means move; report **ranges**. ORT CPU can be
competitive with (or faster than) some GPU paths on this sub-1M model under WSL2 — that is a
**full-model-vs-full-model** observation only. torch.compile with CUDA graph capture **crashes** on
BiLSTM (dynamic recurrent control flow). On this laptop, TensorRT FP16 is slower than eager PyTorch
for this sub-1M parameter model (full-model comparison within Table B).

### Cross-Hardware (UM DICC multi-session campaign, Option A)

**Source:** `docs/DICC_EXTRACTION_TABLES.md` · three sessions (S1, S2, Day2) · SUCCESS on laptop.  
**B3 server latency (Option B, 2026-08-15):** B3 columns below are **historical pre_fix wall-clock only**. They are **not** a post_fix rebench on the server and are **not** parity-gated DICC performance. Comparative post_fix B3 speed claims are **out of the active path** until a new SUCCESS tree exists — see `docs/B3_SERVER_LATENCY_DECISION.md`.

| GPU | CUDA B3 FP16 mean | PT B3 mean | PT full V3 mean | Note |
|-----|------------------:|-----------:|----------------:|------|
| **V100S** | ~**513 µs** | ~**363 µs** | ~**964–973 µs** | **historical pre_fix** only — **not** post_fix rebench |
| **A100** | ~**667–671 µs** | ~**384–391 µs** | ~**945–962 µs** | same (**historical pre_fix**; not post_fix) |
| RTX 3050 (laptop) | Table A / per-block | local ranges | Table B | laptop only; DICC matrix is separate below |

Legacy June single-shot pipeline (~551 / ~592 µs CUDA-only) remains **legacy** (no same-GPU PT that day).  
**Do not** claim full Custom CUDA pipeline vs full V3 PT. Prefer per-block tables (B1/B2/B4 CUDA wins; B3 server cells historical-only under Option B).  
**B3 note (2026-08-15):** source **race + reverse-alignment fixed**; **local** production-weight CUDA–PyTorch full-sequence parity is **closed** (`block3_parity_gate.json` `valid=true`, `kernel_status=post_fix`); local sanitizers 0 errors. **Local parity ≠ DICC latency rebench.** DICC multi-session B3 means remain **historical pre_fix** wall-clock of the original SUCCESS trees.  
Details: `docs/B3_SERVER_LATENCY_DECISION.md`, `docs/DICC_EXTRACTION_TABLES.md`, `docs/DICC_B3_CUDA_VS_PT_REPORT.md`, `docs/ISSUE_REGISTER.md`.

### DICC multi-compiler matrix (full-model absolute, batch-1, not Option A)

**Source:** `docs/DICC_MULTI_COMPILER_MATRIX.md` · jobs 395433 (V100S) / 395417 (A100) · n=20, inner=200, warmup=50 · **batch-1**.

| Method | V100S mean (µs) | A100 mean (µs) |
|--------|----------------:|---------------:|
| Eager full V3 | **1041** | **931.5** |
| torch.compile | **865** | **770** |
| ORT CUDA | **895** | **865** |
| ORT CPU | **500** | **461** |
| ORT TensorRT EP | **766** | **2033** |
| TensorRT native FP16 | **528** | **588** |

Absolute framework latencies only — **not** Custom CUDA parity and **not** to be ratioed against incomplete Custom CUDA block sums. Fastest GPU framework path here is **TensorRT native**. ORT CPU is measured but is not a GPU-deployment claim. Laptop Table B remains a separate local full-model story. Do not mix laptop numbers with these server absolutes.

### Per-Block Performance (RTX 3050)

| Block | PyTorch GPU (us) | Custom CUDA (us) | Speedup |
|---|---|---|---|
| 1: Proj+Conv1+BN+ReLU | 404 | 62 | 6.55x |
| 2: Conv2+BN+ReLU+Pool | 282 | 87 | 3.24x |
| 3: BiLSTM FP16 half2 | 784 | 532–602* | 1.30x–1.47x* |
| 4: Dense Head | 122 | 20 | 6.07x |

\* Range across five independent n=100-trial measurement sessions on this dev box. Optimized B3: source race+align **fixed**; **local** production-weight parity **closed** (`block3_parity_gate.json` valid=true). Laptop matching-op latency ratios are non-portable. DICC B3 wall-clock is **historical pre_fix only** (Option B — not a post_fix server rebench; `docs/B3_SERVER_LATENCY_DECISION.md`).

### Block 3 Optimization Progression (7.55x–9.50x)

Step 0 (naive) is now backed by **three independent n=100-trial sessions** of the **fixed** kernel (see
"Naive Kernel Fix" below) — the race-condition fix landed mid-session-2, so no session-1 measurement of
the fixed kernel exists. Step 1 remains a historical single-run figure with no surviving re-runnable
artifact (its kernel file was overwritten by later optimizations). Steps 2-4 are each backed by **five
independent n=100-trial measurement sessions** (`benchmarks/results/cuda_kernel_stats_rtx3050.json`,
regenerated across sessions — see "Measurement Stability" below) rather than one. The PyTorch cuDNN
reference used for per-step ratios is a real n=50-trial mean, **784us** (std 89us, CV 11.3%) from
`benchmarks/results/pytorch_block3_stats_rtx3050.json` (`scripts/benchmark_pytorch_block3_stats.py`, 50
independent subprocess trials — mirrors the CUDA kernel statistical harness so both sides of the ratio are
backed by a real distribution). This resolves an earlier ambiguity between two single-run point estimates
(740.7us vs 943.6us) that bracketed the true mean. With the real baseline, **the FP16 step beats cuDNN in
all five sessions (1.30x–1.47x) on this laptop**; the transposed-W_hh steps (with or without CUDA Graphs)
land at/around parity with PyTorch across all five sessions (0.77x–1.08x). **DICC B3 wall-clock is
historical pre_fix only** (PT B3 faster in those trees) — **not** a post_fix server rebench
(Option B; `docs/B3_SERVER_LATENCY_DECISION.md`). Source race+alignment is fixed; **local**
production-weight parity is **closed**; claim-eligible **DICC** post_fix latency still requires a new SUCCESS tree.

| Step | Configuration | Latency (us) | Cumulative |
|---|---|---|---|
| 0 | Naive (1 thread/hidden), race-fixed | 4,544–5,050 | 1.00x |
| 1 | + Precomputed W_ih x X | 2,901 | 1.57x–1.74x |
| 2 | + Transposed W_hh (coalesced) | 732–1,023 | 4.44x–6.90x |
| 3 | + CUDA Graphs | 724–905 | 5.02x–6.97x |
| 4 | + FP16 half2 FMA gate packing | 532–602 | **7.55x–9.50x** |

#### Naive Kernel Fix (was a disclosed limitation, now resolved for step 0 only)

The naive kernel (step 0) previously carried a disclosed caveat: it failed numerical validation against
the PyTorch reference in a majority of repeated runs (~6/30 passing), attributed to "accumulated FP32
rounding error over its unoptimized summation order." That attribution was wrong — re-running the SAME
seeded input through the SAME binary produced *different* GPU output each time, which pure rounding-order
error cannot do (that would be deterministic). `compute-sanitizer --tool racecheck` confirmed a genuine
shared-memory data race: the per-timestep hidden-state write and the next timestep's read of it raced
despite an intervening `__syncthreads()`. Fixed by double-buffering the hidden state in
`fused_block3_naive.cu` so a timestep's read and write never target the same shared array. Verified:
**0 hazards under racecheck** (was reporting thousands), **100/100 runs pass** at the standard 1e-2
tolerance (was ~6/30), and **20/20 pass even at a 1e-5 tolerance** — i.e. genuinely close to the CPU
reference, not just passing a loose threshold. The naive kernel's latency figure above is now a real
n=100-trial mean of this fixed, verified kernel. **Optimized Block-3 kernels use the same double-buffer
pattern in source; local production-weight full-sequence CUDA↔PT parity is closed
(`benchmarks/results/block3_parity_gate.json`, `valid=true`, `kernel_status=post_fix`). DICC multi-session
B3 latency remains **historical pre_fix only** (Option B — not a post_fix server rebench).**

#### Measurement Stability (new finding, 2026-07-01)

Re-running the full n=100-trial CUDA kernel statistical harness later the same day (needed to safely add
the newly-fixed naive kernel's stats without overwriting the file with a partial run) produced
meaningfully different means for the transposed-W_hh and FP16 configs than the same harness gave earlier
that day — despite each individual session's own internal CV looking tight (6.8%–24.4%):

| Config | Session 1 mean | Session 2 mean | Delta |
|---|---|---|---|
| Transposed W_hh, no graphs | 804 us | 1,023 us | +27% |
| Transposed W_hh + CUDA Graphs | 789 us | 905 us | +15% |
| FP16 half2 | 602 us | 548 us | −9% |

This means within-session CV understates true measurement uncertainty on this WSL2 dev box: there is real
session-to-session drift (thermal state / background load / WSL2 scheduler) that one n=100 run, however
tight its own std, does not capture. Rather than silently picking one session's numbers, this README
reports both as an explicit range. **Phase 3 DICC re-run (tooling ready on `final-polish`):** use
`dicc_scripts/submit_session.sh` for Day 1 and Day 2 on the **same git SHA and same compiled binaries**,
then `scripts/compare_dicc_sessions.py` (rejects provenance mismatches; requires two distinct dates and
`SUCCESS` markers). If DICC session means are stable, describe that as *consistent with WSL2-specific
drift* — not as proof that WSL2 is the sole cause. Operator guide: `dicc_scripts/README.md`.

### Detection Accuracy — BoT-IoT (principal = sealed multi-seed test)

**Principal result (use this in headlines and abstracts):** sealed multi-seed **test** macro-F1  
**0.9780 ± 0.0033** (n=5, seeds 42–46, protocol `botiot_v1`, path A).  
Source: `benchmarks/results/sealed_test/summary.json`.  
Min-class F1 mean **0.9292**; Theft F1 mean **1.0**. Champion weights md5  
`80a90f7cc210276300eaa90173a5a385` (unchanged by sealed runs).

Protocol-era dual bars (val, same protocol family): LightGBM **0.9818** (still tops pure F1); RF **0.9778**.  
Detection is an **accuracy–efficiency** story — **not** pure-F1 supremacy over all classical models.

#### Development / legacy single-run table (not principal)

Historical single-run two-stage checkpoint evaluation (development era). The rounded **0.9790** figure is
**legacy / historical only** — do not promote it above the sealed multi-seed mean.

| Model | Macro-F1 | Method | Parameters |
|---|---|---|---|
| Two-stage CNN-BiLSTM (**historical / legacy**) | **0.9790** | KD (a=0.6,T=10.0) + focal + real-data FT | 530,181 |
| KD + Focal CNN-BiLSTM | 0.9763 | a=0.6, T=10.0, g=2.0 | 530,181 |
| MLP (distilled) | 0.9624 | Same KD recipe | 400,901 |
| MLP (two-stage) | 0.9542 | Same FT recipe | 400,901 |
| Ensemble KD | 0.9529 | RF+XGB+LGB teacher | 530,181 |
| GPU RF (cuML) | 0.9471 | 200 trees, GPU | -- |
| Original V3 | 0.9352 | CE + SMOTE | 530,181 |
| CPU RF (sklearn, processed splits) | 0.9864* | 200 trees, CPU | -- |

Gap of historical **0.9790** to processed-split RF **0.9864**: **0.74%** (narrative dual bar only).

\* Trained/evaluated on the exact same preprocessed splits (`data/processed/*.npy`) the CNN-BiLSTM
itself uses — the apples-to-apples comparison. `scripts/rf_baseline.py` and `train_distill.py`'s
inline RF teacher each apply their own independent resampling straight from the raw CSVs and give
different (also legitimate, but not directly comparable) numbers — 0.9768 and ~0.975 respectively;
see `scripts/rf_baseline_processed.py` (`benchmarks/results/rf_baseline_processed.json`) for this
figure's source, confirmed reproducible byte-for-byte 2026-07-01. Protocol RF val **0.9778** is the
protocol-fair classical bar for sealed-era dual plots.

### Detection Accuracy — ToN-IoT (corrected leakage-safe protocol)

**Active evidence:** corrected leakage-safe rerun under protocol `toniot_leakage_safe_v1`  
(13-feature allowlist, split seed 42, 60/20/20 stratified, train-only preprocess, **no SMOTE**, **no KD**).  
Source: `benchmarks/results/toniot_corrected/summary.json` · `table.md`.  
Issue / quarantine trail: [`docs/ISSUE_REGISTER.md`](docs/ISSUE_REGISTER.md) **DATA-TON-001**.

| Model | Test macro-F1 | Features | Protocol |
|---|---|---|---|
| **CNN-BiLSTM (hard-label, class-weighted CE)** | **0.8075** | 13 allowlist | `toniot_leakage_safe_v1` |
| CPU RF (same split) | **0.9626** | 13 allowlist | same |

Feature SHA-256: `838239eea277712ed719a17ea5f451eebbea368fa673a0676820741b438ecb61`.  
`valid: true`, `use_in_manuscript: true`. Split is **random stratified**, not an official temporal/host split.  
This is a **simple corrected experiment** on the same feature allowlist family as the older 13-feat path — not BoT weight transfer. The RF gap is disclosed.

**Comparable older package path (honest, not superseded as invalid):** WP8 CAD-CBA-v1 mapped on `data/processed_toniot` — neural test ~**0.811** vs same-split RF ~**0.939** (`benchmarks/results/toniot_final/summary.json`). Prefer the **corrected** table above for multi-dataset manuscript rows.

#### Tombstone — historical “clean” 26-feat path (INVALID / withdrawn)

> **INVALID / tombstone (DATA-TON-001).** Prior tables listed CNN **0.9526** and RF **0.9851**
> (26-feat “clean”) and claimed **+15.4%** CNN lift vs 13 features — all **INVALID**. Those results are
> **withdrawn**: the loader retained target-derived `label` in the feature matrix while predicting `type`,
> fit encoders before split, and applied ordinary SMOTE to integer-encoded categoricals.  
> Artifacts: `benchmarks/results/toniot_clean_comparison.json` (`valid: false`),
> `toniot_clean_comparison.INVALIDATED.json`, `toniot_clean_retrain.json` (`valid: false`).  
> **Do not use in manuscript headline tables.** See `docs/ISSUE_REGISTER.md` DATA-TON-001 and
> `benchmarks/results/toniot_corrected/` for the active replacement.

### KD Sweep (BoT-IoT, 14 configurations)

Round 1 (no focal loss, or a single focal point) established alpha=0.7/T=5.0 as the best region found
at the time. Round 2 (2026-07-01, session 2) extended temperature past 5.0 with focal_gamma=2.0 fixed
across a 3x2 alpha/temperature grid, motivated by Round 1's still-rising T=1->3->5 trend. One config
(a=0.7, T=10.0) is a genuine outlier — collapsed on the Normal/Theft minority classes (precision
0.75/0.67) despite excellent majority-class performance, dragging macro-F1 down; kept in the table as a
real, useful negative result rather than dropped.

| Alpha | Temp | Focal | Val F1 | Test F1 |
|---|---|---|---|---|
| -- | -- | -- | 0.9418 | 0.9330 |
| 0.5 | 1.0 | -- | 0.9703 | 0.9481 |
| 0.3 | 1.0 | -- | 0.9474 | 0.9421 |
| 0.7 | 1.0 | -- | 0.9599 | 0.9284 |
| 0.9 | 1.0 | -- | 0.9567 | 0.9284 |
| 0.5 | 3.0 | -- | 0.9541 | 0.9341 |
| 0.7 | 5.0 | -- | 0.9620 | 0.9547 |
| 0.7 | 5.0 | 2.0 | 0.9728 | 0.9601 |
| 0.6 | 7.0 | 2.0 | 0.9780 | 0.9702 |
| 0.7 | 7.0 | 2.0 | 0.9728 | 0.9687 |
| 0.8 | 7.0 | 2.0 | 0.9751 | 0.9757 |
| 0.7 | 10.0 | 2.0 | 0.9482 | 0.9033 |
| 0.8 | 10.0 | 2.0 | 0.9672 | 0.9745 |
| **0.6** | **10.0** | **2.0** | **0.9757** | **0.9763** |

### MLP Ablation

| Model | Params | Latency (A100, legacy chain) | Test F1 (historical single-run) |
|---|---|---|---|
| **CNN-BiLSTM** | 530,181 | 592 us | **0.9790** (historical / legacy) |
| MLP | 400,901 | 175 us | 0.9542 |

MLP is 3.4x faster on this legacy A100 chain figure but CNN-BiLSTM wins accuracy on the historical
single-run path. Principal detection remains sealed **0.9780 ± 0.0033**. The recurrent architecture's
dynamic control flow exposes compiler limitations — a core systems contribution. Pseudo-sequence
limits: see `docs/KNOWN_LIMITATIONS.md`.

### cuML GPU RF Comparison (A100) — exploratory energy

| Metric | cuML RF | CNN-BiLSTM |
|---|---|---|
| VRAM | 444 MB | ~2 MB |
| F1 (GPU path / historical labels) | 0.9471 | 0.9790 (historical / legacy single-run) |
| Throughput (exploratory) | 2,065,669 f/s | 87,791 f/s |
| Energy (exploratory, GPU-board) | 0.048 mJ/flow | 1.089 mJ/flow |

**Caveats (ENERGY-001):** throughput and energy rows are **exploratory** — heterogeneous measurement
boundaries and board-power sampling; **not** a controlled efficiency contest. Do **not** present **1.089**
as a definitive efficiency loss (or win) vs cuML without those caveats. Prefer WP6b multi-session laptop
energy ranges **0.920–0.943** mJ/flow for systems discussion. CNN-BiLSTM still shows a large **VRAM**
advantage (~222× less in this table) with higher GPU-path F1 under the historical labels used here.

The CNN-BiLSTM A100 throughput above is derived from `a100_energy.json` batch=128 timing
(`128 / (avg_batch_time_ms / 1000)` ≈ 87,791 flows/sec). RTX 3050 **bulk batched** throughput (~25,899 f/s)
is reported under “Bulk batched throughput and exploratory energy” — **not** as a streaming-arrival rate.

### Bulk batched throughput and exploratory energy

| Metric | Value | Label |
|---|---|---|
| Bulk batched throughput (RTX 3050) | ~**25,899** flows/sec (batch=128) | **Not** paced streaming arrivals (BENCH-STREAM-001) |
| Energy (RTX 3050, single-script) | 0.79 mJ/flow | Exploratory |
| Energy (A100, single-script) | 1.089 mJ/flow | Exploratory board-power estimate |
| Energy (WP6b laptop multi-session) | **0.920–0.943** mJ/flow | Preferred systems range |
| Preprocessing overhead | 43.7 us (6.1% of pipeline) | Secondary |
| End-to-end latency (legacy pipeline note) | 717.7 us | Secondary |

The historical “streaming” artifact (`streaming_throughput.json`) does **not** pace arrivals at an offered
rate; report it only as **bulk batched processing throughput**.

### LLM dispatch prototype (not free-form explainability)

| Metric | Value |
|---|---|
| Dispatch overhead | 16.60 us p99 (~2.5% of a local pipeline slice) |
| Generation time | ~8.5 sec/alert (background; not the 16.60 µs figure) |
| Model | TinyLlama 1.1B Q4 (0.77 GB) |
| Alert aggregation | 25,000 DDoS alerts to 10 LLM calls (design note) |
| Deployment intent | On-device / air-gapped prototype |

**LLM-001:** **16.60 µs** is alert construction and queue **dispatch** only. Free-form text quality is
weak in XAI suite checks — do **not** title the work as validated LLM-based explainability.

### GPU Hardware Profile (RTX 3050)

All kernels at **100% theoretical occupancy** (20 SMs, 1536 threads/SM):
- Block 1: 256 threads, 6 blocks/SM
- Block 2: 256 threads, 6 blocks/SM  
- Block 3: 128 threads, 12 blocks/SM
- Block 4: 64 threads, 24 blocks/SM

## CUDA Kernel Design

### Architecture
Four fused kernels replacing PyTorch operators:
- **Block 1**: Linear projection + reshape + Conv1D + BatchNorm + ReLU (5 ops fused)
- **Block 2**: Conv1D + BatchNorm + ReLU + MaxPool (4 ops fused)
- **Block 3**: 2-layer BiLSTM with transposed W_hh and FP16 half2 FMA
- **Block 4**: Dense + ReLU + Dense (3 ops fused)

### FP16 Half2 Optimization
1. Repacks W_hh into half2 pairs: (i_gate, f_gate) and (g_gate, o_gate)
2. Uses `__hfma2` for two gates per instruction
3. Shared memory for half-precision hidden states
4. FP32 only for sigmoid/tanh activations

## Datasets

**BoT-IoT** (Koroniotis et al., FGCS, 2019): 10 features, 5 classes, 733,705 test samples

**ToN-IoT** (Moustafa, 2021): active protocol is **`toniot_leakage_safe_v1`** (13 allowlist features, 10 classes; `benchmarks/results/toniot_corrected/`); historical 26-feature “clean” path is **INVALID** (DATA-TON-001)

## Scoped literature notes

(Formerly phrased as “Verified Research Gaps.” These are **bounded literature notes**, not exhaustive
proofs of global uniqueness.)

1. **Custom CUDA for CNN-BiLSTM IDS** — closest prior work identified in our review (Ibrahim et al., *Computer Networks*, 2026) applies custom CUDA kernels to a GNN-based IDS vs. a CPU baseline only (1.22x-1.48x); we target a recurrent CNN-BiLSTM under Option A with **matched operator-vs-operator** tables and **separate** full-model framework tables. On **DICC**, CUDA wins on B1/B2/B4; B3 server wall-clock is **historical pre_fix only** (Option B — not post_fix rebench; local parity closed separately — `docs/B3_SERVER_LATENCY_DECISION.md`). **Do not** cite partial Custom CUDA pipeline sums as end-to-end speedups over TensorRT/eager/ORT.
2. **On-device LLM dispatch for IDS alerts** — Jamshidi et al. (2026) used cloud APIs; we measure fully local **dispatch** at 16.60 us p99 (not full validated free-form explainability).
3. **Full-model framework latencies for sub-1M models** — on laptop RTX 3050, Table B absolute multi-session ranges show TensorRT FP16 slower than eager for this model size; DICC multi-compiler is full-model frameworks only (fastest GPU path there: TensorRT native). Incomplete Custom CUDA block sums (Table A) are **not** ratioed against these full-model rows.
4. **torch.compile on recurrent models** — laptop CUDA-graph path can crash on BiLSTM; DICC full multi-compiler matrix includes compile means ~**865 µs** (V100S) / ~**770 µs** (A100) vs eager ~**1041 / 932** (see multi-compiler table).

## Limitations

- **Principal accuracy is sealed multi-seed, not pure-F1 SOTA:** **0.9780 ± 0.0033** test; protocol LGBM val **0.9818** still leads pure F1. Historical single-run **0.9790** is development-only.
- **ToN clean path invalid:** 26-feat clean numbers quarantined (DATA-TON-001). Active corrected: CNN **0.8075** / RF **0.9626** (`toniot_leakage_safe_v1`). Older 13-feat package path ~**0.811** vs RF ~**0.939** remains comparable only.
- **SMOTE / order sensitivity:** rare Theft and some Stage-A paths depend on synthetic augmentation; historical ToN clean also mis-applied SMOTE to encoded categoricals (corrected path uses **no SMOTE**).
- **Pseudo-sequence:** MLP ablation shows sequential bias is not essential; architecture retained for compiler stress-testing.
- **Incomplete CUDA:** Option A only; full-pipeline Custom CUDA vs full V3 **forbidden**; no partial-pipeline-vs-full-model ratios (CLAIM-PIPE-001).
- **B3 server latency (Option B):** source race+align fixed; **local** production-weight parity closed; DICC B3 means remain **historical pre_fix only** — not post_fix rebench; active comparative post_fix server B3 claim dropped pending Option A rebench (`docs/B3_SERVER_LATENCY_DECISION.md`).
- **Energy exploratory:** prefer WP6b **0.920–0.943** mJ/flow; do not treat 1.089 vs cuML 0.048 as controlled system energy.
- **Bulk throughput ≠ streaming:** ~25,899 f/s is batched processing, not paced arrivals.
- **LLM dispatch-only:** 16.60 µs is not generation or validated XAI.
- **Measurement environment:** WSL2/RTX 3050 session drift → framework ratios as ranges. DICC multi-session SUCCESS (V100S+A100) with same-GPU PT is available under `benchmarks/results/dicc/`.
- **Numerical fidelity:** export path is bit-identical on n=10 reference samples; CUDA block self-checks PASS at disclosed tolerances (FP16 Block 3: 5e-2); **local** B3 production-weight full-sequence parity closed (`block3_parity_gate.json`); native TensorRT engine numerical equivalence is **not** claimed (TRT path skipped in framework gate) — see `docs/paper_text_blocks.md` §15–§16 and `scripts/numerical_fidelity.py`.

Full narrative: [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

## Citation

```bibtex
@article{colide2026,
  title={COLIDE: CUDA-Optimized CNN-BiLSTM for IoT Intrusion Detection with On-Device Alert Dispatch Prototype},
  author={Haque, Ibteshamul and Por, Lip Yee},
  journal={Future Generation Computer Systems},
  year={2026},
  note={Under preparation; LLM component is dispatch-only, not free-form explainability}
}
```

## Acknowledgments

This research was conducted at FCSIT, Universiti Malaya, under Prof. Dr. Por Lip Yee. Computational resources provided by the Data-Intensive Computing Centre (DICC), Universiti Malaya.

## License

Academic research purposes only. Contact authors for commercial licensing.
