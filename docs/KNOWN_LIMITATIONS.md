# Known Limitations

**Date:** 2026-08-14  
**Purpose:** Explicit scientific and systems limitations for manuscript and README honesty.  
**Related:** `docs/ISSUE_REGISTER.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/PRE_MANUSCRIPT_CLOSURE.md`, `docs/REMEDIATION_STATUS.md`, `COLIDE_Remediation_Update_Review.md`

This document does **not** authorize new experiments. It records what must not be overclaimed.

---

## 1. Pseudo-sequence architecture

Flow features are presented as a short “sequence” for the CNN–BiLSTM stack, but they are not true multi-timestep packet sequences with temporal causality guarantees. MLP ablations show that a pure sequential inductive bias is **not essential** for competitive macro-F1 on BoT-IoT. The recurrent design is retained primarily as a **compiler / kernel stress case** (dynamic control flow, BiLSTM) and as the frozen CAD-CBA package, not as proof that sequential modeling is required for this tabular feature set.

---

## 2. Historical test access

Early development runs (including the historical single-run two-stage macro-F1 **0.9790**) were produced under protocols that are **not** identical to the sealed multi-seed test gate. The principal BoT-IoT result is sealed multi-seed test macro-F1 **0.9780 ± 0.0033** (`benchmarks/results/sealed_test/summary.json`, n=5, seeds 42–46). Historical **0.9790** may appear only as a **development / legacy** figure, never as the principal published accuracy.

Do not mix val-only multirun or HPO numbers with sealed test means without labels.

---

## 3. Baseline split-qualification

Classical baselines appear under more than one pipeline:

| Bar | Approx. macro-F1 | Protocol note |
|-----|------------------|---------------|
| Protocol LightGBM (val) | **0.9818** | Same sealed-era protocol; tops pure F1 |
| Protocol RF (val) | **0.9778** | Protocol-fair classical |
| Published RF (processed splits) | **0.9864** | Apples-to-apples on `data/processed/*.npy` but **not** interchangeable with every KD teacher RF |
| cuML GPU RF | **0.9471** | Different runtime / feature path |

**Rule:** dual-bar detection claims must label which baseline pipeline is used. Do not claim pure-F1 SOTA over all classical models.

---

## 4. Stage-A SMOTE-before-scale (and related order issues)

Historical and some training paths apply SMOTE (or resampling) and scaling in orders that can leak distributional information or apply ordinary SMOTE to **integer-encoded categorical** fields (especially ToN-IoT “clean”). This is a disclosed methodology risk. Package decisions prefer documented freeze-card recipes. The **corrected** ToN path (`toniot_leakage_safe_v1`) uses train-only preprocess and **no SMOTE**; historical clean remains quarantined under **DATA-TON-001**.

---

## 5. Incomplete Custom CUDA scope (Option A) — partial vs full

Custom CUDA implements fused Blocks 1–4 only. Full CAD-CBA V3 includes attention, LayerNorm, residual connections, temporal mean pooling, and the final classifier — stages **not** in the CUDA chain. Therefore:

- Full Custom CUDA pipeline vs full V3 PyTorch speedup is **FORBIDDEN**.
- **Any ratio of a partial custom pipeline sum (Blocks 1–4) vs full-model eager / torch.compile / TensorRT / ORT is FORBIDDEN** (CLAIM-PIPE-001; review Phase 1).
- Maintain **two separate table classes** only:
  1. Matched operator-vs-operator (per-block).
  2. Complete model-vs-complete model frameworks.
- Partial Custom CUDA pipeline sums may appear only as **absolute** latencies with incomplete-scope labels — **never** with a “vs Custom CUDA” ratio column against full-model frameworks.
- Prefer per-block Option A tables (B1/B2/B4 CUDA wins; B3 honesty).

See **CLAIM-PIPE-001**, `docs/CLAIM_MAP_PREWRITE.md`, `COLIDE_Remediation_Update_Review.md`.

---

## 6. B3: source fixed; production-weight parity not established

DICC multi-session measurements show matching **PyTorch Block 3 faster than CUDA Block 3 FP16** on V100S and A100. Those timings are real as **measured wall-clock latency of pre_fix historical binaries**, not as post_fix semantic parity.

**Source status (2026-08-14):** race double-buffer and reverse-alignment fixes are in the optimized FP32/FP16 kernels; local synthetic self-checks PASS; `racecheck` reported 0 hazards on the laptop rebuild. Tracked as **CUDA-B3-001/002/003** with status **CODE_FIXED_AWAITING_REBENCH**.

**Still open:** production-weight CUDA–PyTorch equivalence (export champion LSTM tensors, inject into CUDA, compare full aligned sequence and hybrid PyTorch-suffix logits), full sanitizer suite (synccheck/initcheck/memcheck), and corrected server rebench (or drop of comparative B3 claim).

Until the production-weight parity gate is green:

- Treat B3 CUDA vs PT as **provisional / pre_fix** wall-clock of historical kernels.
- Do not claim closed “matching-op correctness” or post_fix speedups.
- Welch/Cohen stats across CUDA n=100 vs PT n=20 harnesses are secondary; direction is unambiguous for wall-clock **pre_fix**.

---

## 7. Batch-1 multi-compiler matrix

The DICC multi-compiler table (eager / torch.compile / ORT / TensorRT native, jobs 395433 / 395417) reports **batch-1 absolute** latencies under CentOS7 cluster constraints. It is **not**:

- a Custom CUDA parity study;
- portable to laptop multi-session **ranges** without remeasure;
- a batch-128 or throughput-optimized deploy curve.

Do not mix laptop framework ratios with DICC absolute µs in one headline.

---

## 8. Exploratory energy measurements

Energy figures from `energy_efficiency.json` / `a100_energy.json` (e.g. RTX ~**0.79** mJ/flow, A100 ~**1.089** mJ/flow, cuML ~**0.048** mJ/flow) use heterogeneous checkpoints and measurement boundaries (GPU-board power sampling, sparse before/after samples). They are **exploratory**:

- Prefer WP6b multi-session laptop ranges: energy **0.920–0.943** mJ/flow.
- Do **not** present **1.089** as a controlled efficiency win or loss vs cuML without caveats.
- Do not call GPU-board samples “CPU energy” or total-system energy.
- Dense integrated GPU-board energy reruns are optional and not Phase 1.

See **ENERGY-001**.

---

## 9. Bulk throughput, not streaming arrival simulation

The experiment behind ~**25,899** flows/s (batch=128, RTX 3050) does **not** pace arrivals at an offered rate. It measures **bulk batched processing throughput**. Claims about controlled offered load, queueing delay, saturation curves, or sustainable streaming rate are **not supported** by the current harness.

See **BENCH-STREAM-001**.

---

## 10. LLM dispatch-only (not validated free-form explainability)

**16.60 µs p99** is alert-construction / queue-**dispatch** overhead. Background generation (~seconds per alert) and free-form text quality (e.g. weak feature-mention rates in XAI suite) are separate. Title-level “LLM-Based Explainability” overclaims the evidence. Retain only as an **on-device dispatch prototype**.

See **LLM-001**.

---

## 11. ToN-IoT: invalid “clean” path vs corrected leakage-safe path

### Invalid (quarantined / withdrawn)

Historical 26-feature “clean” results — CNN **0.9526**, RF **0.9851**, and **+15.4%** improvement language — are **INVALID** for manuscript use (**DATA-TON-001**). Root causes include target-derived `label` in features, encoder-before-split, and SMOTE on encoded categoricals. Artifacts:

- `benchmarks/results/toniot_clean_comparison.json` (`valid: false`)
- `benchmarks/results/toniot_clean_comparison.INVALIDATED.json`
- `benchmarks/results/toniot_clean_retrain.json` (`valid: false`)

Never re-label these experiments as “clean” in active tables.

### Active (corrected leakage-safe protocol)

Protocol **`toniot_leakage_safe_v1`** (minimal corrected experiment): 13-feature allowlist, split seed **42**, 60/20/20 stratified, **train-only** encoders/scaler, **no SMOTE**, **no KD**, hard-label CNN with class-weighted CE. Artifacts: `benchmarks/results/toniot_corrected/` (`summary.json`, `table.md`; `valid: true`, `use_in_manuscript: true`). Feature SHA-256: `838239eea277712ed719a17ea5f451eebbea368fa673a0676820741b438ecb61`.

| Model | Test macro-F1 | Source |
|-------|---------------|--------|
| CNN-BiLSTM (hard-label) | **0.8075** | `toniot_corrected/summary.json` |
| RF same split | **0.9626** | same |

**Split caveat:** this is a **random stratified** train/val/test split, **not** an official temporal or host-based ToN-IoT split (no official train/test file pair was present under `data/raw/toniot/` for this run). Disclose that limitation in multi-dataset prose.

### Comparable prior (honest older package path)

WP8 final-method on `data/processed_toniot` (13 features, 10 classes) remains a labeled comparable package path — neural ~**0.811** vs RF ~**0.939** (`toniot_final/summary.json`) — **recipe transfer**, not BoT weight transfer. Prefer the **corrected** numbers above for active multi-dataset claims.

---

## 12. Other disclosed limits

- **Class imbalance:** rare Theft class historically depended on synthetic augmentation in some recipes.
- **Measurement environment:** WSL2 / RTX 3050 session drift → report framework numbers as **ranges**.
- **Numerical fidelity:** export path bit-identical on small n; FP16 B3 tolerances disclosed; optimized B3 parity still open.
- **Champion freeze:** md5 `80a90f7cc210276300eaa90173a5a385` — do not overwrite without explicit backup and approval.

*End known limitations.*
