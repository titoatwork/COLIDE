# Table substitutes for manuscript (from locked artifacts)

**Date:** 2026-08-15  
**Purpose:** Honest markdown tables derived only from disk artifacts, for use when figure generation is partial or for camera-ready table paste.  
**Rule:** No invented numbers. Prefer rows with `valid: true` / protocol JSON.  
**Figure inventory:** see [`docs/FIGURE_STATUS.md`](../FIGURE_STATUS.md).

Figure generation status (summary):

- Full suite is **not** covered by a single permanent script (`generate_manuscript_figures.py` does CM + class dist only).
- 2026-08-15 hygiene regenerated dual bars, ablation, WP6b, ToN corrected, class dist, CM, and WP5c Pareto into `benchmarks/results/figures_current/`.
- These tables remain the authoritative numeric fallback.

---

## T1 — BoT-IoT sealed multi-seed TEST (B14)

**Source:** `benchmarks/results/sealed_test/summary.json` · `table.md`  
**Champion md5 unchanged:** `80a90f7cc210276300eaa90173a5a385`

| Seed | val macro-F1 | test macro-F1 | test min-cls | test Theft |
|------|-------------:|--------------:|-------------:|-----------:|
| 42 | 0.9791 | 0.9787 | 0.9333 | 1.0000 |
| 43 | 0.9587 | 0.9798 | 0.9369 | 1.0000 |
| 44 | 0.9797 | 0.9798 | 0.9375 | 1.0000 |
| 45 | 0.9787 | 0.9722 | 0.9014 | 1.0000 |
| 46 | 0.9483 | 0.9796 | 0.9369 | 1.0000 |
| **Mean±std** | 0.9689±0.0145 | **0.9780±0.0033** | **0.9292** | **1.0000** |

---

## T2 — Detection dual-bar values (test vs val honesty)

**Sources:** `sealed_test/summary.json`; `baselines_classical/rf_seed42.json`; `baselines_classical/lgbm_seed42.json`; `rf_baseline_processed.json`; `hpo/summary.json` winner; `multirun/summary.json`

| Series | Split label | Macro-F1 | Notes |
|--------|-------------|----------:|-------|
| B14 sealed multi-seed | **TEST** mean±std | **0.9780±0.0033** | Principal BoT result |
| Protocol RF | val | 0.9778 | Protocol-fair |
| Protocol LGBM | val | **0.9818** | Protocol pure-F1 ceiling |
| Published RF | test (other pipeline) | 0.9864 | **Not** protocol-fair; dual bar only |
| HPO winner seed42 | val | 0.9791 | Train HPs only |
| WP1b multirun | val mean±std | 0.9714±0.0109 | Val-only; do not replace B14 test |

---

## T3 — Ablation ladder (val, seed 42)

**Source:** `benchmarks/results/ablation_ladder/summary.json`

| Row | Name | Val macro-F1 | Min-cls F1 | Theft F1 | Params | µs/sample (bs=256) |
|-----|------|-------------:|-----------:|---------:|-------:|-------------------:|
| A1 | CNN only | 0.6221 | 0.0000 | 0.0000 | 34 821 | 5.03 |
| A2 | BiLSTM only | 0.8058 | 0.5000 | 0.5000 | 372 229 | 15.26 |
| A3 | CNN–BiLSTM | 0.9493 | 0.8571 | 1.0000 | 463 877 | 19.96 |
| A4 | +Attn CE | 0.7378 | 0.0000 | 0.0000 | 530 181 | 24.54 |
| A5 | +Attn+focal | 0.8684 | 0.7059 | 0.7059 | 530 181 | 26.69 |
| A6 | +ens KD | 0.9346 | 0.8462 | 1.0000 | 530 181 | 26.39 |
| **A7** | **full CAD-CBA-v1** | **0.9699** | 0.8974 | 1.0000 | 530 181 | 26.02 |

Protocol RF val **0.9778** · Protocol LGBM val **0.9818** (reference lines; not ladder rows).

---

## T4 — ToN-IoT corrected (prefer for multi-dataset)

**Source:** `benchmarks/results/toniot_corrected/table.md` · `summary.json`  
**Protocol:** `toniot_leakage_safe_v1` · **valid: true** · **use_in_manuscript: true** · SMOTE: false · KD: false  
**Feature SHA-256:** `838239eea277712ed719a17ea5f451eebbea368fa673a0676820741b438ecb61`  
**Split:** stratified random 60/20/20 seed 42 (not official temporal/host split — disclose)

| Model | Val macro-F1 | Test macro-F1 |
|-------|-------------:|--------------:|
| CNN-BiLSTM (hard-label, class-weighted CE) | 0.8066 | **0.8075** |
| RF same split (n_estimators=200, balanced) | 0.9626 | **0.9626** |

### T4a — CNN per-class test (corrected)

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| backdoor | 0.9770 | 0.9992 | 0.9880 | 3742 |
| ddos | 0.9744 | 0.6479 | 0.7783 | 3999 |
| dos | 0.9991 | 0.9113 | 0.9532 | 3799 |
| injection | 0.7287 | 0.7891 | 0.7577 | 3993 |
| mitm | 0.0593 | 0.9087 | **0.1114** | 208 |
| normal | 0.9877 | 0.8660 | 0.9228 | 8408 |
| password | 0.8125 | 0.7256 | 0.7666 | 3972 |
| ransomware | 0.9709 | 0.9857 | 0.9783 | 2947 |
| scanning | 0.9848 | 0.9220 | 0.9524 | 4000 |
| xss | 0.8737 | 0.8596 | 0.8666 | 3027 |

**Rare-class note:** CNN min per-class F1 is **mitm 0.1114** (high recall, low precision under class-weighted CE). Prefer RF for balanced per-class behavior on this protocol. Do not retune against this already-observed test set.

### T4b — RF per-class test (corrected)

| Class | F1 | Precision | Recall | Support |
|-------|---:|----------:|-------:|--------:|
| backdoor | 0.9999 | 1.0000 | 0.9997 | 3742 |
| ddos | 0.9812 | 0.9842 | 0.9782 | 3999 |
| dos | 0.9848 | 0.9968 | 0.9732 | 3799 |
| injection | 0.9719 | 0.9883 | 0.9559 | 3993 |
| mitm | 0.7490 | 0.6547 | 0.8750 | 208 |
| normal | 0.9941 | 0.9938 | 0.9943 | 8408 |
| password | 0.9942 | 0.9990 | 0.9894 | 3972 |
| ransomware | 0.9992 | 0.9983 | 1.0000 | 2947 |
| scanning | 0.9893 | 0.9820 | 0.9968 | 4000 |
| xss | 0.9631 | 0.9410 | 0.9861 | 3027 |

---

## T5 — ToN package path (comparable prior only)

**Source:** `benchmarks/results/toniot_final/table.md`  
**Label:** recipe transfer on `processed_toniot` (13-feat), **not** BoT weight transfer; **not** invalid clean path.

| Model | Val macro-F1 | Test macro-F1 |
|-------|-------------:|--------------:|
| CAD-CBA-v1 (KD selected) | 0.8080 | 0.8110 |
| RF same-split | 0.9400 | 0.9393 |

Prefer **T4** for active multi-dataset claims.

---

## T6 — INVALID ToN clean (tombstone — do not publish as evidence)

**Source:** `benchmarks/results/toniot_clean_comparison.json` (`valid: false`, DATA-TON-001)

| Field | Value | Status |
|-------|------:|--------|
| CNN “clean” macro-F1 | 0.9526 | **INVALID** |
| RF “clean” macro-F1 | 0.9851 | **INVALID** |
| Claimed CNN improvement vs 13-feat | +15.4% | **INVALID — do not cite** |
| Superseded by | `toniot_corrected/summary.json` | use T4 |

---

## T7 — Local systems ranges (WP6b, RTX 3050 only)

**Source:** `benchmarks/results/wp6b_local_ranges/table.md` · `summary.json`  
**n_sessions = 5** · Option A CUDA (per-block / derived pipeline, not full V3 parity)

| Metric | Range (session-mean min–max) | Mean ± std | CV% |
|--------|-----------------------------:|------------|-----|
| PT µs/sample @bs=256 | 24.15–25.68 | 24.90 ± 0.55 | 2.22 |
| PT µs/sample @bs=128 | 29.61–31.32 | 30.59 ± 0.65 | 2.11 |
| Energy mJ/flow @bs=128 | 0.920–0.943 | 0.933 ± 0.010 | 1.05 |
| CUDA block3 FP16 µs | 503.2–508.5 | 505.6 ± 2.2 | 0.44 |
| CUDA derived pipeline µs | 565.2–570.3 | 567.4 ± 1.9 | 0.34 |
| Peak alloc VRAM | **322.2 MiB** (global max) | — | — |
| Historical single-shot energy | 0.786 mJ/flow | HISTORICAL only | — |

**Not** DICC multi-GPU / multi-day.

---

## T8 — Block-3 production-weight parity gate

**Source:** `benchmarks/results/block3_parity_gate.json`  
**valid: true** · **use_in_manuscript: true** · kernel_status: **post_fix**

| Check | max abs error | Pass |
|-------|--------------:|:----:|
| PT vs CPU ref (full sequence) | ~6.49e-6 | yes |
| PT vs CPU ref (last timestep) | ~1.33e-6 | yes |
| GPU inject vs PT (full sequence) | ~3.43e-6 | yes |
| GPU inject vs PT (last timestep) | ~1.24e-6 | yes |
| Hybrid suffix PT vs CPU logits | ~9.54e-6 | yes |

Predeclared FP32 tols: PT↔CPU ref 1e-4; GPU↔PT 1e-3; hybrid logits 1e-4.

---

## T9 — Framework logit parity gate

**Source:** `benchmarks/results/framework_parity_gate.json`  
**valid: true** · **use_in_manuscript: true** · reference: pytorch_eager_cpu  
**summary:** n_backends=5, n_pass=4, n_skipped=1

| Backend | Skipped | Pass | max abs error | Class agree |
|---------|:-------:|:----:|--------------:|------------:|
| pytorch_eager_cuda | no | yes | 0.0364 | 8/8 |
| onnxruntime_cpu | no | yes | 7.63e-6 | 8/8 |
| onnxruntime_cuda | no | yes | 0.0342 | 8/8 |
| torch_compile | no | yes | 0.0364 | 8/8 |
| tensorrt_native | **yes** | — | — | no TRT engine present (not invented) |

---

## T10 — Protocol claims snapshot (selected)

**Source:** `benchmarks/results/claims_package/table.md` (plus post-remediation ToN correction)

| Claim ID | Value | Status |
|----------|------:|--------|
| `bot_sealed_test_multiseed` | 0.9780±0.0033 | LOCKED_TEST |
| `g5_lgbm_val` | 0.9818 | LOCKED_VAL |
| `g3_rf_val` | 0.9778 | LOCKED_VAL |
| `published_rf_test` | 0.9864 | HISTORICAL dual bar |
| `wp5a_a7_val` | 0.9699 | LOCKED_VAL |
| `wp5a_a4_val` | 0.7378 | LOCKED_VAL (attn+CE hurts) |
| `pareto_h8_composite_g6` | 0.9056 @ 4.33 µs | LOCKED_VAL |
| `wp6b_energy_mj_per_flow_range` | 0.920–0.943 | LOCKED_SYSTEMS |
| `wp6b_pt_batch256_us_range` | 24.15–25.68 | LOCKED_SYSTEMS |
| `xai_dispatch_p99_us` | 16.60 | LOCKED_VAL (dispatch only) |
| ToN corrected CNN / RF | **0.8075 / 0.9626** | prefer over package `ton_*` |
| `ton_test` (package prior) | 0.8110 | comparable prior only |
| `dicc_multiday_stats` | PENDING | BLOCKED_DICC |

---

## How to regenerate tables

```bash
# ToN corrected table (already on disk)
cat benchmarks/results/toniot_corrected/table.md

# Sealed test
cat benchmarks/results/sealed_test/table.md

# WP6b systems
cat benchmarks/results/wp6b_local_ranges/table.md

# Claims package
cat benchmarks/results/claims_package/table.md

# Verify claim surface (if verify_claims present)
PYTHONPATH=. .venv/bin/python scripts/verify_claims.py
```

---

*End table substitutes. Prefer these over any prose that still cites invalid ToN clean numbers.*
