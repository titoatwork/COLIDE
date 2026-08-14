# Issue Register — claim quarantine and remediation

**Authority for remediation status.** Linked from `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, and `docs/PRE_MANUSCRIPT_CLOSURE.md`.  
**Created:** 2026-08-14  
**Phase:** 1 (claim quarantine + documentation correction)  
**Do not silently delete invalid historical results** — preserve with explicit invalidity metadata.

Status values: `OPEN` · `QUARANTINED` · `IN_PROGRESS` · `CODE_FIXED_AWAITING_REBENCH` · `CLOSED`  
Severity: `P0` (blocking) · `P1` (required cleanup) · `P2` (small improvement)

---

## DATA-TON-001 — Target-derived `label` in ToN-IoT features

| Field | Value |
|-------|--------|
| **Severity** | P0 |
| **Summary** | The historical “clean” ToN-IoT path includes the binary `label` column as a numeric input feature while predicting multiclass `type`. Encoders are fit before split; ordinary SMOTE is applied to integer-encoded categoricals. Headline results are scientifically invalid. |
| **Affected files** | `scripts/train_toniot_clean.py`; `scripts/run_toniot_final_method.py` (historical hardcodes); `benchmarks/results/toniot_clean_comparison.json`; `benchmarks/results/toniot_clean_retrain.json`; `model/best_model_toniot_clean.pth`; README (was active table) |
| **Affected results** | CNN “clean” macro-F1 **0.9526**; RF “clean” **0.9851**; claimed **+15.4%** vs 13-feat path |
| **Affected claims** | Any headline ToN clean accuracy or “feature cleaning improved CNN by +15.4%” |
| **Remediation decision** | **CLOSED for corrected path.** Historical clean results remain **quarantined/invalid**. Prefer leakage-safe protocol; legacy `train_toniot_clean.py` fail-fast unless `COLIDE_ALLOW_INVALID_TON=1`. Older 13-feat package path (`toniot_final/`: ~**0.811** vs RF ~**0.939**) remains a labeled comparable prior only. |
| **Completion evidence** | `scripts/protocol/toniot_leakage_safe.py`; `benchmarks/results/toniot_corrected/summary.json` (RF macro-F1 **0.9626**, CNN **0.8075**, `valid: true`); `scripts/train_toniot_clean.py` fail-fast; historical clean JSON remains invalid |
| **Status** | **CLOSED** (corrected path; historical clean still quarantined/invalid) |
| **Date** | 2026-08-14 |
| **Closed commit** | `2a6de4b` |

---

## CUDA-B3-001 — Optimized Block-3 hidden-state race

> **2026-08-14 update:** source double-buffer applied; local RTX 3050 rebuild + self-check PASS; `compute-sanitizer --tool racecheck` reports **0 hazards** on FP32 and FP16. Status remains **CODE_FIXED_AWAITING_REBENCH** for claim-eligible multi-session latency (DICC) and real-weight GPU inject.


| Field | Value |
|-------|--------|
| **Severity** | P0 |
| **Summary** | Optimized BiLSTM kernels (`fused_block3.cu` / FP16) write and read per-timestep hidden state with a race pattern; only the naive kernel was double-buffer fixed. |
| **Affected files** | `inference/kernels/fused_block3.cu`; `inference/kernels/fused_block3_fp16.cu` (and related); CUDA stats JSONs under `benchmarks/results/` and DICC SUCCESS trees |
| **Affected results** | All optimized B3 latency means (laptop ranges, DICC ~513 / ~667–671 µs CUDA B3) |
| **Affected claims** | Matching-op B3 speedup vs PyTorch; progression table FP16 step; “beats cuDNN” on laptop |
| **Remediation decision** | **Race+align fixed in source 2026-08-14** (double-buffer + reverse store at original pos). Wall-clock DICC/laptop numbers remain **pre_fix** until rebench + real-weight parity gate green. |
| **Completion evidence** | `inference/kernels/fused_block3.cu`; `inference/kernels/fused_block3_fp16.cu`; `docs/CUDA_WEIGHT_MAPPING.md`; `scripts/parity_block3_cuda_pt.py` → `benchmarks/results/block3_parity_gate.json` (`kernel_status: code_fixed_awaiting_rebench`) |
| **Status** | **CODE_FIXED_AWAITING_REBENCH** |
| **Date** | 2026-08-14 |
| **Closed commit** | — |

---

## CUDA-B3-002 — Reverse-sequence output alignment

> **2026-08-14 update:** reverse stores at original sequence `pos`; CUDA self-check PASS after rebuild. Full real-weight CUDA↔PT gate still intermediate (`valid=false` until inject).


| Field | Value |
|-------|--------|
| **Severity** | P0 |
| **Summary** | Bidirectional reverse path stores outputs at sequence index `t` while reading input at reverse position `pos`, misaligning reverse outputs vs PyTorch. |
| **Affected files** | Optimized Block-3 CUDA sources; numerical validators |
| **Affected results** | Same B3 numerical/timing artifacts as CUDA-B3-001 |
| **Affected claims** | Semantic parity of Custom CUDA B3 with matching PT B3 |
| **Remediation decision** | **Alignment fixed in source 2026-08-14** (reverse store at original `pos`). Wall-clock DICC/laptop numbers remain **pre_fix** until rebench + parity gate green. |
| **Completion evidence** | `inference/kernels/fused_block3.cu` / `fused_block3_fp16.cu` reverse-alignment comments + store path; `docs/CUDA_WEIGHT_MAPPING.md`; `scripts/parity_block3_cuda_pt.py` |
| **Status** | **CODE_FIXED_AWAITING_REBENCH** |
| **Date** | 2026-08-14 |
| **Closed commit** | — |

---

## CUDA-B3-003 — CUDA / PyTorch output-contract mismatch / production-weight parity

> **2026-08-14 update:** contract documented; race+align fixed in source; synthetic/self-check path intermediate. **Production-weight CUDA–PyTorch parity is not established** (`valid: false`, `use_in_manuscript: false`, `kernel_status: code_fixed_awaiting_rebench`). Status remains **CODE_FIXED_AWAITING_REBENCH**.


| Field | Value |
|-------|--------|
| **Severity** | P0 |
| **Summary** | PyTorch path typically takes `output[:, -1, :]` (last time index); CUDA extract uses last recurrence-step semantics that can diverge for the reverse direction under misalignment. Full V3 attention needs the complete aligned sequence. Production champion weights have not been shown to match CUDA Block-3 output on GPU. |
| **Affected files** | Block-3 harnesses; `scripts/numerical_fidelity.py`; real-weight validators; `scripts/parity_block3_cuda_pt.py` |
| **Affected results** | Fidelity / “matching Block 3” language; any post_fix B3 claim |
| **Affected claims** | Full or per-block “same computation” framing; post_fix matching-op speedups |
| **Remediation decision** | **Contract documented + harness in place.** Full sequence aligned so `fw[k]` and `rev[k]` are both at input time `k`; last timestep = `output[:, -1, :]` on aligned sequence. Race+align fixed in source 2026-08-14; wall-clock remains **pre_fix** until rebench + **production-weight** parity gate green. Option A still forbids full-pipeline CUDA vs full V3. |
| **Completion evidence** | `docs/CUDA_WEIGHT_MAPPING.md`; `scripts/parity_block3_cuda_pt.py` → `benchmarks/results/block3_parity_gate.json` (`valid: false`); fused_block3 extract comments |
| **Status** | **CODE_FIXED_AWAITING_REBENCH** (parity open) |
| **Date** | 2026-08-14 |
| **Closed commit** | — |

---

## CLAIM-PIPE-001 — Incomplete CUDA pipeline vs full V3

> **2026-08-14 Phase 1 update:** active surfaces rewritten to **two strictly separated tables** — Table A = Custom CUDA Blocks 1–4 sum **absolute** ranges only; Table B = full-model framework **absolute** multi-session ranges. Explicit ban on computing speedups across Table A and Table B. Literature bullets that cited partial-CUDA-versus-full-TRT ratios removed/qualified. See `COLIDE_Remediation_Update_Review.md` Phase 1.


| Field | Value |
|-------|--------|
| **Severity** | P0 |
| **Summary** | Custom CUDA covers fused Blocks 1–4 only; full CAD-CBA V3 includes attention, LayerNorm, residual, pooling, classifier paths not in the CUDA chain. Partial pipeline sums must never be ratioed against full-model backends. |
| **Affected files** | `inference/kernels/*`; README Framework Comparison; claim map; harness full-pipeline flag; literature notes |
| **Affected results** | Laptop Custom CUDA Blocks 1–4 sum ranges; full-model eager/compile/TRT/ORT ranges (separate) |
| **Affected claims** | End-to-end Custom CUDA vs full V3; any “vs Custom CUDA” ratio of full-model frameworks vs incomplete block sum (e.g. 3.60×–4.99× over TensorRT) |
| **Remediation decision** | **FORBIDDEN** as model-level speedup and as cross-table ratio. Report (1) matched operator-vs-operator per-block only, (2) full-model-vs-full-model frameworks only, (3) partial pipeline as **absolute** incomplete-scope ranges only. No ratios across (3) and (2). |
| **Completion evidence** | `docs/CLAIM_MAP_PREWRITE.md` §C + §E; README Table A / Table B; `docs/KNOWN_LIMITATIONS.md` §5; `docs/PRE_MANUSCRIPT_CLOSURE.md` Phase 1 row |
| **Status** | **QUARANTINED** (claim forbidden; incomplete scope remains by design under Option A; Phase 1 table separation applied on active surfaces) |
| **Date** | 2026-08-14 |
| **Closed commit** | — |

---

## LOSS-FOCAL-001 — Class-weighted focal-loss formulation

| Field | Value |
|-------|--------|
| **Severity** | P1 |
| **Summary** | Focal + class-weight combination may not match the canonical formulation used in literature; CB-weight ablations underperformed plain focal in package decisions. |
| **Affected files** | Training / loss utilities; imbalance_loss results |
| **Affected results** | Focal vs focal_cb comparisons; package freeze notes |
| **Affected claims** | Any claim that class-balanced focal is the proven best loss without disclosure |
| **Remediation decision** | **CLOSED for disclosure.** `StandardFocalLoss` + `LegacyFocalLoss` documented in `scripts/protocol/losses.py`; champion recipe stays on legacy path; no retrain. Do not claim novel loss theory. |
| **Completion evidence** | `scripts/protocol/losses.py` (`StandardFocalLoss`, `LegacyFocalLoss`); `tests/test_focal_loss.py`; package freeze / imbalance notes |
| **Status** | **CLOSED** (disclosure; no retrain) |
| **Date** | 2026-08-14 |
| **Closed commit** | `2a6de4b` |

---

## KD-001 — Noncanonical temperature handling

| Field | Value |
|-------|--------|
| **Severity** | P1 |
| **Summary** | Distillation temperature / alpha handling may differ from textbook KD (soft-target scaling). Recipe is empirical (a=0.6, T=10, γ≈2). |
| **Affected files** | `scripts/train_distill.py` and related; distill result JSONs |
| **Affected results** | KD sweep table; champion stage-1 0.9763 path |
| **Affected claims** | “Standard KD” without protocol disclosure |
| **Remediation decision** | **CLOSED for disclosure.** Historical objective is `legacy_teacher_smoothed_kl` (prob-space \(p^{1/T}\), no student \(T\), no \(T^2\)); see `docs/KD_OBJECTIVES.md`. Do not retrain champion. |
| **Completion evidence** | `docs/KD_OBJECTIVES.md`; README KD table + freeze cards |
| **Status** | **CLOSED** (disclosure; no retrain) |
| **Date** | 2026-08-14 |
| **Closed commit** | `2a6de4b` |

---

## BENCH-STREAM-001 — Offered-rate variable does not pace arrivals

| Field | Value |
|-------|--------|
| **Severity** | P0 |
| **Summary** | `streaming_throughput` experiment sweeps an “offered rate” but does not pace arrivals; it measures bulk batched processing throughput (peak ~**25,899** flows/s at batch=128 on RTX 3050). |
| **Affected files** | `scripts/benchmark_streaming.py` (or equivalent); `benchmarks/results/streaming_throughput.json`; README |
| **Affected results** | 25,899 f/s figure |
| **Affected claims** | Controlled offered load, saturation, queueing delay, sustainable stream rate, “streaming latency” |
| **Remediation decision** | **REFRAME** as bulk batched throughput. DROP true-streaming arrival claims unless a real pacer is implemented later. |
| **Completion evidence** | README reframe; claim map FORBIDDEN/OK rows |
| **Status** | **QUARANTINED** (claim reframed; script rename optional later) |
| **Date** | 2026-08-14 |
| **Closed commit** | — |

---

## ENERGY-001 — Nonintegrated and incomparable power measurements

| Field | Value |
|-------|--------|
| **Severity** | P0 / P1 |
| **Summary** | Energy scripts use different checkpoints and measurement boundaries; GPU-board power may be sampled during CPU inference; A100 path uses sparse before/after samples rather than dense integrated energy. |
| **Affected files** | Energy benchmark scripts; `benchmarks/results/energy_efficiency.json`; `a100_energy.json`; cuML comparison table |
| **Affected results** | RTX **0.79** mJ/flow; A100 **1.089** mJ/flow; cuML **0.048** mJ/flow |
| **Affected claims** | Controlled efficiency win vs cuML; “CPU energy” from GPU-board sensors; system energy |
| **Remediation decision** | Treat as **exploratory**. Prefer WP6b multi-session ranges (**0.920–0.943** mJ/flow, laptop). Do not present 1.089 as a controlled win/loss without caveats. |
| **Completion evidence** | README reframe; claim map; KNOWN_LIMITATIONS |
| **Status** | **QUARANTINED** (exploratory labeling; optional corrected rerun deferred) |
| **Date** | 2026-08-14 |
| **Closed commit** | — |

---

## LLM-001 — Dispatch latency presented too broadly

| Field | Value |
|-------|--------|
| **Severity** | P0 |
| **Summary** | **16.60 µs p99** measures alert construction / queue-dispatch overhead only, not end-to-end free-form LLM generation or validated explainability quality. |
| **Affected files** | `llm_integration/`; `benchmarks/results/llm_explainability.json`; README title/citation; manuscript spine |
| **Affected results** | 16.60 µs p99; ~8.5 s generation (background); feature-mention ~0.333 quality signal |
| **Affected claims** | Title-level “LLM-Based Explainability”; production-ready on-device XAI |
| **Remediation decision** | **NARROW** to async dispatch prototype. Drop full explainability from title unless “prototype/dispatch” is explicit. |
| **Completion evidence** | README title/sections; claim map FORBIDDEN full free-form title claim |
| **Status** | **QUARANTINED** (wording narrowed; product XAI not implemented) |
| **Date** | 2026-08-14 |
| **Closed commit** | — |

---

## Related open themes (no separate ID yet)

| Theme | Note |
|-------|------|
| Principal BoT number | Sealed multi-seed **0.9780 ± 0.0033** is principal; historical single-run **0.9790** is development/legacy only |
| Batch-1 multi-compiler | DICC matrix is batch-1 absolute protocol — do not mix with laptop ranges |
| Partial vs full ratios | CLAIM-PIPE-001: no speedups across incomplete Custom CUDA sum and full-model tables |
| B3 pre_fix server result | DICC “PT wins B3” is wall-clock of **pre_fix** historical binaries; race+align fixed in source (`CODE_FIXED_AWAITING_REBENCH`); **production-weight parity open**; rebench + parity gate green before post_fix claims |
| Project closure | DATA closed; CUDA evidence + publication synchronization **pending** (`COLIDE_Remediation_Update_Review.md`) |

## Maintenance

- When closing an issue: set Status, Date closed, Closed commit, and point Completion evidence at JSON or PR.
- Forbidden active strings: see `scripts/check_stale_claims.py`.
- Limitations narrative: `docs/KNOWN_LIMITATIONS.md`.

*End issue register.*
