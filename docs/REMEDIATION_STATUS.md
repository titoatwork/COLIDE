# Remediation status (handoff)

**Date:** 2026-08-14  
**Scope:** Offline claim hygiene, ToN leakage-safe correction, CUDA B3 source fixes, protocol utilities, tests, license.  
**Commit snapshot:** `2a6de4b` (remediation: full checklist offline correction)
**Authority:** `docs/ISSUE_REGISTER.md` · checklist `COLIDE Remediation and Limited-Scope Improvement Checklist.md`

---

## Champion identity

| Field | Value |
|-------|--------|
| Path | `model/best_model_botiot_twostage.pth` |
| Config | `config/paths.py`, `config/champion.json` |
| MD5 | `80a90f7cc210276300eaa90173a5a385` |
| Principal BoT result | Sealed multi-seed test macro-F1 **0.9780 ± 0.0033** (n=5) |
| Verify | `PYTHONPATH=. python scripts/verify_champion.py` |

Do not overwrite the champion without backup and explicit approval.

---

## Issue ID → status

| Issue ID | Summary | Status |
|----------|---------|--------|
| **DATA-TON-001** | Target-derived `label` in ToN features | **CLOSED offline** — quarantined; superseded by corrected run |
| **CUDA-B3-001** | Optimized B3 hidden-state race | **CODE FIXED** — double-buffer in `fused_block3.cu` / `_fp16.cu`; **AWAITING DICC rebench + sanitizer** |
| **CUDA-B3-002** | Reverse-sequence output alignment | **CODE FIXED** — store at original `pos`; **AWAITING rebench / parity** |
| **CUDA-B3-003** | CUDA / PyTorch output-contract mismatch | **PARTIAL** — documented in `docs/CUDA_WEIGHT_MAPPING.md`; full-sequence parity gate still open |
| **CLAIM-PIPE-001** | Incomplete CUDA pipeline vs full V3 | **QUARANTINED** — full custom-CUDA vs full V3 **FORBIDDEN** (docs) |
| **LOSS-FOCAL-001** | Class-weighted focal formulation | **DISCLOSED** — `StandardFocalLoss` + `LegacyFocalLoss` (no champion retrain) |
| **KD-001** | Noncanonical temperature / KD mix | **DISCLOSED** — `docs/KD_OBJECTIVES.md` (no formula change / no retrain) |
| **BENCH-STREAM-001** | Offered-rate does not pace arrivals | **REFRAMED** — bulk batched throughput; true pacer **DROP** |
| **ENERGY-001** | Nonintegrated / incomparable power | **REFRAMED** — exploratory; controlled remeasure **DROP** unless needed |
| **LLM-001** | Dispatch latency over-claimed | **NARROWED** — alert construction / queue dispatch only |

---

## Corrected ToN-IoT numbers

Protocol: `toniot_leakage_safe_v1` · artifacts: `benchmarks/results/toniot_corrected/` · weights: `model/toniot_corrected/`

| Model | Test macro-F1 | Notes |
|-------|---------------|--------|
| Random Forest | **0.9626** | `class_weight=balanced`, same 13-feat split |
| CNN–BiLSTM (hard-label) | **0.8075** | class-weighted CE, no KD, no SMOTE |
| Label | corrected leakage-safe **random** split (not official temporal/host) | official split **not available** in selected file |

Entry points:

```bash
PYTHONPATH=. python scripts/protocol/toniot_leakage_safe.py
# or
PYTHONPATH=. python scripts/run_toniot_corrected_simple.py
```

Historical “clean” 0.9526 / 0.9851 / +15.4% remain **INVALID** (`DATA-TON-001`); `scripts/train_toniot_clean.py` fail-fasts unless `COLIDE_ALLOW_INVALID_TON=1`.

---

## CUDA Block 3 — fixed source, awaiting rebench

| Item | State |
|------|--------|
| Double-buffer hidden state (FP32 + FP16) | **Done in source** |
| Reverse output alignment to sequence `pos` | **Done in source** |
| `docs/CUDA_WEIGHT_MAPPING.md` | **Done** |
| DICC latency rebench of fixed kernels | **AWAITING_HARDWARE** |
| `compute-sanitizer` racecheck (local sm_86) | **DONE 2026-08-14 — 0 hazards** FP32+FP16 |
| Real-weight numerical parity gate | **OPEN** |

Pre-fix B3 timings remain provisional / not claim-eligible for post-fix speedups.

---

## How to rebench on DICC

Prefer campaign scripts (`docs/DICC_OPS_METHOD.md`, `dicc_scripts/README.md`). Minimal path:

### 1. Compile fixed kernels (GPU node with `nvcc`)

```bash
# From repo root on a CUDA node (or via compile job)
bash dicc_scripts/compile_on_gpu.sh v100   # or a100 / both
# Direct kernels-only (when nvcc is on PATH):
bash dicc_scripts/01_setup.sh --kernels-only --targets sm_70:v100
bash dicc_scripts/01_setup.sh --kernels-only --targets sm_80:a100
```

Equivalent direct `nvcc` (from `inference/kernels/`):

```bash
nvcc -arch=sm_70 -o fused_block3      fused_block3.cu
nvcc -arch=sm_70 -o fused_block3_fp16 fused_block3_fp16.cu
# A100: -arch=sm_80  ·  laptop RTX 3050: -arch=sm_86
```

### 2. Sanitizers (required before claim-eligible B3)

```bash
compute-sanitizer --tool racecheck  ./fused_block3
compute-sanitizer --tool synccheck  ./fused_block3
compute-sanitizer --tool initcheck  ./fused_block3
# Repeat for fused_block3_fp16; save full logs under benchmarks/results/dicc/
```

### 3. Latency / parity sessions

```bash
bash dicc_scripts/submit_session.sh --targets v100   # site-specific partition/gres
bash dicc_scripts/submit_session.sh --targets a100
# or full campaign
bash dicc_scripts/run_campaign.sh
```

Benchmark helpers: `dicc_scripts/lib/run_benchmark.sh`, `scripts/benchmark_cuda_kernels_stats.py`, `scripts/benchmark_pytorch_block3_stats.py`. Record commit, `nvcc` version, arch, and binary SHA-256 in claim-eligible JSON (`source_dirty: false`).

---

## Offline DONE (no DICC required)

- Issue register + stale-claim guard (`scripts/check_stale_claims.py`)
- README / claim map / KNOWN_LIMITATIONS / PRE_MANUSCRIPT claim hygiene
- ToN quarantine + corrected pipeline results
- Focal + KD disclosure; HPO CLI > hpo file > defaults (`train_protocol_ft.py`)
- Champion path centralization + `verify_champion.py`
- `requirements.txt` / `requirements-core.txt`; Dockerfile CUDA arch comments; Academic Research LICENSE
- Tests: toniot blacklist, focal, champion hash, config precedence, result schema, stale-claims import
- Result envelope fields (`scripts/protocol/result_schema.py`); portable shebangs
- Docs reframe: bulk throughput, exploratory energy, LLM dispatch-only, incomplete pipeline FORBIDDEN

## Still open / dropped

| Item | Disposition |
|------|-------------|
| DICC rebench fixed B3 | AWAITING_HARDWARE |
| compute-sanitizer racecheck (laptop) | DONE 0 hazards; synccheck/initcheck optional |
| Full manuscript rewrite + figure regen | OPEN |
| True streaming pacer | DROP (reframe only) |
| Energy remeasure | DROP (exploratory) |
| Official ToN temporal/host split | NOT AVAILABLE |

---

## Git / artifacts note

- `benchmarks/results/*` is ignored **except** `benchmarks/results/toniot_corrected/**` (claim-eligible).
- `model/*.pth` is **not** ignored; champion is already tracked; `model/toniot_corrected/` can be added normally.
- If an ignore rule ever blocks weights: `git add -f model/toniot_corrected/cnn_hardlabel_seed42.pth`.

*End remediation status.*

---

## Local laptop recompile (2026-08-14, RTX 3050 sm_86)

After source fix, kernels recompiled with CUDA 12.6 `nvcc -arch=sm_86 -O3`:

| Binary | Self-check | Sample latency |
|--------|------------|----------------|
| `fused_block3` | **FP32 validation PASSED** | ~854–875 µs (graphs on/off) |
| `fused_block3_fp16` | **FP16 half2 validation PASSED** | ~521 µs |

Parity gate (`scripts/parity_block3_cuda_pt.py`): PT vs CUDA-contract CPU ref **pass** (last max abs ~1.3e-6); CUDA synthetic self-check **PASS**. Gate remains `valid=false` / `use_in_manuscript=false` until real-weight GPU inject + multi-session rebench provenance. Binaries are local rebuilds (not DICC SUCCESS trees).

**compute-sanitizer racecheck (2026-08-14):**

```
fused_block3:      RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
fused_block3_fp16: RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

Self-check PASS under sanitizer as well. Latency under sanitizer is not claim-eligible (instrumentation overhead).

