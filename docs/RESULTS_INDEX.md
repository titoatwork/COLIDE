# Results index — claim → artifact map

**Date:** 2026-08-14  
**Tip (pre-review-implementation):** see `git rev-parse HEAD`  
**Authority:** `docs/ISSUE_REGISTER.md` · `COLIDE_Remediation_Update_Review.md` · `docs/CLAIM_MAP_PREWRITE.md`

Only rows with `use_in_manuscript: true` (or explicit OK in claim map) may appear in active manuscript tables.

---

## Principal detection (BoT-IoT)

| Claim | Value | Artifact | Status |
|-------|-------|----------|--------|
| Sealed multi-seed test macro-F1 | **0.9780 ± 0.0033** (n=5) | sealed-test summary / README | **OK** |
| Champion identity | md5 `80a90f7cc210276300eaa90173a5a385` | `model/best_model_botiot_twostage.pth` · `config/champion.json` | **OK** |
| Historical 0.9790 | development/legacy only | claim map FORBIDDEN as principal | **historical** |

## ToN-IoT

| Claim | Value | Artifact | Status |
|-------|-------|----------|--------|
| Corrected CNN test macro-F1 | **0.8075** | `benchmarks/results/toniot_corrected/summary.json` | **OK** if clean provenance after Phase-6 rerun |
| Corrected RF test macro-F1 | **0.9626** | same | **OK** if clean provenance |
| Protocol | `toniot_leakage_safe_v1` | same | **OK** |
| Invalid clean CNN/RF | 0.9526 / 0.9851 / +15.4% | `toniot_clean_comparison.json` tombstone | **INVALID** |

## Option A CUDA (per-block)

| Claim | Value | Artifact | Status |
|-------|-------|----------|--------|
| B1/B2/B4 CUDA faster than matching PT (DICC) | extraction tables | `docs/DICC_EXTRACTION_TABLES.md` | **OK** wall-clock historical |
| B3 PT faster than CUDA FP16 (DICC) | ~363 vs ~513 V100S; ~385–391 vs ~667–671 A100 | B3 report | **pre_fix** binaries until post_fix rebench |
| B3 production-weight parity | GPU inject vs PT full seq max abs ~6.5e-6 | `benchmarks/results/block3_parity_gate.json` | **OK** (`valid: true`, post_fix) |
| Full custom pipeline vs full V3 | — | — | **FORBIDDEN** |

## Full-model multi-compiler (batch-1, DICC)

| Method | V100S / A100 | Artifact | Status |
|--------|--------------|----------|--------|
| Eager / compile / ORT / TRT | matrix | `docs/DICC_MULTI_COMPILER_MATRIX.md` | **OK** absolute only |
| Framework logit parity | eager CUDA/ORT/compile pass; TRT native skipped | `benchmarks/results/framework_parity_gate.json` | **OK** for non-skipped backends |

## Throughput / energy / LLM

| Claim | Value | Status |
|-------|-------|--------|
| Bulk batched throughput | ~25,899 f/s batch=128 RTX 3050 | **OK** as bulk only |
| Energy | exploratory board power | **exploratory** |
| LLM dispatch | 16.60 µs p99 | **dispatch only** |

## Archived / invalid

| Artifact | Reason |
|----------|--------|
| `toniot_clean_comparison*.json` | DATA-TON-001 |
| Pre-fix optimized B3 latency as post_fix | CUDA-B3-* open |
| Partial custom vs full-model ratios | CLAIM-PIPE-001 |

---

*Regenerate figures only from rows marked OK / valid.*
