# 06 — Consistency / Contradiction Matrix

**HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`

---

## 1. Option A violations / construct risks

| Surface | Statement | Conflict | Severity |
|---------|-----------|----------|----------|
| README L66–68 | chained CUDA and eager full-model are "**the same computation**" | DESIGN_PLAN §5.1: V3 has attention/LN/GAP; CUDA last-timestep; fused_pipeline skips true B3 chain; harness `full_pipeline_cuda_vs_pytorch.valid=false` | **HIGH — INVALID if published as written** |
| README abstract | Primary claims 3.60x–4.99x TRT, 3.04x–3.78x eager | Same architecture parity gap for end-to-end model claims; TRT/eager time full V3 export path vs incomplete custom blocks | **HIGH risk** |
| paper_text_blocks HPC § | "pipeline speedup over eager PyTorch 3.04x–3.78x" | Same | **HIGH risk** |
| README L76–78 | Correctly warns not to publish full-pipeline CUDA/PT until parity | Contradicts L66–68 "same computation" | Internal README inconsistency |
| FINAL_PLAN / DESIGN_PLAN | Option A locked | README abstract still Option-A-unsafe lead | Doc hierarchy: plan wins for truth; README lags |

## 2. Numeric mismatches (rounding / stale / drift)

| Metric | Surface A | Surface B | Delta | Verdict |
|--------|-----------|-----------|-------|---------|
| Champion F1 | README 0.9790 | twostage_botiot 0.978997… | rounding | OK CURRENT |
| RF F1 | README 0.9864 | rf_baseline_processed 0.986387… | rounding | OK |
| Gap | 0.74% | (0.986387−0.978997)*100≈0.739% | OK | |
| ToN gap | README **3.3%** | JSON **3.25** | rounding | OK; verify uses both |
| Streaming | README 25,899 | JSON 25898.678… | rounding | OK |
| LLM p99 | 16.60 | 16.59917… | rounding | OK |
| A100 energy | 1.089 | 1.088789 | rounding | OK |
| V100 total | README 551 | txt 550.664 | rounding | OK LEGACY |
| A100 total | 592 | 592.044 | OK LEGACY | |
| B1 CUDA table | README **62** | live stats mean **57.33** | ~8% | **MISMATCH** — table stale vs multi-session live |
| B2 CUDA table | README **87** | live **117.79** (CV 251%) | large | **MISMATCH** — table not multi-session range |
| B4 CUDA table | README **20** | live **18.72** | small | mild drift |
| B3 FP16 | README 532–602 | live mean 531.59 | at/below low end | live session at floor of range |
| Derived total live | 593.89 | range low 594 | consistent with EXTRA 594 | OK |
| Framework Custom CUDA live | stats_v2 mean 652.42 | derived from kernels 593.89 | different sessions | expected; range spans |
| cuML resources F1 | cnn_bilstm_f1 **0.9639** | champion 0.9790 | stale label | **STALE inside JSON** (table uses 0.9790 in README) |
| cuml_rf_native laptop F1 | custom_cuda_f1 **0.9601** | 0.9790 | stale | **STALE** |
| MLP latency comparison field | comparison_cnn_bilstm_us **674** | A100 chained 592 / range ~594–675 | mixed platforms | note field inconsistency |
| twostage.json | 0.9639 | twostage_botiot 0.9790 | superseded file remains | SUPERSEDED artifact |
| statistical_confidence | early means | statistical_significance_v2 | different | SUPERSEDED for headlines |

## 3. Git tracking contradictions

| File | Needed for claim | Tracked? |
|------|------------------|----------|
| twostage_botiot.json | 0.9790 | **NO (gitignored)** |
| rf_baseline_processed.json | 0.9864 | **NO** |
| cuda_kernel_stats_rtx3050.json | ranges | **NO** |
| pytorch_block3_stats_rtx3050.json | 784 | **NO** |
| numerical_fidelity.json | fidelity | **NO** |
| Round-2 distill JSONs | KD table | **NO** |
| statistical_significance_v2.json | framework | YES |
| llm_explainability.json | 16.60 | YES |
| dicc_*_summary.txt | 551/592 | YES |

## 4. Platform / methodology contradictions

| Issue | Detail |
|-------|--------|
| Energy A100 vs RTX | README now same-GPU for cuML table; note in cuml_rf_resources still mentions RTX 0.79 not comparable |
| Batch=1 only | Framework comparison scope; not always disclosed in abstract |
| WSL2 drift | Ranges required; DICC multi-day still missing for cluster absolutes |
| Rostam vs UM | DESIGN_PLAN Rostam B3 loss vs README local "beats cuDNN" — different sites; official = UM only |
| PROF_POR_STATUS_REPORT | Contingency draft numbers may not match locked sources; **non-authoritative** |

## 5. HANDOFF / plan vs README frozen numbers

| Item | Plans | README | Match? |
|------|-------|--------|--------|
| 0.9790 / 0.9864 / 0.74% | YES | YES | YES |
| 16.60 p99 | YES | YES | YES |
| 551/592 legacy | YES labeled | YES with n/a PT | YES |
| Option A full-pipeline ban | YES | Footnote yes; abstract/same-computation no | **PARTIAL** |
| Multi-day DICC complete | NO | Footnote pending | YES (both say pending) |

## 6. verify_claims green ≠ Option A safe

**Critical:** `verify_claims.py` can be fully green while README still contains construct-invalid "same computation" full-pipeline framing. Verifier checks **string presence of numbers**, not claim validity under Option A.
