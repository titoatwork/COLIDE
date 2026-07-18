# 04 — Claims Register

**HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`  
**verify_claims this audit:** 66 PASSED, 0 FAILED, 0 regressions; 5 bold numbers uncovered (`0.6`, `1.00x`, `10.0`, `2.0`, `3.3%`).

Confidence: **HIGH / MED / LOW**. Labels: CURRENT / LEGACY / SUPERSEDED / INVALID / ABSENT / UNCERTAIN.

---

## A. Accuracy — BoT-IoT

| claim_id | text / numbers | doc location | source | verify Y/N | first intro (approx commit) | re-verified? | conf | notes |
|----------|----------------|--------------|--------|------------|------------------------------|--------------|------|-------|
| C-ACC-001 | Two-stage macro-F1 **0.9790** | README L159; abstract L9; paper_text_blocks | `twostage_botiot.json` macro_f1=0.978997… | Y `twostage_final_test_f1` | f98bf33 (+ docs 547e895) | Stage-1 bit-repro 2560348; Stage-2 not bit-repro | HIGH | Champion; JSON **gitignored** on disk |
| C-ACC-002 | RF **0.9864** apples-to-apples | README L167; limitations | `rf_baseline_processed.json` 0.986387… | Y | acdcba5 (traced/sourced) | yes 2026-07-01 byte-for-byte claim in README | HIGH | Keep bar; not 0.9885 |
| C-ACC-003 | Gap **0.74%** | README L9,L15,L168 | 0.9864−0.9790 | Y `rf_gap_botiot_final` | f98bf33 | computed | HIGH | Was 2.25% with 0.9639 |
| C-ACC-004 | Gap was **5.12%** baseline narrative | README L15 | RF vs early V3 area | Y `rf_gap_botiot_baseline` | e928d8e era | MED | Narrative baseline |
| C-ACC-005 | KD stage-1 **0.9763** (a=0.6,T=10,γ=2) | README L160, KD table | distill_botiot_a0.6_T10.0_focal2.json | Y | f98bf33 | repro identical macros | HIGH | gitignored JSON |
| C-ACC-006 | KD table 14 configs | README L197–212 | each distill_*.json | Y (tracked cells) | 1e40b18 / f98bf33 | focal-γ 38b1ea6 negative | HIGH | |
| C-ACC-007 | Outlier a=0.7 T=10 test **0.9033** | README L211 | distill_botiot_a0.7_T10.0_focal2.json | Y | f98bf33 | kept as negative | HIGH | |
| C-ACC-008 | Original V3 **0.9352** | README L166 | early train | partial | ed62ab3 | single training | MED | |
| C-ACC-009 | MLP distilled **0.9624** / two-stage **0.9542** | README L161–162 | ablation_mlp / mlp_twostage | Y | 0553542 | | HIGH | |
| C-ACC-010 | Ensemble KD **0.9529** | README L163 | ensemble_distill.json | Y | 0553542; S5 31940dd | not champion | HIGH | |
| C-ACC-011 | cuML RF F1 **0.9471** | README L164,L228 | cuml_rf_* | partial | 99e7560 | single | MED | |
| C-ACC-012 | Strengthen RF best **0.9885** | NOT published bar | rf_teacher_strengthen.json | N | 31940dd | | HIGH as fact / NOT published | Keep 0.9864 |
| C-ACC-013 | SUPERSEDED two-stage **0.9639** | twostage.json | historical | guarded | 0553542 | SUPERSEDED | — | REGRESSION_GUARDS |

## B. Accuracy — ToN-IoT

| claim_id | numbers | source | verify | conf | notes |
|----------|---------|--------|--------|------|-------|
| C-TON-001 | Clean CNN-BiLSTM **0.9526** | toniot_clean_* | Y | HIGH | |
| C-TON-002 | Clean RF **0.9851** | toniot_clean_comparison | partial | HIGH | |
| C-TON-003 | Gap **3.3%** / JSON 3.25 | toniot_clean_comparison | Y | HIGH | bold 3.3% uncovered |
| C-TON-004 | Original CNN **0.8254** | distill_toniot_v2 | Y | HIGH | |
| C-TON-005 | Original RF **0.9396** | rf_baseline_toniot | partial | HIGH | |
| C-TON-006 | Multi-eval 10-class ~0.803 / 5-cat ~0.930 / bin ~0.956 | toniot_multi_eval | N | MED | secondary |

## C. Framework latency ranges (RTX 3050)

| claim_id | numbers | source construction | verify | conf | notes |
|----------|---------|---------------------|--------|------|-------|
| C-FW-001 | Custom CUDA FP16 **594–675** µs | HIST 674.7 + S3A 652.4 + live + EXTRA 614.5/594.0 | Y | MED-HIGH | Multi-session partially hardcoded in verify_claims |
| C-FW-002 | Eager **2,050–2,247** | HIST/S3A/live | Y | MED-HIGH | |
| C-FW-003 | torch.compile **1,519–1,777** | same | Y | MED-HIGH | |
| C-FW-004 | TensorRT **2,427–2,966** | same | Y | MED-HIGH | |
| C-FW-005 | ORT GPU **3,862–4,652** | same | Y | MED-HIGH | |
| C-FW-006 | ORT CPU **487–699** | same; significance unstable | Y | MED-HIGH | straddles parity |
| C-FW-007 | vs TRT **3.60x–4.99x** | range math | Y | MED | Option A risk if full-V3 framed |
| C-FW-008 | vs compile **2.25x–2.99x** | | Y | MED | |
| C-FW-009 | vs eager **3.04x–3.78x** | | Y | MED | Option A risk |
| C-FW-010 | vs ORT GPU **5.72x–7.83x** | README | partial | MED | |
| C-FW-011 | p<0.001 TRT/compile/eager/ORT-GPU all 3 sessions | prose | partial | MED | |
| C-FW-012 | SUPERSEDED 4.40x/3.33x/2.63x/674.7 | REGRESSION_GUARDS | blocked | — | bd3777e d9e1f79 |

## D. Block 3 / cuDNN

| claim_id | numbers | source | verify | conf | notes |
|----------|---------|--------|--------|------|-------|
| C-B3-001 | cuDNN **784** µs n=50 | pytorch_block3_stats | Y | HIGH | d85271c |
| C-B3-002 | FP16 **532–602** | multi-session | Y | MED-HIGH | live ~531.6 |
| C-B3-003 | Naive **4,544–5,050** race-fixed | multi-session | Y | MED-HIGH | 3eb773a |
| C-B3-004 | Progression **7.55x–9.50x** | range | Y | MED | d9e1f79 |
| C-B3-005 | Beats cuDNN **1.30x–1.47x** | 784/fp16 | Y | MED | Rostam threatens portability |
| C-B3-006 | Transpose **732–1,023** / graphs **724–905** | | Y | MED | |
| C-B3-007 | Step1 **2,901** historical | no JSON | N | LOW | kernel overwritten |

## E. Per-block table (README) — weaker provenance

| claim_id | numbers | conf | notes |
|----------|---------|------|-------|
| C-PB-001 | B1 404/62/6.55x | LOW-MED | live CUDA B1 mean **57.3** ≠ 62 |
| C-PB-002 | B2 282/87/3.24x | LOW-MED | live B2 mean **117.8** ≠ 87 |
| C-PB-003 | B3 784 / 532–602 | MED-HIGH | |
| C-PB-004 | B4 122/20/6.07x | LOW-MED | live B4 **18.7** |

## F. DICC

| claim_id | numbers | source | conf | notes |
|----------|---------|--------|------|-------|
| C-DICC-001 | V100S **~551** (550.664) | dicc_v100_summary.txt | HIGH as LEGACY single-shot | job 363046 |
| C-DICC-002 | A100 **~592** (592.044) | dicc_a100_summary.txt | HIGH as LEGACY | job 363047 |
| C-DICC-003 | vs PyTorch n/a | README footnote | HIGH discipline | was INVALID 3.39x/3.15x |
| C-DICC-004 | Multi-day SUCCESS tree | path checked | **ABSENT** | PLANNED |
| C-DICC-005 | Rostam B3 ~581 vs PT ~512 (V100); ~706 vs ~353 (A100) | DESIGN_PLAN §5.3 | LOW-MED doc | TOOLING-ONLY |

## G. LLM / streaming / energy

| claim_id | numbers | source | verify | conf | notes |
|----------|---------|--------|--------|------|-------|
| C-LLM-001 | **16.60** µs p99 | llm_explainability overhead_p99 | Y | HIGH | 09b509b fixed 5.19 fab |
| C-LLM-002 | Gen ~7.4–8.5 s | json mean 7400ms | partial | MED | |
| C-STR-001 | **25,899** f/s | streaming 25898.67 | Y | HIGH | |
| C-EN-001 | RTX **0.79** mJ | energy_efficiency | Y | HIGH | |
| C-EN-002 | A100 **1.089** mJ | a100_energy | Y | HIGH | |
| C-EN-003 | A100 thr **87,791** | derived | Y | HIGH | e928d8e fixed mislabel |
| C-EN-004 | cuML 444MB / 2,065,669 / 0.048 mJ | cuml_rf_resources | Y | HIGH | |
| C-EN-005 | preprocess 43.7 us; e2e 717.7 | paper_text_blocks | N | MED | |

## H. Fidelity

| claim_id | numbers | source | verify | conf |
|----------|---------|--------|--------|------|
| C-FID-001 | export bit-identical max abs 0 n=10 | numerical_fidelity | Y | HIGH |
| C-FID-002 | 6/6 CUDA self-check PASS | numerical_fidelity | Y | HIGH |
| C-FID-003 | FP16 tol 5e-2 | numerical_fidelity | Y | HIGH |
| C-FID-004 | real_weight classif 0.98 / 1000 | real_weight_validation | N | MED |

## I. Invalid / risky language still present

| claim_id | text | location | status |
|----------|------|----------|--------|
| C-INV-001 | custom CUDA and eager PyTorch are "**the same computation**" | README L66–68 | **INVALID** per DESIGN_PLAN §5.1 |
| C-INV-002 | Abstract full-pipeline framework speedups as primary | README abstract | **RISKY** under Option A |
| C-INV-003 | paper_text_blocks pipeline speedup 3.04x–3.78x over eager | paper § | same risk |

## J. Champion md5

| Item | Value | Status |
|------|-------|--------|
| Path | `model/best_model_botiot_twostage.pth` | present |
| md5 | `80a90f7cc210276300eaa90173a5a385` | **CONFIRMED** this audit |
| BACKUP_0.9790_s5 | same md5 | confirmed |

## K. verify_claims gaps

**Bold uncovered:** 0.6, 1.00x, 10.0, 2.0, 3.3% (KD hypers + ToN gap formatting).

**Not in CLAIMS but load-bearing:** per-block point speedups (6.55x etc.), param counts, test set sizes, preprocess 43.7, e2e 717.7, 222x VRAM, occupancy 100%, V100S>A100 narrative, alert aggregation 25000→10, Ibrahim 1.22–1.48x.
