# Prof. Por Feedback — Master Tracker (nothing dropped)

**Source of truth for requirements:** `docs/feedback1.docx` + email on interim report  
**Operational plan:** `docs/execution_plan/*`  
**Update rule:** Every session that finishes work must flip status here.  
**Statuses:** `TODO` | `IN_PROGRESS` | `PARTIAL` | `DONE` | `BLOCKED` (reason) | `RUN_DOCUMENTED` (ran; not in final method; see notes/JSON) | `INCORPORATED` (in final package)

**Commitment:** Achieve **all** items below to exceptional (not “just enough”) standard before calling the paper ready.

---

## Project policy (user-locked 2026-07-19) — SKIP NOTHING

1. **No silent skips.** Every Prof requirement that is scientifically significant is **run** (or BLOCKED only for missing ops/data, e.g. DICC access).  
2. **Run → record.** Every run leaves JSON under `benchmarks/results/` (or documented path) with protocol_id, seed, config, metrics, git_sha, timestamp.  
3. **Then decide.** If useful → **INCORPORATED** into the final method/paper tables. If not → status **RUN_DOCUMENTED** with a short note *what happened* and *why not in final package* (negative result is still evidence).  
4. **No “we ignored it.”** Optional-looking items (SupCon, uncertainty, etc.) still get at least a **bounded experiment** + write-up unless truly impossible (compute/ops) — then BLOCKED with reason and a plan to unblock.  
5. **Documentation of failures counts.** Ablations and failed variants go in appendix/results JSON even if not in the abstract.

---

## A. Email / contribution bar

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| A1 | Local progress acknowledged but not submission-ready | PARTIAL | Interim + reply sent; science ongoing |
| A2 | 0.9790 &lt; RF 0.9864 — detection not sole headline unless improved | PARTIAL | Numbers frozen; improvement TODO |
| A3 | CUDA mainly B3 local RTX 3050; V100S/A100 pending | BLOCKED | Need DICC |
| A4 | Clear quantitative advantage on ≥1 major dimension | TODO | Need tables after science |
| A5 | Not rely mainly on implementation/docs quality | TODO | Ongoing discipline |
| A6 | Strengthen before finalising manuscript | TODO | Manuscript after evidence |

---

## B. §1 Systematic HPO

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| B1 | Controlled HPO (not manual few params) | DONE | WP3 Optuna TPE study `botiot_stage_b_ft_hpo_v1`; 20 trials + full-train refine; `hpo/summary.json` |
| B2 | CNN layers and filters | TODO | Arch frozen CAD-CBA-v1 for KD init; WP2c if plateau |
| B3 | Convolution kernel sizes | TODO | Same — deferred with fixed padding=1 for k=3 |
| B4 | BiLSTM hidden dims and layers | TODO | Arch frozen; WP2c if plateau |
| B5 | Dropout rate | DONE (train) | WP3 searched dropout 0.10–0.50 + attention_dropout 0–0.30; winner dropout≈0.148 att≈0.214 |
| B6 | LR and LR scheduler | DONE | WP3: lr log 1e-5–3e-3; scheduler {none,cosine,step}; winner lr≈5.89e-5 cosine |
| B7 | Batch size | DONE | WP3 categorical {128,256,512,1024}; winner **1024** |
| B8 | Focal-loss parameters | DONE (γ) | WP3 γ∈[0.5,3.5]; winner **≈1.917** (α class-weights still open B9) |
| B9 | Class weights | PARTIAL | npy exists; not in WP3 search (focal plain) |
| B10 | Distill T and α | PARTIAL | Historical + WP4b fixed α=0.6 T=10; not in stage_b_ft HPO |
| B11 | Sequence length | TODO | reshape fixed [2,32] with V3 |
| B12 | Decision thresholds minority | RUN_DOCUMENTED | WP2d on `ft_focal_seed42`: all decode variants = argmax val macro 0.9780 (Δ0); keep argmax (`thresholds_focal_seed42.json`) |
| B13 | Objectives: val macro-F1, bal-acc, minority recall | DONE (logged) | Primary max val macro-F1; min-cls / bal-acc / Theft logged per trial |
| B14 | Test untouched until final config | PARTIAL | WP3 test SEALED; multi-seed sealed test of winner still TODO |

---

## C. §2 Novel method contribution

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| C1 | Not only “standard CNN–BiLSTM stack” | TODO | Need motivated package |
| C2 | Class-aware attention after BiLSTM | TODO | Candidate |
| C3 | Lightweight temporal attention | TODO | Candidate |
| C4 | Multi-scale temporal convolution | TODO | Candidate |
| C5 | Gated CNN–BiLSTM fusion | TODO | Candidate |
| C6 | Class-balanced or logit-adjusted loss | RUN_DOCUMENTED | 4-way FT compare: focal best; focal_cb/logit_adj worse macro (`imbalance_loss/`) |
| C7 | Supervised contrastive (minority) | TODO | Later unless chosen |
| C8 | Asymmetric loss minority | TODO | Later unless chosen |
| C9 | Teacher ensemble distillation | INCORPORATED | WP4b: ensemble student val **0.9401** best among 5 teachers (`teachers_kd/`) |
| C10 | Uncertainty-aware detection | TODO | Later unless chosen |
| C11 | Adaptive per-class thresholds | RUN_DOCUMENTED | Val grid search macro/min/joint/Theft/Normal — no lift vs argmax on focal FT; not in package |
| C12 | Component addresses named weakness | TODO | Imbalance default weakness |
| C13 | Mod comparison table before lock | PARTIAL | `MOD_DECISION_TABLE.md` |

**Rule:** Not every C2–C11 is mandatory; **at least one clear package** must be fully evaluated. Tracker keeps all options until chosen/rejected in writing.

---

## D. §3 Class imbalance

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| D1 | Imbalance first-class (not accuracy-only) | PARTIAL | Metrics + loss compare + thresholds done; stratified still open |
| D2 | Weighted CE | RUN_DOCUMENTED | CE FT control val 0.9755 (`imbalance_loss/ft_ce_seed42.json`) |
| D3 | Focal loss | INCORPORATED | Best in 4-way compare val **0.9780** — default CAD-CBA-v1 |
| D4 | Class-balanced focal | RUN_DOCUMENTED | val 0.9121 — worse macro; JSON kept |
| D5 | Logit adjustment | RUN_DOCUMENTED | val 0.9225 — worse macro; JSON kept |
| D6 | Stratified batch sampling | TODO | |
| D7 | Minority-aware threshold tuning | RUN_DOCUMENTED | WP2d DONE: macro/min/joint/class_f1 on focal seed42; Δmacro=0 Δmin=0 vs argmax; keep argmax |
| D8 | Controlled oversampling | PARTIAL | SMOTE stage_a in protocol |
| D9 | Supervised contrastive | TODO | Optional bounded run later |
| D10 | Select on macro-F1 + per-class rare attacks | PARTIAL | metrics logged; loss pick used val macro-F1 + min cls F1 |

---

## E. §4 Teachers / KD trade-off

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| E1 | Improve teacher or ensemble | INCORPORATED | WP4b ensemble soft-label mean beats solo RF/XGB/LGBM student |
| E2 | RF teacher | RUN_DOCUMENTED | Protocol KD student 0.9346 (2nd); teacher val 0.9750; solid fallback |
| E3 | XGBoost teacher | RUN_DOCUMENTED | Teacher val 0.9918 but student 0.9270 — does not win student |
| E4 | LightGBM teacher | RUN_DOCUMENTED | Teacher val 0.5928; student 0.8829 — weak path |
| E5 | Calibrated tree ensemble | INCORPORATED | Unweighted mean RF+XGB+LGBM probs; student **0.9401** best |
| E6 | Strong neural teacher | TODO | Optional later |
| E7 | Heterogeneous ensembles | INCORPORATED | RF+XGB+LGBM heterogeneous mean (WP4b) |
| E8 | Student not mere imitation — deployment trade-off | PARTIAL | Student ≠ copy of best teacher F1 (XGB teacher≠best student); multi-obj table still open |
| E9 | Lower memory / latency / GPU deploy / temporal | PARTIAL | Scattered numbers |

---

## F. §5 Ablations

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| F1 | CNN only | TODO | |
| F2 | BiLSTM only | TODO | |
| F3 | CNN–BiLSTM | PARTIAL | Baseline exists |
| F4 | + KD | PARTIAL | Historical + protocol WP4b (none vs KD ladder) |
| F5 | + imbalance method | TODO | |
| F6 | + attention/fusion | TODO | |
| F7 | Full proposed method | TODO | |
| F8 | Metrics: macro/weighted F1, bal-acc, P/R, per-class F1 | PARTIAL | metrics.py |
| F9 | Latency, params, memory, energy per ablation | TODO | |

---

## G. §6 Fair baselines (same split)

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| G1 | Logistic Regression | DONE (val) | full stage_b_ft val_macro_f1=0.5231; test sealed |
| G2 | SVM | RUN_DOCUMENTED | pilot 150k failed (&lt;3 samples/class); full re-run TODO |
| G3 | Random Forest | DONE (val) | protocol-fair val **0.9778**; published 0.9864 = other pipeline (`rf_baseline_processed`) |
| G4 | XGBoost | DONE (val) | protocol-fair val **0.9762** (`xgb_seed42.json`) |
| G5 | LightGBM | RUN_DOCUMENTED | full train val **0.5512** — weak; fix/re-run later |
| G6 | MLP | PARTIAL | Historical |
| G7 | 1D-CNN | TODO | |
| G8 | LSTM | TODO | |
| G9 | BiLSTM | TODO | |
| G10 | CNN–LSTM | TODO | |
| G11 | CNN–BiLSTM | PARTIAL | |
| G12 | Transformer / temporal-attention | TODO | |
| G13 | Reproducible lightweight IDS | TODO | If available |
| G14 | Same train/val/test partitions | PARTIAL | protocol v1 |
| G15 | Comparable HPO effort | TODO | |

---

## H. §7 Multi-objective

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| H1 | May not beat RF F1 if deployment better | TODO | Framing ready |
| H2 | Near-RF detection | PARTIAL | 0.9790 vs 0.9864 |
| H3 | Much lower GPU memory | PARTIAL | ~2MB vs 444MB cuML numbers exist |
| H4 | Faster neural inference | PARTIAL | Local only |
| H5 | Low explanation dispatch | PARTIAL | 16.60 µs |
| H6 | Good minority detection | TODO | Need method |
| H7 | Stable across GPU platforms | BLOCKED | DICC |
| H8 | Pareto F1–latency–memory | TODO | |

---

## I. §8 DICC / same-GPU / multi-day

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| I1 | B3 CUDA vs matching PT same GPU | BLOCKED | No dicc/ tree |
| I2 | Full PT model latency same GPU | BLOCKED | |
| I3 | V100S results | BLOCKED | Legacy single-shot only |
| I4 | A100 results | BLOCKED | Legacy single-shot only |
| I5 | ≥2 different days | BLOCKED | |
| I6 | mean, median, std, CV, CI | BLOCKED | Harness ready |
| I7 | Warm-up protocol | PARTIAL | Tooling |
| I8 | Batch-size sensitivity | TODO | |
| I9 | Statistical significance + effect size | BLOCKED | compare script exists |
| I10 | No generalise from RTX 3050 alone | TODO | Discipline |
| I11 | Portability central | BLOCKED | Until I1–I5 |

---

## J. §9 Explainability

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| J1 | 16.60 µs is dispatch only | PARTIAL | Measured |
| J2 | Faithfulness | TODO | |
| J3 | Consistency | TODO | |
| J4 | Latency (gen vs dispatch) | PARTIAL | Dispatch done |
| J5 | Analyst usefulness | TODO | |
| J6 | Hallucination rate | TODO | |
| J7 | Agreement with model evidence | TODO | |
| J8 | vs SHAP / LIME / attention / rules | TODO | |
| J9 | Structured evidence into LLM | PARTIAL | Features in prompts |
| J10 | Or drop full “explainable” claim | TODO | Decision open |

---

## K. §10 Multi-dataset + final RQs

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| K1 | BoT primary | PARTIAL | Yes |
| K2 | ToN (or other) on final method | PARTIAL | Historical ToN; not final method |
| K3 | Cross-dataset / transfer | TODO | |
| K4 | RQ: improve detection? | TODO | |
| K5 | RQ: minority recognition? | TODO | |
| K6 | RQ: reduce latency or memory? | PARTIAL | Local evidence only |
| K7 | RQ: valid across GPUs? | BLOCKED | |
| K8 | RQ: explainability measurable value? | TODO | |

---

## L. Process / staged plan (his letter)

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| L1 | DICC first, then focused improvement | PARTIAL | Order locked; DICC not done |
| L2 | Mod table before final model | PARTIAL | Written |
| L3 | Avoid changing many parts at once | PARTIAL | Discipline |
| L4 | Phase: freeze preprocess/split/metrics/seeds/hardware/baseline | PARTIAL | Protocol + freeze card |
| L5 | ≥5 independent training runs mean±std | DONE | multirun mean 0.9714 ± 0.0109 n=5 (val only) |
| L6 | Optuna/Bayesian HPO | DONE | WP3 Optuna TPE val-only; winner INCORPORATE 0.9791 (`config/hpo_best.yaml`) |
| L7 | One clear proposed method | TODO | |
| L8 | Deploy: export, parity, profile, kernels, TRT/ORT/compile, FP16/INT8 | PARTIAL | Local CUDA exists |
| L9 | Realistic: trade-off vs beat RF everywhere | PARTIAL | Framing |
| L10 | Paper structure / tables / ToV / repro | TODO | Outline in plan |
| L11 | Fix gitignored claim-source repro | TODO | |
| L12 | Title words only if evaluated | TODO | |

---

## M. Foundation built so far (enablers, not end goals)

| ID | Item | Status |
|----|------|--------|
| M1 | `scripts/protocol/botiot.py` stages | DONE |
| M2 | Shared metrics | DONE |
| M3 | Sealed eval_checkpoint | DONE |
| M4 | Losses helpers | DONE |
| M5 | Threshold search util | DONE |
| M6 | Result envelope | DONE |
| M7 | Execution plan pack | DONE |
| M8 | This tracker | DONE |

---

## Session update log

| Date | What changed |
|------|----------------|
| 2026-07-19 | Tracker created; protocol foundation DONE; all science items TODO/PARTIAL/BLOCKED as above |
| 2026-07-19 | **Skip-nothing policy locked:** run significant items → record → incorporate or RUN_DOCUMENTED |
| 2026-07-19 | User trust + **rock mode**: execute all tracker rows; protocol foundation DONE; next WP1b/method/DICC |
| 2026-07-19 | **WP1b multirun DONE** mean 0.9714±0.0109 n=5; classical protocol-fair LR/RF/XGB/LGBM documented; **imbalance 4-way DONE** focal INCORPORATE |
| 2026-07-21 | **Handoff-only session:** no new experiments. Tracker G3–G5/D*/C6/L5 aligned; `SESSION_CONTINUITY` + `RESULTS_DISK_MANIFEST` written; next chat continues from continuity §5 |
| 2026-07-21 | **WP2d val thresholds DONE** on `ft_focal_seed42.pth`: all variants = argmax 0.9780 → **RUN_DOCUMENTED** keep argmax; B12/C11/D7 updated; next: teachers / Optuna / ablations |
| 2026-07-21 | **WP4b teacher/KD DONE** stage_a_kd α=0.6 T=10 γ=2: ensemble student **0.9401 INCORPORATE**; rf 0.9346 / none 0.9326 / xgb 0.9270 / lgbm 0.8829 RUN_DOCUMENTED; E*/C9 updated; next: Optuna (WP3) or ablations (WP5) |
| 2026-07-21 | **WP3 Optuna HPO DONE** stage_b_ft val-only: Stage A 20 trials (11 complete/9 pruned, max_train=400k); Stage B full refine top-3; winner trial8 **0.9791 INCORPORATE** (Δ+0.0010 vs multirun seed42 0.9780); B1/B5–B8/L6 updated; arch B2–B4 deferred; next: FT from ensemble+HPO or WP5 |

---

## Definition of “all of them done”

Every row is:

- **DONE** / **INCORPORATED**, or  
- **RUN_DOCUMENTED** (ran; not selected; notes + JSON exist), or  
- **BLOCKED** only with unblock path (e.g. waiting DICC access),

and manuscript only after that.

**We do not mark complete early. We do not skip in silence.**
