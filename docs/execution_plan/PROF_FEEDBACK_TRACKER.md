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
| A1 | Local progress acknowledged but not submission-ready | PARTIAL | Interim sent; reply planned |
| A2 | 0.9790 &lt; RF 0.9864 — detection not sole headline unless improved | PARTIAL | Numbers frozen; improvement TODO |
| A3 | CUDA mainly B3 local RTX 3050; V100S/A100 pending | BLOCKED | Need DICC |
| A4 | Clear quantitative advantage on ≥1 major dimension | TODO | Need tables after science |
| A5 | Not rely mainly on implementation/docs quality | TODO | Ongoing discipline |
| A6 | Strengthen before finalising manuscript | TODO | Manuscript after evidence |

---

## B. §1 Systematic HPO

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| B1 | Controlled HPO (not manual few params) | TODO | Need Optuna study |
| B2 | CNN layers and filters | TODO | |
| B3 | Convolution kernel sizes | TODO | |
| B4 | BiLSTM hidden dims and layers | TODO | |
| B5 | Dropout rate | TODO | |
| B6 | LR and LR scheduler | TODO | |
| B7 | Batch size | TODO | |
| B8 | Focal-loss parameters | PARTIAL | Used; not full search |
| B9 | Class weights | PARTIAL | npy exists; often unused in train |
| B10 | Distill T and α | PARTIAL | Manual sweeps done historically |
| B11 | Sequence length | TODO | |
| B12 | Decision thresholds minority | PARTIAL | `thresholds.py` exists; not in full HPO yet |
| B13 | Objectives: val macro-F1, bal-acc, minority recall | PARTIAL | Metrics module has them |
| B14 | Test untouched until final config | PARTIAL | `eval_checkpoint` seals test by default |

---

## C. §2 Novel method contribution

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| C1 | Not only “standard CNN–BiLSTM stack” | TODO | Need motivated package |
| C2 | Class-aware attention after BiLSTM | TODO | Candidate |
| C3 | Lightweight temporal attention | TODO | Candidate |
| C4 | Multi-scale temporal convolution | TODO | Candidate |
| C5 | Gated CNN–BiLSTM fusion | TODO | Candidate |
| C6 | Class-balanced or logit-adjusted loss | PARTIAL | `losses.py` implemented; not full train sweep |
| C7 | Supervised contrastive (minority) | TODO | Later unless chosen |
| C8 | Asymmetric loss minority | TODO | Later unless chosen |
| C9 | Teacher ensemble distillation | PARTIAL | Script exists; student not final |
| C10 | Uncertainty-aware detection | TODO | Later unless chosen |
| C11 | Adaptive per-class thresholds | PARTIAL | Search util exists; not locked in method |
| C12 | Component addresses named weakness | TODO | Imbalance default weakness |
| C13 | Mod comparison table before lock | PARTIAL | `MOD_DECISION_TABLE.md` |

**Rule:** Not every C2–C11 is mandatory; **at least one clear package** must be fully evaluated. Tracker keeps all options until chosen/rejected in writing.

---

## D. §3 Class imbalance

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| D1 | Imbalance first-class (not accuracy-only) | PARTIAL | Metrics; experiments TODO |
| D2 | Weighted CE | TODO | |
| D3 | Focal loss | PARTIAL | Exists |
| D4 | Class-balanced focal | PARTIAL | CB weights helper; full CB-focal train TODO |
| D5 | Logit adjustment | PARTIAL | Loss class exists; sweep TODO |
| D6 | Stratified batch sampling | TODO | |
| D7 | Minority-aware threshold tuning | PARTIAL | Val search util |
| D8 | Controlled oversampling | PARTIAL | SMOTE stage_a |
| D9 | Supervised contrastive | TODO | Optional |
| D10 | Select on macro-F1 + per-class rare attacks | TODO | |

---

## E. §4 Teachers / KD trade-off

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| E1 | Improve teacher or ensemble | PARTIAL | RF strengthen + ensemble script |
| E2 | RF teacher | PARTIAL | Historical |
| E3 | XGBoost teacher | TODO | |
| E4 | LightGBM teacher | TODO | (ensemble has combo) |
| E5 | Calibrated tree ensemble | PARTIAL | ensemble distill |
| E6 | Strong neural teacher | TODO | |
| E7 | Heterogeneous ensembles | PARTIAL | |
| E8 | Student not mere imitation — deployment trade-off | TODO | Need integrated table |
| E9 | Lower memory / latency / GPU deploy / temporal | PARTIAL | Scattered numbers |

---

## F. §5 Ablations

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| F1 | CNN only | TODO | |
| F2 | BiLSTM only | TODO | |
| F3 | CNN–BiLSTM | PARTIAL | Baseline exists |
| F4 | + KD | PARTIAL | Historical |
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
| G2 | SVM | TODO | LinearSVC still to run |
| G3 | Random Forest | PARTIAL | protocol-fair val 0.9778; published 0.9864 is other path — both kept |
| G4 | XGBoost | TODO | |
| G5 | LightGBM | TODO | |
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
| L5 | ≥5 independent training runs mean±std | IN_PROGRESS | smoke seed42 DONE val_macro_f1=0.9755 (2ep); full 5-seed×10ep multirun launched |
| L6 | Optuna/Bayesian HPO | TODO | |
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

---

## Definition of “all of them done”

Every row is:

- **DONE** / **INCORPORATED**, or  
- **RUN_DOCUMENTED** (ran; not selected; notes + JSON exist), or  
- **BLOCKED** only with unblock path (e.g. waiting DICC access),

and manuscript only after that.

**We do not mark complete early. We do not skip in silence.**
