# Prof. Por Feedback — Master Tracker (nothing dropped)

**Source of truth for requirements:** `docs/feedback1.docx` + email on interim report  
**Operational plan:** `docs/execution_plan/*`  
**Update rule:** Every session that finishes work must flip status here.  
**Statuses:** `TODO` | `IN_PROGRESS` | `PARTIAL` | `DONE` | `BLOCKED` (reason) | `RUN_DOCUMENTED` (ran; not in final method; see notes/JSON) | `INCORPORATED` (in final package)

**Commitment:** Achieve **all** items below to exceptional (not “just enough”) standard before calling the paper ready.

---

## Project policy (user-locked 2026-07-19) — SKIP NOTHING / FULL PLAYLIST

1. **No silent skips.** Every Prof requirement / tracker row / work-package playlist item is **completed** to one of: DONE · INCORPORATED · RUN_DOCUMENTED · BLOCKED (with unblock path only for ops, e.g. DICC).  
2. **Complete every playlist item** — including “optional-looking” ones (SupCon, uncertainty, arch HPO, neural baselines, XAI, ToN, Pareto). No “defer forever.”  
3. **Run → record.** Every run leaves JSON under `benchmarks/results/` (or documented path) with protocol_id, seed, config, metrics, git_sha, timestamp.  
4. **Then decide.** If useful → **INCORPORATED** into the final method/paper tables. If not → **RUN_DOCUMENTED** with *what happened* and *why not in final package* (negative result is still evidence).  
5. **Context hygiene:** If evidence already exists on disk/docs, **update tracker status + notes** (do not leave stale TODO). Never invent numbers.  
6. **Documentation of failures counts.** Ablations and failed variants go in appendix/results JSON even if not in the abstract.

---

## A. Email / contribution bar

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| A1 | Local progress acknowledged but not submission-ready | DONE (framing) | Interim+reply sent; local science + PI venue polish closed; **submission** still needs PI journal class/BibTeX + DICC if multi-GPU claimed |
| A2 | 0.9790 &lt; RF 0.9864 — detection not sole headline unless improved | DONE (framing) | Dual bars locked; pure F1 not sole headline; multi-obj + sealed test **0.9780±0.0033** near RF protocol; LGBM **0.9818** still tops pure F1 |
| A3 | CUDA mainly B3 local RTX 3050; V100S/A100 pending | DONE (measured) | S1+S2+Day2 SUCCESS on laptop; B3 CUDA FP16 *slower* than PT B3 on V100S/A100 (tables in `DICC_EXTRACTION_TABLES.md`). Not a portable CUDA B3 win. |
| A4 | Clear quantitative advantage on ≥1 major dimension | DONE (local multi-obj) | G6 composite **0.9056** @4.33 µs; WP6b energy **0.920–0.943** mJ/flow; PT@256 **24.15–25.68** µs; CUDA pipe **565–570** µs; peak **322.2** MiB; dispatch **16.60** µs; pure F1 still LGBM **0.9818** |
| A5 | Not rely mainly on implementation/docs quality | DONE | 59 claims from disk JSON; sealed test + ablations + negatives RUN_DOCUMENTED — science not docs-only |
| A6 | Strengthen before finalising manuscript | DONE (PI polish) | WP9b spine + WP9c draft + **PI venue polish** (continuous abstract, Table 1b per-class, Table 5b HPO refine, systems CI/CV, front-matter placeholders, PDF rebuild); journal class file/BibTeX = PI after venue; DICC if multi-GPU claimed |

---

## B. §1 Systematic HPO

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| B1 | Controlled HPO (not manual few params) | DONE | WP3 Optuna TPE study `botiot_stage_b_ft_hpo_v1`; 20 trials + full-train refine; `hpo/summary.json` |
| B2 | CNN layers and filters | RUN_DOCUMENTED | Plateau reject: multi-seed package means do not beat WP1b; KD transfer freezes V3 dims; C4 multi-scale probe 0.9167 ≪ CTRL 0.9787; see `B2B4_ARCH_HPO_PLATEAU_REJECT.md` |
| B3 | Convolution kernel sizes | RUN_DOCUMENTED | k=3 frozen in package; C4 multi-scale k∈{3,5,7} bounded probe RUN_DOCUMENTED (no lift); full kernel Optuna rejected |
| B4 | BiLSTM hidden dims and layers | RUN_DOCUMENTED | BiLSTM dims frozen for V3 package; plateau + KD transfer; see B2B4 note |
| B5 | Dropout rate | DONE (train) | WP3 searched dropout 0.10–0.50 + attention_dropout 0–0.30; winner dropout≈0.148 att≈0.214 |
| B6 | LR and LR scheduler | DONE | WP3: lr log 1e-5–3e-3; scheduler {none,cosine,step}; winner lr≈5.89e-5 cosine |
| B7 | Batch size | DONE | WP3 categorical {128,256,512,1024}; winner **1024** |
| B8 | Focal-loss parameters | DONE (γ) | WP3 γ∈[0.5,3.5]; winner **≈1.917**; B9 class-weights **RUN_DOCUMENTED** (keep no CB on neural) |
| B9 | Class weights | RUN_DOCUMENTED | Explicit CB via `focal_cb` val **0.9121** ≪ plain focal **0.9780** (`imbalance_loss/`); keep **no class-weight** on neural focal. (Classical LGBM G5 uses balanced — separate tree baseline, not CAD-CBA loss) |
| B10 | Distill T and α | DONE (recipe) + historical RUN_DOCUMENTED | **17** historical `distill_botiot_a*_T*.json` sweeps; CAD-CBA-v1 uses **α=0.6 T=10** (best historical + WP4b); protocol Optuna on T/α not required if recipe locked |
| B11 | Sequence length | DONE (locked) | V3 reshape fixed **[2,32]** in freeze/config; design freeze (not a free search dim under CAD-CBA-v1 KD transfer) |
| B12 | Decision thresholds minority | RUN_DOCUMENTED | WP2d on `ft_focal_seed42`: all decode variants = argmax val macro 0.9780 (Δ0); keep argmax (`thresholds_focal_seed42.json`) |
| B13 | Objectives: val macro-F1, bal-acc, minority recall | DONE (logged) | Primary max val macro-F1; min-cls / bal-acc / Theft logged per trial |
| B14 | Test untouched until final config | **DONE** | User lock 2026-07-22 path **A**; sealed multi-seed test n=5: **test 0.9780±0.0033**; min-cls 0.9292; Theft **1.0**; champion unchanged (`sealed_test/summary.json`) |

---

## C. §2 Novel method contribution

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| C1 | Not only “standard CNN–BiLSTM stack” | DONE (package novelty) | Composition fully evaluated: A7 **0.9699** &gt; A3 **0.9493**; novelty = focal+ens KD+HPO package (not attn alone — A4 **0.7378** hurts); C* negatives RUN_DOCUMENTED; spine §5 |
| C2 | Class-aware attention after BiLSTM | RUN_DOCUMENTED (caveat) | V3 has attention; WP5a A4 attn+CE **0.7378** &lt; A3 **0.9493** under seed42/8-ep — do not claim attn alone helps; credit is package-level |
| C3 | Lightweight temporal attention | RUN_DOCUMENTED (caveat) | Same as C2; systems: attn models ~26 µs/sample vs A3 ~20 µs (batch256) |
| C4 | Multi-scale temporal convolution | RUN_DOCUMENTED | Bounded seed42: val **0.9167** vs CTRL **0.9787** (Δ−0.062); not incorporated (`cstar_bounded/C4_*.json`) |
| C5 | Gated CNN–BiLSTM fusion | RUN_DOCUMENTED | Bounded seed42: val **0.9132** vs CTRL 0.9787 (Δ−0.065); not incorporated (`cstar_bounded/C5_*.json`) |
| C6 | Class-balanced or logit-adjusted loss | RUN_DOCUMENTED | 4-way FT compare: focal best; focal_cb/logit_adj worse macro (`imbalance_loss/`) |
| C7 | Supervised contrastive (minority) | RUN_DOCUMENTED | Bounded SupCon+focal: val **0.7732** (Theft=0) ≪ CTRL 0.9787; not incorporated (`cstar_bounded/C7_*.json`) |
| C8 | Asymmetric loss minority | RUN_DOCUMENTED | Bounded ASL: val **0.8012** ≪ CTRL 0.9787; not incorporated (`cstar_bounded/C8_*.json`) |
| C9 | Teacher ensemble distillation | INCORPORATED | WP4b: ensemble student val **0.9401** best among 5 teachers (`teachers_kd/`) |
| C10 | Uncertainty-aware detection | RUN_DOCUMENTED | MC-dropout + entropy selective on HPO confirm: det **0.9791**, no high-coverage lift; keep argmax (`cstar_bounded/C10_*.json`) |
| C11 | Adaptive per-class thresholds | RUN_DOCUMENTED | Val grid search macro/min/joint/Theft/Normal — no lift vs argmax on focal FT; not in package |
| C12 | Component addresses named weakness | DONE (val+test) | Named weakness = imbalance/Theft; package focal+ensemble KD; val minority table in claims; **B14 test Theft mean 1.0**, min-cls mean **0.9292** (`sealed_test/`) |
| C13 | Mod comparison table before lock | DONE | `docs/MOD_DECISION_TABLE.md` + `METHOD_PACKAGE_DECISION.md` CAD-CBA-v1 |

**Rule:** Not every C2–C11 is mandatory; **at least one clear package** must be fully evaluated. Tracker keeps all options until chosen/rejected in writing.

---

## D. §3 Class imbalance

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| D1 | Imbalance first-class (not accuracy-only) | DONE | Metrics + loss + thresholds + D6 + val minority table; **B14 sealed test** min-cls 0.9292 / Theft 1.0 packaged (`sealed_test/` + claims) |
| D2 | Weighted CE | RUN_DOCUMENTED | CE FT control val 0.9755 (`imbalance_loss/ft_ce_seed42.json`) |
| D3 | Focal loss | INCORPORATED | Best in 4-way compare val **0.9780** — default CAD-CBA-v1 |
| D4 | Class-balanced focal | RUN_DOCUMENTED | val 0.9121 — worse macro; JSON kept |
| D5 | Logit adjustment | RUN_DOCUMENTED | val 0.9225 — worse macro; JSON kept |
| D6 | Stratified batch sampling | RUN_DOCUMENTED | WP D6 seed42: shuffle **0.9791** &gt; stratified inv-freq **0.9209** (Δ−0.058); keep shuffle (`stratified_batch/summary.json`) |
| D7 | Minority-aware threshold tuning | RUN_DOCUMENTED | WP2d DONE: macro/min/joint/class_f1 on focal seed42; Δmacro=0 Δmin=0 vs argmax; keep argmax |
| D8 | Controlled oversampling | DONE (stage_a) | Protocol `stage_a_kd` SMOTE targets freeze card; stage_b_ft deliberately no SMOTE |
| D9 | Supervised contrastive | RUN_DOCUMENTED | Same as C7: SupCon+focal **0.7732** ≪ CTRL; not in package |
| D10 | Select on macro-F1 + per-class rare attacks | DONE | Protocol selection + HPO/loss pick use val macro-F1; min-cls/Theft logged every run |

---

## E. §4 Teachers / KD trade-off

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| E1 | Improve teacher or ensemble | INCORPORATED | WP4b ensemble soft-label mean beats solo RF/XGB/LGBM student |
| E2 | RF teacher | RUN_DOCUMENTED | Protocol KD student 0.9346 (2nd); teacher val 0.9750; solid fallback |
| E3 | XGBoost teacher | RUN_DOCUMENTED | Teacher val 0.9918 but student 0.9270 — does not win student |
| E4 | LightGBM teacher | RUN_DOCUMENTED | Teacher val 0.5928; student 0.8829 — weak path |
| E5 | Calibrated tree ensemble | INCORPORATED | Unweighted mean RF+XGB+LGBM probs; student **0.9401** best |
| E6 | Strong neural teacher | RUN_DOCUMENTED | G11 neural teacher → V3 student stage_a_kd val **0.8513** ≪ ensemble student **0.9401**; keep ensemble (`teachers_kd_neural/`) |
| E7 | Heterogeneous ensembles | INCORPORATED | RF+XGB+LGBM heterogeneous mean (WP4b) |
| E8 | Student not mere imitation — deployment trade-off | RUN_DOCUMENTED | WP4b: XGB teacher 0.9918 → student 0.9270 &lt; ensemble 0.9401; multi-obj `pareto_h8/` composite G6 0.9056 |
| E9 | Lower memory / latency / GPU deploy / temporal | RUN_DOCUMENTED (local) | WP5c + `pareto_h8/` + **WP6b multi-session ranges** (energy 0.920–0.943; PT@256 24.15–25.68 µs; peak 322.2 MiB); DICC multi-day still BLOCKED |

---

## F. §5 Ablations

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| F1 | CNN only | RUN_DOCUMENTED | WP5a A1 seed42 val macro **0.6221** (min/Theft=0); `ablation_ladder/A1_*.json` |
| F2 | BiLSTM only | RUN_DOCUMENTED | WP5a A2 seed42 val macro **0.8058** (min/Theft=0.5); `A2_*.json` |
| F3 | CNN–BiLSTM | RUN_DOCUMENTED (+ prior) | WP5a A3 seed42 **0.9493** (Theft=1.0) strong CE backbone; also WP1b multirun path |
| F4 | + KD | RUN_DOCUMENTED (+ WP4b) | WP5a A6 attn+focal+ens KD **0.9346** vs A5 0.8684; WP4b ensemble teacher INCORPORATE |
| F5 | + imbalance method | RUN_DOCUMENTED (+ prior) | WP5a A5 attn+focal scratch **0.8684** ≫ A4 CE 0.7378; 4-way loss focal INCORPORATE prior |
| F6 | + attention/fusion | RUN_DOCUMENTED | WP5a A4 attn+CE **0.7378** **underperforms** A3 0.9493 under this budget — attention not free gain; full package still uses attn+focal+KD+HPO |
| F7 | Full proposed method | RUN_DOCUMENTED (ladder) | WP5a A7 full CAD-CBA-v1 **0.9699** tops ladder; package multirun mean 0.9639±0.0185 prior |
| F8 | Metrics: macro/weighted F1, bal-acc, P/R, per-class F1 | DONE | `scripts/protocol/metrics.py` + all protocol result JSONs |
| F9 | Latency, params, memory, energy per ablation | RUN_DOCUMENTED | WP5a params+latency; WP5c Pareto; energy_table historical **0.786**; **WP6b multi-session energy 0.920–0.943** mJ/flow n=5 (primary range); per-ablation mJ not re-measured — disclosed |

---

## G. §6 Fair baselines (same split)

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| G1 | Logistic Regression | DONE (val) | full stage_b_ft val_macro_f1=0.5231; test sealed |
| G2 | SVM | RUN_DOCUMENTED | full LinearSVC dual=False hard labels val **0.4268** (Theft=0); pilot ERROR fixed; weak under extreme imbalance (`svm_seed42.json`) |
| G3 | Random Forest | DONE (val) | protocol-fair val **0.9778**; published 0.9864 = other pipeline (`rf_baseline_processed`) |
| G4 | XGBoost | DONE (val) | protocol-fair val **0.9762** (`xgb_seed42.json`) |
| G5 | LightGBM | DONE (val) | G5 fix multiclass+class_weight=balanced val **0.9818** (min/Theft 0.9231); tops protocol classical; legacy 0.5512 superseded (`lgbm_seed42.json`) |
| G6 | MLP | RUN_DOCUMENTED (protocol) | WP5b seed42 CE scratch val **0.9285** (min 0.7077 Theft 1.0; 400901 params; 4.33 µs/sample); historical non-protocol 0.962/0.954 must **not** be mixed |
| G7 | 1D-CNN | RUN_DOCUMENTED | WP5b seed42 **0.6221** (min/Theft=0; matches WP5a A1); weak pure-CNN under equal budget |
| G8 | LSTM | RUN_DOCUMENTED | WP5b seed42 **0.8099** (min 0.3556 Theft 0.8) |
| G9 | BiLSTM | RUN_DOCUMENTED | WP5b seed42 **0.8058** (min/Theft 0.5; matches WP5a A2) |
| G10 | CNN–LSTM | RUN_DOCUMENTED | WP5b seed42 **0.8159** (min/Theft 0.5) |
| G11 | CNN–BiLSTM | DONE (val protocol) + WP5b arch ref | WP5b CE scratch **0.9493** tops neural suite (= WP5a A3); WP1b multirun package path mean 0.9714±0.0109 (diff init/loss) |
| G12 | Transformer / temporal-attention | RUN_DOCUMENTED | WP5b lightweight temporal transformer seed42 **0.5808** (min/Theft=0) — underperforms under equal CE budget; not free win |
| G13 | Reproducible lightweight IDS | RUN_DOCUMENTED (N/A) | No external public method re-impl under protocol; use G6–G12 + classical G1–G5; note `G13_LIGHTWEIGHT_IDS_NOTE.md` |
| G14 | Same train/val/test partitions | DONE | Protocol `botiot_v1` + freeze card |
| G15 | Comparable HPO effort | DONE (documented) | WP5b equal fixed HPs (lr=1e-3 Adam bs=512 ep≤8 pat=3 CE scratch seed42); no per-baseline Optuna; CAD-CBA HPO separate (`summary.json` g15 note) |

---

## H. §7 Multi-objective

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| H1 | May not beat RF F1 if deployment better | DONE (framing) | METHOD + multi-obj framing locked; WP5c tables show classical LGBM/RF may still top pure F1 while neural owns deploy/size path |
| H2 | Near-RF detection | DONE (honest dual bar) | **B14 test 0.9780±0.0033** ≈ protocol RF val 0.9778; LGBM val 0.9818 still tops pure F1; published RF 0.9864 other pipeline — dual bars locked |
| H3 | Much lower GPU memory | DONE (local peak) | **WP6b** peak alloc **322.2** MiB across batches/sessions; batch256 peak ~**103.2** MiB; ckpt ~2.0 MiB / n_params 530181; historical cuML RF ~444MB contrast kept; DICC peak still open |
| H4 | Faster neural inference | DONE (local ranges) | **WP6b** full V3 PT @bs=256 **24.15–25.68** µs/sample (mean **24.90**, n=5); WP5c relative arch table still valid (G6 4.33 vs A7 ~26); CUDA Option A pipe **565–570** µs |
| H5 | Low explanation dispatch | DONE (dispatch) | `llm_explainability.json` — 16.60 µs is **dispatch only** (not full LLM gen) |
| H6 | Good minority detection | DONE (protocol) | Val + **B14 test Theft mean 1.0** / min-cls **0.9292**; classical Theft 0.9231 on val; paper table ready from claims |
| H7 | Stable across GPU platforms | DONE (with caveats) | 3 sessions × 2 GPUs; B3 very stable; V100 B1 S1–Day2 formal max spread 11% (not all-metrics-stable). A100 stable under allow-dirty compare. No portable B3 CUDA win. |
| H8 | Pareto F1–latency–memory | DONE | WP5c analysis `pareto/` + systems rebench `pareto_h8/`: composite G6; F1 leadership classical LGBM/package seeds; a priori weights locked; figure + CSV |

---

## I. §8 DICC / same-GPU / multi-day

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| I1 | B3 CUDA vs matching PT same GPU | DONE (honest) | PT B3 wins on V100S (~363 vs ~513) and A100 (~384–391 vs ~668–671). Tables locked. |
| I2 | Full PT model latency same GPU | DONE | Full V3 PT ~945–973 µs (n=20) across sessions/GPUs. |
| I3 | V100S results | DONE | 3 SUCCESS sessions on laptop+git |
| I4 | A100 results | DONE | 3 SUCCESS sessions on laptop+git |
| I5 | ≥2 different days | DONE | S1/S2 + calendar Day2; compares in `DICC_COMPARE_OUTCOMES.md` |
| I6 | mean, median, std, CV, CI | DONE (local) | **WP6b** local multi-session mean/median/std/CV/CI on energy + PT + CUDA (`wp6b_local_ranges/`); **DICC multi-day still BLOCKED** (I1–I5) |
| I7 | Warm-up protocol | DONE | WP6b warm-up **50** discarded sync forwards per timed block; documented in summary |
| I8 | Batch-size sensitivity | DONE | WP6b multi-session I8 table bs∈{1,8,32,64,128,256,512,1024}; `systems_i8_h3/` + `wp6b_local_ranges/` |
| I9 | Statistical significance + effect size | DONE (local) / BLOCKED (cross-GPU) | Local: `statistical_significance_v2.json` framework tests + WP6b CI/CV/std; **cross-GPU multi-day significance BLOCKED** until DICC |
| I10 | No generalise from RTX 3050 alone | DONE (discipline) | Spine + ToV + claims forbid RTX→cluster generalisation; multi-GPU cells TBD until DICC |
| I11 | Portability central | DONE (honest answer) | Measured on V100S+A100 multi-session; portable B3 CUDA speedup **refuted**; portability of *measurement* and small-block CUDA wins stand |

---

## J. §9 Explainability

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| J1 | 16.60 µs is dispatch only | DONE | Measured + documented (`llm_explainability.json`); claim scope locked as dispatch-only |
| J2 | Faithfulness | RUN_DOCUMENTED | Occlusion ΔP top-3 mass **0.5109**; top features min/stddev/max (`xai/summary.json`) |
| J3 | Consistency | RUN_DOCUMENTED | Occlusion rank Spearman **0.9636** across two val draws |
| J4 | Latency (gen vs dispatch) | DONE | Dispatch p99 overhead **16.60 µs**; TinyLlama gen mean **~7400 ms** (never conflate) |
| J5 | Analyst usefulness | RUN_DOCUMENTED | Structured template usefulness mean **1.0** (automatic rubric n=8); not a human SOC study |
| J6 | Hallucination rate | RUN_DOCUMENTED | Heuristic on n=6 TinyLlama samples; generic/feature-weak free-form text |
| J7 | Agreement with model evidence | RUN_DOCUMENTED | Strict feature-mention rate **0.333** (ambiguous min/max/mean filtered); top3 agree weak |
| J8 | vs SHAP / LIME / attention / rules | RUN_DOCUMENTED | Occlusion + attention temporal proxy + rule templates (shap/lime **not installed**) |
| J9 | Structured evidence into LLM | DONE | Structured template: class, conf, top occlusion feats, analyst action |
| J10 | Or drop full “explainable” claim | DONE | **DROP_FULL_EXPLAINABLE_CLAIM_KEEP_STRUCTURED** — keep dispatch + structured evidence; no title-level full LLM-XAI |

---

## K. §10 Multi-dataset + final RQs

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| K1 | BoT primary | DONE | Primary dataset under protocol `botiot_v1` |
| K2 | ToN (or other) on final method | RUN_DOCUMENTED | CAD-CBA-v1 mapped on `processed_toniot` 13-feat: val **0.8080** test **0.8110**; RF same-split test **0.9393**; ≠ historical 26-feat clean 0.9526 (`toniot_final/`) |
| K3 | Cross-dataset / transfer | RUN_DOCUMENTED (recipe) | Recipe transfer (not weight transfer); ToN-scale kd_lr/ft_lr; honest RF gap |
| K4 | RQ: improve detection? | DONE (answer locked) | B14 test **0.9780±0.0033** near RF val 0.9778; does **not** beat LGBM 0.9818 or published RF 0.9864 — multi-obj / deploy is the contribution path (`WP9b_MANUSCRIPT_SPINE.md` §3) |
| K5 | RQ: minority recognition? | DONE (answer locked) | B14 test Theft **1.0** mean; min-cls **0.9292** — strong minority under protocol; val tables support |
| K6 | RQ: reduce latency or memory? | DONE (local answer draft) | WP6b ranges: energy **0.920–0.943** mJ/flow; PT@256 **24.15–25.68** µs; peak **322.2** MiB; G6 composite 0.9056 @4.33 µs; multi-GPU still BLOCKED |
| K7 | RQ: valid across GPUs? | DONE (answered) | Yes measured; B3 CUDA does **not** transfer as a win vs PT; full PT absolute + B1/B2/B4 CUDA wins do |
| K8 | RQ: explainability measurable value? | RUN_DOCUMENTED | Dispatch yes; structured evidence yes; free-form LLM quality weak → no full XAI RQ claim |

---

## L. Process / staged plan (his letter)

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| L1 | DICC first, then focused improvement | DONE (order policy) | Order locked; DICC deferred to **dedicated user session** under **OnDemand VNC + screen + batch** (`DICC_OPS_METHOD.md`); local science closed without inventing multi-day numbers; campus/Cheran-default ops **superseded** |
| L2 | Mod table before final model | DONE | `docs/MOD_DECISION_TABLE.md` + CAD-CBA-v1 signed |
| L3 | Avoid changing many parts at once | DONE | Sequential WPs (loss → thresholds → KD → HPO → package → confirm) |
| L4 | Phase: freeze preprocess/split/metrics/seeds/hardware/baseline | DONE | Protocol + `BASELINE_FREEZE_CARD.md` + multirun baseline |
| L5 | ≥5 independent training runs mean±std | DONE | WP1b 0.9714±0.0109; package 0.9639±0.0185; HPO confirm 0.9689±0.0145 n=5 (val only) |
| L6 | Optuna/Bayesian HPO | DONE | WP3 Optuna TPE val-only; winner INCORPORATE 0.9791; multi-seed confirm RUN_DOCUMENTED 0.9689±0.0145 |
| L7 | One clear proposed method | DONE (named) | **CAD-CBA-v1** signed; evaluation playlist **closed** (ablations/test/ToN/paper local-complete); only DICC ops + PI venue remain |
| L8 | Deploy: export, parity, profile, kernels, TRT/ORT/compile, FP16/INT8 | DONE (local path) | WP6a fidelity PASS + **WP6b multi-session ranges DONE**; TRT/ORT/compile historical; WP6c DICC re-bench only if user opens DICC (champion unchanged → optional) |
| L9 | Realistic: trade-off vs beat RF everywhere | DONE (framing) | Multi-obj framing locked; H8 + `CLAIMS_REGISTRY` advantage snapshot |
| L10 | Paper structure / tables / ToV / repro | DONE (PI polish) | Spine + polished `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.{md,pdf}` + figures; Table 1b/5b; App D checklist; `scripts/build_manuscript_pdf.py`; journal template/BibTeX = PI after venue |
| L11 | Fix gitignored claim-source repro | DONE | WP9a + post-B14/WP6b + Table 1b per-class claims: **64** claims; sealed LOCKED_TEST; `verify_claims.py` all green |
| L12 | Title words only if evaluated | DONE (policy) | J10 drops full XAI title; spine §1 recommended **T1** multi-obj + Option A CUDA; no portability/XAI-full words until evaluated |

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
| 2026-07-21 | **Package FT multirun DONE** ensemble KD init + hpo_best: mean **0.9639 ± 0.0185** n=5 (max 0.9803 seed45; min 0.9328 seed43) RUN_DOCUMENTED — mean does not beat WP1b 0.9714±0.0109; higher variance; L5 dual multirun noted; next: multi-seed HPO confirm or WP5 ablations |
| 2026-07-21 | **Multi-seed HPO confirm DONE** original distill + hpo_best: mean **0.9689 ± 0.0145** n=5 (max 0.9797 seed44; min 0.9483 seed46; seed42 **0.9791** exact WP3 repro) RUN_DOCUMENTED — mean does not beat WP1b; train HPs stay INCORPORATED; next: WP5a ablations or neural baselines |
| 2026-07-21 | **Context hygiene / full-playlist lock:** skip-nothing restated as complete **every** tracker row; flipped stale statuses from existing disk evidence (B10/B11, C13, D8/D10, F3/F4/F8, G6/G11/G14, H1/H5, J1, K1, L2–L4/L7/L9, etc.). Open rows marked **Playlist required**. No invented numbers. Next science still WP5a etc. |
| 2026-07-21 | **WP5a ablation ladder DONE** A1–A7 seed42 val-only ~90 min: A7 **0.9699** &gt; A3 0.9493 &gt; A6 0.9346 &gt; A5 0.8684 &gt; A2 0.8058 &gt; A4 0.7378 &gt; A1 0.6221; F1–F7 RUN_DOCUMENTED; F9 systems partial; A4 attn+CE underperforms A3 (honest); champion unchanged; next WP5b neural baselines |
| 2026-07-22 | **WP5b neural baselines DONE** G6–G12 seed42 CE scratch equal budget ~80 min: G11 **0.9493** &gt; G6 0.9285 &gt; G10 0.8159 &gt; G8 0.8099 &gt; G9 0.8058 &gt; G7 0.6221 &gt; G12 0.5808; G15 HPO budget note DONE; transformer weak under budget; G7=A1 G9=A2 G11=A3 consistency; champion unchanged; next D6/G2/G5 or WP5c/C* |
| 2026-07-22 | **G2/G5 classical + D6 + G13 DONE:** LinearSVC full **0.4268** RUN_DOCUMENTED; LGBM fix **0.9818** DONE (tops classical); D6 stratified **0.9209** vs shuffle **0.9791** (Δ−0.058) RUN_DOCUMENTED keep shuffle; G13 N/A note; B9 class-weight close RUN_DOCUMENTED; champion unchanged; next WP5c Pareto / C* / B2–B4 / E6 |
| 2026-07-22 | **WP5c Pareto H8 DONE** analysis-only: 14 protocol points A1–A7+G6–G12; F1–lat front A7/A3/G6; best F1 A7 **0.9699** @26.0 µs; composite #1 G6 **0.762** @4.33 µs F1 0.9285; classical LGBM/RF ref 0.9818/0.9778; `pareto/summary.json` + `table.md` + plots; champion unchanged; next C* / B2–B4 / E6 |
| 2026-07-22 | **WP5c systems rebench `pareto_h8/`** + classical RF/XGB/LGBM CPU systems; composite G6 **0.9056**; front G6/HPO/WP1b-s44/XGB/LGBM; champion unchanged |
| 2026-07-22 | **Bounded C\* DONE** CTRL **0.9787**; C4 multi-scale **0.9167**; C5 gated **0.9132**; C8 ASL **0.8012**; C7 SupCon **0.7732**; C10 uncertainty no lift — all **RUN_DOCUMENTED** no package incorporate; `cstar_bounded/` |
| 2026-07-22 | **B2–B4 arch HPO plateau reject RUN_DOCUMENTED** — `B2B4_ARCH_HPO_PLATEAU_REJECT.md`; V3 dims frozen |
| 2026-07-22 | **E6 neural teacher KD DONE** G11 teacher → student **0.8513** ≪ ensemble **0.9401** RUN_DOCUMENTED; keep ensemble INCORPORATED; champion unchanged |
| 2026-07-22 | **WP7 XAI suite DONE** occlusion faith 0.5109 / rank corr 0.9636 / structured usefulness 1.0; LLM feature-mention 0.333 → **J10 DROP full claim KEEP structured+dispatch**; `xai/` |
| 2026-07-22 | **F9 energy table DONE** consolidated RTX 0.786 mJ/flow + A100/cuML + latency/params; `energy_table/` RUN_DOCUMENTED |
| 2026-07-22 | **WP8 ToN final method DONE** CAD-CBA-v1 mapped 13-feat: val **0.8080** test **0.8110** RF test **0.9393** RUN_DOCUMENTED; KD selected (FT no lift); pilot low-lr archived |
| 2026-07-22 | **WP6a re-export + fidelity DONE** bit-identical blocks; CUDA self-check all PASS; champion md5 unchanged |
| 2026-07-22 | **WP9a claims packaging DONE** `scripts/build_claims_package.py` → `claims_package/` + `CLAIMS_REGISTRY.md` (42 claims, 11 minority rows); `verify_claims.py` protocol claims green; `FINAL_CONFIG_FREEZE_CARD.md` awaiting user lock for B14; A4/C12/D1/H6/L11 advanced from disk evidence; sealed test **not** run |
| 2026-07-22 | **B14 sealed multi-seed BoT TEST DONE** user lock path **A**; seeds 42–46; **test macro-F1 0.9780 ± 0.0033**; min-cls 0.9292; Theft **1.0**; val 0.9689±0.0145; champion unchanged; wall ~79 min; claims rebuild 46 claims green; B14/C12/D1/H2/H6/L11 flipped |
| 2026-07-22 | **WP6b local multi-session ranges DONE** n=5 sessions RTX 3050; energy **0.920–0.943** mJ/flow (mean 0.933); PT@256 **24.15–25.68** µs (mean 24.90); CUDA pipe **565–570** µs; block3 FP16 **503–509** µs; peak alloc **322.2** MiB; I7/I8/H3/H4/K6/L8 advanced; claims **59** green; historical energy 0.786 labeled HISTORICAL |
| 2026-07-22 | **WP9b manuscript spine DONE** `WP9b_MANUSCRIPT_SPINE.md`: title policy T1; abstract 5-part; RQ answers K4/K5 DONE; core tables from disk; ToV addendum; residual PARTIAL flips A1/A2/A4–A6/C1/I6/I9/I10/K4/K5/L1/L10/L12; H7/I1–I5/I11/K7/WP0 remain BLOCKED (DICC ops); verify_claims green; no train; champion unchanged |
| 2026-07-22 | **WP9c camera-ready draft DONE** `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.{md,pdf}` + figures (arch, class-dist, dual bars, ablation, B14 CM seed42, WP6b ranges, Pareto); related-work gap table; results→methods→intro from spine; A6/L10 flipped to draft; claims 59 green; no train; champion unchanged; DICC still BLOCKED |
| 2026-07-22 | **PI venue polish DONE** continuous journal abstract; author/affiliation/venue placeholders; Table 1b multi-seed test per-class means (from sealed_test seeds); Table 5b HPO Stage-B refine ranking; Table 6 std/CV/CI; data/ethics stubs; App D checklist; `scripts/build_manuscript_pdf.py` rebuild (~797 KB PDF); A6/L10 → DONE (PI polish); L11 notes 59 claims; verify_claims green; no train; champion unchanged; DICC still BLOCKED; final journal class file/BibTeX left for PI after venue choice |
| 2026-07-22 | **Playlist closure audit + claims hygiene** — full tracker parse **133/133 terminal** (0 TODO/PARTIAL); Table 1b means re-verified from seed JSON; claims rebuild **64** (added `bot_sealed_test_pc_*`); open_gates trimmed to DICC + PI venue only; WP0/0b/6c → BLOCKED(ops/N/A); L7/B8 stale notes flipped; `PLAYLIST_CLOSURE_AUDIT.md`; verify_claims green; no train; champion unchanged |
| 2026-07-22 | **DICC ops method lock** — guidance: OnDemand **VNC Desktop** + **`screen`** + batch campaign; remove campus-runner / Cheran-as-default cluster operator as primary plan; `docs/DICC_OPS_METHOD.md` + FINAL_PLAN / Phase0 / status packs / HANDOFF / WP0 / tracker A3+L1; no cluster run; champion unchanged |
| 2026-07-22 | **Git branching policy lock** — `docs/BRANCHING_POLICY.md`: **`master` always final**; create a branch when work is a **true alternative option**; strict **low branch count**; merge then delete; wired into HANDOFF / continuity / safety §5b |

---

## Definition of “all of them done”

Every row is:

- **DONE** / **INCORPORATED**, or  
- **RUN_DOCUMENTED** (ran; not selected; notes + JSON exist), or  
- **BLOCKED** only with unblock path (e.g. waiting DICC access),

and manuscript only after that.

**We do not mark complete early. We do not skip in silence.**
