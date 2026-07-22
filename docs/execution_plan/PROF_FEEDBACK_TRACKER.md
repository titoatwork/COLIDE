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
| B2 | CNN layers and filters | TODO | **Still required (playlist):** bounded WP2c arch search or explicit RUN_DOCUMENTED reject after package plateau test; currently frozen for KD weight transfer |
| B3 | Convolution kernel sizes | TODO | **Still required:** same as B2; padding=1 k=3 frozen interim only |
| B4 | BiLSTM hidden dims and layers | TODO | **Still required:** same as B2 |
| B5 | Dropout rate | DONE (train) | WP3 searched dropout 0.10–0.50 + attention_dropout 0–0.30; winner dropout≈0.148 att≈0.214 |
| B6 | LR and LR scheduler | DONE | WP3: lr log 1e-5–3e-3; scheduler {none,cosine,step}; winner lr≈5.89e-5 cosine |
| B7 | Batch size | DONE | WP3 categorical {128,256,512,1024}; winner **1024** |
| B8 | Focal-loss parameters | DONE (γ) | WP3 γ∈[0.5,3.5]; winner **≈1.917** (α class-weights still open B9) |
| B9 | Class weights | RUN_DOCUMENTED | Explicit CB via `focal_cb` val **0.9121** ≪ plain focal **0.9780** (`imbalance_loss/`); keep **no class-weight** on neural focal. (Classical LGBM G5 uses balanced — separate tree baseline, not CAD-CBA loss) |
| B10 | Distill T and α | DONE (recipe) + historical RUN_DOCUMENTED | **17** historical `distill_botiot_a*_T*.json` sweeps; CAD-CBA-v1 uses **α=0.6 T=10** (best historical + WP4b); protocol Optuna on T/α not required if recipe locked |
| B11 | Sequence length | DONE (locked) | V3 reshape fixed **[2,32]** in freeze/config; design freeze (not a free search dim under CAD-CBA-v1 KD transfer) |
| B12 | Decision thresholds minority | RUN_DOCUMENTED | WP2d on `ft_focal_seed42`: all decode variants = argmax val macro 0.9780 (Δ0); keep argmax (`thresholds_focal_seed42.json`) |
| B13 | Objectives: val macro-F1, bal-acc, minority recall | DONE (logged) | Primary max val macro-F1; min-cls / bal-acc / Theft logged per trial |
| B14 | Test untouched until final config | PARTIAL | WP3 + package + HPO multi-seed confirm test SEALED; sealed multi-seed **test** of final lock still TODO |

---

## C. §2 Novel method contribution

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| C1 | Not only “standard CNN–BiLSTM stack” | PARTIAL | Ladder shows package A7 **0.9699** &gt; plain A3 **0.9493**; novelty = composition (attn+focal+ens KD+HPO), not attn alone (A4 hurts) |
| C2 | Class-aware attention after BiLSTM | RUN_DOCUMENTED (caveat) | V3 has attention; WP5a A4 attn+CE **0.7378** &lt; A3 **0.9493** under seed42/8-ep — do not claim attn alone helps; credit is package-level |
| C3 | Lightweight temporal attention | RUN_DOCUMENTED (caveat) | Same as C2; systems: attn models ~26 µs/sample vs A3 ~20 µs (batch256) |
| C4 | Multi-scale temporal convolution | TODO | **Playlist required:** bounded experiment or RUN_DOCUMENTED reject after try |
| C5 | Gated CNN–BiLSTM fusion | TODO | **Playlist required:** bounded experiment or RUN_DOCUMENTED reject after try |
| C6 | Class-balanced or logit-adjusted loss | RUN_DOCUMENTED | 4-way FT compare: focal best; focal_cb/logit_adj worse macro (`imbalance_loss/`) |
| C7 | Supervised contrastive (minority) | TODO | **Playlist required:** bounded run + JSON even if not selected |
| C8 | Asymmetric loss minority | TODO | **Playlist required:** bounded run + JSON even if not selected |
| C9 | Teacher ensemble distillation | INCORPORATED | WP4b: ensemble student val **0.9401** best among 5 teachers (`teachers_kd/`) |
| C10 | Uncertainty-aware detection | TODO | **Playlist required:** bounded run + JSON even if not selected |
| C11 | Adaptive per-class thresholds | RUN_DOCUMENTED | Val grid search macro/min/joint/Theft/Normal — no lift vs argmax on focal FT; not in package |
| C12 | Component addresses named weakness | PARTIAL | Named weakness = imbalance/Theft; package uses focal+ensemble KD; **ablation/minority tables still open** |
| C13 | Mod comparison table before lock | DONE | `docs/MOD_DECISION_TABLE.md` + `METHOD_PACKAGE_DECISION.md` CAD-CBA-v1 |

**Rule:** Not every C2–C11 is mandatory; **at least one clear package** must be fully evaluated. Tracker keeps all options until chosen/rejected in writing.

---

## D. §3 Class imbalance

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| D1 | Imbalance first-class (not accuracy-only) | PARTIAL | Metrics + loss + thresholds + D6 stratified done; minority tables / final claims still open |
| D2 | Weighted CE | RUN_DOCUMENTED | CE FT control val 0.9755 (`imbalance_loss/ft_ce_seed42.json`) |
| D3 | Focal loss | INCORPORATED | Best in 4-way compare val **0.9780** — default CAD-CBA-v1 |
| D4 | Class-balanced focal | RUN_DOCUMENTED | val 0.9121 — worse macro; JSON kept |
| D5 | Logit adjustment | RUN_DOCUMENTED | val 0.9225 — worse macro; JSON kept |
| D6 | Stratified batch sampling | RUN_DOCUMENTED | WP D6 seed42: shuffle **0.9791** &gt; stratified inv-freq **0.9209** (Δ−0.058); keep shuffle (`stratified_batch/summary.json`) |
| D7 | Minority-aware threshold tuning | RUN_DOCUMENTED | WP2d DONE: macro/min/joint/class_f1 on focal seed42; Δmacro=0 Δmin=0 vs argmax; keep argmax |
| D8 | Controlled oversampling | DONE (stage_a) | Protocol `stage_a_kd` SMOTE targets freeze card; stage_b_ft deliberately no SMOTE |
| D9 | Supervised contrastive | TODO | **Playlist required:** bounded run + JSON (same as C7) |
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
| E6 | Strong neural teacher | TODO | **Playlist required:** bounded neural-teacher KD or RUN_DOCUMENTED reject after try |
| E7 | Heterogeneous ensembles | INCORPORATED | RF+XGB+LGBM heterogeneous mean (WP4b) |
| E8 | Student not mere imitation — deployment trade-off | PARTIAL | WP4b: XGB teacher 0.9918 → student 0.9270 &lt; ensemble student 0.9401 (not pure imitation); multi-obj table still open |
| E9 | Lower memory / latency / GPU deploy / temporal | PARTIAL | WP5c consolidated table (`pareto/`); A7 tops F1, G6 wins composite/latency; historical cuML ~2MB vs 444MB; deploy CUDA blocks still separate |

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
| F9 | Latency, params, memory, energy per ablation | PARTIAL | WP5a logs params + CUDA latency; WP5c consolidates F1–lat–params Pareto; **full energy table still open** |

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
| H2 | Near-RF detection | PARTIAL | Protocol neural HPO 0.9791 / multirun mean 0.9714 vs protocol RF 0.9778 / published RF 0.9864 — keep honest dual bars |
| H3 | Much lower GPU memory | PARTIAL | Historical `cuml_rf_resources.json` CNN ~2MB vs cuML RF ~444MB; protocol proxy via n_params/ckpt bytes in WP5c (`pareto/`); peak VRAM re-measure still open |
| H4 | Faster neural inference | PARTIAL | Protocol batch256 µs/sample in WP5c: G6 MLP **4.33** vs A7 **26.02** vs A3 **19.96**; historical `baseline_latency.json` still secondary |
| H5 | Low explanation dispatch | DONE (dispatch) | `llm_explainability.json` — 16.60 µs is **dispatch only** (not full LLM gen) |
| H6 | Good minority detection | PARTIAL | Theft/min-cls logged (HPO seed42 Theft=1.0); systematic minority claim needs ablation/final tables |
| H7 | Stable across GPU platforms | BLOCKED | DICC |
| H8 | Pareto F1–latency–memory | DONE | WP5c: 14 protocol points (A1–A7 + G6–G12); F1–lat front **A7/A3/G6**; composite #1 G6 0.762; best F1 A7 **0.9699**; classical LGBM/RF ref 0.9818/0.9778; `pareto/summary.json` + plots |

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
| J1 | 16.60 µs is dispatch only | DONE | Measured + documented (`llm_explainability.json`); claim scope locked as dispatch-only |
| J2 | Faithfulness | TODO | **Playlist required** (or J10 drop path) |
| J3 | Consistency | TODO | **Playlist required** (or J10 drop path) |
| J4 | Latency (gen vs dispatch) | PARTIAL | Dispatch done; full generation latency still open |
| J5 | Analyst usefulness | TODO | **Playlist required** (or J10 drop path) |
| J6 | Hallucination rate | TODO | **Playlist required** (or J10 drop path) |
| J7 | Agreement with model evidence | TODO | **Playlist required** (or J10 drop path) |
| J8 | vs SHAP / LIME / attention / rules | TODO | **Playlist required** (or J10 drop path) |
| J9 | Structured evidence into LLM | PARTIAL | Features in prompts (existing pipeline) |
| J10 | Or drop full “explainable” claim | TODO | **Must decide after J2–J9 attempt or explicit drop** |

---

## K. §10 Multi-dataset + final RQs

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| K1 | BoT primary | DONE | Primary dataset under protocol `botiot_v1` |
| K2 | ToN (or other) on final method | PARTIAL | Historical `distill_toniot*.json` etc.; **final CAD-CBA-v1 method re-eval still required** |
| K3 | Cross-dataset / transfer | TODO | **Playlist required** after K2 |
| K4 | RQ: improve detection? | TODO | Answer after final tables |
| K5 | RQ: minority recognition? | TODO | Answer after final tables |
| K6 | RQ: reduce latency or memory? | PARTIAL | Local evidence exists; final composite still open |
| K7 | RQ: valid across GPUs? | BLOCKED | DICC |
| K8 | RQ: explainability measurable value? | TODO | After J* or J10 drop |

---

## L. Process / staged plan (his letter)

| ID | Requirement | Status | Evidence / notes |
|----|-------------|--------|------------------|
| L1 | DICC first, then focused improvement | PARTIAL | Order locked; DICC deferred to dedicated session (user); local science continues under Option A |
| L2 | Mod table before final model | DONE | `docs/MOD_DECISION_TABLE.md` + CAD-CBA-v1 signed |
| L3 | Avoid changing many parts at once | DONE | Sequential WPs (loss → thresholds → KD → HPO → package → confirm) |
| L4 | Phase: freeze preprocess/split/metrics/seeds/hardware/baseline | DONE | Protocol + `BASELINE_FREEZE_CARD.md` + multirun baseline |
| L5 | ≥5 independent training runs mean±std | DONE | WP1b 0.9714±0.0109; package 0.9639±0.0185; HPO confirm 0.9689±0.0145 n=5 (val only) |
| L6 | Optuna/Bayesian HPO | DONE | WP3 Optuna TPE val-only; winner INCORPORATE 0.9791; multi-seed confirm RUN_DOCUMENTED 0.9689±0.0145 |
| L7 | One clear proposed method | DONE (named) | **CAD-CBA-v1** signed; full evaluation playlist still open (ablations/test/ToN/paper) |
| L8 | Deploy: export, parity, profile, kernels, TRT/ORT/compile, FP16/INT8 | PARTIAL | Local CUDA/kernels/`tensorrt_native.json`/`torch_compile_native.json` exist; final re-export after lock open |
| L9 | Realistic: trade-off vs beat RF everywhere | DONE (framing) | Multi-obj framing locked; need Pareto tables (H8) |
| L10 | Paper structure / tables / ToV / repro | TODO | Outline in plan pack; write after science green |
| L11 | Fix gitignored claim-source repro | TODO | Manifest + claim scripts; finish before paper |
| L12 | Title words only if evaluated | TODO | Discipline until J/XAI/CUDA claims closed |

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

---

## Definition of “all of them done”

Every row is:

- **DONE** / **INCORPORATED**, or  
- **RUN_DOCUMENTED** (ran; not selected; notes + JSON exist), or  
- **BLOCKED** only with unblock path (e.g. waiting DICC access),

and manuscript only after that.

**We do not mark complete early. We do not skip in silence.**
