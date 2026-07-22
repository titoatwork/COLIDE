# CAD-CBA: A Class-Aware Distilled CNN–BiLSTM for Multi-Objective IoT Intrusion Detection with Operation-Matched CUDA Acceleration

**Manuscript draft status:** Camera-ready **local-complete** draft (WP9c writing pass)  
**Method freeze:** CAD-CBA-v1 · **Option A** CUDA (per-block / operation-matched only)  
**Claims:** 59 protocol claims; `PYTHONPATH=. python3 scripts/verify_claims.py` must stay green  
**Champion md5:** `80a90f7cc210276300eaa90173a5a385` (do not clobber)  
**Authority numbers:** `docs/execution_plan/CLAIMS_REGISTRY.md` + `benchmarks/results/**`  
**Open ops:** DICC multi-GPU / multi-day cells remain **TBD** (never invent)

> Writing order used: results tables locked first (spine §8) → methods from freeze card → intro/abstract last.  
> This file is the PI-facing full prose draft. Figures live in `docs/manuscript/figures/`.

---

## Abstract

**(1) Background/problem.**  
IoT intrusion detection must handle extreme class imbalance and low-latency edge constraints. Tree ensembles often top pure accuracy on static tabular flows, while deep models are preferred for GPU deployment, incremental adaptation, and systems co-design — yet unfair protocols and framework-only acceleration claims obscure real trade-offs.

**(2) Gap.**  
Prior work often siloes detection quality from deployment metrics, tunes without a sealed test, or reports single-shot laptop latencies as multi-platform facts. Full-pipeline “Custom CUDA vs full model” claims are invalid when kernels implement only operation-matched blocks (**Option A**).

**(3) Method — CAD-CBA-v1 only.**  
We freeze **CAD-CBA-v1**: V3 CNN–BiLSTM–Attention; **ensemble** knowledge distillation (mean RF+XGB+LGBM soft labels, α=**0.6**, T=**10**); **focal** loss (γ≈**1.92**); Optuna train HPs (`config/hpo_best.yaml`); **shuffle** sampler; **argmax** decode. We evaluate under protocol `botiot_v1` with fair classical and neural baselines, a full ablation ladder, multi-objective Pareto analysis, local multi-session systems ranges (Option A CUDA), structured evidence + dispatch timing, and a ToN recipe-transfer honesty check.

**(4) Results (locked).**  
- BoT **sealed multi-seed TEST** (n=5, path A): macro-F1 **0.9780±0.0033**; min-cls **0.9292**; Theft **1.0000**.  
- Protocol classical pure-F1 ceiling: LGBM **0.9818** (val); RF **0.9778** (val); published RF **0.9864** is a **different pipeline** dual bar only.  
- Package ladder A7 **0.9699** (seed42) tops the incremental path; attention+CE alone **hurts** vs plain CNN–BiLSTM (A4 **0.7378** ≪ A3 **0.9493**).  
- Multi-obj: a priori composite #1 **G6 MLP 0.9056** @ **4.33** µs (F1 0.9285).  
- Local systems (WP6b, n=5 sessions, RTX 3050): energy **0.920–0.943** mJ/flow; PT@256 **24.15–25.68** µs/sample; CUDA derived pipeline **565–570** µs (Option A); peak alloc **322.2** MiB.  
- XAI: dispatch p99 **16.60** µs; rank corr **0.9636**; faith mass **0.5109**; free-form LLM feature-mention **0.333** → **no** full explainable claim.  
- ToN (13-feat): test **0.8110** vs same-split RF **0.9393** (honest gap).

**(5) Contribution.**  
Under a sealed, protocol-fair evaluation, CAD-CBA-v1 delivers **near-RF multi-seed test detection** with **strong minority (Theft) recognition**, while the primary scientific claim is a **valid accuracy–efficiency multi-objective story** with **operation-matched CUDA** and **local multi-session latency/energy ranges** — not pure-F1 supremacy over LightGBM and not multi-GPU portability until DICC completes.

**Keywords:** IoT intrusion detection; class imbalance; knowledge distillation; multi-objective evaluation; CUDA acceleration; BoT-IoT; protocol-fair baselines.

---

## 1. Introduction

### 1.1 Motivation

Edge and campus IoT deployments need detectors that (i) remain robust under severe class imbalance, (ii) support GPU inference with predictable latency and energy, and (iii) admit honest comparison against strong classical baselines on the **same** data protocol. Much of the literature optimises one axis (pure F1, or single-shot latency) and then generalises beyond the measured hardware and training recipe.

### 1.2 Research questions

| RQ | Question | Locked answer (this paper) |
|----|----------|----------------------------|
| **RQ1** | Does CAD-CBA improve detection under protocol-fair evaluation? | Sealed multi-seed **test** macro-F1 **0.9780±0.0033** ≈ protocol RF val **0.9778**; does **not** beat protocol LGBM val **0.9818** or published RF **0.9864** (other pipeline). Detection alone is not the sole headline. |
| **RQ2** | Does it improve minority recognition? | **Yes on protocol test:** Theft mean **1.0**, min-cls mean **0.9292** (n=5). |
| **RQ3** | Does it reduce latency or memory on the measured GPU? | **Yes on local RTX 3050:** PT@256 **24.15–25.68** µs/sample; energy **0.920–0.943** mJ/flow; peak alloc **322.2** MiB; multi-obj composite G6 **0.9056** @4.33 µs. |
| **RQ4** | Are systems results valid across GPUs / multi-day? | **Not yet claimable** — local ranges ≠ multi-day V100S/A100 (DICC open). |
| **RQ5** | Does explainability add measurable value? | **Dispatch + structured evidence yes; free-form LLM quality no** (J10 drop full claim). |

### 1.3 Contributions

1. **CAD-CBA-v1 method package** fully frozen and evaluated: ensemble KD + focal + Optuna train HPs + V3 backbone under protocol `botiot_v1`, with sealed multi-seed BoT **test**.  
2. **Protocol-fair classical and neural baselines** on the same splits, dual-bar honesty vs published RF.  
3. **Ablation + negative results** showing package-level credit (A7) and rejecting attention-alone, multi-scale CNN, gated fusion, SupCon, ASL, neural-teacher KD, and stratified batching.  
4. **Multi-objective Pareto** and **local multi-session** latency/energy/VRAM ranges under Option A CUDA discipline.  
5. **Scoped XAI**: low-overhead dispatch + structured evidence; no full “LLM-explainable IDS” title claim.  
6. **ToN recipe-transfer honesty** on 13-feat processed data with same-split RF gap disclosed.

### 1.4 Non-claims (explicit)

- We do **not** claim pure-F1 supremacy over protocol LightGBM.  
- We do **not** claim full Custom-CUDA vs full V3 PyTorch **parity** (Option A).  
- We do **not** claim multi-GPU / multi-day portability until DICC artifacts exist.  
- We do **not** claim full free-form LLM explainability (feature-mention **0.333**).

---

## 2. Related work

### 2.1 Deep and hybrid IDS for IoT / tabular flows

CNN, LSTM/BiLSTM, and hybrid CNN–RNN stacks are widely applied to flow and packet features. Many report high accuracy on public datasets but omit sealed-test discipline, protocol-matched classical baselines, or multi-seed stability. Transformer-style temporal models are sometimes proposed as drop-in upgrades; under our equal-budget protocol baseline they underperform (G12 val macro-F1 **0.5808** ≪ G11 **0.9493**).

### 2.2 Class imbalance in intrusion detection

Focal loss, class-balanced losses, oversampling (e.g. SMOTE), and threshold search are common. Our four-way loss compare under protocol FT finds **focal** best (val **0.9780**); class-balanced focal (**0.9121**) and logit adjustment (**0.9225**) hurt macro-F1. Val threshold search does not beat **argmax**. Stage-A SMOTE remains a KD-path tool; Stage-B fine-tune uses real train only.

### 2.3 Knowledge distillation and teacher ensembles

Distilling tree ensembles into compact neural students is attractive for deployment. We compare RF, XGB, LGBM, none, and **ensemble** soft-label teachers under identical Stage-A KD; ensemble student val **0.9401** wins. A strong neural teacher (G11) yields a weaker student (**0.8513**).

### 2.4 GPU acceleration and systems claims

Framework compilers (TensorRT, ORT, `torch.compile`) and hand-written CUDA kernels both appear in IDS systems papers. Invalid comparisons arise when custom kernels cover only operation-matched blocks but are reported as full-model speedups. We adopt **Option A**: report per-block / derived pipeline CUDA latency and full V3 PyTorch absolute latency **separately**, never as parity.

### 2.5 Explainable IDS and LLMs

Recent LLM-assisted IDS work often emphasises narrative quality without faithfulness/consistency metrics, or conflates prompt-dispatch cost with generation latency. We measure occlusion faithfulness (**0.5109** top-3 mass), rank consistency (**0.9636**), dispatch p99 (**16.60** µs), generation mean (**~7400** ms), and free-form feature-mention (**0.333**), then **drop** a full explainable title claim while retaining structured evidence + dispatch.

### 2.6 Gap table (positioning)

| Gap in prior practice | Our response | Evidence |
|----------------------|--------------|----------|
| Test used during selection | Sealed test until B14 lock | `FINAL_CONFIG_FREEZE_CARD.md`, `sealed_test/` |
| Classical baselines on other pipelines | Protocol-fair RF/XGB/LGBM/SVM/LR | `baselines_classical/` |
| Single-shot laptop latency as “the” number | Multi-session ranges n=5 | `wp6b_local_ranges/` |
| Full-pipeline CUDA vs full PT false parity | Option A language + fidelity | `numerical_fidelity.json` |
| Architecture novelty without ablation | A1–A7 + C* negatives | `ablation_ladder/`, `cstar_bounded/` |
| Full LLM-XAI claim without metrics | J10 DROP_FULL; keep structured | `xai/summary.json` |
| Cross-GPU claim from one laptop | Explicit TBD cells | DICC BLOCKED |

---

## 3. Method: CAD-CBA-v1

### 3.1 Architecture (V3)

Input: 10 BoT-IoT flow features → linear projection to 64 → reshape **[2, 32]** → two 1D-CNN layers (64→128, kernel 3, BN, ReLU, pool) → BiLSTM (128→64, bidirectional) → multi-head attention (4 heads) → dense 64 + dropout → 5-class softmax. Trainable parameters of the production package path: **530,181** (Pareto/systems tables).

![Architecture](figures/fig_architecture.png)

**Figure 1.** CAD-CBA-v1 architecture and training stages (dims from `config/config.yaml`; train HPs from `config/hpo_best.yaml`).

### 3.2 Training recipe (frozen)

| Stage | Role | Key settings |
|-------|------|--------------|
| **Stage A KD** | Distill student from soft labels | Teacher = mean(RF, XGB, LGBM) probs; α=**0.6**, T=**10**; SMOTE train targets per freeze card; val for selection |
| **Stage B FT** | Real-data fine-tune | **No** SMOTE; **focal** γ≈**1.9166**; lr≈**5.893e-5**; batch **1024**; dropout≈**0.148**; attn dropout≈**0.214**; weight decay≈**1.92e-4**; **cosine** schedule; **shuffle** sampler |
| **Decode** | Inference decision | **argmax** (val threshold search: no macro/min lift) |

Init path for sealed multi-seed test (**path A**):  
`model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` + Stage-B FT with `hpo_best.yaml`, seeds 42–46.

### 3.3 Production champion (systems / XAI path)

Production weights used for systems ranges, energy historical table, and XAI suite:

- Path: `model/best_model_botiot_twostage.pth`  
- md5: **`80a90f7cc210276300eaa90173a5a385`**  
- Never overwritten without BACKUP + explicit approval.

### 3.4 What is *not* in the package

Multi-scale CNN (C4 **0.9167**), gated fusion (C5 **0.9132**), SupCon (C7 **0.7732**), asymmetric loss (C8 **0.8012**), MC-dropout selective (C10 no high-coverage lift), neural teacher KD (E6 **0.8513**), full arch Optuna (B2–B4 plateau reject), CB class weights on neural focal, stratified inv-freq batching (D6 **0.9209** ≪ shuffle **0.9791**).

---

## 4. Experimental protocol

### 4.1 Dataset and splits (BoT-IoT)

Protocol ID: **`botiot_v1`**. Features: 10 numerical flow features (freeze list in `scripts/protocol/botiot.py`). Classes: DDoS, DoS, Normal, Reconnaissance, Theft. Official test CSV is **sealed** for all model selection (HPO, thresholds, early stopping). Stage A uses stratified val + SMOTE on train only; Stage B uses stratified val + **no** SMOTE.

![Class distribution](figures/fig_class_distribution.png)

**Figure 2.** Extreme train imbalance and official test support (log scale). Theft test support is only **14** — report minority metrics with that honesty.

### 4.2 Metrics

Primary: **macro-F1**. Always log: balanced accuracy, weighted F1, min per-class F1, per-class F1 (especially Theft), confusion matrices. Systems: µs/sample (multi-session), mJ/flow, peak allocated VRAM, n_params, Option A CUDA block/pipeline µs.

### 4.3 Seeds and hardware

- Multiruns / sealed test seeds: **42–46** (n=5).  
- Local systems: NVIDIA GeForce **RTX 3050 6GB** Laptop GPU; WP6b **5** sessions; warm-up **50** sync forwards.  
- DICC V100S/A100 multi-day: **not** in this draft’s numeric tables.

### 4.4 Baselines (same protocol)

**Classical:** LR, LinearSVC, RF, XGB, LGBM (G5 balanced multiclass).  
**Neural (equal fixed CE budget, seed42):** MLP, 1D-CNN, LSTM, BiLSTM, CNN–LSTM, CNN–BiLSTM, lightweight temporal transformer.  
**Published RF 0.9864:** dual bar only (different pipeline) — never mixed as protocol-fair.

---

## 5. Results

### 5.1 Overall detection — sealed multi-seed TEST (B14)

**Table 1.** Sealed multi-seed BoT TEST (CAD-CBA-v1 path A; seeds 42–46). Source: `benchmarks/results/sealed_test/summary.json`.

| Seed | val macro-F1 | test macro-F1 | test min-cls | test Theft |
|------|--------------|---------------|--------------|------------|
| 42 | 0.9791 | 0.9787 | 0.9333 | 1.0000 |
| 43 | 0.9587 | 0.9798 | 0.9369 | 1.0000 |
| 44 | 0.9797 | 0.9798 | 0.9375 | 1.0000 |
| 45 | 0.9787 | 0.9722 | 0.9014 | 1.0000 |
| 46 | 0.9483 | 0.9796 | 0.9369 | 1.0000 |
| **Mean±std** | 0.9689±0.0145 | **0.9780±0.0033** | **0.9292** | **1.0000** |

**Honesty notes.** Seed 46 has weaker **val** (0.9483) but strong **test** (0.9796) — both reported. Do not replace test mean with val multirun **0.9714±0.0109**. Champion production file md5 **unchanged**.

![Detection dual bars](figures/fig_detection_dual_bars.png)

**Figure 3.** Dual accuracy bars: sealed test vs protocol classical val vs published RF (other pipeline).

![Confusion matrix](figures/fig_confusion_matrix_b14_seed42.png)

**Figure 4.** Representative sealed-test confusion matrix (seed 42; test macro-F1 0.9787). Multi-seed mean remains **0.9780±0.0033**.

### 5.2 Protocol-fair classical baselines (val)

**Table 2.** Classical baselines under `botiot_v1` (val; test sealed during selection).

| Model | val macro-F1 | Notes |
|-------|--------------|-------|
| LGBM (balanced) | **0.9818** | Protocol pure-F1 ceiling |
| RF | 0.9778 | Protocol-fair |
| XGB | 0.9762 | |
| LR | 0.5231 | |
| LinearSVC | 0.4268 | Weak under extreme imbalance |
| Published RF | 0.9864 | **Different pipeline** — dual bar only |

**RQ1 answer:** CAD-CBA sealed test **0.9780±0.0033** is near protocol RF and below protocol LGBM pure F1. The paper’s primary contribution is therefore **not** “we beat every classical model on F1 alone.”

### 5.3 Neural baselines (val, equal budget)

**Table 3.** WP5b neural baselines (seed42, CE scratch, equal fixed HPs).

| Model | val macro-F1 |
|-------|--------------|
| G11 CNN–BiLSTM | **0.9493** |
| G6 MLP | 0.9285 |
| G10 CNN–LSTM | 0.8159 |
| G8 LSTM | 0.8099 |
| G9 BiLSTM | 0.8058 |
| G7 1D-CNN | 0.6221 |
| G12 Transformer | **0.5808** |

Transformer is **not** a free win under this budget. G11 matches ablation A3.

### 5.4 Ablations and package composition

**Table 4.** WP5a ablation ladder (seed42, val).

| Row | val macro-F1 | Role |
|-----|--------------|------|
| A7 full CAD-CBA-v1 | **0.9699** | Package top |
| A3 cnn_bilstm CE | 0.9493 | Strong backbone |
| A6 +ens KD | 0.9346 | KD lift vs A5 |
| A5 attn+focal | 0.8684 | Focal helps vs A4 |
| A2 bilstm | 0.8058 | |
| A4 attn+CE | **0.7378** | Attention not free |
| A1 cnn only | 0.6221 | Weak |

![Ablation ladder](figures/fig_ablation_ladder.png)

**Figure 5.** Ablation ladder: full package (A7) wins; attention+CE (A4) underperforms plain CNN–BiLSTM (A3).

**Novelty paragraph (locked).** CAD-CBA-v1 is **not** “just a standard CNN–BiLSTM.” Novelty is **composition under a frozen protocol**: focal training, ensemble tree KD, Optuna-selected train HPs, and a systems stack (export fidelity + Option A CUDA + multi-session energy/latency). Ablations show the package path wins while attention alone is not a free gain.

### 5.5 Teachers / KD

**Table 5.** Stage-A KD student val macro-F1 (α=0.6, T=10, γ=2, seed42).

| Teacher | Student | Decision |
|---------|---------|----------|
| **ensemble** | **0.9401** | **INCORPORATED** |
| RF | 0.9346 | RUN_DOCUMENTED fallback |
| none | 0.9326 | hard-label control |
| XGB | 0.9270 | strong teacher, weaker student |
| LGBM | 0.8829 | weak path |
| Neural G11 (E6) | 0.8513 | rejected |

Do not mix Stage-A KD scores with Stage-B FT multirun means.

### 5.6 Loss / sampler / thresholds (imbalance)

| Experiment | Result | Package decision |
|------------|--------|------------------|
| CE vs focal vs focal_cb vs logit_adj | focal **0.9780** best; CB 0.9121; logit 0.9225 | **focal INCORPORATED** |
| Val thresholds on focal | all variants = argmax 0.9780 | keep **argmax** |
| D6 stratified inv-freq vs shuffle | 0.9209 vs **0.9791** | keep **shuffle** |

### 5.7 Multi-objective Pareto

A priori composite ranking (Pareto-H8): **G6 MLP score 0.9056** @ **4.33** µs/sample with F1 **0.9285** leads efficiency-weighted ranking. F1-oriented points include A7 (**0.9699** @ ~26 µs) and classical LGBM/RF references. Figures: `figures/fig_pareto_f1_latency.png`, `figures/fig_pareto_f1_params.png`.

**Interpretation.** The multi-objective story allows classical models to keep pure-F1 leadership while neural models own deployable size/latency/GPU paths — matching Prof feedback that detection need not be the sole headline.

### 5.8 Systems: latency, energy, VRAM (local multi-session)

**Table 6.** WP6b local multi-session ranges (n=5 sessions, RTX 3050, champion frozen). Source: `wp6b_local_ranges/summary.json`.

| Metric | Session-mean range | Mean |
|--------|--------------------|------|
| Energy mJ/flow @bs=128 | **0.920–0.943** | **0.933** |
| PT µs/sample @bs=256 | **24.15–25.68** | **24.90** |
| CUDA derived pipeline µs | **565.2–570.3** | 567.4 |
| CUDA block3 FP16 µs | **503.2–508.5** | 505.6 |
| Peak alloc MiB | **322.2** | — |

![Systems ranges](figures/fig_wp6b_systems_ranges.png)

**Figure 6.** Local multi-session energy and PT ranges. **Trap:** do not mix WP6b mean energy **0.933** with historical single-shot **0.786** mJ/flow (`energy_table/`, labeled HISTORICAL). **Trap:** CUDA pipeline is Option A **block sum**, not full V3 parity.

Warm-up protocol: **50** discarded sync forwards (I7). Batch-size sensitivity I8: bs∈{1,8,32,64,128,256,512,1024} multi-session tables in `systems_i8_h3/`.

### 5.9 CUDA Option A and numerical fidelity

Export path: **bit-identical** real-weight blocks (max abs error 0). CUDA self-checks: **all PASS**. FP16 block3 uses documented looser tolerance. These establish kernel correctness for operation-matched blocks; they do **not** license a full-pipeline Custom CUDA vs full V3 speedup claim.

### 5.10 Explainability (scoped)

**Table 7.** XAI suite on BoT val, champion frozen (`xai/summary.json`).

| Metric | Value | Scope |
|--------|-------|-------|
| Dispatch p99 overhead | **16.60** µs | Dispatch only |
| Occlusion rank Spearman | **0.9636** | Consistency |
| Faithfulness top-3 mass | **0.5109** | Proxy only |
| Structured usefulness (auto rubric) | **1.0** | Templates; not human SOC study |
| Free-form feature-mention | **0.333** | Weak |
| Generation mean | **~7400** ms | Never conflate with dispatch |

**J10 decision:** **DROP_FULL_EXPLAINABLE_CLAIM_KEEP_STRUCTURED**. Title and abstract do not advertise a fully LLM-explainable IDS.

### 5.11 Multi-dataset: ToN-IoT honesty (13-feat)

**Table 8.** CAD-CBA-v1 recipe transfer on `processed_toniot` (13 features).

| Model | val macro-F1 | test macro-F1 |
|-------|--------------|---------------|
| CAD-CBA-v1 (KD selected) | 0.8080 | **0.8110** |
| RF same-split | 0.9400 | **0.9393** |

≠ historical clean 26-feat CNN **0.9526**. Large RF gap is disclosed; this is recipe transfer, not BoT weight transfer.

### 5.12 Cross-GPU multi-day (TBD)

| Cell | Status |
|------|--------|
| V100S multi-day mean/median/std/CV/CI | **TBD (DICC)** |
| A100 multi-day | **TBD (DICC)** |
| Same-GPU B3 CUDA vs matching PT on cluster | **TBD (DICC)** |

Local WP6b ranges must **not** be generalised to cluster portability.

---

## 6. Discussion

### 6.1 What the sealed test allows us to say

Multi-seed test macro-F1 **0.9780±0.0033** with Theft **1.0** is a strong, protocol-honest detection result. It is **near** protocol RF and **below** protocol LGBM on pure F1. That forces the paper’s lead narrative onto **multi-objective validity** and **systems honesty**, which is exactly the Prof contribution bar.

### 6.2 Package novelty vs single-module novelty

Attention alone can hurt (A4). Multi-scale and gated variants fail bounded probes. SupCon and ASL collapse under budget. The scientific object is therefore the **evaluated package** (focal + ensemble KD + HPO + V3 + systems discipline), not a single new layer sold without measurement.

### 6.3 Dual bars and dual energy numbers

Always label:

- Protocol val vs sealed **test**.  
- Protocol RF/LGBM vs published RF **0.9864**.  
- WP6b multi-session energy **0.920–0.943** vs historical single-shot **0.786**.

### 6.4 Negative results as evidence

Failed variants remain in `benchmarks/results/` and in this manuscript’s tables. They are part of the contribution (what not to add).

---

## 7. Threats to validity

1. **Val vs test.** Most HPO/ablation/baseline numbers are **val-only**. Sealed multi-seed **test** is separate (B14).  
2. **Dual accuracy bars.** Protocol RF/LGBM ≠ published RF 0.9864 pipeline.  
3. **Single-seed ladders.** A1–A7 and G6–G12 are seed **42** / fixed budget — multi-seed stability is B14 / multiruns.  
4. **Local ≠ portable.** WP6b is **RTX 3050 laptop only**.  
5. **Option A construct validity.** Per-block CUDA latency ≠ “full model Custom CUDA vs full V3.”  
6. **Energy construct.** WP6b multi-session range is primary; 0.786 is single-shot historical.  
7. **XAI conclusion validity.** No human SOC study; free-form LLM weak; structured templates use automatic rubric.  
8. **ToN external validity.** 13-feat processed recipe transfer; large RF gap.  
9. **Theft support.** Test Theft n=**14** — perfect Theft F1 is real under this test set but limited support must be stated.  
10. **Historical text blocks.** Older `paper_text_blocks` §1–10 may mix pre-protocol numbers; **protocol-era §11** and this manuscript override for CAD-CBA-v1 claims.

---

## 8. Reproducibility

```text
Protocol: botiot_v1 (scripts/protocol/)
Method: CAD-CBA-v1 (docs/execution_plan/METHOD_PACKAGE_DECISION.md)
Train HPs: config/hpo_best.yaml
Freeze: docs/execution_plan/FINAL_CONFIG_FREEZE_CARD.md
Champion: model/best_model_botiot_twostage.pth
  md5 80a90f7cc210276300eaa90173a5a385
Sealed test: benchmarks/results/sealed_test/
Systems: benchmarks/results/wp6b_local_ranges/
Claims: docs/execution_plan/CLAIMS_REGISTRY.md
Verify: PYTHONPATH=. python3 scripts/verify_claims.py
Option A: per-block CUDA only; no full-pipeline parity claims
DICC: not included until dedicated session SUCCESS tree exists
Spine: docs/execution_plan/WP9b_MANUSCRIPT_SPINE.md
This draft: docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md
Figures: docs/manuscript/figures/
```

All load-bearing public numbers are registered in the claims package (59 claims at last rebuild). Agents and authors must rebuild and re-verify after any prose that introduces new bold numbers.

---

## 9. Conclusion

We presented **CAD-CBA-v1**, a class-aware distilled CNN–BiLSTM package evaluated under a sealed, protocol-fair regime. On BoT-IoT multi-seed **test**, the method reaches **0.9780±0.0033** macro-F1 with Theft **1.0**, near protocol-fair RF, while LightGBM retains the pure-F1 ceiling (**0.9818** val). Ablations credit **package composition** rather than attention alone. Local multi-session systems ranges quantify energy (**0.920–0.943** mJ/flow), latency (PT@256 **24.15–25.68** µs), and peak VRAM (**322.2** MiB) under Option A CUDA discipline. Explainability is scoped to dispatch and structured evidence. Multi-GPU portability remains future work for a dedicated DICC session. The primary claim is therefore a **valid multi-objective accuracy–efficiency evaluation package** with honest limits — not an oversold single-number victory.

---

## Appendix A — HPO winner parameters

| HP | Value |
|----|-------|
| lr | 5.89306076111462e-05 |
| batch_size | 1024 |
| focal_gamma | 1.9166447754858478 |
| dropout_rate | 0.14783769837532068 |
| attention_dropout | 0.21397343616689848 |
| weight_decay | 0.00019158219548093185 |
| scheduler | cosine |

Source: `config/hpo_best.yaml` / WP3 trial 8 full-train refine val **0.9791**.

## Appendix B — Figure checklist

| Figure | File | Status |
|--------|------|--------|
| Architecture | `figures/fig_architecture.png` | DONE |
| Class distribution | `figures/fig_class_distribution.png` | DONE |
| Detection dual bars | `figures/fig_detection_dual_bars.png` | DONE |
| Confusion matrix (B14 s42) | `figures/fig_confusion_matrix_b14_seed42.png` | DONE |
| Ablation ladder | `figures/fig_ablation_ladder.png` | DONE |
| WP6b systems ranges | `figures/fig_wp6b_systems_ranges.png` | DONE |
| Pareto F1–latency | `figures/fig_pareto_f1_latency.png` | DONE (copied) |
| Pareto F1–params | `figures/fig_pareto_f1_params.png` | DONE (copied) |

## Appendix C — Related-work comparison (compact)

| Theme | Typical prior claim | This work |
|-------|---------------------|-----------|
| Detection | Single split / test leakage risk | Sealed multi-seed test after freeze lock |
| Classical comparison | Different preprocess/pipeline | Protocol-fair same split + dual bar |
| Acceleration | Full-pipeline CUDA vs PT | Option A per-block + absolute PT ranges |
| Multi-GPU | Single laptop extrapolated | Explicit TBD until DICC |
| XAI | “Explainable IDS” in title | Metrics first; full claim dropped |

---

*Draft assembled 2026-07-22 writing pass from WP9b spine + on-disk JSON only. No invented DICC numbers.*
