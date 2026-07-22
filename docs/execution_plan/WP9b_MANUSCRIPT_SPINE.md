# WP9b — Manuscript Spine (CAD-CBA-v1 / Prof feedback track)

**Status:** **DONE** (spine locked 2026-07-22)  
**Gate:** Science + local systems closed; claims **59** green; champion md5 **`80a90f7cc210276300eaa90173a5a385`**  
**Not this deliverable:** camera-ready PDF, multi-GPU DICC cells, invented numbers  
**Authority numbers:** `CLAIMS_REGISTRY.md` + `RESULTS_DISK_MANIFEST.md` + `benchmarks/results/**`

---

## 0. Entry-gate self-check (honest)

| Gate (Phase 9) | Status | Evidence |
|----------------|--------|----------|
| Phase 0 DICC complete | **OPEN (ops)** | WP0 BLOCKED — user-scheduled dedicated session only |
| Final method + sealed test | **DONE** | CAD-CBA-v1; B14 test **0.9780±0.0033** |
| Ablations + baselines + Pareto | **DONE** | WP5a/b/c + classical + C* negatives |
| CUDA Option A clean local | **DONE** | WP6a fidelity PASS; WP6b ranges; **cluster open** |
| Numbers-match + verify_claims | **DONE** | 59 claims; all green |
| Claim-source packaging | **DONE** | WP9a + post-B14/WP6b rebuild |
| Title words = evaluated only | **DONE** | J10 + §1 title policy below |

**Manuscript policy under open DICC:** write a **local-complete** paper spine. Leave multi-GPU / multi-day cells as **TBD (DICC)** — never invent. Do **not** claim portability until I1–I5 land.

---

## 1. Title policy (L12) — locked

**Forbidden in title / abstract contribution bullets until evaluated:**
- “Fully explainable” / “LLM-explainable IDS” as primary contribution (J10 **DROP_FULL**)
- Full custom-CUDA vs full V3 PyTorch **parity** speedup (Option A)
- Multi-GPU / multi-day stability as established fact

**Allowed title ingredients (evaluated):**
- Class-aware / imbalance-aware training (focal + KD under protocol)
- Distilled CNN–BiLSTM (ensemble teacher)
- Multi-objective accuracy–efficiency
- Operation-matched / per-block CUDA (Option A)
- Structured evidence / low-overhead **dispatch** (not free-form LLM quality)

### Recommended titles (pick one at PI freeze)

| # | Title | When |
|---|-------|------|
| **T1 (recommended)** | **CAD-CBA: A Class-Aware Distilled CNN–BiLSTM for Multi-Objective IoT Intrusion Detection with Operation-Matched CUDA Acceleration** | Default — XAI light, systems honest |
| T2 | **Near-RF Detection with Deployable Efficiency: Protocol-Fair Evaluation of a Distilled CNN–BiLSTM on Imbalanced IoT Traffic** | Detection + fairness lead |
| T3 | **Accuracy–Efficiency Trade-offs for Edge IoT IDS: Protocol Baselines, Ablations, and Local Multi-Session Latency/Energy Ranges** | Systems / multi-obj lead |
| ~~Full LLM-explainable title~~ | — | **Rejected** (J10; feature-mention **0.333**) |

---

## 2. Abstract five-part draft (Prof structure)

> Replace bracketed notes only with registry numbers. Do not upgrade claims.

**(1) Background/problem.**  
IoT intrusion detection must handle extreme class imbalance and low-latency edge constraints. Tree ensembles often top pure accuracy on static tabular flows, while deep models are preferred for GPU deployment, incremental adaptation, and systems co-design — yet unfair protocols and framework-only acceleration claims obscure real trade-offs.

**(2) Gap.**  
Prior work often siloes detection quality from deployment metrics, tunes without a sealed test, or reports single-shot laptop latencies as multi-platform facts. Full-pipeline “Custom CUDA vs full model” claims are invalid when kernels implement only operation-matched blocks (Option A).

**(3) Method — CAD-CBA-v1 only.**  
We freeze **CAD-CBA-v1**: V3 CNN–BiLSTM–Attention; **ensemble** KD (mean RF+XGB+LGBM soft labels, α=**0.6**, T=**10**); **focal** loss (γ≈**1.92**); Optuna train HPs (`config/hpo_best.yaml`); **shuffle** sampler; **argmax** decode. We evaluate under protocol `botiot_v1` with fair classical and neural baselines, a full ablation ladder, multi-objective Pareto analysis, local multi-session systems ranges (Option A CUDA), structured evidence + dispatch timing, and a ToN recipe-transfer honesty check.

**(4) Results (locked).**  
- BoT **sealed multi-seed TEST** (n=5, path A): macro-F1 **0.9780±0.0033**; min-cls **0.9292**; Theft **1.0000**.  
- Protocol classical pure-F1 ceiling: LGBM **0.9818** (val); RF **0.9778** (val); published RF **0.9864** is a **different pipeline** dual bar only.  
- Package ladder A7 **0.9699** (seed42) tops incremental path; attention+CE alone **hurts** vs plain CNN–BiLSTM (A4 **0.7378** ≪ A3 **0.9493**).  
- Multi-obj: a priori composite #1 **G6 MLP 0.9056** @ **4.33** µs (F1 0.9285).  
- Local systems (WP6b, n=5 sessions, RTX 3050): energy **0.920–0.943** mJ/flow; PT@256 **24.15–25.68** µs/sample; CUDA derived pipeline **565–570** µs (Option A); peak alloc **322.2** MiB.  
- XAI: dispatch p99 **16.60** µs; rank corr **0.9636**; faith mass **0.5109**; free-form LLM feature-mention **0.333** → **no** full explainable claim.  
- ToN (13-feat): test **0.8110** vs same-split RF **0.9393** (honest gap).

**(5) Contribution sentence.**  
Under a sealed, protocol-fair evaluation, CAD-CBA-v1 delivers **near-RF multi-seed test detection** with **strong minority (Theft) recognition**, while the primary scientific claim is a **valid accuracy–efficiency multi-objective story** with **operation-matched CUDA** and **local multi-session latency/energy ranges** — not pure-F1 supremacy over LightGBM and not multi-GPU portability until DICC completes.

---

## 3. Research questions — locked answers (K4–K8)

| RQ | Answer | Status | Primary evidence |
|----|--------|--------|------------------|
| **RQ1 Detection** | Improves / near classical on protocol **test** (**0.9780±0.0033** ≈ RF val **0.9778**) but **does not** beat protocol LGBM val **0.9818** or published RF **0.9864**. Detection alone is **not** the sole headline. | **DONE** | `sealed_test/`, classical handoff |
| **RQ2 Minority** | **Yes under protocol test:** Theft mean **1.0**, min-cls mean **0.9292** (n=5). Val tables consistent (HPO Theft 1.0). | **DONE** | `sealed_test/`, minority tables in claims |
| **RQ3 Latency/memory** | **Yes on local GPU:** PT@256 **24.15–25.68** µs; energy **0.920–0.943** mJ/flow; peak **322.2** MiB; G6 composite **0.9056** @4.33 µs. | **DONE (local)** | `wp6b_local_ranges/`, `pareto_h8/` |
| **RQ4 Across GPUs** | **Not yet claimable.** Local ranges ≠ multi-day V100S/A100. | **BLOCKED** | WP0 / I1–I5 |
| **RQ5 Explainability value** | **Dispatch + structured evidence yes; free-form LLM quality no.** J10 DROP full claim. | **RUN_DOCUMENTED** | `xai/summary.json` |

---

## 4. Contribution bar (A1–A6) — locked framing

| ID | Locked position |
|----|-----------------|
| **A1** | Interim progress acknowledged; **submission-ready only after** DICC portability cells (if claimed) + PI polish of spine→PDF. Science playlist for local path is closed. |
| **A2** | Historical **0.9790 &lt; RF 0.9864** remains true on dual-bar accuracy. Paper **must not** lead with pure detection alone. Lead with multi-obj + sealed-test honesty + systems ranges. |
| **A3** | Local RTX 3050 + Option A CUDA **DONE**. V100S/A100 multi-day **BLOCKED (DICC)**. |
| **A4** | **Clear quantitative advantage on ≥1 dimension:** multi-obj composite (G6 **0.9056**), energy range, PT absolute, peak VRAM, dispatch µs — even while LGBM tops pure F1. |
| **A5** | Contribution is **measured science** (59 claims, sealed test, ablations, negatives RUN_DOCUMENTED), not docs polish. |
| **A6** | Evidence package complete for manuscript drafting; this spine is the strengthen-before-finalise artifact. |

---

## 5. Novelty (C1) — one paragraph

CAD-CBA-v1 is **not** “just a standard CNN–BiLSTM.” Novelty is **composition under a frozen protocol**: class-imbalance-aware **focal** training, **ensemble tree KD**, Optuna-selected train HPs, and a systems stack (export fidelity + Option A CUDA + multi-session energy/latency). Ablations show the package path wins the ladder (A7 **0.9699**) while **attention alone is not a free gain** (A4 **0.7378** ≪ A3 **0.9493**). Bounded probes reject multi-scale CNN, gated fusion, SupCon, ASL, and neural-teacher KD. The contribution is therefore a **fully evaluated method package + honest multi-objective evidence**, not a single new layer claimed without measurement.

---

## 6. Section map → evidence (write order: results → methods → intro)

| § | Working title | Evidence sources (paths) |
|---|---------------|--------------------------|
| 1 | Introduction + RQs | This spine §3; gap matrix; feedback1 |
| 2 | Related work | `literature_review_raw.md`; gap table; Ibrahim et al. CUDA IDS note in paper_text_blocks |
| 3 | Method CAD-CBA-v1 | `METHOD_PACKAGE_DECISION.md`; `FINAL_CONFIG_FREEZE_CARD.md`; `hpo_best.yaml` |
| 4 | Experimental protocol | `scripts/protocol/`; freeze cards; seeds 42–46; test seal story |
| 5.1 | Overall detection (val + **test**) | multirun, hpo, **sealed_test** |
| 5.2 | Classical + neural baselines | `baselines_classical/`, `baselines_neural/` |
| 5.3 | Ablations + negatives | `ablation_ladder/`, `cstar_bounded/`, imbalance_loss |
| 5.4 | Teachers / KD | `teachers_kd/`, `teachers_kd_neural/` |
| 5.5 | Multi-objective Pareto | `pareto/`, `pareto_h8/` + plots |
| 5.6 | Systems latency/energy/VRAM | `wp6b_local_ranges/`, `energy_table/` (HISTORICAL label), `systems_i8_h3/` |
| 5.7 | CUDA Option A + fidelity | `numerical_fidelity.json`, cuda stats, Option A language |
| 5.8 | XAI (scoped) | `xai/` — dispatch + structured; J10 |
| 5.9 | Multi-dataset ToN | `toniot_final/` honesty gap |
| 5.10 | Cross-GPU multi-day | **TBD DICC** — empty cells allowed only if labeled |
| 6 | Discussion | Dual bars; multi-obj; negative results |
| 7 | Threats to validity | paper_text_blocks §15 + protocol-era addendum below |
| 8 | Reproducibility | claims package, md5s, scripts, freeze card |
| 9 | Conclusion | Contribution sentence §2.5 |

---

## 7. Essential tables / figures checklist

| Artifact | Status | Path / plan |
|----------|--------|-------------|
| Architecture diagram | **TODO (draw)** | V3 CAD-CBA block diagram — from model code |
| Class distribution | **TODO (plot from protocol)** | BoT train counts + SMOTE note |
| Related-work comparison table | **TODO (write)** | From literature pass |
| Overall predictive table | **READY (numbers)** | B14 test + classical dual bars |
| Per-class / minority table | **READY** | claims minority rows + sealed test aggregates |
| Ablation table | **READY** | A1–A7 ranking |
| HPO sensitivity | **READY (table)** | top trials in `hpo/`; optional figure later |
| Per-block CUDA latency | **READY (local ranges)** | WP6b block3 + derived pipeline |
| Full-model framework latency | **READY (historical local)** | statistical_significance_v2 + paper_text_blocks ranges |
| Cross-GPU multi-day stability | **BLOCKED** | DICC |
| Pareto F1–latency–memory | **READY** | `benchmarks/plots/pareto_*.png` + `pareto_h8/` |
| Energy / throughput | **READY** | WP6b primary; energy_table HISTORICAL |
| Explanation pipeline + quality | **READY (scoped)** | dispatch + structured; no full LLM claim |
| Confusion matrices | **TODO (export)** | From sealed_test / champion eval if needed |

---

## 8. Core result tables (copy into paper — numbers from disk only)

### 8.1 Sealed multi-seed BoT TEST (B14, path A)

| Seed | val macro-F1 | test macro-F1 | test min-cls | test Theft |
|------|--------------|---------------|--------------|------------|
| 42 | 0.9791 | 0.9787 | 0.9333 | 1.0000 |
| 43 | 0.9587 | 0.9798 | 0.9369 | 1.0000 |
| 44 | 0.9797 | 0.9798 | 0.9375 | 1.0000 |
| 45 | 0.9787 | 0.9722 | 0.9014 | 1.0000 |
| 46 | 0.9483 | 0.9796 | 0.9369 | 1.0000 |
| **Mean±std** | 0.9689±0.0145 | **0.9780±0.0033** | **0.9292** | **1.0000** |

Source: `benchmarks/results/sealed_test/summary.json`. Champion **unchanged**.

### 8.2 Protocol-fair classical (val)

| Model | val macro-F1 | Notes |
|-------|--------------|-------|
| LGBM (balanced) | **0.9818** | Protocol pure-F1 ceiling |
| RF | 0.9778 | Protocol-fair |
| XGB | 0.9762 | |
| LR | 0.5231 | |
| LinearSVC | 0.4268 | Weak under imbalance |
| Published RF | 0.9864 | **Different pipeline** — dual bar only |

### 8.3 Ablation ladder (seed42, val)

| Row | val macro-F1 | Role |
|-----|--------------|------|
| A7 full CAD-CBA-v1 | **0.9699** | Package top |
| A3 cnn_bilstm CE | 0.9493 | Strong backbone |
| A6 +ens KD | 0.9346 | KD lift |
| A5 attn+focal | 0.8684 | Focal helps vs A4 |
| A2 bilstm | 0.8058 | |
| A4 attn+CE | **0.7378** | Attention not free |
| A1 cnn only | 0.6221 | Weak |

### 8.4 Local systems ranges (WP6b, n=5 sessions, RTX 3050)

| Metric | Range (session means) | Mean |
|--------|----------------------|------|
| Energy mJ/flow @bs=128 | **0.920–0.943** | **0.933** |
| PT µs/sample @bs=256 | **24.15–25.68** | **24.90** |
| CUDA derived pipeline µs | **565.2–570.3** | 567.4 |
| CUDA block3 FP16 µs | **503.2–508.5** | 505.6 |
| Peak alloc MiB | **322.2** | — |

**Trap:** Do not mix WP6b mean energy **0.933** with historical single-shot **0.786**.  
**Trap:** CUDA pipeline is Option A **block sum**, not full V3 parity.

### 8.5 XAI scoped claims

| Metric | Value | Claim scope |
|--------|-------|-------------|
| Dispatch p99 | 16.60 µs | Dispatch only |
| Rank Spearman | 0.9636 | Occlusion consistency |
| Faith top-3 mass | 0.5109 | Proxy only |
| Free-form feature mention | 0.333 | Weak → drop full XAI title |
| Gen mean | ~7400 ms | Never conflate with dispatch |

### 8.6 ToN honesty (13-feat)

| Model | val | test |
|-------|-----|------|
| CAD-CBA-v1 (KD selected) | 0.8080 | **0.8110** |
| RF same-split | 0.9400 | **0.9393** |

≠ historical clean 26-feat CNN 0.9526.

---

## 9. Threats to validity — protocol-era addendum

Append to paper_text_blocks §15 (do not delete historical systems ToV):

1. **Val vs test.** Most HPO/ablation/baseline numbers are **val-only**. Sealed multi-seed **test** is reported separately (B14). Never mix without labels.  
2. **Dual accuracy bars.** Protocol RF/LGBM ≠ published RF 0.9864 pipeline.  
3. **Single-seed ladders.** A1–A7 and G6–G12 are seed **42** / fixed budget — multi-seed stability is B14 / multiruns.  
4. **Local ≠ portable.** WP6b multi-session stats are **RTX 3050 laptop only**. Multi-day V100S/A100 still open.  
5. **Option A construct validity.** Per-block CUDA latency is not “full model Custom CUDA vs full V3.”  
6. **Energy construct.** Multi-session WP6b range is primary; historical 0.786 is single-shot.  
7. **XAI conclusion validity.** No human SOC study; free-form LLM weak; structured templates are automatic rubric.  
8. **ToN external validity.** 13-feat processed recipe transfer; large RF gap; not weight transfer from BoT champion.

---

## 10. Reproducibility block (paper §8 draft)

```text
Protocol: botiot_v1 (scripts/protocol/)
Method: CAD-CBA-v1 (METHOD_PACKAGE_DECISION.md)
Train HPs: config/hpo_best.yaml
Champion: model/best_model_botiot_twostage.pth
  md5 80a90f7cc210276300eaa90173a5a385
Sealed test: benchmarks/results/sealed_test/
Systems: benchmarks/results/wp6b_local_ranges/
Claims: docs/execution_plan/CLAIMS_REGISTRY.md
Verify: PYTHONPATH=. python3 scripts/verify_claims.py
Option A: per-block CUDA only; no full-pipeline parity claims
DICC: not included until dedicated session SUCCESS tree exists
```

---

## 11. Writing process (remaining to camera-ready)

1. ~~Lock number table~~ **DONE** (claims registry + this spine)  
2. Draft results sections from §8 tables  
3. Methods from freeze card  
4. Intro/abstract last using §2  
5. Draw architecture + class-dist figures  
6. PI review cycles  
7. **DICC insert** when user opens session (replace TBD cells)  
8. Final verify_claims + claim-source zip  

**No new experiments during prose** unless a number is missing from disk — then open a bounded WP, do not invent.

---

## 12. Session deliverable status

| Item | Status |
|------|--------|
| WP9b spine document | **DONE** (this file) |
| Title policy | **DONE** |
| Abstract draft | **DONE** |
| RQ answers | **DONE** (RQ4 BLOCKED honest) |
| Core tables | **DONE** (from disk) |
| Figure art | **PARTIAL** — Pareto PNGs exist; arch/class-dist TODO draw |
| Full PDF | **TODO** (next writing session; not blocked on science) |
| DICC cells | **BLOCKED** |

---

*WP9b spine authored 2026-07-22. Numbers only from on-disk JSON / CLAIMS_REGISTRY.*
