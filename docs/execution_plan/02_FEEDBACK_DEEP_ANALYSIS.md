# 02 — Deep Analysis of Prof. Por Feedback

**Primary source:** `docs/feedback1.docx`  
**Secondary:** Email on interim report (RF gap + local-only CUDA)  
**Author metadata:** POR LIP YEE @ POR KHOON SUN  

---

## 1. Tone and strategic intent

| He is saying | He is not saying |
|--------------|------------------|
| Good interim process (verify numbers, invalid comparisons, clear DICC gap) | Ready to submit to WoS as-is |
| Need a **research contribution**, not an engineering diary | Abandon the project |
| Accuracy **or** multi-objective systems win must be **convincing** | Accuracy is the only path |
| Staged science (one change at a time) | Change everything in one sprint |
| DICC first, then improvement phase | Write the paper while cluster is empty |

**Bottom line:** Conditional path to a reputable journal — raise scientific + systems evidence quality to exceptional.

---

## 2. The contribution bar (non-negotiable)

A final manuscript must show a **clear advantage in at least one major dimension**:

1. Detection performance  
2. Latency  
3. Memory usage  
4. Energy efficiency  
5. Throughput  
6. Robustness  
7. Portability  
8. **Overall accuracy–efficiency trade-off** (explicitly allowed and preferred if RF still wins F1)

**Disallowed as sole contribution:** implementation effort, documentation quality, claim-verifier hygiene alone.

**Implication for us:** Every phase must produce **tables that a reviewer can attack and still find a hard win**. Soft narratives fail.

---

## 3. Ten numbered requirements — deep reading

### §1 Systematic HPO

**Ask:** Controlled search (not hand tweaks): CNN layers/filters/kernels, BiLSTM size/layers, dropout, LR/scheduler, batch, focal params, class weights, T/α, sequence length, minority thresholds.  
**Objectives:** val macro-F1, balanced accuracy, minority recall.  
**Test untouched until final config.**

**Why:** Reviewers assume cherry-picking if only one lucky test F1.  
**Our state:** Manual α/T/γ sweeps only; single seed; test often reported during exploration.  
**Exceptional bar:** Optuna study with fixed budget, logged trials, sealed test once, multi-seed confirmation of final config.

### §2 Novel method component

**Ask:** Not “just CNN–BiLSTM.” Motivated add-on: class-aware attention, multi-scale CNN, gated fusion, CB/logit-adj loss, SupCon, asymmetric loss, ensemble teacher, uncertainty, adaptive thresholds.  
**Must fix a named weakness** (imbalance, minority, cost, unstable gen).

**Why:** Architecture integration alone is low novelty in 2026 IDS literature.  
**Our state:** V3 already has multi-head attention + LN + GAP — but framing is still “standard stack + CUDA,” and attention is **not** in CUDA path.  
**Exceptional bar:** One named method (e.g. class-aware distilled temporal attention) with ablation proving the component, not kitchen-sink.

### §3 Class imbalance

**Ask:** Compare multiple strategies; select on macro-F1 **and per-class**, rare attacks first-class.

**Why:** BoT-IoT imbalance makes overall accuracy meaningless; Theft/Normal failures kill papers.  
**Our state:** SMOTE + focal; class_weights.npy unused in config (`use_class_weights: false`); outlier KD cell 0.9033 shows minority collapse risk.  
**Exceptional bar:** Head-to-head imbalance table + confusion matrices + minority recall as optim objective.

### §4 Teacher for KD

**Ask:** Improve RF / ensemble teachers; student must justify **deployment trade-off** vs pure imitation.

**Why:** Student 0.9790 < RF 0.9864 invites “why not just RF?”  
**Our state:** RF teacher, strengthen to ~0.9885 diagnostic, ensemble distill exists (0.9529 — weaker).  
**Exceptional bar:** Best teacher protocol + explicit memory/latency vs RF (and cuML RF) multi-objective argument.

### §5 Ablations

**Ask:** CNN only; BiLSTM only; CNN–BiLSTM; +KD; +imbalance; +attention/fusion; full. Metrics include **systems** (latency, params, memory, energy).

**Why:** Without ablations, contribution is unprovable.  
**Our state:** Partial (MLP, ensemble, focal-γ negatives); not the full ladder with systems metrics.  
**Exceptional bar:** One master ablation table, every row same split/protocol.

### §6 Stronger baselines

**Ask:** LR, SVM, RF, XGB, LGBM, MLP, 1D-CNN, LSTM, BiLSTM, CNN–LSTM, CNN–BiLSTM, Transformer/temporal-attn, reproducible lightweight IDS; same split; comparable tuning effort.

**Why:** Beating only a weak neural baseline is not enough when RF wins.  
**Our state:** RF, MLP, ensemble, cuML RF; not full suite under identical protocol.  
**Exceptional bar:** Single `scripts/run_baselines.py` + shared data module + JSON schema.

### §7 Multi-objective presentation

**Ask:** Near-RF detection + lower VRAM + neural latency + low explain dispatch + minority + multi-GPU stability; **Pareto** preferred.

**Why:** He explicitly allows not beating RF if trade-off is quantitative.  
**Our state:** Numbers exist separately (F1, VRAM, energy, ranges); no Pareto figure/script.  
**Exceptional bar:** Pareto plot + composite score defined a priori (not after peeking at test).

### §8 Same-GPU full DICC protocol

**Ask:** B3 CUDA vs matching PT same GPU; full PT absolute; V100S+A100; multi-day; mean/median/std/CV/CI; warm-up; batch sensitivity; significance + effect size.  
**Do not generalise from RTX 3050 alone.** Portability is central.

**Why:** Local B3 win may reverse (Rostam risk).  
**Our state:** Tooling ready; artifacts **ABSENT**.  
**Exceptional bar:** SUCCESS trees + compare accept + all stats in tables before any cluster claim.

### §9 Explainability upgrade

**Ask:** 16.60 µs is dispatch only. Need faithfulness, consistency, latency, analyst usefulness, hallucination, agreement with evidence; vs SHAP/LIME/attention/rules; structured evidence into LLM.

**Why:** Without this, “explainable” in title is overclaim.  
**Our state:** Async ring buffer + generation demo.  
**Exceptional bar:** Either complete XAI eval suite **or** drop “explainable” from title and keep dispatch as systems micro-result only.

### §10 Multi-dataset

**Ask:** BoT primary; ToN or other; transfer helps.

**Our state:** ToN clean 0.9526 / RF 0.9851 exists but not full final-method protocol.  
**Exceptional bar:** Final method frozen on BoT, then evaluated on ToN without re-cheating splits.

---

## 4. Five research questions (must answer in paper)

| RQ | Maps to phases |
|----|----------------|
| RQ1 Detection + minority under imbalance | 1–5, 8 |
| RQ2 CUDA block latency vs matching PT | 0, 6 |
| RQ3 Stability RTX / V100S / A100 | 0, 6 |
| RQ4 Multi-objective trade-offs vs baselines | 5–6, 11 Pareto |
| RQ5 Async LLM negligible critical path | 7 (quality optional) |

---

## 5. Staged plan (Prof) ↔ our phases

| Prof phase | Our phase file | Intent |
|------------|----------------|--------|
| Establish baseline | Phase 1 | Freeze protocol, multi-run |
| HPO | Phase 3 | Optuna/Bayesian |
| One proposed method | Phase 2+4 | Novelty + imbalance/teacher |
| Deploy path | Phase 6 | CUDA/frameworks after arch fixed |
| (+ implied) DICC | Phase 0 | **First** |
| Manuscript structure | Phase 9 | After evidence |

**Order he wrote at end of letter:**  
> Complete DICC first, then focused improvement phase. Prepare comparison table of modifications before deciding final model.

We implement that order strictly.

---

## 6. Suggested method branding (conditional)

**“Class-Aware Distilled Attention CNN–BiLSTM”** only if components are **thoroughly evaluated**.  
Novelty framing he likes: joint minority representation + deployment efficiency — **not** standard CNN–BiLSTM with separate CUDA.

**Warning:** V3 already has attention; novelty must be **class-aware / imbalance-aware / distillation-aware** difference, not “we added attention” if reviewers see stock MHA.

---

## 7. Realistic objectives (his words)

- Outperforming RF on **all** metrics may be unrealistic (cuML RF throughput).  
- Realistic: similar/better vs **best neural** baseline + much lower latency/memory; composite trade-off vs RF.  
- **Not:** tune until one test number beats RF without method + sealed protocol.

---

## 8. Paper structure implications

He provided full section outline (Intro RQs, RW 2.1–2.7, method 3.x, experiments 4.x, results 5.1–5.11, discussion, ToV, reproducibility, conclusion).  
Essential figures include **Pareto**, cross-GPU stability, explanation quality table, ablation, per-class, confusion matrices.

**Repro note he already flagged:** gitignored measurement files must be fixed before submission.

---

## 9. Conflict with prior “champion freeze”

| Old rule | New rule under Prof plan |
|----------|---------------------------|
| No retrain / freeze 0.9790 forever | 0.9790 = **baseline reference**; new candidates allowed under protocol |
| No clobber without backup | **Still true** — BACKUP always |
| Systems paper only | Systems **and/or** method + multi-objective |

---

## 10. Exceptional vs adequate response

| Adequate (rejectable) | Exceptional (what we aim for) |
|----------------------|-------------------------------|
| One more α/T try | Full Optuna + sealed test + multi-seed |
| “We have attention in V3” | Ablation proves class-aware component |
| RF vs one CNN | Full baseline suite same split |
| Local B3 1.3× story only | V100S/A100 multi-day same-GPU stats |
| 16.60 µs as XAI | Faithfulness suite **or** drop XAI claim |
| Scattered JSON | Unified data module + result schema + tracked claim sources |
| Separate F1 and latency chapters | Pareto multi-objective primary figure |

---

## 11. Immediate interpretation for the team

1. Interim work was **process-correct** but **contribution-incomplete**.  
2. Path is long and sequential; rushing HPO before DICC violates his order.  
3. Winning strategy is likely **multi-objective** (near-RF + memory + portable block latency), not RF-F1 oratory alone.  
4. Every implementation must leave **JSON + seed + config + git SHA** so results are paper-ready.

---

*Next: `03_GAP_MATRIX.md`*
