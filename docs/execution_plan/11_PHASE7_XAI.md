# 11 — Phase 7: Explainability Beyond Dispatch

**Status:** DONE 2026-07-22 — suite RUN_DOCUMENTED; J10 = drop full claim / keep structured+dispatch  
**Depends on:** Final detector  
**Prof:** §9  

---

## 1. Binary decision (do first)

| Choice | Condition |
|--------|-----------|
| **A. Full XAI track** | Title/abstract claim “explainable” / LLM explanations as contribution |
| **B. Systems-only dispatch** | Keep 16.60 µs as micro-result; **no** explainable branding |

Exceptional work requires **A** if the word explainable appears.

---

## 2. If Choice A — required evaluations

| Metric | Method ideas |
|--------|----------------|
| Faithfulness | Feature occlusion / perturbation vs explanation claims |
| Consistency | Same input → stable explanation themes |
| Latency | Dispatch (done) + generation time separate |
| Hallucination rate | Rubric: contradicted by features/label |
| Agreement with model evidence | Top features vs text mentions |
| Baselines | SHAP, LIME, attention weights, rule templates |
| Analyst usefulness | Structured rubric (even small internal study) |

**Architecture:** Prefer structured evidence (class, conf, top features, thresholds) → LLM template; not free-form only.

---

## 3. Implementation sketch

| Item | Notes |
|------|-------|
| Extend `llm_explainability.py` | Structured prompt builder |
| `scripts/xai_eval.py` | Automated metrics |
| `scripts/xai_baselines.py` | SHAP/LIME on tabular features |
| Results JSON | Per-metric scores |

---

## 4. Acceptance criteria (Choice A)

- [x] Dispatch + generation reported separately (16.60 µs vs ~7400 ms)  
- [x] ≥1 faithfulness metric (occlusion top-3 mass 0.5109)  
- [x] Comparison vs attention + rules (shap/lime not installed — documented)  
- [x] Hallucination/consistency protocol documented  
- [x] Claims match measured results → **Choice B for title/abstract** + structured evidence kept  

Choice B acceptance: abstract never claims full explanation quality. **Locked.**

---

## 5. Exit

**DONE:** `benchmarks/results/xai/summary.json` · J10 `DROP_FULL_EXPLAINABLE_CLAIM_KEEP_STRUCTURED`.
