# 11 — Phase 7: Explainability Beyond Dispatch

**Status:** NOT STARTED  
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

- [ ] Dispatch + generation reported separately  
- [ ] ≥1 faithfulness metric  
- [ ] Comparison table vs SHAP/LIME or attention  
- [ ] Hallucination/consistency protocol documented  
- [ ] Claims match measured results  

Choice B acceptance: abstract never claims explanation quality.

---

## 5. Exit

Results 5.10 ready **or** XAI claims removed from paper spine.
