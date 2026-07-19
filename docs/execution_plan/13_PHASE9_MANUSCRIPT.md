# 13 — Phase 9: Manuscript (After Evidence Complete)

**Status:** BLOCKED until Phases 0–6 (+7/8 as claimed)  
**Prof:** Full outline in feedback1.docx  

---

## 1. Entry gate (all required)

- [ ] Phase 0 DICC complete  
- [ ] Final method + sealed test  
- [ ] Ablations + baselines + Pareto  
- [ ] CUDA Option A clean on local + cluster  
- [ ] Numbers-match across README/docs/claims  
- [ ] `verify_claims.py` green  
- [ ] Claim-source artifacts packaged  
- [ ] Title words match evaluated components only  

---

## 2. Suggested titles (conditional)

1. `COLIDE: A Class-Aware Distilled CNN–BiLSTM with CUDA-Optimized Inference and Low-Overhead LLM Explanations for IoT Intrusion Detection` — only if class-aware + XAI evaluated  
2. `A CUDA-Optimized and Explainable CNN–BiLSTM Framework for Imbalanced IoT Intrusion Detection` — only if explainable evaluated  
3. Safer systems title if XAI light: emphasise multi-objective + operation-matched CUDA  

---

## 3. Abstract five-part structure (Prof)

1. Background/problem (imbalance + low-latency)  
2. Gap (accuracy vs acceleration silos; unfair kernel comparisons; weak multi-platform)  
3. Method (only what exists)  
4. Results (macro-F1, minority, B3, full-model, memory, thr, energy, dispatch)  
5. Contribution: **valid** accuracy–efficiency advantage; no unsupported full-pipeline claims  

---

## 4. Section map → our evidence

| Section | Evidence phase |
|---------|----------------|
| Intro RQs 1–5 | All |
| RW 2.1–2.7 + gap table | Literature pass |
| Method 3.x | Phase 2–4, 6–7 |
| Experiments 4.x | Phase 1 protocol |
| Results 5.1–5.11 | Phases 0,5,6,7,8 |
| Discussion 6.x | Synthesis |
| ToV 7 | Honest limits |
| Repro 8 | Packaging |
| Conclusion 9 | Summary |

---

## 5. Essential tables/figures checklist (Prof)

- [ ] Architecture diagram  
- [ ] Class distribution  
- [ ] Related-work comparison table  
- [ ] Overall predictive table  
- [ ] Per-class table  
- [ ] Ablation table  
- [ ] HPO sensitivity figure  
- [ ] Per-block CUDA latency  
- [ ] Full-model framework latency  
- [ ] Cross-GPU multi-day stability  
- [ ] Pareto accuracy–latency–memory  
- [ ] Energy/throughput  
- [ ] Explanation pipeline + quality (if claimed)  
- [ ] Confusion matrices  

---

## 6. Writing process (exceptional)

1. Lock number table (single source)  
2. Draft results first  
3. Methods second  
4. Intro/abstract last  
5. PI review cycles  
6. No new experiments hidden in writing phase without protocol  

---

## 7. Exit

Submission-ready PDF + artifact appendix + green verify_claims.
