# 13 — Phase 9: Manuscript (After Evidence Complete)

**Status:** **WP9c + PI VENUE POLISH DONE** (2026-07-22) · local-complete PDF + figures · playlist closure audit · DICC cells open · journal class/BibTeX = PI after venue  
**Prof:** Full outline in feedback1.docx  
**Authority spine:** `docs/execution_plan/WP9b_MANUSCRIPT_SPINE.md`  
**Draft artifacts:** `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` + `.pdf` + `docs/manuscript/figures/`  
**Playlist audit:** `docs/execution_plan/PLAYLIST_CLOSURE_AUDIT.md` (133/133 terminal)

---

## 1. Entry gate (honest)

- [ ] Phase 0 DICC complete — **BLOCKED (ops)** user-scheduled  
- [x] Final method + sealed test — B14 **0.9780±0.0033**  
- [x] Ablations + baselines + Pareto  
- [x] CUDA Option A clean on **local** (cluster open)  
- [x] Numbers-match across claims / registry  
- [x] `verify_claims.py` green (**64** claims)  
- [x] Claim-source artifacts packaged (WP9a + Table 1b per-class)  
- [x] Title words match evaluated components only (J10 + spine §1)  
- [x] Camera-ready **local-complete** draft PDF + figure art (WP9c)  
- [x] PI venue polish pass  
- [x] Full playlist closure audit (local path)  

---

## 2. Locked titles (from WP9b)

1. **Recommended T1:** `CAD-CBA: A Class-Aware Distilled CNN–BiLSTM for Multi-Objective IoT Intrusion Detection with Operation-Matched CUDA Acceleration`  
2. Detection-fairness lead (T2) / systems lead (T3) — see spine  
3. ~~Full LLM-explainable title~~ **Rejected** (J10)

---

## 3. Abstract five-part structure (Prof) — drafted in spine §2 · **in draft PDF**

1. Background/problem (imbalance + low-latency)  
2. Gap (accuracy vs acceleration silos; unfair kernel comparisons; weak multi-platform)  
3. Method (CAD-CBA-v1 only)  
4. Results (sealed test, dual bars, multi-obj, WP6b ranges, scoped XAI, ToN honesty)  
5. Contribution: **valid** accuracy–efficiency advantage; no unsupported full-pipeline / multi-GPU claims  

---

## 4. Section map → our evidence

| Section | Evidence phase | Spine ref | Draft status |
|---------|----------------|-----------|--------------|
| Intro RQs 1–5 | All | WP9b §3 | **DONE (PI polish)** |
| RW 2.1–2.7 + gap table | Literature pass | spine + literature_review_raw | **DONE (compact; BibTeX=PI)** |
| Method 3.x | Phase 2–4, 6–7 | freeze card | **DONE (PI polish)** |
| Experiments 4.x | Phase 1 protocol | protocol scripts | **DONE (PI polish)** |
| Results 5.1–5.13 | Phases 0*,5,6,7,8 | WP9b §8 tables | **DONE (PI polish; §5.13 TBD DICC)** |
| Discussion 6.x | Synthesis | spine §4–5 | **DONE (PI polish)** |
| ToV 7 | Honest limits | paper_text_blocks §15 + protocol addendum | **DONE (PI polish)** |
| Repro 8 | Packaging | claims registry | **DONE (PI polish)** |
| Conclusion 9 | Summary | abstract part 5 | **DONE (PI polish)** |

\*Phase 0 multi-GPU results remain TBD cells until DICC.

---

## 5. Essential tables/figures checklist (Prof)

- [x] Architecture diagram — `docs/manuscript/figures/fig_architecture.png`  
- [x] Class distribution — `fig_class_distribution.png`  
- [x] Related-work comparison table — manuscript §2 + App C  
- [x] Overall predictive table (numbers ready)  
- [x] Per-class table (claims + B14)  
- [x] Ablation table (A1–A7) + figure  
- [x] HPO sensitivity table (hpo/ + App A)  
- [x] Per-block CUDA latency (WP6b)  
- [x] Full-model framework latency (local historical + ranges)  
- [ ] Cross-GPU multi-day stability (**DICC**)  
- [x] Pareto accuracy–latency–memory (plots exist)  
- [x] Energy/throughput (WP6b primary)  
- [x] Explanation pipeline + quality **scoped** (J10)  
- [x] Confusion matrices — B14 seed42 representative export  

---

## 6. Writing process (exceptional)

1. ~~Lock number table~~ DONE  
2. ~~Draft results first~~ DONE (WP9c)  
3. ~~Methods second~~ DONE  
4. ~~Intro/abstract last~~ DONE (spine + draft)  
5. ~~PI venue polish~~ **DONE** (continuous abstract, Table 1b/5b, systems CI/CV, PDF rebuild, App D)  
6. Final journal class file / BibTeX — **PI after venue choice**  
7. No new experiments hidden in writing phase without protocol  

---

## 7. Exit

- [x] Manuscript **spine** + artifact pointers + green verify_claims  
- [x] **Local-complete** camera-ready draft PDF + full local figure art  
- [x] **PI venue polish** pass (not publisher typeset)  
- [ ] Submission-ready after **PI venue template/BibTeX** + DICC cells (if multi-GPU claimed)  
