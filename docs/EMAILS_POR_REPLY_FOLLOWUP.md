> **Status: INTERNAL correspondence / NOT CURRENT AUTHORITY (frozen ~2026-08-12).**  
> Do not cite this file as the live claim surface or as manuscript evidence. Current authority: `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/RESULTS_INDEX.md`, `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
> Email / coordination drafts only. Not a claim register.  
> Kept for audit trail. Stale numbers here may be superseded (ToN clean 0.9526 INVALID; principal BoT is 0.9780±0.0033; DICC B3 latency is pre_fix / Option B).  
> Public GitHub visitors: this is internal correspondence, not a results paper.

# Follow-up emails after Prof. Por’s reply (2026-08-12)

**Prof said:** share all paper materials with Cheran; work with him on manuscript; certificate → Dr. Erma / UM admin (not Por).  
**Cheran said:** please share materials; then he will review and proceed with write-up.

---

## Email 1 — Reply to Prof. Por  
**To:** Prof. Por  
**CC:** Cheranrach Mahandren  
**Subject:** Re: COLIDE — status update, manuscript direction, and completion certificate request

Dear Prof. Por,

Thank you for your guidance.

I will share the full set of paper materials with Cheran as you advised, including the draft writing, experimental analysis, results and data, source code, tables, figures, diagrams, and supporting notes on baselines, ablations, multi-GPU results, and multi-compiler comparisons. I will work closely with him on the manuscript preparation and keep you updated on progress.

For the internship completion certificate, I will contact Dr. Erma / the relevant UM administrative office as you directed.

Thank you again for your supervision and for clarifying the next steps.

Regards,  
Ibteshamul Haque  
FCSIT, Universiti Malaya

---

## Email 2 — Reply to Cheran (materials handoff)  
**To:** Cheranrach Mahandren  
**CC:** Prof. Por (optional but good, since he asked to be kept updated)  
**Subject:** COLIDE paper materials for manuscript preparation

Dear Cheran,

Thank you for coordinating the manuscript preparation as advised by Prof. Por.

I am sharing the available COLIDE materials for the paper. Please find below what is included and how to access it. If anything is missing or you prefer a different packaging (for example a zip archive or a short walkthrough call), I am happy to arrange that.

**1. Source code and repository**  
- GitHub (master): https://github.com/titoatwork/COLIDE  
- Please pull the latest `master`. I can also prepare a zip of the tree if that is easier for you.

**2. Writing prepared so far**  
- Manuscript draft: `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` and `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf`  
- Manuscript spine / structure notes: `docs/execution_plan/WP9b_MANUSCRIPT_SPINE.md`  
- Pre-write claim map (what can and cannot be claimed): `docs/CLAIM_MAP_PREWRITE.md`  
- Pre-manuscript closure summary: `docs/PRE_MANUSCRIPT_CLOSURE.md`  
- Plain-language numbers card: `docs/PROF_PLAIN_NUMBERS_CARD.md`

**3. Figures and diagrams**  
- Folder: `docs/manuscript/figures/`  
  - Architecture  
  - Class distribution  
  - Detection dual bars  
  - Ablation ladder  
  - Confusion matrix (sealed test seed 42)  
  - Local systems ranges  
  - Pareto plots  

**4. Experimental analysis, results, and data**  
- Method package decision: `docs/execution_plan/METHOD_PACKAGE_DECISION.md`  
- Modification decision table: `docs/MOD_DECISION_TABLE.md`  
- Feedback tracker (checklist of completed experiments): `docs/execution_plan/PROF_FEEDBACK_TRACKER.md`  
- Claims registry: `docs/execution_plan/CLAIMS_REGISTRY.md`  
- Result JSON under `benchmarks/results/` (local systems, sealed test, ablations, baselines, HPO, ToN, etc., where present on disk)  
- Champion weights: `model/best_model_botiot_twostage.pth`

**5. Multi-GPU (UM DICC) results**  
- Extraction tables: `docs/DICC_EXTRACTION_TABLES.md`  
- Block-3 CUDA vs PyTorch report (honest finding): `docs/DICC_B3_CUDA_VS_PT_REPORT.md`  
- Compare outcomes: `docs/DICC_COMPARE_OUTCOMES.md`  
- Results pack / flags: `docs/DICC_RESULTS_AND_FLAGS.md`  
- SUCCESS run trees: `benchmarks/results/dicc/core/` (V100S and A100, three sessions)

**6. Multi-compiler comparisons**  
- DICC matrix write-up: `docs/DICC_MULTI_COMPILER_MATRIX.md`  
- JSON: `benchmarks/results/dicc/framework/multi_compiler_v100s.json`  
- JSON: `benchmarks/results/dicc/framework/multi_compiler_a100.json`  
- Laptop multi-compiler narrative remains in the manuscript draft / README (local ranges)

**7. Baselines, ablations, and other experiment notes**  
Documented across the tracker and result folders for classical and neural baselines, ablation ladder, HPO, distillation/teachers, ToN-IoT transfer evaluation, and XAI/dispatch scope. I can walk you through the folder map in a short call if useful.

Please let me know once you have access, and how you prefer we work day to day (shared draft, weekly sync, etc.). I am ready to support the write-up closely.

Regards,  
Ibteshamul Haque

---

## Email 3 — To Dr. Erma (CC Ridzwan / GMC) — internship certificate  
**To:** Dr. Erma (Deputy Dean, Student Affairs, FCSIT)  
**CC:** Muhammad Ridzwan bin Mohd Rashid (Assistant Registrar, Global Mobility Centre)  
**Subject:** Request for internship completion certificate — Ibteshamul Haque (COLIDE / FCSIT)

Dear Dr. Erma,

I hope you are well. I am Ibteshamul Haque, a research intern at the Faculty of Computer Science and Information Technology, Universiti Malaya, under the project supervision of Prof. Dr. Por Lip Yee (COLIDE project).

Prof. Por has advised me that internship completion certificates are an administrative matter and that I should contact you or the relevant UM office. I am copying Mr. Muhammad Ridzwan bin Mohd Rashid from the Global Mobility Centre, as he has been handling the related administrative process.

I would like to request an internship completion certificate (or official completion letter) for submission to my home university.

For clarity on the internship period:
- I was **on-site at UM for most of June** (as covered in my internship cover letter / related documents).
- After that, I continued the research work **remotely** under the same project supervision, including completion of the experimental and multi-GPU evidence package for COLIDE.
- The cover letter currently on file appears to cover **June only**. I would be grateful if the completion certificate could reflect the full internship engagement as recognised by the faculty / GMC, or if you could advise the correct way to document the on-site and remote portions for my home university.

Could you please advise:
1. Whether I may request the certificate now,  
2. What forms or supporting documents are required, and  
3. How the June on-site period and the subsequent remote period should be stated on the official letter.

I am happy to provide a short project summary, dates confirmation from my side, or any other documents you need. If a short confirmation from Prof. Por as project supervisor is required for the remote portion, please let me know and I will request it in the form your office prefers.

Thank you very much for your guidance and assistance.

Regards,  
Ibteshamul Haque  
FCSIT, Universiti Malaya  
[your student/intern ID]  
[your phone]  
[your email]

---

## Suggested send order

1. **Email 1** to Prof (CC Cheran) — same day  
2. **Email 2** to Cheran (CC Prof optional) — same day, with repo link + material map  
3. **Email 3** to Dr. Erma — same day or next working day  

If you do not have Dr. Erma’s address yet, ask Cheran or faculty office politely for the correct admin contact before sending Email 3.

---

## Optional: one-line reply if you only reply-all once

If you prefer a single reply-all to Prof + Cheran:

Dear Prof. Por and Cheran,

Thank you, Prof., for the clear direction. I will share the full paper materials with Cheran as listed in my separate note, work with him on the manuscript, and keep you updated. For the completion certificate I will contact Dr. Erma / the administrative office as advised.

Cheran — I am sending the material map and repository access next / below. Happy to walk through it with you.

Regards,  
Ibteshamul Haque

*End.*
