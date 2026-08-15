> **Status: INTERNAL correspondence / NOT CURRENT AUTHORITY (frozen ~2026-08).**  
> Do not cite this file as the live claim surface or as manuscript evidence. Current authority: `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/RESULTS_INDEX.md`, `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
> Email / coordination draft only. Numbers or closure wording below may be historical.  
> Kept for audit trail. Stale numbers here may be superseded (ToN clean 0.9526 INVALID; principal BoT is 0.9780±0.0033; DICC B3 latency is pre_fix / Option B).  
> Public GitHub visitors: this is internal correspondence, not a results paper.

# Final email to Prof. Por — user-approved draft (COLIDE-first)

**Focus:** COLIDE status + manuscript direction first.  
**Internship / certificate:** short closing asks only (do not dominate the email).  
**Status:** ready to send  

---

## Subject

COLIDE — status update, manuscript direction, and completion certificate request

---

## Body

Dear Prof. Por,

I hope you are well. I am writing with a short status update on COLIDE, and to ask for your guidance on the manuscript. I also have a brief request about internship completion paperwork for my home university.

I treated your feedback document as the working checklist for the rest of the campaign. That covered not only local accuracy work, but also protocol discipline, multi-objective framing, fair baselines and ablations, deployment-style measurements, multi-GPU evidence on UM DICC, and careful claim scope, including not overselling pure detection scores or free-form LLM explainability.

Where the results did not support a simple “win everywhere” story, we kept the measurement and reported it honestly. In particular, on the servers, matching PyTorch is faster than our Custom CUDA Block-3 (BiLSTM) path, while Custom CUDA remains stronger on the CNN and dense blocks. We also do not claim a full Custom CUDA pipeline versus the full PyTorch model, because architecture parity is incomplete. This Block-3 finding is one of the important outcomes of the multi-GPU study, so we would like to include it carefully in the paper rather than hide it.

In summary, we completed a single frozen detection method under a sealed evaluation protocol, with the lead story as a multi-objective accuracy–efficiency trade-off rather than pure detection-score supremacy against strong classical baselines. On the laptop GPU, we finished the operation-matched Custom CUDA comparisons and the multi-compiler measurements (eager PyTorch, torch.compile, TensorRT, and ONNX Runtime), reported carefully as multi-session ranges with the usual caveats. On UM DICC, we completed multi-session runs on V100S and A100 with same-GPU PyTorch baselines (per-block Option A only), and also finished a full multi-compiler matrix on both server GPU classes. Evidence collection and claim hygiene for this package are closed. What remains is mainly manuscript writing and venue formatting, not a missing multi-GPU campaign.

I am not claiming that every optional follow-on has been finished, such as deeper profiling, a cleaner A100 provenance re-run, or a major Block-3 kernel redesign. Those remain open only if you judge them necessary for the paper path you prefer.

May I please ask:

First, how would you like us to proceed with the manuscript writing from here? I would value your guidance on structure, emphasis, venue, and how you would like the multi-GPU and multi-compiler findings written into the paper.

Second, may we mark the internship as completed on the project side, now that this evidence package is finished? I will also need an internship completion certificate for my home university, and would be grateful if I may request it now.

Thank you again for your detailed feedback and guidance. I am ready to follow your direction on the writing and submission path.

Regards,
Ibteshamul Haque
FCSIT, Universiti Malaya

---

## Notes (for you only — not in the email)

- Main weight is on **COLIDE + manuscript**, not internship logistics.  
- Certificate is one short closing sentence, not a separate essay.  
- Block-3 honesty kept as a scientific finding for the paper.  

*End.*
