> **Status: INTERNAL correspondence / NOT CURRENT AUTHORITY (frozen ~2026-07).**  
> Do not cite this file as the live claim surface or as manuscript evidence. Current authority: `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/RESULTS_INDEX.md`, `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
> Email / coordination draft only. Numbers or timelines below may be historical.  
> Kept for audit trail. Stale numbers here may be superseded (ToN clean 0.9526 INVALID; principal BoT is 0.9780±0.0033; DICC B3 latency is pre_fix / Option B).  
> Public GitHub visitors: this is internal correspondence, not a results paper.

# Email reply to Prof. Por (feedback on interim report)

**Use:** paste into mail client; attach nothing required (he already has interim + feedback).  
**Tone:** grateful, precise, committed; no excuses; no invented timelines for “all done.”

---

**To:** Prof. Dr. Por Lip Yee  
**Subject:** Re: COLIDE interim report — plan to meet your technical recommendations

---

Dear Prof. Por,

Thank you for the detailed feedback on the interim report and for the technical improvement plan. I appreciate the clear bar for a Web of Science–indexed contribution and the staged path you outlined.

I agree with the core assessment: local progress is real, but it is not yet sufficient for a strong journal submission. In particular:

- The CNN–BiLSTM (macro-F1 **0.9790**) does not outperform the apples-to-apples Random Forest (**0.9864**), so detection cannot be the sole headline unless we improve it under a proper protocol.  
- CUDA evidence is still centred on Block 3 on the local RTX 3050; V100S/A100 multi-day, same-GPU results remain pending and are required before any portability claim.  
- The final paper must show a **clear quantitative advantage** on at least one major dimension (detection, latency, memory, energy, throughput, robustness, portability, or a multi-objective accuracy–efficiency trade-off)—not implementation effort or documentation quality alone.

I will follow the order and standards in your note:

1. **Complete the UM DICC multi-day campaign first** (same-GPU Block 3 CUDA vs matching PyTorch, full-model PyTorch absolute latency, V100S and A100, multi-day statistics and comparison), without inventing or generalising from the laptop alone.  
2. **Then run a focused improvement phase**, without changing many components at once: freeze a reproducible baseline and split protocol; systematic validation-only hyperparameter optimisation; one clearly motivated methodological package (class-imbalance–aware training/distillation and related components as justified by evidence); proper ablations; fair baselines under the same data split; and multi-objective (e.g. Pareto) analysis.  
3. **Deployment path** after the architecture is fixed: exact export, operation-matched CUDA evaluation (Option A / per-block parity), and re-measurement as needed—including re-running cluster experiments if the model changes.  
4. **Explainability:** treat the current **16.60 µs** figure as dispatch overhead only; either add a proper evaluation of explanation quality (faithfulness / baselines such as SHAP or LIME / structured evidence) or avoid claiming a full explainability contribution in the title and abstract.  
5. **Multi-dataset:** keep BoT-IoT primary and evaluate the final method on ToN-IoT (or equivalent) under the same scientific standards.  
6. **Reproducibility:** address sealed test use, multi-run reporting where feasible, and packaging of claim-source results so a clean environment can verify numbers.

I will prepare a short comparison table of candidate methodological modifications (expected benefit, cost, and evidence) before locking the final model package, as you suggested.

My target is not a minimal patch, but a contribution that can answer your research questions rigorously: detection and minority-class behaviour under imbalance; operation-matched CUDA block latency; cross-GPU stability; multi-objective trade-offs versus strong baselines; and, if claimed, measurable value from the explanation path.

I will update you when the DICC artifacts are complete and again when the improvement-phase results (baseline, method choice, and key tables) are ready for your review—before drafting the full manuscript.

Thank you again for the precise guidance.

Best regards,  
Ibteshamul Haque  

---

*Optional shorter PS (only if useful):*  
*P.S. Parallel to DICC, I am freezing the data/split/metric protocol so the subsequent HPO and baseline suite rest on a single reproducible pipeline.*
