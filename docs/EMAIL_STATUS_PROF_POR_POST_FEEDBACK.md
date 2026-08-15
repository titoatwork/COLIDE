> **Status: INTERNAL correspondence / NOT CURRENT AUTHORITY (frozen ~2026-07-22).**  
> Do not cite this file as the live claim surface or as manuscript evidence. Current authority: `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/RESULTS_INDEX.md`, `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
> Email / coordination draft only. Numbers or DICC timelines below may be historical.  
> Kept for audit trail. Stale numbers here may be superseded (ToN clean 0.9526 INVALID; principal BoT is 0.9780±0.0033; DICC B3 latency is pre_fix / Option B).  
> Public GitHub visitors: this is internal correspondence, not a results paper.

# Email draft — Prof. Por status after feedback

**Status:** Ready to paste (plain text, no tables) · 2026-07-22  
**Student:** Ibteshamul Haque  
**Tone:** grateful, evidence-led; honest about DICC still open; not defensive  
**Includes:** DICC admin response + timeline (local wrap-up this week; DICC from early next week)  
**Do not invent:** multi-day V100S/A100 cells still empty  
**Optional attach:** none on first send  

---

**To:** Prof. Dr. Por Lip Yee  
**Subject:** COLIDE — progress after your feedback (local package nearly closed; DICC multi-day from early next week)

---

Dear Prof. Por,

Thank you again for your detailed and constructive feedback on the COLIDE interim report. I understood clearly that local progress at that stage was not yet enough for a strong Web of Science–oriented contribution, and that the final manuscript must show a clear advantage in at least one major dimension—detection performance, latency, memory usage, energy efficiency, throughput, robustness, portability, or an overall accuracy–efficiency trade-off—without relying mainly on implementation effort or documentation quality.

**Intention and sequencing.** Because the UM DICC multi-day campaign was still blocked on access and reliable interactive execution, and based on your feedback that the local scientific package itself also needed to be strengthened under a proper protocol, we deliberately started and completed the **local work that could proceed without waiting on DICC**. That includes protocol freeze, systematic HPO, one locked method package, imbalance and teacher studies, ablations and fair baselines, multi-objective analysis, Option A CUDA fidelity, local multi-session latency/energy/memory ranges, scoped explainability evaluation, multi-dataset honesty on ToN, sealed multi-seed test, claims packaging, and a local-complete manuscript draft. We did **not** treat documentation as the contribution, and we did **not** invent multi-day cluster numbers or claim GPU portability before SUCCESS artifacts exist.

**Update on DICC access and timeline.** We have now received a response from the DICC administrator regarding interactive-session reliability. Their guidance is to use the **VNC Desktop through DICC OnDemand** for longer interactive work (so temporary network or VPN interruptions do not drop the allocation), and to use tools such as **screen** inside that environment for long-running tasks. With that path clarified, we will **proceed with the UM DICC multi-day campaign**. Our plan is to **finish remaining local wrap-up this week**, and to **start DICC execution from early next week** (Day 1 and Day 2 on V100S and A100 under batch run_campaign.sh, then bring SUCCESS artifacts home and insert multi-GPU cells from JSON only). Multi-day V100S/A100 results remain required before any portability claim.

Below is a concise update of what the local package now contains, and what remains on DICC.

### Contribution framing (your dimension bar)

We locked the primary contribution as an **overall accuracy–efficiency multi-objective package**, not pure detection supremacy and not engineering effort alone.

Under frozen protocol botiot_v1, the sealed multi-seed BoT **test** result for the final method package CAD-CBA-v1 (freeze path A, n=5) is macro-F1 **0.9780 ± 0.0033**, min-class F1 **0.9292**, and Theft F1 **1.0000**. This is near protocol-fair RF (val **0.9778**). Protocol LightGBM still leads pure F1 (val **0.9818**). The published RF **0.9864** is kept only as a dual bar from a different pipeline and is never mixed as protocol-fair. Detection alone is therefore not the sole headline, which matches your guidance.

Against that honest detection dual bar, local multi-session systems evidence on RTX 3050 (n=5 sessions, champion frozen) quantifies latency, energy, and memory: full V3 PyTorch at batch 256 is **24.15–25.68** µs per sample; Option A CUDA derived pipeline is **565–570** µs (per-block / operation-matched only—not full-pipeline Custom CUDA versus full V3); energy is **0.920–0.943** mJ per flow as the multi-session range (historical single-shot 0.786 is labeled historical only); peak allocated memory is **322.2** MiB. Multi-objective Pareto analysis supports an efficiency-weighted path (for example G6 composite **0.9056** at **4.33** µs). Multi-seed sealed test and strong minority behaviour support robustness as a secondary element.

**Portability across GPUs** remains the open major dimension in your list until the multi-day DICC SUCCESS tree exists. Implementation quality and documentation (claims package, verifier, runbooks) are reproducibility infrastructure only, not the scientific contribution.

### How this maps to your technical recommendations

**1. Protocol, HPO, sealed test, one clear method.** We froze CAD-CBA-v1: V3 CNN–BiLSTM–Attention; ensemble tree KD (RF+XGB+LGBM, α=0.6, T=10); focal loss (γ≈1.92); Optuna train hyperparameters; shuffle sampling; argmax decode. Validation-only HPO was completed before sealed multi-seed test. Bounded architecture probes and a plateau reject are documented rather than kitchen-sink architecture search under KD weight transfer.

**2. Class imbalance, teachers, staged decisions.** A four-way loss compare selected focal loss; class-balanced focal and logit adjustment were worse on macro-F1. Validation threshold search gave no lift versus argmax. Stratified inverse-frequency batching hurt versus shuffle. Teacher/KD compare selected ensemble soft labels (student val **0.9401** over RF/XGB/LGBM/none). Neural-teacher KD was weaker and not incorporated.

**3. Ablations and fair baselines (same split).** Full ablation ladder A1–A7 under protocol: the full package tops the ladder; attention with cross-entropy alone underperforms plain CNN–BiLSTM, so package composition is credited honestly. Classical baselines (LR, LinearSVC, RF, XGB, LGBM) and neural baselines (MLP through CNN–BiLSTM plus a lightweight transformer) share the same protocol. Transformer is not a free win under equal budget. Negative results (multi-scale CNN, gated fusion, SupCon, ASL, and others) are retained as evidence.

**4. CUDA / deployment discipline (Option A).** We claim only operation-matched per-block Custom CUDA versus matching PyTorch, plus absolute full-model PyTorch latency. We do not claim full-pipeline Custom CUDA versus full V3 parity. Export fidelity is bit-identical on real-weight blocks; CUDA self-checks pass. Local multi-session ranges replace single lucky points for energy and latency headlines.

**5. Explainability.** As you required, **16.60** µs is dispatch overhead only. We measured faithfulness and consistency proxies and structured evidence usefulness; free-form LLM quality is weak (feature-mention about **0.333**). We therefore drop a full “LLM-explainable IDS” title claim and keep dispatch plus structured evidence only.

**6. Multi-dataset.** BoT remains primary. Final-method recipe transfer on 13-feature ToN is reported with an honest gap (neural test about **0.8110** versus same-split RF about **0.9393**), not conflated with historical 26-feature figures.

**7. Reproducibility.** Sealed-test discipline, multi-seed reporting, frozen champion md5 **80a90f7cc210276300eaa90173a5a385**, and a 64-claim package with automated verification against source JSON (verify_claims green). Every item on our internal Prof-feedback tracker is terminal (done, incorporated, or run-documented) except multi-GPU multi-day rows that remain blocked until DICC SUCCESS artifacts exist.

A local-complete camera-ready manuscript draft (prose, figures, systems tables, threats to validity) has been assembled from these locked numbers. Journal class file, author list, and bibliography remain final venue steps. Multi-GPU cells in the draft are explicit TBD until DICC.

### What remains — DICC multi-day (from early next week)

The remaining hard gate is exactly the multi-platform campaign you prioritised and that we could not complete while DICC execution was blocked:

- same-GPU Block 3 Custom CUDA versus matching PyTorch  
- full-model PyTorch absolute latency  
- V100S and A100  
- at least two days, with mean, median, standard deviation, CV, CI, and Day-1 versus Day-2 comparison  

We will execute this under the administrator-guided ops method: DICC OnDemand VNC Desktop for interactive setup; screen for long terminal work; and batch run_campaign.sh for Day 1 and Day 2 so jobs continue after disconnect. Legacy June single-shot cluster figures remain labeled legacy only until the new SUCCESS tree exists.

Once SUCCESS artifacts are on disk, I will insert multi-GPU multi-day cells into the manuscript and claims package from JSON only, then update you again.

### Closing

In short: because DICC was not yet solvable as a blocker, we used your feedback to complete the local scientific and local systems package that could be finished without cluster multi-day results—under protocol discipline, with a clear multi-objective accuracy–efficiency advantage, without overselling detection or full-pipeline CUDA, and without treating documentation as the contribution. With the DICC administrator’s guidance now in hand, we will finish remaining local wrap-up this week and begin the multi-day DICC campaign from early next week. Portability remains the open major dimension until those SUCCESS results are in.

I would be grateful for any brief comment on whether this multi-objective framing matches the emphasis you expect for the final manuscript. If helpful, I can later send a short plain-text summary of key numbers without attaching the full draft.

Thank you again for raising the scientific standard of this work.

Best regards,  
Ibteshamul Haque  

---

## Internal checklist (not for email)

- [x] Opening intention: local work while DICC blocked  
- [x] DICC admin response + OnDemand VNC + screen  
- [x] Timeline: local wrap-up this week; DICC from early next week  
- [x] Contribution dimension bar addressed  
- [x] No markdown tables (plain-text friendly)  
- [x] Dual-bar honesty; Option A; scoped XAI  
- [x] DICC leftover listed explicitly  
- [ ] User pastes/sends when ready  

*End draft.*
