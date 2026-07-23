# Email draft — Prof. Por status after feedback

**Status:** Ready to paste (2026-07-22)  
**Student:** Ibteshamul Haque  
**Tone:** grateful, evidence-led, humble about open DICC; not defensive  
**Do not invent:** multi-day V100S/A100 cells still empty  
**Optional attach:** none on first send (offer one-page table pack if he wants)

---

**To:** Prof. Dr. Por Lip Yee  
**Subject:** COLIDE — progress after your feedback (multi-objective evidence package; DICC multi-day next)

---

Dear Prof. Por,

Thank you again for your detailed and constructive feedback on the COLIDE interim report. I understood your assessment clearly: local progress was real, but not yet at the level expected for a strong Web of Science–oriented contribution. In particular, you emphasised that although a paper need not always achieve the highest accuracy if it makes a strong systems contribution, the **final manuscript must show a clear advantage in at least one major dimension**—detection performance, latency, memory usage, energy efficiency, throughput, robustness, **portability**, or an **overall accuracy–efficiency trade-off**—and that the contribution **must not rely mainly on implementation effort or documentation quality**.

I have taken that bar as the organising principle of the work since your feedback. Below is a concise, evidence-based update of what has been completed under your technical plan, and what remains.

---

### Contribution framing (your dimension bar)

We have **locked the primary contribution** as an **overall accuracy–efficiency multi-objective package**, not pure detection supremacy and not “engineering effort.”

Under a frozen protocol (`botiot_v1`), the sealed multi-seed BoT **test** result for the final method package (**CAD-CBA-v1**, freeze path A, n=5) is:

- macro-F1 **0.9780 ± 0.0033**  
- min-class F1 **0.9292**  
- Theft F1 **1.0000**  

This is **near** protocol-fair RF (val **0.9778**). Protocol LightGBM still leads pure F1 (val **0.9818**). The published RF **0.9864** is retained only as a **dual bar** (different pipeline) and is never mixed as protocol-fair. Detection alone is therefore **not** the sole headline—consistent with your guidance.

Against that honest detection dual bar, we quantify **local multi-session systems evidence** (RTX 3050, n=5 sessions, champion frozen):

| Dimension | Measured evidence (local) |
|-----------|---------------------------|
| **Latency** | Full V3 PyTorch @ batch 256: **24.15–25.68** µs/sample; Option A CUDA derived pipeline: **565–570** µs (per-block / operation-matched only) |
| **Energy** | **0.920–0.943** mJ/flow (multi-session range; historical single-shot 0.786 labeled HISTORICAL only) |
| **Memory** | Peak alloc **322.2** MiB (batch-256 peak ~103 MiB) |
| **Trade-off** | Pareto / a priori composite analysis; efficient path e.g. G6 composite **0.9056** @ **4.33** µs |
| **Robustness (supportive)** | Multi-seed sealed test; strong minority (Theft **1.0**); multi-session CV/CI on systems metrics |

**Portability across GPUs** remains the open major dimension in your list: multi-day V100S/A100 results are **not** claimed until UM DICC SUCCESS artifacts exist. I will not invent multi-day numbers.

Implementation quality and documentation (claims package, verifier, runbooks) are treated only as **reproducibility infrastructure**, not as the scientific contribution.

---

### How this maps to your technical recommendations

**1. Protocol, HPO, sealed test, one clear method**  
We froze a single method package—**CAD-CBA-v1**: V3 CNN–BiLSTM–Attention; ensemble tree KD (RF+XGB+LGBM, α=0.6, T=10); focal loss (γ≈1.92); Optuna train HPs; shuffle sampling; argmax decode. Validation-only HPO was completed before sealed multi-seed **test**. Architecture dimensions were not kitchen-sink Optuna’d under KD weight transfer; bounded arch probes and a plateau reject are documented.

**2. Class imbalance, teachers, staged decisions**  
Four-way loss compare selected **focal** (class-balanced focal and logit adjustment worse on macro). Val threshold search gave no lift vs argmax. Stratified inv-freq batching hurt vs shuffle. Teacher/KD compare selected **ensemble** soft labels (student val **0.9401** over RF/XGB/LGBM/none). Neural-teacher KD was weaker and not incorporated.

**3. Ablations and fair baselines (same split)**  
Full ablation ladder A1–A7 under protocol: full package tops the ladder; attention+CE alone **underperforms** plain CNN–BiLSTM—so package composition is credited honestly. Classical (LR, LinearSVC, RF, XGB, LGBM) and neural (MLP through CNN–BiLSTM + lightweight transformer) baselines share the same protocol. Transformer is not a free win under equal budget. Negative results (multi-scale CNN, gated fusion, SupCon, ASL, etc.) are retained as evidence, not discarded.

**4. CUDA / deployment discipline (Option A)**  
Per your invalid-comparison concerns: we claim only **operation-matched / per-block** Custom CUDA vs matching PyTorch, plus **absolute** full-model PyTorch latency. We do **not** claim full-pipeline Custom CUDA vs full V3 parity. Export fidelity is bit-identical on real-weight blocks; CUDA self-checks pass. Local multi-session ranges replace single lucky points for energy/latency headlines.

**5. Explainability**  
As you required, **16.60 µs is dispatch overhead only**. We measured faithfulness/consistency proxies and structured evidence usefulness; free-form LLM quality is weak (feature-mention ~0.333). Accordingly we **drop** a full “LLM-explainable IDS” title claim and keep **dispatch + structured evidence** only.

**6. Multi-dataset**  
BoT remains primary. Final-method recipe transfer on 13-feat ToN is reported with an **honest gap** (neural test ~0.8110 vs same-split RF ~0.9393), not conflated with historical 26-feat figures.

**7. Reproducibility**  
Sealed test discipline, multi-seed reporting, frozen champion md5 `80a90f7cc210276300eaa90173a5a385`, and a **64-claim** package with automated verification against source JSON (`verify_claims.py` green). Every item on our Prof-feedback tracker is terminal (DONE / INCORPORATED / RUN_DOCUMENTED), except multi-GPU/multi-day rows that remain **BLOCKED until DICC**.

A **local-complete camera-ready manuscript draft** (prose, figures, systems tables, threats to validity) has been assembled from these locked numbers. Journal class file / author list / BibTeX remain PI/venue steps; multi-GPU cells are explicit **TBD** until DICC.

---

### What remains (aligned with your multi-platform requirement)

The remaining hard gate is the **UM DICC multi-day campaign** you prioritised:

- same-GPU Block 3 Custom CUDA vs matching PyTorch  
- full-model PyTorch absolute latency  
- V100S and A100  
- ≥2 days, with mean/median/std/CV/CI and Day1–Day2 comparison  

**Ops method (current):** DICC **OnDemand → VNC Desktop** for interactive setup (so temporary VPN/network drops do not kill long interactive sessions), **`screen`** for long terminal work, and **batch** `run_campaign.sh` for Day1/Day2 so jobs continue after disconnect. Legacy June single-shot cluster figures remain labeled **legacy only** until the new SUCCESS tree exists.

Once SUCCESS artifacts are on disk, I will insert multi-GPU/multi-day cells into the manuscript and claims package **from JSON only**, then update you again.

---

### Closing

In summary: after your feedback, we have built a **protocol-fair, claim-disciplined evidence package** that targets a **clear multi-objective accuracy–efficiency advantage** with measured local latency, energy, and memory—without resting the contribution on documentation quality, and without overselling detection or full-pipeline CUDA. **Portability** is the remaining major dimension and is the next execution priority on DICC.

I would be grateful for any brief comment on whether this multi-objective framing matches the emphasis you expect for the final manuscript. If helpful, I can send a one-page table pack (sealed test, classical dual bars, ablation top line, WP6b ranges) without attaching the full draft.

Thank you again for raising the scientific standard of this work.

Best regards,  
Ibteshamul Haque  

---

## Internal checklist (not for email)

- [x] Contribution bar paragraph addressed explicitly  
- [x] Dual-bar honesty (LGBM/RF/published RF)  
- [x] Option A CUDA discipline  
- [x] XAI scoped; no full LLM-XAI claim  
- [x] DICC open; no invented multi-day numbers  
- [x] Docs/implementation not framed as contribution  
- [ ] User pastes/sends when ready  
- [ ] Optional: one-page PDF tables only if he asks  

*End draft.*
