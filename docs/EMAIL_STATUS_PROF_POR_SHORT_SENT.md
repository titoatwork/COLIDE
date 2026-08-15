> **Status: INTERNAL correspondence / NOT CURRENT AUTHORITY (frozen ~2026-07, as sent).**  
> Do not cite this file as the live claim surface or as manuscript evidence. Current authority: `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/RESULTS_INDEX.md`, `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
> Email / coordination text only. Numbers or “local wrap-up / DICC next week” wording below are historical.  
> Kept for audit trail. Stale numbers here may be superseded (ToN clean 0.9526 INVALID; principal BoT is 0.9780±0.0033; DICC B3 latency is pre_fix / Option B).  
> Public GitHub visitors: this is internal correspondence, not a results paper.

# Prof Por — short summary email (as sent)

**Status:** **SENT** (user) · CC: CHERANRACH  
**Note:** Prefer this style for future updates. No codebase jargon. No long essays.

---

Dear Prof. Por,

Apologies for the lengthy earlier email, got carried away putting every item in full detail.
I have followed your feedback document entirely. Summary of work after that guidance:

Plan: DICC multi-day was blocked, so we finished local work that did not need the cluster. DICC admin advised OnDemand VNC + screen.
This week: finish local wrap-up.
From early next week: start UM DICC multi-day (Day 1–2, V100S + A100, batch run_campaign.sh). No multi-GPU/portability claims until SUCCESS artifacts exist.

Method: Protocol botiot_v1; package CAD-CBA-v1 = V3 + ensemble KD (α=0.6, T=10) + focal (γ≈1.92) + Optuna HPs + shuffle + argmax; val-only HPO; sealed multi-seed test only after freeze; champion md5 80a90f7… frozen.

Detection (dual bar): Sealed BoT test n=5: macro-F1 0.9780±0.0033, min-cls 0.9292, Theft 1.0. Protocol LGBM val 0.9818 tops pure F1; RF val 0.9778; published RF 0.9864 = other pipeline only. Lead claim = accuracy–efficiency trade-off, not pure F1 supremacy.

Imbalance / KD / ablations / baselines: Focal selected; CB/logit-adj worse; argmax thresholds; stratified batching rejected. Ensemble KD student 0.9401 best; neural teacher weaker. Ablation A1–A7 (full package tops; attention-alone can hurt). Same-split classical (LR/SVM/RF/XGB/LGBM) + neural (MLP→CNN–BiLSTM + transformer; transformer not free win). Negatives kept (multi-scale, gated, SupCon, ASL, etc.).

Systems (local, multi-session n=5): Option A only (per-block CUDA; no full CUDA vs full V3); bit-identical fidelity; CUDA self-check PASS. Energy 0.920–0.943 mJ/flow; PT@256 24.15–25.68 µs; CUDA pipe 565–570 µs; peak 322.2 MiB; multi-obj e.g. G6 composite 0.9056 @ 4.33 µs.

XAI / ToN / packaging: 16.60 µs = dispatch only; free-form LLM weak (feature-mention ~0.333) → drop full explainable title claim; keep structured + dispatch. ToN 13-feat test ~0.811 vs RF ~0.939 (honest gap). Local-complete manuscript draft + figures; 64 claims; verify_claims green.

Best regards,
Ibteshamul Haque

---

**Lesson for future Prof mail:** avoid internal tool names (`verify_claims`, etc.); use plain English from `docs/PROF_PLAIN_NUMBERS_CARD.md`.
