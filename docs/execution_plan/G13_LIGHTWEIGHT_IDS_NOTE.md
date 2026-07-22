# G13 — Reproducible lightweight IDS baseline (playlist note)

**Status:** **RUN_DOCUMENTED N/A** (no suitable public method selected under protocol)  
**Date:** 2026-07-22  
**Tracker:** G13

## Requirement
Prof §6 asks for a reproducible lightweight IDS method under the **same train/val/test partitions** as CAD-CBA-v1 (`botiot_v1` / stage_b_ft).

## Decision
No external “named lightweight IDS” paper method was ported as a third-party code drop-in for this playlist item, because:

1. **Protocol lock:** Any external method must use the identical BoT-IoT freeze card split, feature set (10 tabular features, V3 reshape where applicable), and val-only selection. Most published “lightweight IDS” numbers use different preprocess/undersample/SMOTE pipelines and are **not protocol-fair**.
2. **Internal coverage already provides the lightweight neural bar:**
   - G6 MLP CE scratch val macro-F1 **0.9285** (400k params, ~4.3 µs/sample)
   - G7 1D-CNN **0.6221**
   - G8 LSTM **0.8099**
   - Ablation A1/A3/G11 CNN–BiLSTM CE **0.9493**
3. **Classical lightweight:** LR **0.5231** (weak); LinearSVC (G2 full) and LGBM (G5 fix) documented in `baselines_classical/`.
4. **Risk of inventing numbers:** Claiming a named third-party IDS without a sealed re-implementation would violate skip-nothing honesty.

## What would unblock a named G13 later
- Pick one public repo with a clear multi-class BoT-IoT recipe  
- Re-implement under `scripts/protocol/botiot.py` stage_b_ft only  
- Equal-budget HPs (G15) or document HPO effort  
- Write JSON under `benchmarks/results/baselines_external/`  

Until then, **G13 = N/A with reason** (this note). Paper tables should use the protocol-fair internal lightweight suite (G6–G12 + classical G1–G5), not external published scores.

## Do not claim
- That G13 was “skipped” silently  
- That any historical non-protocol lightweight IDS score is comparable to CAD-CBA-v1 multirun
