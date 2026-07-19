# Method package decision (Phase 2) — signed default

**Date:** 2026-07-19  
**Rule:** one package first; other ideas still RUN_DOCUMENTED later.

## Chosen package (v1): **Class-aware distilled CNN–BiLSTM (CAD-CBA-v1)**

| Component | Decision | Tracker |
|-----------|----------|---------|
| Base arch | Keep V3 CNN–BiLSTM–Attention (`cnn_bilstm_v3_attention`) | F3 |
| KD teacher | RF (n=200) then try ensemble / XGB if student lag | E2–E5 |
| Loss | Compare CE / focal / focal_cb / logit_adj → pick best on **val** | D2–D5 |
| Thresholds | Val-only per-class search after train | D7, C11 |
| Multi-scale CNN / new attention | **Later** only if CAD-CBA-v1 plateaus | C3–C4 |
| SupCon / asymmetric / uncertainty | Bounded run later (skip-nothing) | C7–C10 |

## Named weakness
Extreme class imbalance + minority (Theft) under neural deploy path; RF still stronger on published path — student must win on **trade-off** and/or close minority gap under **same protocol**.

## Success criteria (val first)
1. val macro-F1 ≥ baseline multirun mean  
2. val min per-class F1 / Theft F1 improved vs pure focal FT  
3. Then sealed multi-seed test + multi-obj vs RF/XGB  

## Explicit non-goals for v1
- Full V3 CUDA parity (Option B)  
- Beating RF on every metric before multi-obj tables  

## Status
- Multirun baseline FT in progress (seed 42 done 0.9780; 43–46 running)  
- Loss compare driver ready (`run_imbalance_loss_compare.py`) — after multirun GPU free  
