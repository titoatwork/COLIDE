# B2–B4 Architecture HPO — Plateau Reject (RUN_DOCUMENTED)

**Date:** 2026-07-22  
**Tracker:** B2 (CNN layers/filters), B3 (kernel sizes), B4 (BiLSTM dims/layers)  
**Decision:** **RUN_DOCUMENTED — reject full architecture Optuna; freeze CAD-CBA-v1 V3 dims**

---

## 1. What was asked

Prof §1 Systematic HPO includes architecture dimensions (CNN layers/filters, kernels, BiLSTM hidden sizes/layers), not only train HPs.

## 2. What we already searched (WP3)

`config/hpo_best.yaml` / `benchmarks/results/hpo/summary.json` searched **train** HPs only:

| Searched | Not searched (B2–B4) |
|----------|----------------------|
| lr, batch, focal γ, dropout, attn dropout, weight decay, scheduler | CNN filters, kernel size, BiLSTM units/layers, # CNN layers |

WP3 note (locked): architecture frozen for **KD weight transfer** into V3.

## 3. Plateau evidence (do not invent multi-day re-search)

| Evidence | Value | Implication |
|----------|-------|-------------|
| WP1b multirun (old distill + default HPs) | **0.9714 ± 0.0109** n=5 | Strong multi-seed bar |
| HPO multi-seed confirm (distill + hpo_best) | **0.9689 ± 0.0145** n=5 | Train HPs do not lift multi-seed mean |
| Package multirun (ensemble KD + hpo_best) | **0.9639 ± 0.0185** n=5 | Full package path does not beat WP1b mean |
| WP5a A3 cnn_bilstm CE scratch | **0.9493** | Strong backbone without arch search |
| WP5a A4 attn+CE | **0.7378** | Attention alone can hurt under budget |
| WP5a A7 full package | **0.9699** | Composition wins ladder; not free arch scale-up |
| WP5b G11 = A3 | **0.9493** | Arch consistency under equal CE budget |
| Classical LGBM / RF protocol | **0.9818 / 0.9778** | Detection ceiling still classical under protocol |

**Plateau definition used:** multi-seed package means do not beat WP1b mean; single-seed HPO refine (0.9791) is seed-local and already multi-seed confirmed as non-mean-win.

## 4. Why full B2–B4 Optuna is rejected (now)

1. **KD transfer constraint:** Ensemble/KD teachers target V3 weight shapes. Changing filters/BiLSTM dims invalidates init checkpoints and reopens WP4b.
2. **Plateau at package level:** Extra arch HPO is unlikely to fix multi-seed mean shortfall vs WP1b without a new init recipe.
3. **Cost vs Option A:** Full arch TPE on full BoT train is multi-hour–multi-day on RTX 3050; Option A prioritizes portable claims over unbounded laptop search.
4. **Bounded arch probes still run:** C4 multi-scale kernels and C5 gated fusion are the playlist architecture **deltas** (separate from freezing V3 package dims). Results under `cstar_bounded/`.

## 5. What remains frozen in CAD-CBA-v1

From `config/config.yaml` model block (V3):

- reshape `[2, 32]`
- cnn_filters_1=64, cnn_filters_2=128, kernel=3, pool=2
- bilstm_units_1=128, bilstm_units_2=64
- dense_units=64, attention_heads=4

Train HPs still from `hpo_best.yaml` (INCORPORATED).

## 6. Reopen criteria (only if all hold)

- Multi-seed val mean ≥ WP1b **and** classical gap closed under same protocol, **or**
- User explicitly unlocks multi-day arch HPO / DICC budget, **or**
- C4/C5 bounded probes show ≥ +0.01 macro-F1 vs control under same budget (then limited arch adopt, not full Optuna).

## 7. Status flips

| ID | Status | Note |
|----|--------|------|
| B2 | **RUN_DOCUMENTED** | Plateau + KD freeze; C4 is bounded multi-scale probe |
| B3 | **RUN_DOCUMENTED** | Kernel=3 frozen; C4 multi-scale k∈{3,5,7} is the probe |
| B4 | **RUN_DOCUMENTED** | BiLSTM dims frozen for V3 package |

**No invented numbers. Champion md5 unchanged.**
