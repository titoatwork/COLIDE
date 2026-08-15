# figures_current

**Generated / refreshed:** 2026-08-15  
**Rule:** only claim-eligible locked artifacts (`valid: true` / protocol JSON). No invented numbers. No retrain.

## Contents

| File | Source artifact | Regenerator |
|------|-----------------|-------------|
| `fig_class_distribution.*` | `data/processed/class_distribution.json` + `sealed_test/ft_seed42.json` | `scripts/generate_manuscript_figures.py` |
| `fig_confusion_matrix_b14_seed42.*` | `sealed_test/ft_seed42.json` | `scripts/generate_manuscript_figures.py` |
| `fig_detection_dual_bars.*` | sealed_test + classical + HPO + multirun + published RF JSON | ad-hoc from locked JSON (2026-08-15) |
| `fig_ablation_ladder.*` | `ablation_ladder/summary.json` + classical RF/LGBM | ad-hoc from locked JSON (2026-08-15) |
| `fig_wp6b_systems_ranges.*` | `wp6b_local_ranges/summary.json` headline ranges | ad-hoc from locked JSON (2026-08-15) |
| `fig_toniot_corrected_cnn_per_class.*` | `toniot_corrected/summary.json` (`valid: true`) | ad-hoc from locked JSON (2026-08-15) |
| `fig_pareto_f1_latency.png` / `pareto_f1_*.png` | protocol ablation + neural systems points | `scripts/run_pareto_wp5c.py` |
| `fig_architecture.*` | schematic (not numeric claim surface) | copied from manuscript; not data-driven |

## Explicitly NOT here

- Invalid ToN “clean” CNN **0.9526** / RF **0.9851** / **+15.4%** (DATA-TON-001) — no plot assets used those numbers.
- DICC multi-day / post-campaign B3 latency comparative plots (no post_fix multi-session campaign tree).
- Empty pretty plots.

## Quarantine

Invalid JSON tombstones live under `benchmarks/results/toniot_clean_*.json` (+ `.INVALIDATED.json`).  
No figure files required quarantine into `figures_archived/` (none encoded invalid clean ToN).

See `docs/FIGURE_STATUS.md` for the full inventory.
