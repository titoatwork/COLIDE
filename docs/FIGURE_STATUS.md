# Figure status inventory (publication hygiene)

**Date:** 2026-08-15  
**Policy:** Prefer honesty over pretty empty plots. No retrain. No invented data.  
**Canonical current copies:** `benchmarks/results/figures_current/`  
**Manuscript tree:** `docs/manuscript/figures/` (synced where regenerated)  
**Invalid ToN clean path:** DATA-TON-001 — historical 0.9526 / 0.9851 / +15.4% **FORBIDDEN**

Status codes:

| Status | Meaning |
|--------|---------|
| **CURRENT** | Regenerated from locked JSON on 2026-08-15, or schematic not tied to invalidated claims |
| **STALE** | Older file; numbers still match artifacts or are exploratory only — do not treat as post-remediation canonical |
| **INVALIDATED** | Encodes pre-remediation / invalid ToN clean claims — do not use in manuscript |
| **MISSING** | Expected figure not present |

---

## Manuscript figures (`docs/manuscript/figures/`)

| Figure path | Status | Source artifact | Action needed |
|------------|--------|-----------------|---------------|
| `docs/manuscript/figures/fig_class_distribution.png` (+`.pdf`) | **CURRENT** | `data/processed/class_distribution.json`; test support from `benchmarks/results/sealed_test/ft_seed42.json` | None; regen via `scripts/generate_manuscript_figures.py` |
| `docs/manuscript/figures/fig_confusion_matrix_b14_seed42.png` (+`.pdf`) | **CURRENT** | `benchmarks/results/sealed_test/ft_seed42.json` (CM + seed-42 test macro-F1 0.9787); multi-seed mean caption from `sealed_test/summary.json` | None; regen via `scripts/generate_manuscript_figures.py` |
| `docs/manuscript/figures/fig_detection_dual_bars.png` (+`.pdf`) | **CURRENT** | B14 mean/std `sealed_test/summary.json`; protocol RF/LGBM `baselines_classical/*_seed42.json`; published RF `rf_baseline_processed.json` test_macro_f1; HPO `hpo/summary.json` winner; WP1b `multirun/summary.json` | Prefer `figures_current/` copy; keep published RF labeled non-protocol |
| `docs/manuscript/figures/fig_ablation_ladder.png` (+`.pdf`) | **CURRENT** | `ablation_ladder/summary.json` ranking + protocol RF/LGBM val | Prefer `figures_current/` copy |
| `docs/manuscript/figures/fig_wp6b_systems_ranges.png` (+`.pdf`) | **CURRENT** | `wp6b_local_ranges/summary.json` → `headline.energy_mj_per_flow_range` + `pt_batch256_us_per_sample_range` | Prefer `figures_current/` copy; never label as DICC multi-GPU |
| `docs/manuscript/figures/fig_toniot_corrected_cnn_per_class.png` (+`.pdf`) | **CURRENT** | `toniot_corrected/summary.json` (`valid: true`, CNN test macro-F1 **0.8075**, RF **0.9626**) | Prefer for multi-dataset figure; **do not** use invalid clean path |
| `docs/manuscript/figures/fig_pareto_f1_latency.png` | **CURRENT** | WP5c consolidation via `scripts/run_pareto_wp5c.py` → `benchmarks/plots/` + `benchmarks/results/pareto/` | Also mirrored in `figures_current/` |
| `docs/manuscript/figures/fig_pareto_f1_params.png` | **CURRENT** | same as above | same |
| `docs/manuscript/figures/fig_architecture.png` (+`.pdf`) | **CURRENT** (schematic) | Architecture diagram only (dims / Stage A–B labels); not a numeric claim table | Optional human redraw for venue polish; not invalid ToN |

### Manuscript prose vs figures (honest mismatch to fix in text, not by inventing plots)

| Item | Issue | Action |
|------|-------|--------|
| Manuscript §5.12 Table 8 | Still quotes WP8 package path ToN **0.8110** vs RF **0.9393** (`toniot_final`) | Prefer corrected **0.8075 / 0.9626** from `toniot_corrected` for active multi-dataset claims; keep 0.811 as labeled comparable prior only |
| Invalid clean **0.9526** | Mentioned only as withdrawn contrast in prose | Keep as **FORBIDDEN** active claim |

---

## Benchmarks plots

| Figure path | Status | Source artifact | Action needed |
|------------|--------|-----------------|---------------|
| `benchmarks/plots/pareto_f1_latency.png` | **CURRENT** | `scripts/run_pareto_wp5c.py` (protocol ablation + neural systems points; no retrain) | Copied to `figures_current/` |
| `benchmarks/plots/pareto_f1_params.png` | **CURRENT** | same | same |
| `benchmarks/results/pareto_h8/pareto_f1_latency.png` | **CURRENT** (alternate campaign style) | `scripts/run_pareto_multiobj.py` / `pareto_h8/rows.json` (H8 multi-obj rebench, 2026-07-22) | Valid BoT protocol data; different styling/cohort than WP5c plots. Do not mix IDs casually with WP5c figure captions |
| `benchmarks/results/figures_current/*` | **CURRENT** | see `figures_current/README.md` | Canonical post-hygiene export set |

---

## Notebooks / EDA (not manuscript claim figures)

| Figure path | Status | Source artifact | Action needed |
|------------|--------|-----------------|---------------|
| `notebooks/class_distribution_train.png` | **STALE** (EDA) | Early BoT train counts from notebook EDA | Do not use as sealed-eval figure; prefer `fig_class_distribution` |
| `notebooks/class_distribution_test.png` | **STALE** (EDA) | Notebook EDA | same |
| `notebooks/feature_correlation.png` | **STALE** (EDA) | Notebook EDA correlation heatmap | Optional appendix only; not claim-critical |

---

## Invalid / quarantined result artifacts (no figure files to move)

No PNG/PDF in the repo encoded invalid ToN clean numbers. JSON tombstones already exist:

| Artifact | Status | Notes |
|----------|--------|-------|
| `benchmarks/results/toniot_clean_comparison.json` | **INVALIDATED** | `valid: false`, CNN 0.9526 / RF 0.9851 / +15.4% claim status INVALID |
| `benchmarks/results/toniot_clean_comparison.INVALIDATED.json` | **INVALIDATED** | Sidecar tombstone |
| `benchmarks/results/toniot_clean_retrain.json` | **INVALIDATED** | `valid: false` |
| `benchmarks/results/toniot_clean_retrain.INVALIDATED.json` | **INVALIDATED** | Sidecar tombstone |

**Quarantine action taken:** no figure move into `benchmarks/results/figures_archived/` required (directory created empty). Do not create decorative empty plots of invalidated numbers.

---

## Plot / figure scripts

| Script | What it regenerates | Ran 2026-08-15? | Notes |
|--------|---------------------|-----------------|-------|
| `scripts/generate_manuscript_figures.py` | class distribution + B14 seed-42 CM only | **Yes** | Comment in script: dual bars / ablation / architecture / systems were original one-off |
| `scripts/run_pareto_wp5c.py` | Pareto F1–latency + F1–params under `benchmarks/plots/` | **Yes** | No retrain; consolidates protocol points |
| `scripts/run_pareto_multiobj.py` | H8 multi-obj + `pareto_h8/pareto_f1_latency.png` | **No** | Would rebench latency; existing PNG kept as alternate CURRENT |
| Ad-hoc JSON→plot (2026-08-15 hygiene) | dual bars, ablation, WP6b ranges, ToN corrected per-class | **Yes** | Numbers only from locked JSON; outputs in `figures_current/` and manuscript `figures/` |

### If regeneration scripts are incomplete

`generate_manuscript_figures.py` does **not** cover the full manuscript suite. Dual bars / ablation / WP6b / ToN corrected were regenerated once from JSON during this hygiene pass. For future automation, extend that script or keep using `figures_current/` as the export. **Table substitutes** for manuscript drafting live in:

→ [`docs/manuscript/TABLES_FROM_ARTIFACTS.md`](manuscript/TABLES_FROM_ARTIFACTS.md)

---

## MISSING (optional; not invented)

| Desired figure | Status | Reason |
|----------------|--------|--------|
| DICC multi-day multi-GPU latency/energy ranges | **MISSING** | No multi-day SUCCESS tree; blocked on DICC campaign |
| Post_fix B3 CUDA vs PT comparative multi-session plot | **MISSING** | Local B3 parity gate exists (`block3_parity_gate.json` valid); cluster comparative rebench not present as plot |
| Invalid-ToN “before/after clean” comparison plot | **MISSING** (intentionally) | Would encode INVALID claims; **do not create** |

---

## Regeneration commands (no retrain)

```bash
# Class distribution + sealed-test CM
PYTHONPATH=. .venv/bin/python scripts/generate_manuscript_figures.py

# WP5c Pareto plots from protocol JSON systems points
PYTHONPATH=. .venv/bin/python scripts/run_pareto_wp5c.py

# Copy/export
cp -f docs/manuscript/figures/fig_class_distribution.* \
      docs/manuscript/figures/fig_confusion_matrix_b14_seed42.* \
      benchmarks/plots/pareto_f1_*.png \
      benchmarks/results/figures_current/
```

Dual bars / ablation / WP6b / ToN corrected: re-run the hygiene plot block against the same JSON paths (see session notes / `figures_current/README.md`), or extend `generate_manuscript_figures.py`.

---

*End figure status.*
