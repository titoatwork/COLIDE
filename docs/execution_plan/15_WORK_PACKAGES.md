# 15 — Work Packages (One Package ≈ One Focused Session)

**Rule:** Do not stack DICC + full HPO + manuscript in one chat.  
**Order:** Respect phase dependencies.  
**Live status:** update this board every session; authoritative narrative in `SESSION_CONTINUITY.md`.

---

## Package board

| ID | Name | Phase | Depends | Owner | Status |
|----|------|-------|---------|-------|--------|
| WP0 | UM DICC Day1+Day2+compare+scp | 0 | — | User/ops + coach | **TODO** (BLOCKED — no access/artifacts) |
| WP0b | Extract DICC JSON tables + fork decision | 0 | WP0 | Agent+user | TODO |
| WP1a | Unified protocol + freeze card | 1 | — (prep ok) | Agent | **DONE 2026-07-19** |
| WP1b | Multi-run baseline driver (5 seeds) | 1 | WP1a | Agent | **DONE 2026-07-19** mean 0.9714±0.0109 |
| WP2a | Sign method package (MOD table) | 2 | — | User+agent | **DONE** CAD-CBA-v1 |
| WP2b | Implement losses + thresholds modules | 2 | WP1a | Agent | **DONE** |
| WP2b-run | Imbalance loss 4-way compare | 4 | WP2b | Agent | **DONE** focal INCORPORATE |
| WP2c | Architecture delta if any (attention/multi-scale) | 2 | WP2a | Agent | TODO (only if plateaus) |
| WP2d | Val threshold search on focal best | 2/4 | WP2b-run | Agent | **DONE 2026-07-21** RUN_DOCUMENTED keep argmax |
| WP3 | Optuna HPO study + sealed test | 3 | WP1–2 | Agent | TODO |
| WP4a | Imbalance strategy sweep (beyond loss) | 4 | WP1–3 | Agent | PARTIAL (loss done; stratified/SupCon open) |
| WP4b | Teacher sweep (RF/XGB/LGBM/ensemble) | 4 | WP1–3 | Agent | **DONE 2026-07-21** ensemble student 0.9401 INCORPORATE |
| WP5a | Full ablation ladder | 5 | Final recipe | Agent | TODO |
| WP5b | Fair baseline suite | 5 | WP1a | Agent | PARTIAL (classical LR/RF/XGB/LGBM val; SVM/neural open) |
| WP5c | Pareto figure + composite score | 5 | WP5a/b + systems | Agent | TODO |
| WP6a | Re-export + fidelity | 6 | Final ckpt | Agent | TODO |
| WP6b | Local re-bench ranges Option A | 6 | WP6a | Agent | TODO |
| WP6c | Re-DICC if model changed | 6 | WP6a | User/ops | TODO |
| WP7 | XAI suite **or** drop explainable claims | 7 | Final detector | Agent | TODO |
| WP8 | ToN final-method eval | 8 | Final recipe | Agent | TODO |
| WP9a | Numbers-match + claim JSON packaging | 9 | All | Agent | TODO |
| WP9b | Manuscript spine + figures | 9 | WP9a | User+agent | TODO (not before tracker green) |

---

## What to run **right now** (next chat after verify)

| Priority | Action |
|----------|--------|
| 0 | Verify disk via `RESULTS_DISK_MANIFEST.md` (incl. `teachers_kd/`) |
| 1 | **WP3** Optuna HPO val-only **or** **WP5** ablations/neural baselines |
| Optional | stage_b FT from ensemble KD init (`kd_ensemble_…seed42.pth`) |
| Parallel | **WP0** only when user has DICC SSH |
| Never first | WP9b manuscript with empty cluster cells |

---

## WP0 detail (highest priority when unblocked)

**Inputs:** tarball or git tree on DICC, champion md5  
**Outputs:** `benchmarks/results/dicc/**/SUCCESS`, compare outcome, laptop copy  
**Exit:** Phase 0 checklist in `04_PHASE0_DICC.md` complete  

---

## WP1a / WP1b exit (achieved)

- Protocol: `scripts/protocol/*` + freeze card  
- Multirun: 5 seeds, summary JSON, ckpts under `model/multirun/`  
- Test sealed; production champion md5 unchanged  

---

## Tracking

Update Status column in this file when a WP completes. Link commit hash in HANDOFF.  
Numbers inventory: `RESULTS_DISK_MANIFEST.md`.
