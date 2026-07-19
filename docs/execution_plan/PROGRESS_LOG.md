# Progress log (results as they land)

**Policy:** document everything; label pilot vs full; test sealed unless noted.

---

## 2026-07-19

### Protocol foundation
- `scripts/protocol/*` live (`botiot_v1`, metrics, losses, thresholds)
- Champion sealed eval val macro-F1 **0.9780** (`stage_b_ft`, seed 42)
- Production champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`

### Fine-tune multirun (WP1b)
| Seed | Epochs | Best val macro-F1 | Notes |
|------|--------|-------------------|-------|
| 42 smoke | 2 | **0.9755** | Done early |
| 42 full | 10 target | **0.9780** at epoch 3 (in progress) | ep1 0.9755, ep2 0.9684, ep3 0.9780 |
| 43–46 | 10 | pending | After seed 42 finishes |

Log: `/tmp/multirun_full.log`  
Ckpt: `model/multirun/ft_seed{seed}.pth`  
JSON: `benchmarks/results/multirun/ft_seed{seed}.json`

### Classical baselines (same protocol `stage_b_ft`, seed 42, **val only**)

| Model | Train data | Val macro-F1 | Min per-class F1 | Theft F1 | Notes |
|-------|------------|--------------|------------------|----------|-------|
| LR pilot | 100k subsample | 0.463 | 0.0 | — | RUN_DOCUMENTED pilot |
| RF pilot | 100k subsample | 0.686 | 0.0 | — | RUN_DOCUMENTED pilot |
| **LR full** | full train | **0.5231** | 0.0 | 0.0 | DONE |
| **RF full** | full train | **0.9778** | **0.9231** | **0.9231** | DONE (bal_acc 0.9957) |
| **XGB full** | full train | **0.9762** | **0.9231** | — | DONE |
| LGBM full | full train | pending | | | in flight |

**Important:** Published README RF **0.9864** uses `rf_baseline_processed` path — **not identical** to this protocol FT split. Compare apples-to-apples only within protocol tables. Do not silently replace 0.9864 with 0.9778.

### Next after jobs
1. Multirun summary mean±std  
2. LGBM + classical summary rewrite  
3. Imbalance loss compare (queued)  
4. Method package + HPO  

### Jobs / logs
- Multirun: `run_baseline_multirun`  
- Classical: `/tmp/classical_full.log`  
- Post-multirun: imbalance via watcher on `summary.json`  

### Multirun seed 42 full (completed)
- best val macro-F1 **0.9780** (epoch 3), early-stopped after epoch 5
- elapsed ~1702s (~28 min)
- history: ep1 0.9755, ep2 0.9684, ep3 **0.9780**, ep4 0.9722, ep5 0.9408
- seed 43 started

### Classical full (protocol stage_b_ft, val only)
| model | val_macro_f1 | min_cls_f1 | theft_f1 |
|-------|--------------|------------|----------|
| lr | 0.5231 | 0.0 | 0.0 |
| rf | 0.9778 | 0.9231 | 0.9231 |
| xgb | 0.9762 | 0.9231 | 0.9231 |
| lgbm | 0.4951 | 0.0 | 0.0 (re-run in flight) |

Note: RF 0.9778 on this protocol ≠ published 0.9864 (different feature/pipeline).


### Multirun update (live)
| seed | best val macro-F1 | elapsed | status |
|------|-------------------|---------|--------|
| 42 | **0.9780** | ~28 min | DONE (early stop ep6) |
| 43 | **0.9578** | ~25 min | DONE (early stop ep5) |
| 44 | in progress | — | RUNNING |
| 45–46 | pending | — | |

### Classical full (protocol stage_b_ft, val)
| model | val_macro_f1 | notes |
|-------|--------------|-------|
| lr | 0.523 | DONE |
| rf | 0.978 | DONE (≠ published 0.9864 path) |
| xgb | 0.976 | DONE |
| lgbm | 0.551 | DONE after DataFrame fix (still weak — RUN_DOCUMENTED) |
| svm pilot 150k | FAILED | &lt;3 samples/class — RUN_DOCUMENTED |


### Multirun seeds complete so far
| seed | best val macro-F1 | min_cls_f1 | elapsed |
|------|-------------------|------------|---------|
| 42 | 0.9780 | 0.9315 | 28 min |
| 43 | 0.9578 | 0.9091 | 25 min |
| 44 | **0.9840** | **0.9589** | 36 min |
| 45 | running | | |
| 46 | pending | | |

Partial mean (n=3): **0.9733** ± 0.0137 — desirable; seed 43 is the weak tail.


### Multirun WP1b COMPLETE
| seed | best val macro-F1 | min_cls_f1 |
|------|-------------------|------------|
| 42 | 0.9780 | 0.9315 |
| 43 | 0.9578 | 0.9091 |
| 44 | **0.9840** | **0.9589** |
| 45 | 0.9746 | 0.9143 |
| 46 | 0.9624 | 0.9091 |

**mean 0.9714 ± 0.0109** (n=5)  
min 0.9578 max 0.9840  
Path: `benchmarks/results/multirun/summary.json`  
Test sealed. Champion not overwritten.

### Next (auto-started)
Imbalance loss compare (ce → focal → focal_cb → logit_adj), 5 epochs, seed 42.
