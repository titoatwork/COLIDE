# Results Disk Manifest (handoff snapshot)

**Generated (UTC):** 2026-07-21T11:01:25.771483+00:00  
**Last append (UTC):** 2026-07-21T11:09:20 (WP2d val thresholds)  
**Host path root:** `/home/titoisalive/colide`

`benchmarks/results/` is **gitignored**. This file is committed so the next session can verify local artifacts without inventing numbers.

## Champion (never clobber)

| Path | md5 | bytes |
|------|-----|-------|
| `model/best_model_botiot_twostage.pth` | `80a90f7cc210276300eaa90173a5a385` | 2133730 |

## Result JSON inventory

| Path | exists | md5 | bytes | key metrics |
|------|--------|-----|-------|-------------|
| `benchmarks/results/multirun/summary.json` | yes | `4ef4a675666d0d4139b708216837b50e` | 2141 | mean=0.9714±0.0109 n=5 |
| `benchmarks/results/multirun/ft_seed42.json` | yes | `e7f69c98d3e8e37be174f63a994e81ca` | 6508 | best_val_macro_f1=0.9780 |
| `benchmarks/results/multirun/ft_seed43.json` | yes | `6812ddec5d9f100af86a2c878fc898db` | 6393 | best_val_macro_f1=0.9578 |
| `benchmarks/results/multirun/ft_seed44.json` | yes | `77556518d6adde11282256c51a3f0a3a` | 7461 | best_val_macro_f1=0.9840 |
| `benchmarks/results/multirun/ft_seed45.json` | yes | `25473186e1e970fdd316354b922ec60b` | 6312 | best_val_macro_f1=0.9746 |
| `benchmarks/results/multirun/ft_seed46.json` | yes | `9ab013bfe50fa062bb7c918868b1a773` | 6870 | best_val_macro_f1=0.9624 |
| `benchmarks/results/imbalance_loss/summary.json` | yes | `7682957d5fa237210f7cb804e031971e` | 1710 | best=focal 0.9780 |
| `benchmarks/results/imbalance_loss/ft_ce_seed42.json` | yes | `059a4a60e2ff69e8ab13e9c65a7f139f` | 6282 | best_val_macro_f1=0.9755 |
| `benchmarks/results/imbalance_loss/ft_focal_seed42.json` | yes | `09744f7e5c6022e1183d4570b5c2a12a` | 6332 | best_val_macro_f1=0.9780 |
| `benchmarks/results/imbalance_loss/ft_focal_cb_seed42.json` | yes | `637c2d985a5ef74e93d6659c7fd8bcf1` | 6334 | best_val_macro_f1=0.9121 |
| `benchmarks/results/imbalance_loss/ft_logit_adj_seed42.json` | yes | `56143d80bc55cbc86f27ea8721f311c6` | 6081 | best_val_macro_f1=0.9225 |
| `benchmarks/results/imbalance_loss/thresholds_focal_seed42.json` | yes | `2a6bc98d967883efc53d326535cf9d5b` | 34114 | WP2d; all variants=argmax 0.9780; RUN_DOCUMENTED |
| `benchmarks/results/baselines_classical/summary_handoff.json` | yes | `b4faee6d15c5c53cbc59faa44df89993` | 1786 | lr=0.5231; rf=0.9778; xgb=0.9762; lgbm=0.5512 |
| `benchmarks/results/baselines_classical/TABLE_VAL.json` | yes | `6ff4f624ae0b3c8336e7f1abd733ac50` | 1248 | lr=0.5231; rf=0.9778; xgb=0.9762; lgbm=0.4951 |
| `benchmarks/results/baselines_classical/lr_seed42.json` | yes | `5c68972f00fedbd1308f0f7b6d82d3ff` | 4885 | val.macro_f1=0.5231 |
| `benchmarks/results/baselines_classical/rf_seed42.json` | yes | `092d27bfc532041519bf950301042a7b` | 4885 | val.macro_f1=0.9778 |
| `benchmarks/results/baselines_classical/xgb_seed42.json` | yes | `88d92ab3007d27b36f2aa676b9b3aa35` | 4956 | val.macro_f1=0.9762 |
| `benchmarks/results/baselines_classical/lgbm_seed42.json` | yes | `af2820a196f33fe56bf5d6ae091e14fb` | 4896 | val.macro_f1=0.5512 |
| `benchmarks/results/protocol/eval_best_model_botiot_twostage_stage_b_ft.json` | yes | `40f1009cf8a5d39fd983e0f9c0e6874c` | 4448 |  |
| `benchmarks/results/protocol/botiot_protocol_smoke.json` | yes | `36633eb8620af1c6e25830f98ef4a8c2` | 2608 |  |

## Checkpoints (local, typically not committed)

| Path | exists | md5 | bytes |
|------|--------|-----|-------|
| `model/imbalance_loss/ft_ce_seed42.pth` | yes | `70d0fea813983cb5c0b70d7148a4dc9d` | 2133218 |
| `model/imbalance_loss/ft_focal_cb_seed42.pth` | yes | `a38c1f4d0df9c7e7d5bce2f454ed8521` | 2133410 |
| `model/imbalance_loss/ft_focal_seed42.pth` | yes | `170eaccc584ba12cce2a34ca52ebfbf2` | 2133314 |
| `model/imbalance_loss/ft_logit_adj_seed42.pth` | yes | `c9e126ffe76532c2e4f1431f3c9faa74` | 2133442 |
| `model/multirun/ft_seed42.pth` | yes | `f2afe112f0456cb93b4cc96727c03c88` | 2133122 |
| `model/multirun/ft_seed43.pth` | yes | `7dd67f66f9714b24a364084175602126` | 2133122 |
| `model/multirun/ft_seed44.pth` | yes | `80add9e119d30becdcd9ac8bbb3e9a45` | 2133122 |
| `model/multirun/ft_seed45.pth` | yes | `2a612ae93886ceedd6e1dc7e0b0fc26c` | 2133122 |
| `model/multirun/ft_seed46.pth` | yes | `739a9aaf38a04cb385ace9a6b562a6c1` | 2133122 |
| `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` (multirun init) | yes | `4b7accd4e0a42905ac1b51b80302a85c` | 2134274 |

## Headline numbers (copy for claims only with these sources)

| Claim | Value | Source |
|-------|-------|--------|
| Protocol multirun FT mean±std val macro-F1 | 0.9714 ± 0.0109 (n=5) | `multirun/summary.json` |
| Multirun best seed | 0.9840 seed44 | `multirun/ft_seed44.json` |
| Loss compare winner | focal 0.9780 | `imbalance_loss/summary.json` |
| Val thresholds on focal seed42 | no gain vs argmax (all variants 0.9780); keep argmax | `imbalance_loss/thresholds_focal_seed42.json` |
| Protocol-fair RF val | 0.9778 | `baselines_classical/rf_seed42.json` |
| Protocol-fair XGB val | 0.9762 | `baselines_classical/xgb_seed42.json` |
| Published RF (different pipeline) | 0.9864 | historical / freeze card — not protocol-fair |
| Champion sealed val macro-F1 | 0.9780 | protocol eval JSON |
| Champion md5 | 80a90f7cc210276300eaa90173a5a385 | `model/best_model_botiot_twostage.pth` |
| DICC multi-day tree | ABSENT | no `benchmarks/results/dicc/` |

## Trap warnings

1. `baselines_classical/summary.json` may only contain the **last** model run (LGBM). Use **`summary_handoff.json`** or individual `*_seed42.json`.
2. `TABLE_VAL.json` may disagree slightly with later LGBM re-runs (e.g. 0.495 vs 0.551). Prefer individual `lgbm_seed42.json` / `summary_handoff.json`.
3. Do **not** mix published RF 0.9864 with protocol multirun numbers.
4. All above metrics are **val-only** unless a JSON explicitly allows test.
5. Model `.pth` under `model/multirun/` and `model/imbalance_loss/` are local artifacts; re-run if missing on another machine.
6. `scripts/watch_and_queue_next.sh` is a finished one-shot helper (multirun→imbalance); not an active job.

