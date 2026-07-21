# Results Disk Manifest (handoff snapshot)

**Generated (UTC):** 2026-07-21T11:01:25.771483+00:00  
**Last append (UTC):** 2026-07-21T20:21:00 (multi-seed HPO confirm)
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
| WP4b best KD teacher (student val macro-F1) | ensemble **0.9401** | `teachers_kd/summary.json` |
| WP4b RF / none / XGB / LGBM student | 0.9346 / 0.9326 / 0.9270 / 0.8829 | `teachers_kd/kd_*_seed42.json` |
| WP3 HPO winner (full-train refine val macro-F1) | **0.9791** INCORPORATE | `hpo/summary.json` + `config/hpo_best.yaml` |
| WP3 Stage A best (subsampled train) | trial 11 **0.9787** | `hpo/summary.json` |
| HPO multi-seed confirm (orig distill + hpo_best) | **0.9689 ± 0.0145** n=5 RUN_DOCUMENTED | `multirun_hpo_confirm/summary.json` |
| HPO confirm seed42 (repro) | **0.9791** | `multirun_hpo_confirm/ft_seed42.json` |
| DICC multi-day tree | ABSENT | no `benchmarks/results/dicc/` |

## Trap warnings

1. `baselines_classical/summary.json` may only contain the **last** model run (LGBM). Use **`summary_handoff.json`** or individual `*_seed42.json`.
2. `TABLE_VAL.json` may disagree slightly with later LGBM re-runs (e.g. 0.495 vs 0.551). Prefer individual `lgbm_seed42.json` / `summary_handoff.json`.
3. Do **not** mix published RF 0.9864 with protocol multirun numbers.
4. All above metrics are **val-only** unless a JSON explicitly allows test.
5. Model `.pth` under `model/multirun/` and `model/imbalance_loss/` are local artifacts; re-run if missing on another machine.
6. `scripts/watch_and_queue_next.sh` is a finished one-shot helper (multirun→imbalance); not an active job.

## WP4b Teacher/KD compare (2026-07-21)

**Stage:** `stage_a_kd` · **seed 42** · **α=0.6 T=10 γ=2** · epochs≤10 patience=4 batch=512 · val-only · test sealed  
**Scripts:** `scripts/train_protocol_kd.py`, `scripts/run_teacher_kd_compare.py`  
**Wall:** ~6879 s (~1.9 h)

### Result JSON

| Path | md5 | bytes | key metrics |
|------|-----|-------|-------------|
| `benchmarks/results/teachers_kd/summary.json` | `63ab4bd3f40e24adc6788fa1ca255bd8` | 5105 | best=ensemble 0.9401 |
| `benchmarks/results/teachers_kd/kd_none_seed42.json` | `379df6cbd7aa34e418d35efa0f5acb02` | 8743 | student 0.9326 |
| `benchmarks/results/teachers_kd/kd_rf_seed42.json` | `836473b48d812b47bb604fb4460c0ad0` | 8602 | student 0.9346; teacher 0.9750 |
| `benchmarks/results/teachers_kd/kd_xgb_seed42.json` | `2d056371b3e60a1d131bf098ba06e633` | 9534 | student 0.9270; teacher 0.9918 |
| `benchmarks/results/teachers_kd/kd_lgbm_seed42.json` | `513749956a9248f6fe8c5828098f9695` | 9294 | student 0.8829; teacher 0.5928 |
| `benchmarks/results/teachers_kd/kd_ensemble_seed42.json` | `03257954837840f5ccb7c8ceeff7303a` | 10097 | student **0.9401**; teacher 0.9803 |

### Checkpoints

| Path | md5 | bytes |
|------|-----|-------|
| `model/teachers_kd/kd_none_a0.6_T10.0_g2.0_seed42.pth` | `6ce196e582ce7d79244e9de9270c5b71` | 2133858 |
| `model/teachers_kd/kd_rf_a0.6_T10.0_g2.0_seed42.pth` | `fc0610beb6c28ccfd17d8d59b6b360d9` | 2133794 |
| `model/teachers_kd/kd_xgb_a0.6_T10.0_g2.0_seed42.pth` | `bc90b9dad63c1e39a115d1e07b0ed538` | 2133826 |
| `model/teachers_kd/kd_lgbm_a0.6_T10.0_g2.0_seed42.pth` | `a9101db8f22d551b3a81e48c099f9811` | 2133858 |
| `model/teachers_kd/kd_ensemble_a0.6_T10.0_g2.0_seed42.pth` | `d1eff65eda2a7bac26523cc2952742dd` | 2133986 |

### Ranking (student val macro-F1)

| Rank | Teacher | Student | Min-cls | Theft | Teacher val | Decision |
|------|---------|---------|--------|-------|-------------|----------|
| 1 | **ensemble** | **0.9401** | 0.8434 | 0.9231 | 0.9803 | **INCORPORATE** CAD-CBA-v1 KD teacher |
| 2 | rf | 0.9346 | 0.8000 | 1.0000 | 0.9750 | RUN_DOCUMENTED (simple fallback) |
| 3 | none | 0.9326 | 0.8409 | 0.9231 | — | RUN_DOCUMENTED (hard-label control) |
| 4 | xgb | 0.9270 | 0.8434 | 0.8571 | 0.9918 | RUN_DOCUMENTED (strong teacher, weaker student) |
| 5 | lgbm | 0.8829 | 0.7059 | 0.7059 | 0.5928 | RUN_DOCUMENTED (weak teacher) |

**Note:** These are **stage_a_kd from-scratch** student numbers (not stage_b FT). Do not mix with multirun FT mean 0.9714. Champion md5 unchanged.

## WP3 Optuna HPO (2026-07-21)

**Stage:** `stage_b_ft` · **seed 42** · val-only · test sealed · arch fixed V3  
**Script:** `scripts/hpo_optuna_botiot.py`  
**Stage A:** 20 trials, epochs≤4, patience=2, max_train=400000 stratified, full val · 11 COMPLETE / 9 PRUNED  
**Stage B:** top-3 full train, epochs≤8, patience=3  
**Wall:** ~69 min total  
**Decision:** **INCORPORATE** winner 0.9791 (Δ+0.0010 vs multirun seed42 baseline 0.9780)

### Result JSON / config

| Path | md5 | bytes | key metrics |
|------|-----|-------|-------------|
| `benchmarks/results/hpo/summary.json` | `5ba39a920706100b13975e89c3b20924` | 24094 | winner 0.9791 INCORPORATE |
| `benchmarks/results/hpo/top10_trials.json` | `ad5141befae0dfa55f389f7da72a9967` | 6452 | Stage A ranked |
| `benchmarks/results/hpo/refine_rank2_trial008_seed42.json` | `4248a1417b89515ea49207e1cf2033bd` | 6206 | full-train winner JSON |
| `benchmarks/results/hpo/study.db` | (sqlite) | 155648 | Optuna storage |
| `config/hpo_best.yaml` | `598d87c9ea5d0f26847ce7b860a0eb68` | 1013 | CAD-CBA-v1 train HPs |

### Winner checkpoint

| Path | md5 | bytes |
|------|-----|-------|
| `model/hpo/refine_rank2_trial008_seed42.pth` | `f9360aec2c003815140823cfe9b2a386` | 2135710 |

### Winner params (exact)

| HP | Value |
|----|-------|
| lr | 5.89306076111462e-05 |
| batch_size | 1024 |
| focal_gamma | 1.9166447754858478 |
| dropout_rate | 0.14783769837532068 |
| attention_dropout | 0.21397343616689848 |
| weight_decay | 0.00019158219548093185 |
| scheduler | cosine |

### Stage B refine ranking (full train)

| Rank | Trial | val macro-F1 | Min-cls | Theft | Decision |
|------|-------|--------------|---------|-------|----------|
| 1 | 8 | **0.9791** | 0.9351 | 1.0000 | **INCORPORATE** |
| 2 | 11 | 0.9721 | 0.9014 | 1.0000 | RUN_DOCUMENTED |
| 3 | 13 | 0.8656 | 0.5000 | 0.5000 | RUN_DOCUMENTED |

**Trap:** Stage A used max_train=400k; do not claim Stage A scores as full-train. Winner is Stage B full-train only. Champion not overwritten.

## Package FT multirun — ensemble KD + HPO HPs (2026-07-21)

**Stage:** `stage_b_ft` · seeds 42–46 · val-only · test sealed · arch fixed V3  
**Init:** `model/teachers_kd/kd_ensemble_a0.6_T10.0_g2.0_seed42.pth` (WP4b ensemble)  
**Train HPs:** `config/hpo_best.yaml` (WP3 winner)  
**Scripts:** `scripts/train_protocol_ft.py` (HPO-aware), `scripts/run_package_ft_multirun.py`  
**Tag:** `multirun_ensemble_hpo/` (WP1b `multirun/` intentionally untouched)  
**Wall:** ~5062 s (~84 min)  
**Champion md5 unchanged** `80a90f7cc210276300eaa90173a5a385`

### Result JSON

| Path | md5 | bytes | key metrics |
|------|-----|-------|-------------|
| `benchmarks/results/multirun_ensemble_hpo/summary.json` | `1fa206e34c50e799d531f5eee70629e8` | 7396 | mean **0.9639 ± 0.0185** n=5 |
| `benchmarks/results/multirun_ensemble_hpo/ft_seed42.json` | `b605b08bd52b09d1a95b40883079d95f` | 7820 | 0.9741 |
| `benchmarks/results/multirun_ensemble_hpo/ft_seed43.json` | `24025857074d4573ea0838046007167e` | 8743 | 0.9328 |
| `benchmarks/results/multirun_ensemble_hpo/ft_seed44.json` | `0405e3280a748bcec3f3c5996662bc83` | 8806 | 0.9699 |
| `benchmarks/results/multirun_ensemble_hpo/ft_seed45.json` | `2d0fc39a3c56b13f996527ecdae70867` | 8522 | **0.9803** |
| `benchmarks/results/multirun_ensemble_hpo/ft_seed46.json` | `3857772b192ceb8b232baf02208f8845` | 9038 | 0.9623 |

### Checkpoints

| Path | md5 | bytes |
|------|-----|-------|
| `model/multirun_ensemble_hpo/ft_seed42.pth` | `bda58953cd3c1990a10a72de28f51d39` | 2133122 |
| `model/multirun_ensemble_hpo/ft_seed43.pth` | `02b046ea55c7a77f6461c9850193e9b8` | 2133122 |
| `model/multirun_ensemble_hpo/ft_seed44.pth` | `34fbcfda4c215aa5b71ea31cb7065791` | 2133122 |
| `model/multirun_ensemble_hpo/ft_seed45.pth` | `66e9167359934261dca48d9a58f98b27` | 2133122 |
| `model/multirun_ensemble_hpo/ft_seed46.pth` | `e2674fbd45ab34bdd6628850a3839a65` | 2133122 |

### Per-seed ranking

| Seed | val macro-F1 | Min-cls | Theft |
|------|--------------|---------|-------|
| 45 | **0.9803** | 0.9474 | 1.0000 |
| 42 | 0.9741 | 0.9333 | 1.0000 |
| 44 | 0.9699 | 0.8947 | 1.0000 |
| 46 | 0.9623 | 0.9091 | 0.9091 |
| 43 | 0.9328 | 0.8000 | 0.8000 |

**Mean 0.9639 ± 0.0185** (n=5) · min-cls mean 0.8969 · Theft mean 0.9418  

**Comparators:** WP1b multirun (old distill + default HPs) **0.9714 ± 0.0109**; WP3 HPO seed42 refine **0.9791**.  

**Decision: RUN_DOCUMENTED** — full CAD-CBA-v1 path (ensemble KD init + HPO train HPs) is multi-seed stable in the mid-0.96 band, but **mean does not beat WP1b** and **std is higher** (seed 43 weak). HPO was tuned on a different init; do not claim package multirun as an improvement over WP1b mean. Components (ensemble teacher, HPO HPs, focal) remain individually INCORPORATED; aggregate package mean is evidence, not a new champion.

**Trap:** Do not overwrite WP1b `multirun/summary.json` with package numbers. Do not mix stage_a KD 0.9401 with these stage_b FT scores.

## Multi-seed HPO confirm — original distill + hpo_best (2026-07-21)

**Stage:** `stage_b_ft` · seeds 42–46 · val-only · test sealed · arch fixed V3  
**Init:** `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` (same as WP3 Optuna)  
**Train HPs:** `config/hpo_best.yaml`  
**Scripts:** `scripts/run_hpo_multiseed_confirm.py`, `scripts/train_protocol_ft.py`  
**Tag:** `multirun_hpo_confirm/` (WP1b + package trees untouched)  
**Wall:** ~2981 s (~50 min)  
**Champion md5 unchanged** `80a90f7cc210276300eaa90173a5a385`

### Result JSON

| Path | md5 | bytes | key metrics |
|------|-----|-------|-------------|
| `benchmarks/results/multirun_hpo_confirm/summary.json` | `ef6c92a592474c321a4c1300e19a8065` | 3698 | mean **0.9689 ± 0.0145** n=5 |
| `benchmarks/results/multirun_hpo_confirm/ft_seed42.json` | `1337c4283be9751ff73412b09788afac` | 7501 | **0.9791** (WP3 repro) |
| `benchmarks/results/multirun_hpo_confirm/ft_seed43.json` | `9312b580b88916e303a3f94bb8d2ce4f` | 9067 | 0.9587 |
| `benchmarks/results/multirun_hpo_confirm/ft_seed44.json` | `c4d09884caddc68862e326560de58e25` | 7456 | **0.9797** |
| `benchmarks/results/multirun_hpo_confirm/ft_seed45.json` | `9f728994c53e9c7be2c4cb9312dca472` | 6854 | 0.9787 |
| `benchmarks/results/multirun_hpo_confirm/ft_seed46.json` | `d984ecf3d86befe09d783c56ea7f9dbc` | 8756 | 0.9483 |

### Checkpoints

| Path | md5 | bytes |
|------|-----|-------|
| `model/multirun_hpo_confirm/ft_seed42.pth` | `3d32ffc15d2301331c10a9c9ee9a7aa3` | 2133122 |
| `model/multirun_hpo_confirm/ft_seed43.pth` | `88fede576071f4f5cb4d9b29b96d810f` | 2133122 |
| `model/multirun_hpo_confirm/ft_seed44.pth` | `3a3dc725519a62e7c2547b76f3a4ee7a` | 2133122 |
| `model/multirun_hpo_confirm/ft_seed45.pth` | `dec1a5c68da1574223ae203f21659446` | 2133122 |
| `model/multirun_hpo_confirm/ft_seed46.pth` | `ed550532175be43e78802c03d384c6c5` | 2133122 |

### Per-seed ranking

| Seed | val macro-F1 | Min-cls | Theft |
|------|--------------|---------|-------|
| 44 | **0.9797** | 0.9367 | 1.0000 |
| 42 | **0.9791** | 0.9351 | 1.0000 |
| 45 | 0.9787 | 0.9333 | 1.0000 |
| 43 | 0.9587 | 0.9091 | 0.9091 |
| 46 | 0.9483 | 0.8333 | 0.8333 |

**Mean 0.9689 ± 0.0145** (n=5) · min-cls mean 0.9095 · Theft mean 0.9485  

**Comparators:** WP3 HPO seed42 **0.9791** (reproduced); WP1b **0.9714 ± 0.0109**; package ensemble+HPO **0.9639 ± 0.0185**.  

**Decision: RUN_DOCUMENTED** — multi-seed aggregate does **not** beat WP1b mean; higher std (seed46 weak). Seed42 exact repro of WP3 winner supports keeping train HPs **INCORPORATED**. Not a new champion.

**Trap:** Do not mix with package `multirun_ensemble_hpo/` (different init). Do not claim multi-seed HPO mean > WP1b.

