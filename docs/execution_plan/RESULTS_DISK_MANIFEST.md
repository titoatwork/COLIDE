# Results Disk Manifest (handoff snapshot)

**Generated (UTC):** 2026-07-21T11:01:25.771483+00:00  
**Last append (UTC):** 2026-07-22T17:10:00 (DICC ops method lock: OnDemand VNC + screen + batch; no new train/result JSON)
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
| `benchmarks/results/baselines_classical/summary_handoff.json` | yes | `b93899d2d019f72d8deb97bef2de3b9e` | 2978 | lr=0.5231; svm=0.4268; rf=0.9778; xgb=0.9762; **lgbm=0.9818** |
| `benchmarks/results/baselines_classical/TABLE_VAL.json` | yes | `a97c155ccd879b55fb3273a0bc972061` | 250 | lr/svm/rf/xgb/lgbm val macros |
| `benchmarks/results/baselines_classical/lr_seed42.json` | yes | `5c68972f00fedbd1308f0f7b6d82d3ff` | 4885 | val.macro_f1=0.5231 |
| `benchmarks/results/baselines_classical/rf_seed42.json` | yes | `092d27bfc532041519bf950301042a7b` | 4885 | val.macro_f1=0.9778 |
| `benchmarks/results/baselines_classical/xgb_seed42.json` | yes | `88d92ab3007d27b36f2aa676b9b3aa35` | 4956 | val.macro_f1=0.9762 |
| `benchmarks/results/baselines_classical/lgbm_seed42.json` | yes | `575bf6ce9ede255d2a96d26f776588e6` | 5486 | val.macro_f1=**0.9818** (G5 fix) |
| `benchmarks/results/baselines_classical/svm_seed42.json` | yes | `86f8596204d75f39ef3db81a590458d1` | 5350 | val.macro_f1=**0.4268** (G2 full LinearSVC) |
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
| **B14 sealed multi-seed TEST mean±std** | **0.9780 ± 0.0033 (n=5)** | `sealed_test/summary.json` |
| B14 test min-cls / Theft means | 0.9292 / **1.0000** | `sealed_test/summary.json` |
| Protocol multirun FT mean±std val macro-F1 | 0.9714 ± 0.0109 (n=5) | `multirun/summary.json` |
| Multirun best seed | 0.9840 seed44 | `multirun/ft_seed44.json` |
| Loss compare winner | focal 0.9780 | `imbalance_loss/summary.json` |
| Val thresholds on focal seed42 | no gain vs argmax (all variants 0.9780); keep argmax | `imbalance_loss/thresholds_focal_seed42.json` |
| Protocol-fair RF val | 0.9778 | `baselines_classical/rf_seed42.json` |
| Protocol-fair XGB val | 0.9762 | `baselines_classical/xgb_seed42.json` |
| Protocol-fair LGBM val (G5 fix, balanced) | **0.9818** | `baselines_classical/lgbm_seed42.json` |
| Protocol-fair LinearSVC val (G2 full) | **0.4268** | `baselines_classical/svm_seed42.json` |
| D6 stratified batch (inv-freq) vs shuffle | stratified **0.9209** vs shuffle **0.9791** (Δ−0.058) RUN_DOCUMENTED | `stratified_batch/summary.json` |
| Published RF (different pipeline) | 0.9864 | historical / freeze card — not protocol-fair |
| Champion sealed val macro-F1 | 0.9780 | protocol eval JSON |
| Champion md5 | 80a90f7cc210276300eaa90173a5a385 | `model/best_model_botiot_twostage.pth` |
| WP4b best KD teacher (student val macro-F1) | ensemble **0.9401** | `teachers_kd/summary.json` |
| WP4b RF / none / XGB / LGBM student | 0.9346 / 0.9326 / 0.9270 / 0.8829 | `teachers_kd/kd_*_seed42.json` |
| WP3 HPO winner (full-train refine val macro-F1) | **0.9791** INCORPORATE | `hpo/summary.json` + `config/hpo_best.yaml` |
| WP3 Stage A best (subsampled train) | trial 11 **0.9787** | `hpo/summary.json` |
| HPO multi-seed confirm (orig distill + hpo_best) | **0.9689 ± 0.0145** n=5 RUN_DOCUMENTED | `multirun_hpo_confirm/summary.json` |
| HPO confirm seed42 (repro) | **0.9791** | `multirun_hpo_confirm/ft_seed42.json` |
| WP5b neural baselines top (G11 cnn_bilstm CE) | **0.9493** | `baselines_neural/summary.json` |
| WP5b protocol MLP (G6) | **0.9285** | `baselines_neural/G6_mlp_seed42.json` |
| WP5b transformer (G12) | **0.5808** weak | `baselines_neural/G12_transformer_seed42.json` |
| WP5c Pareto best F1 (protocol systems) | **A7 0.9699** @26.02 µs · 530181 params | `pareto/summary.json` |
| WP5c F1–latency front | A7 / A3 / G6 | `pareto/summary.json` |
| WP5c composite #1 | **G6 0.762** F1 0.9285 @4.33 µs | `pareto/summary.json` |
| DICC multi-day tree | ABSENT | no `benchmarks/results/dicc/` |
| **WP6b energy mJ/flow range (n=5)** | **0.920–0.943** (mean **0.933**) | `wp6b_local_ranges/summary.json` |
| **WP6b PT @bs=256 µs/sample range** | **24.15–25.68** (mean **24.90**) | same |
| **WP6b CUDA derived pipeline µs** | **565.2–570.3** | same (Option A) |
| **WP6b peak alloc MiB** | **322.2** | same |
| Historical single-shot energy | 0.786 mJ/flow | `energy_table/` HISTORICAL |

## WP6b local multi-session ranges (2026-07-22)

**Platform:** RTX 3050 laptop · sessions **5** · warm-up 50 · champion frozen  
**Script:** `scripts/run_wp6b_local_ranges.py` · wall ~147 s  
**Champion md5 unchanged** `80a90f7cc210276300eaa90173a5a385`

| Path | exists | md5 | bytes | key metrics |
|------|--------|-----|-------|-------------|
| `benchmarks/results/wp6b_local_ranges/summary.json` | yes | `34c3ca9ab60f611c8ed2de56fc10fd8f` | 23740 | energy 0.920–0.943; PT@256 24.15–25.68; CUDA 565–570 |
| `benchmarks/results/wp6b_local_ranges/table.md` | yes | — | — | markdown tables |
| `benchmarks/results/wp6b_local_ranges/sessions/session_0.json` | yes | — | — | per-session raw |
| `benchmarks/results/systems_i8_h3/summary.json` | yes | `17e42c269cef17f0d42c4ea14a366084` | 11468 | I8 multi-session mirror |
| `scripts/run_wp6b_local_ranges.py` | yes | — | — | harness |

**Trap:** Do not mix WP6b multi-session energy **0.933 mean** with historical single-shot **0.786**. Do not claim full CUDA vs full V3 PT parity. Local ≠ DICC multi-day.

## WP9c camera-ready manuscript draft (2026-07-22)

**Mode:** prose + figures only · no new result JSON · champion frozen  
**Numbers:** all from CLAIMS_REGISTRY / sealed_test / wp6b / classical / ablation / xai / toniot_final

| Path | exists | notes |
|------|--------|-------|
| `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` | yes | full local-complete prose draft |
| `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf` | yes | camera-ready local-complete PDF |
| `docs/manuscript/figures/fig_architecture.png` | yes | V3 CAD-CBA block diagram |
| `docs/manuscript/figures/fig_class_distribution.png` | yes | train imbalance + test support |
| `docs/manuscript/figures/fig_confusion_matrix_b14_seed42.png` | yes | sealed test CM representative |
| `docs/manuscript/figures/fig_ablation_ladder.png` | yes | A1–A7 |
| `docs/manuscript/figures/fig_detection_dual_bars.png` | yes | test vs classical dual bars |
| `docs/manuscript/figures/fig_wp6b_systems_ranges.png` | yes | energy + PT ranges |
| `docs/manuscript/figures/fig_pareto_f1_latency.png` | yes | copied from benchmarks/plots |
| `docs/manuscript/figures/fig_pareto_f1_params.png` | yes | copied from benchmarks/plots |
| `scripts/generate_manuscript_figures.py` | yes | regenerates class-dist + CM |

**Trap:** Draft is **local-complete**, not DICC-complete.

## PI venue polish (2026-07-22)

**Mode:** prose + PDF packaging only · no new result JSON · champion frozen  
**Builder:** `scripts/build_manuscript_pdf.py` (ReportLab)

| Path | exists | notes |
|------|--------|-------|
| `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` | yes | continuous abstract; Table 1b per-class means; Table 5b HPO refine; systems CI/CV; front matter placeholders |
| `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf` | yes | ~797 KB rebuild |
| `scripts/build_manuscript_pdf.py` | yes | reproducible PDF from markdown + figures |

**Table 1b means (from sealed_test ft_seed{42..46} test per_class F1):** DDoS **0.9838**, DoS **0.9813**, Normal **0.9292**, Recon **0.9958**, Theft **1.0000**.  
**Still open for PI:** journal class file / BibTeX after venue choice. DICC cells still TBD.

## DICC ops method lock (2026-07-22)

**Mode:** documentation only · no cluster SUCCESS tree yet  

| Path | exists | notes |
|------|--------|-------|
| `docs/DICC_OPS_METHOD.md` | yes | **Authoritative** ops: OnDemand VNC + screen + batch |
| `docs/execution_plan/04_PHASE0_DICC.md` | yes | Phase 0 points to ops method; campus/Cheran-default removed |
| `dicc_scripts/README.md` | yes | Connection method section added |
| `benchmarks/results/dicc/` | **no** | Still ABSENT — do not invent multi-day numbers |

**Superseded as primary plan:** campus-stable runner; Cheran-as-default cluster operator; long interactive `srun`/`salloc` over VPN.

## Playlist closure audit + claims hygiene (2026-07-22)

**Mode:** audit + claims packaging only · no new train · champion frozen  

| Path | exists | notes |
|------|--------|-------|
| `docs/execution_plan/PLAYLIST_CLOSURE_AUDIT.md` | yes | 133/133 tracker rows terminal; local playlist CLOSED |
| `docs/execution_plan/CLAIMS_REGISTRY.md` | yes | **64** claims (was 59); Table 1b `bot_sealed_test_pc_*` |
| `benchmarks/results/claims_package/protocol_claims.json` | yes | rebuilt; open_gates = DICC + PI venue only |
| `scripts/build_claims_package.py` | yes | derives Table 1b means from seed JSON only |

**Table 1b re-verify (seed means, 4 d.p.):** DDoS **0.9838**, DoS **0.9813**, Normal **0.9292**, Recon **0.9958**, Theft **1.0000** — match manuscript.  
**Claims count:** **64** · `verify_claims` green · champion **unchanged**.

## B14 sealed multi-seed BoT TEST (2026-07-22)

**User lock:** CAD-CBA-v1 init path **A** · distill + `hpo_best` FT · seeds 42–46 · `--allow-test`  
**Wall:** ~4768 s · Champion **unchanged** `80a90f7cc210276300eaa90173a5a385`

| Path | exists | md5 | bytes | key metrics |
|------|--------|-----|-------|-------------|
| `benchmarks/results/sealed_test/summary.json` | yes | `8958860625b484c5c84eb15da1f9ea3f` | 10194 | test **0.9780±0.0033** n=5; Theft 1.0 |
| `benchmarks/results/sealed_test/ft_seed42.json` | yes | `aac59b6957baf9c506c3905fa1a22163` | 10563 | test 0.9787 |
| `benchmarks/results/sealed_test/ft_seed43.json` | yes | `1182641b36155a0af7ecc501134befca` | 12130 | test 0.9798 |
| `benchmarks/results/sealed_test/ft_seed44.json` | yes | `fe2ba2f58e4048393e844898311aee30` | 10472 | test 0.9798 |
| `benchmarks/results/sealed_test/ft_seed45.json` | yes | `799063fd215e95e35364bf35b77cea60` | 9907 | test 0.9722 |
| `benchmarks/results/sealed_test/ft_seed46.json` | yes | `0ee6af4e1778f86dc99f4e5ae089fd67` | 11819 | test 0.9796 |
| `model/sealed_test/ft_seed42.pth` | yes | `3d32ffc15d2301331c10a9c9ee9a7aa3` | 2133122 | bit-identical to hpo_confirm s42 |
| `model/sealed_test/ft_seed43.pth` | yes | `88fede576071f4f5cb4d9b29b96d810f` | 2133122 | |
| `model/sealed_test/ft_seed44.pth` | yes | `3a3dc725519a62e7c2547b76f3a4ee7a` | 2133122 | |
| `model/sealed_test/ft_seed45.pth` | yes | `dec1a5c68da1574223ae203f21659446` | 2133122 | |
| `model/sealed_test/ft_seed46.pth` | yes | `ed550532175be43e78802c03d384c6c5` | 2133122 | |

**Trap:** Test mean 0.9780 ≠ val multirun claims. Label **test** explicitly. Seed46 val 0.9483 with test 0.9796 — report both.

## WP5c Pareto (2026-07-22)

| Path | exists | md5 | bytes | key metrics |
|------|--------|-----|-------|-------------|
| `benchmarks/results/pareto/summary.json` | yes | `893645611534c7ed681e2846d3c80246` | 19545 | H8 DONE; 14 points; fronts A7/A3/G6 |
| `benchmarks/results/pareto/table.md` | yes | `a80d4f6c4b90fb6a060d795c51f0c4c9` | 4742 | markdown tables |
| `benchmarks/plots/pareto_f1_latency.png` | yes | `dff020a141fe8196fac926c1be1ddf81` | 66653 | F1 vs µs/sample |
| `benchmarks/plots/pareto_f1_params.png` | yes | `77d378127ecc81af174fcc1cfd2a834f` | 69302 | F1 vs n_params |
| `scripts/run_pareto_wp5c.py` | yes | `79a09e4f2b7b8bd6a9a3d2c3318339a8` | — | consolidator (no retrain) |

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

## WP5a Ablation ladder A1–A7 (2026-07-21)

**Stage:** `stage_b_ft` · **seed 42** · epochs≤8 · patience=3 · val-only · test sealed · arch variants under CAD-CBA dims  
**Script:** `scripts/run_ablation_ladder.py` + `model/ablation_variants.py`  
**Tag:** `ablation_ladder/` (does **not** clobber champion, WP1b `multirun/`, `multirun_hpo_confirm/`, or `multirun_ensemble_hpo/`)  
**Wall:** ~5397 s (~90 min)  
**Champion md5 unchanged** `80a90f7cc210276300eaa90173a5a385`

### Result JSON

| Path | md5 | bytes | key metrics |
|------|-----|-------|-------------|
| `benchmarks/results/ablation_ladder/summary.json` | `988b826adcea79ef51f4b8144055825e` | 7297 | n=7 success; ranking A7 best |
| `benchmarks/results/ablation_ladder/A1_cnn_only_seed42.json` | `25217b8cd6f4880467cd4db71fbb950b` | 10743 | 0.6221 |
| `benchmarks/results/ablation_ladder/A2_bilstm_only_seed42.json` | `b6d531c18ed5de970e4258d5d4de3ba3` | 10920 | 0.8058 |
| `benchmarks/results/ablation_ladder/A3_cnn_bilstm_seed42.json` | `bb5321e4f652df44f8145c3f07f91c78` | 10879 | 0.9493 |
| `benchmarks/results/ablation_ladder/A4_cnn_bilstm_attn_ce_seed42.json` | `499dd3b423ed87d199cec5db74c81d3a` | 10718 | 0.7378 |
| `benchmarks/results/ablation_ladder/A5_cnn_bilstm_attn_focal_seed42.json` | `e310822a0b168fac972c6086e74ee223` | 11015 | 0.8684 |
| `benchmarks/results/ablation_ladder/A6_attn_focal_ensemble_kd_seed42.json` | `9658ff43007771f28b1b0a06feeac070` | 10219 | 0.9346 |
| `benchmarks/results/ablation_ladder/A7_full_cad_cba_v1_seed42.json` | `8f2bf9a76dfeef66f4582c991558c9da` | 11070 | **0.9699** |

### Checkpoints

| Path | md5 | bytes |
|------|-----|-------|
| `model/ablation_ladder/A1_cnn_only_seed42.pth` | `fd8944a8ed262df7c9554cd46dc26e93` | 147810 |
| `model/ablation_ladder/A2_bilstm_only_seed42.pth` | `beff44489bababcce461a4445785919f` | 1493736 |
| `model/ablation_ladder/A3_cnn_bilstm_seed42.pth` | `4f5e2f57a251131e4e9bb8bcd16b2d4f` | 1866342 |
| `model/ablation_ladder/A4_cnn_bilstm_attn_ce_seed42.pth` | `c62d5407e1f384f650db0169aedb79a5` | 2133794 |
| `model/ablation_ladder/A5_cnn_bilstm_attn_focal_seed42.pth` | `62e494d861393f953937d04658279c12` | 2133890 |
| `model/ablation_ladder/A6_attn_focal_ensemble_kd_seed42.pth` | `7a2f05930ce499bf80c549d5357dcc6c` | 2133922 |
| `model/ablation_ladder/A7_full_cad_cba_v1_seed42.pth` | `43129f48bc1c3c070f83b55b8c34b396` | 2133698 |

### Ranking (val macro-F1) + systems (batch=256 latency)

| Rank | Row | Name | val macro-F1 | Min-cls | Theft | params | µs/sample | Decision |
|------|-----|------|--------------|---------|-------|--------|-----------|----------|
| 1 | **A7** | full CAD-CBA-v1 (attn+focal+ens KD+HPO) | **0.9699** | 0.8974 | 1.0000 | 530181 | 26.02 | **RUN_DOCUMENTED** ladder top; package path wins incremental table |
| 2 | A3 | cnn_bilstm CE scratch | 0.9493 | 0.8571 | 1.0000 | 463877 | 19.96 | RUN_DOCUMENTED strong backbone w/o attn |
| 3 | A6 | attn+focal+ensemble KD (default HPs) | 0.9346 | 0.8462 | 1.0000 | 530181 | 26.39 | RUN_DOCUMENTED KD lift vs A5 |
| 4 | A5 | attn+focal scratch | 0.8684 | 0.7059 | 0.7059 | 530181 | 26.69 | RUN_DOCUMENTED focal helps vs A4 |
| 5 | A2 | bilstm_only CE | 0.8058 | 0.5000 | 0.5000 | 372229 | 15.26 | RUN_DOCUMENTED |
| 6 | A4 | attn+CE scratch | 0.7378 | 0.0000 | 0.0000 | 530181 | 24.54 | RUN_DOCUMENTED **attn+CE underperforms A3** under this budget |
| 7 | A1 | cnn_only CE | 0.6221 | 0.0000 | 0.0000 | 34821 | 5.03 | RUN_DOCUMENTED weak |

**Incremental story (honest):** A1≪A2≪A3; A4 (attn+CE) **hurts** vs A3 under seed42/8-ep budget; A5 focal recovers; A6 ensemble KD lifts further; A7 HPO HPs top the ladder at **0.9699**. Do **not** claim attention alone is free gain. Do **not** mix single-seed A7 with WP1b multirun mean 0.9714 or package multirun mean 0.9639.

**Trap:** Ladder is seed **42 only**, epochs≤8 — not multi-seed stability. Systems latency is RTX 3050 batch-256 sync forward, not full energy table (F9 energy still open).

## WP5b Protocol-fair neural baselines G6–G12 (2026-07-22)

**Stage:** `stage_b_ft` · **seed 42** · epochs≤8 · patience=3 · **CE** · scratch · val-only · test sealed  
**Script:** `scripts/run_neural_baselines.py` + `model/neural_baselines.py`  
**Shared HPs:** lr=1e-3 Adam · batch=512 · equal budget (G15) · no per-baseline Optuna  
**Tag:** `baselines_neural/` (does **not** clobber champion, multirun trees, or `ablation_ladder/`)  
**Wall:** ~4816 s (~80 min)  
**Champion md5 unchanged** `80a90f7cc210276300eaa90173a5a385`

### Result JSON

| Path | md5 | bytes | key metrics |
|------|-----|-------|-------------|
| `benchmarks/results/baselines_neural/summary.json` | `dc85077cb129c3209d6f6148c18e925b` | 8155 | n=7; G11 best 0.9493 |
| `benchmarks/results/baselines_neural/G6_mlp_seed42.json` | `c2310b0fbc0201c95557baa702ce9cab` | 10993 | 0.9285 |
| `benchmarks/results/baselines_neural/G7_cnn1d_seed42.json` | `5d9b4c98cabd7eafb51bf588f0563ba0` | 10895 | 0.6221 |
| `benchmarks/results/baselines_neural/G8_lstm_seed42.json` | `e622891ba268f0e797a2fabfb0e81f3c` | 11018 | 0.8099 |
| `benchmarks/results/baselines_neural/G9_bilstm_seed42.json` | `b88dc4d44623f071a2bcc6a62cb01dc0` | 11070 | 0.8058 |
| `benchmarks/results/baselines_neural/G10_cnn_lstm_seed42.json` | `6e1a3c491fd8e02237437d3787109a13` | 11033 | 0.8159 |
| `benchmarks/results/baselines_neural/G11_cnn_bilstm_seed42.json` | `04eef1820823f96046b00f32f5ba3803` | 11064 | **0.9493** |
| `benchmarks/results/baselines_neural/G12_transformer_seed42.json` | `b2419843e5ecbfce8241e22094bf372b` | 10716 | 0.5808 |

### Checkpoints

| Path | md5 | bytes |
|------|-----|-------|
| `model/baselines_neural/G6_mlp_seed42.pth` | `eeacb365a384164c33c24571414dfc66` | 1607048 |
| `model/baselines_neural/G7_cnn1d_seed42.pth` | `a3d5e9cee346856a0badbb883079a142` | 147738 |
| `model/baselines_neural/G8_lstm_seed42.pth` | `02bca182152d6aa6f3616c1796648129` | 618388 |
| `model/baselines_neural/G9_bilstm_seed42.pth` | `3c135ed63bdaa12ef157e09becc26430` | 1493676 |
| `model/baselines_neural/G10_cnn_lstm_seed42.pth` | `62d8631f98f4710984382d70fa540723` | 859660 |
| `model/baselines_neural/G11_cnn_bilstm_seed42.pth` | `1c34bd88eda022b0d1b4ab8d23feb945` | 1866368 |
| `model/baselines_neural/G12_transformer_seed42.pth` | `ef3ea08d7abed4ca2ffdd2744918b19c` | 432710 |

### Ranking (val macro-F1) + systems (batch=256 latency)

| Rank | Row | Name | val macro-F1 | Min-cls | Theft | params | µs/sample | Decision |
|------|-----|------|--------------|---------|-------|--------|-----------|----------|
| 1 | **G11** | cnn_bilstm | **0.9493** | 0.8571 | 1.0000 | 463877 | 20.12 | **RUN_DOCUMENTED** suite top (= WP5a A3) |
| 2 | G6 | mlp | 0.9285 | 0.7077 | 1.0000 | 400901 | 4.33 | RUN_DOCUMENTED protocol-fair (≠ historical 0.962) |
| 3 | G10 | cnn_lstm | 0.8159 | 0.5000 | 0.5000 | 212485 | 16.25 | RUN_DOCUMENTED |
| 4 | G8 | lstm | 0.8099 | 0.3556 | 0.8000 | 153605 | 10.39 | RUN_DOCUMENTED |
| 5 | G9 | bilstm | 0.8058 | 0.5000 | 0.5000 | 372229 | 16.01 | RUN_DOCUMENTED (= WP5a A2) |
| 6 | G7 | cnn1d | 0.6221 | 0.0000 | 0.0000 | 34821 | 6.15 | RUN_DOCUMENTED (= WP5a A1) |
| 7 | G12 | transformer | 0.5808 | 0.0000 | 0.0000 | 105221 | 10.72 | RUN_DOCUMENTED **weak under equal budget** |

**G15:** All rows share fixed HPs; CAD-CBA Optuna is separate (not applied to baselines). See `g15_hpo_budget_note` in summary JSON.

**Trap:** Do not mix historical `ablation_mlp.json` / `mlp_twostage.json` with G6. Do not claim transformer beat CNN–BiLSTM. Suite is seed42 / CE / scratch only — not multi-seed package path.

## G2 SVM full + G5 LGBM fix (2026-07-22)

**Stage:** `stage_b_ft` · **seed 42** · full train · val-only · test sealed  
**Script:** `scripts/run_classical_baselines.py`  
**Champion md5 unchanged** `80a90f7cc210276300eaa90173a5a385`

### Result JSON

| Path | md5 | bytes | key metrics |
|------|-----|-------|-------------|
| `benchmarks/results/baselines_classical/svm_seed42.json` | `86f8596204d75f39ef3db81a590458d1` | 5350 | LinearSVC 0.4268 |
| `benchmarks/results/baselines_classical/lgbm_seed42.json` | `575bf6ce9ede255d2a96d26f776588e6` | 5486 | **0.9818** G5 fix |
| `benchmarks/results/baselines_classical/summary_handoff.json` | `b93899d2d019f72d8deb97bef2de3b9e` | 2978 | full G1–G5 table |
| `benchmarks/results/baselines_classical/TABLE_VAL.json` | `a97c155ccd879b55fb3273a0bc972061` | 250 | val macros |
| `benchmarks/results/baselines_classical/summary.json` | `ab97f197e573c1a41f1417898d3900f8` | 1943 | last invocation svm+lgbm only |

### Ranking (protocol-fair classical, val macro-F1)

| Rank | Model | val macro-F1 | Min-cls | Theft | Decision |
|------|-------|--------------|---------|-------|----------|
| 1 | **LGBM (fixed)** | **0.9818** | 0.9231 | 0.9231 | **DONE (val)** G5 |
| 2 | RF | 0.9778 | 0.9231 | 0.9231 | DONE (val) |
| 3 | XGB | 0.9762 | 0.9231 | 0.9231 | DONE (val) |
| 4 | LR | 0.5231 | 0.0000 | 0.0000 | DONE (val) |
| 5 | LinearSVC | 0.4268 | 0.0000 | 0.0000 | RUN_DOCUMENTED G2 weak |

**Trap:** Prefer `summary_handoff.json` over `summary.json`. LGBM official number is **0.9818** (class_weight=balanced fix); legacy 0.5512 is not the paper figure. SVM is hard-label LinearSVC (no CalibratedClassifierCV). LGBM used class_weight=balanced; RF/XGB did not — document that difference.

## D6 stratified batch compare (2026-07-22)

**Stage:** `stage_b_ft` · **seed 42** · epochs≤8 · patience=3 · focal · hpo_best · val-only  
**Init:** `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth`  
**Script:** `scripts/run_stratified_batch_compare.py` + `--train-sampler`  
**Wall:** ~1479 s · Champion unchanged

### Result JSON / checkpoints

| Path | md5 | bytes | key metrics |
|------|-----|-------|-------------|
| `benchmarks/results/stratified_batch/summary.json` | `294669c7a7b2e3d318b54186a0ce917c` | 2334 | Δ−0.058 RUN_DOCUMENTED |
| `benchmarks/results/stratified_batch/ft_shuffle_seed42.json` | `75d225921e48010e4ff873aa213335ed` | 7534 | **0.9791** |
| `benchmarks/results/stratified_batch/ft_stratified_seed42.json` | `ad1542c83d5e6b54c310821e4feb2bcd` | 8102 | 0.9209 |
| `model/stratified_batch/ft_shuffle_seed42.pth` | `27089188f2369751e02aa083e391812b` | 2133378 | |
| `model/stratified_batch/ft_stratified_seed42.pth` | `803f35c27dad15f5cb7ead81c2d05751` | 2133474 | |

### Ranking

| Rank | Sampler | val macro-F1 | Min-cls | Theft | Decision |
|------|---------|--------------|---------|-------|----------|
| 1 | **shuffle** | **0.9791** | 0.9351 | 1.0000 | keep default |
| 2 | stratified (inv-freq WeightedRandomSampler) | 0.9209 | 0.7500 | 0.7500 | RUN_DOCUMENTED hurts |

**Trap:** Do not claim stratified batch helps under CAD-CBA-v1 HPs. Shuffle control also reproduces WP3 HPO seed42 0.9791.

## G13 lightweight IDS note

| Path | md5 | note |
|------|-----|------|
| `docs/execution_plan/G13_LIGHTWEIGHT_IDS_NOTE.md` | `36f3b501a7017cc7d2f0cd6d648d6eea` | RUN_DOCUMENTED N/A |



## Bounded C* playlist (2026-07-22)

**Stage:** `stage_b_ft` · seed 42 · epochs≤8 · val-only · test sealed  
**Script:** `scripts/run_bounded_cstar.py` · variants `model/method_variants.py`  
**Tag:** `cstar_bounded/` · Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`

| Path | md5 | bytes | key |
|------|-----|-------|-----|
| `benchmarks/results/cstar_bounded/summary.json` | `498241338cb114ae4010809302386191` | 6903 | all RUN_DOCUMENTED |
| `benchmarks/results/cstar_bounded/CTRL_control_v3_focal_seed42.json` | `c378542a216b50300d3774f1c07b6c33` | 10235 | **0.9787** control |
| `benchmarks/results/cstar_bounded/C4_multi_scale_seed42.json` | `be1be008713af43ab56da7dbb3f04bbc` | 10702 | 0.9167 |
| `benchmarks/results/cstar_bounded/C5_gated_fusion_seed42.json` | `dc6c8c8435da487015abf587905d383d` | 10095 | 0.9132 |
| `benchmarks/results/cstar_bounded/C7_supcon_focal_seed42.json` | `8878bb3ccc892a68833d4951f38f3a3a` | 10456 | 0.7732 |
| `benchmarks/results/cstar_bounded/C8_asymmetric_seed42.json` | `07955b3863ca6d894c03d141dfe1e0e8` | 9923 | 0.8012 |
| `benchmarks/results/cstar_bounded/C10_uncertainty_mc_dropout_seed42.json` | `1ca9c691bf4e1b3a18b8bfff08286b19` | 11889 | det 0.9791 no lift |

**Decisions:** C4/C5/C7/C8/C10 **RUN_DOCUMENTED** — none beat CTRL; not in CAD-CBA-v1.

## E6 neural teacher KD (2026-07-22)

**Stage:** `stage_a_kd` · seed 42 · G11 teacher · V3 student · α=0.6 T=10 γ=2  
**Script:** `scripts/run_neural_teacher_kd.py` · wall ~1436 s

| Path | md5 | bytes | key |
|------|-----|-------|-----|
| `benchmarks/results/teachers_kd_neural/summary.json` | `e40d997e4cd8b2e4566eae2e2d59a287` | 763 | student **0.8513** RUN_DOCUMENTED |
| `benchmarks/results/teachers_kd_neural/kd_neural_cnn_bilstm_seed42.json` | `0b951c8bb5d24de0021242431192abd0` | 11080 | vs ensemble 0.9401 |

**Decision:** RUN_DOCUMENTED — keep ensemble teacher INCORPORATED.

## WP5c systems rebench pareto_h8 (2026-07-22)

Complements analysis-only `pareto/`. Script: `scripts/run_pareto_multiobj.py`.

| Path | md5 | bytes | key |
|------|-----|-------|-----|
| `benchmarks/results/pareto_h8/summary.json` | `699f86e93a815b53f07f16f60366b33a` | 48256 | composite G6 0.9056 |
| `benchmarks/results/pareto_h8/multiobj_table.csv` | `a305965ae422b175bcb379de61a1c97c` | 5825 | table |
| `benchmarks/results/pareto_h8/pareto_f1_latency.png` | `837011fd738b407690697ac47d483e81` | 111321 | figure |
| `benchmarks/results/pareto_h8/rows.json` | `7a95d68cece0544abb2f097d05b50f3a` | 30505 | raw rows |

**Headline claims:** LGBM tops protocol F1 0.9818; package/HPO near-RF; G6 wins a priori composite; classical CPU latency ≠ GPU µs — label carefully.

## B2–B4 plateau reject

| Path | note |
|------|------|
| `docs/execution_plan/B2B4_ARCH_HPO_PLATEAU_REJECT.md` | RUN_DOCUMENTED freeze V3 dims |

## WP7 XAI suite (2026-07-22)

| Path | md5 | bytes | key |
|------|-----|-------|-----|
| `benchmarks/results/xai/summary.json` | `4e1d869af8cc994db31637603e4a5f5a` | 17264 | J10 DROP_FULL keep structured |
| `benchmarks/results/xai/table.md` | `d28ffb49824520e7168ce5b61b4e0830` | 1236 | markdown |
| `benchmarks/results/xai/structured_examples.json` | `d559139080e8ed418779bc37ee7b28e6` | 5820 | templates |

**Headlines:** occlusion top3 min/stddev/max; faith mass 0.5109; rank corr 0.9636; LLM feature-mention 0.333; dispatch 16.60 µs; gen ~7400 ms.

## F9 energy table (2026-07-22)

| Path | md5 | bytes | key |
|------|-----|-------|-----|
| `benchmarks/results/energy_table/summary.json` | `ae6c9b74a81e199c3abd8e28cb52c761` | 8514 | RTX ~0.786 mJ/flow |
| `benchmarks/results/energy_table/table.md` | `3c271e0174991ab7b83260635192a3fe` | 916 | markdown |

## WP8 ToN final method (2026-07-22)

| Path | md5 | bytes | key |
|------|-----|-------|-----|
| `benchmarks/results/toniot_final/summary.json` | `a30268122037008d59a36214b0108882` | 29433 | val 0.8080 test 0.8110 |
| `benchmarks/results/toniot_final/table.md` | `e0dd394f20e95bc99b9cc4cc6e74f5e8` | 601 | markdown |
| `benchmarks/results/toniot_final/classical_rf.json` | `1f8d05492a6e38a8d8d759d62a9aae76` | 5212 | RF test 0.9393 |
| `benchmarks/results/toniot_final/summary_pilot_lowlr.json` | `6479aa2a1dfb869ecde9b95a8c6608f7` | 28707 | pilot under-tuned archive |
| `model/toniot_final/kd_ensemble_seed42.pth` | `b258d4f822b4c27b8378e0539b30e611` | 2135458 | selected final |
| `model/toniot_final/ft_cad_cba_v1_seed42.pth` | `b258d4f822b4c27b8378e0539b30e611` | 2135458 | = KD (FT no lift) |

**Trap:** Do not mix with historical clean 26-feat CNN 0.9526. This is 13-feat `processed_toniot`.

## WP6a re-export + fidelity (2026-07-22)

| Path | md5 | bytes | key |
|------|-----|-------|-----|
| `benchmarks/results/numerical_fidelity.json` | `73d043da90ffc6f67ca28f39cc2f8c7e` | 2608 | bit-identical + CUDA PASS |
| `benchmarks/results/wp6_reexport/summary.json` | (local) | — | DONE re-export |

## WP9a claims packaging (2026-07-22)

| Path | exists | md5 | key |
|------|--------|-----|-----|
| `scripts/build_claims_package.py` | yes | (source) | rebuilds claims from disk JSON |
| `benchmarks/results/claims_package/protocol_claims.json` | yes | (local gitignored) | 42 claims + minority tables |
| `benchmarks/results/claims_package/table.md` | yes | (local) | markdown twin |
| `docs/execution_plan/CLAIMS_REGISTRY.md` | yes | (committed) | committed numbers registry |
| `docs/execution_plan/FINAL_CONFIG_FREEZE_CARD.md` | yes | (committed) | B14 gate — AWAITING USER LOCK |
| `scripts/verify_claims.py` | yes | (source) | historical + protocol claims green |

**Decision: DONE (packaging)** — sealed-test / DICC claims remain PENDING/BLOCKED rows in the registry. Re-run builder after B14.

**Last append (UTC):** 2026-07-22T11:55:00 (WP9a)

## WP9b manuscript spine (2026-07-22) — docs only

No new `benchmarks/results/` JSON this session (prose + tracker hygiene only).

| Path | exists | key |
|------|--------|-----|
| `docs/execution_plan/WP9b_MANUSCRIPT_SPINE.md` | yes | title T1; abstract; RQs; core tables; ToV |
| `docs/execution_plan/13_PHASE9_MANUSCRIPT.md` | yes | gate checklist updated |
| Champion md5 | unchanged | `80a90f7cc210276300eaa90173a5a385` |
| Claims package | 59 green | no rebuild required (no new numbers) |

**Trap:** Spine tables copy locked registry numbers only — do not invent DICC multi-day cells.
