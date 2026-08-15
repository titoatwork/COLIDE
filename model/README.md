# COLIDE tracked weights

**Date:** 2026-08-15  
**Rule:** Do not clobber the BoT champion without backup + explicit approval.  
**Identity config:** [`config/champion.json`](../config/champion.json) · [`config/paths.py`](../config/paths.py)

Hashes below: only the **known frozen champion MD5** is listed. Do not invent other checksums here.

---

## CURRENT champion (BoT-IoT)

| File | Status | MD5 |
|------|--------|-----|
| [`best_model_botiot_twostage.pth`](best_model_botiot_twostage.pth) | **CURRENT** production champion (CAD-CBA-v1 two-stage) | `80a90f7cc210276300eaa90173a5a385` |

Principal sealed multi-seed test macro-F1 **0.9780 ± 0.0033** is evaluated against this identity. Verify with `scripts/verify_champion.py` / `tests/test_champion_hash.py`.

---

## CURRENT ToN corrected

| File | Status | Note |
|------|--------|------|
| [`toniot_corrected/cnn_hardlabel_seed42.pth`](toniot_corrected/cnn_hardlabel_seed42.pth) | **CURRENT** leakage-safe ToN CNN | Protocol `toniot_leakage_safe_v1`; results in `benchmarks/results/toniot_corrected/` |

No MD5 is recorded here (none frozen in `config/champion.json`).

---

## INVALID

| File | Status | Why |
|------|--------|-----|
| [`best_model_toniot_clean.pth`](best_model_toniot_clean.pth) | **INVALID** | **DATA-TON-001** — target-derived `label` in features; do not use for manuscript tables |

Related tombstones: `benchmarks/results/toniot_clean_comparison.json`, `toniot_clean_retrain.json` (and `.INVALIDATED` copies). Retrain of this path is fail-fast unless `COLIDE_ALLOW_INVALID_TON=1`.

---

## HISTORICAL (development, not principal)

Kept for audit / ablation provenance. **Not** the published champion and **not** the corrected ToN path.

### Distill / backup (explicitly non-principal)

- `best_model_botiot_distill.pth` and `best_model_botiot_distill_*.pth` (α / T / focal sweeps)
- `best_model_botiot_twostage_BACKUP_0.9639.pth`
- `best_model_botiot_twostage_BACKUP_0.9790_s5.pth` — historical single-run **0.9790** development only
- `best_model_toniot_distill.pth`, `best_model_toniot_distill_v2.pth`

### Other development checkpoints on disk

| Group | Paths |
|-------|--------|
| Early / alt BoT | `best_model.pth`, `best_model_botiot_ensemble.pth`, `best_model_botiot_optimized.pth`, `best_model_botiot_mlp_mlp.pth`, `best_model_botiot_mlp_twostage.pth` |
| Older ToN (not corrected; not the invalid clean path) | `best_model_toniot.pth`; `toniot_final/*.pth` (comparable prior only) |
| Ablation ladder | `ablation_ladder/*.pth` |
| Neural baselines | `baselines_neural/*.pth` |
| Bounded C* | `cstar_bounded/*.pth` |
| HPO | `hpo/*.pth`, `hpo_smoke/*.pth` |
| Imbalance / KD teachers | `imbalance_loss/*.pth`, `teachers_kd/*.pth`, `teachers_kd_neural/*.pth` |
| Multi-seed FT campaigns | `multirun/*.pth`, `multirun_ensemble_hpo/*.pth`, `multirun_hpo_confirm/*.pth`, `sealed_test/*.pth`, `stratified_batch/*.pth` |

Stage-1 KD init commonly used before fine-tune (not the champion): `best_model_botiot_distill_a0.6_T10.0_focal2.pth` (`config.paths.STAGE1_KD_PATH`).

---

## Not PyTorch checkpoints

`weights/`, `weights_bin/`, `colide_model.onnx`, `model.onnx` are export / kernel-load artifacts, not training champions.

Python sources in this directory (`cnn_bilstm*.py`, `ablation_variants.py`, …) define architectures; they are not weights.
