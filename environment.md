# COLIDE — Environment Documentation

## Local Development Machine (WSL2)
- **Owner:** Ibteshamul Haque
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **SM Version:** 8.6 (Ampere)
- **VRAM:** 6GB
- **Driver Version:** 595.97
- **CUDA Version:** 13.2 (system) / 12.1 (PyTorch bundled)
- **PyTorch Version:** [fill after install]
- **Python Version:** [fill after install]
- **OS:** Ubuntu (WSL2) on Windows
- **Known measurement issue:** real session-to-session latency drift beyond within-session CV
  (see README "Measurement Stability"). Cross-check on DICC before treating local ranges as
  universal.

## Fill These After Setup (local)
Run and paste output here:
```
nvidia-smi
nvcc --version (if installed)
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

## Any SLURM cluster (portable)

Scripts under `dicc_scripts/` are **site-agnostic**:
- `COLIDE_ROOT` defaults to the checkout that contains `dicc_scripts/` (no `/scr` hardcode)
- No fixed `--nodelist` (gpu05/gpu06 removed)
- Resources via `submit_session.sh --partition/--account/--gres/--constraint` or `site.env`
- GPU profiles in `dicc_scripts/profiles/{v100,a100}.env`

| Item | Value |
|------|--------|
| Operator guide | `dicc_scripts/README.md` |
| Setup | `bash dicc_scripts/01_setup.sh` (login node; no nvidia-smi required) |
| Submit | `bash dicc_scripts/submit_session.sh --targets v100,a100 …` |
| Kernels | `inference/kernels/<profile>/` via `--targets sm_70:v100,sm_80:a100` |
| Results | `benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/` |
| Compare | `scripts/compare_dicc_sessions.py` |
| Checkpoint | `model/best_model_botiot_twostage.pth` |
| Conda env | `colide` |

**Historical DICC (UM) note:** older runs used `/scr/$USER/colide` and nodelists gpu05/gpu06.
New campaigns must not depend on those paths. Fill exact product names/driver from each run’s
`environment.txt` after the first success on a given site.

## Reproducibility
- All training/eval experiments seeded: seed=42 (`config/config.yaml` global default)
- Seed set in: numpy, torch, python random — at top of every training script
- Local virtual environment: `.venv` (WSL2, not committed)
- Cluster: conda env `colide` created by `01_setup.sh`
- Config: `config/config.yaml` (committed)
- DICC latency campaigns: isolate by job dir + require identical git SHA and kernel `SHA256SUMS`
  across Day 1 / Day 2 before accepting numbers
