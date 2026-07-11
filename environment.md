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

## DICC Cluster (Universiti Malaya)
- **Queue system:** SLURM
- **Repo checkout (scratch):** `/scr/$USER/colide` (`COLIDE_ROOT`)
- **Operator guide:** `dicc_scripts/README.md`
- **Branch for hardened campaign:** `final-polish` (from `ba6e0cb` onward)
- **V100 job:** `dicc_scripts/02_benchmark_v100.sh` — `#SBATCH --nodelist=gpu05`, label `v100s`,
  kernels `inference/kernels/v100/` (`sm_70`), expect name matching `V100`, CC `7.0`
- **A100 job:** `dicc_scripts/03_benchmark_a100.sh` — `#SBATCH --nodelist=gpu06`, label `a100`,
  kernels `inference/kernels/a100/` (`sm_80`), expect name matching `A100`, CC `8.0`
- **Submit entrypoint:** `dicc_scripts/submit_session.sh` (not raw `sbatch 02` alone)
- **Setup:** `dicc_scripts/01_setup.sh` (login/build node; **does not require** `nvidia-smi`)
- **Conda env name:** `colide` (minimal: torch, numpy, scipy, pyyaml, scikit-learn)
- **Production checkpoint used by harness:** `model/best_model_botiot_twostage.pth`
- **Results root:** `benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/`
- **Cross-day compare:** `scripts/compare_dicc_sessions.py`
- **Nsight:** opt-in via `submit_session.sh --with-nsight` (depends on A100 `afterok`)
- **CUDA toolkit module (setup default):** `cuda/12.1` if available (override via `module load`)
- **Internet on compute nodes:** TBC — setup assumes packages installable on login node
- **Fill after first successful run:** paste `nvidia-smi` / driver / exact GPU product names from
  each run’s `environment.txt` into this section so paper hardware tables stay honest.

## Reproducibility
- All training/eval experiments seeded: seed=42 (`config/config.yaml` global default)
- Seed set in: numpy, torch, python random — at top of every training script
- Local virtual environment: `.venv` (WSL2, not committed)
- Cluster: conda env `colide` created by `01_setup.sh`
- Config: `config/config.yaml` (committed)
- DICC latency campaigns: isolate by job dir + require identical git SHA and kernel `SHA256SUMS`
  across Day 1 / Day 2 before accepting numbers
