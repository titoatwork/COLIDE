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

## Fill These After Setup
Run and paste output here:
nvidia-smi
nvcc --version (if installed)
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print(torch.cuda.get_device_properties(0))"

## DICC Cluster (June 1–29)
- **GPU:** TBC — run nvidia-smi on arrival
- **SM Version:** TBC
- **Queue System:** TBC (SLURM or PBS)
- **CUDA Toolkit:** TBC
- **Internet on compute nodes:** TBC

## Reproducibility
- All experiments seeded: seed=42
- Seed set in: numpy, torch, python random — at top of every script
- Virtual environment: .venv (WSL2, not committed to git)
- Config: config/config.yaml (committed, version controlled)
