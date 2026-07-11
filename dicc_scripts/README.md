# Cluster campaign scripts (portable SLURM)

Hardened multi-day GPU latency campaigns. **No hardcoded site paths**, **no
hardcoded hostnames**. Works on DICC, LSU Rostam, laptops, or any SLURM site.

## One command (preferred)

```bash
cd /path/to/COLIDE
bash dicc_scripts/run_campaign.sh
```

That single script will, as far as the site allows:

1. Create `.venv-cluster` and install torch (cu121 — V100 + A100 compatible)
2. Detect SLURM partitions (`cuda-V100`, `cuda-A100`, `gpu`, …)
3. Auto-omit `--gres` when `sinfo` reports GRES=(null) (Rostam pattern)
4. Compile kernels (on a GPU node via sbatch if login has no `nvcc`)
5. Submit Day‑1 V100/A100 (or detected) benchmark jobs

```bash
# After Day 1 SUCCESS markers, on a later UTC day (same checkout, no reinstall):
bash dicc_scripts/run_campaign.sh --day 2

# Optional:
bash dicc_scripts/run_campaign.sh --wait          # poll until jobs finish
bash dicc_scripts/run_campaign.sh --targets v100  # one GPU class only
bash dicc_scripts/run_campaign.sh --local         # run on this node (needs GPU)
bash dicc_scripts/run_campaign.sh --dry-run
```

Compare after two successful days:

```bash
source .venv-cluster/bin/activate
export PYTHONPATH=.
python scripts/compare_dicc_sessions.py --gpu v100s --date-a YYYYMMDD --date-b YYYYMMDD
python scripts/compare_dicc_sessions.py --gpu a100  --date-a YYYYMMDD --date-b YYYYMMDD
```

Overrides when auto-detect is wrong:

```bash
export COLIDE_V100_PARTITION=cuda-V100
export COLIDE_A100_PARTITION=cuda-A100
export COLIDE_SBATCH_GRES=            # force no gres
export COLIDE_SBATCH_ACCOUNT=myacct
bash dicc_scripts/run_campaign.sh
```

## Layout

| Path | Role |
|------|------|
| **`run_campaign.sh`** | **One-command entry (use this)** |
| `01_setup.sh` | Python env + kernel compile (called by run_campaign) |
| `submit_session.sh` | SLURM submit helper |
| `job_benchmark.sh` | Generic job body |
| `profiles/*.env` | GPU identity (v100, a100) |
| `lib/run_benchmark.sh` | Measurement body |
| `validate/local_validate.sh` | Offline suite |

## Result isolation

```text
benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/
  manifest.json  environment.txt  kernel_SHA256SUMS
  cuda_kernel_stats.json  pytorch_gpu_stats.json
  raw/  logs/  exit_status  SUCCESS
```

## Comparability

- Full CUDA-vs-PyTorch pipeline speedup is **not valid** until architecture parity.
- Block 3 is the preferred head-to-head.
- Stable multi-day means: “consistent with WSL2-specific drift,” not proof.

## Local validation

```bash
bash dicc_scripts/validate/local_validate.sh
```
