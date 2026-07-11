# Cluster campaign scripts (portable SLURM)

Hardened multi-day GPU latency campaigns. **No hardcoded site paths**, **no
hardcoded hostnames**. Works on DICC, LSU Rostam, laptops, or any SLURM site.

## One command (preferred)

```bash
cd /path/to/COLIDE
bash dicc_scripts/run_campaign.sh --full
```

That **one line** does Day 1 → wait → Day 2 → wait → cross-day compare (as far as the
site allows): setup Python, detect partitions/GRES, compile kernels, submit both
GPU classes, then compare SUCCESS runs. Day-2 date label is `<day>_d2` so it works
even on the same calendar day.

```bash
# Day 1 only / Day 2 only / options:
bash dicc_scripts/run_campaign.sh
bash dicc_scripts/run_campaign.sh --day 2
bash dicc_scripts/run_campaign.sh --wait
bash dicc_scripts/run_campaign.sh --targets v100
bash dicc_scripts/run_campaign.sh --local
bash dicc_scripts/run_campaign.sh --dry-run
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
