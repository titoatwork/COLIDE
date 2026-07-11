# Cluster campaign scripts (portable SLURM)

Hardened multi-day GPU latency campaigns. **No hardcoded site paths** (`/scr/...`),
**no hardcoded hostnames** (`gpu05`/`gpu06`). Works on DICC, LSU Rostam, or any
SLURM cluster with a writable checkout and `nvcc` + conda/modules.

## Quick start (any cluster)

```bash
# You are already inside a COLIDE checkout (clone or rsync — any path is fine)
cd /path/to/COLIDE
export COLIDE_ROOT="$PWD"   # optional; scripts default to this tree

# 1) Setup once on a login/build node (no GPU required)
bash dicc_scripts/01_setup.sh
# single-arch example (only V100 kernels):
# bash dicc_scripts/01_setup.sh --targets sm_70:v100

# 2) Day 1 submit — set YOUR site's partition/account if required
bash dicc_scripts/submit_session.sh \
  --targets v100,a100 \
  --campaign core \
  --date "$(date -u +%Y%m%d)" \
  --partition YOUR_PARTITION \    # omit if default partition is fine
  --account YOUR_ACCOUNT \        # omit if not required
  --gres gpu:1

# 3) Day 2 (next UTC day): SAME commit, do NOT re-run 01_setup.sh
bash dicc_scripts/submit_session.sh \
  --targets v100,a100 \
  --campaign core \
  --date "$(date -u +%Y%m%d)" \
  --partition YOUR_PARTITION \
  --account YOUR_ACCOUNT

# 4) Compare after both days have SUCCESS markers
module load miniconda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate colide
export PYTHONPATH=.
python scripts/compare_dicc_sessions.py --gpu v100s --date-a D1 --date-b D2
python scripts/compare_dicc_sessions.py --gpu a100  --date-a D1 --date-b D2
```

Discover site knobs:

```bash
sinfo -o "%P %G %l %D" | head
module avail 2>&1 | head
```

Optional: `cp dicc_scripts/site.env.example dicc_scripts/site.env` and edit defaults.

## Layout

| Path | Role |
|------|------|
| `01_setup.sh` | Repo-local setup: conda, atomic kernel compile + SHA256 |
| `submit_session.sh` | Preferred entry: absolute logs, `--chdir`, multi-profile |
| `job_benchmark.sh` | Generic SLURM body (no site hostnames) |
| `profiles/*.env` | GPU identity + kernel subdir (v100, a100, …) |
| `site.env.example` | Partition/account/gres template |
| `02`/`03` | Back-compat wrappers → same runner |
| `lib/run_benchmark.sh` | Measurement body |
| `validate/local_validate.sh` | Offline suite |

## Result isolation

```text
benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/
  manifest.json  environment.txt  kernel_SHA256SUMS
  cuda_kernel_stats.json  pytorch_gpu_stats.json
  raw/  logs/  exit_status  SUCCESS
```

## Adding a new GPU profile

1. Compile: `bash dicc_scripts/01_setup.sh --targets sm_86:rtx`
2. Create `profiles/rtx.env` (label, name regex, CC, min mem, kernels subdir).
3. Submit: `bash dicc_scripts/submit_session.sh --targets rtx --constraint …`

## Comparability

- Full CUDA-vs-PyTorch pipeline speedup is **not valid** until architecture parity.
- Block 3 is the preferred head-to-head.
- Stable multi-day means: “consistent with WSL2-specific drift,” not proof.

## Local validation

```bash
bash dicc_scripts/validate/local_validate.sh
```
