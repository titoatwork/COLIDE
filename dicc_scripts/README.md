# DICC cluster scripts

Hardened Slurm workflow for multi-day V100 / A100 latency campaigns.

## Why this layout

Previous scripts mixed relative log paths, shared fixed JSON filenames, soft
conda activation, and non-strict CUDA stats. Concurrent V100/A100 jobs could
overwrite each other, and failed commands could still print “Complete.”

This tree isolates every run, fails closed, and separates submission from
measurement.

## Layout

| Path | Role |
|------|------|
| `01_setup.sh` | Clone/ff-only pull, conda env, atomic V100/A100 kernel compile + SHA256 |
| `02_benchmark_v100.sh` | Thin SBATCH resource wrapper (gpu05 / V100) |
| `03_benchmark_a100.sh` | Thin SBATCH resource wrapper (gpu06 / A100) |
| `04_nsight_profile.sh` | Opt-in Nsight on A100 (prefer `--with-nsight`) |
| `05_run_all.sh` | Back-compat → `submit_session.sh` |
| `submit_session.sh` | Preferred entry: absolute logs, `--chdir`, session manifest |
| `lib/common.sh` | Conda init, GPU asserts, run-dir / manifest helpers |
| `lib/run_benchmark.sh` | Shared measurement body (CUDA strict stats + PyTorch harness) |
| `validate/local_validate.sh` | Offline suite (bash -n, shellcheck, mocks) |

## Cluster usage

```bash
# On login node (no GPU required for compile)
bash dicc_scripts/01_setup.sh
export COLIDE_ROOT=/scr/$USER/colide

# Day 1 (UTC date label defaults to today)
bash dicc_scripts/submit_session.sh --campaign core
# Optional Nsight after A100 succeeds:
# bash dicc_scripts/submit_session.sh --campaign core --with-nsight

# Day 2 — same commit and binaries, new date
bash dicc_scripts/submit_session.sh --campaign core --date YYYYMMDD
```

Results land under:

```text
benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/
  manifest.json
  environment.txt
  kernel_SHA256SUMS
  cuda_kernel_stats.json
  pytorch_gpu_stats.json
  raw/
  logs/
  exit_status
  SUCCESS          # only on clean exit
```

Never write legacy fixed names like `pipeline_benchmark.json` or
`statistical_confidence.json` from this path.

## Cross-day compare

```bash
cd $COLIDE_ROOT
PYTHONPATH=. python scripts/compare_dicc_sessions.py \
  --gpu v100s --date-a YYYYMMDD --date-b YYYYMMDD
PYTHONPATH=. python scripts/compare_dicc_sessions.py \
  --gpu a100  --date-a YYYYMMDD --date-b YYYYMMDD
```

Rejected if dates match, `SUCCESS` missing, or git SHA / kernel checksums /
checkpoint / protocol / GPU identity differ.

Stable DICC means are described as **consistent with WSL2-specific drift**, not
as proof. Update README/paper numbers only from accepted cross-day artifacts.

## Comparability (read this)

- **Full CUDA vs PyTorch pipeline speedup is not valid** until architecture
  parity: V3 PyTorch runs attention + LayerNorm + global average pool; CUDA
  implements none of those; `fused_pipeline.cu` skips Block 3.
- **Block 3** (BiLSTM + last timestep) is a valid head-to-head.
- Per-block 1/2/4 comparisons remain usable; full-model PyTorch latency is
  still collected for absolute numbers.

## Local validation (no cluster)

```bash
bash dicc_scripts/validate/local_validate.sh
```

## Related Python entry points

- `scripts/benchmark_cuda_kernels_stats.py` — default `n=100`, `--strict`, raw samples, CIs
- `scripts/benchmark_pytorch_gpu_stats.py` — 20×1000, production checkpoint, comparability flags
- `scripts/compare_dicc_sessions.py` — cross-day gate + Welch / Cohen’s d
