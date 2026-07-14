# COLIDE — DICC Cluster Runbook (Session 9)

**Purpose:** Multi-day V100S + A100 re-benchmark so the paper can replace “n/a” same-hardware PyTorch baselines and check whether session-to-session latency drift is WSL2-specific.

**Who does what**
- **You (human):** SSH to DICC, run setup once, `sbatch` on **≥2 separate calendar days**, `scp` results home.
- **Agent (Session 10):** Ingest JSON → update README cross-hardware table + `verify_claims.py`.

Do **not** invent V100/A100 ratios until those files exist on disk in the laptop repo.

---

## 0. Prerequisites

- DICC account with access to nodes used in the SLURM scripts:
  - V100: `gpu05` (`dicc_scripts/02_benchmark_v100.sh`)
  - A100: `gpu06` (`dicc_scripts/03_benchmark_a100.sh`)
- GitHub access to clone/pull `titoatwork/COLIDE` (or your fork) onto `/scr/$USER`.
- Local laptop has this repo at `~/colide` (or equivalent) for pull-back.

If node names changed, edit the `#SBATCH --nodelist=` lines **before** submitting.

---

## 1. Day 0 — one-time setup (login node / interactive)

```bash
# From a DICC login node
cd /scr/$USER
# Prefer the repo script (idempotent pull + compile sm_70 + sm_80):
bash /path/to/colide/dicc_scripts/01_setup.sh
# If scripts are only inside the clone after first clone:
#   git clone https://github.com/titoatwork/COLIDE.git colide
#   cd colide && bash dicc_scripts/01_setup.sh
```

What `01_setup.sh` does:
- Clone or `git pull` into `/scr/$USER/colide`
- Create/activate conda env `colide`, install Python deps
- Compile all 7 kernels into `inference/kernels/v100/` (`sm_70`) and `.../a100/` (`sm_80`)
- **No extra `-O3`** (must match Dockerfile / local methodology)

Sanity:

```bash
cd /scr/$USER/colide
ls inference/kernels/v100/fused_block3_fp16 inference/kernels/a100/fused_pipeline
nvidia-smi   # on a GPU node if you have interactive access
```

**Push local Session 4–7 commits first** if the cluster should see the relative-path fix and job-id tagging (`git push` from laptop, then `git pull` on DICC). Cluster must be at least at commit `d8ccd20` or later (ideally current `master`).

---

## 2. Day 1 — first measurement session

From `/scr/$USER/colide` (or with absolute paths to scripts):

```bash
cd /scr/$USER/colide
mkdir -p benchmarks/results

# Optional: skip Nsight on day 1 if queue is tight
sbatch dicc_scripts/02_benchmark_v100.sh
sbatch dicc_scripts/03_benchmark_a100.sh
# Optional profiling:
# sbatch dicc_scripts/04_nsight_profile.sh

squeue -u $USER
```

Each job:
- Tags outputs with `RUNTAG=${HW}_${SLURM_JOB_ID}` (e.g. `v100s_12345`, `a100_12346`)
- Runs single-run kernel sanity, then **`benchmark_cuda_kernels_stats.py` n=20** (paper stats)
- Runs `benchmark_pipeline.py` (hardware-tagged pipeline JSON — **same-hardware PyTorch baseline**)
- Runs batch / ORT / energy / stats scripts; copies selected JSON to `*_${RUNTAG}.json`

### Day-1 artifacts to copy home (minimum set)

From `/scr/$USER/colide/benchmarks/results/`:

| Pattern | Why |
|---------|-----|
| `cuda_kernel_stats_v100s_*.json` | V100 multi-trial CUDA stats |
| `cuda_kernel_stats_a100_*.json` | A100 multi-trial CUDA stats |
| `pipeline_benchmark_*v100s_*.json` or `pipeline_benchmark_*_${RUNTAG}.json` | Same-hardware pipeline + PyTorch |
| `pipeline_benchmark_*a100_*.json` | Same for A100 |
| `dicc_v100_*.log` / `dicc_a100_*.log` | Full stdout |
| `dicc_v100_*.err` / `dicc_a100_*.err` | Errors |
| `energy_efficiency_${RUNTAG}.json` | Optional energy |
| `statistical_confidence_${RUNTAG}.json` | Optional |

```bash
# Example pull-back to laptop (run on laptop)
JOBV=12345   # replace with real SLURM ids
JOBA=12346
scp "USER@DICC:/scr/USER/colide/benchmarks/results/*${JOBV}*" ~/colide/benchmarks/results/
scp "USER@DICC:/scr/USER/colide/benchmarks/results/*${JOBA}*" ~/colide/benchmarks/results/
scp "USER@DICC:/scr/USER/colide/benchmarks/results/dicc_v100_${JOBV}.*" ~/colide/benchmarks/results/
scp "USER@DICC:/scr/USER/colide/benchmarks/results/dicc_a100_${JOBA}.*" ~/colide/benchmarks/results/
```

Record in a notes file (or HANDOFF): date, job IDs, node names from logs.

---

## 3. Day 2+ — second measurement session (required)

**Wait until a different calendar day** (not the same sitting). Goal: test session-to-session drift on native Linux (no WSL2).

```bash
cd /scr/$USER/colide
git pull   # if laptop pushed anything
sbatch dicc_scripts/02_benchmark_v100.sh
sbatch dicc_scripts/03_benchmark_a100.sh
squeue -u $USER
```

Pull **new** `*_${NEWID}*` files the same way. Do **not** delete day-1 files.

You should end with **≥2** independent `cuda_kernel_stats_*` and **≥2** pipeline tags per GPU.

---

## 4. What “done” looks like for Session 10 ingest

On the laptop repo, `benchmarks/results/` contains at least:

```text
cuda_kernel_stats_v100s_<id1>.json
cuda_kernel_stats_v100s_<id2>.json
cuda_kernel_stats_a100_<id1>.json
cuda_kernel_stats_a100_<id2>.json
pipeline_benchmark_*_<v100 tag1>.json
pipeline_benchmark_*_<v100 tag2>.json
pipeline_benchmark_*_<a100 tag1>.json
pipeline_benchmark_*_<a100 tag2>.json
```

Then start **Session 10**: compute means/ranges, same-hardware CUDA vs PyTorch ratios, update README cross-hardware table, extend `verify_claims.py`, update threats-to-validity with DICC stability conclusion.

---

## 5. Failure checklist

| Symptom | Action |
|---------|--------|
| `module load` fails | `module avail`; edit `01_setup.sh` |
| `nvcc` missing | load CUDA toolkit module matching cluster practice |
| Binary not found / double path | Ensure cluster code ≥ path fix (`benchmark_cuda_kernels_stats.py` resolves absolute paths); `git pull` |
| Job killed OOM / time | raise `#SBATCH --mem` / `--time` |
| conda activate fails in batch | use `source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate colide` in the sbatch script |
| Pipeline JSON overwrites | scripts copy to `*_${RUNTAG}.json` — always `scp` tagged files, not only untagged names |

---

## 6. Out of scope for this runbook

- Changing published RTX 3050 ranges
- Retraining models on DICC
- Single-day “good enough” — **two days minimum** per HANDOFF measurement-stability mandate

---

## 7. Quick command card (print this)

```text
Day 0:  bash dicc_scripts/01_setup.sh
Day 1:  sbatch dicc_scripts/02_benchmark_v100.sh
        sbatch dicc_scripts/03_benchmark_a100.sh
        scp results *RUNTAG* → laptop
Day 2:  same sbatch pair + scp new tags
Then:   tell agent "Session 10 — files are in benchmarks/results/"
```
