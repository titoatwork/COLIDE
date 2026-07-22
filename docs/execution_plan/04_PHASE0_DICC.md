# 04 — Phase 0: UM DICC Multi-Day Campaign (HARD GATE)

**Status:** **BLOCKED (ops)** — artifacts ABSENT (`benchmarks/results/dicc/` missing); user must open a **dedicated DICC session**  
**Authority:** feedback §8; FINAL_PLAN P0–P1; dicc_scripts  
**Blocking:** All cluster claims, portability RQ (K7), final multi-objective cluster cells  
**Local path note:** WP6b multi-session ranges on RTX 3050 are DONE and must not be re-labeled as DICC multi-day.  

---

## 1. Goal

Produce **paper-grade** multi-day, same-GPU evidence on **UM DICC** for:

1. Custom CUDA **Block 3** vs **matching** PyTorch Block 3  
2. **Full-model** PyTorch absolute latency (separate from CUDA ratio)  
3. **V100S** and **A100**  
4. Stability across **≥2 days** with statistical comparison  

**Not goals:** Invent numbers; Rostam as official; full-pipeline CUDA vs full V3 ratio.

---

## 2. Deliverables (acceptance)

| Deliverable | Acceptance criteria |
|-------------|---------------------|
| Day1 SUCCESS dirs V100S (and A100 if possible) | `SUCCESS` file, manifest, cuda stats, pytorch stats |
| Day2 SUCCESS same GPU class(es) | Same schema; different job/date |
| Compare report | `compare_dicc_sessions.py` accept **or** documented reject with reason |
| Metrics present | mean, median, std, CV, CI (as harness provides); Welch/p, Cohen’s d for compare |
| Warm-up protocol | Documented in manifest / README of run |
| Tree on laptop | `benchmarks/results/dicc/**` complete |
| Champion md5 on cluster | Matches `80a90f7…` or newer sealed champion (if retrained later, re-run Phase 0) |
| No invented cells | Empty until JSON present |

**Optional later in Phase 0/6:** batch-size sensitivity on cluster; extend campaign if missing.

---

## 3. Operator procedure (exceptional)

### 3.1 Sync code (prefer tarball; fetch may freeze)

```bash
# Laptop
cd /home/titoisalive
tar czf colide-master-for-dicc.tar.gz \
  --exclude='colide/.venv' --exclude='colide/.venv-cluster' \
  --exclude='colide/**/__pycache__' -C /home/titoisalive colide
# scp to login01.dicc.um.edu.my home
```

### 3.2 Environment on DICC

```bash
cd ~/colide   # or unpack path
python3 -m venv .venv-cluster && source .venv-cluster/bin/activate
pip install -U pip
pip install 'numpy>=2.0.0' 'scipy>=1.13.0' 'pyyaml>=6.0' 'scikit-learn>=1.5.0'
pip install --upgrade 'torch>=2.5.0,<2.7' --index-url https://download.pytorch.org/whl/cu121
```

### 3.3 Run

```bash
export COLIDE_V100_PARTITION=gpu-v100s
export COLIDE_A100_PARTITION=gpu-a100
export COLIDE_SBATCH_GRES=gpu:1
bash dicc_scripts/run_campaign.sh           # Day 1
bash dicc_scripts/run_campaign.sh --day 2   # Day 2
# Prefer tmux; prefer partitions over fixed nodes
```

### 3.4 Compare + bring home

```bash
# On laptop after scp of benchmarks/results/dicc/
PYTHONPATH=. python3 scripts/compare_dicc_sessions.py ...  # per README args
```

### 3.5 Extract table (no freehand)

Fill only from JSON:

| GPU | Day | B3 CUDA FP16 mean | B3 PT mean | Full V3 PT mean | CV | Notes |
|-----|-----|-------------------|------------|-----------------|----|-------|
| V100S | 1 | | | | | |
| V100S | 2 | | | | | |
| A100 | 1 | | | | | |
| A100 | 2 | | | | | |

---

## 4. Decision fork after results

| Outcome | Next exceptional action |
|---------|-------------------------|
| CUDA B3 ≤ PT on both GPUs (win/tie band) | Systems latency primary; proceed Phase 1–6 with confidence |
| CUDA B3 > PT (slower) on servers | **M12** kernel optim or protocol audit **before** claiming portable speedups; strengthen multi-objective memory story |
| Compare reject | Report both days honestly; no “stable multi-day” until fixed |
| Only one GPU class | Document; publish complete class; schedule other |

---

## 5. Scripts / files involved

- `dicc_scripts/run_campaign.sh`, `submit_session.sh`, `lib/*`  
- `scripts/benchmark_cuda_kernels_stats.py`  
- `scripts/benchmark_pytorch_gpu_stats.py`  
- `scripts/compare_dicc_sessions.py`  
- `dicc_scripts/README.md`  

---

## 6. Status checklist

- [ ] Tarball/sync current master to DICC  
- [ ] Day1 V100S SUCCESS  
- [ ] Day1 A100 SUCCESS  
- [ ] Day2 V100S SUCCESS  
- [ ] Day2 A100 SUCCESS  
- [ ] Compare run + outcome logged  
- [ ] Tree on laptop under `benchmarks/results/dicc/`  
- [ ] Extraction table filled from JSON only  
- [ ] Fork decision recorded in `MOD_DECISION_TABLE.md` footer  

---

## 7. Exit criteria

Phase 0 complete when checklist done and no cluster number appears in paper/docs without a path to JSON.

---

*Do not start large HPO or architecture rewrites until Phase 0 exit — Prof order.*
