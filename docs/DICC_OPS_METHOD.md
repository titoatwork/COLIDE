# DICC operations method (authoritative)

**Status:** LOCKED 2026-07-22  
**Authority:** Guidance relayed for DICC access (VNC Desktop via OnDemand + `screen`); scientific campaign = Prof feedback §8 + `docs/execution_plan/04_PHASE0_DICC.md`  
**Official cluster:** **UM DICC only** (Rostam is tooling trial, not paper-final)

---

## 1. What this document is

The **how to run** multi-day DICC work without losing sessions to network drops.

It is **not**:
- A campus-stable Wi‑Fi requirement  
- A “local UM runner / Cheran runs the campaign” default  
- An excuse to invent multi-day numbers  

**Supersedes** earlier ops language in older drafts that recommended Cheran-on-his-account, campus runner, or “better link + interactive `srun`/`salloc`” as the primary plan.

---

## 2. Guidance (connection resilience)

| Avoid as primary path | Prefer |
|----------------------|--------|
| Long **interactive** `srun` / `salloc` over VPN/SSH | **DICC OnDemand → VNC Desktop** for interactive setup/downloads/compile |
| Depending on a continuous laptop↔cluster SSH for hours | Session lives on the cluster desktop; **reconnect after drops** |
| Unprotected long shell work | Inside VNC terminal: **`screen`** (or equivalent) for long-running tasks |
| Babysitting multi-hour GPU work interactively | Prefer **batch** (`sbatch` / `dicc_scripts/run_campaign.sh`) so jobs survive disconnect |

**Rationale (from guidance text):** interactive `srun`/`salloc` requires a stable connection; temporary network/VPN interruption disconnects the session and can **lose the allocation**. VNC Desktop via OnDemand keeps the remote desktop running; reconnect later. Within VNC, use `screen` so long tasks continue while you use the terminal for other work.

---

## 3. Recommended workflow

```text
1. Browser → DICC OnDemand → start VNC Desktop
2. Open terminal inside VNC
3. Start screen:   screen -S colide
4. Sync tree (git or tarball); verify champion md5
5. Create .venv-cluster; install deps; compile kernels (interactive OK here)
6. Submit campaign via batch:
     export COLIDE_V100_PARTITION=gpu-v100s
     export COLIDE_A100_PARTITION=gpu-a100
     export COLIDE_SBATCH_GRES=gpu:1
     bash dicc_scripts/run_campaign.sh          # Day 1
     bash dicc_scripts/run_campaign.sh --day 2  # Day 2 after Day1 SUCCESS
7. Disconnect freely; jobs continue on SLURM
8. Later: VNC/login again → check SUCCESS → rsync/scp benchmarks/results/dicc/ to laptop
9. Laptop only: compare_dicc_sessions.py + fill tables from JSON + claims rebuild
```

---

## 4. Operator run card (inside OnDemand VNC + screen)

```bash
# After OnDemand VNC Desktop is up, in a terminal:
screen -S colide   # or: screen -r colide

# Tree: git clone OR tar xzf colide-master-for-dicc.tar.gz
cd ~/colide   # or actual path

python3 -m venv .venv-cluster
source .venv-cluster/bin/activate
pip install -U pip
pip install 'numpy>=2.0.0' 'scipy>=1.13.0' 'pyyaml>=6.0' 'scikit-learn>=1.5.0'
pip install --upgrade 'torch>=2.5.0,<2.7' --index-url https://download.pytorch.org/whl/cu121

# Champion must match production systems path (no clobber)
md5sum model/best_model_botiot_twostage.pth
# expect: 80a90f7cc210276300eaa90173a5a385

export COLIDE_V100_PARTITION=gpu-v100s
export COLIDE_A100_PARTITION=gpu-a100
export COLIDE_SBATCH_GRES=gpu:1

bash dicc_scripts/run_campaign.sh              # Day 1
# after SUCCESS dirs exist:
bash dicc_scripts/run_campaign.sh --day 2      # Day 2 (_d2 label OK)

# Deliverable to laptop: entire benchmarks/results/dicc/
```

**Notes:**
- Prefer **`.venv-cluster`** (conda activate under script `set -u` breaks on DICC).  
- Prefer **partitions** over fixed nodelists (`gpu05` / `gpu07` may change).  
- Prefer **batch** for Day1/Day2 benchmarks; use VNC+screen for setup and monitoring.  
- Detach screen: `Ctrl-a d` · reattach: `screen -r colide`.

---

## 5. Scientific rules (unchanged)

| Rule | Detail |
|------|--------|
| Option A | Per-block Custom CUDA vs matching PyTorch only |
| Primary head-to-head | Block 3 (BiLSTM), same GPU, same session |
| Forbidden | Full-pipeline Custom CUDA vs full PyTorch V3 speedup |
| Multi-day | ≥2 days SUCCESS; compare before “stable multi-day” claim |
| June 2026 551/592 µs | Legacy **single-shot** only — not multi-day official |
| Invented numbers | Never |

---

## 6. Roles

| Who | Does |
|-----|------|
| **User (account holder)** | OnDemand VNC, setup, submit campaign, bring SUCCESS tree home |
| **Agent** | Coaching, laptop compare/claims/manuscript insert; **no DICC login**; **no invented numbers** |
| **Writing support (if any)** | Manuscript only — **not** the default cluster operator path |

---

## 7. Exit when

- `benchmarks/results/dicc/**/SUCCESS` on laptop  
- Compare outcome recorded  
- No cluster number in paper without a path to JSON  

---

## 8. Related files

| Path | Role |
|------|------|
| **This file** | Authoritative ops method |
| `docs/execution_plan/04_PHASE0_DICC.md` | Phase 0 science acceptance |
| `docs/FINAL_PLAN.md` | Overall phased plan (ops section points here) |
| `dicc_scripts/README.md` | Script entry (`run_campaign.sh`) |
| `docs/DICC_RUNBOOK.md` | Historical; superseded for operators |

*End.*
