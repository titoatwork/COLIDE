# DICC D0 — Preflight checklist

**Session:** D0 (laptop portion) · **Date (UTC):** 2026-07-29  
**Goal:** Ready the tree and scripts so Day1 can start in OnDemand VNC without thrash  
**Does not produce:** multi-day paper numbers (no invent)

---

## 1. Laptop preflight (DONE this session)

| Check | Result |
|-------|--------|
| Git tip | `87723ad` (or later after this session’s handoff commit) on **`master`** |
| Champion md5 | `80a90f7cc210276300eaa90173a5a385` **unchanged** |
| Claims verifier | All tracked claims green (local package) |
| GPU local | Idle (not required for DICC) |
| `benchmarks/results/dicc/` | **ABSENT** (expected until campaign) |
| Ops method | `docs/DICC_OPS_METHOD.md` (OnDemand VNC + screen + batch) |
| `run_campaign.sh --dry-run` | **OK** on laptop (no SLURM partitions here — expected) |
| `dicc_scripts/validate/local_validate.sh` | **26 passed, 2 failed** (mock spool missing `profiles/*.env` copy in fake tree — compare/harness core OK; re-check on cluster) |
| Profiles present | `dicc_scripts/profiles/v100.env`, `a100.env` |
| Compare script | `scripts/compare_dicc_sessions.py` present |
| Tarball on laptop | `/home/titoisalive/colide-master-for-dicc.tar.gz` **refreshed 2026-07-29** (~323 MB; excludes heavy model/*/logs; includes champion + `run_campaign.sh`) — or `git clone` current `master` on DICC |
| Prof short summary | Sent (plain bullets); long mail was apology+summarized |
| Plain numbers card | `docs/PROF_PLAIN_NUMBERS_CARD.md` (no repo jargon) |

---

## 2. User-only steps (start of live DICC — next)

Do these **inside DICC OnDemand → VNC Desktop**:

```text
[ ] Open DICC OnDemand in browser → start VNC Desktop
[ ] Terminal → screen -S colide
[ ] Sync code: git clone https://github.com/titoatwork/COLIDE.git
      OR refresh tarball from laptop after: tar of current master
[ ] cd ~/colide (or path) && git checkout master && git pull   # if git
[ ] md5sum model/best_model_botiot_twostage.pth
      # must be 80a90f7cc210276300eaa90173a5a385
[ ] python3 -m venv .venv-cluster && source .venv-cluster/bin/activate
[ ] pip install -U pip
[ ] pip install 'numpy>=2.0.0' 'scipy>=1.13.0' 'pyyaml>=6.0' 'scikit-learn>=1.5.0'
[ ] pip install --upgrade 'torch>=2.5.0,<2.7' --index-url https://download.pytorch.org/whl/cu121
[ ] export COLIDE_V100_PARTITION=gpu-v100s
[ ] export COLIDE_A100_PARTITION=gpu-a100
[ ] export COLIDE_SBATCH_GRES=gpu:1
[ ] bash dicc_scripts/run_campaign.sh --dry-run   # optional sanity on cluster
[ ] bash dicc_scripts/run_campaign.sh             # Day 1 → D1 session
```

**D0 complete (full):** all boxes above checked **and** Day1 env proven on cluster.  
**D0 complete (laptop):** section 1 only — hand off to user for VNC.

---

## 3. Partitions (from prior jobs; override if `sinfo` differs)

```bash
export COLIDE_V100_PARTITION=gpu-v100s
export COLIDE_A100_PARTITION=gpu-a100
export COLIDE_SBATCH_GRES=gpu:1
```

---

## 4. Exit / next session

| After D0 full | Next |
|---------------|------|
| Env + dry-run OK on DICC | **D1** Day1 real submit |
| SUCCESS dirs appear | **D2** Day2 |
| Both days home | **D3** compare |
| JSON only | **D5** manuscript insert |

*No multi-day numbers in paper until SUCCESS on laptop.*
