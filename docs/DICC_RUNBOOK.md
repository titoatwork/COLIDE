> **SUPERSEDED for operators.**  
> **Use:** `docs/DICC_OPS_METHOD.md` (connection + run card) and  
> `bash dicc_scripts/run_campaign.sh` (`dicc_scripts/README.md`).  
> This file is **historical** only (old Session 9 fixed-node / interactive-SSH style).

# COLIDE — DICC Cluster Runbook (historical)

**Do not follow this file as the live ops plan.**

## Current method (2026-07-22)

| Step | Action |
|------|--------|
| 1 | Browser → **DICC OnDemand → VNC Desktop** |
| 2 | Terminal inside VNC → **`screen -S colide`** |
| 3 | Sync tree; verify champion md5 `80a90f7…` |
| 4 | `.venv-cluster` + deps + compile as needed |
| 5 | **Batch:** `bash dicc_scripts/run_campaign.sh` (Day1), then `--day 2` |
| 6 | rsync/scp `benchmarks/results/dicc/` to laptop |
| 7 | Laptop: compare + claims + manuscript insert |

**Avoid as primary:** long interactive `srun`/`salloc` over VPN; campus-runner / third-party operator defaults; inventing multi-day numbers.

---

## Why this file is historical

Earlier drafts assumed bare SSH + fixed nodelists (`gpu05`/`gpu06`) and interactive babysitting. Guidance and practice now prefer **OnDemand VNC** (session survives network drops) + **`screen`** + **batch** campaign scripts with **partitions** (`gpu-v100s`, `gpu-a100`).

*End historical note.*
