# COLIDE — Session Handoff

**MODE:** 🚀 **EXECUTION — next chat continues Prof tracker.** This chat closed for continuity.  
**Closed:** 2026-07-21 · **docs/handoff only** (no new experiments).  
**Authority:** `docs/execution_plan/SESSION_CONTINUITY.md` + `RESULTS_DISK_MANIFEST.md` + `PROF_FEEDBACK_TRACKER.md` + Option A.  
**Policy:** skip nothing → JSON → INCORPORATED or RUN_DOCUMENTED. No invent DICC numbers.  
**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`** — no clobber without BACKUP.

---

## This chat / arc delivered

| Deliverable | Status |
|-------------|--------|
| Protocol `botiot_v1` + metrics/losses/thresholds | DONE |
| Sealed `eval_checkpoint` (champion val **0.9780**) | DONE |
| WP1b multirun 5 seeds | **DONE** mean **0.9714 ± 0.0109** |
| Classical protocol-fair LR/RF/XGB/LGBM | DOCUMENTED (RF 0.978 / XGB 0.976) |
| Imbalance loss compare 4 losses | **DONE** — **focal wins 0.9780** |
| CAD-CBA-v1 method decision | Signed (status refreshed) |
| Full execution plan + tracker | DONE + aligned 2026-07-21 |
| `RESULTS_DISK_MANIFEST.md` | DONE (committed numbers + md5s) |

**DICC:** still **ABSENT** (`benchmarks/results/dicc/`).  
**Jobs:** multirun + imbalance compare **finished** (2026-07-19); expect no train processes.

---

## Open with (next chat)

1. `HANDOFF.md` (this header)  
2. **`docs/execution_plan/SESSION_CONTINUITY.md`**  
3. **`docs/execution_plan/RESULTS_DISK_MANIFEST.md`**  
4. **`docs/execution_plan/PROGRESS_LOG.md`**  
5. **`docs/execution_plan/PROF_FEEDBACK_TRACKER.md`**

---

## Paste-ready next-session prompt

```text
Continue COLIDE — FULL PROF FEEDBACK EXECUTION. Skip nothing. Exceptional quality.
Option A CUDA locked. NO new jobs until you read continuity + verify disk.

Read first (in order):
1) HANDOFF.md header + Session lifecycle
2) docs/execution_plan/SESSION_CONTINUITY.md   ← authoritative handoff
3) docs/execution_plan/RESULTS_DISK_MANIFEST.md
4) docs/execution_plan/PROGRESS_LOG.md
5) docs/execution_plan/PROF_FEEDBACK_TRACKER.md
6) docs/execution_plan/METHOD_PACKAGE_DECISION.md
7) docs/execution_plan/15_WORK_PACKAGES.md

Verify on disk (do not invent if missing):
- benchmarks/results/multirun/summary.json  (mean ~0.9714±0.0109, n=5)
- benchmarks/results/imbalance_loss/summary.json  (focal wins 0.9780)
- benchmarks/results/baselines_classical/*_seed42.json + summary_handoff.json
- model/multirun/ft_seed{42..46}.pth ; model/imbalance_loss/ft_*_seed42.pth
- Champion md5 still 80a90f7cc210276300eaa90173a5a385

Last session (2026-07-21): DOCUMENTATION / HANDOFF ONLY — no new experiments.
Completed prior: protocol; WP1b multirun DONE; classical protocol-fair RF/XGB/LR;
imbalance loss compare DONE (focal INCORPORATE; focal_cb/logit_adj RUN_DOCUMENTED).

Next (pick one, document JSON + update tracker):
A) Val thresholds on model/imbalance_loss/ft_focal_seed42.pth
B) Teacher/KD experiments under protocol
C) Optuna HPO val-only
D) Ablations / neural baselines
E) Guided DICC if user has SSH

Rules: no invent multi-day numbers; no clobber champion without BACKUP;
update tracker every WP; commit/push; end per HANDOFF lifecycle.
```

---

## Session lifecycle (standing)

```bash
cd /path/to/colide
git status -sb
git log -1 --oneline
git rev-parse HEAD origin/master
# after claim edits: PYTHONPATH=. python3 scripts/verify_claims.py
```

| Gate | Requirement |
|------|-------------|
| HANDOFF header | Updated |
| Deliverables | On disk / documented |
| Commit + push | Meaningful message |
| Clean tree | Or list deferred |
| Next prompt | In HANDOFF **and** closing message |
| Champion | Never clobber without backup + OK |

---

## Standing rules

- Option A: no full-pipeline Custom CUDA vs full V3  
- Test sealed until final config  
- Official cluster: UM DICC only  
- Results under `benchmarks/results/` often **gitignored** — use `RESULTS_DISK_MANIFEST.md` + local paths  
- Agents: no invented DICC numbers  
- Classical: prefer `summary_handoff.json` over possibly stale `summary.json`  

---

## Older checkpoints (still valid)

Design plan Option A approved; FINAL_PLAN P0–P5; audit pack `docs/audit/`; interim Word report sent to Prof; feedback in `docs/feedback1.docx`.

**Next science priority after verify disk:** thresholds on focal **or** HPO/teachers/ablations per continuity §5.
