# Session Continuity / Handoff Pack

**Session closed for continuity:** 2026-07-21  
**Mode this session:** documentation + handoff only — **no new experiments**  
**Git tip at handoff:** see latest commit after handoff push (`git log -1 --oneline`)  
**Machine root:** `/home/titoisalive/colide`

---

## 1. Mission (unchanged)

Complete **every** row in `PROF_FEEDBACK_TRACKER.md` for Prof Por / WoS path.

**Policy:** skip nothing → run → JSON → **INCORPORATED** or **RUN_DOCUMENTED**.  
**Option A:** per-block CUDA only; no invent multi-day DICC numbers.  
**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`** — never clobber without BACKUP + explicit OK.

---

## 2. Read first in the next chat (order)

1. `HANDOFF.md` header  
2. **This file** (`SESSION_CONTINUITY.md`)  
3. `docs/execution_plan/RESULTS_DISK_MANIFEST.md` ← committed numbers + md5s  
4. `docs/execution_plan/PROF_FEEDBACK_TRACKER.md`  
5. `docs/execution_plan/PROGRESS_LOG.md`  
6. `docs/execution_plan/METHOD_PACKAGE_DECISION.md` (CAD-CBA-v1)  
7. `docs/execution_plan/15_WORK_PACKAGES.md`  
8. `docs/feedback1.docx` (when interpreting Prof requirements)

---

## 3. Completed this arc (do not redo)

### 3.1 Planning / process
| Item | Location |
|------|----------|
| Execution plan pack (phases 0–9) | `docs/execution_plan/00`–`16` |
| Skip-nothing policy | tracker + `14_EXCEPTIONAL_STANDARDS.md` |
| Method package decision CAD-CBA-v1 | `METHOD_PACKAGE_DECISION.md` |
| Disk results manifest (git) | `RESULTS_DISK_MANIFEST.md` |
| Interim report + Prof reply (user) | Word interim; email sent |

### 3.2 Protocol foundation (DONE)
| Item | Path |
|------|------|
| BoT load `stage_a_kd` / `stage_b_ft` | `scripts/protocol/botiot.py` |
| Metrics | `scripts/protocol/metrics.py` |
| Losses (focal, CE, CB weights, logit_adj) | `scripts/protocol/losses.py` |
| Val-only thresholds | `scripts/protocol/thresholds.py` |
| Result envelope | `scripts/protocol/result_schema.py` |
| Sealed eval | `scripts/eval_checkpoint.py` |
| Freeze card | `docs/execution_plan/BASELINE_FREEZE_CARD.md` |
| Champion sealed eval val macro-F1 | **0.9780** (`stage_b_ft`, seed 42) |

### 3.3 WP1b multirun FT (DONE)
Init: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth`  
Script: `scripts/train_protocol_ft.py` + `scripts/run_baseline_multirun.py`  
Results: `benchmarks/results/multirun/` (**gitignored** — on this machine)

| Seed | Best val macro-F1 | Min per-class F1 |
|------|-------------------|------------------|
| 42 | 0.9780 | 0.9315 |
| 43 | 0.9578 | 0.9091 |
| 44 | **0.9840** | **0.9589** |
| 45 | 0.9746 | 0.9143 |
| 46 | 0.9624 | 0.9091 |

**Mean 0.9714 ± 0.0109** (n=5)  
min 0.9578 · max 0.9840  
Summary: `benchmarks/results/multirun/summary.json`  
Ckpts: `model/multirun/ft_seed{42..46}.pth`  
**Test sealed. Production champion not overwritten.**

### 3.4 Classical baselines protocol-fair (PARTIAL suite; trees done)
Script: `scripts/run_classical_baselines.py`  
**Use:** `benchmarks/results/baselines_classical/summary_handoff.json`  
(**Trap:** `summary.json` may only reflect last single-model run.)

| Model | Val macro-F1 | Theft F1 | Notes |
|-------|--------------|----------|-------|
| LR | 0.5231 | 0.0 | full train |
| **RF** | **0.9778** | **0.923** | full train; ≠ published 0.9864 path |
| **XGB** | **0.9762** | **0.923** | full train |
| LGBM | 0.5512 | 0.0 | weak — RUN_DOCUMENTED / fix later |
| SVM pilot 150k | FAILED | — | &lt;3 samples/class after subsample |

**Critical:** Published RF **0.9864** = `rf_baseline_processed` pipeline. Protocol `stage_b_ft` RF **0.9778** is the fair compare for multirun neural.

### 3.5 Imbalance loss compare (DONE)
Script: `scripts/run_imbalance_loss_compare.py`  
Results: `benchmarks/results/imbalance_loss/`  
Summary: `benchmarks/results/imbalance_loss/summary.json`

| Loss | Best val macro-F1 | Min cls F1 | Decision |
|------|-------------------|------------|----------|
| ce | 0.9755 | 0.9189 | RUN_DOCUMENTED |
| **focal** | **0.9780** | **0.9315** | **INCORPORATE (keep default)** |
| focal_cb | 0.9121 | 0.8000 | RUN_DOCUMENTED (worse macro) |
| logit_adj | 0.9225 | 0.8000 | RUN_DOCUMENTED (worse macro) |

**Winner: focal** (same as multirun baseline recipe).  
Ckpts: `model/imbalance_loss/ft_{ce,focal,focal_cb,logit_adj}_seed42.pth`

### 3.6 Not done (do next chat)
- Val threshold search JSON on best focal (util exists; good first job)  
- Optuna HPO  
- Teacher/KD under protocol  
- Ablations ladder + neural baselines + Pareto  
- SVM full-data fair run  
- LGBM fix / re-run to RF-class quality  
- DICC multi-day  
- XAI / ToN / manuscript  

---

## 4. Background jobs at handoff

**Expect idle** for train/multirun/imbalance (all completed 2026-07-19).  
Verify:

```bash
cd /home/titoisalive/colide
ps -eo pid,cmd | awk '/train_protocol_ft|run_baseline_multirun|run_imbalance|run_classical/{print}'
# should be empty (or only the awk line itself)
test -f benchmarks/results/multirun/summary.json && echo multirun_OK
test -f benchmarks/results/imbalance_loss/summary.json && echo imbalance_OK
md5sum model/best_model_botiot_twostage.pth
# expect: 80a90f7cc210276300eaa90173a5a385
```

Untracked leftover (do not treat as active job): `scripts/watch_and_queue_next.sh` (one-shot queue from multirun → imbalance; already finished).

---

## 5. Next chat work order (strict)

1. **Verify** disk vs `RESULTS_DISK_MANIFEST.md` (summaries + champion md5).  
2. **Optional quick:** thresholds on `model/imbalance_loss/ft_focal_seed42.pth` → JSON under `benchmarks/results/imbalance_loss/` (val only).  
3. Confirm tracker rows match §3 (should already after 2026-07-21 handoff).  
4. **Next science (pick one, full JSON + tracker flip):**  
   - Teacher compare (RF/XGB/ensemble soft labels) under protocol, **or**  
   - Optuna HPO val-only (B*), **or**  
   - Ablation ladder F*  
5. **DICC** when user has access (Phase 0) — parallel track.  
6. Never start manuscript until tracker largely green.  
7. End session: update tracker + progress + HANDOFF + commit/push.

---

## 6. Paste-ready next-session prompt

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

## 7. Key file index

| Role | Path |
|------|------|
| Handoff narrative | `docs/execution_plan/SESSION_CONTINUITY.md` |
| Disk numbers + md5s | `docs/execution_plan/RESULTS_DISK_MANIFEST.md` |
| Tracker | `docs/execution_plan/PROF_FEEDBACK_TRACKER.md` |
| Progress | `docs/execution_plan/PROGRESS_LOG.md` |
| Method decision | `docs/execution_plan/METHOD_PACKAGE_DECISION.md` |
| Work packages | `docs/execution_plan/15_WORK_PACKAGES.md` |
| Multirun summary | `benchmarks/results/multirun/summary.json` |
| Imbalance summary | `benchmarks/results/imbalance_loss/summary.json` |
| Classical handoff table | `benchmarks/results/baselines_classical/summary_handoff.json` |
| Protocol | `scripts/protocol/*` |
| Train FT | `scripts/train_protocol_ft.py` |
| Multirun | `scripts/run_baseline_multirun.py` |
| Classical | `scripts/run_classical_baselines.py` |
| Loss compare | `scripts/run_imbalance_loss_compare.py` |

**Note:** `benchmarks/results/` is largely **gitignored** — results live on this machine; next agent must use laptop paths or re-run. Manifest commits the **headlines + md5s**.

---

## 8. Desirability snapshot

| Result | Assessment |
|--------|------------|
| Multirun mean ~0.971 ± 0.011 | Desirable protocol baseline |
| Best seed 0.984 | Strong; do not report alone |
| Protocol RF/XGB ~0.978/0.976 | Neural in same band — good |
| Focal best among losses | Keep as default |
| CB-focal / logit_adj worse macro | Documented negative results |
| LGBM weak | Not ready as strong baseline |

---

*End handoff. Next chat: verify disk → continue tracker from §5.*
