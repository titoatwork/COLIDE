# Session Continuity / Handover Pack

**Purpose:** Live progress + zero-loss resume if this chat is retired (~400k tokens).  
**Last update:** 2026-07-19 (still executing; multirun + classical in flight)

---

## 1. Mission

Complete **every** item in `PROF_FEEDBACK_TRACKER.md` (Prof Por / WoS).  
**Policy:** skip nothing → run → JSON → **INCORPORATED** or **RUN_DOCUMENTED**.

---

## 2. Read first in any new chat

1. `HANDOFF.md` header  
2. **This file**  
3. `docs/execution_plan/PROF_FEEDBACK_TRACKER.md`  
4. `docs/execution_plan/15_WORK_PACKAGES.md`  
5. `docs/feedback1.docx`

---

## 3. Done (do not redo)

| Item | Status | Notes |
|------|--------|-------|
| Interim report + Prof feedback + reply | Done | Parallel improve + DICC |
| Execution plan pack + skip-nothing | Done | `docs/execution_plan/*` |
| Protocol botiot_v1 | **DONE** | `scripts/protocol/*` |
| eval_checkpoint sealed test | **DONE** | champion val F1 **0.9780** |
| train_protocol_ft + multirun driver | **DONE** | never clobber champion by default |
| Multi-loss FT + imbalance compare driver | **DONE** | `--loss ce/focal/focal_cb/logit_adj` |
| Smoke FT seed42 (2 ep) | **DONE** | best val **0.9755** |
| Classical pilot 100k | **DONE** | lr 0.46 / rf 0.69 — pilot only |
| Full-data classical LR | **DONE** | val_macro_f1 **0.5231**, theft_f1 0, normal_f1 0.26 |

**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`** — do not clobber.  
**DICC:** still **ABSENT**.

---

## 4. In flight (check first!)

```bash
cd /home/titoisalive/colide
ps -eo pid,etime,pcpu,cmd | awk '/run_baseline_multirun|train_protocol_ft\.py|run_classical_baselines/{print}'
tail -40 /tmp/multirun_full.log
tail -40 /tmp/classical_full.log
ls -la benchmarks/results/multirun/ benchmarks/results/baselines_classical/
# Queued after multirun summary.json appears:
#   /tmp/colide_queue_after_multirun.sh → run_imbalance_loss_compare.py
cat /tmp/post_multirun_queue.log 2>/dev/null
```

| Job | Status at last note |
|-----|---------------------|
| Multirun 5×10ep | **RUNNING** seed 42 (epoch 1 done @ 0.9755); seeds 43–46 pending |
| Classical full lr,rf,xgb,lgbm | **RUNNING** — LR done 0.523; RF+ next |
| Post-multirun queue | Watcher waits for `multirun/summary.json` then imbalance compare |

**Do not start a second GPU training** while multirun holds the GPU.

---

## 5. Next order of work

1. Collect multirun `summary.json` → update tracker L5  
2. Collect full classical summary → update G1–G5 (SVM still TODO)  
3. Imbalance loss compare (auto-queued or manual) → D*  
4. Thresholds on best val model  
5. Method package / HPO  
6. Ablations + neural baselines + Pareto  
7. DICC when user has access  
8. XAI / ToN / manuscript after evidence  

---

## 6. Hard rules

- Option A CUDA; no invent DICC numbers  
- Test sealed until final config  
- BACKUP before champion overwrite  
- Update tracker every session  
- Results often **gitignored** under `benchmarks/results/` — local paths matter  

---

## 7. Paste-ready next chat (when retiring)

```text
Continue COLIDE — FULL PROF FEEDBACK EXECUTION. Skip nothing. Exceptional quality.
Option A CUDA locked.

Read:
1) HANDOFF.md header + Session lifecycle
2) docs/execution_plan/SESSION_CONTINUITY.md
3) docs/execution_plan/PROF_FEEDBACK_TRACKER.md
4) docs/execution_plan/15_WORK_PACKAGES.md

First: check multirun/classical processes and JSON under benchmarks/results/multirun/
and baselines_classical/. If multirun summary.json ready, update tracker L5 and proceed
to imbalance loss compare (or check /tmp/imbalance_loss_compare.log).
Then next incomplete WP. No invent DICC; no clobber champion without BACKUP.
Update tracker; commit/push; end per HANDOFF lifecycle.
```

---

## 8. Git tip (update when committing)

See `git log -1 --oneline` on `master` (continuity commits land as work proceeds).
