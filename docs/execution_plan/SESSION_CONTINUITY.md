# Session Continuity / Handover Pack

**Purpose:** Live continuity notes while executing (also usable if a chat is retired later).  
**Not a retirement signal** — this chat continues until ~400k tokens or user says stop.  
**Update:** every significant milestone.  
**Last update:** 2026-07-19 (rock-mode; stay in this chat)

---

## 1. Mission

Complete **every** item in `PROF_FEEDBACK_TRACKER.md` to exceptional standard for Prof Por / WoS path.

**Policy:** skip nothing → run → JSON → **INCORPORATED** or **RUN_DOCUMENTED**.

---

## 2. Authority files (read first in new chat)

1. `HANDOFF.md` (header + lifecycle)  
2. `docs/execution_plan/PROF_FEEDBACK_TRACKER.md`  
3. `docs/execution_plan/00_INDEX.md`  
4. `docs/execution_plan/15_WORK_PACKAGES.md`  
5. `docs/feedback1.docx` (Prof requirements)  
6. This file: `SESSION_CONTINUITY.md`

---

## 3. Done this arc (do not redo)

| Item | Status | Location |
|------|--------|----------|
| Interim report to Prof | Sent by user | `docs/COLIDE_Interim_Status_Report-2.docx` |
| Prof feedback received | Done | `docs/feedback1.docx` |
| Reply email | User sent (parallel improvements + DICC) | — |
| Execution plan pack | Done | `docs/execution_plan/*` |
| Skip-nothing policy | Locked | tracker + standards |
| Protocol `botiot_v1` | **DONE** | `scripts/protocol/botiot.py` |
| Metrics / losses / thresholds / result_schema | **DONE** | `scripts/protocol/*` |
| `eval_checkpoint.py` val-only | **DONE** | champion val F1 **0.9780** |
| Freeze card | **DONE** | `BASELINE_FREEZE_CARD.md` |
| `train_protocol_ft.py` | **DONE** | never overwrites champion by default |
| `run_baseline_multirun.py` | **DONE** | 5-seed driver |
| Smoke FT seed 42 (2 epoch) | **DONE** | val_macro_f1 **0.9755**; `multirun/ft_seed42.json`; ~411s |
| Classical pilot lr/rf 100k | **DONE (pilot only)** | lr 0.463 / rf 0.686 val — **not** full-data; re-run full for G* |
| Full multirun 5×10ep | **LAUNCHED** | check `/tmp/multirun_full.log` + `run_baseline_multirun` |

**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`** — do not clobber.

**DICC:** `benchmarks/results/dicc/` still **ABSENT**.

---

## 4. In flight (check first in new chat)

```bash
ps aux | grep train_protocol_ft | grep -v grep
ls -la model/multirun/ benchmarks/results/multirun/
# If ft_seed42.json missing but process dead → re-run seed 42 full epochs
# If process alive → wait; do not start second GPU train
```

After smoke OK, run full multirun:

```bash
cd /home/titoisalive/colide
PYTHONPATH=. .venv/bin/python scripts/run_baseline_multirun.py \
  --seeds 42,43,44,45,46 --epochs 10 --patience 3
```

(Long: ~2.6M train samples/epoch on RTX 3050.)

---

## 5. Next work packages (order)

| Order | WP | Action |
|------:|----|--------|
| 1 | WP1b finish | Complete multirun; update tracker L5; summary.json |
| 2 | Classical baselines G1–G5 | `scripts/run_classical_baselines.py` on protocol (same split) |
| 3 | Method package | Sign MOD table; train with focal/logit-adj/thresholds |
| 4 | Optuna HPO B* | Val-only study |
| 5 | Ablations F* | Ladder + systems metrics where possible |
| 6 | Neural baselines G6–G12 | Same protocol |
| 7 | Pareto H* | Local multi-obj figure |
| 8 | WP0 DICC | When user has SSH |
| 9 | XAI J* / ToN K* | Per tracker |
| 10 | Manuscript | Only after tracker green |

---

## 6. Hard rules (never drop)

- Option A: per-block CUDA only; no full-pipeline vs full V3  
- Test sealed until final config  
- No invented multi-day DICC numbers  
- BACKUP before any champion overwrite  
- Update tracker every session  
- Negative runs → RUN_DOCUMENTED + JSON  

---

## 7. Paste-ready next chat prompt

```text
Continue COLIDE — FULL PROF FEEDBACK EXECUTION. Skip nothing. Exceptional quality.
Option A CUDA locked.

Read first:
1) HANDOFF.md header + Session lifecycle
2) docs/execution_plan/SESSION_CONTINUITY.md
3) docs/execution_plan/PROF_FEEDBACK_TRACKER.md
4) docs/execution_plan/15_WORK_PACKAGES.md

Check: is train_protocol_ft / multirun still running? Collect multirun JSON if ready.
Then continue next incomplete WP in order (WP1b finish → classical baselines → method → HPO…).
Policy: RUN every significant Prof item → JSON → INCORPORATED or RUN_DOCUMENTED.
No invent DICC numbers; no clobber best_model_botiot_twostage.pth without BACKUP.
Update tracker; commit/push; end per HANDOFF lifecycle.
```

---

## 8. Git tip at last continuity edit

`cd411bc`+ (classical LR fix); results under `benchmarks/results/` are **gitignored** — local only until packaging WP.

### Running jobs to check

```bash
ps -eo pid,cmd | awk '/run_baseline_multirun|train_protocol_ft|run_classical/{print}'
tail -30 /tmp/multirun_full.log
ls benchmarks/results/multirun/
```
