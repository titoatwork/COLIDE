# COLIDE — Session Handoff

**MODE:** 🚀 **EXECUTION — next chat continues Prof tracker.** This chat closed for continuity.  
**Closed:** 2026-07-22 · **WP7 XAI + WP8 ToN + F9 energy + WP6a fidelity** (science + systems).  
**Authority:** `docs/execution_plan/SESSION_CONTINUITY.md` + `RESULTS_DISK_MANIFEST.md` + `PROF_FEEDBACK_TRACKER.md` + Option A.  
**Policy:** skip nothing → **complete every playlist/tracker row** → JSON → INCORPORATED or RUN_DOCUMENTED (BLOCKED only for ops). **Perfection over LOR hurry.** No invent DICC numbers. Context hygiene: flip statuses when evidence already exists.  
**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`** — no clobber without BACKUP.  
**Session end:** always paste full next-session prompt in closing message (not only in this file).

---

## This chat / arc delivered

| Deliverable | Status |
|-------------|--------|
| Protocol + sealed eval + WP1b multirun 0.9714±0.0109 | DONE (prior) |
| Classical / imbalance / thresholds / WP4b ensemble KD 0.9401 | DONE (prior) |
| WP3 Optuna HPO 0.9791 INCORPORATE train HPs | DONE (prior) |
| Package / HPO confirm / ablation / neural baselines | DONE (prior) |
| G2/G5 classical + D6 stratified + WP5c analysis Pareto | DONE (prior) |
| WP5c systems rebench `pareto_h8/` | DONE composite G6 **0.9056** |
| Bounded C* + B2–B4 + E6 | DONE all RUN_DOCUMENTED |
| **WP7 XAI suite** | **DONE** J10 **DROP_FULL_EXPLAINABLE_CLAIM_KEEP_STRUCTURED** |
| **F9 energy table** | **DONE** RTX ~**0.786 mJ/flow** batch128 |
| **WP8 ToN final method** | **DONE** val **0.8080** test **0.8110** RF test **0.9393** |
| **WP6a re-export + fidelity** | **DONE** bit-identical; CUDA self-check PASS |

**CAD-CBA-v1 locked:** V3 + focal + ensemble KD + hpo_best + shuffle + argmax.  
**XAI paper path:** dispatch 16.60 µs + structured evidence only — **no full LLM-explainable title.**  
**DICC:** still **ABSENT** — dedicated session when user opens it.  
**Jobs:** expect idle GPU.

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
Perfection over LOR hurry. Option A CUDA locked. NO new jobs until you read continuity + verify disk.

FULL PLAYLIST LAW (user-locked):
- Complete EVERY tracker / WP playlist item → DONE | INCORPORATED | RUN_DOCUMENTED | BLOCKED(ops only).
- No silent skips of “optional” work (sealed test, WP6b ranges, claims packaging, manuscript gates, etc.).
- If evidence already exists on disk/docs → flip tracker status + notes. Never invent numbers.
- End session: update tracker/progress/manifest/HANDOFF + commit/push + paste full next-session prompt.

Read first (in order):
1) HANDOFF.md header + Session lifecycle
2) docs/execution_plan/SESSION_CONTINUITY.md
3) docs/execution_plan/RESULTS_DISK_MANIFEST.md
4) docs/execution_plan/PROGRESS_LOG.md
5) docs/execution_plan/PROF_FEEDBACK_TRACKER.md
6) docs/execution_plan/METHOD_PACKAGE_DECISION.md
7) docs/execution_plan/15_WORK_PACKAGES.md
8) config/hpo_best.yaml

Verify on disk:
- benchmarks/results/xai/summary.json  (J10 DROP_FULL keep structured; rank_corr ~0.9636; faith ~0.5109)
- benchmarks/results/toniot_final/summary.json  (val ~0.8080; test ~0.8110; RF test ~0.9393)
- benchmarks/results/energy_table/summary.json  (RTX ~0.786 mJ/flow)
- benchmarks/results/numerical_fidelity.json  (bit-identical + CUDA PASS)
- benchmarks/results/cstar_bounded/summary.json  (CTRL ~0.9787; C* RUN_DOCUMENTED)
- benchmarks/results/teachers_kd_neural/summary.json  (E6 student ~0.8513)
- benchmarks/results/pareto_h8/summary.json  (composite G6 ~0.9056)
- benchmarks/results/hpo/summary.json  (winner ~0.9791 INCORPORATE)
- config/hpo_best.yaml
- benchmarks/results/multirun/summary.json  (WP1b ~0.9714±0.0109)
- Champion md5 still 80a90f7cc210276300eaa90173a5a385
- No train jobs; GPU cool before start

Last session (2026-07-22): WP7 XAI J10 drop full claim keep structured+dispatch;
WP8 ToN CAD-CBA mapped val 0.8080 test 0.8110 RF 0.9393; F9 energy table;
WP6a re-export fidelity PASS; CAD-CBA-v1 locked; champion unchanged.

Next:
A) Final config freeze → sealed multi-seed TEST (B14)  ← only if user confirms lock
B) WP6b local multi-session latency/energy ranges after lock
C) WP9a claims packaging / numbers-match (verify_claims + manifest)
D) WP9b manuscript spine only when tracker largely green
E) DICC only if user opens dedicated session

Rules: no invent multi-day numbers; no clobber champion without BACKUP;
thermal guard if sustained train; commit/push; end per HANDOFF lifecycle.
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
- Official cluster: UM DICC only — **user will open a dedicated DICC session**  
- Results under `benchmarks/results/` often **gitignored** — use `RESULTS_DISK_MANIFEST.md` + local paths  
- Agents: no invented DICC numbers  
- Classical: prefer `summary_handoff.json` (LGBM official **0.9818** fixed; not legacy 0.5512)  
- Teacher KD: stage_a KD ≠ stage_b FT numbers  
- HPO: `hpo/summary.json` + `hpo_best.yaml`  
- C*: `cstar_bounded/` — all negative vs CTRL; not package  
- E6: `teachers_kd_neural/` — weaker than ensemble  
- XAI: `xai/` — drop full explainable claim; keep dispatch + structured  
- ToN: `toniot_final/` — 13-feat protocol ≠ historical 26-feat clean 0.9526  
- Pareto: `pareto/` analysis + `pareto_h8/` systems rebench  
- D6 stratified: keep **shuffle** default  
- Laptop thermal: soft 85 / hard 90  

---

## Older checkpoints (still valid)

Design plan Option A approved; FINAL_PLAN P0–P5; audit pack `docs/audit/`; interim Word report sent to Prof; feedback in `docs/feedback1.docx`.

**Next science priority after verify disk:** sealed multi-seed test (user lock) / WP6b ranges / WP9 claims. DICC only when user opens that session.
