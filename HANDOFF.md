# COLIDE — Session Handoff

**MODE:** 🚀 **EXECUTION — next chat continues Prof tracker.** This chat closed for continuity.  
**Closed:** 2026-07-22 · **WP5b neural baselines G6–G12** (science + docs).  
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
| Package FT multirun ensemble KD + hpo_best 0.9639±0.0185 | DONE (prior) |
| Multi-seed HPO confirm 0.9689±0.0145 | DONE (prior) |
| WP5a ablation ladder A1–A7 A7 0.9699 | DONE (prior) |
| **WP5b neural baselines G6–G12** | **DONE** G11 **0.9493** tops suite; G6–G12 **RUN_DOCUMENTED**; G15 **DONE** |
| Systems metrics per baseline (params/latency/mem) | DONE (suite); F9 energy still open |

**DICC:** still **ABSENT** — dedicated session when user opens it.  
**Jobs:** neural baselines **finished** (~80 min wall); expect idle GPU.

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
- No silent skips of “optional” work (SupCon, arch HPO B2–B4, stratified D6, neural baselines, XAI, ToN, Pareto, sealed test, etc.).
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
- benchmarks/results/baselines_neural/summary.json  (G11 top ~0.9493 n=7 RUN_DOCUMENTED)
- benchmarks/results/ablation_ladder/summary.json  (A7 top ~0.9699 n=7)
- benchmarks/results/multirun_hpo_confirm/summary.json  (mean ~0.9689±0.0145 n=5)
- benchmarks/results/multirun_ensemble_hpo/summary.json  (mean ~0.9639±0.0185 n=5)
- benchmarks/results/hpo/summary.json  (winner ~0.9791 INCORPORATE)
- config/hpo_best.yaml
- benchmarks/results/teachers_kd/summary.json  (ensemble ~0.9401)
- benchmarks/results/multirun/summary.json  (WP1b ~0.9714±0.0109)
- Champion md5 still 80a90f7cc210276300eaa90173a5a385
- No train jobs; GPU cool before start

Last session (2026-07-22): WP5b neural baselines G6–G12 seed42 CE scratch equal budget ~80 min —
G11 0.9493 > G6 0.9285 > G10 0.8159 > G8 0.8099 > G9 0.8058 > G7 0.6221 > G12 0.5808;
G15 HPO note DONE; transformer weak; G7=A1 G9=A2 G11=A3 consistency; champion unchanged.

Next:
A) G2 SVM full + G5 LGBM classical fixes  ← recommended CPU
B) D6 stratified batch FT compare  ← recommended GPU
C) Bounded C* / WP5c Pareto / B2–B4 arch HPO if ready
D) DICC only if user opens dedicated session

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
- Classical: prefer `summary_handoff.json`  
- Teacher KD: stage_a KD ≠ stage_b FT numbers  
- HPO: `hpo/summary.json` + `hpo_best.yaml`  
- Ablation: `ablation_ladder/` ≠ multirun trees  
- Neural baselines: `baselines_neural/` ≠ historical `ablation_mlp.json`  
- HPO multi-seed confirm: `multirun_hpo_confirm/` ≠ package `multirun_ensemble_hpo/` ≠ WP1b `multirun/`  
- Laptop thermal: soft 85 / hard 90  

---

## Older checkpoints (still valid)

Design plan Option A approved; FINAL_PLAN P0–P5; audit pack `docs/audit/`; interim Word report sent to Prof; feedback in `docs/feedback1.docx`.

**Next science priority after verify disk:** G2/G5 classical fixes or D6 stratified. DICC only when user opens that session.
