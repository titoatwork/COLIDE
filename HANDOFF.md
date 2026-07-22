# COLIDE — Session Handoff

**MODE:** 🚀 **EXECUTION — next chat continues Prof tracker.** This chat closed for continuity.  
**Closed:** 2026-07-22 · **B14 sealed multi-seed BoT TEST DONE** (path A; test 0.9780±0.0033).  
**Authority:** `docs/execution_plan/SESSION_CONTINUITY.md` + `RESULTS_DISK_MANIFEST.md` + `PROF_FEEDBACK_TRACKER.md` + Option A.  
**Policy:** skip nothing → **complete every playlist/tracker row** → JSON → INCORPORATED or RUN_DOCUMENTED (BLOCKED only for ops). **Perfection over LOR hurry.** No invent DICC numbers. Context hygiene: flip statuses when evidence already exists.  
**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`** — no clobber without BACKUP.  
**Session end:** always paste full next-session prompt in closing message (not only in this file).

---

## This chat / arc delivered

| Deliverable | Status |
|-------------|--------|
| Protocol + WP1b–WP9a science arc | DONE (prior) |
| User freeze lock CAD-CBA-v1 path **A** | **DONE** |
| **B14 sealed multi-seed TEST** seeds 42–46 | **DONE** test **0.9780±0.0033**; Theft **1.0**; min-cls **0.9292** |
| Claims package rebuild (46 claims) | **DONE** verify_claims green |
| Freeze card | **USER-LOCKED** + B14 result table |
| Champion | **unchanged** |
| WP6b multi-session ranges | **NOT RUN** (next) |
| WP9b manuscript spine | **NOT RUN** |
| DICC | **ABSENT** — dedicated session when user opens it |

**CAD-CBA-v1 locked (science + sealed test):** V3 + focal + ensemble KD + hpo_best + shuffle + argmax.  
**XAI paper path:** dispatch + structured only — no full LLM-explainable title.  
**Jobs:** expect idle GPU after this handoff.

---

## Open with (next chat)

1. `HANDOFF.md` (this header)  
2. **`docs/execution_plan/SESSION_CONTINUITY.md`**  
3. **`docs/execution_plan/RESULTS_DISK_MANIFEST.md`**  
4. **`docs/execution_plan/PROGRESS_LOG.md`**  
5. **`docs/execution_plan/PROF_FEEDBACK_TRACKER.md`**  
6. **`docs/execution_plan/CLAIMS_REGISTRY.md`**  
7. `benchmarks/results/sealed_test/summary.json` (B14)

---

## Paste-ready next-session prompt

```text
Continue COLIDE — FULL PROF FEEDBACK EXECUTION. Skip nothing. Exceptional quality.
Perfection over LOR hurry. Option A CUDA locked. NO new jobs until you read continuity + verify disk.

FULL PLAYLIST LAW (user-locked):
- Complete EVERY tracker / WP playlist item → DONE | INCORPORATED | RUN_DOCUMENTED | BLOCKED(ops only).
- No silent skips of “optional” work (WP6b ranges, manuscript gates, etc.).
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
8) docs/execution_plan/CLAIMS_REGISTRY.md
9) docs/execution_plan/FINAL_CONFIG_FREEZE_CARD.md
10) config/hpo_best.yaml

Verify on disk:
- benchmarks/results/sealed_test/summary.json  (B14 DONE; test ~0.9780±0.0033; Theft 1.0)
- benchmarks/results/claims_package/protocol_claims.json  (46 claims; sealed LOCKED_TEST)
- benchmarks/results/xai/summary.json  (J10 DROP_FULL; rank_corr ~0.9636; faith ~0.5109)
- benchmarks/results/toniot_final/summary.json  (val ~0.8080; test ~0.8110; RF test ~0.9393)
- benchmarks/results/energy_table/summary.json  (RTX ~0.786 mJ/flow)
- benchmarks/results/numerical_fidelity.json  (bit-identical + CUDA PASS)
- benchmarks/results/hpo/summary.json + config/hpo_best.yaml
- benchmarks/results/multirun/summary.json  (WP1b ~0.9714±0.0109)
- Champion md5 still 80a90f7cc210276300eaa90173a5a385
- No train jobs; GPU cool before start
- PYTHONPATH=. python3 scripts/verify_claims.py  → all green

Last session (2026-07-22): User locked CAD-CBA-v1 path A; B14 sealed multi-seed TEST DONE
(test 0.9780±0.0033 n=5; min-cls 0.9292; Theft 1.0; champion unchanged); claims rebuild
46 claims green; freeze card LOCKED; WP6b NOT run.

Next:
A) WP6b local multi-session latency/energy ranges (Option A)
B) Keep verify_claims green after new systems JSON
C) WP9b manuscript spine when tracker largely green
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
# after claim edits:
PYTHONPATH=. python3 scripts/build_claims_package.py
PYTHONPATH=. python3 scripts/verify_claims.py
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
- BoT sealed multi-seed **test DONE** (B14 path A) — numbers in `sealed_test/`  
- Official cluster: UM DICC only — **user will open a dedicated DICC session**  
- Results under `benchmarks/results/` often **gitignored** — use `RESULTS_DISK_MANIFEST.md` + local paths  
- Agents: no invented DICC numbers  
- Claims: rebuild with `build_claims_package.py`; verify with `verify_claims.py`  
- Classical: prefer `summary_handoff.json` (LGBM official **0.9818**)  
- B14 test **0.9780±0.0033** — label as **test**; do not mix with val multirun  
- XAI: drop full explainable claim; keep dispatch + structured  
- ToN: `toniot_final/` — 13-feat protocol ≠ historical 26-feat clean  
- Laptop thermal: soft 85 / hard 90  

---

## Older checkpoints (still valid)

Design plan Option A approved; FINAL_PLAN P0–P5; audit pack `docs/audit/`; interim Word report sent to Prof; feedback in `docs/feedback1.docx`.

**Next science priority after verify disk:** **WP6b** local multi-session ranges → manuscript when green. DICC only when user opens that session.
