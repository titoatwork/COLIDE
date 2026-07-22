# COLIDE — Session Handoff

**MODE:** 🚀 **EXECUTION — next chat continues Prof tracker.** This chat closed for continuity.  
**Closed:** 2026-07-22 · **WP6b local multi-session ranges DONE** (energy 0.920–0.943; PT@256 24.15–25.68 µs).  
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
| **WP6b local multi-session ranges** | **DONE** energy **0.920–0.943**; PT@256 **24.15–25.68** µs; CUDA pipe **565–570** µs; peak **322.2** MiB |
| I7 warm-up / I8 batch sensitivity / H3 peak VRAM | **DONE** (under WP6b) |
| Claims package rebuild | **DONE** **59** claims; verify_claims green |
| Freeze card | **USER-LOCKED** + B14 result table |
| Champion | **unchanged** |
| WP9b manuscript spine | **NOT RUN** (next when tracker largely green) |
| DICC | **ABSENT** — dedicated session when user opens it |

**CAD-CBA-v1 locked (science + sealed test + local systems ranges):** V3 + focal + ensemble KD + hpo_best + shuffle + argmax.  
**XAI paper path:** dispatch + structured only — no full LLM-explainable title.  
**Systems:** report **ranges** (WP6b); historical single-shot energy 0.786 labeled HISTORICAL.  
**Jobs:** expect idle GPU after this handoff.

---

## Open with (next chat)

1. `HANDOFF.md` (this header)  
2. **`docs/execution_plan/SESSION_CONTINUITY.md`**  
3. **`docs/execution_plan/RESULTS_DISK_MANIFEST.md`**  
4. **`docs/execution_plan/PROGRESS_LOG.md`**  
5. **`docs/execution_plan/PROF_FEEDBACK_TRACKER.md`**  
6. **`docs/execution_plan/CLAIMS_REGISTRY.md`**  
7. `benchmarks/results/wp6b_local_ranges/summary.json`  
8. `benchmarks/results/sealed_test/summary.json` (B14)

---

## Paste-ready next-session prompt

```text
Continue COLIDE — FULL PROF FEEDBACK EXECUTION. Skip nothing. Exceptional quality.
Perfection over LOR hurry. Option A CUDA locked. NO new jobs until you read continuity + verify disk.

FULL PLAYLIST LAW (user-locked):
- Complete EVERY tracker / WP playlist item → DONE | INCORPORATED | RUN_DOCUMENTED | BLOCKED(ops only).
- No silent skips of “optional” work (manuscript gates, residual PARTIAL rows, etc.).
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
10) docs/execution_plan/13_PHASE9_MANUSCRIPT.md
11) config/hpo_best.yaml

Verify on disk:
- benchmarks/results/wp6b_local_ranges/summary.json  (WP6b DONE; energy 0.920–0.943; PT@256 24.15–25.68)
- benchmarks/results/systems_i8_h3/summary.json  (I8/H3 mirror)
- benchmarks/results/sealed_test/summary.json  (B14 DONE; test ~0.9780±0.0033; Theft 1.0)
- benchmarks/results/claims_package/protocol_claims.json  (59 claims)
- benchmarks/results/xai/summary.json  (J10 DROP_FULL; rank_corr ~0.9636; faith ~0.5109)
- benchmarks/results/toniot_final/summary.json  (val ~0.8080; test ~0.8110; RF test ~0.9393)
- benchmarks/results/energy_table/summary.json  (historical RTX ~0.786 mJ/flow)
- benchmarks/results/numerical_fidelity.json  (bit-identical + CUDA PASS)
- Champion md5 still 80a90f7cc210276300eaa90173a5a385
- No train jobs; GPU cool before start
- PYTHONPATH=. python3 scripts/verify_claims.py  → all green

Last session (2026-07-22): WP6b local multi-session ranges DONE (n=5; energy 0.920–0.943
mean 0.933; PT@256 24.15–25.68 mean 24.90; CUDA pipe 565–570; peak 322.2 MiB;
champion unchanged); claims 59 green; B14 already DONE; WP9b next.

Next:
A) WP9b manuscript spine (tracker largely green; science + local systems closed)
B) Flip any remaining PARTIAL tracker rows from existing disk evidence (A1/A2/A5/A6, C1, K4/K5, L10/L12, I9–I11, H7)
C) Keep verify_claims green after prose
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
- Local systems ranges **DONE** (WP6b) — numbers in `wp6b_local_ranges/`  
- Official cluster: UM DICC only — **user will open a dedicated DICC session**  
- Results under `benchmarks/results/` often **gitignored** — use `RESULTS_DISK_MANIFEST.md` + local paths  
- Agents: no invented DICC numbers  
- Claims: rebuild with `build_claims_package.py`; verify with `verify_claims.py`  
- Classical: prefer `summary_handoff.json` (LGBM official **0.9818**)  
- B14 test **0.9780±0.0033** — label as **test**; do not mix with val multirun  
- WP6b energy **0.920–0.943** — do not mix with historical single-shot **0.786**  
- XAI: drop full explainable claim; keep dispatch + structured  
- ToN: `toniot_final/` — 13-feat protocol ≠ historical 26-feat clean  
- Laptop thermal: soft 85 / hard 90  

---

## Older checkpoints (still valid)

Design plan Option A approved; FINAL_PLAN P0–P5; audit pack `docs/audit/`; interim Word report sent to Prof; feedback in `docs/feedback1.docx`.

**Next science priority after verify disk:** **WP9b manuscript spine** (tracker largely green). DICC only when user opens that session.
