# 07 — Planned Work Left (FROM REPO PLANS ONLY)

Sources: `docs/FINAL_PLAN.md`, `docs/PROF_POR_3DAY.md`, `docs/DESIGN_PLAN.md`, `HANDOFF.md`.  
**No invented deadlines beyond what docs state.**

**Audit HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`

---

## Strategy freeze (DONE / LOCKED)

| Item | Evidence |
|------|----------|
| Option A approved | DESIGN_PLAN status APPROVED 2026-07-14; HANDOFF; FINAL_PLAN §1 |
| Champion freeze md5 `80a90f7cc210276300eaa90173a5a385` | FINAL_PLAN; HANDOFF; **confirmed on disk this audit** |
| Official cluster = UM DICC only | FINAL_PLAN; HANDOFF |
| No full-pipeline CUDA vs full V3 claim | DESIGN_PLAN §5.1; FINAL_PLAN forbidden claim |
| Numbers-match hard gate before final Prof email | FINAL_PLAN P2; 23be30b |

---

## FINAL_PLAN phases

### P0 — Unblock DICC — **BLOCKED / IN PROGRESS**

| Task | Status | Evidence |
|------|--------|----------|
| Await Prof reply on Cheran cluster help | **BLOCKED** (pending human) | FINAL_PLAN §3 P0; HANDOFF header |
| Cheran or user runs campaign on DICC | **NOT DONE** | no SUCCESS tree on laptop |
| No credential sharing | standing rule | FINAL_PLAN |
| Exit: ≥1 GPU class Day1 SUCCESS | **NOT MET** | `benchmarks/results/dicc/` **ABSENT** |

### P1 — Multi-day complete + artifacts home — **NOT DONE**

| Task | Status |
|------|--------|
| Day2 SUCCESS same GPU class | NOT DONE |
| `compare_dicc_sessions.py` accept | NOT RUN (no inputs) |
| scp `benchmarks/results/dicc/` to laptop | **ABSENT** |
| Clock after unblock | ~2–5 days wall (FINAL_PLAN) |

### P2 — Extract + codebase-wide numbers match + Prof report — **BLOCKED on P1; partial local ready**

| Task | Status |
|------|--------|
| P2a extract from DICC JSON | **BLOCKED** (no multi-day artifacts) |
| P2b numbers match across README/docs/claims/email | **NOT COMPLETE** as final gate; local verify_claims green but Option A language still risky |
| P2c `verify_claims.py` green | **PASS this audit** (66/0) — necessary not sufficient |
| P2d send Prof report | **FORBIDDEN until P2a–c** per FINAL_PLAN; contingency draft exists but **non-authoritative** |
| Clock after P1 | ~1–2 days |

### P3 — Pre-manuscript residual hygiene — **NOT STARTED as exit**

Threats-to-validity text partially exists (S7 `3f99243`, DESIGN_PLAN, paper_text_blocks). Exit after P2.

### P4 — Manuscript spine — **EXPLICITLY DEFERRED**

FINAL_PLAN: not started until P2 complete. Clock ~1.5–3 months calendar after.

### P5 — Stretch — **EXPLICITLY DEFERRED / OPTIONAL**

| Item | When (per FINAL_PLAN) |
|------|------------------------|
| Nsight V100 vs A100 note | after P3 if time |
| Full TRT/ORT/compile on DICC | only if reviewers demand |
| Option B CUDA = full V3 | large; only if full-pipeline claim mandatory |
| Retrain to close RF gap | out of scope for freeze |

---

## From HANDOFF — completed local arc (evidence)

| Work | Status |
|------|--------|
| S4 baseline + multi-session roadmap | CLOSED |
| S5 ensemble Val-F1 fix + RF strengthen | CLOSED (champion kept) |
| S6 balanced-RF KD | SKIPPED low EV |
| S7 threats-to-validity + fidelity | CLOSED |
| S8 batch-size note | SKIPPED |
| S9 DICC_RUNBOOK | SUPERSEDED by dicc_scripts |
| Branch unify final-polish→master | DONE |
| Rostam Day 1 | Trial only |
| Historical UM June 2026 summaries | EXISTS as LEGACY files |
| Design plan approval | APPROVED |
| Deep codebase audit (this session) | IN PROGRESS → deliverable |

---

## From PROF_POR_3DAY — deliverable shape when unblocked

- Fill §4 tables from **JSON only** (Block 3 CUDA vs PT same GPU; absolute full V3 PT; **no** invalid full CUDA/full V3 ratio).
- Label June 551/592 only as **legacy single-shot**.
- Do not substitute Rostam as UM official.

---

## DESIGN_PLAN work packages (post-approval order)

Summarized from DESIGN_PLAN §6/§10:

1. Design plan — **DONE**  
2. WP1 sync tarball to DICC — **ops / blocked**  
3. WP2 Day1 run_campaign — **blocked**  
4. WP3 Day2 + compare — **blocked**  
5. Ingest + claim hygiene — **after artifacts**  
6. Prof pack — **after numbers match**  
7. Manuscript — deferred  

---

## What is NEXT after unblock (quote synthesis from FINAL_PLAN §7)

1. Prof decision on Cheran/DICC.  
2. If yes → run card P0→P1.  
3. When `benchmarks/results/dicc/` on laptop → P2a–P2d (match + verify before email).  
4. Only then P3 / P4 / P5.

**While waiting (this audit's intent):** rebuild evidence feedstock; do **not** treat multi-day cells as filled.
