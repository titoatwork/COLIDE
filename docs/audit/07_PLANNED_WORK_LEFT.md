# 07 — Planned Work Left (FROM REPO PLANS ONLY)

Sources: `docs/FINAL_PLAN.md`, `docs/DICC_OPS_METHOD.md`, `docs/PROF_POR_3DAY.md`, `docs/DESIGN_PLAN.md`, `HANDOFF.md`.  
**No invented deadlines beyond what docs state.**

**Last hygiene update:** 2026-07-22 (DICC ops → OnDemand VNC + screen + batch)

---

## Strategy freeze (DONE / LOCKED)

| Item | Evidence |
|------|----------|
| Option A approved | DESIGN_PLAN; HANDOFF; FINAL_PLAN §1 |
| Champion freeze md5 `80a90f7cc210276300eaa90173a5a385` | FINAL_PLAN; HANDOFF; disk |
| Official cluster = UM DICC only | FINAL_PLAN; HANDOFF |
| No full-pipeline CUDA vs full V3 claim | DESIGN_PLAN; FINAL_PLAN |
| DICC ops method | **`docs/DICC_OPS_METHOD.md`** — OnDemand VNC + `screen` + batch |
| Local science playlist closed | PLAYLIST_CLOSURE_AUDIT; tracker 133/133 terminal |
| Numbers-match hard gate before final multi-day Prof email | FINAL_PLAN P2 |

**Stale (removed as primary plan):** campus-stable runner; Cheran-as-default cluster operator; long interactive `srun`/`salloc` over VPN as the campaign path.

---

## FINAL_PLAN phases

### P0 — Unblock DICC — **BLOCKED (execution) / method LOCKED**

| Task | Status | Evidence |
|------|--------|----------|
| Ops method: OnDemand VNC + screen + batch | **LOCKED** | `DICC_OPS_METHOD.md`; FINAL_PLAN P0 |
| User runs campaign on DICC (own account) | **NOT DONE** | no SUCCESS tree on laptop |
| Exit: ≥1 GPU class Day1 SUCCESS | **NOT MET** | `benchmarks/results/dicc/` **ABSENT** |

### P1 — Multi-day complete + artifacts home — **NOT DONE**

| Task | Status |
|------|--------|
| Day2 SUCCESS same GPU class | NOT DONE |
| `compare_dicc_sessions.py` accept | NOT RUN (no inputs) |
| scp/rsync `benchmarks/results/dicc/` to laptop | **ABSENT** |
| Clock after start | ~2–5 days wall (queue-dominated) |

### P2 — Extract + numbers match + multi-day insert — **BLOCKED on P1**

| Task | Status |
|------|--------|
| P2a extract from DICC JSON | **BLOCKED** (no multi-day artifacts) |
| P2b numbers match across README/docs/claims | Local claims green; multi-day cells still empty |
| P2c `verify_claims.py` green | **PASS** for local package (64 claims as of 2026-07-22) |
| P2d multi-day Prof/manuscript update | **FORBIDDEN** until SUCCESS tree + match |

### P3–P5 — Local residual

Local-complete manuscript + PI venue polish **DONE**. Remaining PI: authors/venue/BibTeX. Stretch optional. DICC §5.13 insert after P1.

---

## What is NEXT

1. User: OnDemand VNC + `screen` + run card in `docs/DICC_OPS_METHOD.md`.  
2. When `benchmarks/results/dicc/` on laptop → compare + extract + claims + manuscript §5.13.  
3. No invent multi-day numbers.

**While waiting:** do **not** treat multi-day cells as filled; keep local claims green.
