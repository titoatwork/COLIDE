# 15 — Work Packages (One Package ≈ One Focused Session)

**Rule:** Do not stack DICC + full HPO + manuscript in one chat.  
**Order:** Respect phase dependencies.

---

## Package board

| ID | Name | Phase | Depends | Owner | Status |
|----|------|-------|---------|-------|--------|
| WP0 | UM DICC Day1+Day2+compare+scp | 0 | — | User/ops + coach | **TODO** |
| WP0b | Extract DICC JSON tables + fork decision | 0 | WP0 | Agent+user | TODO |
| WP1a | Unified `botiot_protocol.py` + freeze card | 1 | — (prep ok) | Agent | **DONE 2026-07-19** |
| WP1b | Multi-run baseline driver (5 seeds) | 1 | WP1a | Agent | **IN_PROGRESS** (driver done; seed42 smoke running) |
| WP2a | Sign method package (MOD table) | 2 | WP0b preferred | User+agent | TODO |
| WP2b | Implement losses + thresholds modules | 2 | WP1a | Agent | **DONE** |
| WP2c | Architecture delta if any (attention/multi-scale) | 2 | WP2a | Agent | TODO |
| WP3 | Optuna HPO study + sealed test | 3 | WP1–2 | Agent | TODO |
| WP4a | Imbalance strategy sweep | 4 | WP1–3 | Agent | TODO |
| WP4b | Teacher sweep (RF/XGB/LGBM/ensemble) | 4 | WP1–3 | Agent | TODO |
| WP5a | Full ablation ladder | 5 | Final recipe | Agent | TODO |
| WP5b | Fair baseline suite | 5 | WP1a | Agent | TODO |
| WP5c | Pareto figure + composite score | 5 | WP5a/b + systems | Agent | TODO |
| WP6a | Re-export + fidelity | 6 | Final ckpt | Agent | TODO |
| WP6b | Local re-bench ranges Option A | 6 | WP6a | Agent | TODO |
| WP6c | Re-DICC if model changed | 6 | WP6a | User/ops | TODO |
| WP7 | XAI suite **or** drop explainable claims | 7 | Final detector | Agent | TODO |
| WP8 | ToN final-method eval | 8 | Final recipe | Agent | TODO |
| WP9a | Numbers-match + claim JSON packaging | 9 | All | Agent | TODO |
| WP9b | Manuscript spine + figures | 9 | WP9a | User+agent | TODO |

---

## What to run **right now**

| If DICC access available | Start **WP0** |
| If DICC blocked | Start **WP1a** (protocol unification) — prep only |
| Never first | WP9b manuscript with empty cluster cells |

---

## WP0 detail (highest priority)

**Inputs:** tarball or git tree on DICC, champion md5  
**Outputs:** `benchmarks/results/dicc/**/SUCCESS`, compare outcome, laptop copy  
**Exit:** Phase 0 checklist in `04_PHASE0_DICC.md` complete  

---

## WP1a detail (best parallel prep)

**Inputs:** current CSV loaders  
**Outputs:** `scripts/data/botiot_protocol.py`, freeze markdown, unit smoke test  
**Exit:** New train scripts can import one protocol  

---

## Tracking

Update Status column in this file when a WP completes. Link commit hash in HANDOFF.
