# 08 — Unplanned Improvements (OPTIONAL)

**Label:** All items below are **OPTIONAL / UNPLANNED** auditor recommendations.  
They are **not** approved work packages and **do not** appear as required gates in FINAL_PLAN unless noted.

---

| ID | Problem | Proposed change | Effort | Risk | Blocks status report? | Helps manuscript? |
|----|---------|-----------------|--------|------|----------------------|-------------------|
| U1 | `benchmarks/results/` gitignored; load-bearing JSONs missing from clone | Force-add / un-ignore curated "claim sources" set; or publish `results_manifest` + checksums | S–M | Low if curated | **Yes for external repro** | Yes |
| U2 | Multi-session ranges hardcoded in verify_claims | Persist `sessions/*.json` array; range computed only from files | M | Medium (rewrite verify) | No | Yes (reviewer auditability) |
| U3 | verify_claims ignores Option A language | Add linter forbidding "same computation" full-pipeline CUDA vs V3; require "per-block"/"implemented blocks" | S | Low | **Yes for honest Prof/paper text** | Yes |
| U4 | Per-block README 62/87/20 ≠ live means | Re-run multi-session for B1/B2/B4; publish ranges or update points | M | Needs GPU time | No for contingency local status | Yes |
| U5 | Stale F1 inside cuml JSON (0.9639/0.9601) | Update labels or add `note: stale` | S | Low | No | Hygiene |
| U6 | No multi-day DICC | Run campaign (this is **planned** P0–P1 — listed here only as evidence strength) | L | Ops | **Yes for final DICC numbers** | Yes |
| U7 | Missing same-GPU PT on legacy DICC | Campaign pytorch_gpu_stats (planned tooling) | L | Ops | Yes for cluster ratios | Yes |
| U8 | Additive pipeline / skip B3 | Footnote everywhere OR true chain timing | M–L | Medium | No if footnoted | Yes |
| U9 | Energy methodology weak (pynvml sampling) | Document protocol; multi-run CIs | M | Low | No | Yes |
| U10 | Dead scripts litter | Archive `scripts/legacy/` | S | Low | No | Clarity |
| U11 | Stage-2 not bit-repro | Optional re-prove or stop claiming | M | Train risk — **do not clobber champion** | No | Integrity |
| U12 | README↔JSON CI | GitHub Action: verify_claims on PR | S | Low | No | Yes continuous |
| U13 | Claim language vs framework full V3 | Split tables: (a) per-block CUDA (b) full V3 framework absolutes | M | Doc rewrite | **Recommended before paper abstract freeze** | Yes |
| U14 | Rostam provisional B3 loss | Confirm/refute on UM (planned science risk R2) | L | Science | Yes for portable "beats cuDNN" | Critical |
| U15 | PROF_POR_STATUS_REPORT non-auth | Mark banner SUPERSEDED or delete after audit | S | Low | Avoids confusion | Process |
| U16 | Session tagging | Every results JSON: git_sha, hostname, gpu, timestamp, n_trials | M | Low | No | Yes |
| U17 | Expand CLAIMS for per-block points / preprocess 43.7 | Add or remove from README | S | Low | No | Yes |
| U18 | Multi-day artifact schema checklist | `validate_dicc_tree.py` checks SUCCESS schema | S | Low | Helps P1 ingest | Yes |

---

## Priority if only 3 OPTIONAL items before report writing

1. **U3** Option A claim linter / fix "same computation" wording (truthfulness).  
2. **U1** Track claim-source JSONs (reproducibility of evidence pack).  
3. **U13** Split abstract tables per Option A (report quality).

Multi-day DICC remains the **planned** critical path for cluster cells — not optional.
