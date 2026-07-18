# 10 — Evidence Gaps

**HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`

---

## Critical gaps (block honest "final" cluster status)

| Gap | What was checked | Impact |
|-----|------------------|--------|
| **Multi-day DICC tree ABSENT** | `ls benchmarks/results/dicc` → none; no SUCCESS dirs | Cannot fill Prof pack §4 multi-day cells; cannot claim multi-day stability |
| **No same-GPU PyTorch on legacy DICC** | dicc_*_summary.txt CUDA only | Cluster vs-PT ratios invalid until campaign |
| **Official campaign not run on hardened stack at UM** | HANDOFF / FINAL_PLAN | Planned only |
| **Prof decision pending** | FINAL_PLAN P0 | Ops unblock |

## Packaging / provenance gaps

| Gap | Detail |
|-----|--------|
| `.gitignore` ignores `benchmarks/results/` | 17 load-bearing files on disk only (twostage, RF processed, cuda stats, B3 PT stats, fidelity, round-2 KD, etc.) |
| Multi-session history not file-backed | Ranges depend on hardcoded HIST/S3A/EXTRA in verify_claims |
| Stage-2 FT not bit-repro | Only stage-1 repro documented |
| data/processed gitignored | RF/train re-runs need local data |

## Scientific / construct gaps

| Gap | Detail |
|-----|--------|
| Option A language still in README | "same computation" full-pipeline |
| CUDA ≠ full V3 | attention/LN/GAP missing; last-timestep vs GAP |
| fused_pipeline additive B3 | not true device chain |
| Per-block table points stale vs live stats | B1/B2 especially |
| Rostam B3 slower than PT | unconfirmed on UM; threatens portable beats-cuDNN |
| ORT CPU non-robust significance | already disclosed |
| Energy sampling methodology | limited documentation |
| LLM generation quality | illustrative only |
| Ibrahim PDF not fully read | DESIGN_PLAN 4.7 |

## What is enough for a **local-only honest status** (no multi-day)

Usable with labels: champion 0.9790, RF 0.9864, gap 0.74%, KD path, LLM 16.60, laptop latency **ranges**, fidelity PASS, legacy DICC 551/592 labeled legacy, multi-day **pending**, Option A caveats.

## What is **not** enough for final Prof "all numbers incl. multi-day DICC"

Missing P1 artifacts + compare accept + numbers match including new µs.

## verify_claims gaps

- Bold: 0.6, 1.00x, 10.0, 2.0, 3.3%
- No Option A semantic checks
- No check that per-block 62/87/20 match cuda_kernel_stats
- Green verifier ≠ complete manuscript readiness
