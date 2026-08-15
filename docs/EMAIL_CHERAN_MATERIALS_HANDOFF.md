> **Status: INTERNAL correspondence / NOT CURRENT AUTHORITY (frozen ~2026-08-15).**  
> Do not cite this file as the live claim surface or as manuscript evidence. Current authority: `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/RESULTS_INDEX.md`, `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
> Email / coordination draft only. Numbers or “ready to write” wording below may be historical.  
> Kept for audit trail. Stale numbers here may be superseded (ToN clean 0.9526 INVALID; principal BoT is 0.9780±0.0033; DICC B3 latency is pre_fix / Option B).  
> Public GitHub visitors: this is internal correspondence, not a results paper.

# Email to Cheran — manuscript materials handoff

**To:** Cheranrach Mahandren  
**CC:** Prof. Por (recommended)  
**Subject:** COLIDE paper materials for manuscript preparation (handoff pack)

---

Dear Cheran,

Thank you for coordinating the manuscript as advised by Prof. Por. I am sharing the COLIDE materials so you can lead the paper writing.

### How to get everything

**Option A — GitHub (recommended, full tree)**  
Repository: https://github.com/titoatwork/COLIDE  
Branch: `master` (please `git pull` the latest).

**Option B — Zip pack (writing + figures + key results)**  
Attached / shared: `COLIDE_Cheran_manuscript_pack_20260815.zip`  
Also on my side under the repo root and Downloads if you need a re-send.

### Start here

Please open first:

`docs/CHERAN_MANUSCRIPT_HANDOFF.md`

That one-page map lists locked numbers, claim rules (OK / forbidden), draft path, figures, and JSON evidence. Suggested reading order is in that file (~30–45 minutes).

### What is included for writing

| Item | Location |
|------|----------|
| Manuscript draft (synced text) | `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` |
| Figures | `docs/manuscript/figures/` |
| Tables from locked artifacts | `docs/manuscript/TABLES_FROM_ARTIFACTS.md` |
| Claim map (OK / FORBIDDEN) | `docs/CLAIM_MAP_PREWRITE.md` |
| Claim → result index | `docs/RESULTS_INDEX.md` |
| Corrected ToN results | CNN **0.8075** / RF **0.9626** (`benchmarks/results/toniot_corrected/`) |
| Principal BoT | **0.9780 ± 0.0033** (champion md5 `80a90f7…`) |
| CUDA / DICC honesty notes | Option A blocks; B3 local parity closed; **DICC B3 latency historical only** (`docs/B3_SERVER_LATENCY_DECISION.md`) |

### Important (please keep while drafting)

- Do **not** use historical ToN “clean” **0.9526 / 0.9851 / +15.4%** (invalid).  
- Do **not** claim full custom CUDA vs full V3 model speedup.  
- Do **not** treat DICC Block-3 timings as post-fix server rebench (we use Option B: historical label only).  
- Throughput ~25,899 f/s is **bulk batched**, not streaming arrivals.  
- LLM **16.60 µs** is dispatch overhead only.  

If any number is unclear, use the JSON under `benchmarks/results/` or ask me before putting it in the paper. I can turn around fact-checks quickly while you lead structure and prose.

Happy to do a short call on venue/template (Word vs LaTeX) whenever you are free.

Regards,  
Ibteshamul Haque  
FCSIT, Universiti Malaya

---

*Plain-text version for paste into email clients is the body above (from “Dear Cheran” through “Universiti Malaya”).*
