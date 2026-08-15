# COLIDE manuscript handoff for Cheranrach Mahandren

**Date:** 2026-08-15  
**From:** Ibteshamul Haque  
**Pre-manuscript:** **CLOSED** — see `docs/PRE_MANUSCRIPT_INDEX.md`.  
**Role for this pack:** frozen evidence + a *working-draft* MD (not a finished paper) so **you can lead paper writing**.  
**Do not use** `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf` (22 Jul) as current — it is stale vs the Markdown.  
**Supervisor:** Prof. Por (updated on progress as you prefer).

---

## 1. How to access materials

### Preferred: GitHub (full tree)

- **Repo:** https://github.com/titoatwork/COLIDE  
- **Branch:** `master`  
- **Tip at handoff packaging:** pull latest `master` (includes manuscript sync, figures, claim gates, Option B B3 decision).

```bash
git clone https://github.com/titoatwork/COLIDE.git
cd COLIDE
git pull origin master
```

### Optional: zip package

If you prefer not to use git, use the zip named:

`COLIDE_Cheran_manuscript_pack_YYYYMMDD.zip`

(built from this repo; see §8). It contains writing, claim docs, figures, and key result JSON — not full multi-GB training trees.

---

## 2. Start here (read order, ~30–45 min)

| Order | Path | Why |
|------:|------|-----|
| 1 | **This file** | Map + hard rules |
| 2 | `docs/CLAIM_MAP_PREWRITE.md` | What is OK / FORBIDDEN to claim |
| 3 | `docs/RESULTS_INDEX.md` | Claim → artifact map |
| 4 | `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` | Current draft spine (synced 2026-08) |
| 5 | `docs/manuscript/TABLES_FROM_ARTIFACTS.md` | Tables pulled from locked JSON |
| 6 | `docs/B3_SERVER_LATENCY_DECISION.md` | Server B3: historical only (Option B) |
| 7 | `docs/PUBLICATION_REMAINING_2026-08-15.md` | What is still open vs closed |
| 8 | `docs/KNOWN_LIMITATIONS.md` | Limitations language for Discussion |

Optional deep dives: `README.md`, `docs/DICC_*`, `COLIDE_Remediation_Update_Review.md` (technical review status).

---

## 3. Locked numbers (use only these as principal)

### BoT-IoT (principal detection)

| Item | Value |
|------|--------|
| Principal test macro-F1 | **0.9780 ± 0.0033** (n=5, sealed multi-seed) |
| Champion | `model/best_model_botiot_twostage.pth` |
| Champion MD5 | **`80a90f7cc210276300eaa90173a5a385`** |
| Historical 0.9790 | Development/legacy only — **not** principal |

### ToN-IoT (secondary / external)

| Item | Value |
|------|--------|
| Protocol | `toniot_leakage_safe_v1` |
| CNN test macro-F1 | **0.8075** |
| RF test macro-F1 | **0.9626** |
| Seed | 42 · stratified random 60/20/20 · **not** official temporal/host split |
| Weak class | **mitm** CNN F1 ≈ **0.111** (low precision, high recall) |
| Invalid “clean” path | CNN **0.9526** / RF **0.9851** / **+15.4%** — **INVALID only** (label leakage) |

### CUDA / systems (claim carefully)

| Item | Status for the paper |
|------|----------------------|
| Local B3 production-weight full-sequence parity | **Closed** — `benchmarks/results/block3_parity_gate.json` (`valid=true`, GPU vs PT max abs ~3.43e-6) |
| Local sanitizers (RTX 3050) | **Closed** — racecheck/synccheck/initcheck/memcheck, 0 errors (`benchmarks/results/sanitizer_b3/`) |
| DICC B3 latency (V100S/A100) | **Historical pre_fix only** — do **not** claim post_fix server rebench (Option B) |
| Full custom CUDA vs full V3 | **FORBIDDEN** (Option A: Blocks 1–4 incomplete) |
| Framework tables | Separate: operator/block vs full-model; no cross-table speedups |
| Throughput ~25,899 f/s | **Bulk batched** only — not true streaming arrivals |
| Energy | **Exploratory** board-power language |
| LLM 16.60 µs p99 | **Alert dispatch** only — not free-form XAI quality |
| Native TensorRT numerical equivalence | **Not validated** (no engine in gate) — do not claim |

---

## 4. Writing assets

| Asset | Path |
|-------|------|
| Manuscript draft (MD) | `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` |
| Older PDF snapshot | `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf` (**prefer MD**; PDF may lag) |
| Figures | `docs/manuscript/figures/` (also copy under `benchmarks/results/figures_current/`) |
| New ToN per-class figure | `docs/manuscript/figures/fig_toniot_corrected_cnn_per_class.png` |
| Figure status | `docs/FIGURE_STATUS.md` |
| Text blocks / reusable prose | `docs/paper_text_blocks.md` |
| Plain numbers card | `docs/PROF_PLAIN_NUMBERS_CARD.md` |

---

## 5. Evidence folders (JSON)

Do not invent numbers. Prefer:

| Topic | Path |
|-------|------|
| Sealed BoT multi-seed | `benchmarks/results/sealed_test/` (where present) |
| Corrected ToN | `benchmarks/results/toniot_corrected/summary.json` |
| B3 parity gate | `benchmarks/results/block3_parity_gate.json` |
| Framework logit gate | `benchmarks/results/framework_parity_gate.json` |
| Sanitizers | `benchmarks/results/sanitizer_b3/` |
| DICC multi-session | `benchmarks/results/dicc/core/` + `docs/DICC_*` |
| Multi-compiler DICC | `docs/DICC_MULTI_COMPILER_MATRIX.md` + framework JSON under `benchmarks/results/dicc/` |
| Invalid ToN tombstone | `benchmarks/results/toniot_clean_comparison.json` (`valid: false`) |

---

## 6. Hard claim rules (please keep)

1. Principal accuracy = sealed **0.9780 ± 0.0033**, multi-objective / not pure-F1 SOTA vs protocol LGBM.  
2. Never revive ToN clean **0.9526 / 0.9851 / +15.4%** as valid.  
3. Never ratio incomplete Custom CUDA pipeline vs full-model eager/compile/TRT as end-to-end model speedup.  
4. Never present DICC B3 µs as **post_fix** until a new SUCCESS rebench exists (we chose Option B).  
5. Local B3 parity ≠ server latency claim.  
6. Pseudo-sequence architecture: latent reshape, not raw packet chronology.  
7. Title/abstract: detection + systems/CUDA Option A + scoped dispatch — not “production LLM explainability.”

When unsure: open `docs/CLAIM_MAP_PREWRITE.md` or ask Ibteshamul before drafting a new quantitative claim.

---

## 7. Suggested division of labour

| You (Cheran) — lead writing | Ibteshamul — support |
|-----------------------------|----------------------|
| Structure, narrative, venue style, related work polish | Artifact lookup, number checks, CUDA/DICC wording |
| Abstract / intro / discussion / conclusion | Tables from JSON, figure captions fact-check |
| Unify MD → Word/LaTeX/PDF for venue | Run parity / regenerate a figure if needed |
| Flag any claim that feels overstrong | Confirm against gate JSON within 24–48 h |

Prof. Por: high-level updates only, as he directed.

---

## 8. Zip package contents (if built)

Typical include list:

- `docs/manuscript/` (md + figures)  
- `docs/CLAIM_MAP_PREWRITE.md`, `RESULTS_INDEX.md`, `B3_SERVER_LATENCY_DECISION.md`, `KNOWN_LIMITATIONS.md`, `CHERAN_MANUSCRIPT_HANDOFF.md`, `PUBLICATION_REMAINING_2026-08-15.md`, `FIGURE_STATUS.md`, `PRE_MANUSCRIPT_CLOSURE.md`, `paper_text_blocks.md`, key `DICC_*.md`  
- `benchmarks/results/toniot_corrected/`  
- `benchmarks/results/block3_parity_gate.json`  
- `benchmarks/results/framework_parity_gate.json`  
- `benchmarks/results/sanitizer_b3/summary.json`  
- `benchmarks/results/figures_current/`  
- `README.md`  

Exclude: raw `data/`, large untracked `model/*` experiment trees, `logs/`, `.venv/`.

Champion weights are large; link from GitHub or share separately if needed:

`model/best_model_botiot_twostage.pth` (md5 `80a90f7…`).

---

## 9. Contact / next step

After you have pulled the repo (or unzipped the pack), a short call or email on:

1. Target venue / template  
2. Whether you want LaTeX or Word first  
3. Any table you want regenerated from JSON  

is enough to start drafting.

— End handoff —
