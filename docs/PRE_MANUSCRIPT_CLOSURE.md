# Pre-manuscript closure pack

**Date (UTC):** 2026-08-12  
**Scope:** Everything **before** manuscript prose/PDF/venue.  
**Authority:** Option A · JSON only · `DICC_RESULTS_AND_FLAGS.md` · `DICC_EXTRACTION_TABLES.md` · `DICC_COMPARE_OUTCOMES.md`

---

## 1. Pre-manuscript gate checklist

| Gate | Status |
|------|--------|
| Local science playlist closed | **DONE** |
| Champion frozen `80a90f7…` | **DONE** |
| DICC S1 + S2 + Day2 SUCCESS × V100S + A100 | **DONE** (6 runs) |
| Results on laptop under `benchmarks/results/dicc/` | **DONE** |
| Force-added SUCCESS runs to git | **DONE** |
| Extraction tables from JSON | **DONE** → `docs/DICC_EXTRACTION_TABLES.md` |
| Cross-session compare V100 | **DONE** (S1–S2 stable; S1–Day2 B1 spread high) |
| Cross-session compare A100 | **DONE** with `--allow-dirty` (stable); provenance caveat noted |
| Fork decision (B3 CUDA vs PT) | **DONE** §2 below |
| Tracker DICC rows updated | **DONE** (PARTIAL→closed with honesty) |
| Multi-compiler DICC matrix | **NOT required** for pre-manuscript (stretch) |
| Manuscript multi-GPU insert | **OUT OF SCOPE** (next phase) |

**Pre-manuscript status: CLOSED for evidence collection.**  
Remaining is **claim strategy + writing**, not missing SUCCESS trees.

---

## 2. Fork decision (MOD / systems)

### Decision: **Option A honest — no portable B3 CUDA win**

| Finding | Action |
|---------|--------|
| On V100S/A100, **PT B3 faster** than CUDA B3 FP16 (~1.41× / ~1.72×) | **Do not claim** portable CUDA B3 beats matching PT |
| B1/B2/B4 CUDA ≫ PT | **Claim** per-block CUDA wins (Option A) |
| Full CUDA vs full V3 | **Forbidden** |
| Multi-session B3 means stable | **Claim** measurement stability for B3/full PT |
| Laptop Custom vs TRT/compile/ORT | **Claim** as **local** multi-session ranges only |
| Detection multi-obj package | **Primary** scientific contribution alongside systems measurement |

**Recorded:** 2026-08-12 · evidence: six SUCCESS JSON dirs + compare JSON under `benchmarks/results/dicc/core/`.

---

## 3. What “exceptional” needs beyond minimum (stretch menu)

Prioritized for Q1-leaning **systems + multi-obj** paper (not accuracy-SOTA).

### Tier S1 — high value / moderate effort (recommended)

| ID | Stretch | Why exceptional | Effort |
|----|---------|-----------------|--------|
| **S1a** | **torch.compile on DICC** V100+A100 (absolute full-model µs, multi-trial) | Closes “compilers on server” without TRT install; torch already in micromamba | 1–2 days |
| **S1b** | **Clean A100 re-run** (clean git tree) so formal compare needs no `--allow-dirty` | Provenance polish for reviewers | 0.5–1 day + queue |
| **S1c** | **Claim map + README hygiene** — remove portable overclaims; dual tables (block vs framework absolute) | Exceptional honesty; prevents desk reject | 0.5 day |
| **S1d** | **Welch/d write-up** of B3 CUDA vs PT (report PT win with CI) | Turns “loss” into rigorous result | 0.5 day |

### Tier S2 — high impact / higher effort

| ID | Stretch | Why | Effort |
|----|---------|-----|--------|
| **S2a** | **B3 kernel optimization pass** + re-DICC B3 only | Only path to reverse PT B3 win | days–weeks + risk |
| **S2b** | **ORT-GPU on DICC** if `onnxruntime-gpu` installs | Extends multi-compiler story to servers | 1–2 days if pip works |
| **S2c** | **Nsight brief** V100 vs A100 bottleneck (why B3 CUDA slower on A100) | Strong systems depth | 1–2 days |
| **S2d** | Energy/power on DICC if tools available | Multi-obj on cluster | unknown / may block |

### Tier S3 — optional / reviewer-driven

| ID | Stretch | Note |
|----|---------|------|
| **S3a** | TensorRT on DICC | Hard (no module; install TRT); FINAL_PLAN: only if demanded |
| **S3b** | Option B full CUDA = V3 (attention/LN/GAP) | Large; only if full-pipeline claim mandatory |
| **S3c** | Retrain to beat RF F1 | Out of freeze scope unless PI prioritizes |

### Recommended exceptional package (default)

```text
Pre-manuscript (this pack)     ✅
+ S1c claim/README hygiene
+ S1d B3 statistical report (honest)
+ S1a torch.compile on DICC (V100+A100)
+ S1b clean A100 re-run if time
(+ S2c Nsight if time before write)
```

**Do not** bet the paper on S2a (B3 optim) unless you accept delay and possible still-lose.

---

## 4. Contribution spine for exceptional framing

1. **Multi-objective CAD-CBA package** under sealed protocol (detection competitive, not pure F1 king).  
2. **Measurement-first multi-GPU study**: three sessions, two GPU classes, JSON-backed, formal compares.  
3. **Option A systems results**: CUDA wins B1/B2/B4; **PT wins B3 on servers** (portable nuance).  
4. **Laptop multi-compiler** (TRT/compile/ORT) with incomplete-CUDA caveats.  
5. **LLM dispatch** micro-result + no full XAI overclaim.  
6. Stretch: **server torch.compile** + optional Nsight “why B3”.

This is defensible for **systems / FGCS-leaning** venues; weaker for “we beat cuDNN everywhere” HPC-only pitches.

---

## 5. Immediate next actions (execute after you approve stretch picks)

| # | Action | Owner |
|---|--------|-------|
| 1 | Approve stretch set (default S1a–d) | **You** |
| 2 | Run S1a compile jobs on DICC if approved | Agent + VPN |
| 3 | S1c README claim strip | Agent |
| 4 | S1d one-page B3 stats summary | Agent |
| 5 | Then manuscript multi-GPU section | Writing phase |

---

*Pre-manuscript evidence gate: CLOSED. Stretch is for exceptional polish, not basic completeness.*
