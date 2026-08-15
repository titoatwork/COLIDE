# Pre-manuscript closure pack

**Date (UTC):** 2026-08-15  
**Status:** **PRE-MANUSCRIPT CLOSED**  
**Scope:** Evidence freeze, claim hygiene, tables/figures from artifacts, B3 Option B, writer handoff — **before** venue manuscript writing.  
**Next:** Cheran leads paper writing from `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
**Start page:** `docs/PRE_MANUSCRIPT_INDEX.md`  
**Not the same as camera-ready:** a working draft MD exists; the Jul 22 PDF is stale; venue prose is still to write.

**Decision already taken (was listed as open):** DICC B3 post_fix latency → **Option B** (`docs/B3_SERVER_LATENCY_DECISION.md`): drop active comparative post_fix server-B3 claim; keep historical pre_fix labeled.

| Artifact | Path |
|----------|------|
| External review (readiness) | `COLIDE_Remediation_Update_Review.md` |
| Extraction tables | `docs/DICC_EXTRACTION_TABLES.md` |
| B3 report (S1d) | `docs/DICC_B3_CUDA_VS_PT_REPORT.md` |
| Multi-compiler matrix | `docs/DICC_MULTI_COMPILER_MATRIX.md` |
| Compare outcomes | `docs/DICC_COMPARE_OUTCOMES.md` |
| Results + flags | `docs/DICC_RESULTS_AND_FLAGS.md` |
| Claim map (pre-write) | `docs/CLAIM_MAP_PREWRITE.md` |
| Issue register | `docs/ISSUE_REGISTER.md` |
| Known limitations | `docs/KNOWN_LIMITATIONS.md` |
| Remediation status | `docs/REMEDIATION_STATUS.md` |
| Session handoff | `HANDOFF.md` |

**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`**

### Phase 1 claim quarantine (2026-08-14) — documentation only

Evidence freeze above is unchanged for BoT/DICC (no BoT retrain, no new DICC). Active claim surfaces were corrected for honesty:

| Quarantine | Action |
|------------|--------|
| Principal BoT | Sealed **0.9780 ± 0.0033**; historical **0.9790** development-only |
| ToN clean 0.9526 / 0.9851 / +15.4% | **INVALID** tombstone (DATA-TON-001) |
| Streaming | Reframed as **bulk batched throughput** (~25,899 f/s) |
| Energy | **Exploratory**; WP6b ranges preferred over bare 1.089 vs cuML |
| B3 DICC | Wall-clock honesty retained; **pre_fix** historical until multi-session post_fix latency rebench (local production-weight parity already **closed**) |
| Partial CUDA vs full model | **No cross-table speedups** (CLAIM-PIPE-001); Table A absolutes vs Table B full-model absolutes only |
| LLM | Dispatch-only; title narrowed |

Guard: `python scripts/check_stale_claims.py`.

### Phase 2 corrected ToN (complete) + CUDA B3 code status

| Item | Status |
|------|--------|
| Corrected ToN leakage-safe experiment | **DONE** — protocol `toniot_leakage_safe_v1`; test macro-F1 **CNN 0.8075** / **RF 0.9626**; no SMOTE, no KD; `benchmarks/results/toniot_corrected/` (`valid: true`, `use_in_manuscript: true`) |
| Invalid clean path | Remains **tombstoned** (0.9526 / 0.9851 / +15.4% never active) |
| Older 13-feat package path | Optional comparable prior only (~0.811 vs RF ~0.939, `toniot_final/`) |
| CUDA B3 race/alignment | Source fixed (double-buffer + reverse `pos`) |
| CUDA B3 production-weight parity (local) | **DONE** — `block3_parity_gate.json` `valid=true`, `kernel_status=post_fix` |
| CUDA B3 sanitizers (local sm_86) | **DONE** — racecheck/synccheck/initcheck/memcheck 0 errors FP32+FP16 |
| CUDA B3 DICC multi-session post_fix latency | **CLOSED as Option B** — comparative post_fix claim dropped; historical pre_fix retained only as historical (`docs/B3_SERVER_LATENCY_DECISION.md`) |

**Pre-manuscript readiness:** **CLOSED** (see `docs/PRE_MANUSCRIPT_INDEX.md`).  
**Camera-ready / venue manuscript:** **NOT started as a finished paper** — working-draft MD + tables/figures are feedstock for Cheran.

---

## 1. Pre-manuscript gate checklist

| Gate | Status |
|------|--------|
| Local science playlist closed | **DONE** (historical freeze) |
| Champion frozen `80a90f7…` | **DONE** |
| DICC S1 + S2 + Day2 SUCCESS × V100S + A100 | **DONE** (6 SUCCESS markers; B3 means are **pre_fix** binaries) |
| Results on laptop under `benchmarks/results/dicc/` | **DONE** |
| Force-added SUCCESS + multi-compiler JSON to git | **DONE** |
| Extraction tables from JSON | **DONE** |
| Cross-session compare V100 | **DONE** (S1–S2 stable; S1–Day2 B1 spread high) |
| Cross-session compare A100 | **DONE** with `--allow-dirty` (stable); dirty provenance noted |
| Fork decision (B3 CUDA vs PT) | **DONE** — PT wins wall-clock **pre_fix**; Welch/d recorded in B3 report |
| Tracker DICC rows honest language | **DONE** (subject to Phase 1 claim hygiene) |
| README / claim hygiene (S1c) | **DONE** — Table A/B split; Option B B3; stale files labeled |
| Full multi-compiler DICC (eager/compile/ORT/TRT) | **DONE** both GPUs |
| Production-weight B3 CUDA–PT parity (local) | **DONE** (`block3_parity_gate.json` valid=true) |
| Corrected server B3 rebench **or drop claim** | **DONE — Option B drop** (rebench remains optional later, not blocking pre-ms) |
| Full sanitizer suite (local) | **DONE** sm_86 four tools × FP32+FP16 |
| Figures / tables from artifacts | **DONE** — `FIGURE_STATUS.md`, `TABLES_FROM_ARTIFACTS.md` |
| Writer handoff | **DONE** — `CHERAN_MANUSCRIPT_HANDOFF.md` |
| Venue manuscript / BibTeX / journal class | **NEXT PHASE** (Cheran) — not a pre-ms gate |

**Pre-manuscript evidence freeze: CLOSED.**  
**Venue submission: not yet — that is manuscript writing.**

---

## 2. Fork decision (locked)

### Decision: **Option A honest — no portable B3 CUDA win**

| Finding | Action |
|---------|--------|
| On V100S/A100, **PT B3 faster** than CUDA B3 FP16 (~1.41× / ~1.72×) **pre_fix** wall-clock | **Do not claim** portable CUDA B3 beats matching PT; label **pre_fix** |
| B1/B2/B4 CUDA ≫ PT | **Claim** per-block CUDA wins (Option A) |
| Full CUDA vs full V3 / partial pipeline vs full-model frameworks | **Forbidden** (no cross-table ratios) |
| Multi-session B3 means stable | **Claim** measurement stability for B3 **pre_fix** only until rebench |
| DICC multi-compiler absolute | **Claim** with protocol + CentOS7 constraints |
| Laptop multi-compiler ranges | **Claim** as **laptop only** — absolute full-model table separate from incomplete Custom CUDA ranges |
| Detection multi-obj package | **Primary** scientific contribution alongside systems measurement |

---

## 3. Stretch menu — final disposition

| ID | Item | Final status |
|----|------|--------------|
| S1a | torch.compile DICC | **DONE** (superseded by full matrix) |
| **Full multi-compiler** | eager / compile / ORT / TRT on V100S+A100 | **DONE** (395433 / 395417) |
| S1b | Clean A100 re-run | **DEFERRED** (optional provenance; not blocking) |
| S1c | README / claim hygiene | **IN PROGRESS** (review Phase 1) |
| S1d | B3 CUDA vs PT stats report | **DONE** (**pre_fix** wall-clock) |
| S2a | B3 kernel optim | **NOT DONE** (optional science delay) |
| S2b | ORT-GPU on DICC | **DONE** (in full matrix) |
| S2c | Nsight V100 vs A100 | **NOT DONE** (optional) |
| S2d | Energy on DICC | **NOT DONE** (optional) |
| S3a | TensorRT on DICC | **DONE** (native TRT 8.6 + ORT TRT EP) |
| S3b | Option B full CUDA = V3 | **NOT DONE** (out of Option A freeze) |
| S3c | Retrain to beat RF F1 | **NOT DONE** (out of freeze) |

---

## 4. Contribution spine (pre-write)

1. **Multi-objective CAD-CBA package** under sealed protocol (detection competitive, not pure-F1 king).  
2. **Measurement-first multi-GPU study**: three sessions, two GPU classes, JSON-backed, formal compares.  
3. **Option A systems**: CUDA wins B1/B2/B4; **PT wins B3 on servers** (**pre_fix** wall-clock; local production-weight parity closed; DICC latency rebench open).  
4. **Multi-compiler**: laptop **full-model absolute ranges** **and** DICC absolute matrix (eager/compile/ORT/TRT) — separate from incomplete Custom CUDA block-sum ranges.  
5. **LLM dispatch** micro-result; no full free-form XAI title claim.  

Defensible for **systems / FGCS-leaning** venues; not for “we beat cuDNN BiLSTM everywhere.”

**Principal detection number for prose:** sealed test **0.9780 ± 0.0033** (not historical 0.9790).  
**ToN:** corrected leakage-safe **CNN 0.8075 / RF 0.9626** (`toniot_leakage_safe_v1`); clean 0.9526/0.9851 **FORBIDDEN**; older 13-feat ~0.811 vs ~0.939 optional comparable only.

---

## 5. Next phase (CUDA evidence + publication synchronization)

Per `COLIDE_Remediation_Update_Review.md` (Phases 2–8), remaining work includes:

| # | Action |
|---|--------|
| 1 | ~~Production-weight B3 parity gate~~ **DONE** local (`block3_parity_gate.json` valid=true) |
| 2 | ~~Full sanitizer suite~~ **DONE** local sm_86; optional provenance archive polish |
| 3 | Corrected V100S/A100 B3 **post_fix latency** rebench **or** drop comparative server B3 claim |
| 4 | Clean ToN provenance rerun (categorical missing-value order) if retained |
| 5 | Full-model framework numerical validation where claimed (native TRT engine **not** built in framework gate — do not claim TRT numerical equivalence) |
| 6 | Manuscript + figures synchronized to corrected claim surfaces |
| 7 | PI venue class file / BibTeX when journal chosen |

Optional only if PI prioritizes: S1b, S2c Nsight, S2a B3 optim.

---

## 6. Integrity snapshot (2026-08-12 historical freeze)

```text
SUCCESS markers: 6 (S1/S2/Day2 × v100s+a100)
multi_compiler_v100s.json + multi_compiler_a100.json: errors {}
champion md5: 80a90f7cc210276300eaa90173a5a385
```

*Data remediation closed. CUDA evidence and publication synchronization pending. Do not treat this pack as submission-ready closure.*
