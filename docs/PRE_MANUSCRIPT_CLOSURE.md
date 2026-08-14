# Pre-manuscript closure pack — **FINAL**

**Date (UTC):** 2026-08-12  
**Status:** ✅ **CLOSED** — evidence + default exceptional stretch complete  
**Scope of this pack:** Everything **before** manuscript prose/PDF/venue formatting  
**Authority:** Option A · JSON only · champion frozen  

| Artifact | Path |
|----------|------|
| Extraction tables | `docs/DICC_EXTRACTION_TABLES.md` |
| B3 report (S1d) | `docs/DICC_B3_CUDA_VS_PT_REPORT.md` |
| Multi-compiler matrix | `docs/DICC_MULTI_COMPILER_MATRIX.md` |
| Compare outcomes | `docs/DICC_COMPARE_OUTCOMES.md` |
| Results + flags | `docs/DICC_RESULTS_AND_FLAGS.md` |
| Claim map (pre-write) | `docs/CLAIM_MAP_PREWRITE.md` |
| Issue register (Phase 1 quarantine) | `docs/ISSUE_REGISTER.md` |
| Known limitations | `docs/KNOWN_LIMITATIONS.md` |
| Session handoff | `HANDOFF.md` |

**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`**

### Phase 1 claim quarantine (2026-08-14) — documentation only

Evidence freeze above is unchanged for BoT/DICC (no BoT retrain, no new DICC). Active claim surfaces were corrected:

| Quarantine | Action |
|------------|--------|
| Principal BoT | Sealed **0.9780 ± 0.0033**; historical **0.9790** development-only |
| ToN clean 0.9526 / 0.9851 / +15.4% | **INVALID** tombstone (DATA-TON-001) |
| Streaming | Reframed as **bulk batched throughput** (~25,899 f/s) |
| Energy | **Exploratory**; WP6b ranges preferred over bare 1.089 vs cuML |
| B3 DICC | Wall-clock honesty retained; **pre_fix** until CUDA-B3-* closed + rebench |
| LLM | Dispatch-only; title narrowed |

Guard: `python scripts/check_stale_claims.py`.

### Phase 2 corrected ToN (complete) + CUDA B3 code status

| Item | Status |
|------|--------|
| Corrected ToN leakage-safe experiment | **DONE** — protocol `toniot_leakage_safe_v1`; test macro-F1 **CNN 0.8075** / **RF 0.9626**; no SMOTE, no KD; `benchmarks/results/toniot_corrected/` (`valid: true`, `use_in_manuscript: true`) |
| Invalid clean path | Remains **tombstoned** (0.9526 / 0.9851 / +15.4% never active) |
| Older 13-feat package path | Optional comparable prior only (~0.811 vs RF ~0.939, `toniot_final/`) |
| CUDA B3 race/alignment/contract fixes | Code fixed in remediation tree — **awaiting rebench** before any post_fix B3 claim; DICC B3 numbers stay **pre_fix** until SUCCESS rebench artifacts exist |

---

## 1. Pre-manuscript gate checklist

| Gate | Status |
|------|--------|
| Local science playlist closed | **DONE** |
| Champion frozen `80a90f7…` | **DONE** |
| DICC S1 + S2 + Day2 SUCCESS × V100S + A100 | **DONE** (6 SUCCESS markers) |
| Results on laptop under `benchmarks/results/dicc/` | **DONE** |
| Force-added SUCCESS + multi-compiler JSON to git | **DONE** |
| Extraction tables from JSON | **DONE** |
| Cross-session compare V100 | **DONE** (S1–S2 stable; S1–Day2 B1 spread high) |
| Cross-session compare A100 | **DONE** with `--allow-dirty` (stable); dirty provenance noted |
| Fork decision (B3 CUDA vs PT) | **DONE** — PT wins; Welch/d recorded in B3 report |
| Tracker DICC rows honest language | **DONE** |
| README / claim hygiene (S1c) | **DONE** |
| Full multi-compiler DICC (eager/compile/ORT/TRT) | **DONE** both GPUs |
| Manuscript multi-GPU prose | **OUT OF SCOPE** (next phase) |
| PI journal class file / BibTeX | **OUT OF SCOPE** (PI after venue) |

**Pre-manuscript status: CLOSED.**  
No further evidence collection is required for the Option A + multi-GPU + multi-compiler systems package.

---

## 2. Fork decision (locked)

### Decision: **Option A honest — no portable B3 CUDA win**

| Finding | Action |
|---------|--------|
| On V100S/A100, **PT B3 faster** than CUDA B3 FP16 (~1.41× / ~1.72×) | **Do not claim** portable CUDA B3 beats matching PT |
| B1/B2/B4 CUDA ≫ PT | **Claim** per-block CUDA wins (Option A) |
| Full CUDA vs full V3 | **Forbidden** |
| Multi-session B3 means stable | **Claim** measurement stability for B3 |
| DICC multi-compiler absolute | **Claim** with protocol + CentOS7 constraints |
| Laptop multi-compiler ranges | **Claim** as **laptop only** — do not mix with DICC means |
| Detection multi-obj package | **Primary** scientific contribution alongside systems measurement |

---

## 3. Stretch menu — final disposition

| ID | Item | Final status |
|----|------|--------------|
| S1a | torch.compile DICC | **DONE** (superseded by full matrix) |
| **Full multi-compiler** | eager / compile / ORT / TRT on V100S+A100 | **DONE** (395433 / 395417) |
| S1b | Clean A100 re-run | **DEFERRED** (optional provenance; not blocking) |
| S1c | README / claim hygiene | **DONE** |
| S1d | B3 CUDA vs PT stats report | **DONE** |
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
3. **Option A systems**: CUDA wins B1/B2/B4; **PT wins B3 on servers**.  
4. **Multi-compiler**: laptop ranges **and** DICC absolute matrix (eager/compile/ORT/TRT).  
5. **LLM dispatch** micro-result; no full free-form XAI title claim.  

Defensible for **systems / FGCS-leaning** venues; not for “we beat cuDNN BiLSTM everywhere.”

**Principal detection number for prose:** sealed test **0.9780 ± 0.0033** (not historical 0.9790).  
**ToN:** corrected leakage-safe **CNN 0.8075 / RF 0.9626** (`toniot_leakage_safe_v1`); clean 0.9526/0.9851 **FORBIDDEN**; older 13-feat ~0.811 vs ~0.939 optional comparable only.

---

## 5. Next phase (explicitly **not** pre-manuscript)

| # | Action |
|---|--------|
| 1 | Manuscript multi-GPU + multi-compiler section from locked tables |
| 2 | PI venue class file / BibTeX when journal chosen |
| 3 | Optional only if PI prioritizes: S1b, S2c Nsight, S2a B3 optim |

---

## 6. Integrity snapshot (2026-08-12)

```text
SUCCESS markers: 6 (S1/S2/Day2 × v100s+a100)
multi_compiler_v100s.json + multi_compiler_a100.json: errors {}
champion md5: 80a90f7cc210276300eaa90173a5a385
```

*End final pre-manuscript pack. Do not reopen evidence gates without new JSON.*
