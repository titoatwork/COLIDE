# COLIDE Remediation Update Review

**Document type:** Living review status (original findings + post-implementation verification)  
**For re-review:** every status below is tied to a path, commit, or command output that was re-checked when this fill was written.  
**No assumed closes.** Items still open are marked **OPEN**. Items closed only on laptop evidence are marked **CLOSED (local)** — not the same as DICC multi-session campaign evidence.

---

## 0. Verification basis (re-checked for this fill)

| Field | Value |
|-------|--------|
| **Repo tip when filled** | `12e8aa1` (`results: close review gates — B3 parity, sanitizers, clean ToN, framework`) |
| **Working-tree source_dirty** | `False` via `scripts.protocol.result_schema.git_dirty` (results/logs/kernel binaries excluded from source dirty) |
| **Champion path** | `model/best_model_botiot_twostage.pth` |
| **Champion MD5** | `80a90f7cc210276300eaa90173a5a385` — `scripts/verify_champion.py` → **MATCH** |
| **Principal BoT claim** | Sealed multi-seed test macro-F1 **0.9780 ± 0.0033** (unchanged; **no retrain**) |
| **pytest** | **28 passed** (`tests/`) |
| **Stale-claim guard** | `scripts/check_stale_claims.py` → **OK** on configured active surfaces |
| **Original review commits** | Pre-remediation `2608c71` · remediation `2a6de4b` · pin `cf3e0f5` · checklist log `a796957` · review file `24ac44f` · implementation `2d7acf8`…`12e8aa1` |
| **Releases** | `remediation-2026-08-14` · `review-gates-2026-08-14` |

### Artifacts used as primary evidence

| Artifact | Path |
|----------|------|
| B3 parity gate | `benchmarks/results/block3_parity_gate.json` |
| Weight bins for inject | `benchmarks/results/block3_parity_weights/` |
| Sanitizer suite | `benchmarks/results/sanitizer_b3/` (+ `summary.json`) |
| Framework logit gate | `benchmarks/results/framework_parity_gate.json` |
| ToN corrected | `benchmarks/results/toniot_corrected/summary.json`, `table.md`, `seed42.json` |
| ToN tombstones | `benchmarks/results/toniot_clean_comparison.json` (+ `.INVALIDATED`) |
| Issue / status docs | `docs/ISSUE_REGISTER.md`, `docs/REMEDIATION_STATUS.md`, `docs/RESULTS_INDEX.md`, `docs/PRE_MANUSCRIPT_CLOSURE.md` |
| Claim map / README | `docs/CLAIM_MAP_PREWRITE.md`, `README.md` |

### How this fill was produced

1. Re-read the original review requirements (sections below).  
2. Re-loaded JSON gates and re-ran tests / champion verify / stale-claim script.  
3. Re-ran `./inference/kernels/fused_block3` (exit **0**, full-seq self-check PASS).  
4. Grepped sources for `CUDA_CHECK`, weight inject, `series_to_cat`, Table A/B wording.  
5. Assigned each original requirement **CLOSED / CLOSED (local) / PARTIAL / OPEN** with evidence — not from memory.

---

# Overall verdict (updated after implementation)

The original review was correct: remediation after `2a6de4b` was **genuine and substantial**, but **not submission-ready** because production-weight B3 parity, full sanitizers, clean ToN provenance, and claim-surface consistency were incomplete.

**After implementation tip `12e8aa1`, the local verification backlog named in that review is largely closed with machine-readable evidence.** That is a material change in readiness for **local correctness**, not a free pass to submission.

| Area | Original review (on ~`cf3e0f5`) | Status at tip `12e8aa1` (verified) |
|------|--------------------------------|-------------------------------------|
| ToN leakage removal | Ready | **CLOSED** |
| ToN clean provenance + cat fix + per-class | Mostly ready / needs rerun | **CLOSED** (clean `source_dirty=false`, `use_in_manuscript=true`) |
| B3 race + reverse source repair | Implemented | **CLOSED** (source) |
| B3 production-weight CUDA↔PT full-seq | **Not established** | **CLOSED (local)** — inject parity PASS |
| Full sanitizer suite | Racecheck only / incomplete | **CLOSED (local sm_86)** — 4 tools × FP32+FP16, 0 errors |
| Independent fw/rev self-check weights | Required | **CLOSED** |
| Nonzero exit on validation fail | Required | **CLOSED** (self-check exit 0 when pass; inject missing dir exits 1) |
| Claim Table A vs B separation | Incomplete | **CLOSED** on README + claim map |
| Framework backend logit parity | Required | **CLOSED (local)** for backends present; TRT native **skipped** |
| DICC multi-session post_fix B3 latency | Not performed | **OPEN** |
| Manuscript + figures regenerated | Stale | **OPEN** |
| Overall submission | Not ready | **Still not fully submission-ready** (manuscript + optional DICC B3 rebench/drop) |

**Honest one-line verdict for the next reviewer:**  
Core validity and local CUDA/ToN gates are now evidence-backed; **do not treat the paper as done** until manuscript/figures match these artifacts and server B3 latency is either rebench’d post_fix or dropped.

---

# A. Original “successfully completed” items — still true?

## A1. ToN-IoT target leakage removed — **CLOSED** (still true; strengthened)

**Protocol:** `toniot_leakage_safe_v1` · `scripts/protocol/toniot_leakage_safe.py`

Still present and re-verified in corrected summary:

- Explicit 13-feature allowlist  
- Blacklist `label` / `type` / `attack` / `category` (+ variants)  
- Fatal asserts target not in `X`  
- Stratified 60/20/20 **before** encoders/scaler; train-only fit  
- No SMOTE, no KD  
- RF `class_weight=balanced` + hard-label CNN  
- Split/feature hashes, predictions, checkpoint  

### Post-review strengthening (Phase 6) — **CLOSED**

| Requirement | Evidence |
|-------------|----------|
| Categorical missing: `fillna` **before** `astype(str)` | `series_to_cat()` lines ~116–123 in `toniot_leakage_safe.py`; unit tests in `tests/test_toniot_blacklist.py` |
| Numeric missing documented as fixed zero | `preprocessing.numeric_missing = "fixed_zero_imputation"` |
| Clean provenance | `summary.json`: `source_dirty=false`, `git_sha=fd08f36…`, `command` present, `checkpoint_sha256` present, `use_in_manuscript=true` |
| Per-class honesty | `table.md` + classification_report; **mitm** test F1 **0.1114**, precision **0.0593**, recall **0.9087**, support **208** |

### Corrected numbers (single seed 42, re-run after cat fix)

| Model | Val macro-F1 | Test macro-F1 |
|-------|-------------:|--------------:|
| RF | 0.962645… | **0.962648…** (report **0.9626**) |
| CNN | 0.806599… | **0.807523…** (report **0.8075**) |

**Limitation (must remain visible):** stratified random split, **not** official temporal/host split (`split.note` in JSON). Single seed only.

**Invalid forever:** clean path CNN **0.9526** / RF **0.9851** / **+15.4%** — tombstoned JSON only.

---

## A2. CUDA Block-3 source repairs — **CLOSED** (source)

Verified in `inference/kernels/fused_block3.cu` and `_fp16.cu`:

- Double-buffer `read_buf` / `write_buf` (`s_h[2]`)  
- Reverse store at original `pos`  
- CPU ref aligned to same contract  

---

## A3. BoT framing — **CLOSED**

- Principal: **0.9780 ± 0.0033**  
- Historical **0.9790** development-only  
- Champion MD5 match; no retrain  

---

## A4. Peripheral claim narrowing — **CLOSED** (active surfaces)

- Streaming → bulk batched throughput  
- Energy → exploratory  
- LLM → dispatch **16.60 µs p99**  

---

## A5. Engineering hygiene — **CLOSED** / improved further

- Dual focal + KD disclosure, HPO precedence, champion paths, LICENSE, tests (**28** now)  
- `source_dirty` semantics refined so results/logs/kernel **binaries** do not block claim gates after clean source commits (documented in `result_schema.py`)  

---

# B. Original “remaining publication blockers” — item-by-item fill

## B1. Direct real-weight CUDA–PyTorch parity — **CLOSED (local)**

Original review required inject + full-sequence + hybrid suffix + machine-readable gate.

### Measured evidence (`block3_parity_gate.json`)

| Field | Value |
|-------|--------|
| `valid` | **true** |
| `use_in_manuscript` | **true** |
| `source_dirty` | **false** |
| `kernel_status` | **post_fix** |
| `status` | `pt_cpu_ref_ok_cuda_inject_ok` |
| `champion_md5_ok` | **true** |
| `executable_sha256` | `62508b915a1e9ba3a798d0a6b039b6617094710e1b2bca8bc1897660dd95fe75` |
| `git_sha` (gate run) | `fd08f36925762978c2ca73b63c477e95a9fbc86f` |
| Geometry | B=4, SEQ=16, IN_CH=128, H1=128, H2=64 → full `[B,16,128]` |
| Modules | `bilstm1`, `bilstm2` from champion |

| Comparison | max_abs_error | Pass? |
|------------|--------------:|-------|
| PT vs CPU-ref **full sequence** | 6.49e-06 | yes (tol 1e-4) |
| PT vs CPU-ref last | 1.33e-06 | yes |
| **GPU inject vs PT full sequence** | **3.43e-06** | **yes** |
| GPU inject vs PT last | 1.24e-06 | yes |
| Hybrid suffix PT B3→attn/LN/pool/head logits | 9.54e-06 | yes; class agree **4/4** |
| Hybrid **CUDA** B3→suffix vs PT logits | 5.72e-06 | yes; class agree **4/4** |
| NaN counts | 0 | |

**How inject works (implemented, not assumed):**  
`scripts/export_block3_weights.py` writes float32 bins → `inference/kernels/fused_block3 <dir>` (`WEIGHT_INJECT_MODE` / `COLIDE_B3_WEIGHTS`) → `out_full.bin` / `out_last.bin` compared to PyTorch.

### What this does **not** claim

- Not multi-seed / multi-shape stress (single fixed seed=42, B=4, SEQ=16).  
- Not FP16 production-weight inject gate (FP16 self-check is synthetic-weight).  
- Not DICC GPU binary parity.  
- Not full end-to-end custom CUDA pipeline vs V3 (Option A; CLAIM-PIPE-001 still forbids that).  

---

## B2. Full-sequence Block-3 contract as active executable contract — **CLOSED (local harness + binary)**

| Location | Behavior |
|----------|----------|
| Parity harness | Full sequence primary; last-timestep auxiliary |
| `fused_block3.cu` | `cpu_pipeline_full` + pack `[SEQ, 2*H2]`; self-check compares full seq; last labeled **legacy_last_state** |
| Hybrid path | Feeds **sequence** into V3 attention/LN/pool/head |

**Residual:** some older benchmark scripts may still emphasize last-timestep; new claim-eligible gate is full-sequence.

---

## B3. Complete CUDA sanitizer + determinism — **CLOSED (local sanitizers); determinism PARTIAL**

### Sanitizers — **CLOSED (local sm_86, RTX 3050)**

Artifact: `benchmarks/results/sanitizer_b3/`  
Session meta timestamp: `20260814T054822Z`  
GPU: NVIDIA GeForce RTX 3050 6GB Laptop · arch **sm_86** · nvcc **12.6**

| Binary | racecheck | synccheck | initcheck | memcheck |
|--------|-----------|-----------|-----------|----------|
| `fused_block3` | **0 hazards** | **0 errors** | **0 errors** | **0 errors** |
| `fused_block3_fp16` | **0 hazards** | **0 errors** | **0 errors** | **0 errors** |

Self-check under suite: FP32 full-seq max abs ~1.18e-6 PASS; FP16 full-seq max abs ~6.22e-3 PASS (looser FP16 tol).

### Determinism — **PARTIAL**

- Repeated parity runs produced consistent max-abs order of magnitude.  
- Built-in self-check exit 0 when PASS.  
- **Not done:** formal multi-hundred bitwise-identical campaign, removal of every historical statistical retry in older harnesses.  

**Do not claim “fully deterministic under all loads” from this fill.**

---

## B4. Corrected V100S/A100 Block-3 benchmarks — **OPEN**

No new DICC SUCCESS tree for post_fix B3 latency was produced in this work.

**Defensible options (unchanged from original review):**

| Option | Action |
|--------|--------|
| **A** | One clean post_fix session per retained server GPU after compile for sm_70/sm_80, with parity gate recorded |
| **B** | Drop comparative post_fix B3 latency from main paper; keep historical pre_fix as historical only |

**Current honest labeling:** DICC B3 means remain **pre_fix wall-clock of historical binaries**. Local laptop self-check latencies exist but are **not** a multi-session claim-eligible campaign and vary with load/sanitizer overhead.

---

## B5. Active claim surfaces consistency — **CLOSED on primary surfaces; residual OPEN on manuscript**

### Done

- README **Table A** (Custom CUDA Blocks 1–4 sum, **absolute only**) vs **Table B** (full-model frameworks, absolute) — **no cross-table speedups** (wording + structure re-checked).  
- `docs/CLAIM_MAP_PREWRITE.md` FORBIDDEN partial-vs-full ratios.  
- `docs/PRE_MANUSCRIPT_CLOSURE.md` status: **DATA REMEDIATION CLOSED; CUDA EVIDENCE AND PUBLICATION SYNCHRONIZATION PENDING** (not “fully CLOSED for submission”).  
- `docs/RESULTS_INDEX.md` maps claims → artifacts.  

### Residual / was fixed during this fill

- README abstract previously still said “parity remains open” after gate was green — **corrected in this fill** to local parity closed + DICC latency historical.  
- PRE_MANUSCRIPT “not submission-ready” line still listed parity/sanitizers as open — **corrected** to reflect closed local gates and open manuscript/DICC.  

### Still OPEN

- Full manuscript (`docs/manuscript/…`, `paper_text_blocks.md`, figures) not regenerated from corrected JSON.  
- Historical PROF/status emails may still contain old numbers as historical context.  

---

## B6. Project closure status premature? — **PARTIALLY RESOLVED**

Accurate status now:

> **DATA REMEDIATION CLOSED; LOCAL CUDA CORRECTNESS GATES CLOSED; SERVER B3 LATENCY + PUBLICATION SYNC PENDING**

Not: “everything closed for camera-ready.”

---

## B7. Manuscript and figures stale — **OPEN**

Must still use for final prose:

- BoT **0.9780 ± 0.0033**  
- ToN RF **0.9626** / CNN **0.8075** + mitm weakness  
- No invalid clean ToN  
- B3: local parity closed; DICC latency pre_fix or rebench  
- Separate operator vs full-model tables  
- Bulk / exploratory energy / dispatch wording  
- Pseudo-sequence architecture wording  

**Figures:** not regenerated in this implementation pass.

---

# C. Original “additional issues” — fill

| # | Issue | Status | Evidence |
|---|--------|--------|----------|
| C1 | Independent fw/rev self-check weights | **CLOSED** | Seeds 42/43/44/45 separate fills in `.cu` (no `w_ih1_r=w_ih1_f`) |
| C2 | Validator compares complete sequence | **CLOSED** | Full-seq primary in binary + harness |
| C3 | Nonzero exit on validation failure | **CLOSED** | `return all_pass ? 0 : 1`; inject fail → exit 1; live self-check EXIT_FP32=0 |
| C4 | Explicit post-launch CUDA error checking | **CLOSED** | `CUDA_CHECK` / `CUDA_CHECK_LAST` throughout launch path |
| C5 | Full-sequence contract in executable tests | **CLOSED (local)** | Parity + inject + hybrid; not every legacy script rewritten |

---

# D. Original “small ToN corrections” — fill

| # | Item | Status |
|---|------|--------|
| D1 | Categorical fillna order | **CLOSED** — `series_to_cat` |
| D2 | Accurate numeric imputation docs | **CLOSED** — fixed zero |
| D3 | Clean committed-source rerun | **CLOSED** — `use_in_manuscript=true`, `source_dirty=false` |
| D4 | Show weak class (mitm) | **CLOSED** — table + metrics; **do not retune on observed test** |

Three-seed ToN: still **not required** under limited scope if paper labels single-seed secondary evaluation.

---

# E. Full-model framework parity — **CLOSED (local, non-TRT)**

Artifact: `benchmarks/results/framework_parity_gate.json`  
`valid=true`, `source_dirty=false`, champion MD5 ok.

| Backend | Numerical pass | Notes |
|---------|----------------|-------|
| pytorch_eager_cuda | yes | Cross-device vs eager CPU ref; larger abs error band (~3.6e-2) |
| onnxruntime_cpu | yes | max abs ~7.6e-6 |
| onnxruntime_cuda | yes | ~3.4e-2 cross-device |
| torch_compile | yes | No BiLSTM crash on this path |
| tensorrt_native | **skipped** | No engine present; **not invented** |

**Honest:** framework gate is **logit agreement on a fixed batch**, not a full multi-compiler latency rebench and not a TRT-native claim.

---

# F. Checklist / status hygiene — **IMPROVED**

Original review said some checklist rollups overstated completion. After implementation:

- Prefer **gate JSON `valid` fields** and this document over checkbox optimism.  
- `docs/RESULTS_INDEX.md` is the claim→artifact map for manuscript drafting.  
- External review file remains authority for residual OPEN items.  

---

# G. Minimum remaining work (honest, for next re-review)

Ordered for a **limited-scope publication**:

### Must before camera-ready

1. **Regenerate manuscript tables/figures** only from OK artifacts in `docs/RESULTS_INDEX.md`.  
2. **Decide B3 server latency:** rebench post_fix on V100S+A100 **or** drop comparative B3 speed claims and keep historical pre_fix labeled.  
3. **Final repo-wide stale-claim sweep** including manuscript files (guard currently covers a fixed active-file list, not every historical email).  

### Should if time

4. FP16 **real-weight** inject parity (today FP16 is synthetic self-check + sanitizer only).  
5. Broader shape/batch determinism campaign.  
6. Native TensorRT engine build + parity if TRT remains in a numerical table.  

### Explicitly still DROP / out of scope (agree with original review)

- Three-seed ToN campaign (unless you choose otherwise)  
- New KD / architecture / dataset / streaming platform / RAG LLM  
- Full custom CUDA V3 (attention/LN/etc.)  

---

# H. Readiness assessment (re-filled)

| Area | Status at tip `12e8aa1` |
|------|-------------------------|
| Sealed BoT-IoT result | **Ready** |
| Historical BoT framing | **Ready** |
| ToN leakage removal | **Ready** |
| Corrected ToN methodology + clean provenance + per-class | **Ready** (single-seed random split limitation remains) |
| CUDA B3 race source repair | **Ready** |
| CUDA reverse-alignment source repair | **Ready** |
| CUDA production-weight equivalence (local inject + hybrid) | **Ready (local)** |
| Complete CUDA sanitizer gate (local) | **Ready (local sm_86)** |
| Deterministic repeated execution (formal campaign) | **Partial** |
| Corrected V100S/A100 Block-3 **latency** benchmark | **Not performed** |
| Partial-vs-full comparison cleanup (README/claim map) | **Ready on primary surfaces** |
| Full-framework numerical equivalence (present backends) | **Ready (local); TRT skipped** |
| README / claim-map consistency | **Ready** (stale abstract line fixed this fill) |
| Manuscript and figures | **Stale / OPEN** |
| Release provenance | **Partial → improved** (tags `remediation-2026-08-14`, `review-gates-2026-08-14`) |
| **Overall publication status** | **Not fully submission-ready** — correctness gates advanced; **publication synchronization + server B3 decision remain** |

---

# I. Commands for the next reviewer (reproduce, do not trust prose alone)

```bash
cd /path/to/colide
git rev-parse HEAD   # expect 12e8aa1 or descendant with same gates
.venv/bin/python scripts/verify_champion.py
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/check_stale_claims.py
PYTHONPATH=. .venv/bin/python scripts/parity_block3_cuda_pt.py
PYTHONPATH=. .venv/bin/python scripts/parity_framework_backends.py
# Inspect:
python -c "import json;print(json.load(open('benchmarks/results/block3_parity_gate.json'))['valid'],
 json.load(open('benchmarks/results/toniot_corrected/summary.json'))['use_in_manuscript'],
 json.load(open('benchmarks/results/framework_parity_gate.json'))['valid'])"
# Sanitizer logs: benchmarks/results/sanitizer_b3/
# Optional live self-check:
./inference/kernels/fused_block3; echo exit:$?
```

---

# J. Final conclusion (for re-review)

**What is real and closed locally**

1. ToN leakage path removed; invalid clean numbers quarantined.  
2. Corrected ToN secondary evaluation with clean provenance and candid mitm weakness.  
3. BoT principal result and champion identity preserved.  
4. B3 race + reverse alignment in source.  
5. **Production-weight full-sequence CUDA↔PT parity + hybrid suffix** with `valid=true`.  
6. **Full local sanitizer suite** 0 errors FP32+FP16.  
7. Claim tables separated; peripheral claims narrowed.  
8. Framework logit parity for available backends.  

**What remains for a defensible submission**

1. Manuscript/figure regeneration from corrected artifacts only.  
2. Explicit decision on DICC B3 **latency** (rebench post_fix or drop comparative claim).  
3. Optional FP16 real-weight inject and broader determinism.  
4. Final stale-claim pass on manuscript surfaces.  

The project has moved from **“major validity problems”** to **“local correctness gates closed with artifacts; publication and server-latency decisions still open.”**  

It is **appropriate for a second technical review of evidence quality**.  
It is **not appropriate to claim camera-ready** solely from this document without manuscript sync.

---

## Appendix — original review snapshot (historical)

The first version of this file (committed as `24ac44f`) assessed tip around `cf3e0f5` / pre-implementation and correctly flagged missing production-weight parity, incomplete sanitizers, dirty ToN provenance, and claim-surface issues. **That assessment was accurate at that time.** This filled version supersedes the overall verdict and section statuses while preserving the original requirement structure for auditability.

*End of filled review status.*
