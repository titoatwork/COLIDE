# Deep multi-agent audit of  
# `COLIDE Remediation and Limited-Scope Improvement Checklist.md`

**Source checklist:** repo root (1453 lines, ~713 checkboxes)  
**Audited tree:** local `master` ≈ `2608c71` (aligned with GitHub)  
**Date:** 2026-08-14  
**Method:** 4 parallel codebase explore agents (ToN, CUDA B3, claims/docs, loss/KD/deps) + direct greps  

**Legend for each item group**

| Verdict | Meaning |
|---------|---------|
| **CONFIRMED** | Issue is real in code/docs; checklist is right |
| **PARTIAL** | Partly true, partly already mitigated, or split by path |
| **ALREADY FIXED (policy)** | Claim map / MS / harness already forbids or corrects the *claim* |
| **NOT DONE** | Remediation action still open (no code/docs fix) |
| **OUT OF SCOPE** | Correctly listed as do-not-do / optional |

---

## Executive summary

The checklist is **high-quality and largely correct**. Multi-agent verification found:

1. **P0 scientific integrity issues are real**, especially:
   - ToN-IoT **clean** path **label leakage** (`label` in features while predicting `type`)
   - Optimized CUDA Block-3 **hidden-state race** (naive only is double-buffered)
   - Bidirectional **reverse output alignment** mismatch vs PyTorch
   - Block-3 **contract mismatch** (`output[:, -1, :]` vs CUDA recurrence finals)
   - Active README still selling **invalid ToN clean numbers** and **0.9790** as principal

2. **Your later honesty work (claim map, PRE_MANUSCRIPT, DICC multi-compiler, Option A) is real** but **incomplete as remediation**: it fixed *policy* more than *code/quarantine/README*.

3. **DICC B3 “PT wins” timings** remain valid as **wall-clock of current binaries**, but checklist is right that they are **not verified matching-op speedups** until race + alignment + real-weight parity.

4. **Do not execute all 713 boxes.** Follow checklist Phase 1–5 order; many items are P2 or DROP.

**Highest-leverage next actions (if acting on checklist):**

| Priority | Action |
|----------|--------|
| P0 | Quarantine ToN clean 0.9526 / 0.9851 / +15.4% from README + hardcodes |
| P0 | Principal BoT → sealed **0.9780±0.0033** in README (not 0.9790) |
| P0 | Label streaming as bulk throughput; drop stream-arrival claims |
| P0 | CUDA B3: double-buffer + reverse align + real-weight parity before any new B3 claim |
| P0 | Title: drop full “LLM-Based Explainability” branding or mark prototype |
| P1 | Energy exploratory reframe; fix CPU-path NVML labeling |
| P1 | ISSUE_REGISTER + KNOWN_LIMITATIONS + stale-claim sweep |

---

# Section-by-section audit

## § Scope + priority labels

| Item | Verdict | Notes |
|------|---------|-------|
| Scope limited to fix/not redesign | **Sound** | Matches Option A + pre-ms freeze |
| P0/P1/P2/DROP labels | **Sound** | Use them; do not treat all boxes equal |
| Snapshot commit `2608c71` | **OK** | Matches current GitHub tip |

---

## §1 Recommended minimum-change strategy

| Checklist decision | Verdict vs repo | Action |
|--------------------|-----------------|--------|
| Keep sealed BoT champion | **ALREADY FIXED (policy)** + practice | Do not retrain; md5 frozen |
| Principal result `0.9780±0.0033` | **PARTIAL** | Claim map + MS yes; **README still 0.9790** |
| Withdraw ToN “clean”; simple corrected run | **CONFIRMED needed** | Clean path still active; no quarantine |
| Fix CUDA B3, don’t new kernels | **CONFIRMED needed** | Race+align still open on optimized kernels |
| Don’t implement full V3 CUDA | **ALREADY FIXED (policy)** | Claim map forbids full pipeline claim |
| Rename streaming → bulk | **CONFIRMED needed** | Still “streaming” in README + script |
| Energy exploratory | **PARTIAL** | MS better; README still strong claims |
| LLM = dispatch only | **PARTIAL** | Numbers OK; title still overclaims |

---

## §2 Create one remediation authority

### 2.1 Issue register

| Item | Verdict |
|------|---------|
| Create `docs/ISSUE_REGISTER.md` | **NOT DONE** |
| IDs DATA-TON-001 … LLM-001 | **NOT DONE** (IDs are good; file missing) |
| Link from README / claim map | **NOT DONE** |

### 2.2 Stale-claim search

| Value/phrase | Still present? | Verdict |
|--------------|----------------|---------|
| `0.9526` | README, scripts, status docs | **CONFIRMED** |
| `0.9851` | README, docs | **CONFIRMED** |
| `15.4%` | README | **CONFIRMED** |
| `0.9790` as principal | README, many docs | **CONFIRMED** |
| `25,899` streaming | README, status docs | **CONFIRMED** |
| `16.60` | Many places (mostly OK if dispatch-scoped) | **PARTIAL** |
| `1.089` energy | README | **CONFIRMED** |
| Fail-on-stale script | Absent | **NOT DONE** |

---

## §3 Immediately quarantine

### 3.1 ToN-IoT — **CRITICAL CONFIRMED**

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Clean loader keeps `label` in X, predicts `type` | **CONFIRMED** | `scripts/train_toniot_clean.py`: preserves `label` with target, never strips from features |
| Encoders fit before split | **CONFIRMED** | Full-DF `LabelEncoder` then split |
| Ordinary SMOTE on encoded cats | **CONFIRMED** | `SMOTE` after int encoding |
| 0.9526 / 0.9851 / +15.4% still active | **CONFIRMED** | README table + prose |
| Quarantine metadata on JSON | **NOT DONE** | No `valid:false` |
| Hardcoded in `run_toniot_final_method.py` | **CONFIRMED** | historical_clean_* fields |
| 13-feat path excludes label | **PARTIAL mitigation** | `preprocess_toniot.py` allowlist OK; not “clean” path |

**Important nuance:** 13-feature `toniot_final` (~0.811 vs RF ~0.939) is a **different, more honest** path. Checklist is right that **clean** results must not stay headline evidence.

### 3.2 CUDA benchmark quarantine

| Claim | Verdict |
|-------|---------|
| Optimized B3 pre_fix metadata | **NOT DONE** |
| Speed claims provisional until parity | **CONFIRMED needed** |
| Full pipeline non_parity | **PARTIAL** | harness + claim map OK; README still markets pipeline ranges vs full frameworks |

### 3.3 Streaming / energy / LLM quarantine

| Claim | Verdict |
|-------|---------|
| Streaming rename | **CONFIRMED needed** |
| Remove offered-load/saturation claims | **CONFIRMED needed** (script computes interval, never paces) |
| Energy exploratory | **CONFIRMED needed in README** |
| 16.60 dispatch-only | **PARTIAL** (numbers OK; title bad) |
| Drop counts next to LLM | **PARTIAL** (script prints; README/MS tables weak) |

---

## §4 Correct ToN-IoT experiment

| Subsection | Verdict | Notes |
|------------|---------|-------|
| 4.1 Leakage-safe schema | **CONFIRMED needed** | Clean path fails; 13-feat allowlist exists but incomplete as full fix |
| 4.2 Split before preprocess | **CONFIRMED broken** (encoders) | Across ToN loaders |
| 4.3 Train-only fit | **PARTIAL** | Scaler often train-only; encoders not |
| 4.4 Remove SMOTE on cats | **CONFIRMED still present** | |
| 4.5 Simple RF+hard CNN no KD | **NOT DONE** as corrected protocol | Final method is KD package |
| 4.6 Full reporting | **PARTIAL** for 13-feat only | |
| 4.7 Completion gate | **FAIL** overall | |

**Agent conclusion:** Checklist ToN diagnosis is **accurate**. Remediation largely **not started**.

---

## §5 Correct CUDA Block 3 — **CRITICAL CONFIRMED**

| Subsection | Verdict | Evidence |
|------------|---------|----------|
| 5.1 Hidden-state race (optimized) | **CONFIRMED** | `fused_block3.cu` / `_fp16.cu`: write `s_h_prev[h]` then `__syncthreads()` |
| Naive double-buffer | **ALREADY FIXED** | `fused_block3_naive.cu` only |
| 5.2 Reverse store at `t` not `pos` | **CONFIRMED** | `output_hidden[..., t]` while input at `pos` |
| 5.3 Contract mismatch | **CONFIRMED** | PT `x[:, -1, :]` vs CUDA extract last **recurrence** index for reverse |
| 5.4 Weight mapping doc | **NOT DONE** | |
| 5.5 Sanitizers on optimized | **NOT DONE** (docs claim racecheck for naive only) | |
| 5.6 Determinism / remove retries | **CONFIRMED issue** | `benchmark_cuda_kernels_stats.py` retries FAILED |
| 5.7 Completion gate | **NOT MET** | |

### Impact on DICC B3 numbers (must not miss)

| Use | OK? |
|-----|-----|
| Wall-clock of binaries on V100/A100 | **Yes** (with pre_fix label) |
| “Matching PyTorch Block-3” speedup | **No** until parity |
| Paper “PT faster than our B3 CUDA” as engineering observation | **Yes if labeled provisional / pre-fix** |
| Use as closed Option A correctness | **No** |

---

## §6 Real-weight numerical parity

| Subsection | Verdict |
|------------|---------|
| 6.1 Validators overclaim CUDA without running CUDA | **CONFIRMED** (`validate_real_weights.py` language) |
| `numerical_fidelity` careful split | **PARTIAL** (good comments) |
| 6.2 Export via `named_parameters` misses BN stats | **CONFIRMED** for `export_weights()`; `weights_bin` path has BN |
| 6.3 Direct PT↔CUDA per block | **NOT DONE** |
| 6.4 Hybrid CUDA B3 → PT attention suffix | **NOT DONE** |
| 6.5 Parity JSON gate | **NOT DONE** |

---

## §7 Rerun necessary CUDA benchmarks

| Item | Verdict |
|------|---------|
| Clean-tree claim-eligible results | **PARTIAL** | A100 campaign dirty; claim map notes allow-dirty |
| Minimal B3-only re-run after fix | **Sound plan** | Not done yet |
| De-emphasize Welch/d from mismatched harnesses | **CONFIRMED good advice** | We added Welch to B3 report; checklist is right to treat as secondary |

---

## §8 Custom-pipeline framing

| Item | Verdict |
|------|---------|
| Forbid full CUDA vs full V3 | **ALREADY FIXED (claim map / PRE_MS / harness)** |
| README still sells pipeline vs full-model frameworks | **CONFIRMED residual** |
| Rename `fused_pipeline` | **NOT DONE** |

---

## §9 Full-model framework output validation

| Item | Verdict |
|------|---------|
| Logit/class agreement across eager/compile/ORT/TRT | **NOT DONE** |
| EP-enabled TRT labeling | **PARTIAL** (docs say EP; good) |
| Batch-1 scope explicit | **PARTIAL** (multi-compiler docs OK) |

**Note:** DICC multi-compiler matrix is **latency-only**; checklist wants numerical parity too — fair residual risk.

---

## §10 BoT-IoT framing

| Item | Verdict |
|------|---------|
| Principal sealed 0.9780±0.0033 | **FIXED in MS/claim map; NOT in README** |
| Historical 0.9790 labeled | **PARTIAL** |
| Split-qualified baselines | **FIXED in MS; weaker in README** |
| No pure-F1 SOTA | **FIXED** (wording denies SOTA) |
| Stage-A SMOTE-before-scale document | **Sound; don’t silently change** |

---

## §11 Loss and KD utilities

| Item | Verdict |
|------|---------|
| Weighted focal `pt=exp(-weighted CE)` | **CONFIRMED** in `scripts/protocol/losses.py` |
| Sealed path default unweighted focal | **PARTIAL** (bug path less likely for champion) |
| Historical KD teacher-only T | **CONFIRMED** (`train_protocol_kd.py`) |
| Some scripts have correct T/T² | **PARTIAL** (neural teacher / toniot final) |
| Retrain champion for loss cleanup | **OUT OF SCOPE** (checklist correctly says don’t) |

---

## §12 Config precedence

| Item | Verdict |
|------|---------|
| HPO overwrites CLI despite docs | **CONFIRMED** | `train_protocol_ft.py` “always apply full HPO” |

---

## §13 Checkpoint consistency

| Item | Verdict |
|------|---------|
| Many benches still default `best_model.pth` | **CONFIRMED** | streaming, energy, ort, pipeline, blocks, llm, … |
| Champion known in FT/DICC paths | **PARTIAL** |

---

## §14 Streaming

| Item | Verdict |
|------|---------|
| Not a real streaming simulation | **CONFIRMED** | interval unused |
| Rename to bulk throughput | **CONFIRMED needed** |
| True streaming rebuild | **OUT OF SCOPE / DROP** | checklist recommends rename only |

---

## §15 Energy

| Item | Verdict |
|------|---------|
| CPU energy from GPU NVML | **CONFIRMED** | `benchmark_energy.py` |
| Cross cuML vs CNN as controlled claim | **CONFIRMED overclaim in README** |
| Low-effort: appendix/exploratory | **Sound** |
| Full multi-GPU energy campaign | **OUT OF SCOPE** |

---

## §16 LLM / XAI

| Item | Verdict |
|------|---------|
| 16.60 is dispatch not generation | **ALREADY FIXED (numbers)** |
| Title LLM-Based Explainability | **CONFIRMED issue (README)** |
| Drop counts beside dispatch | **PARTIAL** |
| Faithfulness wording vs concentration | **PARTIAL residual in MS** |
| No new LLM/RAG/human study | **OUT OF SCOPE** (correct) |

---

## §17 Dependency / environment

| Item | Verdict |
|------|---------|
| requirements incomplete | **CONFIRMED** (pandas, xgb, lgbm, matplotlib, pynvml) |
| Dockerfile sm_86 = A100 comment | **CONFIRMED wrong** (A100 is sm_80) |
| Hardcoded /home paths | **CONFIRMED** | shebangs + SLURM logs |
| env.md placeholders | **CONFIRMED** |

---

## §18 Tests

| Item | Verdict |
|------|---------|
| Required unit tests | **NOT DONE** (empty/missing tests tree) |
| GPU validation script as checklist | **NOT DONE** as unified gate |
| Makefile validate-* | **NOT DONE** |

---

## §19 Result metadata

| Item | Verdict |
|------|---------|
| Standard `valid` / `result_id` schema | **NOT DONE** (thin envelope only) |
| RESULTS_INDEX.md | **NOT DONE** |
| Figures from JSON only | **PARTIAL** |

---

## §20 Documentation / manuscript

| Item | Verdict |
|------|---------|
| README worst active surface | **CONFIRMED** |
| MS better aligned | **CONFIRMED** |
| KNOWN_LIMITATIONS.md | **NOT DONE** |
| Verified Research Gaps wording | **CONFIRMED** README |
| MS still TBD DICC vs PRE_MS closed | **PARTIAL inconsistency** |

---

## §21 Licensing / hygiene

| Item | Verdict |
|------|---------|
| No LICENSE file | **CONFIRMED** |
| Academic-only statement | **CONFIRMED** |
| Full packaging rewrite | **OUT OF SCOPE** |

---

## §22–25 P2 polish, out of scope, phases, gates, stop rules

| Block | Verdict |
|-------|---------|
| §22 P2 improvements | **Optional**; do after P0 |
| §23 Explicitly out of scope | **Sound** — agree, do not expand into second thesis |
| §24 Phase order | **Sound** — Phase 1 stop-claims first is correct |
| §25 Final release gate | **Not met** until P0s done |
| Final stop rules | **Sound** — especially “if B3 still slower after fix, stop optimizing” |

---

# Agreement vs disagreement with the checklist

### Checklist is right (do not soft-pedal)

1. ToN clean leakage path is real and still in README.  
2. Optimized Block-3 race + reverse alignment + contract issues are real.  
3. Streaming experiment is bulk throughput mislabeled.  
4. Energy cross-device claims are weak.  
5. Stale numbers still pollute active surfaces.  
6. Real-weight PT↔CUDA parity is not proven.  

### Checklist is slightly harsher / already partially handled

1. Claim map + PRE_MANUSCRIPT already forbid full-pipeline CUDA vs full V3 and pure F1 SOTA.  
2. DICC multi-compiler + multi-session campaign **did land** after many docs were written; MS “TBD DICC” is stale.  
3. Principal sealed F1 is already correct in MS/claim map — **README is the lagging surface**.  
4. Welch/d on B3 is useful for direction; checklist is right not to oversell p-values.  

### Do not over-react

1. Do **not** retrain sealed champion for loss utility cleanup.  
2. Do **not** rebuild full streaming platform or full V3 CUDA.  
3. Do **not** hide corrected ToN if it is lower.  
4. DICC B3 latency story can stay as **provisional engineering measurement** while correctness is fixed.

---

# Recommended execution (compressed from §24)

| Phase | What | Why |
|-------|------|-----|
| **1 — Stop claims** | README rewrite; quarantine ToN clean; rename streaming; exploratory energy; LLM title; issue register | Protect paper/Cheran handoff **today** |
| **2 — ToN correct** | Allowlist, split-first, no SMOTE, RF+hard CNN | Multi-dataset integrity |
| **3–5 — CUDA B3** | Double-buffer, reverse align, parity, one clean rebench | Systems claims integrity |
| **6–7 — Utils/docs** | Focal/KD labels, checkpoint defaults, deps, LICENSE, limitations | Hygiene |
| **8 — P2 only if time** | Plots, bootstrap, one XAI sanity | Polish |

---

# Bottom line for you + Cheran

| Question | Answer |
|----------|--------|
| Is the checklist trustworthy? | **Yes — treat it as a serious audit, not noise** |
| Is everything already fixed by pre-ms closure? | **No** — policy yes; code + README + quarantine **no** |
| Can you send Cheran materials as-is? | **Yes**, but lead with claim map + PRE_MS + DICC packs and **explicitly flag** ToN clean / B3 pre_fix / streaming as issues |
| Biggest paper risk if ignored? | ToN clean leakage + selling incomplete CUDA pipeline + unverified B3 “matching” |

*End multi-agent deep audit.*
