# COLIDE — Design & Analysis Plan

**Status:** **APPROVED 2026-07-14** (user: full plan approval)  
**Strategy freeze:** **Option A** — scope claims to valid per-block contracts; no full-pipeline CUDA vs full V3 PyTorch until parity WP (stretch)  
**Date:** 2026-07-14  
**Sources:** `HANDOFF.md` (Paused checkpoint), `CLAUDE.md` / `AGENTS.md`, `README.md` (abstract + limitations), `docs/paper_text_blocks.md`, kernel/model sources, result JSONs, `dicc_scripts/`.

**Production anchor (do not silently change):**
| Item | Value |
|------|--------|
| Branch | `master` @ laptop/GitHub (tip at plan write may include this doc) |
| Checkpoint | `model/best_model_botiot_twostage.pth` |
| md5 | `80a90f7cc210276300eaa90173a5a385` |
| Macro-F1 | **0.9790** (BoT-IoT two-stage) |
| Published RF bar | **0.9864** (`rf_baseline_processed.json`) — not 0.9885 |
| Official multi-day site | **UM DICC** (Rostam = tooling trial only) |

---

## 1. Purpose of this plan

COLIDE is in a dangerous-but-recoverable state for a Q1 systems paper:

- Much of the **local scientific hygiene work is solid** (provenance, ranges, KD, threats-to-validity text, kernel tooling).
- Several **headline latency narratives are still construct-invalid or provisionally threatened** (full-pipeline CUDA vs full V3 PyTorch; possible Block 3 loss to cuDNN on server GPUs).
- **Paper-official multi-day cluster evidence on the current stack does not exist yet** at UM DICC.

This document answers, before any more execution:

1. What is solid enough to keep and cite carefully?  
2. What is weak (needs re-measurement, reframing, or disclosure)?  
3. What is **invalid** if published as currently implied?  
4. What **must** re-run on UM DICC (and what must not be claimed even after)?  
5. What is the manuscript path and risk register?  
6. What is the ordered sequence of work packages after approval?

---

## 2. Inventory of work done so far

### 2.1 Scientific / model track

| Area | Outcome | Artifact anchors |
|------|---------|------------------|
| BoT-IoT preprocessing | Per-flow, no windowing (v2); windowed pipeline abandoned | `preprocessing/preprocess_v2.py`, `DAILY_LOG` history |
| Production model | Two-stage: KD (α=0.6, T=10, γ=2) + real-data FT → **0.9790** | `twostage_botiot.json`, `best_model_botiot_twostage.pth` |
| KD sweep | 14 configs; γ∈{1,3,4} negative vs γ=2; Stage-1 winner bit-repro | `distill_botiot_*.json`, repro JSON |
| RF baseline (apples-to-apples) | **0.9864** on same `data/processed/*.npy` | `rf_baseline_processed.py/json` |
| RF strengthen | balanced → 0.9885; **published bar kept 0.9864** | `rf_teacher_strengthen.json` |
| Ensemble teacher | Diagnostic Val-F1 bug fixed; still not champion | Session 5 |
| Balanced-RF KD (S6) | **Skipped** (low EV) | HANDOFF |
| MLP ablation | ~0.95 F1, faster; supports “systems stress case” narrative | README MLP table |
| ToN-IoT clean | **0.9526** macro-F1 (26-feat); RF still higher | `toniot_clean_*.json` |
| Weight export vs 0.9790 | Re-exported; real-weight validation pass | `weights_bin/`, `real_weight_validation.json` |
| Numerical fidelity (S7) | Export path bit-identical n=10; 6 CUDA self-checks PASS | `numerical_fidelity.json`, paper §15–§16 |

### 2.2 Systems / CUDA track

| Area | Outcome |
|------|---------|
| Blocks 1–4 kernels | Standalone fused CUDA (FP32 + B3 FP16 half2 + naive ablation) |
| Naive B3 race | Real shared-memory race fixed (double-buffer); racecheck clean |
| Optimization progression | Documented steps; latency **ranges** across sessions (7.55×–9.50×) |
| Framework comparison (local) | Ranges over multi-session WSL2 noise; Welch two-sample design |
| Measurement stability | Session-to-session drift quantified; ranges required on laptop |
| `fused_pipeline.cu` | Times **B1+B2+B4 only**; B3 added **additively** from separate binary; hardcoded baseline leftovers remain in comments/prints |
| LLM dispatch | Real p99 **16.60 µs** (5k trials); generation async ~7–8 s |
| torch.compile finding | CUDA-graph crash on BiLSTM documented |

### 2.3 Provenance / process track

| Area | Outcome |
|------|---------|
| `verify_claims.py` | Manifest + regression guards (fabricated numbers, lucky point ratios, bad citation phrasing) |
| Sophimatics false citation | Removed; replaced with Ibrahim et al. (Computer Networks 2026) with honest GNN/CPU caveats |
| One-chat / slow-down rule | User-confirmed after multi-session rush |
| Branch unify | `final-polish` → `master` |

### 2.4 Cluster track

| Area | Outcome |
|------|---------|
| Portable campaign stack | `run_campaign.sh`, setup, submit, n=100 CUDA stats, PyTorch same-GPU harness, compare gate |
| Rostam Day 1 | **Tooling trial SUCCESS** (not paper-official); provisional means recorded |
| Historical UM DICC (2026-06-21) | Single-shot summaries only; no multi-day; no same-GPU PyTorch |
| UM path discovery | Clone at `/home/user/ibteshamulhaque/colide`; git diverged; **fetch freezes** India→UM→GitHub |
| Official multi-day on hardened stack @ UM | **Not started** |

---

## 3. What is solid

These pieces can underpin the paper **if claims stay within their proven scope**.

### 3.1 Accuracy and training narrative (BoT-IoT)

- Champion **0.9790** is reproducible in method (seeded Stage-1 bit-identical repro; Stage-2 uses same determinism flags though not re-proved bit-identical).
- RF gap story is honest and sourced: **0.9864 − 0.9790 = 0.74%**, not fabricated.
- KD grid + focal-γ negative result + outlier at (0.7, T=10) are good experimental science (include negative cells).
- MLP ablation correctly demotes “BiLSTM needed for tabular accuracy” while elevating systems motivation.
- SMOTE/Theft minority-class limitation is already drafted honestly.

**Manuscript use:** Detection section + KD appendix/table + RF positioning (§12 narrative). Do not claim SOTA accuracy over RF.

### 3.2 Per-block CUDA engineering (especially Block 3)

- Blocks 1–4 are real, compiled, validated at disclosed tolerances.
- Block 3 optimization path (naive → transpose → graphs → FP16 half2) is a coherent HPC contribution.
- Race fix on naive kernel is a strength (methods + integrity), not a liability if disclosed as resolved.
- Local Block 3 vs cuDNN-style PyTorch block harness exists (`pytorch_block3_stats_rtx3050.json`); Block 3 is the **intended valid head-to-head**.

**Manuscript use:** Core systems section should **lead with Block 3 methodology + per-block results**, not with a single full-pipeline ×-vs-TensorRT headline.

### 3.3 Measurement hygiene infrastructure

- Multi-session ranges instead of lucky points.
- Two-sample Welch tests vs real CUDA distributions.
- ORT CPU significance correctly marked **non-robust**.
- `verify_claims.py` + regression guards reduce reintroduction of known-bad strings.
- Result isolation for campaigns (`manifest`, `environment.txt`, `kernel_SHA256SUMS`, `SUCCESS`).

**Manuscript use:** Methods (experimental protocol) + threats-to-validity. Keep running verify after every README/paper edit.

### 3.4 LLM systems integration (dispatch path)

- Async ring buffer + non-blocking detection path is real design.
- **16.60 µs p99** is measured, not placeholder.
- Generation quality is **illustrative only** (already disclosed in §15) — keep it that way unless a user study is planned (not recommended pre-submission).

### 3.5 Campaign tooling

- Portable SLURM entry is production-ready enough for UM once tree is synced.
- Comparability flags in `benchmark_pytorch_gpu_stats.py` correctly encode full-pipeline invalidity.
- Rostam Day 1 proves the stack can complete jobs end-to-end (even if numbers are not paper-final and site differs).

### 3.6 Literature / citation integrity

- Fabricated Sophimatics claim removed and guarded.
- Ibrahim et al. is a real, differentiated closest-prior-work candidate (GNN + CPU baseline vs recurrent + production frameworks).

---

## 4. What is weak (salvageable with work or careful wording)

### 4.1 Local (WSL2 RTX 3050) as the primary latency venue

- Session-to-session drift (roughly 6–27% on some Block 3 configs; 14–17% on torch.compile/TensorRT in back-to-back sittings) means **all laptop ratios are ranges**, not pin-point product claims.
- High CVs on some blocks (WSL2 scheduling) require n≥50–100 and multi-session protocol.
- Abstract currently leads with laptop framework ranges; cluster multi-day stability is still missing for V100/A100.

**Mitigation:** UM multi-day campaign; manuscript should present laptop results as **dev-box characterized ranges** and cluster as **replicated absolute latencies + valid per-block ratios**.

### 4.2 “Pipeline total” methodology (additive reconstruction)

Even ignoring architecture parity (next section):

- `fused_pipeline.cu` does **not** measure a true B1→B2→B3→B4 device chain with real B3 outputs feeding B4.
- Headline “594–675 µs Custom CUDA FP16” is a **derived / additive** quantity (B1+B2+B4 chained or separate + B3 separate).
- Comments still mix live measurement with **hardcoded** B3 / PyTorch baseline figures inside the binary printout — fine for historical debug, dangerous if treated as live authority.

**Mitigation (later WP):** either (a) implement true chained timing including B3, or (b) permanently brand totals as “sum-of-block means” in every table footnote. Prefer (b) for submission speed; (a) if time allows after DICC.

### 4.3 Framework comparison fairness surface

Partially disclosed (TensorRT build notes, no INT8, Python API, batch=1 only), but still weak under review:

| Issue | Why weak |
|-------|----------|
| Batch size = 1 only | TensorRT / torch.compile often look better at larger batches; paper is about edge stream, but must own that scope |
| TensorRT Python path | vs C++ enqueue may differ; already partially disclosed |
| ORT CPU straddles parity | Cannot claim “beats all frameworks” without qualification — abstract already hedges somewhat |
| torch.compile configs | Graph-mode crash vs reduce-overhead without graphs must stay unconflated |

**Mitigation:** Explicit “deployment regime: batch-1, single-stream, sub-1M model” subsection; do not claim general compiler inferiority.

### 4.4 Accuracy still second to RF; energy second to cuML RF

- RF higher accuracy; cuML RF higher throughput/energy efficiency on A100 table.
- Neural model’s defense is **VRAM + LLM latent path + adaptability narrative** — soft under a hostile accuracy-only reviewer.

**Mitigation:** Lead contribution list with systems findings; keep RF defense paragraph; never hide 0.9864.

### 4.5 Pseudo-sequence / construct of “sequence model on tabular flows”

- Scientifically honest in README/limitations; still invites “why not just RF/MLP?”  
- BiLSTM is justified as **compiler stress case** — that justification must be front-and-center, not buried.

### 4.6 Cross-hardware story partially anecdotal

- “V100S outperforms A100 for sequential LSTM” from **legacy single-shot** June 2026 DICC + provisional Rostam Day 1.
- Needs multi-day confirmation on UM with same SHA/binaries and PyTorch same-GPU block metrics.
- Rostam peek already complicates the story (see §5.3).

### 4.7 Closest-prior-work PDF

- Ibrahim et al. metadata corroborated; **primary PDF not fully read** (ScienceDirect blocks).  
**Mitigation:** one manual PDF read before submission (low effort, high integrity).

### 4.8 Stage-2 training bit-repro not empirically shown

Low priority if no more FT runs; if any retune, re-prove or avoid claiming “bit-identical fine-tune.”

### 4.9 ToN-IoT secondary

Useful external validation; less systems instrumentation (no full CUDA feature-dim path for 26-feat clean model). Keep as accuracy transfer table, not second CUDA campaign, unless time surplus.

---

## 5. What is invalid (do not publish as currently implied)

### 5.1 Full-pipeline Custom CUDA vs full PyTorch V3 speedup — **INVALID**

This is the central construct-validity defect. Multiple independent gaps stack:

| Gap | Detail |
|-----|--------|
| **Missing V3 ops on CUDA** | Production model (`cnn_bilstm_v3_attention.py`) runs **MultiheadAttention + residual LayerNorm + global average pool** after BiLSTM. **No CUDA kernel implements these.** |
| **Reduce mismatch for dense head** | CUDA Block 3 / export path use **last BiLSTM timestep**. V3 full forward uses **mean over sequence after attention**. Numerical fidelity explicitly notes: *“V3 (attention, using V2 last-timestep for CUDA)”*. |
| **Pipeline binary skips real B3** | `fused_pipeline.cu` chains B1+B2+B4 and injects/uses B3 path only as **separate timed addend**, not as true dataflow into B4 during the timed chain. |
| **Framework table conflation** | README claims chained custom CUDA and eager PyTorch full forward are “the same computation.” They are **not**: eager path is full V3; custom path is incomplete + last-timestep contract. |
| **Harness already admits this** | `benchmark_pytorch_gpu_stats.py` sets `comparability.full_pipeline_cuda_vs_pytorch.valid = false` and documents the same gaps. |

**Consequence for manuscript:**

- **Do not** lead the abstract with “3.04×–3.78× over eager PyTorch” as a *full-model apples-to-apples* claim until parity is fixed **or** the claim is rewritten as:
  - sum-of-CUDA-blocks vs sum-of-matched PyTorch blocks, and/or
  - Block-3-only head-to-head, and/or
  - full V3 PyTorch absolute latency reported **without** a CUDA full-model ratio.
- TensorRT / torch.compile / ORT comparisons that time **full exported/compiled V3** against **incomplete custom CUDA pipeline totals** inherit the same problem for *end-to-end model* claims. Framework-vs-framework comparisons among TRT/ORT/eager remain valid among themselves; custom CUDA must be scoped to **implemented blocks**.

**Valid today:**

- Per-block CUDA vs per-block PyTorch modules that match the CUDA contract (Block 3 last-timestep BiLSTM is the flagship).
- Intra-CUDA optimization progression (naive → FP16) on the same kernel contract.
- Absolute full-model V3 latency (PyTorch / TRT / ORT) as “framework tax on the *production classifier*,” separate from custom kernels.

### 5.2 Cross-hardware “vs PyTorch” from legacy DICC — **INVALID** (already mostly fixed)

- June 2026 summaries: CUDA-only, single job, no multi-day, no same-GPU PyTorch.
- Historically, ratios reused ~1864 µs RTX PyTorch baseline on V100/A100 — fixed in docs; **must never return**.
- README footnote correctly says n/a until accepted Day1/Day2 compare artifacts; keep that discipline.

### 5.3 Treating Rostam Day 1 as paper-final — **INVALID**

- Wrong official site (UM required for acknowledgments/resources narrative).
- No Day 2 + `compare_dicc_sessions.py` accept gate.
- Provisional peek is scientifically **concerning**, not just incomplete:

| GPU (Rostam Day 1, approx.) | CUDA B3 FP16 | PyTorch B3 | Direction |
|----------------------------|--------------|------------|-----------|
| V100 | ~581 µs | ~512 µs | Custom **slower** |
| A100 | ~706 µs | ~353 µs | Custom **much slower** |

If UM replicates this, the paper **cannot** claim “FP16 BiLSTM beats cuDNN” as a portable hardware fact — only as a **local RTX 3050 / specific protocol** finding, or the kernel must improve, or the comparison protocol must be shown inequitable (e.g. cuDNN vs hand-rolled last-timestep module differences). **This is a top scientific risk**, not a documentation nit.

### 5.4 “Beats all major frameworks” without ORT CPU caveat — **INVALID / overclaim**

- ORT CPU ratio range straddles parity; significance flips across sessions.
- Abstract already partially hedges; title-level / contribution-level wording must never drop the hedge.

### 5.5 Lucky single-session point ratios as primary claims — **INVALID** (mostly fixed)

- Old 4.40× / 3.33× / 2.63× style points superseded by ranges; regression guards exist. Maintain forever.

### 5.6 Accuracy claims that erase RF — **INVALID**

- Any framing of CNN-BiLSTM as accuracy-superior on BoT-IoT hard labels is false.
- Publishing balanced-RF 0.9885 as “the” baseline without updating all gap math is also invalid; decision was to keep **0.9864**.

### 5.7 LLM “first” / quality claims beyond dispatch measurement — **WEAK→INVALID if overstated**

- Dispatch overhead is solid.
- “First fully on-device…” style claims need careful scoping vs Jamshidi et al.; generation quality is not evaluated at scale.

---

## 6. What must be re-run on UM DICC

### 6.1 Must re-run (paper-blocking for cross-hardware + stability story)

Official site: **`login01.dicc.um.edu.my`**, user `ibteshamulhaque`, intended tree after sync under home (path currently known as `.../colide`; do **not** assume `/scr`).

| Item | Why |
|------|-----|
| **Code sync of current `master`** | DICC clone paused at old SHA without `run_campaign.sh`; fetch freezes → **laptop tarball/scp** method |
| **Day 1 campaign** | `bash dicc_scripts/run_campaign.sh` (or day 1 default) on V100 + A100 profiles |
| **Day 2 campaign** | Same git SHA, same kernel `SHA256SUMS`, new date label (`--day 2` / `_d2`) |
| **Cross-day compare** | `scripts/compare_dicc_sessions.py` must **accept** before any README/paper table update |
| **CUDA n=100 strict stats** | Per-block means/std on bare metal (or site-native stack), not WSL2 |
| **PyTorch same-GPU stats** | `benchmark_pytorch_gpu_stats.py`: full V3 absolute + **Block 3** matched contract |
| **Artifact scp home** | Results dirs typically gitignored; copy `benchmarks/results/dicc/...` off cluster |

**Publishable outputs after accept:**

- Absolute per-block CUDA latencies (V100, A100) with multi-day stability statement.
- Absolute full V3 PyTorch latencies (no invalid CUDA/full ratio).
- **Valid** CUDA-vs-PyTorch **Block 3** ratios only (and B1/B2/B4 if contracts match).
- Qualitative confirmation or refutation of “V100 vs A100 sequential” story with error bars / day compare.

### 6.2 Must **not** claim from DICC even after success

- Full-pipeline CUDA vs full V3 PyTorch speedup (until parity WP done).
- Laptop framework ranges “confirmed on DICC” unless the **same framework suite** is also run on DICC (currently campaign focuses CUDA + PyTorch GPU stats, not full TRT/ORT matrix).
- Rostam numbers as UM numbers.

### 6.3 Optional on DICC (nice-to-have, not first)

| Item | Priority |
|------|----------|
| Full TensorRT / ORT / torch.compile matrix on V100/A100 | Medium — strengthens portability of framework-tax story; large setup cost |
| Energy re-measure | Low–medium |
| Nsight deep dive | Low for v1 submission if Block 3 story clear |
| Training on DICC | **Out of scope** (accuracy campaign already local) |

### 6.4 Pre-DICC operational prerequisites (not science)

1. Park diverged DICC clone (`colide-old-diverged-*` / existing backup branch).  
2. Deliver tarball of clean `master` via scp (no `git fetch` on DICC).  
3. Site env: partitions, modules/CUDA on GPU nodes, torch **cu121** if V100 present, GRES policy.  
4. Dry-run / `--local` only if a GPU node interactive is available; else Day 1 is the smoke test.  
5. Confirm production checkpoint md5 on cluster matches laptop.

---

## 7. Manuscript path

### 7.1 Target and contribution spine

- **Venue:** FGCS-class systems / cyber-physical (README); IoT-J stretch.  
- **PI:** Prof. Dr. Por Lip Yee (FCSIT, UM); DICC acknowledgment after official runs.  
- **Contribution type:** Systems/performance engineering + measurement integrity — **not** model novelty.

**Recommended narrative order (revise golden arc §13):**

1. **Edge batch-1 inference tax** for sub-1M recurrent models (frameworks optimized for large models/batches).  
2. **torch.compile CUDA-graph failure** on BiLSTM as concrete compiler limitation.  
3. **Hand-written CUDA Block 3** optimization progression + valid per-block comparisons.  
4. **Measurement validity discipline** (ranges, multi-day cluster, comparability rules) as first-class contribution — this differentiates from papers with single-run speedups.  
5. **Async on-device LLM dispatch** as systems integration (not NLP novelty).  
6. **KD closes most of RF gap** while enabling GPU+LLM pipeline — accuracy as supporting, not lead.

### 7.2 Claim tiers (use in writing)

| Tier | Meaning | Examples |
|------|---------|----------|
| **A — Cite freely** (within scope) | Multi-sourced, scoped correctly | 0.9790 F1; 0.9864 RF; 16.60 µs p99; KD table; naive race fix; fidelity bit-identical export |
| **B — Cite with range / venue** | True but environment-bound | Laptop framework ranges; Block 3 progression ranges on WSL2; energy on named GPUs |
| **C — Cite only after UM Day1+Day2 accept** | Cluster tables | V100/A100 block latencies; multi-day stability; Block 3 CUDA vs PT on DICC |
| **D — Do not cite until fixed** | Construct-invalid | Full-pipeline CUDA vs full V3 PT speedup; “same computation” wording for incomplete CUDA vs V3 |
| **E — Threat / may reverse** | Must not pre-commit abstract | “FP16 beats cuDNN on server GPUs”; “V100 always faster than A100” |

### 7.3 Document map

| Doc | Role |
|-----|------|
| `README.md` | Living results dashboard; update only with verify_claims green |
| `docs/paper_text_blocks.md` | Manuscript prose modules (already has §15–§16) |
| `docs/DESIGN_PLAN.md` | This plan; decision log |
| `HANDOFF.md` | Operational resume order |
| `scripts/verify_claims.py` | Gate on claim edits |
| `dicc_scripts/README.md` | Operator path |

### 7.4 Abstract rewrite direction (when claim edits allowed)

Current abstract leads with multi-framework pipeline speedups that sit on **Tier D** construct issues and **Tier B** laptop noise.

**Safer structure:**

1. Custom CUDA kernels for CNN-BiLSTM blocks; **Block 3** FP16 path with documented optimization ladder.  
2. Statistically characterized latency under multi-session protocol; cluster multi-day (after DICC).  
3. Valid per-block speedups; full V3 framework latencies reported separately.  
4. KD accuracy 0.9790 (0.74% from RF).  
5. Async LLM dispatch 16.60 µs p99.  

Exact numbers after DICC ingest — do not rewrite abstract in the DICC operational chat without a dedicated claims pass.

### 7.5 Sections outline (paper)

1. Introduction (edge paradox + framework tax + measurement crisis)  
2. Background / related work (Ibrahim et al. + IDS DL + compiler/TRT literature)  
3. System design (model, **explicit CUDA vs V3 parity diagram**, LLM async)  
4. CUDA kernel design (Blocks 1–4, FP16 half2, race fix)  
5. Experimental setup (datasets, seeds, hardware, **comparability rules**)  
6. Results — accuracy (BoT-IoT/ToN-IoT, KD, RF, MLP)  
7. Results — per-block latency & optimization progression  
8. Results — framework comparison **scoped** + threats  
9. Results — cross-hardware (post-DICC)  
10. Results — LLM dispatch / aggregation  
11. Discussion (RF, energy, pseudo-sequence, when custom CUDA wins/loses)  
12. Threats to validity (expand §15)  
13. Conclusion  

### 7.6 Figures / tables minimum set

- Architecture diagram with **red “not in CUDA”** box for attention/LN/GAP.  
- Block 3 optimization progression (ranges).  
- Per-block latency table (laptop + DICC).  
- Framework table with measurement-stability note OR replaced by block-matched table.  
- KD sweep table (keep negatives).  
- Accuracy vs RF/MLP.  
- Threats/comparability callout box.

---

## 8. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| R1 | Full-pipeline claims rejected in review | High if unchanged | Critical | Reframe now; parity WP or permanent non-claim |
| R2 | UM Block 3 CUDA loses to PyTorch (Rostam pattern) | Medium–High | Critical | Treat as open science question; improve kernel or scope claim to laptop; show protocol fairness |
| R3 | Multi-day means unstable even on bare metal | Medium | High | Report ranges; adjust “WSL2-specific” language to “environment-dependent” |
| R4 | DICC network / sync friction (India latency, no fetch) | High operationally | Medium | Tarball/scp only; no remote GitHub dependency |
| R5 | V100 cuDNN / torch wheel mismatch | Medium | Medium | cu121 pin already in tooling; verify on first job |
| R6 | A100 FP16 validation flakes | Medium | Medium | Retries + bounded skips already in harness |
| R7 | Accuracy-only reviewer dismisses paper | Medium | High | Lead systems story; honest RF; LLM integration |
| R8 | Citation integrity (Ibrahim PDF unread) | Low | High if wrong | Manual PDF read before submit |
| R9 | Claim drift / reintroduce fixed bugs | Medium | High | Always `verify_claims.py`; one claims chat per edit batch |
| R10 | `train_twostage.py` clobbers champion | Medium if training resumes | Critical | Backup rule; no training in DICC chats |
| R11 | Scope creep (more KD, TRT sweeps, energy) | High historically | Medium | One chat = one WP; this plan’s order |
| R12 | Parity implementation time sink | Medium | Medium | Prefer claim scoping (Option A) over full V3 CUDA port unless R2 forces deeper CUDA work |

---

## 9. Strategic options (decision needed)

### Option A — **Scope claims to valid contracts** (Recommended default)

- Do **not** implement attention/LN/GAP in CUDA before submission unless forced by R2.  
- Publish: per-block CUDA (esp. B3), framework latencies for **full V3** as “what production classifiers pay,” additive block sums only with footnotes.  
- Fastest path to an honest Q1 systems paper.

### Option B — **Architecture parity on CUDA**

- Implement attention + LayerNorm + GAP (and true B3→head dataflow) in CUDA; re-export; re-validate; re-benchmark all venues.  
- Enables true full-model custom CUDA vs frameworks.  
- Large engineering cost; delays DICC story if sequenced wrong.  
**Only choose if** full-model speedup is non-negotiable for the venue story.

### Option C — **Dual-model paper**

- Explicit “V2 inference core” (last-timestep, no attention) for CUDA path + “V3 accuracy model” for F1.  
- Must prove V2 core F1 is still acceptable or report both.  
- Clean scientifically if labeled; risks “two systems” confusion.

**Plan recommendation:** **Option A** for submission path; keep Option B as post-review or stretch WP. Do not mix options silently in tables.

---

## 10. Ordered work packages (after this plan is approved)

Each package ≈ **one chat**. No cluster work inside design or manuscript-only chats.

### WP0 — Plan approval (this document)

- User reviews §5 invalid list, §6 DICC must-list, §9 option A/B/C.  
- Freeze production checkpoint and RF bar.  
- **Exit:** written approval note in HANDOFF (option chosen).

### WP1 — UM DICC code sync (operational, no science claims)

- Tarball/scp `master` → DICC; park old tree.  
- Verify `run_campaign.sh`, md5 of champion checkpoint, CUDA module/path notes in run log.  
- **Exit:** `git log -1` and script presence on DICC; no benchmarks required yet.

### WP2 — UM DICC Day 1 campaign

- `bash dicc_scripts/run_campaign.sh` (or site-adjusted).  
- Confirm `SUCCESS` dirs for V100 and A100 (or document which GPUs exist).  
- **Exit:** artifacts under `benchmarks/results/dicc/...` with manifests; **no README table rewrite yet**.

### WP3 — UM DICC Day 2 + compare

- Same SHA/binaries; `--day 2`.  
- `compare_dicc_sessions.py` accept/reject.  
- scp results home.  
- **Exit:** accept report; if reject, diagnose (do not force paper numbers).

### WP4 — Session 10 ingest (claims-safe)

- Update cross-hardware tables from **accepted** artifacts only.  
- Block 3 CUDA vs PyTorch ratios only for CUDA/PT speedup language.  
- Full V3 PT as absolute latency.  
- Run `verify_claims.py`.  
- If R2 realizes (CUDA loses B3): rewrite contribution language **before** abstract freeze.

### WP5 — Claim hygiene pass (may merge with WP4 if small)

- Remove residual “same computation” full-pipeline wording wherever still present.  
- Align abstract/key contributions with Tier A–C only.  
- Footnote additive pipeline totals.  
- `verify_claims.py` green.

### WP6 — Manuscript spine (Session 11)

- Assemble `paper_text_blocks.md` → full draft structure (§7.5).  
- Insert parity diagram and threats.  
- Manual read of Ibrahim et al. PDF; adjust §14 if needed.  
- No new experiments in this chat.

### WP7 — Optional stretch (only if time / reviewer demand)

| Sub | Trigger |
|-----|---------|
| True fused B1–B4 timing (still without attention) | Reviewer questions additive totals |
| Option B parity CUDA | Need full-model custom speedup |
| Framework matrix on DICC | Portability attacks |
| Batch-size fairness sweep | TRT fairness attacks |
| More KD / teacher | Only if accuracy gap becomes blocking (unlikely for systems venue) |
| Stage-2 bit-repro | If claiming perfect FT reproducibility |

### Explicit non-goals until WP0–WP4 done

- No more local KD sweeps by default.  
- No balanced-RF teacher KD (S6 remains skipped).  
- No training runs that touch `best_model_botiot_twostage.pth`.  
- No Rostam as substitute for UM official campaign.  
- No README “final” cluster numbers from Day 1 alone.

---

## 11. Solid / weak / invalid — one-page scorecard

| Component | Rating | Note |
|-----------|--------|------|
| BoT-IoT champion F1 0.9790 + provenance | **Solid** | Keep |
| RF 0.9864 apples-to-apples | **Solid** | Keep published bar |
| KD sweep + negatives | **Solid** | Keep |
| MLP / pseudo-sequence honesty | **Solid** | Keep front-and-center |
| Per-block CUDA B1–B4 + race fix | **Solid** | Core contribution |
| Local multi-session range discipline | **Solid** (process) / **Weak** (venue for absolute truth) | DICC must complement |
| LLM dispatch 16.60 µs | **Solid** | Generation quality weak |
| Threats-to-validity + fidelity tables | **Solid draft** | Expand with DICC |
| Campaign tooling | **Solid** | Proven on Rostam trial |
| Framework “custom beats all” full-model | **Invalid as stated** | Scope to blocks / fix parity |
| Additive “pipeline total” as e2e | **Weak** | Footnote or true chain |
| Cross-HW ratios (legacy) | **Invalid** | Replace via UM multi-day |
| “B3 beats cuDNN” portable | **Threatened** | Rostam provisional opposite |
| Citation Ibrahim | **Mostly solid** | PDF read pending |
| UM official multi-day data | **Missing** | WP1–WP3 |

---

## 12. Immediate next action after approval

1. User marks **Option A / B / C** and any edits to this plan.  
2. New chat: **WP1 UM DICC code sync only** (tarball/scp; no jobs if user prefers split, or sync+Day1 if user has a dedicated cluster window).  
3. Do not start WP4 claim edits until compare accepts.

---

## 13. Appendix — key file pointers

```
HANDOFF.md                          operational truth + pause state
CLAUDE.md / AGENTS.md               architecture + comparability rules
README.md                           published numbers + limitations
docs/paper_text_blocks.md           manuscript modules §1–§16
docs/DESIGN_PLAN.md                 this file
model/cnn_bilstm_v3_attention.py    production forward (attn+LN+GAP)
inference/kernels/fused_*.cu        CUDA blocks; fused_pipeline skips B3 chain
scripts/benchmark_pytorch_gpu_stats.py   DICC PT harness + comparability JSON
scripts/benchmark_cuda_kernels_stats.py  n=100 CUDA harness
scripts/compare_dicc_sessions.py    multi-day gate
scripts/verify_claims.py            claim regression gate
scripts/numerical_fidelity.py       fidelity table
dicc_scripts/run_campaign.sh        one-command campaign entry
dicc_scripts/README.md              operator guide
benchmarks/results/dicc_*_summary.txt    LEGACY June 2026 single-shot only
```

---

## 14. Change log

| Date | Change |
|------|--------|
| 2026-07-14 | Initial design/analysis plan from paused checkpoint session (no implementation, no DICC jobs). |
| 2026-07-14 | **User approved entire plan** (Option A locked; WP order §10 authorized). |
| 2026-07-14 | **Design/planning phase closed.** WP1–WP3 are **user-manual on UM DICC** (agents have no cluster access). Next agent work = WP4+ after user brings compare artifacts. |
| 2026-07-14 | **Deadline replan:** Prof Por update in ≤3 days needs **all numbers including DICC first**. Operator playbook: `docs/PROF_POR_3DAY.md`. Defer WP5–WP7 until after that update. |
