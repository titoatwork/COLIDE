# B3 server latency — publication decision

**Decision date:** 2026-08-15  
**Scope:** UM DICC multi-session Block-3 (BiLSTM) wall-clock latency for **active manuscript tables**  
**Authority chain:** this file · `docs/CLAIM_MAP_PREWRITE.md` · `docs/ISSUE_REGISTER.md` · `docs/REMEDIATION_STATUS.md` · `docs/RESULTS_INDEX.md`  
**Rule:** Do **not** invent DICC JSON, SUCCESS trees, or post_fix server means. No fabricated numbers.

---

## Decision (default limited-scope): **Option B now**

| Option | Meaning | Status |
|--------|---------|--------|
| **Option B (chosen)** | Do **not** present DICC B3 numbers as **post_fix** or parity-gated performance. Keep historical pre_fix DICC B3 wall-clock as **historical only**. Active comparative “CUDA B3 vs PT B3 on servers (claim-eligible / post_fix)” path is **dropped** until a new SUCCESS tree exists. | **ACTIVE** |
| **Option A (later)** | Rebench fixed kernels on DICC (V100S + A100, multi-session) and only then promote post_fix B3 latency into active tables if claim-eligible JSON is green. | **Deferred** — path documented below; not run |

### What this means for manuscript / README

1. **FORBIDDEN (active):** post_fix DICC B3 speed claims, “parity-gated server B3 latency,” or replacing pre_fix means with invented post_fix µs.
2. **OK (labeled historical only):** existing pre_fix DICC B3 wall-clock (~363 vs ~513 µs V100S; ~385–391 vs ~667–671 µs A100 PT vs CUDA FP16) from the 2026-08 multi-session SUCCESS trees — always with **pre_fix / historical binaries** language.
3. **OK (local correctness):** production-weight CUDA↔PyTorch full-sequence parity closed on laptop (`block3_parity_gate.json` `valid=true`, `kernel_status=post_fix`); local sanitizer suite green. These are **not** a DICC latency rebench.
4. **Active Option A systems claims that remain:** CUDA B1/B2/B4 matched-op wins on DICC (historical wall-clock as already documented); full-model multi-compiler absolute matrix; no full Custom CUDA pipeline vs full V3.

---

## Local evidence that exists (no DICC invention)

| Evidence | Artifact | What it establishes | What it does **not** establish |
|----------|----------|---------------------|--------------------------------|
| Production-weight B3 parity (local) | `benchmarks/results/block3_parity_gate.json` | `valid=true`, `use_in_manuscript=true`, `kernel_status=post_fix`; GPU inject vs PT full-seq; hybrid suffix logits | Multi-session **server** latency under fixed post_fix binaries |
| Sanitizer suite (local sm_86) | `benchmarks/results/sanitizer_b3/summary.json` | racecheck/synccheck/initcheck/memcheck **0 errors** FP32+FP16 | DICC sm_70/sm_80 latency; claim-eligible multi-session means |
| Source fixes | `inference/kernels/fused_block3.cu`, `fused_block3_fp16.cu` | Double-buffer race fix; reverse store at original `pos` | That historical DICC binaries were built from this source |
| Historical DICC B3 wall-clock | `docs/DICC_B3_CUDA_VS_PT_REPORT.md`, `docs/DICC_EXTRACTION_TABLES.md`, SUCCESS under `benchmarks/results/dicc/` | Pre_fix PT-faster-than-CUDA-FP16 direction and session stability | Post_fix performance after kernel fix |

Related issue IDs: **CUDA-B3-001 / 002 / 003** — local correctness **CLOSED**; **server latency rebench OPEN** and **comparative active claim dropped under Option B** (see `docs/ISSUE_REGISTER.md`).

---

## Exact rebench path if Option A is chosen later

Prefer campaign scripts. Full ops: `docs/DICC_OPS_METHOD.md`, `dicc_scripts/README.md`. Commands below are copied from `docs/REMEDIATION_STATUS.md` (minimal path) and the portable campaign entrypoints — **run only on DICC / CUDA nodes; do not invent outputs**.

### 1. Compile fixed kernels (GPU node with `nvcc`)

```bash
# From repo root on a CUDA node (or via compile job)
bash dicc_scripts/compile_on_gpu.sh v100   # or a100 / both
# Direct kernels-only (when nvcc is on PATH):
bash dicc_scripts/01_setup.sh --kernels-only --targets sm_70:v100
bash dicc_scripts/01_setup.sh --kernels-only --targets sm_80:a100
```

Equivalent direct `nvcc` (from `inference/kernels/`):

```bash
nvcc -arch=sm_70 -o fused_block3      fused_block3.cu
nvcc -arch=sm_70 -o fused_block3_fp16 fused_block3_fp16.cu
# A100: -arch=sm_80  ·  laptop RTX 3050: -arch=sm_86
```

### 2. Sanitizers (required before claim-eligible B3)

```bash
compute-sanitizer --tool racecheck  ./fused_block3
compute-sanitizer --tool synccheck  ./fused_block3
compute-sanitizer --tool initcheck  ./fused_block3
# Repeat for fused_block3_fp16; save full logs under benchmarks/results/dicc/
```

### 3. Latency / multi-session campaign

```bash
# Preferred one-command campaign (Day1 → Day2 → compare when site allows)
bash dicc_scripts/run_campaign.sh --full

# Or per-session submit
bash dicc_scripts/submit_session.sh --targets v100
bash dicc_scripts/submit_session.sh --targets a100
```

Benchmark helpers: `dicc_scripts/lib/run_benchmark.sh`, `scripts/benchmark_cuda_kernels_stats.py`, `scripts/benchmark_pytorch_block3_stats.py`.  
Compare: `scripts/compare_dicc_sessions.py` (requires two SUCCESS trees, distinct dates, matching provenance).

Expected layout (from `dicc_scripts/README.md`):

```text
benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/
  manifest.json  environment.txt  kernel_SHA256SUMS
  cuda_kernel_stats.json  pytorch_gpu_stats.json
  raw/  logs/  exit_status  SUCCESS
```

### 4. After SUCCESS lands on laptop

1. Extract means only from real JSON (no invention) → update `docs/DICC_EXTRACTION_TABLES.md` / B3 report if direction or µs change.  
2. Flip claim map: post_fix DICC B3 rows become **OK** only if fields below are green.  
3. Replace README / RESULTS_INDEX “historical pre_fix only” language with post_fix labels tied to the new SUCCESS tree paths.  
4. Do **not** mix laptop sm_86 latencies with V100S/A100 means.

---

## Required fields for claim-eligible **post_fix** B3 JSON

A post_fix server B3 latency artifact is claim-eligible for **active** tables only if it is a real SUCCESS-tree (or envelope) product and satisfies **all** of:

| Field | Required value / meaning |
|-------|---------------------------|
| `valid` | `true` |
| `use_in_manuscript` | `true` |
| `source_dirty` | `false` (clean git tree at measurement) |
| `git_sha` | Non-empty, matches the measured commit (not `unknown` for formal claims) |
| `kernel_status` or equivalent provenance | Explicit **`post_fix`** (fixed race+align binaries; not historical pre_fix trees) |
| `pre_fix_vs_post_fix` (if present) | `post_fix` |
| `timestamp_utc` | Present |
| Binary integrity | `kernel_SHA256SUMS` (or `executable_sha256`) recorded for `fused_block3` / `fused_block3_fp16` used in the run |
| Arch / GPU | sm_70 (V100S) and/or sm_80 (A100) stated; GPU class matches path |
| Protocol | Multi-session SUCCESS markers; same-GPU PT B3 harness present for comparative claims |
| `command` | Reproducible command or job script reference |
| `invalid_reason` | `null` / absent when `valid=true` |

Envelope helpers: `scripts/protocol/result_schema.py` (`valid`, `use_in_manuscript`, `source_dirty`, `git_sha`, `command`, …).

**Not claim-eligible:** laptop self-check sample µs; sanitizer-instrumented timings; historical 2026-08 SUCCESS trees without a new post_fix recompile/rebench; any JSON with `source_dirty: true` or missing SUCCESS.

---

## Active-table guidance (summary)

| Surface | Under Option B (now) |
|---------|----------------------|
| Manuscript active B3 server speed table | **Omit** comparative post_fix B3; optional appendix/historical table only with **pre_fix** label |
| README Cross-Hardware B3 cells | **Historical pre_fix** only; never “post_fix rebench” |
| Local parity / sanitizers | **OK** as correctness evidence |
| B1/B2/B4 DICC | Unchanged (matched-op wall-clock as already documented) |
| Full-model multi-compiler DICC | Unchanged (not Option A block parity) |

When Option A rebench completes with claim-eligible JSON, update this file’s decision row, claim map, RESULTS_INDEX, issue register, and README in one coordinated pass.

---

*End B3 server latency decision. No DICC results fabricated.*
