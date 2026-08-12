# DICC multi-session campaign — results pack + flags

**Status date (UTC):** 2026-08-08  
**Machine (cluster):** `/home/user/ibteshamulhaque/colide`  
**Laptop tree:** `benchmarks/results/dicc/` — **synced 2026-08-12** (~1.9 MB, 6 SUCCESS). Compare outcomes: `docs/DICC_COMPARE_OUTCOMES.md`  
**Champion:** `model/best_model_botiot_twostage.pth` md5 `80a90f7cc210276300eaa90173a5a385`  
**Option A:** per-block Custom CUDA vs matching ops only; full pipeline CUDA vs full V3 **invalid**

---

## 1. Campaign sessions completed (cluster)

| Session | Label | V100S | A100 |
|---------|--------|-------|------|
| **S1** | `20260807` | job **390642** SUCCESS · gpu05 | job **390643** SUCCESS · gpu07 |
| **S2** (same calendar day) | `20260807_s2` | job **390653** SUCCESS · gpu05 | job **390654** SUCCESS · gpu07 |
| **Day2** (calendar label) | `20260808` | job **390781** SUCCESS · gpu05 | job **390782** SUCCESS · gpu06 |

Cluster SUCCESS paths:

```text
benchmarks/results/dicc/core/v100s/20260807_job390642/SUCCESS
benchmarks/results/dicc/core/v100s/20260807_s2_job390653/SUCCESS
benchmarks/results/dicc/core/v100s/20260808_job390781/SUCCESS
benchmarks/results/dicc/core/a100/20260807_job390643/SUCCESS
benchmarks/results/dicc/core/a100/20260807_s2_job390654/SUCCESS
benchmarks/results/dicc/core/a100/20260808_job390782/SUCCESS
```

Each SUCCESS run includes (when complete):  
`manifest.json`, `cuda_kernel_stats.json`, `pytorch_gpu_stats.json`, `kernel_SHA256SUMS`, `exit_status`, `raw/`, logs.

**Laptop:** as of doc write, only `DAY1_LABEL.txt` local — **full tree not yet rsynced**.

---

## 2. Extraction tables (from cluster JSON means — re-verify after rsync)

### 2.1 Block 3 Option A head-to-head (primary systems compare)

| GPU | Session | CUDA B3 FP16 mean (n=100) | PT B3 mean (n=20) | CUDA/PT |
|-----|---------|---------------------------:|------------------:|--------:|
| V100S | S1 | 513.3 µs | 363.5 µs | **1.41× slower CUDA** |
| V100S | S2 | 513.0 µs | 363.6 µs | **1.41×** |
| V100S | Day2 | 513.1 µs | 363.3 µs | **1.41× slower CUDA** |
| A100 | S1 | 668.0 µs | 383.7 µs | **1.74× slower CUDA** |
| A100 | S2 | 667.4 µs | 389.0 µs | **1.72×** |
| A100 | Day2 | 671.2 µs | 390.9 µs | **1.72×** |

**FLAG-1 (CRITICAL):** Matching **Block 3** Custom CUDA FP16 is **slower** than same-GPU PyTorch Block 3 on both V100S and A100. Portable “CUDA B3 beats PT/cuDNN” is **not** supported. Stable across sessions (not a fluke).

### 2.2 Full-model PyTorch absolute (valid claim type)

| GPU | Session | Full V3 PT mean (n=20) |
|-----|---------|------------------------:|
| V100S | S1 / S2 | ~963.5 / ~966.7 µs |
| A100 | S1 / S2 / Day2 | ~945.2 / ~961.8 / ~956.6 µs |

### 2.3 Small blocks (CUDA wins)

Order of magnitude (S1-class means): CUDA B1 ~10–12 µs, B2 ~35 µs, B4 ~10 µs vs PT B1 ~165–170 µs, B2 ~155–167 µs, B4 ~84–96 µs.

**FLAG-2:** B1/B2/B4 Custom CUDA remain much faster than PT counterparts — still Option A valid wins.

### 2.4 Full pipeline CUDA vs full V3 PT

Harness: `comparability.full_pipeline_cuda_vs_pytorch.valid = false`.  
**FLAG-3:** Do **not** publish full-pipeline Custom CUDA vs full V3 speedup.

### 2.5 Laptop multi-compiler matrix (separate story — not DICC campaign)

Documented ranges (RTX 3050 / WSL): Custom pipeline 594–675 µs vs eager / torch.compile / TensorRT / ORT-GPU.  
Scripts: `benchmark_tensorrt_native.py`, `benchmark_torch_compile_native.py`, `benchmark_ort.py`.  
**FLAG-4:** Those ratios are **laptop**; TRT/ORT **not** installed on DICC micromamba; FINAL_PLAN marks full TRT matrix on DICC as **stretch / if reviewers demand**. Scripts hardcode old custom µs and partly old checkpoints — not drop-in for production champion without rework.

---

## 3. How the campaign was unblocked (ops)

| Issue | Fix |
|-------|-----|
| `sbatch --wrap` under `/bin/sh` → `pipefail` fail | Force `bash -c` in `run_campaign.sh` |
| Login Python 3.9 venv broken on GPU (py3.6 + no libpython3.9 / glibc) | **micromamba** env `colide` (py3.10+torch cu121); wrapper at `.venv-cluster/bin/python` |
| `--export=ALL` env poison | Prefer `--export=NONE` + COLIDE_* (submit_session) |
| Kernel compile C++11 on g++ 4.8 GPU nodes | `nvcc -std=c++11 -Xcompiler -std=c++11` on-node |
| OnDemand flaky | SSH + batch; jobs independent of VNC |

---

## 4. Flags / risks for claims and compare

| ID | Severity | Flag |
|----|----------|------|
| **FLAG-1** | **CRITICAL** | B3 CUDA FP16 **slower** than matching PT B3 on V100S & A100 |
| **FLAG-2** | Info | B1/B2/B4 CUDA still much faster than PT |
| **FLAG-3** | Hard rule | Full CUDA vs full V3 PT speedup **invalid** |
| **FLAG-4** | Scope | Multi-compiler (TRT/compile/ORT) wins are **laptop**; not DICC campaign |
| **FLAG-5** | **HIGH** | A100 runs all `git_dirty=true` → formal compare **REJECTED**. V100 runs `git_dirty=false` but `git_sha=unknown` → compare ran. See `DICC_COMPARE_OUTCOMES.md` |
| **FLAG-6** | Medium | Same-day S2 is a second **session**, not a new calendar day; Day2 label `20260808` is the calendar multi-day point |
| **FLAG-7** | Medium | README still markets portable framework beat-all language — must align with DICC B3 honesty before submit |
| **FLAG-8** | Ops | ~~Laptop empty~~ **CLEARED 2026-08-12** — rsync complete |
| **FLAG-9** | Science | Por §8 gate for portable “beats cuDNN” is **failed** on servers; multi-obj + measurement still viable |

---

## 5. Formal compare (`compare_dicc_sessions.py`)

```bash
# After rsync:
PYTHONPATH=. python scripts/compare_dicc_sessions.py \
  --run-a benchmarks/results/dicc/core/v100s/20260807_job390642 \
  --run-b benchmarks/results/dicc/core/v100s/20260808_job390781

PYTHONPATH=. python scripts/compare_dicc_sessions.py \
  --run-a benchmarks/results/dicc/core/a100/20260807_job390643 \
  --run-b benchmarks/results/dicc/core/a100/20260808_job390782
```

Script **requires**: different date labels, matching manifest fields, matching kernel checksums, **not git_dirty**.  
**Expect possible REJECT on FLAG-5** — if so, document “manual multi-session means table from JSON; formal compare rejected dirty provenance” **or** clean re-run.

---

## 6. Sync command (laptop, VPN on)

```bash
bash scripts/rsync_dicc_results.sh
# or:
rsync -avz -e ssh dicc:/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/ \
  /home/titoisalive/colide/benchmarks/results/dicc/
```

Also useful (ops recipe, not paper numbers):

```bash
rsync -avz -e ssh dicc:/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/june_*.out \
  /home/titoisalive/colide/benchmarks/results/dicc/ 2>/dev/null || true
```

---

## 7. Pre-manuscript remaining (after sync)

1. Rsync tree to laptop  
2. Re-extract tables from local JSON (confirm Day2 V100 cells)  
3. Run formal compare; record accept/reject  
4. Flip tracker A3/H7/I1–I5/I11/K7/WP0 with honest B3 language  
5. Claims rebuild only if new DICC claims added  
6. **Not** manuscript prose until you decide claim strategy  

---

## 8. Claim strategy note (for your decision — not auto-applied)

- **Justified:** multi-GPU measured; multi-session stability; full PT absolute; B1/B2/B4 CUDA vs PT; laptop multi-compiler ranges with caveats; multi-obj detection package.  
- **Not justified:** portable CUDA B3 beats matching PT; full-pipeline Custom vs full V3; DICC TRT/compile without new runs.  

*End pack.*
