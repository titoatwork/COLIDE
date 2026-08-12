# COLIDE — Session Handoff

**MODE:** ✅ **PRE-MANUSCRIPT CLOSED** · stretch + **full DICC multi-compiler** complete · **next = manuscript multi-GPU writing**  
**Date:** 2026-08-12  
**Authority:** `docs/PRE_MANUSCRIPT_CLOSURE.md` · Option A · JSON only · champion frozen  
**Champion:** md5 **`80a90f7cc210276300eaa90173a5a385`**  
**Continuity:** `docs/execution_plan/SESSION_CONTINUITY.md`

---

## Delivered (this arc)

| Item | Status |
|------|--------|
| 6 DICC SUCCESS (S1/S2/Day2 × V100S+A100) | **DONE** |
| Option A B3 fork (PT wins on servers) | **DONE** · `DICC_B3_CUDA_VS_PT_REPORT.md` |
| S1c/S1d claim hygiene + B3 report | **DONE** |
| **Full multi-compiler on DICC** | **DONE** · V100S job **395433** · A100 job **395417** |
| S1b clean A100 re-run | **DEFERRED** optional |

### Multi-compiler means (µs) — authoritative

| Method | V100S | A100 |
|--------|------:|-----:|
| Eager | 1041 | 932 |
| torch.compile | 865 | 770 |
| ORT CUDA | 895 | 865 |
| ORT CPU | 500 | 461 |
| ORT TRT EP | 766 | 2033 |
| **TRT native FP16** | **528** | **588** |

Pack: `docs/DICC_MULTI_COMPILER_MATRIX.md` · JSON under `benchmarks/results/dicc/framework/multi_compiler_*.json`

---

## Locked science (do not re-argue without new JSON)

- **Option A B3:** PT faster than CUDA FP16 on servers; B1/B2/B4 CUDA wins; no full CUDA vs full V3.  
- **DICC multi-compiler complete** (eager/compile/ORT/TRT). Native TRT fastest GPU framework path here.  
- Laptop multi-compiler ranges remain separate; do not mix environments.

### Cluster env pitfalls (if re-running)

- GPU nodes = **CentOS 7 / glibc 2.17** → ORT **1.16.3** only; not modern manylinux_2_27.  
- TRT **8.6** libs + cuDNN8 in `~/colide/third_party/trt8_libs`; torch uses cuDNN9 — preload TRT path **after** eager/compile.  
- Never force-reinstall random nvidia-* wheels from default PyPI (breaks glibc / torch).

---

## Next phase

1. **Write manuscript multi-GPU + multi-compiler section** from:
   - `docs/DICC_EXTRACTION_TABLES.md`
   - `docs/DICC_B3_CUDA_VS_PT_REPORT.md`
   - `docs/DICC_MULTI_COMPILER_MATRIX.md`
   - `docs/DICC_COMPARE_OUTCOMES.md`
2. Optional: S1b clean A100, S2c Nsight, S2a B3 optim.  
3. PI venue class file / BibTeX when journal chosen.

---

## Paste-ready prompt for next chat

```text
Resume COLIDE — manuscript multi-GPU + multi-compiler writing phase.

Context (do not redo):
- Pre-manuscript CLOSED. Option A; champion md5 80a90f7cc210276300eaa90173a5a385 — never clobber.
- 6 SUCCESS campaign runs under benchmarks/results/dicc/core/.
- B3 on DICC: PT wins (~363 vs ~513 V100S; ~385–391 vs ~667–671 A100). docs/DICC_B3_CUDA_VS_PT_REPORT.md
- Full multi-compiler DONE both GPUs (docs/DICC_MULTI_COMPILER_MATRIX.md):
  V100S: eager 1041, compile 865, ORT-CUDA 895, ORT-CPU 500, ORT-TRT 766, TRT-native 528
  A100:  eager 932,  compile 770, ORT-CUDA 865, ORT-CPU 461, ORT-TRT 2033, TRT-native 588
- Read: HANDOFF.md → SESSION_CONTINUITY.md → PRE_MANUSCRIPT_CLOSURE.md → DICC_EXTRACTION_TABLES.md → DICC_MULTI_COMPILER_MATRIX.md → DICC_B3_CUDA_VS_PT_REPORT.md → WP9b_MANUSCRIPT_SPINE.md → docs/manuscript/.

Your job:
1) Write manuscript multi-GPU / multi-compiler systems section from JSON only (no invent; no full CUDA vs V3; no portable B3 CUDA win; don't mix laptop ratios with DICC absolutes).
2) Keep README/docs consistent.
3) Commit on master when a coherent writing unit lands.

SSH: host `dicc`. Micromamba env `colide`. VPN if DNS fails.
```

---

## Quick ops

```bash
ls benchmarks/results/dicc/framework/multi_compiler_*.json
bash scripts/rsync_dicc_results.sh   # optional full tree
```

*End handoff.*
