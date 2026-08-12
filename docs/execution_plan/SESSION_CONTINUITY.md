# Session Continuity / Handoff Pack

**Session closed for continuity:** 2026-08-12  
**Mode this session:** **PRE-MANUSCRIPT FULLY CLOSED** (evidence + multi-compiler + claim map)  
**Git tip at handoff:** see latest commit after handoff push (`git log -1 --oneline`)  
**Machine root:** `/home/titoisalive/colide`  
**Cluster:** `dicc` → `login01.dicc.um.edu.my` · user `ibteshamulhaque` · repo `~/colide`

---

## 1. Mission (current)

**Pre-manuscript is FULLY CLOSED** (campaign + stretch multi-compiler + claim map).  
**Next phase is manuscript writing only** — no more evidence collection unless PI prioritizes optional S1b/Nsight/B3 optim.

**Policy:** Option A · JSON only · never invent · never clobber champion without BACKUP + OK.  
**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`**.  
**Git:** **`master` is always final** — `docs/BRANCHING_POLICY.md`.  
**DICC ops:** SSH ControlMaster host `dicc` preferred for agent work; OnDemand VNC + `screen` still valid for interactive; batch for long jobs — `docs/DICC_OPS_METHOD.md`.

---

## 2. Read first in the next chat (order)

1. **`HANDOFF.md`**  
2. **This file**  
3. `docs/PRE_MANUSCRIPT_CLOSURE.md`  
4. `docs/CLAIM_MAP_PREWRITE.md`  
5. `docs/DICC_EXTRACTION_TABLES.md`  
6. `docs/DICC_B3_CUDA_VS_PT_REPORT.md`  
7. `docs/DICC_MULTI_COMPILER_MATRIX.md`  
8. `docs/DICC_COMPARE_OUTCOMES.md`  
9. `docs/execution_plan/WP9b_MANUSCRIPT_SPINE.md`  
10. `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md`

---

## 3. What is DONE (do not redo)

| Deliverable | Location |
|-------------|----------|
| 6 SUCCESS campaign runs | `benchmarks/results/dicc/core/{v100s,a100}/` |
| Extraction tables | `docs/DICC_EXTRACTION_TABLES.md` |
| Formal compares V100; A100 `--allow-dirty` | `docs/DICC_COMPARE_OUTCOMES.md` |
| B3 PT-win report (S1d) | `docs/DICC_B3_CUDA_VS_PT_REPORT.md` |
| README Option A hygiene (S1c) | `README.md` abstract/contributions/cross-HW/limitations |
| torch.compile V100S+A100 (S1a) | jobs **395338/395339** · `framework/torch_compile_{v100s,a100}.json` |
| Script + SLURM recipes | `scripts/benchmark_torch_compile_dicc.py` · `logs/job_torch_compile_*.sh` |
| Pre-manuscript gate | `docs/PRE_MANUSCRIPT_CLOSURE.md` |

### Key scientific facts (locked)

- **Option A B3:** PT faster than CUDA FP16 on V100S (~363 vs ~513 µs) and A100 (~385–391 vs ~667–671 µs); stable across 3 sessions.  
- **B1/B2/B4:** CUDA still much faster than matching PT.  
- **Full CUDA vs full V3:** invalid.  
- **Laptop multi-compiler:** local ranges only.  
- **DICC torch.compile:** V100S eager ~1033 / compile ~818 µs; A100 eager ~957 / compile ~761 µs (~1.26×; absolute full-model; not Option A).

---

## 4. Open / deferred (explicit)

| Item | Status | Action if resuming |
|------|--------|-------------------|
| Full multi-compiler V100S+A100 | **DONE** 395433/395417 | `DICC_MULTI_COMPILER_MATRIX.md` |
| S1b clean A100 re-run | **DEFERRED** optional | Only if provenance reviewers demand no `--allow-dirty` |
| S2c Nsight | **NOT RUN** | Optional systems depth |
| Manuscript multi-GPU section | **NEXT** | Write from JSON-backed tables only |
| PI venue / BibTeX | Open | After journal choice |

---

## 5. Ops cheatsheet

```bash
# Laptop
cd /home/titoisalive/colide
ssh -o BatchMode=yes dicc 'squeue -u $USER; sacct -j 395339 --format=JobID,State,ExitCode,Elapsed -P'

# Rsync framework + core
rsync -avz -e ssh dicc:~/colide/benchmarks/results/dicc/framework/ \
  benchmarks/results/dicc/framework/
# or: bash scripts/rsync_dicc_results.sh

# Micromamba on cluster (do not use broken system py3.9 venv on GPU nodes)
# $HOME/micromamba/bin/micromamba run -n colide python …
```

**Do not:** invent multi-day numbers; claim portable B3 CUDA win; full-pipeline CUDA vs V3; clobber champion; password-spam SSH.

---

## 6. Stretch status board

| ID | Item | Status |
|----|------|--------|
| S1a / full multi-compiler | eager+compile+ORT+TRT V100+A100 | **DONE** (matrix pack) |
| S1b | Clean A100 re-run | **DEFERRED** |
| S1c | README hygiene | **DONE** |
| S1d | B3 stats one-pager | **DONE** |
| S2c | Nsight | not run |

---

*Update this file only when phase boundary changes (e.g. manuscript draft land, A100 compile JSON filled).*
