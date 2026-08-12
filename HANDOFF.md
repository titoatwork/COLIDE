# COLIDE — Session Handoff

**MODE:** DICC multi-session **COMPLETE** · tree **on laptop** · compare **partial** (V100 ran; A100 rejected dirty).  
**Authority:** Option A · no invent · `docs/DICC_RESULTS_AND_FLAGS.md` · `docs/DICC_COMPARE_OUTCOMES.md`.  
**Champion:** md5 **`80a90f7cc210276300eaa90173a5a385`**.

---

## SUCCESS (laptop `benchmarks/results/dicc/`)

| Session | Label | V100S | A100 |
|---------|--------|-------|------|
| S1 | `20260807` | 390642 | 390643 |
| S2 | `20260807_s2` | 390653 | 390654 |
| Day2 | `20260808` | 390781 | 390782 |

---

## Critical flags

1. **B3 CUDA FP16 slower than PT B3** both GPUs (~513 vs ~363 V100; ~668–671 vs ~384–391 A100).  
2. Full CUDA vs full V3 PT **invalid**.  
3. A100 formal compare **REJECTED** (`git_dirty=true`); V100 S1–Day2 compare ran but **not session-stable** on all metrics (B1 11% spread); S1–S2 V100 **stable**.  
4. Multi-compiler TRT/compile/ORT = **laptop only**.  

---

## Next (user claim decision)

- Paper tables from local JSON only  
- Tracker/claims wording for honest B3  
- Optional: clean re-run A100 for formal compare accept  
- Manuscript only after you approve claim strategy  

**Do not** invent numbers. **Do not** clobber champion.

---

## Ops that worked

- On-node `nvcc -std=c++11` kernels (sm_70 / sm_80)  
- micromamba `colide` env on GPU nodes (login venv py3.9 broken on compute py3.6)  
- batch `run_campaign.sh` with kernels prebuilt  

---

## Paste-ready next prompt

```text
Continue COLIDE — post-DICC pre-manuscript.
Read: HANDOFF.md, docs/DICC_RESULTS_AND_FLAGS.md
VPN + rsync dicc tree if missing; run compare_dicc_sessions; document accept/reject;
update tracker A3/H7/I* for multi-GPU with honest B3 language. No invent. Option A.
```
