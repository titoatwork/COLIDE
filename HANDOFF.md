# COLIDE — Session Handoff

**MODE:** DICC multi-session campaign **COMPLETE on cluster** (S1 + S2 + Day2 both GPUs SUCCESS).  
**Laptop:** results tree **not yet rsynced** (VPN required).  
**Authority:** Option A · no invent · `docs/DICC_RESULTS_AND_FLAGS.md` · `docs/DICC_OPS_METHOD.md`.  
**Champion:** `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`**.

---

## Cluster SUCCESS (all six)

| Session | Label | V100S | A100 |
|---------|--------|-------|------|
| S1 | `20260807` | 390642 | 390643 |
| S2 | `20260807_s2` | 390653 | 390654 |
| Day2 | `20260808` | 390781 | 390782 |

Paths under cluster `~/colide/benchmarks/results/dicc/core/{v100s,a100}/`.

---

## Critical flags (see `docs/DICC_RESULTS_AND_FLAGS.md`)

1. **B3 CUDA FP16 slower than matching PT B3** on V100S (~513 vs ~363 µs) and A100 (~668–671 vs ~384–391 µs).  
2. Full CUDA vs full V3 PT speedup **invalid**.  
3. Formal `compare_dicc_sessions.py` may **reject** runs with **git_dirty=true**.  
4. Multi-compiler (TRT/compile/ORT) evidence is **laptop**, not DICC campaign.  
5. Laptop `benchmarks/results/dicc/` still nearly empty until rsync.

---

## Next (user decides claim strategy after)

```bash
# 1) VPN + ssh dicc once (ControlMaster), then laptop:
bash scripts/rsync_dicc_results.sh

# 2) Compare (may flag dirty git):
PYTHONPATH=. python scripts/compare_dicc_sessions.py \
  --run-a benchmarks/results/dicc/core/v100s/20260807_job390642 \
  --run-b benchmarks/results/dicc/core/v100s/20260808_job390781
# similarly a100 S1 vs Day2

# 3) Update tracker + claims only with JSON-backed language
```

**Do not** invent multi-day tables. **Do not** clobber champion.

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
