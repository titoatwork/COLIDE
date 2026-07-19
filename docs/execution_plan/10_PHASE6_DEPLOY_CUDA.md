# 10 — Phase 6: Deployment Path & CUDA (Option A)

**Status:** NOT STARTED (local harness DONE; cluster BLOCKED on Phase 0)  
**Depends on:** Final weights from Phases 2–5; Phase 0 for cluster claims  
**Prof:** staged Phase 4 deploy; §8 cluster; Option A  

---

## 1. Goals

1. Export **exact** final model weights.  
2. Prove **operation parity** for implemented CUDA blocks.  
3. Optimise **dominant** kernels only after profiling.  
4. Report: matching block comparisons + full-model framework absolutes **separately**.  
5. Precision study: FP32 / FP16 / (INT8 optional) with accuracy delta.

---

## 2. Steps

| Step | Action | Scripts |
|------|--------|---------|
| 1 | Export weights from final ckpt | `validate_weights.py` / update export path |
| 2 | Numerical fidelity | `numerical_fidelity.py` |
| 3 | Profile blocks | existing benches + optional Nsight |
| 4 | Kernel optim if Phase 0 shows server loss | CUDA sources |
| 5 | Local multi-session ranges re-measure | `benchmark_*_stats*` |
| 6 | Cluster re-run if model changed | Phase 0 repeat |
| 7 | Framework matrix as needed | ORT/TRT/compile |
| 8 | Memory/energy/throughput update | energy, streaming, cuml |

---

## 3. Claim rules (non-negotiable)

| Allowed | Forbidden |
|---------|-----------|
| Block i CUDA vs Block i PT matching ops | Full custom pipeline vs full V3 speedup as parity |
| Absolute full V3 PT/TRT/ORT latency | “Same computation” incomplete CUDA vs V3 |
| Multi-day cluster stats from JSON | Invented or RTX-only portable claims |

---

## 4. Acceptance criteria

- [ ] Final weights md5 + fidelity PASS  
- [ ] Per-block tables Option A clean  
- [ ] Full-model framework table separate  
- [ ] Cluster stats if model final (re-DICC if arch changed)  
- [ ] README/paper language audited for Option A  
- [ ] Precision accuracy deltas reported  

---

## 5. Exit

Systems Results 5.6–5.9 manuscript-ready.
