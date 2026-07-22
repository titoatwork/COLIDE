# 16 — Safety, Claim Rules, Session Rules

---

## 1. Checkpoint safety

| Rule | Detail |
|------|--------|
| Never silent clobber | Do not overwrite `best_model_botiot_twostage.pth` without copy to `BACKUP_<tag>_<date>.pth` |
| New candidates | Separate filenames: `best_model_botiot_cad_seed42.pth` etc. |
| Promote champion | Only after sealed test + user explicit OK + md5 logged |
| Historical | Keep 0.9790 backup forever for audit |

---

## 2. Number safety

| Rule | Detail |
|------|--------|
| No invention | Empty multi-day cells until JSON exists |
| JSON only | Cluster tables from SUCCESS artifacts |
| Ranges | Laptop latency multi-session ranges |
| Legacy | June 551/592 V100S/A100 labeled legacy until re-verified |
| Option A | Per-block CUDA vs matching PT only |

---

## 3. Option A (still binding)

- No full-pipeline Custom CUDA vs full PyTorch V3 apples-to-apples speedup  
- No “same computation” for incomplete CUDA vs V3  
- Full-model framework latencies OK as absolute “production cost”  
- Harness `full_pipeline_cuda_vs_pytorch.valid = false` respected  

---

## 4. Prof order (binding)

1. DICC first  
2. Focused improvements with decision table  
3. One method at a time  
4. Manuscript after evidence  

---

## 5. Session lifecycle

- One major WP per chat when possible  
- End: `git status`, commit, push, update HANDOFF  
- `verify_claims` after public number edits  
- Agents: no DICC invent; no DICC login; ops method = OnDemand VNC + screen + batch (`docs/DICC_OPS_METHOD.md`) 

---

## 6. Venue honesty

- RF may remain higher F1 — then **multi-objective** must be the win  
- Contribution ≠ documentation quality alone  
- Title words must be earned  

---

## 7. Official cluster

- **UM DICC only** for paper multi-day  
- Rostam = tooling trial only  

---

*Violating this file is not exceptional work.*
