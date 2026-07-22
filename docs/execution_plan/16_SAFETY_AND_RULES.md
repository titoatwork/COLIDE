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

## 5b. Git branching (binding — all agents)

**Canonical:** `docs/BRANCHING_POLICY.md`

| Rule | Detail |
|------|--------|
| **Final line** | **`master` is always final** — handoff, claims, manuscript tip, next-session resume |
| **When to branch** | **Must** open a new branch when work is a **true alternative option** (could be discarded, Option-B-style fork, risky isolation) |
| **When not to** | Continuity/docs/claims hygiene, locked-path incremental WPs — stay on **`master`** |
| **Budget** | Keep branch count **low** (prefer **0–2** open feature branches; soft-cap **≤3** remote non-master); no vanity/`wip/agent-*` spam |
| **After merge** | Delete local + remote feature branch; do not leave second “final” branches |
| **Force-push** | Never force-push `master` without explicit user OK |

Historical single-branch work on `master` was fine. Future option-forks use short-lived branches, then merge back to **`master`**.

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
