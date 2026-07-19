# 14 — Exceptional Standards (Not “Just Enough”)

This file defines the quality bar for every phase. If a deliverable fails these checks, it is **not done**.

---

## 1. Scientific integrity

| Standard | Rule |
|----------|------|
| Test seal | Test set never used for model selection, HPO, or threshold search |
| Pre-registration lite | Composite score / minority constraints written before final selection |
| Multi-seed | Final configs confirmed with ≥3–5 seeds when compute allows |
| Negative results kept | Failed γ, collapsed minority configs stay in appendix |
| No cherry ranges | Latency as multi-session ranges; no lucky single points as headlines |
| Stats | Report mean/median/std/CV/CI; tests + effect sizes for key comparisons |

---

## 2. Engineering integrity

| Standard | Rule |
|----------|------|
| Single data protocol | All new trains call unified loader |
| Result schema | Every JSON: git_sha, hostname, gpu, seed, n_trials, timestamp, config hash |
| Checkpoints | Never overwrite champion without `BACKUP_*` + log |
| Claims | Every public number has source path |
| Option A | Per-block only; no same-computation falsehoods |
| Repro | Claim-source JSONs tracked or checksum manifest published |

---

## 3. Contribution integrity

| Standard | Rule |
|----------|------|
| Named win | Paper states primary dimension of advantage in one sentence |
| Ablation proof | Every claimed component removed and measured |
| Fair baselines | Same split; tuning budget disclosed |
| Portability | No cross-GPU claim without DICC evidence |
| XAI | No “explainable” without Phase 7 metrics |

---

## 4. Definition of done (per experiment)

An experiment is done only when:

1. Code merged/committed  
2. JSON results on disk with schema  
3. Val metrics logged  
4. Test metrics only if sealed stage  
5. README/paper table updated **or** explicitly deferred with ticket  
6. verify_claims updated if number is public  

---

## 5. Anti-patterns (reject in review)

- “We’ll fix stats in the paper later”  
- Tuning on test “just to see”  
- Full-pipeline CUDA vs V3 speedup  
- Claiming multi-day without SUCCESS tree  
- Kitchen-sink architecture without ablation  
- Title promises (class-aware/explainable) without eval  
- Shipping interim process quality as the contribution  
- **Silently skipping a Prof-required experiment** without RUN_DOCUMENTED / BLOCKED note  

---

## 5b. Skip-nothing experiment policy (user-locked)

| Step | Action |
|------|--------|
| 1 | Identify tracker ID (e.g. C7 SupCon, D2 weighted CE) |
| 2 | Run bounded experiment under frozen protocol |
| 3 | Save JSON + log path in tracker notes |
| 4 | If useful → INCORPORATED into final package / paper table |
| 5 | If not → RUN_DOCUMENTED: one paragraph + metrics (negative result kept) |

Final model may be lean; **evidence trail must be complete**.

---

## 6. Exceptional bar summary

> **Adequate** satisfies a checklist.  
> **Exceptional** makes a skeptical WoS reviewer unable to dismiss the work as “another CNN-BiLSTM + some CUDA” without engaging your tables.

We optimise for the second.
