# 06 — Phase 2: One Clear Methodological Contribution

**Status:** NOT STARTED  
**Depends on:** Phase 0 (fork), Phase 1 (protocol)  
**Prof:** §2 novel method; staged plan Phase 3 “one clear proposed method”

---

## 1. Goal

Introduce **one** clearly motivated improvement package that addresses a **named weakness**, with ablations later proving it.

**Named weakness (default):**  
Severe class imbalance + minority failure modes under deployment-oriented neural IDS; RF remains stronger on macro-F1, so the method must improve **minority-aware detection** and/or **accuracy–efficiency trade-off**, not only average F1.

---

## 2. Recommended package (default; revisable)

**Working name:** Class-Aware Distilled CNN–BiLSTM (CAD-CBA)

| Component | Include? | Rationale |
|-----------|----------|-----------|
| Multi-scale 1D CNN | Candidate | Multi-scale flow patterns |
| Existing BiLSTM | Yes | Temporal stack |
| Lightweight temporal / class-aware attention | Candidate | Prof lean; differentiate from stock MHA if redesigned |
| Class-balanced focal **or** logit-adjusted loss | **Yes** | Imbalance |
| RF or ensemble teacher KD | **Yes** | Teacher ceiling |
| Class-specific thresholds (val-calibrated) | **Yes** | Minority |
| Prune/quant | Later Phase 6 | After accuracy fixed |

**Do not in Phase 2:** SupCon + asymmetric + uncertainty + full Transformer all at once.

---

## 3. Novelty framing (paper sentence)

> Existing CUDA-accelerated IDS models optimise inference but often under-treat minority classes under extreme imbalance. We jointly improve minority-class recognition and deployment efficiency via class-aware distillation, calibrated decisions, and [optional attention], evaluated under operation-matched CUDA protocols.

Only use if experiments support every clause.

---

## 4. Implementation plan

### 4.1 Code modules (new)

| Module | Responsibility |
|--------|----------------|
| `model/cad_cba.py` or extend v3 | Architecture deltas only if needed |
| `scripts/losses.py` | Focal, CB-focal, logit adjustment |
| `scripts/thresholds.py` | Per-class threshold search on **val** |
| `scripts/train_cad.py` | Unified trainer using Phase 1 protocol |

### 4.2 Experiments (before calling it final)

1. Baseline freeze (Phase 1)  
2. + class-aware loss only  
3. + thresholds only  
4. + stronger teacher only  
5. Combined package  
6. Optional attention add-on last  

Each: val metrics only until selection.

---

## 5. Interaction with CUDA

| If architecture changes ops | Action |
|-----------------------------|--------|
| New attention / multi-scale | Re-export weights; extend kernels **or** limit CUDA claims to still-matched blocks |
| Loss/threshold only | CUDA kernels may stay; re-bench still required |

**Option A:** Never claim full-model custom CUDA vs full V3 without parity.

---

## 6. Acceptance criteria

- [ ] Decision recorded in `MOD_DECISION_TABLE.md` (which M* IDs)  
- [ ] Code for package exists and trains under frozen protocol  
- [ ] Val improvement over baseline on macro-F1 **or** minority metrics (pre-registered)  
- [ ] Ablation plan ready for Phase 5  
- [ ] Title words (“class-aware”, “distilled”) only if evaluated  

---

## 7. Exit

Signed method package + checkpoints with non-colliding filenames (never silent overwrite of `best_model_botiot_twostage.pth`).
