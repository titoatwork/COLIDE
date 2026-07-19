# COLIDE — Prof Por Feedback Execution Roadmap

**Source:** `docs/feedback1.docx` (Prof. Por Lip Yee) + email reply  
**Date locked:** 2026-07-19  
**Status:** AUTHORITATIVE for post-interim work  
**Supersedes for ordering:** “manuscript next” thinking. **Does not** cancel Option A CUDA claim rules.

---

## 0. One-line mandate

> **DICC first → focused scientific/systems strengthening → only then manuscript.**  
> Contribution must be a **clear quantitative advantage** on ≥1 major dimension (detection, latency, memory, energy, throughput, robustness, portability, or multi-objective trade-off) — **not** implementation effort or documentation quality alone.

---

## 1. What is locked from Option A (still)

| Keep | Do not re-open casually |
|------|-------------------------|
| Per-block Custom CUDA vs **matching** PyTorch only | Full-pipeline CUDA vs full V3 as apples-to-apples |
| No “same computation” language | Invented multi-day / cross-HW ratios |
| Official cluster = **UM DICC** | Rostam as paper-final |
| June 551/592 = **legacy single-shot** until re-verified | Raising RF bar silently without protocol |
| Operation parity on CUDA claims | Leading with invalid speedups |

**Changed vs old freeze:** Systematic HPO, method novelty, retrain under protocol, stronger baselines, and real XAI eval are **now expected** for the WoS path Prof described. Champion `0.9790` remains **baseline reference** until a new champion is selected on **validation only** with backup + test freeze.

---

## 2. Execution order (do not skip)

```text
PHASE 0  Complete UM DICC multi-day campaign (HARD GATE)
   ↓
PHASE 1  Freeze reproducible baseline protocol (splits, seeds, metrics, 5-run if feasible)
   ↓
PHASE 2  Modification decision table → pick ONE primary method (not everything at once)
   ↓
PHASE 3  Controlled HPO on validation only (Optuna/Bayesian); test untouched
   ↓
PHASE 4  Imbalance + teacher/KD improvements under that method
   ↓
PHASE 5  Ablations + fair baselines (same split) + multi-objective tables
   ↓
PHASE 6  Deployment path: export, parity, kernels, FP16/(INT8), TRT/ORT as needed
   ↓
PHASE 7  Explainability beyond dispatch (only if title claims it)
   ↓
PHASE 8  ToN-IoT / second dataset pass on final method
   ↓
PHASE 9  Numbers-match + verify_claims + manuscript (Prof structure)
```

**Parallel while Phase 0 waits (allowed prep only):**

- This roadmap + mod decision table (below)  
- Inventory of existing scripts/results vs Prof checklist  
- Draft reply / status notes  
- **No** inventing DICC numbers; **no** full manuscript; **no** clobber champion without backup  

---

## 3. Phase 0 — UM DICC (CURRENT HARD GATE)

### 3.1 Required deliverables (from feedback §8)

| Item | Status now |
|------|------------|
| Block 3 custom CUDA vs matching PyTorch **same GPU** | **ABSENT** |
| Full PyTorch model absolute latency same GPU | **ABSENT** |
| V100S + A100 | **ABSENT** |
| ≥2 days, compare | **ABSENT** |
| mean / median / std / CV / CI | **ABSENT** |
| warm-up protocol documented | tooling exists |
| batch-size sensitivity | **NOT RUN** (cluster) |
| significance + effect size | **NOT RUN** (cluster) |
| Laptop path `benchmarks/results/dicc/` | **ABSENT** |

### 3.2 Operator path

`dicc_scripts/run_campaign.sh` Day1 → Day2 → `compare_dicc_sessions.py` → scp tree home.  
Partitions: `gpu-v100s`, `gpu-a100`.  
**Agents do not invent results.** User/ops run cluster.

### 3.3 Exit Phase 0

Local `benchmarks/results/dicc/**/SUCCESS` for intended GPUs + compare outcome recorded + no fabricated cells.

### 3.4 Decision fork after Phase 0 (read results cold)

| Cluster B3 CUDA vs PT | Implication |
|----------------------|-------------|
| CUDA competitive / faster | Primary systems win = **latency + portability**; reinforce with memory/trade-off |
| CUDA slower | Latency-as-headline at risk → **kernel fix re-run** OR pivot primary to **memory/trade-off/detection** |

---

## 4. Phase 1 — Strong reproducible baseline

Freeze and document:

- preprocessing pipeline  
- train / val / test split  
- metrics (macro-F1, balanced acc, minority recall, per-class F1, …)  
- seeds  
- hardware protocol for train vs inference  
- **baseline CNN–BiLSTM** (current path as reference: two-stage 0.9790 / md5 `80a90f7…`)  

Report mean ± std over **≥5 independent training runs where feasible** (new protocol).  
**Do not touch test set for selection.**

### Exit Phase 1

Written baseline card + scripts/configs that reproduce reference under frozen protocol.

---

## 5. Phase 2 — ONE clear proposed method (after decision table)

Prof suggestion (not mandatory branding until evaluated):

**Class-Aware Distilled Attention CNN–BiLSTM** (example package):

- multi-scale 1D CNN  
- BiLSTM  
- lightweight temporal / class-aware attention  
- class-balanced focal or logit-adjusted loss  
- RF or ensemble teacher distillation  
- class-specific thresholds  
- (later) prune / quant for deployment  

**Rule:** Introduce **one** core novelty story aimed at a named weakness (imbalance / minority / deployment).  
Do **not** change architecture + loss + teacher + CUDA + XAI all at once.

### Exit Phase 2

Signed decision: which single method package is in / out (use §11 table).

---

## 6. Phase 3 — Hyperparameter optimisation

- Tool: Optuna / Bayesian (reproducible)  
- Search space: as in feedback (filters, kernels, CNN blocks, BiLSTM h/layers, dropout, LR, batch, γ, T, α, seq length, class thresholds)  
- Objective: **validation** macro-F1 + balanced acc + minority recall (optionally multi-obj with latency/params later)  
- **Test set locked** until final config chosen  

### Exit Phase 3

Best config on val + sealed test evaluation once + logs/JSON under `benchmarks/results/`.

---

## 7. Phase 4 — Imbalance + teacher (under chosen method)

Compare strategies Prof listed (weighted CE, focal, CB-focal, logit adj, stratified sampling, thresholds, controlled oversampling, SupCon if in scope).  
Teachers: RF / XGB / LGBM / ensemble / neural — justify student via **deployment trade-off**, not pure imitation.

### Exit Phase 4

Selected imbalance + teacher with val evidence; test only at end of selection chain.

---

## 8. Phase 5 — Ablations + baselines + multi-objective

### Ablations (minimum)

CNN only; BiLSTM only; CNN–BiLSTM; +KD; +imbalance; +attention/fusion; full method.  
Each: macro/weighted F1, bal acc, P/R, per-class F1, latency, params, memory, energy.

### Baselines (same split)

LR, SVM, RF, XGB, LGBM, MLP, 1D-CNN, LSTM, BiLSTM, CNN–LSTM, CNN–BiLSTM, Transformer/temporal-attn, reproducible lightweight IDS.

### Multi-objective

Near-RF detection + memory + latency + throughput + energy + minority + cross-GPU.  
**Pareto** preferred.

### Exit Phase 5

Tables that answer Prof’s five questions (detection, minority, latency/memory, portability, explain value).

---

## 9. Phase 6 — Deployment / CUDA path

- Export **exact** trained final model  
- Operation parity checks  
- Profile → optimise **dominant** kernels only  
- Custom CUDA vs matching PT (blocks); full model framework absolutes separate  
- FP32 / FP16 / (INT8 optional) with accuracy delta  
- TRT / ORT / compile as appropriate  

Option A still governs claims.

### Exit Phase 6

JSON-backed latency/memory/energy on laptop + DICC for final model.

---

## 10. Phase 7–9 — XAI, second dataset, manuscript

| Phase | Content | Only if |
|-------|---------|---------|
| 7 XAI | Faithfulness, consistency, vs SHAP/LIME/attn, structured evidence into LLM | Paper claims explainability |
| 8 ToN / other | Final method on second dataset; transfer if possible | Always for final paper |
| 9 Manuscript | Prof outline (title only if components evaluated); numbers-match; fix gitignore claim sources | After 0–6(+7/8) |

---

## 11. Modification decision table (fill before Phase 2 commit)

| ID | Modification | Targets weakness | Expected benefit | Cost (GPU-days) | Risk | Depends on DICC? | Decision |
|----|--------------|------------------|------------------|-----------------|------|------------------|----------|
| M0 | Complete multi-day DICC protocol | Portability unknown | Systems validity | Ops/queue | High if skip | **IS** Phase 0 | **MUST** |
| M1 | Systematic HPO (Optuna) on val | Under-tuned net | Macro-F1 / minority | Med | Overfit if test leaked | No | **SHOULD** after 0–1 |
| M2 | Class-balanced focal / logit-adj | Imbalance | Minority F1 | Low–Med | None large | No | Candidate |
| M3 | Lightweight temporal attention | Weak temporal focus | Macro / minority | Med | Complexity; CUDA parity later | No (CUDA later) | Candidate (Prof lean) |
| M4 | Multi-scale 1D CNN | Multi-scale patterns | Macro-F1 | Med | Arch change | No | Candidate |
| M5 | Gated CNN–BiLSTM fusion | Feature fusion | Macro-F1 | Med | Complexity | No | Optional |
| M6 | Ensemble teacher (RF+XGB+LGBM) | Teacher ceiling | Student F1 | Med | Student still &lt; RF | No | Candidate |
| M7 | Class-specific thresholds | Minority | Rare-class recall | Low | Calibration leakage if done on test | No | **SHOULD** |
| M8 | SupCon / asymmetric loss | Minority separation | Rare-class | High | Scope creep | No | Later |
| M9 | Full baseline suite fair split | Weak comparison | Reviewer defense | Med | Time | No | **MUST** before paper |
| M10 | Ablation suite | Black-box method | Reviewer defense | Med | Time | No | **MUST** |
| M11 | Multi-obj Pareto (F1–lat–mem) | “Only accuracy or only CUDA” | WoS contribution clarity | Low (analysis) | Weak if numbers poor | Partial | **MUST** |
| M12 | Kernel optim post-DICC | Server B3 loss | Portability win | High | Only if DICC fails | **Yes** | Conditional |
| M13 | Full V3 CUDA (attn/LN/GAP) | Full-pipeline claim | Option B | Very high | Delay | No | **OUT** default |
| M14 | XAI quality eval | Dispatch-only XAI | Explain contribution | High | Subjective | No | If title claims |
| M15 | ToN / transfer | Single-dataset | Generalisation | Med | — | No | **MUST** for final |
| M16 | Package claim JSONs | Repro | Trust | Low | — | No | **SHOULD** early |
| M17 | Retrain to beat RF only | F1 gap | Accuracy headline | High | May fail; not only goal | No | Not sole goal |

**Initial recommended package (to confirm after Phase 0):**  
**M0 → M1 baseline freeze → M2+M7+M6 (class-aware distill path) + optional M3 → M9–M11 → M12 if needed → M14 if claiming XAI → M15 → manuscript.**

---

## 12. Contribution targets (realistic)

| Goal | Realistic? |
|------|------------|
| Beat RF on all metrics | Unrealistic (Prof: cuML RF thr very high) |
| Near-RF macro-F1 + much lower memory + competitive neural latency | **Primary realistic** |
| Valid same-GPU B3 CUDA win on V100S/A100 | **If DICC agrees** |
| Explainability beyond 16.60 µs dispatch | Only with extra eval |
| Full-pipeline custom CUDA vs full V3 | **Not** without Option B |

---

## 13. Session packaging rule

One major package ≈ one chat:

| Package ID | Work |
|------------|------|
| Pkg-0 | Guided / execute DICC → artifacts home |
| Pkg-1 | Baseline freeze card + 5-run protocol |
| Pkg-2 | Decide M-package from table + implement method skeleton |
| Pkg-3 | HPO campaign |
| Pkg-4 | Imbalance + teacher sweeps |
| Pkg-5 | Baselines + ablations |
| Pkg-6 | CUDA/deploy re-export + re-bench |
| Pkg-7 | XAI eval (optional title-dependent) |
| Pkg-8 | ToN final |
| Pkg-9 | Numbers match + manuscript spine |

---

## 14. Champion / training safety

1. Never overwrite `best_model_botiot_twostage.pth` without copy to `BACKUP_*`.  
2. New candidates: separate filenames + JSON metrics.  
3. Promote to “production champion” only after sealed test eval + user OK.  
4. Keep md5 of any published checkpoint in claims.

---

## 15. Immediate next action (now)

| Priority | Action | Owner |
|----------|--------|-------|
| **1** | Run **Phase 0 DICC** (user/ops) | User |
| **2** | Keep this roadmap as authority | Repo |
| **3** | While waiting: baseline inventory + implement prep for M1/M9 scripts | Agent (separate packages) |
| **4** | After DICC JSON home: extract + fork decision + start Phase 1–2 | Agent + user |

**Do not start full HPO/retrain mega-sweep in the same chat as DICC ops.**

---

## 16. Success criteria (pre-manuscript “Prof bar”)

- [ ] Phase 0 DICC complete with stats he listed  
- [ ] Clear quantitative win on ≥1 major dimension (or strong Pareto)  
- [ ] Method justified + val-only selection + untouched test  
- [ ] Ablations + fair baselines  
- [ ] Option A CUDA claims only  
- [ ] Minority-class reporting solid  
- [ ] Repro: claim sources not only gitignored local  
- [ ] XAI depth matches title claims  
- [ ] ToN or second dataset on final method  

---

*End of roadmap. Authority: feedback1.docx + this file. Option A CUDA rules remain.*
