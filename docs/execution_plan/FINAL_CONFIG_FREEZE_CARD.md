# Final Config Freeze Card — CAD-CBA-v1 (B14 gate)

**Purpose:** Explicit user lock before **BoT sealed multi-seed TEST** (tracker B14).  
**Status:** **USER-LOCKED 2026-07-22** · Init path **A** · B14 **DONE**  
**User lock text (verbatim):** *I LOCK CAD-CBA-v1 for sealed multi-seed BoT TEST (B14). Init path: A. Champion md5 must remain 80a90f7cc210276300eaa90173a5a385 unless I say BACKUP+replace. Proceed with sealed test only.*

### B14 sealed multi-seed TEST result (path A)

| Metric | Value | Source |
|--------|-------|--------|
| Init | distill `a0.6_T10.0_focal2` + `hpo_best` FT | path A |
| Seeds | 42–46 (n=5) | `sealed_test/` |
| **Test macro-F1 mean±std** | **0.9780 ± 0.0033** | `sealed_test/summary.json` |
| Test min-cls mean | **0.9292** | same |
| Test Theft mean | **1.0000** | same |
| Val macro-F1 mean±std | 0.9689 ± 0.0145 | same (matches HPO confirm) |
| Champion md5 after | `80a90f7cc210276300eaa90173a5a385` **unchanged** | |
| Wall | ~4768 s (~79 min) | |

---

## Method package (locked science; val-only evidence complete)

| Component | Freeze value | Evidence |
|-----------|--------------|----------|
| Name | **CAD-CBA-v1** | `METHOD_PACKAGE_DECISION.md` |
| Architecture | `cnn_bilstm_v3_attention` V3 dims | F3 / B2–B4 plateau reject |
| KD teacher | Ensemble mean(RF+XGB+LGBM) soft labels | WP4b student **0.9401** INCORPORATE |
| KD recipe | α=**0.6**, T=**10** | historical + WP4b |
| Loss | **focal** (no CB class-weight on neural) | D3; CB/logit_adj RUN_DOCUMENTED worse |
| Train HPs | `config/hpo_best.yaml` (trial 8) | WP3 winner val **0.9791** INCORPORATE |
| Decode | **argmax** | WP2d thresholds Δ0 |
| Train sampler | **shuffle** | D6 stratified hurts (0.9209 ≪ 0.9791) |
| Production champion ckpt | `model/best_model_botiot_twostage.pth` | md5 **`80a90f7cc210276300eaa90173a5a385`** |

### `hpo_best.yaml` winner params (exact)

| HP | Value |
|----|-------|
| lr | 5.89306076111462e-05 |
| batch_size | 1024 |
| focal_gamma | 1.9166447754858478 |
| dropout_rate | 0.14783769837532068 |
| attention_dropout | 0.21397343616689848 |
| weight_decay | 0.00019158219548093185 |
| scheduler | cosine |

---

## Explicitly **not** in final package

| Idea | Result | Decision |
|------|--------|----------|
| Multi-scale CNN (C4) | 0.9167 ≪ CTRL 0.9787 | RUN_DOCUMENTED out |
| Gated fusion (C5) | 0.9132 | out |
| SupCon+focal (C7) | 0.7732 | out |
| Asymmetric loss (C8) | 0.8012 | out |
| MC-dropout selective (C10) | no high-cov lift | keep argmax |
| Neural teacher (E6) | student 0.8513 ≪ ensemble | keep ensemble |
| Full arch Optuna B2–B4 | plateau | freeze V3 dims |
| Full LLM-explainable title | free-form mention 0.333 | J10 DROP full claim |

---

## What B14 sealed test will measure (after lock)

| Item | Spec |
|------|------|
| Dataset | BoT protocol `botiot_v1` **test** (currently sealed) |
| Seeds | ≥5 (recommend 42–46, same as val multiruns) |
| Init / recipe | Final freeze above — pick **one** init path and document it |
| Metrics | macro-F1, min-cls, Theft, bal-acc, per-class |
| Output | `benchmarks/results/sealed_test/` summary + per-seed JSON |
| Champion | **Do not overwrite** production champion without BACKUP + explicit OK |

### Init path decision (choose one when locking)

| Option | Init checkpoint | Rationale |
|--------|-----------------|-----------|
| **A (recommended)** | `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` + `hpo_best` FT per seed | Matches WP3 / HPO confirm protocol |
| B | Ensemble KD init + `hpo_best` | Matches package multirun (mean **lower** than WP1b) |
| C | Eval existing best ckpts only (no retrain) | Fast; weaker “fresh multi-seed train” story |

**Default recommendation:** Option A (same family as HPO confirm / WP1b distill init).

---

## User lock statement (paste to unlock B14)

```text
I LOCK CAD-CBA-v1 for sealed multi-seed BoT TEST (B14).
Init path: A | B | C  (pick one)
Champion md5 must remain 80a90f7cc210276300eaa90173a5a385 unless I say BACKUP+replace.
Proceed with sealed test only.
```

Until that message appears, agents must **keep BoT test sealed**.

---

## After B14

1. ~~WP6b local multi-session latency/energy ranges~~ **DONE 2026-07-22** (energy 0.920–0.943; PT@256 24.15–25.68 µs; CUDA pipe 565–570)  
2. ~~WP9a claim flips for test numbers~~ **DONE** (59 claims post-WP6b)  
3. ~~WP9b manuscript spine~~ **DONE 2026-07-22** (`WP9b_MANUSCRIPT_SPINE.md`)  
4. ~~Camera-ready local-complete draft~~ **DONE 2026-07-22** (`docs/manuscript/CAD_CBA_v1_MANUSCRIPT.{md,pdf}` + figures)  
5. DICC only in a dedicated user-opened session  
6. PI venue polish / submission formatting — open  

---

*Card written 2026-07-22 (WP9a session). User-locked + B14 + WP6b + WP9b spine + WP9c draft complete.*
