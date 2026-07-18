# 05 — Git Re-verification Archaeology

**HEAD inspected:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`  
**Method:** `git log -S/-G`, `git show` on key commits, `git log -- path` for results.

---

## NUMBERS RE-VERIFIED VIA HISTORY

### 1) Fabricated / unsourced numbers found and fixed (2026-07-01 cluster)

| What was wrong | Fix commit | Current SoT | Residual risk |
|----------------|------------|-------------|---------------|
| LLM p99 **5.19 us** fabricated (hardcoded in ablation_study; no percentile) | **`09b509b`** (2026-07-01) | `llm_explainability.json` overhead_p99_us=**16.599…→16.60** n=5000 | Single machine/run; not multi-session |
| Pipeline speedup **2.76x** from unsourced **1864.0** PyTorch baseline in fused_pipeline.cu; V100 **3.39x** / A100 **3.15x** cross-hardware | **`a0ff1a8`** | RTX ratio consolidated to eager framework comparison; V100/A100 ratios **removed** | Additive pipeline methodology still weak |
| Significance one-sample vs bare constant | **`99b7f80`** | two-sample Welch in stats_v2 | ORT CPU still unstable across sessions |
| Weight export pointed at **stale pre-distillation** checkpoint | **`96dfc58`** then **`c27ac2a`** re-export for 0.9790 | weights_bin + real_weight_validation | CUDA still last-timestep vs V3 GAP |
| README/paper batch of stale/fabricated | **`e928d8e`** | verify_claims tracks | Some per-block points never multi-sessioned |
| Sophimatics "2.7x CNN CUDA" fabricated citation | **`df958f1`** | Ibrahim et al. CN 2026 | PDF not fully read |
| Uncorroborated RF **0.9864** | **`acdcba5`** | `rf_baseline_processed.py/json` | JSON gitignored |
| Ambiguous B3 cuDNN **740.7 vs 943.6** | **`d85271c`** | `pytorch_block3_stats` mean **784** n=50 | n=50 single session file |
| Naive B3 "FP32 rounding" was **data race** | **`3eb773a`** | race-fixed; 100/100 @1e-2 | — |
| Stale 0.9639/0.9601 comparison strings after 0.9790 | **`547e895`** | 0.9790 surfaces | cuml JSON still has 0.9639/0.9601 labels |

### 2) Measurement stability / multi-session range widenings

| Event | Commit | Effect |
|-------|--------|--------|
| Framework comparison → session ranges | **`bd3777e`** (2026-07-02) | Point 4.40x/3.33x/2.63x → ranges; HIST+SESSION3A hardcode + live file |
| Full 4-block re-verification; include EXTRA totals 594/614.5 | **`d9e1f79`** (2026-07-02) | Custom CUDA range **594–675** (was 652–675); ratios widen to 3.60–4.99 / 2.25–2.99 / 3.04–3.78 / 5.72–7.83; B3 progression 7.55–9.50 |
| verify_claims + kernel stats harness introduced | **`9c8d86f`** | regression infrastructure |
| Intermediate ranges also guarded | inside d9e1f79 REGRESSION_GUARDS | 3.60x–4.55x, 652–675, 8.08x–9.21x etc. banned |

**Mechanism (critical):** Multi-session framework ranges are **NOT** stored as a multi-session JSON array. They are:

```
HIST_LATENCY + SESSION3A_LATENCY + live statistical_significance_v2.json
+ CUSTOM_CUDA_EXTRA_TOTALS [614.5, 594.0]
```

in `scripts/verify_claims.py` `build_claims()`. Overwriting live JSON without updating hardcodes can shrink ranges silently until next audit.

### 3) KD path → champion 0.9790

| Step | Commit / artifact | Result |
|------|-------------------|--------|
| Round 1 KD+focal best | distill_botiot_focal_T5 0.9601; twostage 0.9639 | SUPERSEDED |
| Round 2 T>5 grid | **`f98bf33`** | a=0.6 T=10 γ=2 stage-1 **0.9763** |
| Two-stage FT | twostage_botiot.json | **0.9790** |
| Stage-1 bit-identical repro | **`2560348`** docs + `*_repro.json` | identical macros |
| Focal γ∈{1,3,4} | **`38b1ea6`** | negative vs γ=2; champion unchanged |
| S5 ensemble/RF strengthen | **`31940dd`** | champion kept; RF bar not raised to 0.9885 |

### 4) Fidelity self-check introduction

| Item | Commit | Artifact |
|------|--------|----------|
| Session 7 fidelity table | **`3f99243`** | numerical_fidelity.json (on disk; **gitignored**) |
| Real weight validation pass after re-export | **`c27ac2a`** / f80b9133 era | real_weight_validation.json TRACKED |

### 5) DICC summary vs campaign hardening

| Item | Commit | Status |
|------|--------|--------|
| Legacy summaries introduced | **`dd923c6`** (2026-06-20) | CURRENT as LEGACY files |
| DICC multi-day harden | **`ba6e0cb`** et al. final-polish | tooling ready |
| run_campaign one-shot | **`7c8a97f`**, **`6e1baaa`** --full | scripted |
| Portable / Rostam fixes | ef3b3ba…991a941 | TOOLING |
| Official multi-day SUCCESS on laptop | — | **ABSENT** |

---

## NUMBERS NEVER RE-RUN / SINGLE-SHOT ONLY

| Metric | Artifact | Label |
|--------|----------|-------|
| V100S 550.664 / A100 592.044 pipeline | dicc_*_summary.txt | LEGACY single-shot CUDA-only |
| A100 energy 1.089 mJ | a100_energy.json | single commit ca24ccf |
| MLP latency 175 us | mlp_latency.json | single |
| Streaming 25,899 | streaming_throughput.json | single file (not multi-session) |
| Energy RTX 0.79 | energy_efficiency.json | single |
| cuML resources table | cuml_rf_resources.json | n_runs=2 but not multi-day campaign |
| Most KD train cells | distill_*.json | each training once (repro only for winner stage-1) |
| Stage-2 fine-tune bit-identity | — | **not** empirically re-proved |
| Per-block PT 404/282/122 and CUDA 62/87/20 | pipeline_benchmark era | single-run; drift known |
| Step1 2,901 us | historical | no artifact |
| baseline_latency.json | early | LEGACY |
| tensorrt_native / torch_compile_native point ratios | | SUPERSEDED by ranges for headlines |
| statistical_confidence.json early CIs | | SUPERSEDED by v2 |

---

## NUMBERS WITH MULTI-SESSION RANGES

| Metric | Sessions claimed | Where encoded | On-disk multi-session store? |
|--------|------------------|---------------|------------------------------|
| Framework latencies | 3 framework-side | verify_claims HIST+S3A+live | **NO** (hardcoded + live overwrite) |
| Custom CUDA derived total | 5 | + EXTRA totals | **NO** full history file |
| Block3 FP16 / transpose / graphs | 5 | verify_claims + README prose | live file = last session only |
| Block3 naive | 3 | verify_claims | live only |
| ORT CPU significance | 3 | prose | live p from latest |

---

## NUMBERS DOC-ONLY OR UNSOURCED (residual)

| Item | Status |
|------|--------|
| README "same computation" full-pipeline vs V3 | **INVALID claim language** (sourced as false by DESIGN_PLAN) |
| Preprocess 43.7 us / e2e 717.7 us | paper_text_blocks; weak/no JSON |
| Rostam Day1 means | DESIGN_PLAN table only; not results JSON |
| Theoretical 100% occupancy | gpu_hardware_profile; Nsight not on WSL2 |
| Alert aggregation 25k→10 | design script; not measured campaign metric |
| Ibrahim 1.22–1.48x | external paper claim |

---

## Key commit timeline (hashes inspected)

```
9c8d86f  tools: claim verifier + CUDA kernel statistical harness
e928d8e  docs: correct fabricated/stale numbers
09b509b  fix: LLM 5.19 → 16.60 real p99
a0ff1a8  fix: unsourced 1864 / 2.76x / cross-HW ratios
99b7f80  fix: one-sample → two-sample significance
96dfc58  fix: stale export checkpoint pointer
acdcba5  fix: RF 0.9864 traced to real script
d85271c  fix: cuDNN baseline n=50 → 784us
3eb773a  fix: naive B3 race
f98bf33  feat: KD T-sweep → gap 0.74% / path to 0.9790
547e895  fix: stale 0.9639 strings
2560348  docs: KD bit-identical repro
c27ac2a  fix: re-export weights for 0.9790
bd3777e  fix: framework numbers → session ranges
d9e1f79  fix: widen ranges after 4-block re-verification
df958f1  fix: Sophimatics → Ibrahim
38b1ea6  feat: focal-γ negative sweep
3f99243  docs: fidelity + threats-to-validity
ba6e0cb  feat: harden DICC multi-day workflow
7c8a97f  feat: run_campaign.sh
fea5204  docs: DESIGN_PLAN + PROF_POR_3DAY
37374ca  docs: FINAL_PLAN
23be30b  docs: hard-gate numbers match
ecc37e6  docs: contingency PROF_POR_STATUS_REPORT (NON-AUTH)
803c157  docs: audit prompt
dd923c6  Phase1: dicc_v100/a100 summaries (LEGACY)
```

---

## Gitignore impact on re-verification

Because `benchmarks/results/` is in `.gitignore`, several SoT files for re-verified claims are **not on GitHub** unless force-added. Tracked vs untracked split is a **first-class residual risk** for any external re-verifier. See `10_EVIDENCE_GAPS.md`.
