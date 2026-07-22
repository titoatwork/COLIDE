# 11 — Report Writer Brief (machine-oriented)

**NOT a professor letter.** Feedstock rules for a later report-writing LLM.  
**Audit HEAD:** `803c157ddc1ed4103452f08044a7e0cd74cc728b`  
**Authority:** Option A + FINAL_PLAN.  
**Non-auth:** `docs/PROF_POR_STATUS_REPORT.md`.

---

## Facts you MAY state (with sources)

### Accuracy (CURRENT, HIGH)

- BoT-IoT two-stage CNN-BiLSTM test macro-F1 **0.9790** (`benchmarks/results/twostage_botiot.json`; checkpoint md5 **`80a90f7cc210276300eaa90173a5a385`**).
- Apples-to-apples sklearn RF test macro-F1 **0.9864** (`rf_baseline_processed.json`).
- Gap **0.74%**; do **not** claim beating RF / SOTA accuracy.
- Stage-1 KD winner a=0.6, T=10, γ=2 → test **0.9763**; bit-identical repro JSON matches.
- Focal γ∈{1,3,4} did not displace γ=2 champion path.
- ToN-IoT clean CNN-BiLSTM **0.9526**, RF **0.9851**, gap ~**3.25–3.3%**.
- MLP two-stage **0.9542**; ensemble **0.9529** (not champion).
- Published RF bar remains **0.9864** even though strengthen sweep hit ~**0.9885** (`rf_teacher_strengthen.json`).

### Laptop latency (CURRENT, MED–HIGH; ranges required)

- Custom CUDA FP16 derived pipeline **594–675 µs** (multi-session composition; not a single lucky point).
- Framework ranges as README table (Eager 2050–2247, compile 1519–1777, TRT 2427–2966, ORT GPU 3862–4652, ORT CPU 487–699).
- Speedup ranges vs TRT/compile/eager as README **only if** you add Option A caveats (below).
- Block 3 FP16 **532–602 µs**; cuDNN baseline **784 µs** n=50; local beats-cuDNN **1.30x–1.47x** as **RTX 3050 / this protocol** finding.
- Measurement stability: WSL2 session drift documented; within-session CV understates uncertainty.
- ORT CPU: significance **not robust** across sessions.

### LLM / streaming / energy (CURRENT)

- Dispatch overhead **16.60 µs p99** (n=5000); generation multi-second async; quality illustrative.
- Streaming **~25,899** flows/sec batch=128 RTX 3050.
- Energy ~**0.79 mJ/flow** RTX; ~**1.089 mJ/flow** A100; cuML RF more energy-efficient on A100 (**0.048 mJ**); CNN uses far less VRAM (**~2 MB vs 444 MB**).

### Fidelity (CURRENT)

- Export path bit-identical n=10; 6/6 CUDA self-checks PASS at disclosed tolerances (FP16 5e-2).
- Note: CUDA contract is **V2 last-timestep**, not full V3 attention path.

### DICC (LEGACY only)

- June 21, 2026 single-shot: V100S pipeline total **~551 µs** (550.664), A100 **~592 µs** (592.044).
- Jobs 363046 / 363047; validation PASSED in summaries.
- **CUDA kernels only**; no same-GPU PyTorch that day; **not** multi-day; **not** compare-gated.

### Process / tooling

- `verify_claims.py`: 66 claims pass, 0 regressions (this audit).
- Multi-day campaign **scripted** (`dicc_scripts/run_campaign.sh`) but **not completed** on laptop artifacts.
- Official site: **UM DICC**; Rostam = tooling trial only.

---

## Facts you must NOT state

1. **Any multi-day DICC means, CVs, or Day1/Day2 compare results** — artifacts **ABSENT**.  
2. **Full-pipeline Custom CUDA vs full PyTorch V3** as apples-to-apples speedup without heavy invalidation — **FORBIDDEN (Option A)**.  
3. That custom CUDA and eager V3 are "**the same computation**" (README still says this; it is **wrong**).  
4. V100/A100 "vs PyTorch" ratios from June 2026 (were cross-HW contaminated; removed).  
5. Fabricated/superseded: **5.19 µs** LLM, **2.76x** pipeline, **3.39x/3.15x** DICC, Sophimatics 2.7x, single-point **4.40x/3.33x/2.63x** as sole headlines without ranges.  
6. Champion F1 **0.9639** as current (SUPERSEDED).  
7. Published RF bar **0.9885** (diagnostic only).  
8. Rostam numbers as **UM official** or paper-final.  
9. Invented cluster numbers or SSH claims.  
10. That multi-day campaign is complete.

---

## Required caveats (every status report)

1. Systems/measurement paper; RF still higher accuracy.  
2. Laptop latency = **ranges** due to WSL2 session drift.  
3. Custom CUDA implements **subset** of production V3 (no attention/LN/GAP; last-timestep).  
4. Pipeline total is **derived/additive** (B1+B2+B4 chain + B3 separate), not necessarily true full device pipeline.  
5. June DICC 551/592 = **legacy single-shot**.  
6. Multi-day + same-GPU PT = **pending** (ops/Prof).  
7. Local "beats cuDNN" may **not** hold on server GPUs (Rostam Day1 provisional opposite direction — confirm on UM).  
8. ORT CPU not a clean "beats all frameworks" win.  
9. LLM explanations illustrative; dispatch is the measured systems result.  
10. Some claim-source JSONs are **gitignored** (repro packaging risk).

---

## Sources by paragraph topic

| Topic | Primary sources |
|-------|-----------------|
| Accuracy champion | twostage_botiot.json; md5sum; train_twostage.py |
| RF / gap | rf_baseline_processed.json; rf_baseline_processed.py |
| KD path | distill_botiot_*.json; f98bf33; 2560348 |
| Framework ranges | statistical_significance_v2.json + verify_claims HIST/S3A/EXTRA; bd3777e; d9e1f79 |
| Block 3 | cuda_kernel_stats_rtx3050.json; pytorch_block3_stats; d85271c; 3eb773a |
| LLM | llm_explainability.json; 09b509b |
| Streaming/energy | streaming_throughput.json; energy_efficiency.json; a100_energy.json |
| Fidelity | numerical_fidelity.json; 3f99243 |
| Legacy DICC | dicc_v100_summary.txt; dicc_a100_summary.txt; dd923c6 |
| Option A / invalid | DESIGN_PLAN.md §5; FINAL_PLAN.md §1 |
| Planned work | FINAL_PLAN P0–P5; HANDOFF |
| Re-verification history | 05_GIT_REVERIFICATION.md; commits listed there |

---

## Open questions for human

1. DICC campaign executed under OnDemand VNC + screen + batch (`docs/DICC_OPS_METHOD.md`)? 
2. When will Day1+Day2 SUCCESS land on laptop?  
3. Should report-writer fix Option A wording in README as part of P2, or only in email prose?  
4. Disclose Rostam provisional B3 loss in status update or wait for UM confirm?  
5. Force-add gitignored claim JSON to repo (U1) before sharing with coauthors?

---

## Multi-day DICC cells

| Cell | Value |
|------|-------|
| Day1 V100 SUCCESS | **EMPTY / ABSENT** |
| Day2 V100 SUCCESS | **EMPTY / ABSENT** |
| Day1 A100 SUCCESS | **EMPTY / ABSENT** |
| Day2 A100 SUCCESS | **EMPTY / ABSENT** |
| compare accept | **NOT RUN** |
| Block3 CUDA vs PT same-GPU cluster | **EMPTY** |
| Full V3 PT absolute cluster | **EMPTY** |

Use **legacy** 551/592 **only if labeled legacy single-shot CUDA-only**.

---

## Suggested report skeleton (for writer LLM — not polished email)

1. Goal / Option A scope  
2. Local accuracy frozen numbers + sources  
3. Local latency ranges + measurement stability  
4. LLM dispatch  
5. Legacy DICC + multi-day **pending**  
6. Threats-to-validity (RF gap, drift, architecture parity, Rostam risk)  
7. Next ops step (P0/P1)  
8. Appendix: raw tables from `09_RAW_NUMBER_TABLES.md`
