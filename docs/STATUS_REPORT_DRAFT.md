# COLIDE — Full Interim Status Report

> **SUPERSEDED for current status (2026-08-12).**  
> Living authority: `docs/PRE_MANUSCRIPT_CLOSURE.md`, `docs/DICC_EXTRACTION_TABLES.md`, `HANDOFF.md`.  
> DICC multi-session SUCCESS **exists** on laptop; this draft’s “ABSENT / blocker” sections are **historical**.

**Audience:** Prof. Dr. Por Lip Yee (PI); internal status; feedstock for a short email or slides  
**Draft date:** 2026-07-18 (full depth expansion)  
**Repo / audit HEAD:** evidence pack at `803c157…`; report commits on `master` thereafter  
**Authority:** `docs/DESIGN_PLAN.md` (**Option A**, approved 2026-07-14) + `docs/FINAL_PLAN.md`  
**Evidence feedstock:** `docs/audit/` (forensic pack, 2026-07-18) — **not** freehand README invention  
**Non-authoritative prior draft:** `docs/PROF_POR_STATUS_REPORT.md` (contingency; do not treat as truth)

---

## Document map (how to use this file)

| Section | Purpose |
|---------|---------|
| **0** | Executive summary (1 page) |
| **1** | Goal, venue, Option A claim scope |
| **2** | **Everything done** — timeline + work packages + frozen numbers |
| **3** | Current **blocker** (where interim status stops) |
| **4** | **Leftover** work — blocked on DICC and after unblock (P0–P3) |
| **5** | **Before final manuscript** — must / should / stretch (P4–P5 + optional) |
| **6** | Threats to validity & honesty rules |
| **7** | Short ask for Prof |
| **Appendix A–D** | Empty multi-day cells, KD table, raw number anchors, source paths |

**Scope rule:** This report goes **through the current DICC blocker**. Multi-day cluster means are **EMPTY**. Remaining cluster work is **leftover (blocked on DICC)**. Manuscript writing (P4) is **deferred** until after multi-day numbers exist and pass the match gate.

---

## 0. Executive summary

COLIDE is a **systems / measurement** project for **custom CUDA inference** of a CNN–BiLSTM IoT IDS classifier, with multi-framework latency protocol and low-overhead LLM dispatch for explanations. It is **not** an accuracy-SOTA paper.

**Done (local, early July 2026 arc + freeze):**

- Production two-stage model **macro-F1 0.9790** (checkpoint md5 frozen); RF apples-to-apples **0.9864** → honest gap **0.74%** (we do **not** beat RF).
- Laptop (WSL2 RTX 3050) latency reported as **multi-session ranges**, not lucky points; Block 3 FP16 **532–602 µs** vs cuDNN-style baseline **784 µs** locally.
- Real LLM dispatch **16.60 µs p99**; streaming ~**25,899** flows/s; energy and cuML comparison tables sourced.
- Numerical fidelity: export bit-identical; **6/6** CUDA self-checks PASS (CUDA contract = last-timestep, not full V3).
- Extensive claim hygiene: fabricated/unsourced numbers removed; `verify_claims.py` green (**66 pass / 0 fail** this audit).
- **Option A** locked: **per-block** Custom CUDA vs matching PyTorch only; **no** full-pipeline Custom CUDA vs full V3 as apples-to-apples speedup; **no** “same computation” language.
- Multi-day campaign **tooling** ready (`dicc_scripts/run_campaign.sh`); June 2026 V100/A100 totals kept only as **LEGACY single-shot**.

**Blocker:**

- Official **multi-day UM DICC** campaign (Day1 + Day2 + compare) **not completed**.  
- Path `benchmarks/results/dicc/` on laptop → **ABSENT**.  
- **Ops method locked:** OnDemand **VNC Desktop** + **`screen`** + batch `run_campaign.sh` (`docs/DICC_OPS_METHOD.md`). Campaign execution pending.

**Leftover (critical path):** OnDemand VNC setup → Day1/Day2 SUCCESS → artifacts home → extract → codebase numbers match → multi-day cells + Prof update.  
**Local science + local-complete manuscript:** already done; only multi-GPU/portability cells remain.

---

## 1. Goal, venue, and claim scope (Option A — locked)

### 1.1 Scientific / systems goal

Build and measure a **hand-written CUDA inference path** for a production-style CNN–BiLSTM flow classifier (BoT-IoT primary; ToN-IoT secondary), compare fairly to framework stacks, and integrate **async LLM explanations** with measured dispatch cost.

### 1.2 Venue framing

- **Primary lean:** FGCS-style **systems / measurement** paper (CUDA kernels + protocol + multi-platform latency + LLM dispatch).  
- **Accuracy** supports deployment narrative but is **not** the headline (RF remains higher).  
- IoT journal only if application/deployment is written carefully without overclaiming SOTA F1.

### 1.3 Option A (approved strategy)

| Rule | Detail |
|------|--------|
| **Valid claims** | Per-block Custom CUDA vs **matching** PyTorch modules (flagship: **Block 3 BiLSTM**, last-timestep contract) |
| **Also valid** | Intra-CUDA optimization progression (naive → FP16); absolute full-V3 framework latencies (eager/compile/TRT/ORT) as “what production pays” |
| **Forbidden** | Full-pipeline Custom CUDA vs **full** PyTorch V3 as apples-to-apples speedup |
| **Forbidden language** | Custom CUDA and eager V3 are “**the same computation**” — they are **not** |
| **Why invalid** | V3 has MultiheadAttention + residual LayerNorm + global average pool after BiLSTM; **no CUDA kernel implements these**. CUDA uses **last timestep**; V3 uses **mean over sequence after attention**. `fused_pipeline.cu` times B1+B2+B4 and adds B3 as a **separate timed addend**, not true B3→B4 device dataflow. Harness already sets `full_pipeline_cuda_vs_pytorch.valid = false`. |
| **Accuracy honesty** | RF **0.9864** > CNN **0.9790**; do not claim beating RF / SOTA |
| **Champion freeze** | `model/best_model_botiot_twostage.pth` md5 **`80a90f7cc210276300eaa90173a5a385`** — **no training, no clobber** |
| **Official cluster** | **UM DICC only** (not Rostam) |
| **June 551/592 µs** | **LEGACY single-shot** only |
| **Invented numbers** | Never |

### 1.4 PI / ops roles (as in FINAL_PLAN + DICC_OPS_METHOD)

- **PI:** Prof. Dr. Por Lip Yee  
- **Operator:** User account holder via **DICC OnDemand VNC + screen + batch**  
- **Agent:** laptop-side compare/claims only — no DICC login, no invented numbers

---

## 2. Everything done (through the blocker)

This section is the bulk of the interim report: **work packages + frozen results**. Multi-day official UM campaign is **not** included as complete.

### 2.1 Timeline overview

| Period | Theme | Outcome |
|--------|--------|---------|
| **~2026-06-21** | Historical UM single jobs | V100S ~**551 µs**, A100 ~**592 µs** pipeline totals (CUDA-only summaries) — now **LEGACY** |
| **~2026-07-01** | Provenance crisis / claim hygiene | Fabricated LLM overhead, unsourced speedups, bad citation, wrong weight export path **found and fixed**; verifier + CUDA stats harness introduced |
| **~2026-07-01–02** | Accuracy + latency re-grounding | KD path → **0.9790**; RF **0.9864** sourced; Block 3 cuDNN **784 µs**; race fix; multi-session **ranges** |
| **~2026-07 mid** | Local sessions S4–S7; cluster tooling | Ensemble/RF strengthen (champion kept); fidelity; DICC `run_campaign` hardened; Rostam Day 1 **tooling trial** |
| **2026-07-14** | Design freeze | **Option A approved**; Prof 3-day playbook written; multi-day UM still not started on hardened stack |
| **2026-07-17** | Final plan | P0–P5 phases; **numbers-match hard gate** before final Prof multi-day email |
| **2026-07-18** | Evidence audit + this report | Full forensic feedstock `docs/audit/`; interim status through DICC blocker |

### 2.2 Work package inventory — accuracy / training

| Work | Status | Outcome |
|------|--------|---------|
| BoT-IoT preprocessing (per-flow v2) | **Done** | Windowed pipeline abandoned; processed features shared with RF baseline |
| Two-stage training: KD + real-data fine-tune | **Done / frozen** | Test macro-F1 **0.9790** |
| Stage-1 KD winner | **Done** | α=**0.6**, T=**10**, focal γ=**2** → test **0.9763** |
| Stage-1 bit-identical repro | **Done** | Repro JSON macros match winner |
| Focal γ ∈ {1, 3, 4} | **Done (negative)** | Did not displace γ=2 path; champion unchanged |
| KD grid (14+ configs) | **Done** | Includes outlier a=0.7 T=10 test **0.9033** (negative cell kept) |
| sklearn RF apples-to-apples | **Done** | Test macro-F1 **0.9864** on same processed features |
| RF teacher strengthen sweep | **Done (diagnostic)** | Best ~**0.9885**; **published bar kept 0.9864** |
| Ensemble KD path | **Done (not champion)** | Val-F1 bug fixed (S5); ensemble F1 **0.9529** |
| Balanced-RF KD (S6) | **Skipped** | Low expected value |
| MLP ablation | **Done** | Two-stage MLP **0.9542** (faster, not champion) |
| ToN-IoT clean | **Done** | CNN **0.9526**, RF **0.9851**, gap ~**3.25–3.3%** |
| Weight export for CUDA | **Done** | Re-exported against 0.9790; real-weight validation path updated |
| Superseded two-stage **0.9639** | **Superseded** | Must not be stated as current |

#### Frozen accuracy numbers (with sources)

| Metric | Value | Source path | Confidence |
|--------|------:|-------------|------------|
| Champion test macro-F1 | **0.9790** | `benchmarks/results/twostage_botiot.json` | HIGH |
| Champion md5 | **`80a90f7cc210276300eaa90173a5a385`** | `model/best_model_botiot_twostage.pth` | HIGH (disk-confirmed audit) |
| RF test macro-F1 | **0.9864** | `rf_baseline_processed.json` | HIGH |
| Gap RF − CNN | **0.74%** | computed | HIGH |
| Stage-1 KD test F1 | **0.9763** | `distill_botiot_a0.6_T10.0_focal2.json` | HIGH |
| MLP two-stage | **0.9542** | `mlp_twostage.json` | HIGH |
| Ensemble | **0.9529** | `ensemble_distill.json` | HIGH (not champion) |
| ToN clean CNN / RF | **0.9526** / **0.9851** | `toniot_clean_*` | HIGH |
| RF strengthen best | ~**0.9885** | `rf_teacher_strengthen.json` | HIGH as **diagnostic only** |

**Honesty line for every status surface:** We **do not** claim beating RF or SOTA accuracy. Systems contribution is the lead.

### 2.3 Work package inventory — CUDA / systems (local)

| Work | Status | Outcome |
|------|--------|---------|
| Blocks 1–4 fused CUDA kernels | **Done** | Standalone binaries; FP32 + Block 3 FP16 (half2) + naive ablation |
| Naive Block 3 data race | **Fixed** | Was real shared-memory race (not “FP32 rounding”); race-fixed validation |
| Block 3 optimization progression | **Done** | Naive → transpose → graphs → FP16; progression ratios as **ranges** (~7.55×–9.50×) |
| cuDNN / PyTorch Block 3 baseline | **Done** | Mean **784 µs**, n=50 (`pytorch_block3_stats_rtx3050.json`) |
| Multi-session framework latency | **Done** | Ranges over WSL2 session drift; two-sample Welch design |
| Custom CUDA derived pipeline total | **Done as ranges** | **594–675 µs** multi-session composition (derived/additive) |
| Measurement stability narrative | **Done** | Within-session CV understates uncertainty; **ranges required** |
| LLM async dispatch | **Done** | p99 **16.60 µs** (n=5000); generation multi-second / illustrative quality |
| Streaming throughput | **Done** | ~**25,899** flows/s batch=128 RTX 3050 |
| Energy (RTX / A100) | **Done (single-protocol)** | ~**0.79** / ~**1.089** mJ/flow |
| cuML RF resources | **Done** | Higher throughput/energy efficiency on A100; **much higher VRAM** (~444 MB vs ~2 MB) |
| torch.compile BiLSTM crash note | **Documented** | Methods / limitations material |

#### Frozen laptop latency ranges (WSL2 RTX 3050, batch=1 protocol)

| Path | Latency (µs) | Notes |
|------|-------------:|-------|
| Custom CUDA FP16 **derived** pipeline | **594–675** | Implemented blocks composition; not full V3 |
| Eager PyTorch (full V3) | **2,050–2,247** | Production model absolute |
| torch.compile | **1,519–1,777** | Framework absolute |
| TensorRT | **2,427–2,966** | Framework absolute |
| ONNX Runtime GPU | **3,862–4,652** | Framework absolute |
| ONNX Runtime CPU | **487–699** | Significance **not robust** across sessions |
| Block 3 FP16 | **532–602** | Flagship per-block |
| Block 3 vs cuDNN local | **1.30×–1.47×** | **RTX 3050 / this protocol only** — portability unconfirmed on UM |

**Option A framing for any ratio involving custom CUDA:** Framework columns time **full V3**. Custom totals are **not** the same computation. Do not lead abstract/status with “full-model Custom CUDA × over full V3” as parity-safe claims. If range ratios vs TRT/eager/compile are mentioned, they must carry that caveat (range math under incomplete CUDA scope).

### 2.4 Work package inventory — fidelity, provenance, process

| Work | Status | Outcome |
|------|--------|---------|
| Export numerical fidelity | **Done** | Bit-identical max abs **0** (n=10) |
| CUDA self-checks | **Done** | **6/6 PASS** at disclosed tols (FP16 **5e-2**) |
| Threats-to-validity draft (S7) | **Done** | Paper text blocks + fidelity table |
| `verify_claims.py` | **Done / green** | 66 pass, 0 fail, 0 regressions (audit); 5 bold numbers uncovered (0.6, 1.00x, 10.0, 2.0, 3.3%) |
| Fabricated LLM **5.19 µs** | **Fixed** | Real p99 **16.60** |
| Unsourced pipeline **2.76×** / cross-HW DICC ratios | **Fixed / removed** | No June vs-PyTorch ratios |
| One-sample significance abuse | **Fixed** | Two-sample Welch |
| Stale pre-KD weight export | **Fixed** | Re-export for 0.9790 |
| Fabricated Sophimatics citation | **Fixed** | Replaced with Ibrahim et al. (CN 2026) with honest caveats |
| RF 0.9864 uncorroborated | **Fixed** | Traced to real script/JSON |
| Branch unify `final-polish` → `master` | **Done** | Single canonical branch |
| Session lifecycle (verify → commit → push → next prompt) | **Standing process** | User-confirmed |
| One major package ≈ one chat | **Standing process** | After rushed multi-session lesson |
| Deep evidence audit pack | **Done (2026-07-18)** | `docs/audit/00`–`11` |

**Verifier limitation:** Green `verify_claims` checks **number presence** in docs, **not** Option A construct validity. README can still contain invalid “same computation” framing while claims pass.

**Packaging risk:** `.gitignore` ignores `benchmarks/results/`. Several load-bearing claim JSONs (champion F1, RF, CUDA stats, B3 PT, fidelity, round-2 KD) exist **on disk only**. Clean GitHub clone cannot regenerate all README numbers without those files or re-runs.

### 2.5 Work package inventory — cluster tooling (not multi-day results)

| Work | Status | Outcome |
|------|--------|---------|
| Portable DICC/Rostam campaign stack | **Done (tooling)** | `run_campaign.sh`, setup, submit, n=100 CUDA stats, same-GPU PyTorch harness, compare gate, SUCCESS layout |
| `run_campaign --full` | **Scripted** | Day1+Day2+compare one-shot path exists |
| Site-portable SLURM fixes | **Done** | Partitions, GRES optional, multi-GPU pin, compile on GPU node, etc. |
| Rostam Day 1 | **Tooling trial only** | SUCCESS dirs on Rostam; **not** UM official / paper-final |
| Rostam provisional B3 direction | **Risk note only** | Approx. CUDA slower than PT on V100/A100 peek — **must confirm/refute on UM** |
| Historical UM June 2026 | **LEGACY files only** | ~551 / ~592 µs; CUDA-only; no multi-day; no same-GPU PT |
| UM path discovery | **Partial** | Login known; tree at `/home/user/ibteshamulhaque/colide`; git fetch freezes; prefer **tarball/scp** from laptop |
| Official multi-day on hardened stack @ UM | **NOT DONE** | **Blocker** |

#### Legacy June 2026 (cite only if labeled)

| GPU | Pipeline total (µs) | Job | Label |
|-----|--------------------:|-----|-------|
| V100S | **550.664** (~551) | 363046 | LEGACY single-shot, CUDA-only |
| A100 | **592.044** (~592) | 363047 | LEGACY single-shot, CUDA-only |

Sources: `benchmarks/results/dicc_v100_summary.txt`, `dicc_a100_summary.txt`. Validation PASSED in those summaries. **Not** multi-day; **not** vs-PyTorch.

### 2.6 Local session roadmap (S4–S9) — closed / skipped

| Session | Status |
|---------|--------|
| S4 baseline + multi-session roadmap + one-chat rule | **CLOSED** |
| S5 ensemble Val-F1 fix + RF teacher strengthen | **CLOSED** (champion kept) |
| S6 balanced-RF KD | **SKIPPED** (low EV) |
| S7 threats-to-validity + numerical fidelity | **CLOSED** |
| S8 batch-size note | **SKIPPED** |
| S9 simple DICC_RUNBOOK | **SUPERSEDED** by `dicc_scripts/` + `run_campaign.sh` |

### 2.7 Other systems numbers (local anchors)

| Metric | Value | Source |
|--------|------:|--------|
| LLM dispatch p99 | **16.60 µs** (n=5000) | `llm_explainability.json` |
| Streaming max (batched) | **~25,899** flows/s | `streaming_throughput.json` |
| Energy RTX | **~0.79 mJ/flow** | `energy_efficiency.json` |
| Energy A100 | **~1.089 mJ/flow** | `a100_energy.json` |
| A100 CNN throughput (derived) | **~87,791** flows/s | from A100 energy timing |
| cuML VRAM / thr / energy | **444 MB** / **~2.07e6** / **0.048 mJ** | `cuml_rf_resources.json` |

---

## 3. Current blocker (where this interim report stops)

| Item | Detail |
|------|--------|
| **Blocker name** | Multi-day **UM DICC** campaign incomplete; artifacts not on laptop |
| **Path checked** | `benchmarks/results/dicc/` → **ABSENT** |
| **Why** | Campaign not yet executed under locked ops method (OnDemand VNC + screen + batch). Earlier interactive SSH/`srun` path was fragile; campus-runner/Cheran defaults **superseded**. |
| **What is ready despite blocker** | Campaign scripts, Option A, local science closed, local-complete manuscript, claims package, this interim report |
| **What is not ready** | Day1/Day2 SUCCESS means, compare accept, same-GPU PT on cluster, manuscript §5.13 multi-day cells |

### Multi-day cells (intentionally EMPTY)

| Cell | Status |
|------|--------|
| Day1 V100 SUCCESS | **EMPTY / ABSENT** |
| Day2 V100 SUCCESS | **EMPTY / ABSENT** |
| Day1 A100 SUCCESS | **EMPTY / ABSENT** |
| Day2 A100 SUCCESS | **EMPTY / ABSENT** |
| `compare_dicc_sessions.py` | **NOT RUN** |
| Block 3 CUDA vs PT same-GPU (cluster) | **EMPTY** |
| Full V3 PT absolute (cluster) | **EMPTY** |

**We do not invent multi-day means, CVs, or Day1/Day2 compare results.**

---

## 4. Leftover work (blocked on DICC → after unblock)

These are **planned remaining tasks** from `FINAL_PLAN` / audit `07`. They are **leftover due to the DICC block** (or sequential after it). **Not done in this interim report.**

### 4.1 Critical path (must for multi-day numbers + final numbered email)

| ID | Phase | Work | Depends on | Clock (plan) |
|----|-------|------|------------|--------------|
| **L1** | **P0** | OnDemand VNC Desktop + `screen` + sync tree + verify champion md5 | Human | now |
| **L2** | **P0 exit** | ≥1 GPU class Day1 SUCCESS (ideally V100 **and** A100) | L1 | queue-bound |
| **L3** | **P1** | Day2 SUCCESS same GPU class(es); `compare_dicc_sessions.py` accept (or document reject) | L2 | ~2–5 days wall typical |
| **L4** | **P1** | scp entire `benchmarks/results/dicc/` to laptop | L3 | hours |
| **L5** | **P2a** | Extract from JSON only: Block 3 CUDA vs PT same GPU; absolute full V3 PT; **no** invalid full CUDA/full V3 ratio; fill `PROF_POR_3DAY` §4 | L4 | ~0.5 day |
| **L6** | **P2b** | Codebase-wide numbers match (README, docs, claims, email/slides) | L5 | ~1–2 days if many stale strings |
| **L7** | **P2c** | `PYTHONPATH=. python3 scripts/verify_claims.py` green | L6 | minutes |
| **L8** | **P2d** | Final Prof update with **local + multi-day** locked numbers | L7 | after gate |

**Hard gate:** Do **not** send a final multi-day numbers email until **L6–L7** pass.  
**Total after unblock (plan):** ~**5–7 days typical** (buffer ~1–1.5 weeks) for P0→P2.

### 4.2 Operator run card (for when executing)

**Full method:** `docs/DICC_OPS_METHOD.md`  
**Connection:** OnDemand → VNC Desktop → `screen -S colide` (not long interactive `srun`/`salloc` over VPN).

```bash
# Inside OnDemand VNC + screen:
# Tree: clone OR tar xzf colide-master-for-dicc.tar.gz
cd ~/colide

python3 -m venv .venv-cluster
source .venv-cluster/bin/activate
pip install -U pip
pip install 'numpy>=2.0.0' 'scipy>=1.13.0' 'pyyaml>=6.0' 'scikit-learn>=1.5.0'
pip install --upgrade 'torch>=2.5.0,<2.7' --index-url https://download.pytorch.org/whl/cu121
md5sum model/best_model_botiot_twostage.pth   # expect 80a90f7…

export COLIDE_V100_PARTITION=gpu-v100s
export COLIDE_A100_PARTITION=gpu-a100
export COLIDE_SBATCH_GRES=gpu:1

bash dicc_scripts/run_campaign.sh              # Day 1
bash dicc_scripts/run_campaign.sh --day 2      # Day 2
# Deliverable: entire benchmarks/results/dicc/
```

Prefer partitions over fixed nodelists; prefer venv over conda under `set -u`; prefer **batch** for Day1/Day2. Full notes: `docs/DICC_OPS_METHOD.md`, `dicc_scripts/README.md`.

### 4.3 Contingencies (from FINAL_PLAN)

| Situation | Action |
|-----------|--------|
| VPN/SSH drops during setup | OnDemand VNC keeps desktop; reattach `screen` |
| Interactive allocation lost | Do not use long `srun`/`salloc` as primary; batch campaign instead |
| Only Day 1 finishes | Provisional single-day DICC + local pack; **label clearly**; still numbers-match before multi-day claims |
| Queue blocked entire window | Local pack + June **legacy** + “campaign in flight”; match local numbers |
| Compare rejects | Report both days; no “stable multi-day” claim until fixed/explained |
| One partition missing | Document; publish the GPU that completed |

### 4.4 Pre-manuscript residual (P3) — after L8

| Item | Notes |
|------|-------|
| Threats-to-validity one-pager aligned with final numbers | Much already in S7 / this report |
| Internal doc hygiene | Comments, banners on non-auth drafts |
| Confirm no training / champion touch | Standing rule |
| **Clock** | ~0.5–1 day if P2 thorough |

---

## 5. Before final manuscript writing — significant suggestions

**Manuscript (P4) is explicitly deferred until pre-manuscript is done** (multi-day pack + match gate + Prof numbered update). Below is the ordered “what matters before you write the paper,” split into **must / should / stretch / optional auditor**.

### 5.1 MUST before honest multi-day + final evidence pack (planned)

1. Complete **L1–L8** (UM multi-day + extract + numbers match + verify).  
2. Keep **Option A** in every public surface (README abstract, paper_text_blocks, email).  
3. Confirm or refute **portable “beats cuDNN”** on UM Block 3 same-GPU (Rostam provisional opposite direction is a **top science risk**).  
4. Never restore cross-hardware June “vs PyTorch” ratios; never invent multi-day cells.

### 5.2 SHOULD before freezing abstract / paper tables (high value)

| ID | Suggestion | Why |
|----|------------|-----|
| **U3** | Fix residual “same computation” / full-pipeline framing in README | Truthfulness; verify_claims won’t catch this |
| **U13** | Split tables: (a) per-block CUDA vs matching PT; (b) full V3 framework absolutes | Option A-safe abstract |
| **U1** | Force-add curated claim-source JSONs or publish checksum manifest | Clean clone / coauthor repro |
| **U8** | Footnote additive pipeline / B3 not true device chain **everywhere** ratios appear | Methods honesty |
| **U14** | Treat UM Block 3 CUDA vs PT as decide-to-publish gate for “beats cuDNN” portability | Avoid portable overclaim |
| P3 | Threats-to-validity one-pager locked to final numbers | Reviewer defense |

### 5.3 Manuscript spine when allowed (P4) — plan, not started

**Contribution spine (FINAL_PLAN / DESIGN_PLAN):**

1. Custom CUDA kernels for CNN–BiLSTM IoT IDS inference (**Block 3 focus**).  
2. Measurement methodology: multi-session local **ranges** + UM multi-day DICC + same-GPU PyTorch.  
3. On-device LLM explainability path with **measured** dispatch overhead.  
4. Honest limits: RF accuracy edge; **no** full V3 CUDA parity yet (unless Option B later).

**Minimum figures/tables:**

- Per-block latency (laptop + DICC when available)  
- Block 3 CUDA vs PT (valid head-to-head)  
- Multi-day stability (when L3 accepts)  
- Accuracy vs RF (0.9790 vs 0.9864)  
- LLM dispatch overhead  

**Clock (plan):** ~**1.5–3 months** calendar for solid FGCS-leaning draft + PI feedback (separate from P0–P3).

**Writing role:** User (and any writing collaborators assigned by PI); cluster ops are the account holder via OnDemand VNC.

### 5.4 STRETCH after pre-manuscript (P5) — optional, not v1 path

| Item | When / note |
|------|-------------|
| Nsight / short bottleneck note V100 vs A100 | If time after P3 |
| Full TRT/ORT/compile matrix on DICC | Only if reviewers demand |
| **Option B** — implement attention/LN/GAP on CUDA for true full-model parity | Large; only if full-pipeline claim is mandatory |
| Retrain to close RF gap | **Out of scope** for current freeze; do not clobber champion casually |

### 5.5 Other optional auditor items (hygiene; not FINAL_PLAN gates)

| ID | Item |
|----|------|
| U2 | Persist multi-session latency JSON arrays (stop hardcoding HIST/S3A/EXTRA in verify_claims) |
| U4 | Multi-session ranges for B1/B2/B4 (README point values drift vs live stats) |
| U5 | Fix stale F1 labels inside cuml JSON (0.9639/0.9601) |
| U9 | Strengthen energy methodology docs / multi-run CIs |
| U11 | Stage-2 bit-repro **or** stop implying bit-identity (do not clobber champion) |
| U12 | GitHub Action: `verify_claims` on PR |
| U15 | Banner/delete non-authoritative `PROF_POR_STATUS_REPORT.md` after this pack is canonical |
| U16 | Tag every results JSON with git_sha, host, GPU, timestamp, n_trials |
| U17 | Expand claim manifest for remaining bold numbers / weak fields |
| U18 | `validate_dicc_tree.py` SUCCESS schema checker |

**If only three optional items before abstract freeze:** **U3 → U1 → U13**.

### 5.6 Explicit non-goals until multi-day + match done

- Full manuscript drafting as if cluster numbers exist  
- Raising published RF bar to 0.9885  
- Treating Rostam as UM paper-final  
- Option B CUDA port (unless forced later)  
- Retraining champion without backup + explicit decision  

---

## 6. Threats to validity (status-facing summary)

1. **Accuracy second to RF** (0.9864 vs 0.9790) — systems paper framing required.  
2. **WSL2 session drift** — laptop latency must stay **ranges**.  
3. **Architecture parity** — custom CUDA ≠ full V3; no “same computation.”  
4. **Derived pipeline totals** — additive composition; B3 not necessarily true device chain into head.  
5. **Multi-day DICC absent** — no stability claim on cluster yet.  
6. **Portable beats-cuDNN unproven** — local only; Rostam provisional risk.  
7. **ORT CPU** — non-robust significance.  
8. **LLM generation quality** — illustrative; systems result is dispatch.  
9. **Gitignored claim sources** — packaging/repro risk for coauthors.  
10. **June 551/592** — legacy single-shot only.  
11. **Stage-2 not bit-reproved** — Stage-1 is; Stage-2 uses same discipline but not empirically bit-identical.  
12. **Energy / streaming** — single-protocol artifacts (not multi-session ranges).

---

## 7. Short ask (if this becomes a Prof email)

1. Local accuracy, latency ranges, fidelity, claim hygiene, and Option A scope are **frozen and ready**.  
2. Early July delivered the bulk of scientific and systems evidence; mid-July locked design and tooling.  
3. Remaining **critical path** is multi-day **UM DICC** execution: OnDemand VNC + Day1+Day2+compare → extract → numbers match → §5.13 + final multi-day update.  
4. Ops method locked: **OnDemand VNC Desktop + screen + batch `run_campaign.sh`** (`docs/DICC_OPS_METHOD.md`).  
5. Run card and tarball path prepared; no inventing cluster numbers.  
6. Local-complete manuscript already exists; multi-GPU cells filled only after SUCCESS tree lands.

---

## Appendix A — Multi-day table template (empty until artifacts)

| GPU | Day1 CUDA B3 FP16 | Day1 PT B3 | Day2 CUDA B3 FP16 | Day2 PT B3 | Compare | Full V3 PT abs |
|-----|-------------------|------------|-------------------|------------|---------|----------------|
| V100S | — | — | — | — | NOT RUN | — |
| A100 | — | — | — | — | NOT RUN | — |

## Appendix B — KD sweep snapshot (winner + context)

| Config | α | T | γ | test_f1 | Role |
|--------|--:|--:|--:|--------:|------|
| a0.6 T10 focal2 | 0.6 | 10 | 2 | **0.9763** | Stage-1 winner (+ bit-repro) |
| a0.6 T10 focal1/3/4 | 0.6 | 10 | 1/3/4 | lower / not champion path | Negative γ sweep |
| a0.7 T10 focal2 | 0.7 | 10 | 2 | **0.9033** | Outlier negative |
| Two-stage FT after winner | — | — | — | **0.9790** | **Champion** |

Full grid: audit `09_RAW_NUMBER_TABLES.md` / on-disk `distill_botiot_*.json`.

## Appendix C — Dense number anchors (copy feedstock)

| metric | value | source | label |
|--------|------:|--------|-------|
| champion_macro_f1 | 0.9790 | twostage_botiot.json | CURRENT |
| champion_md5 | 80a90f7cc210276300eaa90173a5a385 | model ckpt | CURRENT |
| rf_test_macro_f1 | 0.9864 | rf_baseline_processed.json | CURRENT published bar |
| rf_gap_pct | 0.74 | computed | CURRENT |
| kd_stage1_f1 | 0.9763 | distill_…focal2.json | CURRENT |
| ton_clean_cnn_f1 | 0.9526 | toniot_clean_* | CURRENT |
| llm_overhead_p99_us | 16.60 | llm_explainability.json | CURRENT |
| streaming_gpu_batched_max | ~25899 | streaming_throughput.json | CURRENT |
| energy_rtx_mj_per_flow | 0.79 | energy_efficiency.json | CURRENT |
| energy_a100_mj_per_flow | 1.089 | a100_energy.json | CURRENT |
| pytorch_b3_mean_us | 784 | pytorch_block3_stats_rtx3050.json | CURRENT |
| cuda_b3_fp16_range_us | 532–602 | multi-session | CURRENT |
| cuda_pipeline_range_us | 594–675 | multi-session composition | CURRENT |
| fw_eager / compile / trt / ort_gpu / ort_cpu | 2050–2247 / 1519–1777 / 2427–2966 / 3862–4652 / 487–699 | multi-session | CURRENT |
| dicc_v100 / a100 pipeline | 550.664 / 592.044 | dicc_*_summary.txt | **LEGACY single-shot** |
| dicc_multiday_any | — | benchmarks/results/dicc/ | **ABSENT** |
| fidelity_export_max_abs | 0.0 | numerical_fidelity.json | CURRENT |
| fidelity_cuda_selfcheck | 6/6 PASS | numerical_fidelity.json | CURRENT |
| rf_strengthen_best | ~0.9885 | rf_teacher_strengthen.json | DIAGNOSTIC not published bar |
| verify_claims (audit) | 66 pass / 0 fail | scripts/verify_claims.py | CURRENT |

## Appendix D — Where everything lives on disk

| Role | Path |
|------|------|
| **This full report** | `docs/STATUS_REPORT_DRAFT.md` |
| Evidence audit pack (source of truth for claims) | `docs/audit/` (`00_INDEX.md` … `11_REPORT_WRITER_BRIEF.md`) |
| Strategy Option A | `docs/DESIGN_PLAN.md` |
| Phased plan P0–P5 | `docs/FINAL_PLAN.md` |
| 3-day playbook tables (fill after DICC) | `docs/PROF_POR_3DAY.md` |
| Non-authoritative old contingency draft | `docs/PROF_POR_STATUS_REPORT.md` |
| Session handoff / next-session prompts | `HANDOFF.md` |
| Campaign scripts | `dicc_scripts/` |
| Claim verifier | `scripts/verify_claims.py` |
| Champion checkpoint | `model/best_model_botiot_twostage.pth` |
| Result JSONs (many gitignored) | `benchmarks/results/` |
| Multi-day tree (missing) | `benchmarks/results/dicc/` **ABSENT** |

### How to generate a short email or slides from this file

1. **Email to Prof:** §0 + §3 + §7 (+ optional §2.1 timeline bullets).  
2. **Internal deep brief:** full document.  
3. **Slides:** §0 bullets; one slide frozen numbers (App C); one slide leftover L1–L8; one slide Option A.  
4. **After DICC lands:** do **not** paste invented numbers — run L5–L8 and replace Appendix A empties from JSON only.

---

*End of full interim report. Scope ends at the DICC blocker. Multi-day campaign work remains leftover until unblocked. Manuscript remains deferred until after L1–L8 and P3.*
