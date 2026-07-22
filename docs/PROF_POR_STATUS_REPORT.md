# COLIDE — Status Report for Prof. Dr. Por Lip Yee

**Date:** 2026-07-18  
**Student:** Ibteshamul Haque (`titoatwork`)  
**Project:** COLIDE — CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection  
**Target venue (working):** Future Generation Computer Systems (FGCS) — systems / measurement paper  
**Claim policy (locked):** **Option A** — valid **per-block** Custom CUDA vs PyTorch only; **no** full-pipeline Custom CUDA vs full PyTorch V3 speedup  

**Purpose of this document:** numbered status pack with frozen local results, clearly labeled legacy DICC figures, and an honest statement of what is still pending. This is **not** a finished manuscript.

**Numbers gate:** All headline figures below are checked against source JSON via `scripts/verify_claims.py` (green as of 2026-07-18). Champion checkpoint md5: `80a90f7cc210276300eaa90173a5a385`.

---

## 1. One-paragraph summary

COLIDE is a **systems / performance** project: hand-written CUDA C++ inference kernels for a trained CNN-BiLSTM IoT IDS, plus an async on-device LLM explainability path. Accuracy is **competitive but not SOTA** versus a same-split Random Forest (we disclose the gap). Laptop (RTX 3050 / WSL2) measurements show Custom CUDA faster than eager PyTorch, torch.compile, TensorRT, and ORT-GPU when reported as **multi-session ranges**. Cross-hardware CUDA pipeline figures exist from a **June 2026 single-shot** on UM DICC (V100S / A100); a **multi-day** campaign with same-GPU PyTorch baselines is scripted and ready. **Ops method (locked):** DICC **OnDemand → VNC Desktop** for interactive setup, **`screen`** for long terminal tasks, and **batch** `run_campaign.sh` for Day1/Day2 (avoids loss of interactive `srun`/`salloc` allocations on VPN drops). We do **not** invent or over-claim DICC multi-day numbers.

---

## 2. Accuracy (final local — frozen; no retraining)

| Metric | Value | Source |
|--------|-------|--------|
| Production model | Two-stage CNN-BiLSTM (KD + focal + fine-tune) | `model/best_model_botiot_twostage.pth` |
| Checkpoint md5 | `80a90f7cc210276300eaa90173a5a385` | laptop |
| BoT-IoT test macro-F1 | **0.9790** | `benchmarks/results/twostage_botiot.json` |
| CPU RF (same preprocessed splits) | **0.9864** | `benchmarks/results/rf_baseline_processed.json` |
| Gap to RF | **0.74%** | 0.9864 − 0.9790 |
| ToN-IoT clean macro-F1 | **0.9526** | `toniot_clean_comparison.json` (README-aligned) |

**Honest framing:** We do **not** claim beating RF or SOTA detection accuracy. The architecture is retained in part because its recurrent control flow stresses production compilers (e.g. torch.compile CUDA-graph path crashes on BiLSTM).

---

## 3. Latency & systems results (laptop — RTX 3050 / WSL2)

### 3.1 Framework comparison (ranges, not single lucky runs)

Multi-session measurement: framework side 3 sessions × 20 trials; Custom CUDA side 5 independent sessions. Session-to-session drift is real (up to ~14–27% on some configs); we report **ranges**.

| Method | Mean latency range (µs) | vs Custom CUDA (range) |
|--------|-------------------------|-------------------------|
| **Custom CUDA FP16** (block-sum pipeline) | **594–675** | **1.00×** |
| Eager PyTorch | 2,050–2,247 | **3.04–3.78×** |
| torch.compile | 1,519–1,777 | **2.25–2.99×** |
| TensorRT FP16 | 2,427–2,966 | **3.60–4.99×** |
| ORT GPU | 3,862–4,652 | **5.72–7.83×** |
| ORT CPU | 487–699 | 0.72–1.18× *(not robust; straddles parity)* |

Welch’s t-test: Eager / compile / TRT / ORT-GPU remain p<0.001 in all three framework sessions. **ORT CPU is not consistently significant** — do not headline “beats all frameworks including ORT CPU.”

### 3.2 Block 3 (BiLSTM) optimization

| Item | Number |
|------|--------|
| FP16 progression vs naive | **7.55–9.50×** |
| Final FP16 vs cuDNN-style block baseline | **1.30–1.47×** |
| Block 3 FP16 absolute (laptop) | **532–602 µs** (5 sessions) |

### 3.3 LLM explainability (on-device)

| Item | Number |
|------|--------|
| Model | TinyLlama-1.1B, 4-bit |
| Dispatch overhead p99 | **16.60 µs** (5,000 trials) |
| Generation (async, non-blocking) | ~7–8 s / alert (separate thread) |

Detection path never waits on LLM generation.

### 3.4 Streaming throughput

| Item | Number |
|------|--------|
| Peak GPU batched throughput | **25,899 flows/s** (batch=128, RTX 3050) |

### 3.5 Numerical fidelity

Export path bit-identical (n=10); six CUDA self-checks PASS (`numerical_fidelity.json`).

---

## 4. UM DICC (Universiti Malaya)

### 4.A Legacy single-shot (June 2026) — **not multi-day official**

These are **real** cluster measurements from jobs **363046** (V100S) and **363047** (A100), but:

- **Single day / single shot** only  
- **CUDA kernels only** — **no** same-hardware PyTorch baseline in that run  
- Therefore **no** honest “CUDA vs PyTorch on V100/A100” ratio yet  

| GPU | Job | Pipeline total (B1+B2+B3 FP16+B4) | Block 3 FP16 |
|-----|-----|-----------------------------------|--------------|
| **V100S-PCIE-32GB** | 363046 | **~551 µs** (550.664) | 511.852 µs |
| **A100-SXM4-80GB** | 363047 | **~592 µs** (592.044) | 548.368 µs |

**Observation (systems):** V100S is faster than A100 on this sequential BiLSTM-heavy path → consistent with **clock-bound**, not SM-count-bound, behaviour. Label as **preliminary / legacy** until multi-day replicate + PT baselines exist.

Source files: `benchmarks/results/dicc_v100_summary.txt`, `dicc_a100_summary.txt`.

### 4.B Multi-day campaign with PyTorch baselines — **pending**

| Item | Status |
|------|--------|
| Campaign scripts | Ready (`dicc_scripts/run_campaign.sh`, Day1 + Day2) |
| Laptop tarball | `~/colide-master-for-dicc.tar.gz` |
| Partitions (from prior jobs) | `gpu-v100s`, `gpu-a100`; GRES `gpu:1` |
| Day 1 SUCCESS (new campaign) | **Not completed** |
| Day 2 SUCCESS | **Not completed** |
| Same-GPU PyTorch Block 3 / full V3 absolute | **Not collected** |
| Block 3 CUDA / PyTorch ratio on V100 & A100 | **TBD** (only valid cluster speedup language under Option A) |
| Day1 vs Day2 stability compare | **TBD** |

**Ops method (current):** Run interactive setup via **DICC OnDemand VNC Desktop**; use **`screen`** for long tasks; submit Day1/Day2 with **`bash dicc_scripts/run_campaign.sh`** (batch). Do **not** rely on long interactive `srun`/`salloc` over VPN. Canonical: `docs/DICC_OPS_METHOD.md`.  
**Blocker remaining:** Campaign not yet executed — SUCCESS tree still absent on laptop (not a missing scientific design).

**What we will add after multi-day SUCCESS (no invention until then):**

| Metric | V100 Day1 | V100 Day2 | A100 Day1 | A100 Day2 |
|--------|-----------|-----------|-----------|-----------|
| CUDA Block 3 FP16 | — | — | — | — |
| PyTorch Block 3 (matched) | — | — | — | — |
| PyTorch full V3 (**absolute only**) | — | — | — | — |
| CUDA B3 / PT B3 ratio | — | — | — | — |

Full-pipeline Custom CUDA / full V3 PT ratio: **n/a** (invalid under Option A — V3 has attention / LayerNorm / GAP not in the CUDA path).

---

## 5. Scientific caveats (we are handling correctly)

1. **Option A:** Do not claim full-pipeline Custom CUDA vs full PyTorch V3 speedup until architecture parity exists.  
2. **Measurement stability:** Within-session CVs can look tight while **session-to-session** means swing 9–27%; all laptop headlines are **ranges**.  
3. **RF dual finding:** RF still wins accuracy; on A100, RF can also be more energy-efficient per flow while CNN-BiLSTM uses far less VRAM than a large GPU RF — both directions disclosed.  
4. **Legacy DICC ≠ multi-day DICC:** June 551 / 592 µs must stay labeled single-shot.  
5. **Rostam:** tooling trial only — not UM official results.

---

## 6. What is ready vs what needs you / cluster

| Ready now | Needs multi-day DICC execution |
|-----------|--------------------------------|
| Local science playlist closed (B14, WP6b, claims, local-complete manuscript) | V100/A100 Day1+Day2 SUCCESS dirs |
| Accuracy freeze + md5 | Same-GPU PyTorch baselines on DICC |
| Laptop multi-session latency/energy ranges (WP6b) | Block 3 CUDA vs PT ratios on cluster |
| LLM dispatch + structured XAI scope | Day1/Day2 stability accept/reject |
| Legacy June DICC CUDA totals (labeled) | Manuscript §5.13 fill from JSON only |
| Claim verification green (64 claims) | Optional Nsight / stretch work later |
| Ops method locked (OnDemand VNC + screen + batch) | — |

---

## 7. Decisions / status (ops)

1. **Connection method (locked):** DICC **OnDemand → VNC Desktop** for interactive setup; **`screen`** for long tasks; **batch** campaign for Day1/Day2. Avoid long interactive `srun`/`salloc` over VPN as the primary path.  
2. **Stale plans removed:** campus-stable runner / third-party operator as default — not required by current guidance.  
3. **Next action:** User executes campaign under `docs/DICC_OPS_METHOD.md`; agent ingests SUCCESS tree on laptop only.  
4. Venue emphasis remains FGCS-leaning **systems** framing (accuracy secondary to measurement + multi-objective honesty).

---

## 8. Email-ready short version (copy/paste)

> **Subject:** COLIDE — status pack (local results frozen; multi-day DICC pending execution)  
>  
> Dear Prof. Por,  
>  
> Please find a short numbered status for COLIDE. Contribution framing remains **systems/measurement** (custom CUDA inference + multi-platform protocol + on-device LLM dispatch), not SOTA accuracy.  
>  
> **Accuracy (protocol-era):** sealed multi-seed BoT **test** macro-F1 **0.9780±0.0033** (champion md5 `80a90f7c…` unchanged). Protocol LGBM val still tops pure F1 (**0.9818**). We do **not** claim beating every classical model on F1 alone.  
>  
> **Latency (laptop RTX 3050):** multi-session WP6b ranges — energy **0.920–0.943** mJ/flow; PT@256 **24.15–25.68** µs; Option A CUDA pipeline **565–570** µs. Historical framework ranges remain labeled separately.  
>  
> **Claim policy:** full-model Custom CUDA vs full PyTorch V3 speedup is **not** claimed. Valid head-to-head is **per-block**, especially **Block 3 (BiLSTM)**.  
>  
> **UM DICC:** June 2026 **legacy single-shot** CUDA pipeline totals — V100S **~551 µs**, A100 **~592 µs** (no same-GPU PyTorch baseline that day). Multi-day campaign + PyTorch baselines are scripted. Ops path: **OnDemand VNC Desktop + screen + batch `run_campaign.sh`** (so interactive allocation loss on VPN drops is avoided). Day1/Day2 SUCCESS pending execution.  
>  
> Full tables and sources: `docs/PROF_POR_STATUS_REPORT.md` / `docs/DICC_OPS_METHOD.md` in the COLIDE repo.  
>  
> Best regards,  
> Ibteshamul Haque  

---

## 9. Source map (for audit)

| Claim area | Artifact |
|------------|----------|
| F1 / RF | `twostage_botiot.json`, `rf_baseline_processed.json` |
| Framework / CUDA ranges | `README.md` + `cuda_kernel_stats_rtx3050.json` (multi-session) |
| LLM | `llm_explainability.json` |
| Streaming | `streaming_throughput.json` |
| Legacy DICC | `dicc_v100_summary.txt`, `dicc_a100_summary.txt` |
| Fidelity | `numerical_fidelity.json` |
| Automation | `PYTHONPATH=. python scripts/verify_claims.py` |

---

*End of status report. Multi-day §4 cells intentionally blank until real SUCCESS artifacts exist.*
