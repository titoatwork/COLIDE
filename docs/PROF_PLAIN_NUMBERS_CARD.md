# Plain-English numbers card (for Prof updates — no repo jargon)

**Purpose:** future short emails / talk tracks. Prof has **no codebase access**.  
**Authority:** sealed_test / wp6b / claims / DICC SUCCESS JSON on laptop · never invent cells  
**Champion weights fingerprint:** `80a90f7cc210276300eaa90173a5a385` (frozen)  
**Updated:** 2026-08-12

---

## Detection (BoT-IoT, same evaluation protocol)

- Sealed multi-seed **test** (5 independent runs): macro-F1 **0.9780 ± 0.0033**
- Weakest class mean F1 **0.9292**; rare attack class Theft **1.0**
- Strong classical baseline on same protocol: LightGBM validation macro-F1 **0.9818** (still tops pure F1)
- Random Forest same protocol validation **0.9778**
- Older published RF **0.9864** is a **different pipeline** — dual bar only
- **Lead claim:** accuracy–efficiency trade-off, not pure F1 supremacy

## Method (one frozen package)

- CNN–BiLSTM with attention, distilled from an **ensemble of tree models**, focal loss, Optuna-selected train settings, standard shuffle training, argmax decisions
- Test used only after final freeze

## Local systems (laptop GPU, multi-session)

- Energy per flow: **0.920–0.943** mJ
- Full model PyTorch latency @ batch 256: **24.15–25.68** µs per sample
- Custom CUDA (operation-matched blocks only): derived pipeline **565–570** µs — **not** claimed as full-model CUDA vs full model PyTorch
- Peak GPU memory: **322.2** MiB
- Multi-objective efficiency path example: composite score **0.9056** at **4.33** µs
- Framework comparison vs Custom pipeline: multi-session **ranges** (eager / compile / TensorRT / ORT-GPU) — laptop only

## Explainability

- **16.60 µs** = cost to **dispatch** an explanation path only (not full LLM generation time)
- Free-form LLM text is weak → we do **not** title the paper as fully LLM-explainable
- We keep structured evidence templates + dispatch timing

## Second dataset (ToN-IoT)

- **Active (prefer):** leakage-safe CNN test **0.8075** / RF **0.9626** (`toniot_leakage_safe_v1`)
- Optional older 13-feat package path: neural ~**0.811** vs RF ~**0.939** (comparable prior only)
- Clean 26-feat **0.9526 / 0.9851 / +15.4%**: **INVALID** (do not cite)

## UM DICC multi-session (measured — three sessions, V100S + A100)

- **Block 3 (same ops):** matching PyTorch is **faster** than our FP16 Custom CUDA on both GPUs (**pre_fix** historical binaries; not post_fix rebench)  
  - V100S: ~**363 µs** PT vs ~**513 µs** CUDA  
  - A100: ~**385–391 µs** PT vs ~**667–671 µs** CUDA  
- Custom CUDA remains much faster on Blocks **1, 2, and 4**  
- Local production-weight B3 CUDA↔PT parity: **closed** (`block3_parity_gate.json` valid=true)  
- Full Custom CUDA pipeline vs full PyTorch model: **not claimed** (Option A)  
- Full-model PyTorch absolute: V100S ~**964–973 µs**; A100 ~**945–962 µs**  
- Full multi-compiler on DICC (batch-1 protocol): TensorRT native ~**528 µs** (V100S) / ~**588 µs** (A100); torch.compile ~**865 / 770**; eager ~**1041 / 932** 

## Still open (writing / optional)

- Manuscript multi-GPU section write-up  
- Optional: clean A100 provenance re-run; Nsight “why B3”; B3 kernel optim only if PI prioritizes  

*End card.*
