# Plain-English numbers card (for Prof updates — no repo jargon)

**Purpose:** future short emails / talk tracks. Prof has **no codebase access**.  
**Authority:** sealed_test / wp6b / claims on laptop · never invent DICC multi-day cells  
**Champion weights fingerprint:** `80a90f7cc210276300eaa90173a5a385` (frozen)

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

## Local systems (laptop GPU, 5 measurement sessions)

- Energy per flow: **0.920–0.943** mJ
- Full model PyTorch latency @ batch 256: **24.15–25.68** µs per sample
- Custom CUDA (operation-matched blocks only): derived pipeline **565–570** µs — **not** claimed as full-model CUDA vs full model PyTorch
- Peak GPU memory: **322.2** MiB
- Multi-objective efficiency path example: composite score **0.9056** at **4.33** µs

## Explainability

- **16.60 µs** = cost to **dispatch** an explanation path only (not full LLM generation time)
- Free-form LLM text is weak → we do **not** title the paper as fully LLM-explainable
- We keep structured evidence templates + dispatch timing

## Second dataset (ToN-IoT)

- Same-style recipe on 13 features: neural test ~**0.811** vs same-split RF ~**0.939** (honest gap)

## Still open

- UM cluster multi-day (V100S + A100, two days, same-GPU CUDA block vs matching PyTorch + full model PyTorch absolute latency)
- No portability claim until those results exist on disk

*End card.*
