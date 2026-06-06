# COLIDE

## CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection

**Institution:** Universiti Malaya (UM), FCSIT
**Supervisor:** Prof. Dr. Por Lip Yee
**Researcher:** Ibteshamul Haque

---

## Project Overview

COLIDE is a three-part real-time intrusion detection system for IoT networks:

1. A CNN-BiLSTM neural network classifies network flows as benign or one of multiple attack categories.
2. Custom hand-tuned CUDA kernels execute inference at bare-metal GPU speed without framework runtime dependency.
3. When an anomaly is detected, an asynchronous LLM generates a human-readable alert for SOC analysts — without blocking detection latency.

**This is an engineering and systems paper.** The contribution is reproducible bare-metal CUDA inference, LLM explainability without latency destruction, and rigorous hardware-comparative benchmarking — not a novel ML architecture or attack taxonomy.

---

## Current Status

### Phase 0 — Remote Prototyping ✅ COMPLETE (May–Early June 2026)

- [x] Literature review and project planning (Early May)
- [x] Development environment setup — WSL2, CUDA, PyTorch (Late May)
- [x] BoT-IoT EDA — 5 classes, 10 features, 23,712:1 imbalance (Late May)
- [x] Preprocessing V1 — windowed sequences (archived after diagnosis)
- [x] CNN-BiLSTM V1 — windowed (archived, macro-F1 0.22)
- [x] Random Forest baseline — per-flow, macro-F1 **0.9768** (Early June)
- [x] Preprocessing V2 — per-flow pipeline (canonical)
- [x] CNN-BiLSTM V2 — per-flow, macro-F1 **0.9330** (Early June)
- [x] Model weights exported FP32 + FP16
- [x] Experimental design document

### Phase 1 — CUDA Optimization (Upcoming)

- [ ] Custom CUDA kernels (Conv1D, BiLSTM gates, fused ops)
- [ ] CUDA Graphs pipeline
- [ ] Tensor core integration (FP16)
- [ ] LLM async integration
- [ ] Streaming load benchmark
- [ ] Five-level benchmark stack
- [ ] Benchmark freeze + plots

---

## Key Results

### Classification Performance (Test Set — 733,705 samples)

| Model | Macro-F1 | Weighted-F1 | All Classes |
|-------|----------|-------------|-------------|
| Random Forest (baseline) | **0.9768** | 0.9906 | 5/5 |
| CNN-BiLSTM V2 (per-flow) | **0.9330** | 0.9683 | 5/5 |
| CNN-BiLSTM V1 (windowed) | 0.22 | — | 2/5 |

### CNN-BiLSTM V2 Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| DDoS | 0.9844 | 0.9555 | 0.9698 | 385,309 |
| DoS | 0.9545 | 0.9819 | 0.9680 | 330,112 |
| Normal | 0.7571 | 0.9907 | 0.8583 | 107 |
| Reconnaissance | 0.9029 | 0.9927 | 0.9457 | 18,163 |
| Theft | 1.0000 | 0.8571 | 0.9231 | 14 |

---

## Dataset

- **Source:** BoT-IoT (Koroniotis et al., FGCS 2019)
- **Version:** 5% subset, 10-best features, pre-made train/test split
- **Training samples:** 268,627 (after resampling)
- **Validation samples:** 293,482 (real data, no synthetic)
- **Test samples:** 733,705 (real data, no synthetic)
- **Classes:** DDoS, DoS, Normal, Reconnaissance, Theft
- **Features:** 10 per-flow network statistics

---

## Model Architecture

```
Input: (batch, 10) — per-flow features
    ↓
Linear Projection: 10 → 64
    ↓
Reshape: (batch, 2, 32)
    ↓
Conv1D(2→64, k=3) → BatchNorm → ReLU
Conv1D(64→128, k=3) → BatchNorm → ReLU
MaxPool1D(2) → Dropout(0.3)
    ↓
BiLSTM(128→128, bidirectional)
BiLSTM(256→64, bidirectional)
    ↓
Dense(128→64) → ReLU → Dropout(0.3)
Dense(64→5) → logits
```

- **Parameters:** 463,877
- **FP32 size:** 1.77 MB
- **FP16 size:** 0.89 MB

---

## Methodology Note

The initial approach used temporal windowing (20 consecutive flows per sample), assuming sequential dependencies between network flows. This produced severe overfitting (train accuracy 100%, val macro-F1 0.22), predicting only 2 of 5 classes.

Systematic investigation identified the root cause: the BoT-IoT 10-best features are pre-aggregated flow-level statistics (mean, stddev, srate, drate, etc.) with no meaningful temporal dependency between consecutive rows. A literature review confirmed that every published paper on this dataset uses per-flow classification.

A Random Forest baseline on per-flow data immediately achieved macro-F1 0.9768, confirming the features are highly separable without temporal context. The pipeline was redesigned as a per-flow system, and the CNN-BiLSTM V2 achieved macro-F1 0.9330 with all 5 classes correctly classified.

The windowed approach is archived in the repository (V1 files) as documentation of the investigation process.

---

## Repository Structure

```
colide/
├── config/config.yaml              # All hyperparameters
├── data/raw/                        # BoT-IoT CSVs (not committed)
├── data/processed/                  # Preprocessed .npy files (not committed)
├── model/
│   ├── cnn_bilstm.py               # CNN-BiLSTM V2 (per-flow, active)
│   ├── cnn_bilstm_v1_windowed.py   # V1 architecture (archived)
│   ├── best_model.pth              # Best checkpoint (epoch 22)
│   └── weights/                     # FP32 + FP16 .npy weights
├── preprocessing/
│   ├── preprocess_v2.py            # Per-flow pipeline (active)
│   └── preprocess_v1_windowed.py   # Windowed pipeline (archived)
├── scripts/
│   ├── train.py                     # Training script V2 (active)
│   ├── train_v1_windowed.py        # V1 training (archived)
│   └── rf_baseline.py              # Random Forest baseline
├── notebooks/
│   └── 01_eda.ipynb                 # Exploratory Data Analysis
├── benchmarks/results/              # Training history JSON
├── environment.md                   # Hardware + software versions
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Setup

```bash
# Clone
git clone https://github.com/titoatwork/COLIDE.git
cd colide

# Create virtual environment (WSL2 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Download BoT-IoT dataset to data/raw/
# (see environment.md for download instructions)

# Run preprocessing
python preprocessing/preprocess_v2.py

# Train model
python scripts/train.py

# Run RF baseline
python scripts/rf_baseline.py
```

---

## Hardware

- **Development:** RTX 3050 Laptop (SM 8.6, 6GB VRAM, Ampere)
- **Benchmarking (Phase 1):** Cloud A100 instance

---

## Citation

If you use the BoT-IoT dataset, please cite:

> Koroniotis, N., Moustafa, N., Sitnikova, E., & Turnbull, B. (2019). Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-IoT dataset. Future Generation Computer Systems, 100, 779-796.
