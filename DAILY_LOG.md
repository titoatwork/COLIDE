# COLIDE — Daily Progress Log

---

## Week 1 — May 1–8, 2026 (Early May — Project Foundation)

### Completed
- [x] Reviewed BoT-IoT dataset documentation (Koroniotis et al., FGCS 2019)
- [x] Studied CNN-BiLSTM architectures used in IoT IDS literature
- [x] Drafted initial project proposal and discussed scope with Prof. Por
- [x] Defined project contribution framing: engineering/systems paper, not ML novelty
- [x] Created COLIDE team brief v1 with project scope, timeline, and methodology
- [x] Surveyed target publication venues (IEEE IoT-J, FGCS, JNCA)
- [x] Identified key risks: class imbalance, LLM latency, hardware migration
- [x] Designed async LLM architecture (decoupled from detection path)
- [x] Planned five-level benchmark stack: CPU → PyTorch GPU → TensorRT → Custom CUDA

### Key Decisions
- Contribution claim locked: bare-metal CUDA inference + LLM explainability + hardware benchmarking
- Publication targets: IoT-J (stretch), FGCS (realistic)
- LLM integration: async dispatch, never in detection critical path

---

## Week 2–3 — May 9–23, 2026 (End-Semester Examinations)

### Status
- Project work paused during end-semester examinations
- Maintained awareness of literature and refined project plan offline

---

## Week 4 — May 24–28, 2026 (Late May — Development Begins)

### Completed
- [x] Set up WSL2 development environment with CUDA support
- [x] Installed Python 3.12, PyTorch (CUDA 12.1), and all dependencies in virtual environment
- [x] GPU confirmed: NVIDIA RTX 3050 Laptop (SM 8.6, 6GB VRAM, CUDA 13.2)
- [x] Initialised Git repository with project scaffold
- [x] Created config.yaml with all hyperparameters, seeds, and paths
- [x] Created environment.md documenting hardware and software versions
- [x] Pushed initial commit to GitHub (private repository)
- [x] Downloaded BoT-IoT dataset (5%, 10-best features, pre-made train/test split)
- [x] Downloaded dataset documentation PDFs (Read_Me.pdf, The Bot-IoT Dataset.pdf)
- [x] Began Exploratory Data Analysis

### Environment
- WSL2 Ubuntu on Windows
- Python 3.12 with isolated virtual environment
- PyTorch 2.x with CUDA 12.1 bundled runtime
- GPU: RTX 3050 Laptop, Driver 595.97

### Key Findings
- Dataset uses comma separator (initial assumption of semicolon was incorrect)
- 19 columns total: 10 numeric features, 3 categorical, 3 target-related, 3 identifiers
- `attack` column is binary (0/1), `category` contains class names — `category` selected as label

---

## Week 5 — June 1–4, 2026 (Early June — After Arriving in Malaysia)

### Day 1–2: EDA Completion & Preprocessing V1

**EDA Results:**
- 5 classes confirmed: DDoS, DoS, Normal, Reconnaissance, Theft
- Extreme imbalance: 23,712:1 (DDoS vs Theft)
- 10 clean numeric features (matching dataset authors' 10-best selection)
- Zero missing values, zero duplicates, no zero-variance features
- High correlation: max ↔ mean (r=0.9087), both retained
- config.yaml auto-populated with real dataset values
- 3 publication-ready plots saved: class_distribution_train.png, class_distribution_test.png, feature_correlation.png

**Preprocessing V1 (Windowed):**
- Built windowed sequence pipeline: SMOTE → MinMax → sliding window (size=20)
- Output shapes: X_train (295608, 20, 10), X_val (293463, 20, 10), X_test (733686, 20, 10)
- Computed class weights (float32)

**CNN-BiLSTM V1 (Windowed):**
- Architecture: input (batch, 20, 10), 464K params
- Trained 30 epochs on RTX 3050

**V1 Failure — Diagnosed:**
- Train accuracy: 100%, Val macro-F1: 0.22
- Model only predicted DDoS and DoS (2 of 5 classes)
- Root cause analysis identified three compounding issues:
  1. Windowing across mixed SMOTE'd data created impossible sequences
  2. SMOTE + class weights = double correction, destabilising gradients
  3. SMOTE expansion too aggressive: Theft 65→10,000 (154×), Normal 370→20,000 (54×)

### Day 3: Critical Discovery & Pipeline Redesign

**RF Baseline Experiment:**
- Hypothesised that windowing was the root cause, not the model
- Trained Random Forest on per-flow data (no windowing), same resampling
- **Test macro-F1: 0.9768** — all 5 classes correctly classified
- Proved BoT-IoT 10-best features are separable at per-flow level without temporal modeling

**Literature Confirmation:**
- Searched published papers using BoT-IoT 10-best features
- Every paper uses per-flow classification — no windowing
- Features (mean, stddev, srate, drate) are pre-aggregated flow statistics — no inter-flow temporal dependency
- Windowing was adding noise, not information

**Preprocessing V2 (Per-Flow):**
- Rebuilt pipeline: Load → Split → Undersample → SMOTE → Scale → Save
- No windowing, per-flow: X_train (268627, 10)
- Conservative resampling targets: Normal→2,000 (6×), Theft→1,000 (17×)
- Class weights computed but NOT used in training (saved for ablation)
- V1 files archived for reference

### Day 4: CNN-BiLSTM V2 Training & Results

**Model V2:**
- Architecture updated for per-flow input (batch, 10)
- Added input projection (10→64), reshape (2, 32), BatchNorm
- 463,877 parameters, 1.77 MB

**Training:**
- 32 epochs, early stopped (best at epoch 22, val macro-F1: 0.9418)
- Each epoch ~40 seconds on RTX 3050

**Test Results (CNN-BiLSTM V2):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| DDoS | 0.9844 | 0.9555 | 0.9698 | 385,309 |
| DoS | 0.9545 | 0.9819 | 0.9680 | 330,112 |
| Normal | 0.7571 | 0.9907 | 0.8583 | 107 |
| Recon | 0.9029 | 0.9927 | 0.9457 | 18,163 |
| Theft | 1.0000 | 0.8571 | 0.9231 | 14 |
| **Macro-F1** | | | **0.9330** | |

**Weight Export:**
- FP32 and FP16 .npy files exported for all layers
- Ready for CUDA kernel loading in Phase 1

### Day 5: Documentation & Submission Preparation

**Completed:**
- [x] Progress Report PDF (12 pages, LaTeX)
- [x] Experimental Design Document PDF (8 pages, LaTeX)
- [x] README.md updated with actual results and methodology
- [x] DAILY_LOG.md updated with complete findings
- [x] Repository cleaned and all artifacts committed
- [x] Phase 0 submission package prepared for Prof. Por

---

## Methodology Decisions Made During Phase 0

| # | Problem | Investigation | Outcome |
|---|---------|---------------|---------|
| 1 | CSV loaded as 1 column | Inspected raw bytes | Comma separator, not semicolon |
| 2 | Three candidate label columns | Inspected sample values | `category` for multiclass (not binary `attack`) |
| 3 | V1 macro-F1 = 0.22, only 2 classes predicted | Diagnosed 3 root causes | Mixed sequences, double correction, aggressive SMOTE |
| 4 | Windowing assumption questioned | Literature review + RF baseline test | Per-flow confirmed (macro-F1 0.9768 without windowing) |
| 5 | SMOTE targets too aggressive | Analysed expansion ratios | Reduced: Normal→2K, Theft→1K |
| 6 | Early stopping on val loss | Compared loss vs macro-F1 behaviour | Switched to macro-F1 (better for imbalanced data) |
| 7 | Port numbers as potential leakage | Verified feature list from config | sport/dport not in 10-best — no leakage |
| 8 | SMOTE before windowing created impossible sequences | Analysed interaction | V2: SMOTE on per-flow vectors (semantically valid) |
| 9 | Val set larger than train set after undersampling | Evaluated alternatives | Kept as-is — val must be real data only |

---

## Phase 0 Summary

### What Was Accomplished
- Complete EDA of BoT-IoT 10-best features dataset
- Two preprocessing pipelines developed (V1 windowed → V2 per-flow)
- Two model iterations (V1 failed → V2 succeeded)
- Random Forest baseline establishing accuracy ceiling
- Model weights exported FP32 + FP16 for CUDA kernel development
- Experimental design document defining Phase 1 methodology
- 6 Git commits documenting the full development progression

### Key Discovery
The BoT-IoT 10-best features are pre-aggregated flow statistics. Temporal windowing of consecutive flows is unnecessary and counterproductive. Per-flow classification is both simpler and more effective, consistent with the published literature.

### Phase 1 Readiness
All prerequisites for CUDA kernel development are met: trained model, exported weights, verified accuracy, defined benchmarking methodology.
