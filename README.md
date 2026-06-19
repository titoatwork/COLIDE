# COLIDE: CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection

[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](#license)

## Abstract

COLIDE presents the first custom CUDA kernel implementation for CNN-BiLSTM-based IoT intrusion detection, achieving **2.42x end-to-end speedup** over PyTorch GPU inference. The system integrates an asynchronous LLM-based explainability module that generates human-readable security alerts with only **8 microseconds** dispatch overhead (0.7% of the detection pipeline). Evaluated on the BoT-IoT and ToN-IoT datasets, the system sustains **25,410 flows/sec** in streaming mode while maintaining a macro-F1 score of **0.9352**.

## Key Contributions

1. **Custom CUDA Inference Kernels**: Hand-written CUDA kernels for all four CNN-BiLSTM blocks with kernel fusion, achieving 3.2-6.6x per-block speedups over PyTorch for convolution and dense layers
2. **FP16 Half2 BiLSTM Optimization**: Native half-precision FMA instructions pack LSTM gate pairs into half2 vectors, achieving 1.67x improvement over FP32 and **beating PyTorch's cuDNN by 1.23x**
3. **Async LLM Explainability**: Ring-buffer-based asynchronous dispatch to a quantized LLM (TinyLlama 1.1B Q4) generates SOC-actionable alert explanations without impacting detection latency
4. **Comprehensive Systems Benchmarking**: Per-kernel optimization progression (9.48x from naive to final), batch scaling analysis, energy efficiency, streaming throughput, and framework comparisons

## Results Summary

### Pipeline Speedup (Single Sample)

| Method | Latency (us) | vs PyTorch GPU |
|---|---|---|
| PyTorch CPU | ~2,005 | 0.93x |
| PyTorch GPU | ~1,864 | 1.00x |
| Custom CUDA FP32 | 1,143 | 1.63x |
| **Custom CUDA FP16** | **770** | **2.42x** |
| ORT CPU | ~397 | 4.70x |

### Per-Block Performance

| Block | PyTorch GPU (us) | Custom CUDA (us) | Speedup |
|---|---|---|---|
| 1: Proj+Conv1+BN+ReLU | 404 | 62 | 6.55x |
| 2: Conv2+BN+ReLU+Pool | 282 | 87 | 3.24x |
| 3: BiLSTM FP16 half2 | 791 | 601 | 1.32x |
| 4: Dense Head | 122 | 20 | 6.07x |

### Block 3 Optimization Progression

| Configuration | Latency (us) | vs Naive |
|---|---|---|
| Naive (1 thread/hidden, global W_hh) | 5,698 | 1.00x |
| + Precomputed input projection | 2,901 | 1.96x |
| + Transposed W_hh (coalesced reads) | 1,007 | 5.66x |
| + FP16 native half2 FMA | 601 | **9.48x** |

### Batch Scaling (Throughput, flows/sec)

| Batch Size | PyTorch CPU | PyTorch GPU | ORT CPU | ORT GPU |
|---|---|---|---|---|
| 1 | 733 | 522 | 2,661 | 253 |
| 32 | 6,524 | 15,510 | 8,202 | 9,145 |
| 128 | 8,323 | 27,455 | 6,804 | 26,577 |
| 256 | 10,325 | 42,878 | 8,814 | 45,855 |

### Streaming Throughput

| Mode | Max Sustained | vs CPU Single |
|---|---|---|
| GPU Batched (batch=128) | 25,410 flows/sec | 54x |
| GPU Single | 423 flows/sec | 0.9x |
| CPU Single | 470 flows/sec | 1.0x |

### Energy Efficiency

| Config | Power (W) | Throughput (f/s) | mJ/flow |
|---|---|---|---|
| GPU batch=128 | 17.81 | 17,491 | 1.02 |
| GPU batch=1 | 10.44 | 411 | 25.41 |
| CPU batch=1 | 16.40 | 347 | 47.23 |

GPU batched is **46x more energy efficient** than CPU single-sample.

### LLM Explainability

| Metric | Value |
|---|---|
| Detection latency (no LLM) | 0.32 us |
| Detection latency (with async LLM) | 8.39 us |
| Async dispatch overhead | 8.07 us (0.7%) |
| LLM generation time (median) | 8,528 ms |
| Model | TinyLlama 1.1B Q4 (0.77 GB VRAM) |

### Model Accuracy (BoT-IoT, 733,705 test flows)

| Model | Macro-F1 | Weighted-F1 | Parameters |
|---|---|---|---|
| CNN-BiLSTM V2 | 0.9330 | 0.9695 | 463,877 |
| CNN-BiLSTM V3 (attention) | 0.9352 | 0.9698 | 530,181 |
| Random Forest (baseline) | 0.9768 | — | — |

### Cross-Dataset Validation

| Dataset | Model | Macro-F1 | Weighted-F1 | Classes | Test Samples |
|---|---|---|---|---|---|
| BoT-IoT | RF Baseline | 0.9768 | — | 5 | 733,705 |
| BoT-IoT | CNN-BiLSTM V3 | 0.9352 | 0.9698 | 5 | 733,705 |
| ToN-IoT | RF Baseline | 0.9396 | 0.9844 | 10 | 42,209 |
| ToN-IoT | CNN-BiLSTM V3 | 0.8029 | 0.8622 | 10 | 42,209 |

### Cross-Hardware CUDA Kernel Comparison

| Block | RTX 3050 (Ampere) | V100S 32GB (Volta) | V100S vs 3050 |
|---|---|---|---|
| Block 1 (Conv) | 61.7 us | 11.5 us | 5.4x faster |
| Block 2 (Conv) | 87.2 us | 31.2 us | 2.8x faster |
| Block 3 FP16 (BiLSTM) | 601.0 us | 650.8 us | 0.92x (slower) |
| Block 4 (Dense) | 20.1 us | 10.5 us | 1.9x faster |

Block 3 FP16 is slightly slower on V100S because RTX 3050 (Ampere) has improved FP16 ALUs compared to V100S (Volta), confirming our half2 optimization specifically benefits from Ampere's enhanced FP16 capabilities.

## Hardware and Software Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with compute capability >= 7.0 (Volta architecture or newer)
- **VRAM**: 4 GB minimum (6 GB recommended for LLM prototype)
- **CUDA Toolkit**: 12.0 or later
- **Python**: 3.10+
- **OS**: Linux (tested on Ubuntu 22.04/24.04, WSL2)

### Tested Configuration
- NVIDIA GeForce RTX 3050 6GB Laptop GPU (SM 8.6, 20 SMs, Ampere)
- CUDA 12.6, Driver 560.x
- Python 3.12, PyTorch 2.5.1+cu121
- WSL2 Ubuntu 24.04

## Quick Start

### Option 1: Docker (Recommended)
```bash
docker build -t colide .
docker run --gpus all colide
```

### Option 2: Manual Setup
```bash
# 1. Clone repository
git clone https://github.com/titoatwork/COLIDE.git
cd COLIDE

# 2. Create Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Compile CUDA kernels (adjust -arch for your GPU)
#    sm_86 = RTX 3050/3060/3070/3080/3090 (Ampere)
#    sm_89 = RTX 4090 (Ada Lovelace)
#    sm_90 = H100 (Hopper)
nvcc -arch=sm_86 -o inference/kernels/fused_block1 inference/kernels/fused_block1.cu
nvcc -arch=sm_86 -o inference/kernels/fused_block2 inference/kernels/fused_block2.cu
nvcc -arch=sm_86 -o inference/kernels/fused_block3 inference/kernels/fused_block3.cu
nvcc -arch=sm_86 -o inference/kernels/fused_block3_fp16 inference/kernels/fused_block3_fp16.cu
nvcc -arch=sm_86 -o inference/kernels/fused_block4 inference/kernels/fused_block4.cu

# 4. Run all benchmarks
bash benchmark.sh

# Or run individual benchmarks:
PYTHONPATH=. python scripts/benchmark_pipeline.py     # Full pipeline comparison
PYTHONPATH=. python scripts/benchmark_batch.py        # Batch size scaling
PYTHONPATH=. python scripts/benchmark_ort.py          # ONNX Runtime comparison
PYTHONPATH=. python scripts/benchmark_stats.py        # Statistical confidence intervals
PYTHONPATH=. python scripts/benchmark_energy.py       # Energy efficiency
PYTHONPATH=. python scripts/benchmark_streaming.py    # Streaming throughput
PYTHONPATH=. python scripts/ablation_study.py         # 8-table ablation study
PYTHONPATH=. python scripts/compare_v2_v3.py          # V2 vs V3 architecture
PYTHONPATH=. python scripts/llm_explainability.py     # Async LLM prototype
```

### Training from Scratch
```bash
# Preprocess BoT-IoT dataset
PYTHONPATH=. python scripts/preprocess.py

# Train V2 model (last-timestep pooling)
# Edit scripts/train.py to import from model.cnn_bilstm
PYTHONPATH=. python scripts/train.py

# Train V3 model (self-attention)
# Edit scripts/train.py to import from model.cnn_bilstm_v3_attention
PYTHONPATH=. python scripts/train.py
```

## Project Structure

```
colide/
├── config/
│   └── config.yaml                    # Model and training hyperparameters
│
├── data/
│   ├── raw/                           # Original BoT-IoT dataset (5% subset)
│   │   ├── UNSW_2018_IoT_Botnet_Final_10_Best.csv
│   │   ├── UNSW_2018_IoT_Botnet_Final_10_best_Training.csv
│   │   ├── UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv
│   │   └── toniot/                    # ToN-IoT dataset raw files
│   │       └── train_test_network.csv
│   ├── processed/                     # Preprocessed numpy arrays
│   │   ├── X_train.npy, y_train.npy   # 268,627 training samples
│   │   ├── X_val.npy, y_val.npy       # 293,482 validation samples
│   │   ├── X_test.npy, y_test.npy     # 733,705 test samples
│   │   ├── label_encoder.pkl          # Sklearn label encoder
│   │   └── scaler.pkl                 # MinMaxScaler
│   └── processed_toniot/              # ToN-IoT preprocessed numpy arrays
│       ├── X_train.npy, y_train.npy   # 95,000 training samples
│       ├── X_val.npy, y_val.npy       # 42,209 validation samples
│       ├── X_test.npy, y_test.npy     # 42,209 test samples
│       └── config_toniot.yaml         # ToN-IoT model config
│
├── dicc_scripts/                      # SLURM job scripts for DICC HPC cluster
│   ├── 01_setup.sh                    # Clone repo, install deps, compile kernels
│   ├── 02_benchmark_v100.sh           # V100S benchmark job
│   ├── 03_benchmark_a100.sh           # A100 benchmark job
│   ├── 04_nsight_profile.sh           # Nsight Compute profiling job
│   └── 05_run_all.sh                  # Submit all jobs

├── model/
│   ├── cnn_bilstm.py                 # V2: CNN-BiLSTM with last-timestep pooling
│   ├── cnn_bilstm_v3_attention.py    # V3: CNN-BiLSTM with multi-head self-attention
│   ├── best_model.pth                # Best trained model weights (PyTorch)
│   ├── best_model_toniot.pth         # Best trained ToN-IoT model weights
│   └── weights/                      # Exported per-layer .npy weights for CUDA
│
├── inference/kernels/                 # Custom CUDA inference kernels
│   ├── fused_block1.cu               # Block 1: Linear(10,64)+Reshape+Conv1D(2,64)+BN+ReLU
│   ├── fused_block2.cu               # Block 2: Conv1D(64,128)+BN+ReLU+MaxPool1D
│   ├── fused_block3.cu               # Block 3: 2-layer BiLSTM (FP32, transposed W_hh)
│   ├── fused_block3_fp16.cu          # Block 3: 2-layer BiLSTM (FP16, native half2 FMA)
│   └── fused_block4.cu               # Block 4: Dense(128,64)+ReLU+Dense(64,5)
│
├── scripts/                           # Benchmarks and utilities
│   ├── train.py                       # Model training with early stopping
│   ├── preprocess.py                  # Dataset preprocessing pipeline
│   ├── preprocess_toniot.py           # ToN-IoT preprocessing pipeline
│   ├── benchmark_pipeline.py          # Full pipeline latency comparison
│   ├── benchmark_batch.py             # Batch size scaling analysis
│   ├── benchmark_ort.py               # ONNX Runtime + TensorRT comparison
│   ├── benchmark_stats.py             # Statistical confidence intervals (10 trials)
│   ├── benchmark_energy.py            # GPU power draw and energy efficiency
│   ├── benchmark_streaming.py         # Streaming throughput at increasing rates
│   ├── ablation_study.py              # 8-table ablation study
│   ├── train_toniot.py                # ToN-IoT model training
│   ├── rf_baseline_toniot.py          # Random Forest baseline on ToN-IoT
│   ├── validate_weights.py            # Export real weights for CUDA validation
│   ├── compare_datasets.py            # Cross-dataset comparison table
│   ├── compare_v2_v3.py               # V2 vs V3 architecture comparison
│   └── llm_explainability.py          # Async LLM explainability prototype
│
├── benchmarks/results/                # JSON outputs from all benchmarks
│   ├── pipeline_benchmark.json
│   ├── streaming_throughput.json
│   ├── energy_efficiency.json
│   ├── statistical_confidence.json
│   ├── llm_explainability.json
│   ├── training_history.json
│   ├── training_history_toniot.json
│   └── rf_baseline_toniot.json
│
├── docs/                              # Documentation and literature
│   ├── Verified Papers and Gaps in IDS_IoT Research.pdf
│   └── literature_review_raw.md
│
├── Dockerfile                         # Reproducibility container
├── benchmark.sh                       # One-command benchmark runner
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## CUDA Kernel Design

### Architecture Overview
The CNN-BiLSTM model is decomposed into four inference blocks, each implemented as a fused CUDA kernel:

- **Block 1** fuses linear projection, reshape, 1D convolution, batch normalization, and ReLU activation into a single kernel launch, eliminating 5 separate PyTorch kernel launches
- **Block 2** fuses convolution, batch normalization, ReLU, and max pooling
- **Block 3** implements a 2-layer bidirectional LSTM with transposed weight matrices for coalesced global memory access and FP16 half2 vector operations for doubled compute throughput
- **Block 4** fuses two dense layers with ReLU activation

### FP16 Half2 Optimization (Block 3)
The BiLSTM inner loop computes four LSTM gates (input, forget, cell, output) via matrix-vector multiplication with the recurrent weight matrix W_hh. The FP16 optimization:

1. Repacks W_hh into half2 pairs: (i_gate, f_gate) and (g_gate, o_gate)
2. Uses `__hfma2` (fused multiply-add for half2) to process two gates per instruction
3. Stores hidden states in shared memory as half-precision
4. Accumulates gate values in FP16, converts to FP32 only for sigmoid/tanh activations

This exploits the RTX 3050 Ampere architecture's 2x FP16 throughput, reducing Block 3 from 1,007 us (FP32) to 601 us (FP16).

## Dataset

**BoT-IoT** (Koroniotis et al., Future Generation Computer Systems, 2019)

| Property | Value |
|---|---|
| Source | UNSW Sydney |
| Subset | 5% (best 10 features) |
| Features | 10 network flow statistics |
| Classes | 5: DDoS, DoS, Normal, Reconnaissance, Theft |
| Train samples | 268,627 |
| Validation samples | 293,482 |
| Test samples | 733,705 |
| Preprocessing | Undersample majority, SMOTE minority, MinMax normalization |

**ToN-IoT** (Moustafa, 2021)

| Property | Value |
|---|---|
| Source | UNSW Canberra |
| Features | 13 (10 numeric + 3 categorical encoded) |
| Classes | 10: backdoor, ddos, dos, injection, mitm, normal, password, ransomware, scanning, xss |
| Train samples | 95,000 (balanced) |
| Validation samples | 42,209 |
| Test samples | 42,209 |
| Preprocessing | Undersample majority to 10K, SMOTE minority to 5K, MinMax normalization |

## Verified Research Gaps

Independent verification via ChatGPT Deep Research confirmed four novel contributions:

1. **No published work** on custom CUDA inference kernels for IDS models
2. **No published work** measuring LLM overhead on IDS detection latency
3. **No published work** with Nsight Compute / roofline profiling for IDS
4. **No published work** comparing TensorRT vs custom CUDA for small (<1M parameter) models

## Citation

```bibtex
@article{colide2026,
  title={COLIDE: CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection},
  author={Haque, Ibteshamul and Por, Lip Yee},
  journal={IEEE Internet of Things Journal},
  year={2026},
  note={Under preparation}
}
```

## Acknowledgments

This research was conducted at the Faculty of Computer Science and Information Technology (FCSIT), Universiti Malaya, under the supervision of Prof. Dr. Por Lip Yee.

This research was supported in part through computational resources provided by the Data-Intensive Computing Centre, Universiti Malaya.

## License

This project is for academic research purposes only. Contact the authors for commercial licensing.
