# COLIDE
## CUDA-Optimized CNN-BiLSTM with LLM-Based Explainability for IoT Intrusion Detection

**Institution:** Universiti Malaya (UM), FCSIT  
**Supervisor:** Prof. Dr. Por Lip Yee  
**Project Lead:** Ibteshamul Haque  

---

## Project Status

### Phase 0 — Remote Prototyping (Active)
- [x] Repository scaffold
- [ ] BoT-IoT EDA complete
- [ ] Preprocessing pipeline
- [ ] CPU baseline with metrics
- [ ] CUDA kernel prototypes
- [ ] H2D pipeline + IPC skeleton

### Phase 1 — On-site DICC (June 1–29)
- [ ] DICC environment confirmed
- [ ] Custom CUDA kernels (CNN + BiLSTM)
- [ ] CUDA Graphs pipeline
- [ ] Tensor core integration (FP16)
- [ ] LLM async integration
- [ ] Streaming load benchmark
- [ ] Benchmark freeze + plots

---

## Setup

```bash
# Clone repo
git clone [repo-url]
cd colide

# Create and activate virtual environment (WSL2)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Hardware
- Local dev: RTX 3050 Laptop (SM 8.6, 6GB VRAM, CUDA 13.2)
- Benchmark target: DICC cluster — Ampere/Hopper (TBC)
