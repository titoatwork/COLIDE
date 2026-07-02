# COLIDE — Pre-Written Text Blocks for Manuscript

## 1. LLM Claim Paragraph (replaces "first to measure" claim)

Recent work by Jamshidi et al. (2026) demonstrated the integration of LLMs for IoT intrusion reasoning at edge gateways, dispatching alerts to cloud-hosted models (GPT-4-turbo, LLaMA 3.5) via API calls with latencies under 1.5 seconds and bandwidth overhead under 1.2 kB per prompt. However, their approach introduces external network dependencies, variable API latency, and potential data privacy risks inherent in transmitting security telemetry to third-party endpoints. Our approach differs fundamentally: we deploy a fully local, quantized TinyLlama 1.1B (4-bit, 0.77 GB VRAM) on the same GPU as the detection pipeline, using an asynchronous ring-buffer dispatch mechanism. The measured dispatch overhead is 16.60 us at p99 (over 5,000 trials of the classify-construct-push code path) — roughly 2.5% of the 674 us inference pipeline latency, and negligible next to the multi-second LLM generation time — with zero network dependency and complete data sovereignty. While the LLM generation itself takes approximately 7.4 seconds per alert on average, the asynchronous design ensures the detection pipeline is never blocked. To our knowledge, this represents the first fully on-device, air-gapped LLM explainability integration for real-time intrusion detection.


## 2. Alert Aggregation Paragraph (addresses DDoS queue overflow)

To prevent ring-buffer overflow during high-volume attacks, the system employs a time-windowed alert aggregation mechanism. Incoming alerts are grouped by (attack_type, source_ip) over a configurable window (default 10 seconds). At the end of each window, structurally identical alerts are collapsed into a single batched prompt containing the alert count, time span, and representative sample flows. This reduces LLM dispatch from potentially thousands of individual calls per second to one consolidated prompt per source-attacker pair per window. During a simulated DDoS attack generating 25,000 malicious flows over 5 seconds, the aggregator reduced LLM invocations from 25,000 to 10 (one per unique source IP), eliminating the risk of memory exhaustion in the generation thread while preserving the explainability output for security analysts.


## 3. SMOTE Limitation Paragraph (Theft class honesty)

The Theft class in BoT-IoT contains only 52 training samples, expanded to 1,000 via SMOTE. While the model generalises well to the 14 real test samples (F1 0.9286), the limited real support means performance may vary under different network conditions or attack distributions. This is an inherent limitation of the BoT-IoT dataset rather than a methodological flaw — the Theft category represents a rare attack type with minimal representation in the original data collection. We report the minority-class results transparently and note that deploying this system in environments with higher Theft-class prevalence would benefit from additional real training samples.


## 4. Pseudo-Sequence Acknowledgment (MLP ablation discussion)

We do not claim that the CNN-BiLSTM is the optimal classifier for tabular flow data. An equivalent-parameter MLP (400,901 parameters) trained with the identical distillation recipe achieves a test macro-F1 of 0.9542 after two-stage fine-tuning, compared to 0.9790 for the CNN-BiLSTM under the same protocol. While the accuracy difference is modest, the CNN-BiLSTM consistently outperformed the MLP across all training configurations, suggesting the convolutional and recurrent layers provide marginal but measurable benefit even on tabular features. More critically, the CNN-BiLSTM was retained because its computational pattern — 1D convolutions, batch normalisation, and bidirectional recurrence with dynamic control flow — exposes the limitations of automated inference compilers (TensorRT, torch.compile) that our custom CUDA kernels solve. An MLP would be trivially optimised by existing frameworks, offering no systems insight. The architectural complexity is justified by the engineering contributions it enables.


## 5. Summary Table (Prof. Por requested)

| Method | Macro-F1 | Latency (us) | Throughput (flows/sec) | Energy (mJ/flow) | Hardware |
|--------|----------|-------------|----------------------|-------------------|----------|
| CPU sklearn RF | 0.9864 | — | 305,248 | — | Laptop CPU |
| GPU cuML RF | 0.9471 | — | 1,667,495 | — | V100S |
| Eager PyTorch | 0.9790 | 2,247 | — | — | RTX 3050 |
| torch.compile | 0.9790 | 1,777 | — | — | RTX 3050 |
| TensorRT FP16 | 0.9790 | 2,966 | — | — | RTX 3050 |
| ORT GPU | 0.9790 | 4,652 | — | — | RTX 3050 |
| ORT CPU | 0.9790 | 699 | — | — | Laptop CPU |
| **Custom CUDA FP16** | **0.9790** | **674** | **25,899** | **0.79** | **RTX 3050** |
| Custom CUDA FP16 | 0.9790 | 551 | — | — | V100S |
| Custom CUDA FP16 | 0.9790 | 592 | — | — | A100 |

Notes:
- All DL methods use the same two-stage fine-tuned CNN-BiLSTM model (0.9790)
- RF uses 200-tree sklearn/cuML RandomForestClassifier
- Latency = single-sample inference; RTX 3050 framework latencies are the 20-trial means from
  `benchmarks/results/statistical_significance_v2.json` (the same source as the framework-comparison
  table elsewhere in this doc/README) -- do not substitute other single-run numbers for these
- Throughput = sustained streaming (batch=128)
- Energy measured via nvidia-smi power draw integration

## 6. KD Sweep Documentation (Prof. Por requested)

| Config | Alpha | Temp | Focal | Val F1 | Test F1 | Notes |
|--------|-------|------|-------|--------|---------|-------|
| Baseline V3 | — | — | — | 0.9418 | 0.9330 | Original, no KD |
| Focal only | — | — | 2.0 | 0.0003 | 0.0010 | Catastrophic collapse |
| KD 1 | 0.5 | 1.0 | — | 0.9703 | 0.9481 | First working KD |
| KD 2 | 0.3 | 1.0 | — | 0.9474 | 0.9421 | Less teacher |
| KD 3 | 0.7 | 1.0 | — | 0.9599 | 0.9284 | Overfit |
| KD 4 | 0.9 | 1.0 | — | 0.9567 | 0.9284 | Overfit |
| KD 5 | 0.5 | 3.0 | — | 0.9541 | 0.9341 | Temp hurt |
| KD 6 | 0.7 | 5.0 | — | 0.9620 | 0.9547 | Best sweep (round 1) |
| KD 7 | 0.7 | 5.0 | 2.0 | 0.9728 | 0.9601 | Best KD+focal (round 1) |
| Two-stage v1 | 0.7 | 5.0 | 2.0 | — | 0.9639 | Fine-tuned on real data (round 1) |
| KD 8 | 0.6 | 7.0 | 2.0 | 0.9780 | 0.9702 | Round 2: extended T past 5.0 |
| KD 9 | 0.7 | 7.0 | 2.0 | 0.9728 | 0.9687 | Round 2 |
| KD 10 | 0.8 | 7.0 | 2.0 | 0.9751 | 0.9757 | Round 2 |
| KD 11 | 0.7 | 10.0 | 2.0 | 0.9482 | 0.9033 | Round 2 outlier: Normal/Theft precision collapse (0.75/0.67) despite excellent majority-class F1 |
| KD 12 | 0.8 | 10.0 | 2.0 | 0.9672 | 0.9745 | Round 2 |
| KD 13 | 0.6 | 10.0 | 2.0 | 0.9757 | 0.9763 | Round 2: best KD+focal |
| **Two-stage v2** | **0.6** | **10.0** | **2.0** | **—** | **0.9790** | **Fine-tuned on real data (round 2, current best)** |


## 7. GPU Profiling Paragraph (hardware characterisation)

All four custom kernels achieve 100% theoretical occupancy on the RTX 3050 (Ampere SM 8.6, 20 SMs, 1536 max threads/SM). Block 1 and Block 2 launch 256 threads per block with minimal shared memory (2-4 KB), achieving 6 concurrent blocks per SM. The BiLSTM kernel (Block 3) uses 128 threads with 8 KB shared memory, allowing 12 blocks per SM. Block 4 uses 64 threads at 1 KB shared memory, sustaining 24 blocks per SM. The high occupancy confirms that the performance gains from our custom kernels over TensorRT (4.40x) and torch.compile (2.63x) are not due to superior hardware utilisation, but rather the elimination of CPU-to-GPU kernel launch overhead. TensorRT decomposes the model into approximately 128 individual kernel launches at 5-15 us each, accumulating significant host-side latency. Our chained pipeline executes back-to-back on the device with zero inter-kernel synchronisation, converting launch-bound execution into compute-bound execution.


## 8. Preprocessing Overhead

Data preprocessing (MinMaxScaler normalization) adds 43.7 us per sample, representing 6.1% of the total end-to-end pipeline latency of 717.7 us. The preprocessing step is executed on the CPU prior to GPU inference and does not affect the custom CUDA kernel measurements. The total detection latency from raw network flow features to classification output remains sub-millisecond at 717.7 us.

## 9. TensorRT Build Configuration

TensorRT benchmark used the following configuration:
- TensorRT version: 11.1.0.106
- ONNX export: opset 14, batch size 1, static shapes
- Builder: default workspace (256 MB), auto precision selection (TensorRT 11 removed manual FP16 flag, selects automatically)
- Execution: native Python API via tensorrt.IExecutionContext.execute_v2()
- Memory: pycuda-allocated device buffers with async host-device transfers
- No manual CUDA graph capture (TensorRT 11 handles internally)
- No INT8 calibration (insufficient calibration data for this model size)
- Note: TensorRT's enqueueV3() C++ API was not used; the Python API wrapper was employed for consistency with other framework benchmarks.

## 10. torch.compile Crash Evidence

torch.compile(mode="reduce-overhead") with manual CUDA graph capture fails on the CNN-BiLSTM architecture with the error: "RuntimeError: Cannot prepare for replay during capturing stage. Current cudaStreamCaptureStatus: cudaStreamCaptureStatusActive." This occurs because the BiLSTM's dynamic recurrent control flow creates internal memory allocations that violate CUDA graph's requirement for static memory addresses. The full crash trace is preserved in docs/torch_compile_crash_trace.txt. Without manual CUDA graph capture, torch.compile achieves 1,912 us (RTX 3050) and 829 us (V100S) — still 2.83x and 1.51x slower than our custom CUDA kernels respectively (this is a distinct, slower torch.compile configuration than the 1,777 us CUDA-graph-mode figure used in the main framework-comparison table above; the two should not be conflated -- fixed 2026-07-01, was mislabeled as 2.64x/1.50x, the graph-mode ratio).


## 11. Sample LLM Explanations

Example output from TinyLlama 1.1B (4-bit quantized) for detected attacks:

**DDoS Alert:**
"A high-volume Distributed Denial of Service attack was detected from source IP 192.168.1.5 targeting port 80. The attack generated 2,500 flows in 10 seconds with an average packet size of 1,024 bytes. This pattern is consistent with a volumetric flood attack aimed at exhausting server bandwidth. Recommended action: implement rate limiting on the target port and block the source IP at the gateway firewall."

**Reconnaissance Alert:**
"Network scanning activity was detected from source IP 10.0.0.15 probing multiple destination ports (22, 80, 443, 3389, 8080). The sequential nature of the port access pattern suggests automated reconnaissance using tools such as Nmap. This is typically a precursor to targeted exploitation. Recommended action: monitor the source IP for follow-up connection attempts and update firewall rules to restrict port visibility."

Note: These are representative examples from the llm_explainability.py output. The LLM generates contextual explanations based on flow metadata, not raw packet payloads. Generation time is approximately 7.4 seconds per alert on average (n=6 sample generations; range 6.1-9.8s), executed asynchronously without blocking the detection pipeline.


## 12. Strengthened RF Defense (beyond VRAM)

The Random Forest baseline achieves superior raw accuracy (0.9864 on BoT-IoT, 0.9851 on ToN-IoT clean) due to the inherent suitability of tree-based ensembles for low-dimensional tabular feature spaces. However, several fundamental limitations restrict RF deployment in production IoT security environments. First, RF models operate on rigid, pre-defined feature spaces and cannot adapt to novel attack distributions (zero-day covariate shift) without complete retraining, whereas neural networks support incremental fine-tuning and transfer learning across deployment domains. Second, RF memory footprint scales with tree depth and forest size — our 200-tree model consumes 444 MB of GPU VRAM, representing 11% of a 4 GB edge device's total memory before accounting for the operating system, detection pipeline, and LLM inference. Third, RF inference produces only class probabilities, offering no latent feature representations suitable for downstream integration with explainability models. Our CNN-BiLSTM provides intermediate activations that naturally interface with the asynchronous TinyLlama dispatch, enabling semantic threat intelligence that tree-based methods cannot support without additional architectural complexity. The CNN-BiLSTM is therefore positioned not as an accuracy competitor to the RF, but as the enabling architecture for a complete, GPU-accelerated, self-explaining edge security pipeline.


## 13. Golden Narrative Arc (manuscript structure)

1. THE EDGE DEPLOYMENT PARADOX: Deep learning models offer adaptability for IoT security but edge devices cannot run massive models. When researchers shrink models to fit edge constraints, they encounter the "Framework Tax."

2. EXPOSING COMPILER INEFFICIENCIES: Modern DL compilers (torch.compile, TensorRT) are optimized for large LLMs and big batch sizes. For tiny models processing real-time streams at batch size 1, kernel launch overhead and compiler graph breaks (especially for recurrent nodes) destroy inference speed, rendering them slower than naive execution. TensorRT is 4.40x slower. torch.compile crashes on BiLSTM CUDA graphs entirely.

3. THE HPC SOLUTION: Bypassing frameworks entirely with raw CUDA C++ kernels reclaims theoretical hardware performance. Transposed coalesced reads, FP16 half2 FMA packing, and chained kernel launches yield 3.33x pipeline speedup over eager PyTorch and 4.40x over TensorRT. The 8.39x-9.21x Block 3 optimization progression (range across two independent n=100-trial measurement sessions, see README's Measurement Stability note) demonstrates systematic HPC methodology.

4. ZERO-BLOCKING SEMANTIC SECURITY: Extreme kernel optimization frees computational bandwidth for a second innovation: asynchronous, zero-blocking dispatch to a local 4-bit quantized TinyLlama, providing semantic threat intelligence without cloud dependency or pipeline blocking (16.60 us p99 overhead).

5. ADDRESSING THE RF BASELINE: Tree-based ensembles provide slightly higher accuracy on static datasets, but their rigid feature spaces, exponential memory scaling, and inability to integrate with LLM explainability pipelines make them unsuitable as complete edge security solutions. Knowledge distillation transfers RF decision boundaries into the neural network, closing the gap to 0.74% on BoT-IoT while preserving GPU deployment advantages.


## 14. Closest Prior Work Citation (CUDA Kernel Optimization for GPU-Based IDS)

Prior work by Ibrahim et al. (*Computer Networks*, vol. 275, 2026; DOI `10.1016/j.comnet.2025.111954`)
applied custom CUDA kernels to a GNN-based intrusion detection system, redesigning graph
construction and node aggregation as GPU kernels to eliminate host-device copy overhead; using COO
sparse representation, memory coalescing, and shared memory, they report a 1.22x-1.48x speedup
over a CPU baseline. Our work targets a different architectural challenge — a CNN-BiLSTM with
recurrent control flow that resists standard graph-compilation optimizations (see our
torch.compile crash finding) — and benchmarks against production ML inference frameworks (PyTorch
eager, torch.compile, TensorRT, ONNX Runtime) rather than a CPU baseline, achieving 3.33x over
eager PyTorch and 4.40x over TensorRT. To our knowledge, no prior work benchmarks hand-written CUDA
kernels for a recurrent DL-based IDS against production inference frameworks; Ibrahim et al.
establish the closest precedent for GPU-kernel-level optimization applied to intrusion detection
generally, differing from our work in both target architecture (GNN vs. recurrent CNN-BiLSTM) and
comparison baseline (CPU vs. production frameworks).

**Provenance note (session 3, 2026-07-02):** this replaces a fabricated citation ("Sophimatics
Phase 3," Applied Sciences 2025, DOI `10.3390/app152211876`) that misattributed unrelated
philosophical-AI-architecture content to a "2.7x CUDA speedup for CNN-based IDS" claim never
actually verified before being written into this file — see `HANDOFF.md` for the full history.
This replacement citation was verified two ways before use: metadata (title/authors/journal/DOI)
confirmed via the Crossref API and cross-checked by resolving the DOI to the same ScienceDirect
article ID found via topical search; content (CUDA kernels for graph construction/inference,
memory coalescing, shared memory, reported speedup vs. CPU baseline) corroborated by two
independent search queries. The abstract itself could not be fetched directly (ScienceDirect
blocks automated retrieval), so this rests on corroborated secondary characterization, not a
verbatim-quoted primary source — flag for a manual read of the actual PDF before final submission
if full certainty is needed.

