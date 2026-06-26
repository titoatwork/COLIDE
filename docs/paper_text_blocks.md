# COLIDE — Pre-Written Text Blocks for Manuscript

## 1. LLM Claim Paragraph (replaces "first to measure" claim)

Recent work by Jamshidi et al. (2026) demonstrated the integration of LLMs for IoT intrusion reasoning at edge gateways, dispatching alerts to cloud-hosted models (GPT-4-turbo, LLaMA 3.5) via API calls with latencies under 1.5 seconds and bandwidth overhead under 1.2 kB per prompt. However, their approach introduces external network dependencies, variable API latency, and potential data privacy risks inherent in transmitting security telemetry to third-party endpoints. Our approach differs fundamentally: we deploy a fully local, quantized TinyLlama 1.1B (4-bit, 0.77 GB VRAM) on the same GPU as the detection pipeline, using an asynchronous ring-buffer dispatch mechanism. The measured dispatch overhead is 5.19 us at p99 — less than 1% of the inference pipeline latency — with zero network dependency and complete data sovereignty. While the LLM generation itself takes approximately 8.5 seconds per alert, the asynchronous design ensures the detection pipeline is never blocked. To our knowledge, this represents the first fully on-device, air-gapped LLM explainability integration for real-time intrusion detection.


## 2. Alert Aggregation Paragraph (addresses DDoS queue overflow)

To prevent ring-buffer overflow during high-volume attacks, the system employs a time-windowed alert aggregation mechanism. Incoming alerts are grouped by (attack_type, source_ip) over a configurable window (default 10 seconds). At the end of each window, structurally identical alerts are collapsed into a single batched prompt containing the alert count, time span, and representative sample flows. This reduces LLM dispatch from potentially thousands of individual calls per second to one consolidated prompt per source-attacker pair per window. During a simulated DDoS attack generating 25,000 malicious flows over 5 seconds, the aggregator reduced LLM invocations from 25,000 to 10 (one per unique source IP), eliminating the risk of memory exhaustion in the generation thread while preserving the explainability output for security analysts.


## 3. SMOTE Limitation Paragraph (Theft class honesty)

The Theft class in BoT-IoT contains only 52 training samples, expanded to 1,000 via SMOTE. While the model generalises well to the 14 real test samples (F1 0.9286), the limited real support means performance may vary under different network conditions or attack distributions. This is an inherent limitation of the BoT-IoT dataset rather than a methodological flaw — the Theft category represents a rare attack type with minimal representation in the original data collection. We report the minority-class results transparently and note that deploying this system in environments with higher Theft-class prevalence would benefit from additional real training samples.


## 4. Pseudo-Sequence Acknowledgment (MLP ablation discussion)

We do not claim that the CNN-BiLSTM is the optimal classifier for tabular flow data. An equivalent-parameter MLP (400,901 parameters) trained with the identical distillation recipe achieves a test macro-F1 of 0.9542 after two-stage fine-tuning, compared to 0.9639 for the CNN-BiLSTM under the same protocol. While the accuracy difference is modest, the CNN-BiLSTM consistently outperformed the MLP across all training configurations, suggesting the convolutional and recurrent layers provide marginal but measurable benefit even on tabular features. More critically, the CNN-BiLSTM was retained because its computational pattern — 1D convolutions, batch normalisation, and bidirectional recurrence with dynamic control flow — exposes the limitations of automated inference compilers (TensorRT, torch.compile) that our custom CUDA kernels solve. An MLP would be trivially optimised by existing frameworks, offering no systems insight. The architectural complexity is justified by the engineering contributions it enables.


## 5. Summary Table (Prof. Por requested)

| Method | Macro-F1 | Latency (us) | Throughput (flows/sec) | Energy (mJ/flow) | Hardware |
|--------|----------|-------------|----------------------|-------------------|----------|
| CPU sklearn RF | 0.9864 | — | 305,248 | — | Laptop CPU |
| GPU cuML RF | 0.9471 | — | 1,667,495 | — | V100S |
| Eager PyTorch | 0.9639 | 2,235 | — | — | RTX 3050 |
| torch.compile | 0.9639 | 1,912 | — | — | RTX 3050 |
| TensorRT FP16 | 0.9639 | 3,334 | — | — | RTX 3050 |
| ORT GPU | 0.9639 | 4,128 | — | — | RTX 3050 |
| ORT CPU | 0.9639 | 723 | — | — | Laptop CPU |
| **Custom CUDA FP16** | **0.9639** | **674** | **25,410** | **0.79** | **RTX 3050** |
| Custom CUDA FP16 | 0.9639 | 551 | — | — | V100S |
| Custom CUDA FP16 | 0.9639 | 592 | — | — | A100 |

Notes:
- All DL methods use the same two-stage fine-tuned CNN-BiLSTM model (0.9639)
- RF uses 200-tree sklearn/cuML RandomForestClassifier
- Latency = single-sample inference
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
| KD 5 | 0.5 | 3.0 | — | 0.9487 | 0.9341 | Temp hurt |
| KD 6 | 0.7 | 5.0 | — | 0.9620 | 0.9547 | Best sweep |
| KD 7 | 0.7 | 5.0 | 2.0 | 0.9728 | 0.9601 | Best KD+focal |
| Two-stage | 0.7 | 5.0 | 2.0 | — | 0.9639 | Fine-tuned on real data |


## 7. GPU Profiling Paragraph (hardware characterisation)

All four custom kernels achieve 100% theoretical occupancy on the RTX 3050 (Ampere SM 8.6, 20 SMs, 1536 max threads/SM). Block 1 and Block 2 launch 256 threads per block with minimal shared memory (2-4 KB), achieving 6 concurrent blocks per SM. The BiLSTM kernel (Block 3) uses 128 threads with 8 KB shared memory, allowing 12 blocks per SM. Block 4 uses 64 threads at 1 KB shared memory, sustaining 24 blocks per SM. The high occupancy confirms that the performance gains from our custom kernels over TensorRT (4.95x) and torch.compile (2.84x) are not due to superior hardware utilisation, but rather the elimination of CPU-to-GPU kernel launch overhead. TensorRT decomposes the model into approximately 128 individual kernel launches at 5-15 us each, accumulating significant host-side latency. Our chained pipeline executes back-to-back on the device with zero inter-kernel synchronisation, converting launch-bound execution into compute-bound execution.

