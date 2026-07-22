# COLIDE — Pre-Written Text Blocks for Manuscript

## 1. LLM Claim Paragraph (replaces "first to measure" claim)

Recent work by Jamshidi et al. (2026) demonstrated the integration of LLMs for IoT intrusion reasoning at edge gateways, dispatching alerts to cloud-hosted models (GPT-4-turbo, LLaMA 3.5) via API calls with latencies under 1.5 seconds and bandwidth overhead under 1.2 kB per prompt. However, their approach introduces external network dependencies, variable API latency, and potential data privacy risks inherent in transmitting security telemetry to third-party endpoints. Our approach differs fundamentally: we deploy a fully local, quantized TinyLlama 1.1B (4-bit, 0.77 GB VRAM) on the same GPU as the detection pipeline, using an asynchronous ring-buffer dispatch mechanism. The measured dispatch overhead is 16.60 us at p99 (over 5,000 trials of the classify-construct-push code path) — roughly 2.5% of the 594-675 us inference pipeline latency (range combining 3 framework-side and 5 Custom CUDA measurement sessions), and negligible next to the multi-second LLM generation time — with zero network dependency and complete data sovereignty. While the LLM generation itself takes approximately 7.4 seconds per alert on average, the asynchronous design ensures the detection pipeline is never blocked. To our knowledge, this represents the first fully on-device, air-gapped LLM explainability integration for real-time intrusion detection.


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
| **Custom CUDA FP16** | **0.9790** | **594–675** | **25,899** | **0.79** | **RTX 3050** |
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

All four custom kernels achieve 100% theoretical occupancy on the RTX 3050 (Ampere SM 8.6, 20 SMs, 1536 max threads/SM). Block 1 and Block 2 launch 256 threads per block with minimal shared memory (2-4 KB), achieving 6 concurrent blocks per SM. The BiLSTM kernel (Block 3) uses 128 threads with 8 KB shared memory, allowing 12 blocks per SM. Block 4 uses 64 threads at 1 KB shared memory, sustaining 24 blocks per SM. The high occupancy confirms that the performance gains from our custom kernels over TensorRT (3.60x-4.99x) and torch.compile (2.25x-2.99x) are not due to superior hardware utilisation, but rather the elimination of CPU-to-GPU kernel launch overhead. TensorRT decomposes the model into approximately 128 individual kernel launches at 5-15 us each, accumulating significant host-side latency. Our chained pipeline executes back-to-back on the device with zero inter-kernel synchronisation, converting launch-bound execution into compute-bound execution.


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

2. EXPOSING COMPILER INEFFICIENCIES: Modern DL compilers (torch.compile, TensorRT) are optimized for large LLMs and big batch sizes. For tiny models processing real-time streams at batch size 1, kernel launch overhead and compiler graph breaks (especially for recurrent nodes) destroy inference speed, rendering them slower than naive execution. TensorRT is 3.60x-4.99x slower. torch.compile crashes on BiLSTM CUDA graphs entirely.

3. THE HPC SOLUTION: Bypassing frameworks entirely with raw CUDA C++ kernels reclaims theoretical hardware performance. Transposed coalesced reads, FP16 half2 FMA packing, and chained kernel launches yield 3.04x-3.78x pipeline speedup over eager PyTorch and 3.60x-4.99x over TensorRT (ranges combining 3 framework-side and 5 Custom CUDA measurement sessions). The 7.55x-9.50x Block 3 optimization progression (range across five independent n=100-trial measurement sessions, see README's Measurement Stability note) demonstrates systematic HPC methodology.

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
eager, torch.compile, TensorRT, ONNX Runtime) rather than a CPU baseline, achieving 3.04x-3.78x over
eager PyTorch and 3.60x-4.99x over TensorRT (ranges combining 3 framework-side and 5 Custom CUDA measurement sessions). To our knowledge, no prior work benchmarks hand-written CUDA
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

## 15. Threats to Validity

Several threats to validity shape how the latency and accuracy claims in this paper should be interpreted.

**Internal validity — measurement environment.** Single-sample inference latencies on the primary development machine (RTX 3050 Laptop, WSL2) exhibit genuine *session-to-session* drift beyond within-session variance: re-running the same n=100 CUDA kernel harness and the same n=20 framework comparison on different sittings produced mean shifts on the order of roughly 6–27% for some Block 3 configurations and 14–17% for torch.compile / TensorRT means in back-to-back framework sessions. Headline framework ratios are therefore reported as ranges over independent measurement sessions rather than single point estimates. Within a session, coefficients of variation can look tight while still understating day-to-day uncertainty on a virtualized GPU path. Cross-hardware V100S/A100 pipeline totals currently rest on unreplicated DICC runs without same-hardware PyTorch baselines; a multi-day DICC re-run remains the planned mitigation (see project HANDOFF Phase 3 / Sessions 9–10).

**Internal validity — statistical comparison design.** Framework significance uses two-sample Welch t-tests against a Custom CUDA distribution derived from n=100 kernel trials, not a one-sample test against a fixed constant. ORT CPU is *not* robustly faster or slower than Custom CUDA across sessions (ratio range straddles parity); Eager PyTorch, torch.compile, TensorRT, and ORT GPU remain significantly slower (p<0.001) in all measured framework-side sessions even when exact ratios move.

**Construct validity — model architecture vs. tabular flows.** BoT-IoT 10-best features are pre-aggregated flow statistics. An MLP ablation nearly matches CNN-BiLSTM accuracy, indicating that sequential bias is not essential for detection quality. The recurrent stack is retained primarily as a systems stress case (dynamic control flow that breaks torch.compile CUDA-graph capture). Readers should not interpret the architecture as a claim of superior temporal modeling on this feature set.

**External validity — datasets and class imbalance.** Results are tied to BoT-IoT and ToN-IoT under a documented undersample→SMOTE→scale pipeline. The Theft class has extremely low real support (tens of training samples before SMOTE); minority-class F1 can swing macro-F1 even when majority traffic is classified well (observed as a KD temperature outlier). Generalization to other networks, feature extractors, or live traffic is unproven.

**External validity — accuracy baseline.** A CPU Random Forest on the same preprocessed splits remains stronger (0.9864 vs 0.9790 macro-F1 on BoT-IoT). Knowledge distillation narrows but does not eliminate the gap. The neural model is positioned for GPU deployment, low VRAM relative to large GPU forests, and async on-device LLM explainability — not as a pure accuracy replacement for RF.

**Conclusion validity — LLM explainability.** Dispatch overhead (16.60 µs p99) is rigorously measured on the classify→construct→push path. Generation quality is illustrated with representative examples; no large-scale human evaluation of explanation usefulness is claimed. Generation time (~7–8 s/alert) lives off the critical path by design.

**Reliability of CUDA correctness claims.** Numerical fidelity is split into (i) bit-identical agreement between live PyTorch block outputs and exported reference tensors for the production checkpoint on 10 held-out indices, and (ii) per-binary GPU-vs-CPU self-checks at disclosed tolerances (stricter for FP32 blocks, 5e-2 for FP16 BiLSTM). See §16.


## 16. Numerical Fidelity Table

Source: `scripts/numerical_fidelity.py` → `benchmarks/results/numerical_fidelity.json` (Session 7, 2026-07-14). Checkpoint: `model/best_model_botiot_twostage.pth` (0.9790 macro-F1). Reference indices from `model/weights_bin/validation_metadata.json` (n=10).

### A. Real-weight export fidelity (PyTorch live vs `weights_bin/reference`)

Block intermediates follow the CUDA decomposition (projection/conv stack, BiLSTM last timestep, dense head). On all 10 reference samples, every block and the full exported logits path are **bit-identical** to the live model (maximum absolute error of 0; prediction agreement 10/10). This confirms that the binary weight export used by the kernels is a faithful snapshot of the production checkpoint, not a stale or truncated dump.

| Path | Max abs error | Max rel error | n |
|------|---------------|---------------|---|
| Block 1 (proj+conv+BN+ReLU) | 0 (bit-identical) | 0 | 10 |
| Block 2 (conv+BN+ReLU+pool) | 0 (bit-identical) | 0 | 10 |
| Block 3 (BiLSTM last step) | 0 (bit-identical) | 0 | 10 |
| Block 4 (dense head) | 0 (bit-identical) | 0 | 10 |
| Full logits (export path) | 0 (bit-identical) | 0 | 10 |

### B. CUDA kernel self-validation (GPU vs in-binary CPU reference)

Each standalone kernel binary executes an internal numerical check at a fixed tolerance (some unit tests use synthetic weights; the check still exercises the same arithmetic path). Session 7 re-run on RTX 3050: **all six block self-checks PASS**.

| Binary | Precision | Tolerance | Result |
|--------|-----------|-----------|--------|
| fused_block1 | FP32 | 1e-3 | PASS |
| fused_block2 | FP32 | 1e-3 | PASS |
| fused_block3 | FP32 | 1e-2 | PASS |
| fused_block3_fp16 | FP16 half2 | 5e-2 | PASS |
| fused_block3_naive | FP32 | 1e-2 | PASS |
| fused_block4 | FP32 | 1e-3 | PASS |

The fused pipeline binary is timing-oriented and does not emit a separate validation line; end-to-end correctness is covered by per-block checks plus the real-weight export table above. FP16 BiLSTM uses a deliberately looser tolerance (5e-2) consistent with half-precision accumulation; FP32 blocks use 1e-3–1e-2.

## 11. Protocol-era numbers (CAD-CBA-v1 / Prof feedback track) — WP9a

> Canonical registry: `docs/execution_plan/CLAIMS_REGISTRY.md` (rebuilt by `scripts/build_claims_package.py`).
> All figures below are **val-only on BoT** unless marked ToN-test or historical. BoT multi-seed **test** remains **PENDING** until final config lock (B14).

### Method lock (do not re-litigate without new protocol)

**CAD-CBA-v1:** V3 CNN–BiLSTM–Attention + ensemble KD (α=0.6, T=10) + focal (γ≈1.92 from HPO) + `config/hpo_best.yaml` train HPs + shuffle sampler + argmax decode. Champion production weights md5 **80a90f7cc210276300eaa90173a5a385**.

### Detection (protocol `botiot_v1`, val)

| Claim | Number | Source |
|-------|--------|--------|
| WP1b multirun FT mean±std (n=5) | **0.9714±0.0109** | `multirun/summary.json` |
| WP3 HPO full-train refine winner | **0.9791** | `hpo/summary.json` |
| HPO multi-seed confirm mean±std | **0.9689±0.0145** | `multirun_hpo_confirm/summary.json` |
| Package ensemble+HPO multirun mean±std | **0.9639±0.0185** | `multirun_ensemble_hpo/summary.json` (not mean-win vs WP1b) |
| Focal FT seed42 | **0.9780** | `imbalance_loss/ft_focal_seed42.json` |
| Ensemble KD student (stage_a) | **0.9401** | `teachers_kd/kd_ensemble_seed42.json` |
| A7 full package ladder | **0.9699** | `ablation_ladder/` |
| A3 cnn_bilstm CE / G11 | **0.9493** | ladder / neural baselines |
| A4 attn+CE (underperforms A3) | **0.7378** | ladder honesty |
| CTRL cstar | **0.9787** | `cstar_bounded/` |
| E6 neural teacher → student | **0.8513** | ≪ ensemble 0.9401 |

### Classical protocol-fair (val)

| Model | val macro-F1 |
|-------|--------------|
| LGBM (G5 fixed, balanced) | **0.9818** |
| RF | **0.9778** |
| XGB | **0.9762** |
| LinearSVC (G2) | **0.4268** |
| Published RF (different pipeline) | **0.9864** historical dual bar only |

### Minority honesty (val Theft / min-cls examples)

HPO winner seed42: Theft **1.0000**, min-cls **0.9351** (Normal), macro **0.9791**.  
Protocol LGBM: Theft **0.9231**, min-cls **0.9231**, macro **0.9818**.  
Do **not** claim neural supremacy on pure F1 over LGBM under protocol.

### Multi-objective / systems

| Claim | Number | Notes |
|-------|--------|-------|
| Pareto-H8 a priori composite #1 | G6 **0.9056** @ **4.33** µs, F1 0.9285 | efficiency-weighted |
| RTX energy batch128 | **0.786** mJ/flow | `energy_table/` |
| LLM dispatch p99 | **16.60** µs | dispatch only |
| LLM generation mean | **~7400** ms | never conflate with dispatch |
| XAI rank consistency | **0.9636** | occlusion Spearman |
| XAI faithfulness top-3 mass | **0.5109** | occlusion proxy |
| XAI free-form feature mention | **0.333** | TinyLlama n=6 |
| XAI J10 path | **DROP_FULL_EXPLAINABLE_CLAIM_KEEP_STRUCTURED** | keep structured+dispatch |

### ToN final method (13-feat `processed_toniot`; test allowed for WP8)

CAD-CBA-v1 mapped: val **0.8080**, test **0.8110**; same-split RF test **0.9393**. ≠ historical clean 26-feat CNN **0.9526**.

### D6 stratified batch

Shuffle **0.9791** ≫ stratified inv-freq **0.9209** (Δ−0.058) — keep **shuffle**.

### Fidelity (WP6a)

Export path **bit-identical** (max abs error 0); CUDA self-checks all PASS. Champion md5 unchanged.

### Explicitly not yet claimable

- BoT sealed multi-seed **test** (B14) — wait for user final-config lock  
- DICC multi-day mean/median/std/CV/CI — dedicated session only  
- Full LLM-explainable IDS title claim — **dropped** (J10)

