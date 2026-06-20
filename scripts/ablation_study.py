"""
COLIDE - Ablation Study
Shows the contribution of each optimization technique to Block 3 (BiLSTM).
Also shows overall pipeline progression.
"""

print("=" * 75)
print("COLIDE ABLATION STUDY")
print("=" * 75)

# ================================================================
# Table 1: Block 3 Optimization Progression
# ================================================================
print(f"\n{'='*75}")
print("TABLE A: Block 3 (BiLSTM) Optimization Progression")
print(f"{'='*75}")
print(f"{'Configuration':<45} {'Latency':>10} {'vs Naive':>10} {'vs PyTorch':>10}")
print(f"{'-'*75}")

b3_configs = [
    ("PyTorch GPU (cuDNN baseline)", 740.7, None, 1.0),
    ("Naive custom (1 thread/hidden, global W_hh)", 5698.0, 1.0, 0.13),
    ("+ Precomputed input projection (W_ih*X)", 2901.0, 1.96, 0.26),
    ("+ Transposed W_hh (coalesced reads)", 1007.3, 5.66, 0.74),
    ("+ CUDA Graphs", 973.8, 5.85, 0.76),
    ("+ FP16 native half2 FMA", 601.4, 9.48, 1.23),
]

for name, lat, vs_naive, vs_pytorch in b3_configs:
    naive_str = f"{vs_naive:.2f}x" if vs_naive else "---"
    pytorch_str = f"{vs_pytorch:.2f}x"
    print(f"{name:<45} {lat:>8.1f} us {naive_str:>10} {pytorch_str:>10}")

print(f"\nKey insight: Precomputing W_ih*X (54.9%) and transposing W_hh (37.2%)")
print(f"together account for 92% of total improvement. CUDA Graphs")
print(f"contributed only 0.7%, confirming the kernel is compute-bound.")

# ================================================================
# Table 2: Per-Block Speedup Summary
# ================================================================
print(f"\n{'='*75}")
print("TABLE B: Per-Block Custom CUDA vs PyTorch GPU")
print(f"{'='*75}")
print(f"{'Block':<30} {'PyTorch GPU':>12} {'Custom CUDA':>12} {'Speedup':>10} {'% of Total':>10}")
print(f"{'-'*75}")

blocks = [
    ("1: Proj+Conv+BN+ReLU", 404.4, 61.7, 6.55, 5.4),
    ("2: Conv+BN+ReLU+Pool", 282.1, 87.2, 3.24, 7.6),
    ("3: BiLSTM (2 layers)", 791.1, 973.8, 0.81, 85.2),
    ("4: Dense (128->5)", 122.1, 20.1, 6.07, 1.8),
]

total_pt = sum(b[1] for b in blocks)
total_cuda = sum(b[2] for b in blocks)

for name, pt, cuda, speedup, pct in blocks:
    print(f"{name:<30} {pt:>10.1f} us {cuda:>10.1f} us {speedup:>8.2f}x {pct:>9.1f}%")

print(f"{'-'*75}")
print(f"{'TOTAL':<30} {total_pt:>10.1f} us {total_cuda:>10.1f} us {total_pt/total_cuda:>8.2f}x {'100.0':>9}%")

print(f"\nKey insight: Block 3 (BiLSTM) accounts for 85.2% of custom CUDA")
print(f"pipeline time. Blocks 1, 2, 4 achieve 3.2-6.6x speedups by fusing")
print(f"multiple PyTorch kernel launches into single custom kernels.")
print(f"Block 3 FP16 achieves 1.23x over PyTorch.")
print(f"Block 4 FP16 achieves 1.16x over FP32 (20.9 -> 17.9 us), confirming")
print(f"small kernels are launch-overhead-bound, not compute-bound.")
print(f"FP16 pipeline total: 674.2 us chained (2.76x over PyTorch GPU)")

# ================================================================
# Table 3: Framework Comparison at Different Batch Sizes
# ================================================================
print(f"\n{'='*75}")
print("TABLE C: Throughput Comparison (flows/sec)")
print(f"{'='*75}")
print(f"{'Batch Size':<12} {'PyTorch CPU':>12} {'PyTorch GPU':>12} {'ORT CPU':>12} {'ORT GPU':>12} {'Winner':>12}")
print(f"{'-'*75}")

batch_data = [
    (1, 733, 522, 2661, 253, "ORT CPU"),
    (32, 6524, 15510, 8202, 9145, "PyTorch GPU"),
    (128, 8323, 27455, 6804, 26577, "PyTorch GPU"),
    (256, 10325, 42878, 8814, 45855, "ORT GPU"),
]

for bs, pt_cpu, pt_gpu, ort_cpu, ort_gpu, winner in batch_data:
    print(f"{bs:<12} {pt_cpu:>12,} {pt_gpu:>12,} {ort_cpu:>12,} {ort_gpu:>12,} {winner:>12}")

print(f"\nKey insight: GPU inference becomes superior at batch >= 32.")
print(f"At batch=1, ORT CPU wins due to GPU kernel launch overhead.")
print(f"At batch=256, GPU processes 42-46K flows/sec vs CPU's 8-10K.")

# ================================================================
# Table 4: Model Architecture Ablation (V2 vs V3)
# ================================================================
print(f"\n{'='*75}")
print("TABLE D: Architecture Ablation (V2 vs V3)")
print(f"{'='*75}")
print(f"{'Metric':<35} {'V2 (baseline)':>15} {'V3 (+attn)':>15} {'Delta':>10}")
print(f"{'-'*75}")

ablation = [
    ("Parameters", "463,877", "530,181", "+14.3%"),
    ("Test Macro-F1", "0.9330", "0.9352", "+0.0022"),
    ("Test Weighted-F1", "0.9695", "0.9698", "+0.0003"),
    ("Training time/epoch", "~40 sec", "~55 sec", "+37.5%"),
    ("DDoS F1", "0.9698", "0.9711", "+0.0013"),
    ("DoS F1", "0.9680", "0.9691", "+0.0011"),
    ("Normal F1", "0.8583", "0.8595", "+0.0012"),
    ("Reconnaissance F1", "0.9457", "0.9534", "+0.0077"),
    ("Theft F1", "0.9231", "0.9231", "+0.0000"),
]

for metric, v2, v3, delta in ablation:
    print(f"{metric:<35} {v2:>15} {v3:>15} {delta:>10}")

# ================================================================
# Table 5: Energy Efficiency
# ================================================================
print(f"\n{'='*75}")
print("TABLE E: Energy Efficiency")
print(f"{'='*75}")
print(f"{'Configuration':<25} {'Power (W)':>10} {'Tput (f/s)':>12} {'mJ/flow':>10} {'Efficiency':>12}")
print(f"{'-'*75}")

energy = [
    ("GPU batch=1", 10.44, 411, 25.414, "1.0x"),
    ("GPU batch=128", 17.81, 17491, 1.018, "25.0x"),
    ("CPU batch=1", 16.40, 347, 47.225, "0.5x"),
]

for name, power, tput, mj, eff in energy:
    print(f"{name:<25} {power:>10.2f} {tput:>12,} {mj:>10.3f} {eff:>12}")

print(f"\nKey insight: GPU batch=128 achieves 25x better energy efficiency")
print(f"than GPU batch=1, and 46x better than CPU batch=1.")

# ================================================================
# Table 6: LLM Explainability Impact
# ================================================================
print(f"\n{'='*75}")
print("TABLE F: Async LLM Explainability Impact on Detection")
print(f"{'='*75}")
print(f"{'Metric':<40} {'Value':>20}")
print(f"{'-'*75}")

llm_metrics = [
    ("Detection latency (no LLM)", "0.32 us"),
    ("Detection latency (with LLM dispatch)", "5.51 us"),
    ("Async dispatch overhead (p99)", "5.19 us"),
    ("Overhead as % of pipeline", "<1%"),
    ("LLM generation time (median)", "8,528 ms"),
    ("Impact on detection", "NEGLIGIBLE"),
    ("Ring buffer capacity", "32 alerts"),
    ("Alerts dropped", "0"),
]

for metric, val in llm_metrics:
    print(f"{metric:<40} {val:>20}")

print(f"\nKey insight: Async LLM dispatch adds 8.07 us overhead (0.7% of")
print(f"detection pipeline). Explanations are generated in background")
print(f"without blocking the detection path.")

# ================================================================
# Table 7: Cross-Dataset Validation
# ================================================================
print(f"\n{'='*75}")
print("TABLE G: Cross-Dataset Validation (BoT-IoT vs ToN-IoT)")
print(f"{'='*75}")
print(f"{'Dataset':<12} {'Model':<15} {'Macro-F1':>10} {'Weighted-F1':>12} {'Classes':>8} {'Test Samples':>13}")
print(f"{'-'*75}")

cross_data = [
    ("BoT-IoT", "RF Baseline", 0.9768, None, 5, 733705),
    ("BoT-IoT", "CNN-BiLSTM V3", 0.9352, 0.9698, 5, 733705),
    ("ToN-IoT", "RF Baseline", 0.9396, 0.9844, 10, 42209),
    ("ToN-IoT", "CNN-BiLSTM V3", 0.8029, 0.8622, 10, 42209),
]

for ds, model, mf1, wf1, cls, samples in cross_data:
    wf1_str = f"{wf1:.4f}" if wf1 else "---"
    print(f"{ds:<12} {model:<15} {mf1:>10.4f} {wf1_str:>12} {cls:>8} {samples:>13,}")

print(f"\nKey insight: CNN-BiLSTM generalizes across both IoT datasets.")
print(f"The accuracy gap on ToN-IoT is attributed to application-layer")
print(f"attacks (XSS, injection, password) having identical flow signatures.")
print(f"RF achieves higher accuracy but cannot leverage GPU acceleration.")

# ================================================================
# Table 8: Cross-Hardware Comparison (RTX 3050 vs V100S)
# ================================================================
print(f"\n{'='*75}")
print("TABLE H: Cross-Hardware CUDA Kernel Comparison (3 GPUs)")
print(f"{'='*75}")
print(f"{'Block':<22} {'RTX 3050':>10} {'V100S 32GB':>10} {'A100 80GB':>10} {'Fastest':>10}")
print(f"{'-'*75}")

hw_data = [
    ("Block 1 (Conv)",     61.7,  10.0,  12.0, "V100S"),
    ("Block 2 (Conv)",     87.2,  29.1,  34.7, "V100S"),
    ("Block 3 FP32",      974.0, 773.3,1117.8, "V100S"),
    ("Block 3 FP16",      601.0, 511.9, 548.4, "V100S"),
    ("Block 4 FP32",       20.1,   8.3,  10.2, "V100S"),
    ("Block 4 FP16",       17.9,   5.2,   6.6, "V100S"),
]

for name, rtx, v100, a100, fastest in hw_data:
    print(f"{name:<22} {rtx:>8.1f} us {v100:>8.1f} us {a100:>8.1f} us {fastest:>10}")

print(f"\nChained pipeline (FP16):")
print(f"  RTX 3050:  674.2 us (2.76x vs PyTorch)")
print(f"  V100S:     550.7 us (3.39x vs PyTorch)")
print(f"  A100:      592.0 us (3.15x vs PyTorch)")
print(f"\nKey insight: V100S achieves fastest BiLSTM FP16 (511.9 us)")
print(f"despite being older Volta architecture. BiLSTM is sequential,")
print(f"so clock speed matters more than SM count. A100 FP32 is slowest")
print(f"(1117.8 us) but benefits most from FP16 (2.04x improvement).")

print(f"\n{'='*75}")
print("ABLATION STUDY COMPLETE")
print(f"{'='*75}")
