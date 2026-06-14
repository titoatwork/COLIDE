"""
COLIDE - V2 vs V3 Model Comparison
Compares CNN-BiLSTM (V2) vs CNN-BiLSTM-Attention (V3)
As requested by Prof. Por Lip Yee.
"""

print("=" * 70)
print("COLIDE MODEL COMPARISON: V2 (last-timestep) vs V3 (self-attention)")
print("=" * 70)

# ================================================================
# Architecture Comparison
# ================================================================
print(f"\n{'='*70}")
print(f"{'ARCHITECTURE':<35} {'V2':>15} {'V3':>15}")
print(f"{'='*70}")
print(f"{'Parameters':<35} {'463,877':>15} {'530,181':>15}")
print(f"{'FP32 Size (MB)':<35} {'1.77':>15} {'2.02':>15}")
print(f"{'Pooling method':<35} {'Last timestep':>15} {'Self-attention':>15}")
print(f"{'Attention heads':<35} {'N/A':>15} {'4':>15}")
print(f"{'Extra parameters':<35} {'N/A':>15} {'+66,304':>15}")
print(f"{'Parameter overhead':<35} {'N/A':>15} {'+14.3%':>15}")

# ================================================================
# Accuracy Comparison
# ================================================================
print(f"\n{'='*70}")
print(f"{'ACCURACY (Test Set, 733,705 flows)':<35} {'V2':>15} {'V3':>15}")
print(f"{'='*70}")
print(f"{'Macro-F1':<35} {'0.9330':>15} {'0.9352':>15}")
print(f"{'Weighted-F1':<35} {'0.9695':>15} {'0.9698':>15}")
print(f"{'Accuracy':<35} {'0.9693':>15} {'0.9697':>15}")

# ================================================================
# Per-Class F1 Comparison
# ================================================================
print(f"\n{'='*70}")
print(f"{'PER-CLASS F1':<35} {'V2':>15} {'V3':>15} {'Delta':>10}")
print(f"{'='*70}")
classes = [
    ("DDoS (52.5%)", 0.9698, 0.9711),
    ("DoS (45.0%)", 0.9680, 0.9691),
    ("Normal (0.013%)", 0.8583, 0.8595),
    ("Reconnaissance (2.5%)", 0.9457, 0.9534),
    ("Theft (0.002%)", 0.9231, 0.9231),
]
for name, v2, v3 in classes:
    delta = v3 - v2
    sign = "+" if delta >= 0 else ""
    print(f"{name:<35} {v2:>15.4f} {v3:>15.4f} {sign}{delta:>9.4f}")

# ================================================================
# Training Comparison
# ================================================================
print(f"\n{'='*70}")
print(f"{'TRAINING':<35} {'V2':>15} {'V3':>15}")
print(f"{'='*70}")
print(f"{'Best epoch':<35} {'22':>15} {'28':>15}")
print(f"{'Best val macro-F1':<35} {'0.9418':>15} {'0.9633':>15}")
print(f"{'Total epochs (early stop)':<35} {'32':>15} {'38':>15}")
print(f"{'Epoch time (approx)':<35} {'~40 sec':>15} {'~55 sec':>15}")
print(f"{'Training overhead':<35} {'N/A':>15} {'+37.5%':>15}")

# ================================================================
# Inference Latency Impact
# ================================================================
print(f"\n{'='*70}")
print(f"{'INFERENCE IMPACT':<35} {'V2':>15} {'V3':>15}")
print(f"{'='*70}")
print(f"{'Attention adds to inference':<35} {'N/A':>15} {'~20-50 us':>15}")
print(f"{'CUDA kernel compatibility':<35} {'Full':>15} {'Partial':>15}")
print(f"{'Note: CUDA kernels implement V2 architecture (last-timestep selection)':<70}")

# ================================================================
# Recommendation
# ================================================================
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")
print("""
Self-attention (V3) provides consistent but marginal improvements:
  - Macro-F1: +0.0022 (0.9330 -> 0.9352)
  - Best improvement on Reconnaissance: +0.0077
  - No improvement on Theft (smallest class)
  - 14.3% more parameters, 37.5% longer training

The improvement is statistically small given the class imbalance.
The attention mechanism captures inter-timestep relationships in the
BiLSTM output, but since features are pre-aggregated per-flow statistics,
temporal dependencies are inherently limited.

For the paper, both architectures should be reported as an ablation study.
The V2 architecture is recommended for the CUDA kernel benchmarks since:
  1. All custom CUDA kernels are designed for V2
  2. The accuracy difference is negligible
  3. V2 has fewer parameters (lower inference cost)
  4. The attention layer would require an additional CUDA kernel
""")

print("Done.")