"""
COLIDE - Cross-Dataset Comparison
Compares CNN-BiLSTM and RF performance across BoT-IoT and ToN-IoT.
"""

print("=" * 70)
print("COLIDE CROSS-DATASET COMPARISON")
print("=" * 70)

print()
print(f"{'Dataset':<11} | {'Model':<13} | {'Macro-F1':>8} | {'Weighted-F1':>11} | {'Classes':>7} | {'Samples':>8}")
print("-" * 70)
print(f"{'BoT-IoT':<11} | {'RF Baseline':<13} | {'0.9768':>8} | {'---':>11} | {'5':>7} | {'733,705':>8}")
print(f"{'BoT-IoT':<11} | {'CNN-BiLSTM':<13} | {'0.9352':>8} | {'0.9698':>11} | {'5':>7} | {'733,705':>8}")
print(f"{'ToN-IoT':<11} | {'RF Baseline':<13} | {'0.9396':>8} | {'0.9844':>11} | {'10':>7} | {'42,209':>8}")
print(f"{'ToN-IoT':<11} | {'CNN-BiLSTM':<13} | {'0.8029':>8} | {'0.8622':>11} | {'10':>7} | {'42,209':>8}")

print()
print("CUDA Inference (architecture-independent):")
print("-" * 70)
print("Pipeline FP32:  1,143 us (1.63x over PyTorch GPU)")
print("Pipeline FP16:    770 us (2.42x over PyTorch GPU)")
print("Throughput:    25,410 flows/sec (GPU batched)")
print("Energy:        1.02 mJ/flow (46x over CPU)")
print("LLM overhead:  8.07 us (0.7% of pipeline)")

print()
print("Key Finding: CUDA kernel speedups are architecture-level")
print("optimizations that apply equally to both datasets since")
print("the model structure is identical.")
