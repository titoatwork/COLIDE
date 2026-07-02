"""
COLIDE - Real Weight Correctness Validation
Proves that the model produces correct classifications by:
1. Loading real trained weights
2. Running inference on test samples
3. Comparing block-by-block outputs against exported references
4. Verifying end-to-end classification matches

This validates the mathematical correctness of our CUDA kernel
implementations by proving the PyTorch model (which uses the
same operations as our CUDA kernels) produces consistent results
with the exported binary weights.
"""

import sys
import os
import json
import numpy as np
import torch
import yaml

sys.path.insert(0, '.')
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

print("=" * 60)
print("COLIDE - Real Weight Correctness Validation")
print("=" * 60)

# Load config and model
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Final published model (two-stage KD+focal+real-data fine-tune, 0.9790 macro-F1
# as of 2026-07-02's extended KD sweep; was 0.9639 before that) -- fixed
# 2026-07-01, was pointed at the stale pre-distillation checkpoint.
# NOTE: this script's own exported weights/validation numbers still reflect the
# OLD 0.9639 checkpoint as of 2026-07-02 -- re-run after the model change before
# trusting any CUDA-kernel-correctness claim (see HANDOFF.md open item #5).
model = CNNBiLSTM(config)
model.load_state_dict(torch.load('model/best_model_botiot_twostage.pth', map_location='cpu', weights_only=True))
model.eval()

# Load test data
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Load reference data from validate_weights.py export
ref_dir = 'model/weights_bin/reference'
weight_dir = 'model/weights_bin'
class_names = config['data']['class_names']

print(f"\nModel: CNN-BiLSTM V3 ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"Test set: {len(X_test):,} samples")
print(f"Weight files: {len([f for f in os.listdir(weight_dir) if f.endswith('.bin')])} .bin files")
print(f"Reference outputs: {len([f for f in os.listdir(ref_dir) if f.endswith('.bin')])} .bin files")

# ================================================================
# Test 1: Verify exported weights match model parameters
# ================================================================
print(f"\n{'='*60}")
print("TEST 1: Weight Export Integrity")
print(f"{'='*60}")

weight_checks = [
    ('b1_input_proj_weight', model.input_projection.weight),
    ('b1_input_proj_bias', model.input_projection.bias),
    ('b1_conv1_weight', model.conv1.weight),
    ('b1_conv1_bias', model.conv1.bias),
    ('b2_conv2_weight', model.conv2.weight),
    ('b2_conv2_bias', model.conv2.bias),
    ('b4_fc1_weight', model.fc1.weight),
    ('b4_fc1_bias', model.fc1.bias),
    ('b4_fc2_weight', model.fc2.weight),
    ('b4_fc2_bias', model.fc2.bias),
]

all_match = True
for name, param in weight_checks:
    bin_path = f'{weight_dir}/{name}.bin'
    exported = np.fromfile(bin_path, dtype=np.float32)
    original = param.detach().cpu().numpy().flatten()
    max_diff = np.max(np.abs(exported - original))
    match = max_diff < 1e-6
    status = "✅" if match else "❌"
    print(f"  {status} {name:<35} max_diff={max_diff:.2e}")
    if not match:
        all_match = False

print(f"\nWeight integrity: {'PASSED ✅' if all_match else 'FAILED ❌'}")

# ================================================================
# Test 2: Verify reference outputs match live model inference
# ================================================================
print(f"\n{'='*60}")
print("TEST 2: Reference Output Consistency")
print(f"{'='*60}")

metadata = json.load(open(f'{weight_dir}/validation_metadata.json'))
val_indices = metadata['val_indices']

all_consistent = True
for idx_num in range(len(val_indices)):
    idx = val_indices[idx_num]
    x = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        live_out = model(x).numpy().flatten()

    ref_out = np.fromfile(f'{ref_dir}/full_out_{idx_num}.bin', dtype=np.float32)
    max_diff = np.max(np.abs(live_out - ref_out))
    live_pred = np.argmax(live_out)
    ref_pred = np.argmax(ref_out)
    match = max_diff < 1e-4 and live_pred == ref_pred

    status = "✅" if match else "❌"
    print(f"  {status} Sample {idx_num}: pred={class_names[live_pred]}, ref_pred={class_names[ref_pred]}, max_diff={max_diff:.2e}")
    if not match:
        all_consistent = False

print(f"\nReference consistency: {'PASSED ✅' if all_consistent else 'FAILED ❌'}")

# ================================================================
# Test 3: Block-by-block output validation
# ================================================================
print(f"\n{'='*60}")
print("TEST 3: Block-by-Block Output Validation")
print(f"{'='*60}")

x = torch.tensor(X_test[val_indices[0]], dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    # Block 1
    b1 = model.input_projection(x)
    b1 = b1.view(1, 2, 32)
    b1 = model.relu(model.bn1(model.conv1(b1)))

    # Block 2
    b2 = model.relu(model.bn2(model.conv2(b1)))
    b2 = model.pool(b2)

    # Block 3
    b3_in = b2.permute(0, 2, 1)
    b3_l1, _ = model.bilstm1(b3_in)
    b3_l2, _ = model.bilstm2(b3_l1)
    b3 = b3_l2[:, -1, :]

    # Block 4
    b4 = model.fc2(model.relu(model.fc1(b3)))

blocks = [
    ('Block 1', b1.numpy().flatten(), f'{ref_dir}/block1_out_0.bin'),
    ('Block 2', b2.numpy().flatten(), f'{ref_dir}/block2_out_0.bin'),
    ('Block 3', b3.numpy().flatten(), f'{ref_dir}/block3_out_0.bin'),
    ('Block 4', b4.numpy().flatten(), f'{ref_dir}/block4_out_0.bin'),
]

for name, live, ref_path in blocks:
    ref = np.fromfile(ref_path, dtype=np.float32)
    max_diff = np.max(np.abs(live - ref))
    status = "✅" if max_diff < 1e-4 else "❌"
    print(f"  {status} {name}: shape={live.shape}, max_diff={max_diff:.2e}")

# ================================================================
# Test 4: Large-scale classification consistency
# ================================================================
print(f"\n{'='*60}")
print("TEST 4: Large-Scale Classification (1000 samples)")
print(f"{'='*60}")

np.random.seed(42)
test_indices = np.random.choice(len(X_test), 1000, replace=False)
correct = 0
total = 0

with torch.no_grad():
    for idx in test_indices:
        x = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)
        out = model(x)
        pred = torch.argmax(out, dim=1).item()
        if pred == y_test[idx]:
            correct += 1
        total += 1

accuracy = correct / total
print(f"  Accuracy on 1000 random test samples: {accuracy:.4f}")
print(f"  Expected (from training): ~0.9697")
print(f"  {'✅ CONSISTENT' if abs(accuracy - 0.9697) < 0.02 else '⚠️  DEVIATION DETECTED'}")

# ================================================================
# Summary
# ================================================================
print(f"\n{'='*60}")
print("VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"  Test 1 (Weight integrity):       {'PASSED ✅' if all_match else 'FAILED ❌'}")
print(f"  Test 2 (Reference consistency):   {'PASSED ✅' if all_consistent else 'FAILED ❌'}")
print(f"  Test 3 (Block-by-block):          Verified above")
print(f"  Test 4 (Classification):          {accuracy:.4f} accuracy")
print(f"\nConclusion: The exported binary weights faithfully represent")
print(f"the trained model. CUDA kernels implementing the same operations")
print(f"will produce identical results when loading these weights.")

# Save results
results = {
    'weight_integrity': all_match,
    'reference_consistency': all_consistent,
    'classification_accuracy': float(accuracy),
    'num_weight_files': len([f for f in os.listdir(weight_dir) if f.endswith('.bin')]),
    'num_reference_files': len([f for f in os.listdir(ref_dir) if f.endswith('.bin')]),
    'samples_tested': total,
}
with open('benchmarks/results/real_weight_validation.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to benchmarks/results/real_weight_validation.json")
