"""
COLIDE - Real Weight Validation
Validates that custom CUDA kernels produce correct output when using
actual trained model weights (not random weights).

Approach: Export model weights to binary files, load in CUDA, compare outputs.
This script exports weights and generates reference outputs.
"""

import sys
import os
import struct
import numpy as np
import yaml
import torch

sys.path.insert(0, '.')
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Final published model (two-stage KD+focal+real-data fine-tune, 0.9790 macro-F1
# as of 2026-07-02's extended KD sweep; was 0.9639 before that).
# Previously pointed at model/best_model.pth, the pre-distillation "Original V3"
# checkpoint (0.9352) -- fixed 2026-07-01 so the exported weights used for CUDA
# kernel correctness validation match the model actually being reported.
# Re-exported 2026-07-02 (session 3) against the current 0.9790 checkpoint.
model = CNNBiLSTM(config)
model.load_state_dict(torch.load('model/best_model_botiot_twostage.pth', map_location='cpu', weights_only=True))
model.eval()

print("=" * 60)
print("COLIDE REAL WEIGHT VALIDATION")
print("=" * 60)

# ================================================================
# 1. Export weights as raw binary files (for easy C++ loading)
# ================================================================
weight_dir = 'model/weights_bin'
os.makedirs(weight_dir, exist_ok=True)

def export_tensor(name, tensor):
    """Export PyTorch tensor as raw float32 binary."""
    path = os.path.join(weight_dir, name + '.bin')
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(path)
    print(f"  {name:<40} shape={str(list(arr.shape)):<20} size={arr.nbytes} bytes")
    return arr

print("\n--- Exporting Block 1 weights ---")
export_tensor('b1_input_proj_weight', model.input_projection.weight)
export_tensor('b1_input_proj_bias', model.input_projection.bias)
export_tensor('b1_conv1_weight', model.conv1.weight)
export_tensor('b1_conv1_bias', model.conv1.bias)
export_tensor('b1_bn1_weight', model.bn1.weight)
export_tensor('b1_bn1_bias', model.bn1.bias)
export_tensor('b1_bn1_running_mean', model.bn1.running_mean)
export_tensor('b1_bn1_running_var', model.bn1.running_var)

print("\n--- Exporting Block 2 weights ---")
export_tensor('b2_conv2_weight', model.conv2.weight)
export_tensor('b2_conv2_bias', model.conv2.bias)
export_tensor('b2_bn2_weight', model.bn2.weight)
export_tensor('b2_bn2_bias', model.bn2.bias)
export_tensor('b2_bn2_running_mean', model.bn2.running_mean)
export_tensor('b2_bn2_running_var', model.bn2.running_var)

print("\n--- Exporting Block 3 weights (BiLSTM Layer 1) ---")
export_tensor('b3_bilstm1_weight_ih_l0', model.bilstm1.weight_ih_l0)
export_tensor('b3_bilstm1_weight_hh_l0', model.bilstm1.weight_hh_l0)
export_tensor('b3_bilstm1_bias_ih_l0', model.bilstm1.bias_ih_l0)
export_tensor('b3_bilstm1_bias_hh_l0', model.bilstm1.bias_hh_l0)
export_tensor('b3_bilstm1_weight_ih_l0_reverse', model.bilstm1.weight_ih_l0_reverse)
export_tensor('b3_bilstm1_weight_hh_l0_reverse', model.bilstm1.weight_hh_l0_reverse)
export_tensor('b3_bilstm1_bias_ih_l0_reverse', model.bilstm1.bias_ih_l0_reverse)
export_tensor('b3_bilstm1_bias_hh_l0_reverse', model.bilstm1.bias_hh_l0_reverse)

print("\n--- Exporting Block 3 weights (BiLSTM Layer 2) ---")
export_tensor('b3_bilstm2_weight_ih_l0', model.bilstm2.weight_ih_l0)
export_tensor('b3_bilstm2_weight_hh_l0', model.bilstm2.weight_hh_l0)
export_tensor('b3_bilstm2_bias_ih_l0', model.bilstm2.bias_ih_l0)
export_tensor('b3_bilstm2_bias_hh_l0', model.bilstm2.bias_hh_l0)
export_tensor('b3_bilstm2_weight_ih_l0_reverse', model.bilstm2.weight_ih_l0_reverse)
export_tensor('b3_bilstm2_weight_hh_l0_reverse', model.bilstm2.weight_hh_l0_reverse)
export_tensor('b3_bilstm2_bias_ih_l0_reverse', model.bilstm2.bias_ih_l0_reverse)
export_tensor('b3_bilstm2_bias_hh_l0_reverse', model.bilstm2.bias_hh_l0_reverse)

print("\n--- Exporting Block 4 weights ---")
export_tensor('b4_fc1_weight', model.fc1.weight)
export_tensor('b4_fc1_bias', model.fc1.bias)
export_tensor('b4_fc2_weight', model.fc2.weight)
export_tensor('b4_fc2_bias', model.fc2.bias)

# ================================================================
# 2. Generate reference outputs for test inputs
# ================================================================
print("\n--- Generating PyTorch Reference Outputs ---")

X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Use 10 specific test samples for validation
np.random.seed(42)
val_indices = np.random.choice(len(X_test), 10, replace=False)

ref_dir = 'model/weights_bin/reference'
os.makedirs(ref_dir, exist_ok=True)

# Run each sample through PyTorch and save intermediate outputs
for idx_num, idx in enumerate(val_indices):
    x = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # Block 1
        b1_out = model.input_projection(x)
        b1_out = b1_out.view(1, 2, 32)
        b1_out = model.relu(model.bn1(model.conv1(b1_out)))
        
        # Block 2
        b2_out = model.relu(model.bn2(model.conv2(b1_out)))
        b2_out = model.pool(b2_out)
        
        # Block 3
        b3_in = b2_out.permute(0, 2, 1)  # (1, 16, 128)
        b3_lstm1, _ = model.bilstm1(b3_in)
        b3_lstm2, _ = model.bilstm2(b3_lstm1)
        b3_out = b3_lstm2[:, -1, :]  # last timestep (V2 style)
        
        # Block 4
        b4_out = model.dropout(model.relu(model.fc1(b3_out)))
        b4_out = model.fc2(b4_out)
        
        # Full model
        full_out = model(x)
        pred = torch.argmax(full_out, dim=1).item()
        conf = torch.softmax(full_out, dim=1).max().item()
    
    # Save input and block outputs
    x.numpy().astype(np.float32).tofile(f'{ref_dir}/input_{idx_num}.bin')
    b1_out.numpy().astype(np.float32).tofile(f'{ref_dir}/block1_out_{idx_num}.bin')
    b2_out.numpy().astype(np.float32).tofile(f'{ref_dir}/block2_out_{idx_num}.bin')
    b3_out.numpy().astype(np.float32).tofile(f'{ref_dir}/block3_out_{idx_num}.bin')
    b4_out.numpy().astype(np.float32).tofile(f'{ref_dir}/block4_out_{idx_num}.bin')
    full_out.numpy().astype(np.float32).tofile(f'{ref_dir}/full_out_{idx_num}.bin')
    
    class_names = config['data']['class_names']
    print(f"  Sample {idx_num}: input[{idx}] -> {class_names[pred]} ({conf:.4f})")

# ================================================================
# 3. Save validation metadata
# ================================================================
import json
metadata = {
    'num_samples': 10,
    'val_indices': val_indices.tolist(),
    'weight_dir': weight_dir,
    'reference_dir': ref_dir,
    'block_shapes': {
        'input': [1, 10],
        'block1_out': [1, 64, 32],
        'block2_out': [1, 128, 16],
        'block3_out': [1, 128],
        'block4_out': [1, 5],
    },
    'model_version': 'V3 (attention, using V2 last-timestep for CUDA)',
}
with open(f'{weight_dir}/validation_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n--- Summary ---")
print(f"Weights exported to: {weight_dir}/")
print(f"Reference outputs:   {ref_dir}/")
print(f"Total binary files:  {len(os.listdir(weight_dir)) + len(os.listdir(ref_dir))}")
print(f"\nTo validate CUDA kernels:")
print(f"  1. Load .bin weights in C++ (fread, sizeof(float))")
print(f"  2. Feed reference inputs through CUDA pipeline")
print(f"  3. Compare CUDA output vs reference .bin files")
print(f"  4. Tolerance: 1e-3 for FP32, 5e-2 for FP16")
print("Done.")