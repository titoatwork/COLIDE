"""
COLIDE - ToN-IoT Multi-Granularity Evaluation
Evaluates CNN-BiLSTM on ToN-IoT at three granularities:
  - 10-class fine-grained
  - 5-category grouped
  - Binary (normal vs attack)
"""

import json
import numpy as np
import torch
import yaml
import sys

sys.path.insert(0, '.')

from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.amp import autocast
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

print("=" * 60)
print("COLIDE - ToN-IoT Multi-Granularity Evaluation")
print("=" * 60)

with open('data/processed_toniot/config_toniot.yaml') as f:
    config = yaml.safe_load(f)

model = CNNBiLSTM(config).cuda()
model.load_state_dict(torch.load('model/best_model_toniot.pth', map_location='cuda', weights_only=True))
model.eval()

X_test = np.load('data/processed_toniot/X_test.npy')
y_test = np.load('data/processed_toniot/y_test.npy')

# Run inference
preds = []
with torch.no_grad():
    for i in range(0, len(X_test), 256):
        batch = torch.tensor(X_test[i:i+256], dtype=torch.float32).cuda()
        with autocast(device_type='cuda'):
            out = model(batch)
        preds.extend(torch.argmax(out, dim=1).cpu().numpy())
preds = np.array(preds)

class_names = config['data']['class_names']

# 1. 10-class fine-grained
print(f"\n{'='*60}")
print("10-CLASS FINE-GRAINED")
print(f"{'='*60}")
print(classification_report(y_test, preds, target_names=class_names, digits=4))
macro_f1_10 = f1_score(y_test, preds, average='macro')
weighted_f1_10 = f1_score(y_test, preds, average='weighted')
acc_10 = accuracy_score(y_test, preds)

# 2. 5-category grouped
# 0=backdoor→Malware, 1=ddos→DoS, 2=dos→DoS, 3=injection→Web,
# 4=mitm→Web, 5=normal→Normal, 6=password→Web, 7=ransomware→Malware,
# 8=scanning→Recon, 9=xss→Web
group_map = {0:2, 1:1, 2:1, 3:4, 4:4, 5:0, 6:4, 7:2, 8:3, 9:4}
group_names = ['Normal', 'DoS/DDoS', 'Malware', 'Recon', 'Web/Access']
y_grouped = np.array([group_map[y] for y in y_test])
preds_grouped = np.array([group_map[p] for p in preds])

print(f"\n{'='*60}")
print("5-CATEGORY GROUPED")
print(f"{'='*60}")
print(classification_report(y_grouped, preds_grouped, target_names=group_names, digits=4))
macro_f1_5 = f1_score(y_grouped, preds_grouped, average='macro')
weighted_f1_5 = f1_score(y_grouped, preds_grouped, average='weighted')
acc_5 = accuracy_score(y_grouped, preds_grouped)

# 3. Binary (normal vs attack)
# class 5 = normal
y_binary = (y_test != 5).astype(int)
preds_binary = (preds != 5).astype(int)

print(f"\n{'='*60}")
print("BINARY (Normal vs Attack)")
print(f"{'='*60}")
print(classification_report(y_binary, preds_binary, target_names=['Normal', 'Attack'], digits=4))
macro_f1_bin = f1_score(y_binary, preds_binary, average='macro')
weighted_f1_bin = f1_score(y_binary, preds_binary, average='weighted')
acc_bin = accuracy_score(y_binary, preds_binary)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Granularity':<20} {'Macro-F1':>10} {'Weighted-F1':>12} {'Accuracy':>10}")
print(f"{'-'*52}")
print(f"{'10-class':<20} {macro_f1_10:>10.4f} {weighted_f1_10:>12.4f} {acc_10:>10.4f}")
print(f"{'5-category':<20} {macro_f1_5:>10.4f} {weighted_f1_5:>12.4f} {acc_5:>10.4f}")
print(f"{'Binary':<20} {macro_f1_bin:>10.4f} {weighted_f1_bin:>12.4f} {acc_bin:>10.4f}")

# Save
results = {
    'dataset': 'ToN-IoT',
    'model': 'CNN-BiLSTM V3',
    '10_class': {'macro_f1': float(macro_f1_10), 'weighted_f1': float(weighted_f1_10), 'accuracy': float(acc_10)},
    '5_category': {'macro_f1': float(macro_f1_5), 'weighted_f1': float(weighted_f1_5), 'accuracy': float(acc_5)},
    'binary': {'macro_f1': float(macro_f1_bin), 'weighted_f1': float(weighted_f1_bin), 'accuracy': float(acc_bin)},
}
with open('benchmarks/results/toniot_multi_eval.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to benchmarks/results/toniot_multi_eval.json")
