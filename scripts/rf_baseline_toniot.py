"""
COLIDE - Random Forest Baseline on ToN-IoT
Per-class comparison with CNN-BiLSTM.
"""

import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

X_train = np.load('data/processed_toniot/X_train.npy')
y_train = np.load('data/processed_toniot/y_train.npy')
X_test = np.load('data/processed_toniot/X_test.npy')
y_test = np.load('data/processed_toniot/y_test.npy')
le = joblib.load('data/processed_toniot/label_encoder.pkl')

print("=" * 60)
print("COLIDE - Random Forest Baseline (ToN-IoT)")
print("=" * 60)

print("\nTraining RF (100 trees)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, preds, target_names=le.classes_, digits=4))

macro_f1 = f1_score(y_test, preds, average="macro")
weighted_f1 = f1_score(y_test, preds, average="weighted")
acc = accuracy_score(y_test, preds)

print(f"Macro-F1:    {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")
print(f"Accuracy:    {acc:.4f}")

# Save results
results = {
    'model': 'RandomForest',
    'dataset': 'ToN-IoT',
    'n_estimators': 100,
    'macro_f1': float(macro_f1),
    'weighted_f1': float(weighted_f1),
    'accuracy': float(acc),
}
with open('benchmarks/results/rf_baseline_toniot.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to benchmarks/results/rf_baseline_toniot.json")
