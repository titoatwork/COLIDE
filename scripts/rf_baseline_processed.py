"""
COLIDE - RF Baseline on Preprocessed Data (canonical source for README's
"CPU RF (sklearn), 0.9864, 200 trees" headline figure used in the RF
accuracy-gap claim)

This is the apples-to-apples RF comparison: it trains on the EXACT same
preprocessed splits (data/processed/*.npy, from preprocessing/preprocess_v2.py's
undersample -> SMOTE -> scale pipeline) that the CNN-BiLSTM itself trains and
is evaluated on -- unlike scripts/rf_baseline.py or the inline RF teacher in
scripts/train_distill.py, which each apply their OWN independent resampling
straight from the raw CSVs and therefore give different, not-directly-comparable
numbers (0.9768 and ~0.975 respectively; see HANDOFF.md for how those were
investigated and ruled out as the source of the 0.9864 figure).

Before this script existed, 0.9864 had been in README.md since the line was
first added with no traceable source (verified 2026-07-01 against git history,
DAILY_LOG.md, and three other RF configurations -- none reproduced it). The
user supplied the original ad-hoc terminal command that produced it; this
script saves that exact recipe permanently so it can't drift back into
unverified territory.

Usage:
    PYTHONPATH=. python scripts/rf_baseline_processed.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEED = 42

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)
test_pred = rf.predict(X_test)

val_macro_f1 = f1_score(y_val, val_pred, average="macro")
val_weighted_f1 = f1_score(y_val, val_pred, average="weighted")
test_macro_f1 = f1_score(y_test, test_pred, average="macro")
test_weighted_f1 = f1_score(y_test, test_pred, average="weighted")
test_acc = accuracy_score(y_test, test_pred)

print(f"Val   Macro-F1: {val_macro_f1:.4f} | Weighted-F1: {val_weighted_f1:.4f}")
print(f"Test  Macro-F1: {test_macro_f1:.4f} | Weighted-F1: {test_weighted_f1:.4f} | Accuracy: {test_acc:.4f}")

results = {
    "n_estimators": 200,
    "random_state": SEED,
    "data_source": "data/processed/*.npy (preprocessing/preprocess_v2.py: undersample -> SMOTE -> scale)",
    "train_shape": list(X_train.shape),
    "val_shape": list(X_val.shape),
    "test_shape": list(X_test.shape),
    "val_macro_f1": float(val_macro_f1),
    "val_weighted_f1": float(val_weighted_f1),
    "test_macro_f1": float(test_macro_f1),
    "test_weighted_f1": float(test_weighted_f1),
    "test_accuracy": float(test_acc),
}
out_path = PROJECT_ROOT / "benchmarks" / "results" / "rf_baseline_processed.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
