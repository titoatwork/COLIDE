"""
COLIDE

Preprocessing V2
Per-Flow Pipeline

No Windowing
No Sequence Construction

Pipeline:

Load
↓
Split
↓
Undersample
↓
SMOTE
↓
Scale
↓
Save
"""

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# Helpers
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Config
# ============================================================

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FEATURE_COLS = config["data"]["feature_columns"]
LABEL_COL = config["data"]["label_column"]

print_header("LOADING CONFIG")
print(f"Features: {len(FEATURE_COLS)}")
print(f"Label: {LABEL_COL}")

# ============================================================
# Load Data
# ============================================================

print_header("LOADING DATA")

TRAIN_PATH = RAW_DIR / "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
TEST_PATH = RAW_DIR / "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

print(f"Train Shape: {df_train.shape}")
print(f"Test Shape: {df_test.shape}")

# ============================================================
# Keep Required Columns
# ============================================================

required_cols = FEATURE_COLS + [LABEL_COL]

df_train = df_train[required_cols]
df_test = df_test[required_cols]

# ============================================================
# Label Encoding
# ============================================================

print_header("LABEL ENCODING")

label_encoder = LabelEncoder()

y_train_full = label_encoder.fit_transform(df_train[LABEL_COL])
y_test = label_encoder.transform(df_test[LABEL_COL])

X_train_full = df_train[FEATURE_COLS]
X_test = df_test[FEATURE_COLS]

print("Classes:")
for idx, name in enumerate(label_encoder.classes_):
    print(f"{idx}: {name}")

# ============================================================
# Train / Validation Split
# ============================================================

print_header("TRAIN / VALIDATION SPLIT")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.10,
    stratify=y_train_full,
    random_state=SEED,
)

print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")

# ============================================================
# Class Mapping
# ============================================================

class_map = {
    name: idx
    for idx, name in enumerate(label_encoder.classes_)
}

print_header("CLASS MAPPING")
for k, v in class_map.items():
    print(f"{k:<20} -> {v}")

# ============================================================
# Distribution Before Resampling
# ============================================================

print_header("TRAIN DISTRIBUTION BEFORE RESAMPLING")

before_distribution = {}
unique, counts = np.unique(y_train, return_counts=True)
for cls_id, count in zip(unique, counts):
    cls_name = label_encoder.classes_[cls_id]
    before_distribution[cls_name] = int(count)
    pct = (count / len(y_train)) * 100
    print(f"{cls_name:<20}{count:>10,} ({pct:.2f}%)")

# ============================================================
# Undersample DDoS / DoS
# ============================================================

print_header("UNDERSAMPLING MAJORITY CLASSES")

undersample_strategy = {
    class_map["DDoS"]: 100000,
    class_map["DoS"]: 100000,
}

rus = RandomUnderSampler(
    sampling_strategy=undersample_strategy,
    random_state=SEED,
)

X_train_resampled, y_train_resampled = rus.fit_resample(
    X_train,
    y_train,
)

print(f"Shape After Undersampling: {X_train_resampled.shape}")

# ============================================================
# MinMax Scaling
# ============================================================

print_header("MINMAX SCALING")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Scaler fitted on training data.")

# ============================================================
# SMOTE - Normal
# ============================================================

print_header("SMOTE - NORMAL")

smote_normal = SMOTE(
    sampling_strategy={class_map["Normal"]: 2000},
    k_neighbors=3,
    random_state=SEED,
)

X_train_scaled, y_train_resampled = smote_normal.fit_resample(
    X_train_scaled,
    y_train_resampled,
)

print(f"Shape After Normal SMOTE: {X_train_scaled.shape}")

# ============================================================
# SMOTE - Theft
# ============================================================

print_header("SMOTE - THEFT")

smote_theft = SMOTE(
    sampling_strategy={class_map["Theft"]: 1000},
    k_neighbors=2,
    random_state=SEED,
)

X_train_scaled, y_train_resampled = smote_theft.fit_resample(
    X_train_scaled,
    y_train_resampled,
)

print(f"Shape After Theft SMOTE: {X_train_scaled.shape}")

# ============================================================
# Distribution After Resampling
# ============================================================

print_header("TRAIN DISTRIBUTION AFTER RESAMPLING")

after_distribution = {}
unique, counts = np.unique(y_train_resampled, return_counts=True)
for cls_id, count in zip(unique, counts):
    cls_name = label_encoder.classes_[cls_id]
    after_distribution[cls_name] = int(count)
    pct = (count / len(y_train_resampled)) * 100
    print(f"{cls_name:<20}{count:>10,} ({pct:.2f}%)")

# ============================================================
# Save Distribution JSON
# ============================================================

distribution_json = {
    "before_resampling": before_distribution,
    "after_resampling": after_distribution,
}

with open(PROCESSED_DIR / "class_distribution.json", "w") as f:
    json.dump(distribution_json, f, indent=4)

print("\nSaved: class_distribution.json")

# ============================================================
# Class Weights
# ============================================================

print_header("CLASS WEIGHTS")

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_resampled),
    y=y_train_resampled,
)

weights = weights.astype(np.float32)

for idx, weight in enumerate(weights):
    print(f"Class {idx}: {weight:.4f}")

np.save(PROCESSED_DIR / "class_weights.npy", weights)

# ============================================================
# Save Artifacts
# ============================================================

print_header("SAVING ARTIFACTS")

np.save(
    PROCESSED_DIR / "X_train.npy",
    X_train_scaled.astype(np.float32),
)

np.save(
    PROCESSED_DIR / "y_train.npy",
    y_train_resampled.astype(np.int64),
)

np.save(
    PROCESSED_DIR / "X_val.npy",
    X_val_scaled.astype(np.float32),
)

np.save(
    PROCESSED_DIR / "y_val.npy",
    y_val.astype(np.int64),
)

np.save(
    PROCESSED_DIR / "X_test.npy",
    X_test_scaled.astype(np.float32),
)

np.save(
    PROCESSED_DIR / "y_test.npy",
    y_test.astype(np.int64),
)

joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
joblib.dump(label_encoder, PROCESSED_DIR / "label_encoder.pkl")

# ============================================================
# Final Summary
# ============================================================

print_header("PREPROCESSING COMPLETE")

print(f"X_train: {X_train_scaled.shape}")
print(f"y_train: {y_train_resampled.shape}")
print(f"X_val:   {X_val_scaled.shape}")
print(f"y_val:   {y_val.shape}")
print(f"X_test:  {X_test_scaled.shape}")
print(f"y_test:  {y_test.shape}")

print("\nSaved:")
print("X_train.npy")
print("y_train.npy")
print("X_val.npy")
print("y_val.npy")
print("X_test.npy")
print("y_test.npy")
print("scaler.pkl")
print("label_encoder.pkl")
print("class_weights.npy")
print("class_distribution.json")
