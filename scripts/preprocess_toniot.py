"""
COLIDE - ToN-IoT Dataset Preprocessing
Matches the BoT-IoT preprocessing pipeline methodology:
  - Select numeric + categorical network flow features
  - Handle class imbalance (undersample majority, SMOTE minority)
  - MinMax normalization
  - Stratified Train/Val/Test split
  - Save as .npy files with config

Dataset: ToN-IoT (Moustafa, 2021)
Source: https://research.unsw.edu.au/projects/toniot-datasets
Classes: 10 (backdoor, ddos, dos, injection, mitm, normal, password, ransomware, scanning, xss)
Features: 13 (10 numeric flow statistics + 3 encoded categorical)

Usage: PYTHONPATH=. python scripts/preprocess_toniot.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

sys.path.insert(0, '.')

# ================================================================
# Configuration
# ================================================================
RAW_PATH = 'data/raw/toniot/train_test_network.csv'
OUT_DIR = 'data/processed_toniot'
SEED = 42

NUMERIC_FEATURES = [
    'duration',
    'src_bytes',
    'dst_bytes',
    'src_pkts',
    'dst_pkts',
    'src_ip_bytes',
    'dst_ip_bytes',
    'src_port',
    'dst_port',
    'missed_bytes',
]

CATEGORICAL_FEATURES = [
    'proto',
    'service',
    'conn_state',
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MAX_SAMPLES_PER_CLASS = 10000
MIN_SAMPLES_PER_CLASS = 5000
TEST_SIZE = 0.2
VAL_SIZE = 0.25
LABEL_COL = 'type'

# ================================================================
# Main
# ================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("COLIDE ToN-IoT PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(RAW_PATH)
    print(f"\nRaw data shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df[LABEL_COL].value_counts().to_string())

    X = df[NUMERIC_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    cat_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        cat_encoders[col] = le
        print(f"\n{col} categories ({len(le.classes_)}): {list(le.classes_[:10])}{'...' if len(le.classes_) > 10 else ''}")

    print(f"\nFeatures ({len(ALL_FEATURES)}): {ALL_FEATURES}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COL])
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {class_names}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X.values, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=SEED, stratify=y_temp)

    print(f"\nSplit sizes:")
    print(f"  Train: {X_train.shape[0]:>8,}")
    print(f"  Val:   {X_val.shape[0]:>8,}")
    print(f"  Test:  {X_test.shape[0]:>8,}")

    print(f"\nTrain distribution BEFORE balancing:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {class_names[u]:<15} {c:>6,}")

    indices = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        if len(cls_idx) > MAX_SAMPLES_PER_CLASS:
            chosen = np.random.RandomState(SEED).choice(cls_idx, MAX_SAMPLES_PER_CLASS, replace=False)
            indices.extend(chosen)
        else:
            indices.extend(cls_idx)
    X_train = X_train[indices]
    y_train = y_train[indices]

    smote_strategy = {}
    for cls in np.unique(y_train):
        count = (y_train == cls).sum()
        if count < MIN_SAMPLES_PER_CLASS:
            smote_strategy[cls] = MIN_SAMPLES_PER_CLASS

    if smote_strategy:
        print(f"\nApplying SMOTE for: {smote_strategy}")
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"\nTrain distribution AFTER balancing:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {class_names[u]:<15} {c:>6,}")
    print(f"  {'TOTAL':<15} {len(y_train):>6,}")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    np.save(f'{OUT_DIR}/X_train.npy', X_train)
    np.save(f'{OUT_DIR}/y_train.npy', y_train)
    np.save(f'{OUT_DIR}/X_val.npy', X_val)
    np.save(f'{OUT_DIR}/y_val.npy', y_val)
    np.save(f'{OUT_DIR}/X_test.npy', X_test)
    np.save(f'{OUT_DIR}/y_test.npy', y_test)
    joblib.dump(label_encoder, f'{OUT_DIR}/label_encoder.pkl')
    joblib.dump(scaler, f'{OUT_DIR}/scaler.pkl')

    config = {
        'seed': SEED,
        'data': {
            'dataset': 'ToN-IoT',
            'source': 'https://research.unsw.edu.au/projects/toniot-datasets',
            'feature_columns': ALL_FEATURES,
            'num_classes': num_classes,
            'class_names': class_names,
        },
        'model': {
            'input_features': len(ALL_FEATURES),
            'projection_dim': 64,
            'reshape': [2, 32],
            'cnn_filters_1': 64,
            'cnn_filters_2': 128,
            'cnn_kernel_size': 3,
            'pool_size': 2,
            'bilstm_units_1': 128,
            'bilstm_units_2': 64,
            'dense_units': 64,
            'dropout_rate': 0.3,
            'num_classes': num_classes,
            'attention_heads': 4,
            'attention_dropout': 0.1,
        },
        'training': {
            'batch_size': 256,
            'epochs': 50,
            'lr': 0.001,
            'early_stopping_patience': 10,
        },
    }

    with open(f'{OUT_DIR}/config_toniot.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nSaved to {OUT_DIR}/:")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:<30} {size:>10,} bytes")

    print("\nPreprocessing complete.")

if __name__ == '__main__':
    main()