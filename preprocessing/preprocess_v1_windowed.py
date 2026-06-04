# preprocessing/preprocess.py

"""
COLIDE
Preprocessing Pipeline

Steps
------
1. Load config
2. Load train/test CSVs
3. Drop non-feature columns
4. Encode labels
5. Stratified train/validation split
6. Resample TRAIN ONLY
7. Fit MinMax scaler on TRAIN ONLY
8. Transform train/val/test
9. Window all datasets
10. Save outputs
"""

import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Utility Functions
# ============================================================

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 ** 2)


def create_windows(X, y, window_size):
    """
    Create sliding windows.

    Window label = label of last flow
    in the window.
    """

    X_windows = []
    y_windows = []

    total = len(X)

    for i in range(total - window_size + 1):

        X_windows.append(
            X[i:i + window_size]
        )

        y_windows.append(
            y[i + window_size - 1]
        )

    X_windows = np.asarray(
        X_windows,
        dtype=np.float32
    )

    y_windows = np.asarray(
        y_windows,
        dtype=np.int64
    )

    return X_windows, y_windows


def print_class_distribution(y, encoder, title):

    print_header(title)

    counts = Counter(y)

    total = len(y)

    for cls_idx in sorted(counts.keys()):

        cls_name = encoder.inverse_transform(
            [cls_idx]
        )[0]

        pct = (
            counts[cls_idx] / total
        ) * 100

        print(
            f"{cls_name:<20}"
            f"{counts[cls_idx]:>12,}"
            f" ({pct:.2f}%)"
        )


# ============================================================
# Main
# ============================================================

def main():

    print_header("COLIDE PREPROCESSING PIPELINE")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    config_path = (
        project_root /
        "config" /
        "config.yaml"
    )

    processed_dir = (
        project_root /
        "data" /
        "processed"
    )

    raw_dir = (
        project_root /
        "data" /
        "raw"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    print_header("LOADING CONFIG")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    feature_columns = (
        config["data"]["feature_columns"]
    )

    label_column = (
        config["data"]["label_column"]
    )

    window_size = (
        config["preprocessing"]["window_size"]
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Label: {label_column}"
    )

    print(
        f"Window Size: {window_size}"
    )

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    print_header("LOADING DATA")

    train_path = (
        raw_dir /
        "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
    )

    test_path = (
        raw_dir /
        "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"
    )

    train_df = pd.read_csv(
        train_path
    )

    test_df = pd.read_csv(
        test_path
    )

    print(
        f"Train Shape: {train_df.shape}"
    )

    print(
        f"Test Shape: {test_df.shape}"
    )

    # --------------------------------------------------------
    # Features / Labels
    # --------------------------------------------------------

    X_train_full = train_df[
        feature_columns
    ].copy()

    y_train_full = train_df[
        label_column
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        label_column
    ].copy()

    # --------------------------------------------------------
    # Label Encoding
    # --------------------------------------------------------

    print_header("LABEL ENCODING")

    label_encoder = LabelEncoder()

    y_train_full = (
        label_encoder.fit_transform(
            y_train_full
        )
    )

    y_test = (
        label_encoder.transform(
            y_test
        )
    )

    joblib.dump(
        label_encoder,
        processed_dir /
        "label_encoder.pkl"
    )

    print(
        "Classes:",
        list(label_encoder.classes_)
    )

    # --------------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------------

    print_header(
        "TRAIN / VALIDATION SPLIT"
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.10,
            random_state=SEED,
            stratify=y_train_full
        )
    )

    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"Val: {X_val.shape}"
    )

    # --------------------------------------------------------
    # Before Resampling
    # --------------------------------------------------------

    print_class_distribution(
        y_train,
        label_encoder,
        "TRAIN DISTRIBUTION BEFORE RESAMPLING"
    )

    # --------------------------------------------------------
    # Custom Sampling Strategy
    # --------------------------------------------------------

    class_mapping = {
        name: idx
        for idx, name
        in enumerate(
            label_encoder.classes_
        )
    }

    undersample_strategy = {
        class_mapping["DDoS"]: 100000,
        class_mapping["DoS"]: 100000
    }

    smote_strategy = {
        class_mapping["Normal"]: 20000,
        class_mapping["Theft"]: 10000
    }

    # --------------------------------------------------------
    # Undersample
    # --------------------------------------------------------

    print_header(
        "UNDERSAMPLING MAJORITY CLASSES"
    )

    rus = RandomUnderSampler(
        sampling_strategy=undersample_strategy,
        random_state=SEED
    )

    X_train, y_train = rus.fit_resample(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # SMOTE Normal
    # --------------------------------------------------------

    print_header(
        "SMOTE NORMAL CLASS"
    )

    smote_normal = SMOTE(
        sampling_strategy={
            class_mapping["Normal"]: 20000
        },
        k_neighbors=3,
        random_state=SEED
    )

    X_train, y_train = (
        smote_normal.fit_resample(
            X_train,
            y_train
        )
    )

    # --------------------------------------------------------
    # SMOTE Theft
    # --------------------------------------------------------

    print_header(
        "SMOTE THEFT CLASS"
    )

    smote_theft = SMOTE(
        sampling_strategy={
            class_mapping["Theft"]: 10000
        },
        k_neighbors=2,
        random_state=SEED
    )

    X_train, y_train = (
        smote_theft.fit_resample(
            X_train,
            y_train
        )
    )

    print_class_distribution(
        y_train,
        label_encoder,
        "TRAIN DISTRIBUTION AFTER RESAMPLING"
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    print_header("MINMAX SCALING")

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    joblib.dump(
        scaler,
        processed_dir /
        "scaler.pkl"
    )

    # --------------------------------------------------------
    # Windowing
    # --------------------------------------------------------

    print_header(
        "CREATING WINDOWS"
    )

    X_train, y_train = create_windows(
        X_train,
        y_train,
        window_size
    )

    X_val, y_val = create_windows(
        X_val,
        y_val,
        window_size
    )

    X_test, y_test = create_windows(
        X_test,
        y_test,
        window_size
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_val: {X_val.shape}"
    )

    print(
        f"X_test: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Save Arrays
    # --------------------------------------------------------

    print_header(
        "SAVING PROCESSED DATA"
    )

    save_targets = {

        "X_train.npy": X_train,
        "y_train.npy": y_train,

        "X_val.npy": X_val,
        "y_val.npy": y_val,

        "X_test.npy": X_test,
        "y_test.npy": y_test

    }

    for filename, array in (
        save_targets.items()
    ):

        filepath = (
            processed_dir /
            filename
        )

        np.save(
            filepath,
            array
        )

        print(
            f"{filename:<20}"
            f"{get_file_size_mb(filepath):.2f} MB"
        )

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    print_header(
        "PREPROCESSING COMPLETE"
    )

    print(
        f"X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"y_train shape: "
        f"{y_train.shape}"
    )

    print(
        f"X_val shape: "
        f"{X_val.shape}"
    )

    print(
        f"y_val shape: "
        f"{y_val.shape}"
    )

    print(
        f"X_test shape: "
        f"{X_test.shape}"
    )

    print(
        f"y_test shape: "
        f"{y_test.shape}"
    )

    print("\nArtifacts Saved:")

    print("scaler.pkl")
    print("label_encoder.pkl")
    print("X_train.npy")
    print("y_train.npy")
    print("X_val.npy")
    print("y_val.npy")
    print("X_test.npy")
    print("y_test.npy")


if __name__ == "__main__":
    main()

