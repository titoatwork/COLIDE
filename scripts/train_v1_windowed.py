"""
COLIDE
Training Pipeline

CNN-BiLSTM Training Script

Features:
- Weighted CrossEntropy
- Mixed Precision (AMP)
- Macro-F1 checkpointing
- Early stopping
- CosineAnnealingLR
- Confusion matrix export
- Classification report
"""

# ============================================================
# Imports
# ============================================================

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)




# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# Utility Functions
# ============================================================

def print_header(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_config(config_path):

    with open(config_path, "r") as f:

        config = yaml.safe_load(f)

    return config


def print_gpu_info():

    print_header("GPU INFORMATION")

    if not torch.cuda.is_available():

        print("CUDA NOT AVAILABLE")
        return

    device_name = (
        torch.cuda.get_device_name(0)
    )

    total_memory_gb = (

        torch.cuda
        .get_device_properties(0)
        .total_memory

        / (1024 ** 3)

    )

    print(
        f"Device: {device_name}"
    )

    print(
        f"VRAM  : {total_memory_gb:.2f} GB"
    )


def get_vram_usage():

    if not torch.cuda.is_available():

        return 0.0

    allocated = (

        torch.cuda.memory_allocated()

        / (1024 ** 3)

    )

    return allocated


# ============================================================
# Dataset
# ============================================================

class NPYDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.from_numpy(
            X
        ).float()

        self.y = torch.from_numpy(
            y
        ).long()

    def __len__(self):

        return len(self.X)

    def __getitem__(
        self,
        idx
    ):

        return (
            self.X[idx],
            self.y[idx]
        )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    return {

        "accuracy":
            accuracy,

        "precision_macro":
            precision_macro,

        "recall_macro":
            recall_macro,

        "f1_macro":
            f1_macro,

        "precision_weighted":
            precision_weighted,

        "recall_weighted":
            recall_weighted,

        "f1_weighted":
            f1_weighted

    }


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)
import sys

sys.path.append(
    str(PROJECT_ROOT)
)

from model.cnn_bilstm import CNNBiLSTM
CONFIG_PATH = (
    PROJECT_ROOT /
    "config" /
    "config.yaml"
)

DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "model"
)

BENCHMARK_DIR = (
    PROJECT_ROOT /
    "benchmarks" /
    "results"
)

BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load Config
# ============================================================

config = load_config(
    CONFIG_PATH
)

print_gpu_info()


# ============================================================
# Load Data
# ============================================================

print_header(
    "LOADING PROCESSED DATA"
)

X_train = np.load(
    DATA_DIR / "X_train.npy"
)

y_train = np.load(
    DATA_DIR / "y_train.npy"
)

X_val = np.load(
    DATA_DIR / "X_val.npy"
)

y_val = np.load(
    DATA_DIR / "y_val.npy"
)

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)

class_weights = np.load(
    DATA_DIR /
    "class_weights.npy"
)

label_encoder = joblib.load(
    DATA_DIR /
    "label_encoder.pkl"
)

print(
    f"X_train: {X_train.shape}"
)

print(
    f"X_val  : {X_val.shape}"
)

print(
    f"X_test : {X_test.shape}"
)


# ============================================================
# DataLoaders
# ============================================================

BATCH_SIZE = (
    config["training"]["batch_size"]
)

NUM_WORKERS = 2

PIN_MEMORY = True

train_dataset = NPYDataset(
    X_train,
    y_train
)

val_dataset = NPYDataset(
    X_val,
    y_val
)

test_dataset = NPYDataset(
    X_test,
    y_test
)


# ============================================================
# History Tracking
# ============================================================

history = {

    "train_loss": [],
    "val_loss": [],

    "train_acc": [],
    "val_acc": [],

    "val_macro_f1": [],
    "val_weighted_f1": []

}

print_header(
    "SETUP COMPLETE"
)
# ============================================================
# Device
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print(
    f"\nUsing Device: {DEVICE}"
)


# ============================================================
# DataLoader Creation
# ============================================================

def create_dataloaders(
    batch_size
):

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    return (

        train_loader,

        val_loader,

        test_loader

    )


# ============================================================
# OOM Safe Batch Size
# ============================================================

try:

    train_loader, val_loader, test_loader = (

        create_dataloaders(
            BATCH_SIZE
        )

    )

except RuntimeError:

    print(
        "\nOOM DETECTED"
    )

    print(
        "Reducing batch size "
        "128 -> 64"
    )

    BATCH_SIZE = 64

    train_loader, val_loader, test_loader = (

        create_dataloaders(
            BATCH_SIZE
        )

    )


# ============================================================
# Model
# ============================================================

print_header(
    "INITIALIZING MODEL"
)

model = CNNBiLSTM(
    config
)

model = model.to(
    DEVICE
)

model.get_model_summary()

print(
    f"\nTrainable Parameters: "
    f"{model.count_parameters():,}"
)


# ============================================================
# Loss Function
# ============================================================

class_weights_tensor = torch.tensor(

    class_weights,

    dtype=torch.float32,

    device=DEVICE

)

criterion = nn.CrossEntropyLoss(

    weight=class_weights_tensor

)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=config["training"]["lr"]

)


# ============================================================
# Scheduler
# ============================================================

scheduler = (

    torch.optim.lr_scheduler
    .CosineAnnealingLR(

        optimizer,

        T_max=config["training"]["epochs"]

    )

)


# ============================================================
# AMP
# ============================================================

USE_AMP = (

    DEVICE.type == "cuda"

)

scaler = GradScaler(
    enabled=USE_AMP
)


# ============================================================
# Training Epoch
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_preds = []

    all_labels = []

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        with autocast(
            enabled=USE_AMP
        ):

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (

            loss.item()

            * X_batch.size(0)

        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        all_preds.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            y_batch.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (

        running_loss

        / len(train_loader.dataset)

    )

    metrics = compute_metrics(

        all_labels,

        all_preds

    )

    return (

        epoch_loss,

        metrics

    )


# ============================================================
# Validation Epoch
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    all_preds = []

    all_labels = []

    for X_batch, y_batch in val_loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True
        )

        with autocast(
            enabled=USE_AMP
        ):

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

        running_loss += (

            loss.item()

            * X_batch.size(0)

        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            y_batch.cpu().numpy()
        )

    epoch_loss = (

        running_loss

        / len(val_loader.dataset)

    )

    metrics = compute_metrics(

        all_labels,

        all_preds

    )

    return (

        epoch_loss,

        metrics

    )


# ============================================================
# Training Loop
# ============================================================

print_header(
    "STARTING TRAINING"
)

BEST_MACRO_F1 = 0.0

PATIENCE = (
    config["training"]
    ["early_stopping_patience"]
)

patience_counter = 0

BEST_MODEL_PATH = (
    MODEL_DIR /
    "best_model.pth"
)

EPOCHS = (
    config["training"]["epochs"]
)

for epoch in range(EPOCHS):

    epoch_start = time.time()

    train_loss, train_metrics = (
        train_one_epoch()
    )

    val_loss, val_metrics = (
        validate()
    )

    scheduler.step()

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["train_acc"].append(
        train_metrics["accuracy"]
    )

    history["val_acc"].append(
        val_metrics["accuracy"]
    )

    history["val_macro_f1"].append(
        val_metrics["f1_macro"]
    )

    history["val_weighted_f1"].append(
        val_metrics["f1_weighted"]
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    print(

        f"\nEpoch "
        f"{epoch+1}/{EPOCHS}"

    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Train Acc: "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Acc: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Macro-F1: "
        f"{val_metrics['f1_macro']:.4f}"
    )

    print(
        f"VRAM Used: "
        f"{get_vram_usage():.2f} GB"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time:.2f}s"
    )

    # --------------------------------------------
    # Checkpoint
    # --------------------------------------------

    if (
        val_metrics["f1_macro"]

        > BEST_MACRO_F1
    ):

        BEST_MACRO_F1 = (
            val_metrics["f1_macro"]
        )

        patience_counter = 0

        torch.save(

            model.state_dict(),

            BEST_MODEL_PATH

        )

        print(
            "New Best Model Saved"
        )

    else:

        patience_counter += 1

    # --------------------------------------------
    # Early Stopping
    # --------------------------------------------

    if (
        patience_counter
        >= PATIENCE
    ):

        print(
            "\nEarly stopping "
            "triggered."
        )

        break
# ============================================================

# Load Best Model

# ============================================================

print_header(
"LOADING BEST MODEL"
)

model.load_state_dict(

torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)

)

model.eval()

# ============================================================

# Test Evaluation

# ============================================================

@torch.no_grad()
def evaluate_test():

    all_preds = []

    all_labels = []

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True
        )

    with autocast(
        enabled=USE_AMP
    ):

        logits = model(
            X_batch
        )

    preds = torch.argmax(
        logits,
        dim=1
    )

    all_preds.extend(
        preds.cpu().numpy()
    )

    all_labels.extend(
        y_batch.cpu().numpy()
    )

    return (

        np.array(all_labels),

        np.array(all_preds)

    )

print_header(
"RUNNING TEST EVALUATION"
)

y_true, y_pred = (
evaluate_test()
)

test_metrics = compute_metrics(

y_true,

y_pred

)

# ============================================================

# Classification Report

# ============================================================

print_header(
"CLASSIFICATION REPORT"
)

class_names = (
label_encoder.classes_
)

report = classification_report(
    all_labels,
    all_preds,
    target_names=label_encoder.classes_,
    labels=list(range(len(label_encoder.classes_))),
    zero_division=0
)

print(report)

print(
f"\nAccuracy     : "
f"{test_metrics['accuracy']:.4f}"
)

print(
f"Macro F1     : "
f"{test_metrics['f1_macro']:.4f}"
)

print(
f"Weighted F1  : "
f"{test_metrics['f1_weighted']:.4f}"
)

# ============================================================

# Confusion Matrix

# ============================================================

print_header(
"CONFUSION MATRIX"
)

cm = confusion_matrix(

y_true,

y_pred

)

print(cm)

np.save(

BENCHMARK_DIR /
"confusion_matrix.npy",

cm

)

# ============================================================

# Save Classification Report

# ============================================================

with open(

    BENCHMARK_DIR /
    "classification_report.txt",

    "w"

) as f:

    f.write(report)

print(
    "\nSaved classification report."
)

# ============================================================

# Save Metrics JSON

# ============================================================

metrics_json = {

"accuracy":
    float(
        test_metrics["accuracy"]
    ),

"macro_f1":
    float(
        test_metrics["f1_macro"]
    ),

"weighted_f1":
    float(
        test_metrics["f1_weighted"]
    ),

"macro_precision":
    float(
        test_metrics["precision_macro"]
    ),

"macro_recall":
    float(
        test_metrics["recall_macro"]
    )

}

with open(

    BENCHMARK_DIR /
    "test_metrics.json",

    "w"

) as f:

    json.dump(

        metrics_json,

        f,

        indent=4

    )

print(
    "Saved test metrics."
)

# ============================================================

# Save Training History

# ============================================================

with open(

    BENCHMARK_DIR /
    "training_history.json",

    "w"

) as f:

    json.dump(

        history,

        f,

    indent=4

)

print(
"Saved training history."
)

# ============================================================

# Export Weights

# ============================================================

print_header(
"EXPORTING WEIGHTS"
)

export_dir = (

PROJECT_ROOT /

config["model"]
["weight_export_path"]

)

model.export_weights(
export_dir
)

# ============================================================

# Final Summary

# ============================================================

print_header(
"TRAINING COMPLETE"
)

print(
f"Best Validation Macro-F1: "
f"{BEST_MACRO_F1:.4f}"
)

print(
f"Test Accuracy          : "
f"{test_metrics['accuracy']:.4f}"
)

print(
f"Test Macro-F1          : "
f"{test_metrics['f1_macro']:.4f}"
)

print(
f"Test Weighted-F1       : "
f"{test_metrics['f1_weighted']:.4f}"
)

print(
"\nArtifacts Generated:"
)

print(
"model/best_model.pth"
)

print(
"benchmarks/results/training_history.json"
)

print(
"benchmarks/results/test_metrics.json"
)

print(
"benchmarks/results/classification_report.txt"
)

print(
"benchmarks/results/confusion_matrix.npy"
)

print(
"model/weights/fp32/"
)

print(
"model/weights/fp16/"
)

print(
"\nCOLIDE Training Pipeline Finished Successfully."
)
