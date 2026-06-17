"""
COLIDE - ToN-IoT Training Script
Same CNN-BiLSTM architecture, different dataset (10 classes).
"""

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Paths (ToN-IoT specific)
# ============================================================
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_toniot"
CONFIG_PATH = PROCESSED_DIR / "config_toniot.yaml"
MODEL_DIR = PROJECT_ROOT / "model"
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "results"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_model_toniot.pth"

# ============================================================
# Config
# ============================================================
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TRAIN_CFG = config["training"]

# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "=" * 80)
print("COLIDE ToN-IoT TRAINING")
print("=" * 80)

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {total_mem:.2f} GB")

# ============================================================
# Dataset
# ============================================================
class NPYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# Load Data
# ============================================================
print("\n" + "=" * 80)
print("LOADING ToN-IoT PROCESSED DATA")
print("=" * 80)

X_train = np.load(PROCESSED_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy")
X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

print(f"X_train: {X_train.shape}")
print(f"X_val  : {X_val.shape}")
print(f"X_test : {X_test.shape}")

# ============================================================
# DataLoaders
# ============================================================
batch_size = TRAIN_CFG["batch_size"]

train_loader = DataLoader(
    NPYDataset(X_train, y_train),
    batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True
)

val_loader = DataLoader(
    NPYDataset(X_val, y_val),
    batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True
)

test_loader = DataLoader(
    NPYDataset(X_test, y_test),
    batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True
)

# ============================================================
# Label Encoder
# ============================================================
label_encoder = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
CLASS_NAMES = list(label_encoder.classes_)
print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

# ============================================================
# Model
# ============================================================
print("\n" + "=" * 80)
print("INITIALIZING MODEL")
print("=" * 80)

model = CNNBiLSTM(config).to(DEVICE)
model.get_model_summary()

# ============================================================
# Training setup
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=TRAIN_CFG["lr"])
scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CFG["epochs"])
scaler = GradScaler("cuda")

# ============================================================
# Training and Validation functions
# ============================================================
def train_one_epoch():
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_macro_f1

@torch.no_grad()
def validate():
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    for inputs, targets in val_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, val_macro_f1

# ============================================================
# Training Loop
# ============================================================
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

history = {
    "train_loss": [], "train_acc": [], "train_macro_f1": [],
    "val_loss": [], "val_acc": [], "val_macro_f1": []
}

best_macro_f1 = 0.0
epochs_without_improvement = 0
patience = TRAIN_CFG["early_stopping_patience"]

for epoch in range(TRAIN_CFG["epochs"]):
    start_time = time.time()

    train_loss, train_acc, train_macro_f1 = train_one_epoch()
    val_loss, val_acc, val_macro_f1 = validate()
    scheduler.step()

    history["train_loss"].append(float(train_loss))
    history["train_acc"].append(float(train_acc))
    history["train_macro_f1"].append(float(train_macro_f1))
    history["val_loss"].append(float(val_loss))
    history["val_acc"].append(float(val_acc))
    history["val_macro_f1"].append(float(val_macro_f1))

    epoch_time = time.time() - start_time

    print(f"\nEpoch {epoch+1}/{TRAIN_CFG['epochs']}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro-F1: {train_macro_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
    print(f"Epoch Time: {epoch_time:.2f} sec")

    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(">> Best model saved")
    else:
        epochs_without_improvement += 1
        print(f"No improvement ({epochs_without_improvement}/{patience})")

    if epochs_without_improvement >= patience:
        print("\nEarly stopping triggered.")
        break

# ============================================================
# Load Best Model and Evaluate on Test Set
# ============================================================
print("\n" + "=" * 80)
print("LOADING BEST MODEL AND EVALUATING")
print("=" * 80)

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        with autocast(device_type="cuda"):
            outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT (ToN-IoT)")
print("=" * 80)

report = classification_report(
    all_targets, all_preds,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    digits=4, zero_division=0
)
print(report)

macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")

cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASS_NAMES))))
print(f"\nConfusion Matrix:\n{cm}")

# Save history
history_path = BENCHMARK_DIR / "training_history_toniot.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=4)
print(f"\nSaved: {history_path}")
print("\nToN-IoT Training Complete.")