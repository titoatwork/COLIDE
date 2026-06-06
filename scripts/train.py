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
from torch.optim.lr_scheduler import (
CosineAnnealingLR
)

from torch.utils.data import (
Dataset,
DataLoader
)

from torch.amp import (
autocast,
GradScaler
)

import sys

PROJECT_ROOT = (
Path(__file__)
.resolve()
.parent
.parent
)

sys.path.append(
str(PROJECT_ROOT)
)

from model.cnn_bilstm import (
CNNBiLSTM
)

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

# Paths

# ============================================================

CONFIG_PATH = (
PROJECT_ROOT /
"config" /
"config.yaml"
)

PROCESSED_DIR = (
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

BEST_MODEL_PATH = (
MODEL_DIR /
"best_model.pth"
)

# ============================================================

# Config

# ============================================================

with open(
CONFIG_PATH,
"r"
) as f:

    config = yaml.safe_load(f)

TRAIN_CFG = config["training"]

# ============================================================

# Device

# ============================================================

DEVICE = torch.device(

"cuda"

if torch.cuda.is_available()

else "cpu"

)

print(
"\n" +
"=" * 80
)

print(
"GPU INFORMATION"
)

print(
"=" * 80
)

if torch.cuda.is_available():
    print(
        f"Device: "
        f"{torch.cuda.get_device_name(0)}"
    )
    
    total_mem = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / 1024**3
    )
    
    print(
        f"VRAM: "
        f"{total_mem:.2f} GB"
    )
    
else:
    print(
        "CUDA not available."
    )

# ============================================================

# Dataset

# ============================================================

class NPYDataset(Dataset):
    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(
            self.X
        )

    def __getitem__(
        self,
        idx
    ):

        return (

            self.X[idx],

            self.y[idx]

        )

# ============================================================

# Load Data

# ============================================================

print(
"\n" +
"=" * 80
)

print(
"LOADING PROCESSED DATA"
)

print(
"=" * 80
)

X_train = np.load(
PROCESSED_DIR /
"X_train.npy"
)

y_train = np.load(
PROCESSED_DIR /
"y_train.npy"
)

X_val = np.load(
PROCESSED_DIR /
"X_val.npy"
)

y_val = np.load(
PROCESSED_DIR /
"y_val.npy"
)

X_test = np.load(
PROCESSED_DIR /
"X_test.npy"
)

y_test = np.load(
PROCESSED_DIR /
"y_test.npy"
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

batch_size = (
TRAIN_CFG["batch_size"]
)

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

try:
    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=2,

        pin_memory=True

    )

except RuntimeError:
    batch_size = 64

    print(
        "\nOOM fallback:"
        " batch_size=64"
    )

train_loader = DataLoader(

    train_dataset,

    batch_size=batch_size,

    shuffle=True,

    num_workers=2,

    pin_memory=True

)

val_loader = DataLoader(

val_dataset,

batch_size=batch_size,

shuffle=False,

num_workers=2,

pin_memory=True

)

test_loader = DataLoader(

test_dataset,

batch_size=batch_size,

shuffle=False,

num_workers=2,

pin_memory=True

)
# ============================================================

# Label Encoder

# ============================================================

label_encoder = joblib.load(

PROCESSED_DIR /
"label_encoder.pkl"

)

CLASS_NAMES = list(
label_encoder.classes_
)

# ============================================================

# Model

# ============================================================

print(
"\n" +
"=" * 80
)

print(
"INITIALIZING MODEL"
)

print(
"=" * 80
)

model = CNNBiLSTM(
config
).to(
DEVICE
)

model.get_model_summary()

# ============================================================

# Loss

# ============================================================

criterion = (
nn.CrossEntropyLoss()
)

# ============================================================

# Optimizer

# ============================================================

optimizer = Adam(

model.parameters(),

lr=TRAIN_CFG["lr"]

)

# ============================================================

# Scheduler

# ============================================================

scheduler = CosineAnnealingLR(

optimizer,

T_max=TRAIN_CFG["epochs"]

)

# ============================================================

# AMP

# ============================================================

scaler = GradScaler(
"cuda"
)

# ============================================================

# Training Step

# ============================================================

def train_one_epoch():
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
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

# ============================================================

# Validation Step

# ============================================================

@torch.no_grad()
def validate():
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
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

# History

# ============================================================

history = {

"train_loss": [],

"train_acc": [],

"train_macro_f1": [],

"val_loss": [],

"val_acc": [],

"val_macro_f1": []

}

# ============================================================

# Early Stopping

# ============================================================

best_macro_f1 = 0.0

epochs_without_improvement = 0

patience = (
TRAIN_CFG[
"early_stopping_patience"
]
)
# ============================================================

# Label Encoder

# ============================================================
label_encoder = joblib.load(

    PROCESSED_DIR / "label_encoder.pkl"

)

CLASS_NAMES = list(label_encoder.classes_)

# ============================================================

# History

# ============================================================

history = {

"train_loss": [],

"train_acc": [],

"train_macro_f1": [],

"val_loss": [],

"val_acc": [],

"val_macro_f1": []

}

# ============================================================

# Early Stopping

# ============================================================

best_macro_f1 = 0.0

epochs_without_improvement = 0

patience = (
TRAIN_CFG[
"early_stopping_patience"
]
)
# ============================================================

# Training Loop

# ============================================================

print(
"\n" +
"=" * 80
)

print(
"STARTING TRAINING"
)

print(
"=" * 80
)

for epoch in range(
    TRAIN_CFG["epochs"]
    ):
    
    start_time = time.time()
    
    train_loss, train_acc, train_macro_f1 = (
    
    train_one_epoch()
    
    )
    
    val_loss, val_acc, val_macro_f1 = (
    
    validate()
    
    )
    
    scheduler.step()
    
    history["train_loss"].append(
    float(train_loss)
    )
    
    history["train_acc"].append(
    float(train_acc)
    )
    
    history["train_macro_f1"].append(
    float(train_macro_f1)
    )
    
    history["val_loss"].append(
    float(val_loss)
    )
    
    history["val_acc"].append(
    float(val_acc)
    )
    
    history["val_macro_f1"].append(
    float(val_macro_f1)
    )
    
    epoch_time = (
    time.time() -
    start_time
    )
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
    else:
        allocated = 0.0
        reserved = 0.0
    
    print(
    f"\nEpoch "
    f"{epoch+1}/"
    f"{TRAIN_CFG['epochs']}"
    )
    
    print(
    f"Train Loss: {train_loss:.4f} | "
    f"Train Acc: {train_acc:.4f} | "
    f"Train Macro-F1: {train_macro_f1:.4f}"
    )
    
    print(
    f"Val Loss: {val_loss:.4f} | "
    f"Val Acc: {val_acc:.4f} | "
    f"Val Macro-F1: {val_macro_f1:.4f}"
    )
    
    print(
    f"GPU Allocated: "
    f"{allocated:.2f} GB | "
    f"Reserved: "
    f"{reserved:.2f} GB"
    )
    
    print(
    f"Epoch Time: "
    f"{epoch_time:.2f} sec"
    )
    
    # ====================================================
    # Checkpoint on Macro-F1
    # ====================================================

    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("✓ Best model saved")
    else:
        epochs_without_improvement += 1
        print(f"No improvement ({epochs_without_improvement}/{patience})")

    if epochs_without_improvement >= patience:
        print("\nEarly stopping triggered.")
        break
    
# ============================================================

# Load Best Model

# ============================================================

print(
"\n" +
"=" * 80
)

print(
"LOADING BEST MODEL"
)

print(
"=" * 80
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

print(
"\n" +
"=" * 80
)

print(
"RUNNING TEST EVALUATION"
)

print(
"=" * 80
)

all_preds = []

all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        with autocast(device_type="cuda"):
            outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
        
# ============================================================

# Metrics

# ============================================================

print(
"\n" +
"=" * 80
)

print(
"CLASSIFICATION REPORT"
)

print(
"=" * 80
)

report = classification_report(

all_targets,

all_preds,

labels=list(
    range(
        len(CLASS_NAMES)
    )
),

target_names=CLASS_NAMES,

digits=4,

zero_division=0

)

print(
report
)

macro_f1 = f1_score(

all_targets,

all_preds,

average="macro",

zero_division=0

)

weighted_f1 = f1_score(

all_targets,

all_preds,

average="weighted",

zero_division=0

)

print(
f"\nMacro-F1: "
f"{macro_f1:.4f}"
)

print(
f"Weighted-F1: "
f"{weighted_f1:.4f}"
)

# ============================================================

# Confusion Matrix

# ============================================================

cm = confusion_matrix(

all_targets,

all_preds,

labels=list(
    range(
        len(CLASS_NAMES)
    )
)

)

print(
"\nConfusion Matrix:"
)

print(
cm
)

# ============================================================

# Save History

# ============================================================

history_path = (

BENCHMARK_DIR /

"training_history.json"

)

with open(
    history_path,
    "w"
) as f:
    json.dump(
        history,
        f,
        indent=4
    )

print(
f"\nSaved: "
f"{history_path}"
)

# ============================================================

# Export Weights

# ============================================================

model.export_weights()

print(
"\nTraining Complete."
)
