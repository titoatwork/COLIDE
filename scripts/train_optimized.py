"""
COLIDE - Optimized Training with Focal Loss + Knowledge Distillation
Replaces SMOTE with WeightedRandomSampler.
Distills from RF teacher into CNN-BiLSTM student.

Usage:
  PYTHONPATH=. python scripts/train_optimized.py --dataset botiot
  PYTHONPATH=. python scripts/train_optimized.py --dataset toniot
  PYTHONPATH=. python scripts/train_optimized.py --dataset botiot --no-distill  (focal loss only)
"""

import argparse
import json
import random
import time
import sys
from pathlib import Path

import numpy as np
import yaml
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

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
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy/majority samples, focuses on hard/minority examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights tensor
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# Dataset
# ============================================================
class NPYDataset(Dataset):
    def __init__(self, X, y, rf_probs=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.rf_probs = torch.tensor(rf_probs, dtype=torch.float32) if rf_probs is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.rf_probs is not None:
            return self.X[idx], self.y[idx], self.rf_probs[idx]
        return self.X[idx], self.y[idx], torch.tensor(0)  # dummy


# ============================================================
# Data Loading (NO SMOTE)
# ============================================================
def load_data_no_smote(dataset_name):
    """Load and preprocess data WITHOUT SMOTE, using only undersampling."""

    if dataset_name == 'botiot':
        import pandas as pd
        # Load raw BoT-IoT (COMMA-separated, NOT semicolon)
        df_train = pd.read_csv('data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv', sep=',')
        df_test  = pd.read_csv('data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv',  sep=',')

        feature_cols = ['N_IN_Conn_P_DstIP', 'N_IN_Conn_P_SrcIP', 'drate', 'max',
                       'mean', 'min', 'seq', 'srate', 'state_number', 'stddev']
        label_col = 'category'

        # *** FIX: Process train and test independently – NO concat-leakage ***
        X_train_raw = df_train[feature_cols].values.astype(np.float32)
        X_test_raw  = df_test[feature_cols].values.astype(np.float32)
        X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_raw  = np.nan_to_num(X_test_raw,  nan=0.0, posinf=0.0, neginf=0.0)

        # LabelEncoder fitted on training labels only
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[label_col])
        y_test  = le.transform(df_test[label_col])          # transform only, no fit
        class_names = list(le.classes_)

        # Train/val split (from training set only)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_raw, y_train, test_size=0.2, random_state=SEED, stratify=y_train)

        # Undersample majority only (NO SMOTE)
        max_per_class = 50000
        indices = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            if len(cls_idx) > max_per_class:
                chosen = np.random.RandomState(SEED).choice(cls_idx, max_per_class, replace=False)
                indices.extend(chosen)
            else:
                indices.extend(cls_idx)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Scale – fit on training, transform val and test
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val   = scaler.transform(X_val).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)

        config_path = 'config/config.yaml'
        num_classes = len(class_names)

    elif dataset_name == 'toniot':
        import pandas as pd
        df = pd.read_csv('data/raw/toniot/train_test_network.csv')

        # *** FIX: sort columns for deterministic feature order ***
        numeric = sorted(['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts',
                          'src_ip_bytes', 'dst_ip_bytes', 'src_port', 'dst_port', 'missed_bytes'])
        categorical = sorted(['proto', 'service', 'conn_state'])

        X = df[numeric].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        for col in categorical:
            cat_le = LabelEncoder()
            X[col] = cat_le.fit_transform(df[col].astype(str))
        X = X.values.astype(np.float32)

        le = LabelEncoder()
        y = le.fit_transform(df['type'])
        class_names = list(le.classes_)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp)

        # Undersample majority only (NO SMOTE)
        max_per_class = 10000
        indices = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            if len(cls_idx) > max_per_class:
                chosen = np.random.RandomState(SEED).choice(cls_idx, max_per_class, replace=False)
                indices.extend(chosen)
            else:
                indices.extend(cls_idx)
        X_train = X_train[indices]
        y_train = y_train[indices]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        config_path = 'data/processed_toniot/config_toniot.yaml'
        num_classes = len(class_names)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nDataset: {dataset_name}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"Train class distribution:")
    for i, name in enumerate(class_names):
        count = (y_train == i).sum()
        print(f"  {name:<20} {count:>6,}")

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names, config_path, num_classes


# ============================================================
# RF Teacher
# ============================================================
def train_rf_teacher(X_train, y_train, X_val, y_val, class_names):
    """Train RF and generate soft probability labels."""
    print(f"\n{'='*60}")
    print("TRAINING RF TEACHER")
    print(f"{'='*60}")

    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    val_preds = rf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    print(f"RF Validation Macro-F1: {val_f1:.4f}")

    # Generate soft labels for training data, with clipping for numerical stability
    rf_train_probs = rf.predict_proba(X_train).astype(np.float32)
    rf_train_probs = np.clip(rf_train_probs, 1e-7, 1.0)
    rf_train_probs /= rf_train_probs.sum(axis=1, keepdims=True)   # re-normalize
    print(f"RF soft labels shape: {rf_train_probs.shape}")

    return rf, rf_train_probs


# ============================================================
# Training Functions
# ============================================================
def train_one_epoch(model, loader, optimizer, scaler, focal_loss, distill, alpha_kd, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        if distill:
            inputs, targets, rf_probs = batch
            rf_probs = rf_probs.to(device, non_blocking=True)
        else:
            inputs, targets, _ = batch

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            logits = model(inputs)

            # Focal loss on hard labels
            loss_ce = focal_loss(logits, targets)

            if distill:
                # KL divergence on soft labels from RF
                student_log_probs = F.log_softmax(logits, dim=1)
                loss_kd = F.kl_div(student_log_probs, rf_probs, reduction='batchmean')
                loss = alpha_kd * loss_kd + (1 - alpha_kd) * loss_ce
            else:
                loss = loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, epoch_f1


@torch.no_grad()
def validate(model, loader, focal_loss, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for inputs, targets, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            logits = model(inputs)
            loss = focal_loss(logits, targets)

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return val_loss, val_f1


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['botiot', 'toniot'], required=True)
    parser.add_argument('--no-distill', action='store_true', help='Disable knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='KD weight (0=pure CE, 1=pure KD)')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distill = not args.no_distill

    print("=" * 60)
    print(f"COLIDE OPTIMIZED TRAINING")
    print(f"Dataset: {args.dataset}")
    print(f"Focal Loss: gamma={args.gamma}")
    print(f"Distillation: {'ON (alpha={})'.format(args.alpha) if distill else 'OFF'}")
    print("=" * 60)

    # Load data (NO SMOTE)
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, config_path, num_classes = \
        load_data_no_smote(args.dataset)

    # Load model config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # RF teacher
    rf_train_probs = None
    if distill:
        rf, rf_train_probs = train_rf_teacher(X_train, y_train, X_val, y_val, class_names)

    # Weighted sampler (replaces SMOTE)
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(y_train),
        replacement=True
    )

    # DataLoaders
    train_dataset = NPYDataset(X_train, y_train, rf_train_probs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(NPYDataset(X_val, y_val), batch_size=args.batch_size,
                           shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(NPYDataset(X_test, y_test), batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = CNNBiLSTM(config).to(DEVICE)
    model.get_model_summary()

    # Focal loss with class weights
    alpha_weights = torch.tensor(class_weights / class_weights.sum() * num_classes,
                                dtype=torch.float32).to(DEVICE)
    focal_loss = FocalLoss(alpha=alpha_weights, gamma=args.gamma)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda")

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    best_f1 = 0.0
    patience_counter = 0
    model_save_path = f"model/best_model_{args.dataset}_optimized.pth"

    for epoch in range(args.epochs):
        start = time.time()

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scaler, focal_loss,
            distill, args.alpha, DEVICE)
        val_loss, val_f1 = validate(model, val_loader, focal_loss, DEVICE)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
              f"{elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  >> Best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda"):
                logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    print(classification_report(all_targets, all_preds,
          target_names=class_names, digits=4, zero_division=0))

    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
    acc = accuracy_score(all_targets, all_preds)

    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Accuracy:    {acc:.4f}")

    # Save results
    results = {
        'dataset': args.dataset,
        'distillation': distill,
        'alpha_kd': args.alpha if distill else None,
        'focal_gamma': args.gamma,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(acc),
        'best_val_f1': float(best_f1),
        'smote': False,
        'sampling': 'WeightedRandomSampler',
    }

    suffix = 'distill' if distill else 'focal_only'
    results_path = f"benchmarks/results/optimized_{args.dataset}_{suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == '__main__':
    main()