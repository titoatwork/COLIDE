"""
COLIDE – ToN-IoT Distillation Training (SMOTE + RF Teacher)
Keeps the proven SMOTE pipeline, adds knowledge distillation.
Saves to a NEW checkpoint file – original model remains untouched.
"""

import argparse, json, random, time, sys, os
from pathlib import Path
import numpy as np, pandas as pd, yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Dataset
# ============================================================
class SimpleDataset(Dataset):
    def __init__(self, X, y, rf_probs=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.rf_probs = torch.tensor(rf_probs, dtype=torch.float32) if rf_probs is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.rf_probs is not None:
            return self.X[idx], self.y[idx], self.rf_probs[idx]
        return self.X[idx], self.y[idx], torch.tensor(0)

# ============================================================
# Data Loading (SMOTE)
# ============================================================
def load_data_toniot():
    """Load ToN-IoT, apply SMOTE with proven targets."""
    df = pd.read_csv('data/raw/toniot/train_test_network.csv')
    
    numeric = sorted(['duration','src_bytes','dst_bytes','src_pkts','dst_pkts',
                      'src_ip_bytes','dst_ip_bytes','src_port','dst_port','missed_bytes'])
    categorical = sorted(['proto','service','conn_state'])
    
    X = df[numeric].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in categorical:
        le_cat = LabelEncoder()
        X[col] = le_cat.fit_transform(df[col].astype(str))
    X = X.values.astype(np.float32)
    
    le = LabelEncoder()
    y = le.fit_transform(df['type'])
    class_names = list(le.classes_)
    num_classes = len(class_names)
    
    # 60/20/20 stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp)
    
    # Undersample majority to 10,000 per class
    max_per_class = 10000
    indices = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        if len(cls_idx) > max_per_class:
            chosen = np.random.RandomState(SEED).choice(cls_idx, max_per_class, replace=False)
            indices.extend(chosen)
        else:
            indices.extend(cls_idx)
    X_train_us = X_train[indices]
    y_train_us = y_train[indices]
    
    # SMOTE on mitm only (class index for 'mitm' in class_names)
    mitm_idx = class_names.index('mitm')
    smote_target = {mitm_idx: 5000}
    # Only apply if mitm has fewer than target
    if Counter(y_train_us)[mitm_idx] < 5000:
        smote = SMOTE(sampling_strategy=smote_target, random_state=SEED, k_neighbors=3)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_us, y_train_us)
    else:
        X_train_smote, y_train_smote = X_train_us.copy(), y_train_us.copy()
    
    scaler = MinMaxScaler()
    X_train_smote = scaler.fit_transform(X_train_smote).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    
    print(f"Dataset: ToN-IoT | Classes: {num_classes}")
    print(f"Train: {X_train_smote.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train_smote, y_train_smote, X_val, y_val, X_test, y_test, class_names, num_classes

# ============================================================
# RF Teacher
# ============================================================
def train_rf_teacher(X_train, y_train, X_val, y_val, class_names):
    print("\n" + "="*60)
    print("TRAINING RF TEACHER (200 trees)")
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_f1 = f1_score(y_val, rf.predict(X_val), average='macro')
    print(f"RF Val Macro-F1: {val_f1:.4f}")
    probs = rf.predict_proba(X_train).astype(np.float32)
    probs = np.clip(probs, 1e-7, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return rf, probs

# ============================================================
# Training Functions
# ============================================================
def train_one_epoch(model, loader, optimizer, scaler, ce_loss, distill, alpha_kd, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        if distill:
            inputs, targets, rf_probs = batch
            rf_probs = rf_probs.to(device)
        else:
            inputs, targets, _ = batch
        inputs = inputs.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(inputs)
            loss = ce_loss(logits, targets)
            if distill:
                student_log_probs = F.log_softmax(logits, dim=1)
                loss_kd = F.kl_div(student_log_probs, rf_probs, reduction='batchmean')
                loss = alpha_kd * loss_kd + (1 - alpha_kd) * loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, epoch_f1

@torch.no_grad()
def validate(model, loader, ce_loss, device):
    model.eval()
    all_preds, all_targets = [], []
    for inputs, targets, _ in loader:
        inputs = inputs.to(device); targets = targets.to(device)
        with autocast('cuda'):
            logits = model(inputs)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    return f1_score(all_targets, all_preds, average='macro', zero_division=0)

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60 + "\nCOLIDE DISTILLATION TRAINING (SMOTE) - ToN-IoT\n" + "="*60)

    X_train, y_train, X_val, y_val, X_test, y_test, class_names, num_classes = load_data_toniot()

    # Load config
    config_path = 'data/processed_toniot/config_toniot.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # RF teacher
    rf, rf_probs = train_rf_teacher(X_train, y_train, X_val, y_val, class_names)

    # DataLoaders
    train_ds = SimpleDataset(X_train, y_train, rf_probs)
    val_ds   = SimpleDataset(X_val, y_val)
    test_ds  = SimpleDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Model
    model = CNNBiLSTM(config).to(DEVICE)
    model.get_model_summary()

    # Loss
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    print("\n" + "="*60 + f"\nTRAINING (Distillation α={args.alpha:.2f})\n" + "="*60)
    best_f1 = 0.0
    patience_counter = 0
    save_path = "model/best_model_toniot_distill.pth"

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, scaler,
                                                ce_loss, True, args.alpha, DEVICE)
        val_f1 = validate(model, val_loader, ce_loss, DEVICE)
        scheduler.step()
        elapsed = time.time() - start
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val F1: {val_f1:.4f} | {elapsed:.1f}s")
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >> Best model saved (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Test evaluation
    print("\n" + "="*60 + "\nTEST SET EVALUATION\n" + "="*60)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(DEVICE)
            with autocast('cuda'):
                logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0))
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
    acc = accuracy_score(all_targets, all_preds)
    print(f"Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f} | Accuracy: {acc:.4f}")

    results = {
        'dataset': 'toniot',
        'method': 'SMOTE + Distillation',
        'alpha_kd': args.alpha,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(acc),
        'best_val_f1': float(best_f1)
    }
    with open("benchmarks/results/distill_toniot.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()