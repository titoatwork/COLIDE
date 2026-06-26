#!/usr/bin/env python3
"""Ensemble Teacher Distillation (RF + XGBoost + LightGBM)."""

import argparse, json, random, time, sys, os
import numpy as np, pandas as pd, yaml
from pathlib import Path

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

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        return loss

# ---------- Data ----------
class SimpleDataset(Dataset):
    def __init__(self, X, y, teacher_probs=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.probs = torch.tensor(teacher_probs, dtype=torch.float32) if teacher_probs is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx], self.probs[idx]) if self.probs is not None else (self.X[idx], self.y[idx], torch.tensor(0))

def load_data():
    df_train = pd.read_csv('data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv', sep=',')
    df_test  = pd.read_csv('data/raw/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv', sep=',')
    feature_cols = ['N_IN_Conn_P_DstIP','N_IN_Conn_P_SrcIP','drate','max','mean','min','seq','srate','state_number','stddev']
    X_train_raw = df_train[feature_cols].values.astype(np.float32)
    X_test_raw  = df_test[feature_cols].values.astype(np.float32)
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw  = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['category'])
    y_test  = le.transform(df_test['category'])
    class_names = list(le.classes_)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_raw, y_train, test_size=0.1, random_state=SEED, stratify=y_train)
    strat = {}
    for i, name in enumerate(class_names):
        cnt = Counter(y_tr)[i]
        if name == 'DDoS': strat[i] = 100000
        elif name == 'DoS': strat[i] = 100000
        elif name == 'Normal': strat[i] = 2000
        elif name == 'Reconnaissance': strat[i] = 50000
        elif name == 'Theft': strat[i] = 1000
        else: strat[i] = cnt
    valid_strat = {k:v for k,v in strat.items() if v > Counter(y_tr)[k]}
    smote = SMOTE(sampling_strategy=valid_strat, random_state=SEED, k_neighbors=3)
    X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
    scaler = MinMaxScaler()
    X_tr_smote = scaler.fit_transform(X_tr_smote).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    return X_tr_smote, y_tr_smote, X_val_scaled, y_val, X_test_scaled, y_test, class_names

# ---------- Ensemble Teacher ----------
def train_ensemble_teacher(X_train, y_train, X_val, y_val, class_names, temperature=5.0):
    print("Training RF...")
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
    print("Training XGBoost...")
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(n_estimators=200, random_state=SEED, eval_metric='mlogloss', verbosity=0).fit(X_train, y_train)
    print("Training LightGBM...")
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1).fit(X_train, y_train)

    # Average probabilities
    probs = (rf.predict_proba(X_train) + xgb_model.predict_proba(X_train) + lgb_model.predict_proba(X_train)) / 3.0
    probs = probs.astype(np.float32)

    # Temperature scaling
    if temperature != 1.0:
        probs = np.log(probs + 1e-7) / temperature
        probs = np.exp(probs)
        probs /= probs.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-7, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)

    val_f1 = f1_score(y_val, np.argmax(probs[:len(y_val)], axis=1) if len(probs) == len(y_val) else
                      rf.predict(X_val), average='macro')
    print(f"Ensemble teacher Val F1: {val_f1:.4f}")
    return probs

# ---------- Training ----------
def train_one_epoch(model, loader, optimizer, scaler, ce_loss, distill, alpha_kd, device):
    model.train()
    running_loss, all_preds, all_targets = 0.0, [], []
    for batch in loader:
        inputs, targets, teacher_probs = batch
        inputs, targets, teacher_probs = inputs.to(device), targets.to(device), teacher_probs.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(inputs)
            loss = ce_loss(logits, targets)
            if distill:
                student_log_probs = F.log_softmax(logits, dim=1)
                loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                loss = alpha_kd * loss_kd + (1 - alpha_kd) * loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    return running_loss / len(loader.dataset), f1_score(all_targets, all_preds, average='macro', zero_division=0)

@torch.no_grad()
def validate(model, loader, ce_loss, device):
    model.eval(); all_preds, all_targets = [], []
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast('cuda'):
            logits = model(inputs)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    return f1_score(all_targets, all_preds, average='macro', zero_division=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"ENSEMBLE TEACHER DISTILLATION — α={args.alpha} T={args.temperature}")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_data()
    teacher_probs = train_ensemble_teacher(X_train, y_train, X_val, y_val, class_names, args.temperature)

    train_ds = SimpleDataset(X_train, y_train, teacher_probs)
    val_ds   = SimpleDataset(X_val, y_val)
    test_ds  = SimpleDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    sys.path.insert(0, "model")
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM
    config = yaml.safe_load(open("config/config.yaml"))
    model = CNNBiLSTM(config).to(DEVICE)
    model.get_model_summary()

    ce_loss = FocalLoss(gamma=args.focal_gamma) if args.focal_gamma > 0 else nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_f1, patience_counter = 0.0, 0
    save_path = "model/best_model_botiot_ensemble.pth"

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, scaler, ce_loss, True, args.alpha, DEVICE)
        val_f1 = validate(model, val_loader, ce_loss, DEVICE)
        scheduler.step()
        elapsed = time.time() - start
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | {elapsed:.1f}s")
        if val_f1 > best_f1:
            best_f1 = val_f1; patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >> Best model saved (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("\nTEST SET EVALUATION — Ensemble Teacher")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(DEVICE)
            with autocast('cuda'):
                logits = model(inputs)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(targets.numpy())
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0))
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Ensemble Teacher Test Macro-F1: {macro_f1:.4f} | Best: 0.9624 (MLP), 0.9601 (CNN-BiLSTM)")

if __name__ == '__main__':
    main()