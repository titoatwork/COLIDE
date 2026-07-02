#!/usr/bin/env python3
"""Two-Stage Fine-Tuning: Distilled checkpoint → Focal fine-tune on real data."""

import argparse, json, random, time, sys
import numpy as np, pandas as pd, yaml, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import classification_report, f1_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class RealDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_real_data():
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
    # NO SMOTE — pure real data
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    print(f"Real data only: Train {X_tr.shape[0]:,} | Val {X_val.shape[0]:,} | Test {X_test.shape[0]:,}")
    return X_tr, y_tr, X_val, y_val, X_test, y_test, class_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='model/best_model_botiot_distill_focal_T5.pth')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_real_data()

    train_loader = DataLoader(RealDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(RealDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(RealDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    sys.path.insert(0, "model")
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM
    config = yaml.safe_load(open("config/config.yaml"))
    model = CNNBiLSTM(config).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.get_model_summary()

    ce_loss = FocalLoss(gamma=args.focal_gamma)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda')

    best_f1, patience_counter = 0.0, 0
    save_path = "model/best_model_botiot_twostage.pth"

    for epoch in range(args.epochs):
        model.train()
        running_loss, all_preds, all_targets = 0.0, [], []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(inputs)
                loss = ce_loss(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        train_loss = running_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        model.eval()
        val_preds, val_targets = [], []
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            with autocast('cuda'):
                logits = model(inputs)
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1; patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >> Best model saved (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\nTEST SET EVALUATION — Two‑Stage Fine‑Tuning")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            with autocast('cuda'):
                logits = model(inputs)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0))
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
    acc = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()
    print(f"Two‑Stage Test Macro-F1: {macro_f1:.4f} | Best: 0.9624 (MLP), 0.9790 (CNN-BiLSTM)")

    # Previously this final headline number was never saved to a JSON --
    # verify_claims.py had it as a hand-typed literal, same class of
    # provenance gap as every other fix this session. Fixed 2026-07-01.
    results = {
        'checkpoint': args.checkpoint,
        'focal_gamma': args.focal_gamma,
        'epochs_requested': args.epochs,
        'lr': args.lr,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(acc),
        'best_val_f1': float(best_f1),
    }
    results_path = 'benchmarks/results/twostage_botiot.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()