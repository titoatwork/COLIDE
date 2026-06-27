#!/home/user/ibteshamulhaque/.conda/envs/colide/bin/python
"""
Retrain CNN‑BiLSTM on ToN‑IoT after dropping columns that are >99.8% empty.
Uses the same KD + Focal Loss recipe as the winning BoT‑IoT configuration.
Saves results to benchmarks/results/toniot_clean_retrain.json.
"""

import argparse, json, random, time, sys, os
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

sys.path.insert(0, "model")
from cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Data loading with empty‑feature removal
# ---------------------------------------------------------------------------
def load_toniot_clean():
    df = pd.read_csv("data/raw/toniot/train_test_network.csv")

    # Identify columns to drop (>99.8% missing or constant '-')
    drop_cols = []
    for col in df.columns:
        if col in ['type', 'label']:   # preserve target
            continue
        null_rate = (df[col].isna() | (df[col].astype(str) == '-')).mean()
        if null_rate > 0.998:
            drop_cols.append(col)
    print(f"Dropping {len(drop_cols)} columns: {drop_cols}")
    df.drop(columns=drop_cols, inplace=True)

    # Remaining features: numeric + categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Remove target from feature lists
    for c in ['type', 'attack', 'category']:
        if c in numeric_cols: numeric_cols.remove(c)
        if c in categorical_cols: categorical_cols.remove(c)

    # Encode categoricals
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[numeric_cols + categorical_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df['type'])
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # Split 60/20/20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp)

    # SMOTE with aggressive targets (same as v2)
    target_map = {
        'backdoor': 50000, 'ddos': 50000, 'dos': 50000,
        'injection': 20000, 'mitm': 10000, 'normal': 80000,
        'password': 20000, 'ransomware': 50000, 'scanning': 50000,
        'xss': 20000
    }
    smote_strat = {}
    for name, target in target_map.items():
        if name in class_names:
            idx = class_names.index(name)
            cnt = Counter(y_train)[idx]
            if cnt < target:
                smote_strat[idx] = target
    if smote_strat:
        smote = SMOTE(sampling_strategy=smote_strat, random_state=SEED, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names, num_classes, X_train.shape[1]

# ---------------------------------------------------------------------------
# Training functions (same as distill)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, ce_loss, alpha_kd, device):
    model.train()
    running_loss = 0.0; all_preds, all_targets = [], []
    for batch in loader:
        inputs, targets, rf_probs = batch
        inputs, targets, rf_probs = inputs.to(device), targets.to(device), rf_probs.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(inputs)
            loss = ce_loss(logits, targets)
            student_log_probs = F.log_softmax(logits, dim=1)
            loss_kd = F.kl_div(student_log_probs, rf_probs, reduction='batchmean')
            loss = 0.7 * loss_kd + 0.3 * loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    return running_loss/len(loader.dataset), f1_score(all_targets, all_preds, average='macro', zero_division=0)

@torch.no_grad()
def validate(model, loader, device):
    model.eval(); all_preds, all_targets = [], []
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast('cuda'):
            logits = model(inputs)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    return f1_score(all_targets, all_preds, average='macro', zero_division=0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, num_classes, input_dim = load_toniot_clean()

    # Train RF teacher
    print("Training RF teacher (200 trees)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_train).astype(np.float32)
    rf_probs = np.clip(rf_probs, 1e-7, 1.0)
    rf_probs /= rf_probs.sum(axis=1, keepdims=True)
    # Apply temperature scaling T=5.0
    T = 5.0
    rf_probs = np.log(rf_probs) / T
    rf_probs = np.exp(rf_probs)
    rf_probs /= rf_probs.sum(axis=1, keepdims=True)

    # DataLoaders
    train_ds = SimpleDataset(X_train, y_train, rf_probs)
    val_ds = SimpleDataset(X_val, y_val)
    test_ds = SimpleDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    config_path = "data/processed_toniot/config_toniot.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['model']['input_features'] = input_dim
    config['model']['num_classes'] = num_classes
    model = CNNBiLSTM(config).to(device)
    model.get_model_summary()

    ce_loss = FocalLoss(gamma=2.0)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_f1, patience_cnt = 0.0, 0
    save_path = "model/best_model_toniot_clean.pth"

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, scaler, ce_loss, 0.7, device)
        val_f1 = validate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | {time.time()-start:.1f}s")
        if val_f1 > best_f1:
            best_f1 = val_f1; patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >> Best saved (Val F1: {val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("\nTEST SET EVALUATION")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    for inputs, targets, _ in test_loader:
        inputs = inputs.to(device)
        with autocast('cuda'):
            logits = model(inputs)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0))
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Test Macro‑F1: {macro_f1:.4f}")
    results = {"dataset": "toniot_clean", "macro_f1": macro_f1}
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/toniot_clean_retrain.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()