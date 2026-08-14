#!/usr/bin/env python3
"""
ToN-IoT leakage-safe corrected minimal experiment (checklist §4).

Protocol: toniot_leakage_safe_v1
  - Explicit 13-feature allowlist (NUMERIC + CATEGORICAL from preprocess_toniot)
  - Blacklist target-derived columns: label, type, attack, category (+ variants)
  - Fatal asserts: label not in X; target (type) not in X
  - Stratified 60/20/20 split BEFORE fitting encoders/scalers (seed 42)
  - Encoders/scaler fit on train only; unknown cats → dedicated code
  - NO SMOTE
  - RF class_weight=balanced + hard-label CNN-BiLSTM (class-weighted CE, no KD)
  - Select by val macro-F1; evaluate test once
  - Results: benchmarks/results/toniot_corrected/

Does NOT touch BoT champion weights.

Usage:
  PYTHONPATH=. python scripts/protocol/toniot_leakage_safe.py
  PYTHONPATH=. python scripts/run_toniot_corrected_simple.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import git_sha  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
PROTOCOL_ID = "toniot_leakage_safe_v1"
SEED = 42
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "toniot" / "train_test_network.csv"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "toniot_corrected"
CKPT_DIR = PROJECT_ROOT / "model" / "toniot_corrected"
LABEL_COL = "type"

# Explicit feature allowlist (same 13 as preprocess_toniot.py NUMERIC+CATEGORICAL)
NUMERIC_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "src_ip_bytes",
    "dst_ip_bytes",
    "src_port",
    "dst_port",
    "missed_bytes",
]
CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "conn_state",
]
FEATURE_ALLOWLIST = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Target-derived blacklist (normalized forms checked after normalize_col)
TARGET_BLACKLIST_CANONICAL = frozenset(
    {
        "label",
        "type",
        "attack",
        "category",
        "attack_cat",
        "attackcat",
        "class",
        "target",
        "y",
    }
)

UNKNOWN_CAT_TOKEN = "__UNK__"


def normalize_col(name: str) -> str:
    """Trim, lowercase, collapse punctuation/underscores for blacklist checks."""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def feature_list_sha256(features: list[str]) -> str:
    payload = "\n".join(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def indices_hash(idx: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(idx, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainOnlyCatEncoder:
    """Label-encode categoricals fit on train only; unknown → dedicated code."""

    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.map_: dict[str, int] = {}
        self.unknown_code: int = 0

    def fit(self, values: pd.Series | np.ndarray) -> "TrainOnlyCatEncoder":
        vals = pd.Series(values).astype(str).fillna(UNKNOWN_CAT_TOKEN)
        classes = sorted(vals.unique().tolist())
        self.classes_ = classes
        self.map_ = {c: i for i, c in enumerate(classes)}
        self.unknown_code = len(classes)  # reserved code for unseen
        return self

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        vals = pd.Series(values).astype(str).fillna(UNKNOWN_CAT_TOKEN)
        return np.array(
            [self.map_.get(v, self.unknown_code) for v in vals], dtype=np.float32
        )


def assert_features_safe(feature_cols: list[str], label_col: str) -> None:
    """Fatal if any target-derived / blacklist column leaks into X."""
    norm_feats = [normalize_col(c) for c in feature_cols]
    bad = []
    for raw, norm in zip(feature_cols, norm_feats):
        if norm in TARGET_BLACKLIST_CANONICAL:
            bad.append(raw)
        if norm == normalize_col(label_col):
            bad.append(raw)
    if bad:
        raise AssertionError(
            f"FATAL: target-derived columns in feature list: {sorted(set(bad))}"
        )
    # Explicit required checks
    assert "label" not in norm_feats, "FATAL: label must not appear in X"
    assert normalize_col(label_col) not in norm_feats, (
        f"FATAL: target column {label_col!r} must not appear in X"
    )
    # Case / spelling variants deliberately injected would also hit blacklist
    for variant in ("Label", "TYPE", "Category", "ATTACK", "label ", " Type"):
        assert normalize_col(variant) in TARGET_BLACKLIST_CANONICAL or normalize_col(
            variant
        ) in {"label", "type", "category", "attack"}


def load_and_split(raw_path: Path, seed: int) -> dict[str, Any]:
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"ToN-IoT raw CSV not found at {raw_path}. "
            "Place train_test_network.csv under data/raw/toniot/."
        )

    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    # Normalize column names for matching (keep original map)
    col_map = {c: normalize_col(c) for c in df.columns}
    inv = {}
    for orig, norm in col_map.items():
        inv.setdefault(norm, orig)

    # Resolve allowlist against actual columns
    missing = [f for f in FEATURE_ALLOWLIST if normalize_col(f) not in inv]
    if missing:
        raise KeyError(f"Allowlist features missing from CSV: {missing}")

    feature_cols = [inv[normalize_col(f)] for f in FEATURE_ALLOWLIST]
    # Stable ordered list uses canonical allowlist names
    ordered_features = list(FEATURE_ALLOWLIST)

    print(f"[ton-safe] ordered feature list ({len(ordered_features)}): {ordered_features}")
    assert_features_safe(ordered_features, LABEL_COL)
    assert "label" not in [normalize_col(c) for c in ordered_features]
    assert normalize_col(LABEL_COL) not in [normalize_col(c) for c in ordered_features]

    label_raw = inv.get(normalize_col(LABEL_COL))
    if label_raw is None:
        raise KeyError(f"Target column {LABEL_COL!r} not found in CSV")

    n_before = len(df)
    y_str = df[label_raw].astype(str)
    valid = y_str.notna() & (y_str.str.strip() != "") & (y_str.str.lower() != "nan")
    n_dropped_target = int((~valid).sum())
    df = df.loc[valid].reset_index(drop=True)
    y_str = df[label_raw].astype(str)

    # Optional: drop exact full-row duplicates before split (record count)
    n_before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dup_removed = n_before_dedup - len(df)
    y_str = df[label_raw].astype(str)

    # Build raw feature frame (pre-encode) with allowlist order
    X_raw = pd.DataFrame(index=df.index)
    for canon, actual in zip(FEATURE_ALLOWLIST, feature_cols):
        X_raw[canon] = df[actual].values

    # Row ids for split persistence / disjointness
    row_ids = np.arange(len(df), dtype=np.int64)

    # Label encode target globally by sorted unique names (mapping only — not a feature fit)
    class_names = sorted(y_str.unique().tolist())
    label_to_id = {n: i for i, n in enumerate(class_names)}
    y = np.array([label_to_id[v] for v in y_str], dtype=np.int64)

    # ---- Split BEFORE fitting encoders/scalers (60/20/20 stratified) ----
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        row_ids,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.25,  # 0.25 of 0.8 → 0.2 overall
        random_state=seed,
        stratify=y_temp,
    )

    # Disjointness
    s_tr, s_va, s_te = set(idx_train), set(idx_val), set(idx_test)
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te), (
        "FATAL: train/val/test row indices are not disjoint"
    )

    # Fit categorical encoders on TRAIN only
    cat_encoders: dict[str, TrainOnlyCatEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        enc = TrainOnlyCatEncoder()
        enc.fit(X_raw.loc[idx_train, col])
        cat_encoders[col] = enc

    def encode_partition(indices: np.ndarray) -> np.ndarray:
        parts = []
        for col in NUMERIC_FEATURES:
            v = pd.to_numeric(X_raw.loc[indices, col], errors="coerce")
            v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
            parts.append(v.values.reshape(-1, 1))
        for col in CATEGORICAL_FEATURES:
            parts.append(cat_encoders[col].transform(X_raw.loc[indices, col]).reshape(-1, 1))
        return np.hstack(parts).astype(np.float32)

    X_train = encode_partition(idx_train)
    X_val = encode_partition(idx_val)
    X_test = encode_partition(idx_test)

    # Scaler fit on train only
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Final leakage asserts on feature matrix column names (not values)
    assert_features_safe(ordered_features, LABEL_COL)

    cat_vocab = {
        col: {
            "classes": enc.classes_,
            "unknown_code": enc.unknown_code,
            "n_fit_rows": int(len(idx_train)),
        }
        for col, enc in cat_encoders.items()
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": class_names,
        "feature_columns": ordered_features,
        "feature_hash": feature_list_sha256(ordered_features),
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "n_dropped_invalid_target": n_dropped_target,
        "n_duplicates_removed": n_dup_removed,
        "n_rows_raw": n_before,
        "n_rows_after_clean": len(df),
        "source_sha256": file_sha256(raw_path),
        "cat_vocab": cat_vocab,
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
        "label_col": LABEL_COL,
        "label_mapping": {n: int(i) for i, n in enumerate(class_names)},
        "smote": False,
        "split": {
            "scheme": "stratified_random_60_20_20",
            "note": (
                "Corrected leakage-safe random split (not official temporal/host split). "
                "No official train/test file pair present under data/raw/toniot/."
            ),
            "seed": seed,
            "train_frac": 0.6,
            "val_frac": 0.2,
            "test_frac": 0.2,
        },
    }


def balanced_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """sklearn-style balanced weights from train labels only."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n = float(len(y))
    return (n / (n_classes * counts)).astype(np.float32)


def make_loader(X, y, batch_size, shuffle, device, drop_last=False):
    pin = device.type == "cuda"
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=pin,
        drop_last=drop_last,
    )


@torch.no_grad()
def eval_model(model, loader, device, class_names):
    model.eval()
    preds, targets = [], []
    use_amp = device.type == "cuda"
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return compute_classification_metrics(y_true, y_pred, class_names), y_pred


def build_model_cfg(n_features: int, n_classes: int) -> dict:
    return {
        "model": {
            "input_features": n_features,
            "projection_dim": 64,
            "reshape": [2, 32],
            "cnn_filters_1": 64,
            "cnn_filters_2": 128,
            "cnn_kernel_size": 3,
            "pool_size": 2,
            "bilstm_units_1": 128,
            "bilstm_units_2": 64,
            "dense_units": 64,
            "dropout_rate": 0.3,
            "num_classes": n_classes,
            "attention_heads": 4,
            "attention_dropout": 0.1,
        }
    }


def train_cnn(
    model,
    train_loader,
    val_loader,
    device,
    class_names,
    class_weights: torch.Tensor,
    epochs: int,
    lr: float,
    patience: int,
):
    """Hard-label CNN with class-weighted CE; select by val macro-F1. No KD."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    best_f1 = -1.0
    best_state = None
    best_val_metrics = None
    bad = 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        scheduler.step()

        val_metrics, _ = eval_model(model, val_loader, device, class_names)
        vf1 = float(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": ep,
                "train_loss": running / max(n, 1),
                "val_macro_f1": vf1,
            }
        )
        print(f"  [CNN] ep {ep:02d} loss={running/max(n,1):.4f} val_macro_f1={vf1:.4f}")
        if vf1 > best_f1 + 1e-6:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [CNN] early stop at ep {ep} (best val={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_metrics, {"history": history, "best_val_macro_f1": best_f1}


def train_rf(X_tr, y_tr, X_va, y_va, X_te, y_te, class_names, seed: int):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )
    rf.fit(X_tr, y_tr)
    pred_va = rf.predict(X_va)
    pred_te = rf.predict(X_te)
    val_m = compute_classification_metrics(y_va, pred_va, class_names)
    test_m = compute_classification_metrics(y_te, pred_te, class_names)
    return rf, val_m, test_m, pred_te


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument(
        "--raw-path",
        type=str,
        default=str(RAW_PATH),
        help="Path to train_test_network.csv",
    )
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"ToN-IoT leakage-safe protocol: {PROTOCOL_ID}")
    print("=" * 60)

    data = load_and_split(Path(args.raw_path), args.seed)
    n_features = data["X_train"].shape[1]
    n_classes = len(data["class_names"])
    assert n_features == len(data["feature_columns"]) == 13
    assert "label" not in data["feature_columns"]
    assert LABEL_COL not in data["feature_columns"]

    print(
        f"[ton-safe] split train={len(data['y_train'])} "
        f"val={len(data['y_val'])} test={len(data['y_test'])} "
        f"classes={n_classes} feats={n_features}"
    )
    print(f"[ton-safe] feature_hash={data['feature_hash'][:16]}…")
    print(f"[ton-safe] train counts: {dict(Counter(data['y_train']))}")
    print(f"[ton-safe] SMOTE=False  KD=False  class_weight=balanced")

    # ---- RF (same split / features) ----
    print("[ton-safe] training RF (class_weight=balanced)…")
    t_rf = time.time()
    rf, rf_val, rf_test, rf_pred_te = train_rf(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        data["class_names"],
        args.seed,
    )
    rf_wall = time.time() - t_rf
    print(
        f"  RF val_macro_f1={rf_val['macro_f1']:.4f} "
        f"test_macro_f1={rf_test['macro_f1']:.4f} ({rf_wall:.1f}s)"
    )

    # ---- CNN hard-label, class-weighted CE ----
    cw = balanced_class_weights(data["y_train"], n_classes)
    print(f"[ton-safe] class weights (train-only): {cw.tolist()}")

    model_cfg = build_model_cfg(n_features, n_classes)
    model = CNNBiLSTMAttention(model_cfg).to(device)
    train_loader = make_loader(
        data["X_train"], data["y_train"], args.batch_size, True, device, drop_last=True
    )
    val_loader = make_loader(
        data["X_val"], data["y_val"], args.batch_size, False, device
    )
    test_loader = make_loader(
        data["X_test"], data["y_test"], args.batch_size, False, device
    )

    print("[ton-safe] training CNN-BiLSTM-Attention (hard labels, weighted CE)…")
    t_cnn = time.time()
    cnn_val, cnn_hist = train_cnn(
        model,
        train_loader,
        val_loader,
        device,
        data["class_names"],
        torch.from_numpy(cw),
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )
    cnn_wall = time.time() - t_cnn

    # Test once after val selection
    cnn_test, cnn_pred_te = eval_model(model, test_loader, device, data["class_names"])
    print(
        f"  CNN val_macro_f1={cnn_val['macro_f1']:.4f} "
        f"test_macro_f1={cnn_test['macro_f1']:.4f} ({cnn_wall:.1f}s)"
    )

    ckpt_path = CKPT_DIR / f"cnn_hardlabel_seed{args.seed}.pth"
    torch.save(model.state_dict(), ckpt_path)

    # Predictions (compact)
    pred_path = OUT_DIR / f"predictions_seed{args.seed}.npz"
    np.savez_compressed(
        pred_path,
        y_test=data["y_test"],
        rf_pred=rf_pred_te.astype(np.int64),
        cnn_pred=cnn_pred_te.astype(np.int64),
    )

    counts = {
        "train": {
            data["class_names"][i]: int(c)
            for i, c in sorted(Counter(data["y_train"]).items())
        },
        "val": {
            data["class_names"][i]: int(c)
            for i, c in sorted(Counter(data["y_val"]).items())
        },
        "test": {
            data["class_names"][i]: int(c)
            for i, c in sorted(Counter(data["y_test"]).items())
        },
        "n_train": int(len(data["y_train"])),
        "n_val": int(len(data["y_val"])),
        "n_test": int(len(data["y_test"])),
    }

    result = {
        "valid": True,
        "use_in_manuscript": True,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "toniot_corrected_leakage_safe_minimal",
        "label": "corrected leakage-safe random split",
        "seed": args.seed,
        "split_seed": data["split"]["seed"],
        "feature_columns": data["feature_columns"],
        "feature_hash_sha256": data["feature_hash"],
        "target_column": data["label_col"],
        "label_mapping": data["label_mapping"],
        "class_names": data["class_names"],
        "n_features": n_features,
        "n_classes": n_classes,
        "counts": counts,
        "split": {
            **data["split"],
            "idx_train_sha256": indices_hash(data["idx_train"]),
            "idx_val_sha256": indices_hash(data["idx_val"]),
            "idx_test_sha256": indices_hash(data["idx_test"]),
            "disjoint": True,
        },
        "preprocessing": {
            "encoders_fit_on": "train_only",
            "scaler_fit_on": "train_only",
            "smote": False,
            "unknown_category_token": UNKNOWN_CAT_TOKEN,
            "cat_vocab": data["cat_vocab"],
            "n_dropped_invalid_target": data["n_dropped_invalid_target"],
            "n_duplicates_removed": data["n_duplicates_removed"],
            "n_rows_raw": data["n_rows_raw"],
            "n_rows_after_clean": data["n_rows_after_clean"],
            "source_file": str(Path(args.raw_path).relative_to(PROJECT_ROOT))
            if Path(args.raw_path).is_relative_to(PROJECT_ROOT)
            else str(args.raw_path),
            "source_sha256": data["source_sha256"],
            "class_weights_train": cw.tolist(),
            "class_weights_formula": "n_samples / (n_classes * count_c)",
        },
        "rf": {
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "class_weight": "balanced",
            "val_metrics": rf_val,
            "test_metrics": rf_test,
            "val_macro_f1": float(rf_val["macro_f1"]),
            "test_macro_f1": float(rf_test["macro_f1"]),
            "per_class_f1_test": {
                k: v["f1"] for k, v in rf_test["per_class"].items()
            },
            "wall_sec": float(rf_wall),
        },
        "cnn": {
            "model": "CNNBiLSTMAttention",
            "loss": "class_weighted_cross_entropy",
            "kd": False,
            "focal": False,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "val_metrics": cnn_val,
            "test_metrics": cnn_test,
            "val_macro_f1": float(cnn_val["macro_f1"]),
            "test_macro_f1": float(cnn_test["macro_f1"]),
            "per_class_f1_test": {
                k: v["f1"] for k, v in cnn_test["per_class"].items()
            },
            "history": cnn_hist["history"],
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "wall_sec": float(cnn_wall),
        },
        "comparators_note": (
            "Do not compare these metrics against invalidated clean-path "
            "results (0.9526 CNN / 0.9851 RF). Those are marked invalid."
        ),
        "bot_champion_untouched": True,
        "git_sha": git_sha(PROJECT_ROOT),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "wall_sec": float(time.time() - t0),
        "predictions_file": str(pred_path.relative_to(PROJECT_ROOT)),
    }

    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    (OUT_DIR / f"seed{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")

    md = [
        "# ToN-IoT corrected leakage-safe results\n\n",
        f"- Protocol: `{PROTOCOL_ID}`\n",
        f"- Features ({n_features}): `{data['feature_columns']}`\n",
        f"- Feature SHA-256: `{data['feature_hash']}`\n",
        f"- Split seed: **{args.seed}** (60/20/20 stratified, train-only preprocess)\n",
        f"- RF test macro-F1: **{rf_test['macro_f1']:.4f}** "
        f"(val {rf_val['macro_f1']:.4f})\n",
        f"- CNN test macro-F1: **{cnn_test['macro_f1']:.4f}** "
        f"(val {cnn_val['macro_f1']:.4f})\n",
        f"- valid: true | use_in_manuscript: true\n",
        f"- SMOTE: false | KD: false\n",
    ]
    (OUT_DIR / "table.md").write_text("".join(md))

    print(
        json.dumps(
            {
                "valid": True,
                "protocol_id": PROTOCOL_ID,
                "rf_test_macro_f1": float(rf_test["macro_f1"]),
                "cnn_test_macro_f1": float(cnn_test["macro_f1"]),
                "rf_val_macro_f1": float(rf_val["macro_f1"]),
                "cnn_val_macro_f1": float(cnn_val["macro_f1"]),
                "out": str(out_json.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
