#!/usr/bin/env python3
"""
WP8 / K2 — Final CAD-CBA-v1 method on ToN-IoT (secondary dataset).

Recipe mapped from BoT package (not weight transfer — different feature space):
  - Arch: CNN–BiLSTM–Attention V3 dims (filters/BiLSTM) with ToN input_features + num_classes
  - Teachers: ensemble mean(RF, XGB, LGBM) soft labels on train
  - Stage A: KD α=0.6 T=10 + focal γ≈1.92
  - Stage B: FT focal + hpo_best train HPs (lr, batch, dropouts, wd, cosine)
  - Decode: argmax
  - Sampler: shuffle

Data: data/processed_toniot/ (13 features, 10 classes) — pre-built splits.
Test: reported once after val selection (final multi-dataset eval; not iterative fishing).

Does not touch BoT champion.

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_toniot_final_method.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention  # noqa: E402
from scripts.protocol.losses import FocalLoss  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import git_sha  # noqa: E402

TON_DIR = PROJECT_ROOT / "data" / "processed_toniot"
OUT_RES = PROJECT_ROOT / "benchmarks" / "results" / "toniot_final"
OUT_CKPT = PROJECT_ROOT / "model" / "toniot_final"
BOT_CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
BOT_MD5_EXPECTED = "80a90f7cc210276300eaa90173a5a385"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ton() -> dict[str, Any]:
    cfg = yaml.safe_load((TON_DIR / "config_toniot.yaml").read_text())
    X_train = np.load(TON_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(TON_DIR / "y_train.npy").astype(np.int64)
    X_val = np.load(TON_DIR / "X_val.npy").astype(np.float32)
    y_val = np.load(TON_DIR / "y_val.npy").astype(np.int64)
    X_test = np.load(TON_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(TON_DIR / "y_test.npy").astype(np.int64)
    class_names = list(cfg["data"]["class_names"])
    return {
        "cfg": cfg,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": class_names,
        "n_features": int(X_train.shape[1]),
        "n_classes": len(class_names),
    }


def make_loader(X, y, batch_size, shuffle, device, drop_last=False, teacher_probs=None):
    pin = device.type == "cuda"
    if teacher_probs is not None:
        ds = TensorDataset(
            torch.from_numpy(np.asarray(X, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
            torch.from_numpy(np.asarray(teacher_probs, dtype=np.float32)),
        )
    else:
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
    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return compute_classification_metrics(y_true, y_pred, class_names)


def build_model_cfg(ton_cfg: dict, hpo: dict) -> dict:
    m = dict(ton_cfg["model"])
    # CAD-CBA V3 backbone dims (same as BoT package) with ToN I/O
    m.update(
        {
            "projection_dim": 64,
            "reshape": [2, 32],
            "cnn_filters_1": 64,
            "cnn_filters_2": 128,
            "cnn_kernel_size": 3,
            "pool_size": 2,
            "bilstm_units_1": 128,
            "bilstm_units_2": 64,
            "dense_units": 64,
            "attention_heads": 4,
            "dropout_rate": float(hpo.get("dropout_rate", m.get("dropout_rate", 0.15))),
            "attention_dropout": float(
                hpo.get("attention_dropout", m.get("attention_dropout", 0.2))
            ),
            "input_features": int(ton_cfg["model"]["input_features"]),
            "num_classes": int(ton_cfg["model"]["num_classes"]),
        }
    )
    return {"model": m, "data": ton_cfg.get("data", {}), "seed": ton_cfg.get("seed", 42)}


def fit_ensemble_teacher(X_tr, y_tr, X_val, y_val, seed: int) -> dict[str, Any]:
    """RF + XGB + LGBM soft-label ensemble (mean proba)."""
    teachers = {}
    # RF
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=seed,
        class_weight=None,
    )
    rf.fit(X_tr, y_tr)
    rf_tr = rf.predict_proba(X_tr)
    rf_va = rf.predict_proba(X_val)
    teachers["rf"] = {
        "val_macro_f1": float(
            f1_score(y_val, rf_va.argmax(1), average="macro", zero_division=0)
        ),
    }

    # XGB
    try:
        import xgboost as xgb

        n_classes = int(y_tr.max()) + 1
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        x_tr = clf.predict_proba(X_tr)
        x_va = clf.predict_proba(X_val)
        teachers["xgb"] = {
            "val_macro_f1": float(
                f1_score(y_val, x_va.argmax(1), average="macro", zero_division=0)
            ),
        }
    except Exception as e:
        teachers["xgb"] = {"error": str(e)}
        x_tr = rf_tr
        x_va = rf_va

    # LGBM
    try:
        import lightgbm as lgb

        n_classes = int(y_tr.max()) + 1
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            max_depth=8,
            objective="multiclass",
            num_class=n_classes,
            class_weight="balanced",
            min_child_samples=5,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        l_tr = clf.predict_proba(X_tr)
        l_va = clf.predict_proba(X_val)
        teachers["lgbm"] = {
            "val_macro_f1": float(
                f1_score(y_val, l_va.argmax(1), average="macro", zero_division=0)
            ),
        }
    except Exception as e:
        teachers["lgbm"] = {"error": str(e)}
        l_tr = rf_tr
        l_va = rf_va

    ens_tr = (rf_tr + x_tr + l_tr) / 3.0
    ens_va = (rf_va + x_va + l_va) / 3.0
    teachers["ensemble"] = {
        "val_macro_f1": float(
            f1_score(y_val, ens_va.argmax(1), average="macro", zero_division=0)
        ),
    }
    # Also RF-only baseline metrics for paper table
    teachers["rf"]["val_macro_f1"] = float(
        f1_score(y_val, rf_va.argmax(1), average="macro", zero_division=0)
    )
    return {
        "teachers": teachers,
        "train_soft": ens_tr.astype(np.float32),
        "val_soft": ens_va.astype(np.float32),
        "rf_val_macro_f1": teachers["rf"]["val_macro_f1"],
        "ensemble_val_macro_f1": teachers["ensemble"]["val_macro_f1"],
    }


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-7, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    if T != 1.0:
        p = np.exp(np.log(p) / T)
        p = p / p.sum(axis=1, keepdims=True)
    return np.clip(p, 1e-7, 1.0).astype(np.float32)


def train_kd_stage(
    model,
    train_loader,
    val_loader,
    device,
    class_names,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    alpha: float,
    temperature: float,
    focal_gamma: float,
    patience: int,
    scheduler_name: str,
) -> tuple[dict, dict]:
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (
        CosineAnnealingLR(opt, T_max=max(epochs, 1))
        if scheduler_name == "cosine"
        else None
    )
    focal = FocalLoss(gamma=focal_gamma)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    best_state = None
    best_metrics = {}
    bad = 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb, tb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss_ce = focal(logits, yb)
                log_p = F.log_softmax(logits / temperature, dim=1)
                soft = tb
                # soft already temperature-scaled offline; still use KL form
                loss_kd = F.kl_div(log_p, soft, reduction="batchmean") * (temperature**2)
                loss = alpha * loss_ce + (1.0 - alpha) * loss_kd
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item())
            n_batches += 1
        if sched is not None:
            sched.step()
        metrics = eval_model(model, val_loader, device, class_names)
        f1 = metrics["macro_f1"]
        history.append({"epoch": ep, "train_loss": total_loss / max(n_batches, 1), "val_macro_f1": f1})
        print(f"  [KD] ep {ep}/{epochs} val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [KD] early stop at ep {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, {"history": history, "best_val_macro_f1": best_f1}


def train_ft_stage(
    model,
    train_loader,
    val_loader,
    device,
    class_names,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    focal_gamma: float,
    patience: int,
    scheduler_name: str,
) -> tuple[dict, dict]:
    # Seed best with pre-FT (KD) val so FT cannot regress the selected checkpoint.
    pre_metrics = eval_model(model, val_loader, device, class_names)
    best_f1 = float(pre_metrics["macro_f1"])
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_metrics = pre_metrics
    print(f"  [FT] pre-FT (KD) val_macro_f1={best_f1:.4f} (floor)")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (
        CosineAnnealingLR(opt, T_max=max(epochs, 1))
        if scheduler_name == "cosine"
        else None
    )
    focal = FocalLoss(gamma=focal_gamma)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    bad = 0
    history = [{"epoch": 0, "train_loss": None, "val_macro_f1": best_f1, "note": "pre_ft_kd"}]
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss = focal(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item())
            n_batches += 1
        if sched is not None:
            sched.step()
        metrics = eval_model(model, val_loader, device, class_names)
        f1 = metrics["macro_f1"]
        history.append({"epoch": ep, "train_loss": total_loss / max(n_batches, 1), "val_macro_f1": f1})
        print(f"  [FT] ep {ep}/{epochs} val_macro_f1={f1:.4f}")
        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [FT] early stop at ep {ep} (kept best val={best_f1:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, {
        "history": history,
        "best_val_macro_f1": best_f1,
        "pre_ft_val_macro_f1": float(pre_metrics["macro_f1"]),
        "ft_improved_over_kd": bool(best_f1 > float(pre_metrics["macro_f1"]) + 1e-6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kd-epochs", type=int, default=20)
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument(
        "--kd-lr",
        type=float,
        default=1e-3,
        help="Stage-A KD lr (from-scratch on ToN; BoT hpo_best lr is FT-scale)",
    )
    ap.add_argument(
        "--ft-lr",
        type=float,
        default=1e-4,
        help="Stage-B FT lr (between BoT hpo_best ~6e-5 and KD lr)",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-test", action="store_true", help="Keep ToN test sealed")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_RES.mkdir(parents=True, exist_ok=True)
    OUT_CKPT.mkdir(parents=True, exist_ok=True)

    bot_md5 = _md5(BOT_CHAMPION)
    if bot_md5 != BOT_MD5_EXPECTED:
        print(f"ERROR: BoT champion md5 mismatch {bot_md5}", file=sys.stderr)
        return 2

    data = load_ton()
    hpo_doc = yaml.safe_load((PROJECT_ROOT / "config" / "hpo_best.yaml").read_text())
    hpo = hpo_doc.get("hpo", {}).get("best_params", {})
    # Recipe transfer: loss/KD/arch from CAD-CBA-v1. Absolute BoT FT lr (~6e-5) is too
    # low for from-scratch ToN — use ToN-scale KD/FT lrs; keep γ/dropouts/wd/cosine.
    batch_size = int(args.batch_size)
    focal_gamma = float(hpo.get("focal_gamma", 1.92))
    weight_decay = float(hpo.get("weight_decay", 1e-4))
    scheduler_name = str(hpo.get("scheduler", "cosine"))
    kd_lr = float(args.kd_lr)
    ft_lr = float(args.ft_lr)

    print(f"[ton] data train={data['X_train'].shape} val={data['X_val'].shape} "
          f"test={data['X_test'].shape} classes={data['n_classes']} feats={data['n_features']}")
    print(f"[ton] HPs: kd_lr={kd_lr} ft_lr={ft_lr} bs={batch_size} γ={focal_gamma} "
          f"wd={weight_decay} sched={scheduler_name}")

    print("[ton] fitting ensemble teacher (RF+XGB+LGBM)…")
    t_teacher = time.time()
    teacher_pack = fit_ensemble_teacher(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"], args.seed
    )
    teacher_wall = time.time() - t_teacher
    print(f"  teachers val macro-F1: {teacher_pack['teachers']}")

    soft_tr = temperature_scale(teacher_pack["train_soft"], args.temperature)

    model_cfg = build_model_cfg(data["cfg"], hpo)
    model = CNNBiLSTMAttention(model_cfg).to(device)

    train_kd_loader = make_loader(
        data["X_train"],
        data["y_train"],
        batch_size,
        True,
        device,
        drop_last=True,
        teacher_probs=soft_tr,
    )
    val_loader = make_loader(data["X_val"], data["y_val"], batch_size, False, device)
    train_ft_loader = make_loader(
        data["X_train"], data["y_train"], batch_size, True, device, drop_last=True
    )

    print("[ton] Stage A KD…")
    kd_metrics, kd_hist = train_kd_stage(
        model,
        train_kd_loader,
        val_loader,
        device,
        data["class_names"],
        epochs=args.kd_epochs,
        lr=kd_lr,
        weight_decay=weight_decay,
        alpha=args.alpha,
        temperature=args.temperature,
        focal_gamma=focal_gamma,
        patience=args.patience,
        scheduler_name=scheduler_name,
    )
    kd_path = OUT_CKPT / f"kd_ensemble_seed{args.seed}.pth"
    torch.save(model.state_dict(), kd_path)

    print("[ton] Stage B FT…")
    ft_metrics, ft_hist = train_ft_stage(
        model,
        train_ft_loader,
        val_loader,
        device,
        data["class_names"],
        epochs=args.ft_epochs,
        lr=ft_lr,
        weight_decay=weight_decay,
        focal_gamma=focal_gamma,
        patience=args.patience,
        scheduler_name=scheduler_name,
    )
    ft_path = OUT_CKPT / f"ft_cad_cba_v1_seed{args.seed}.pth"
    torch.save(model.state_dict(), ft_path)

    test_metrics = None
    if not args.skip_test:
        test_loader = make_loader(data["X_test"], data["y_test"], batch_size, False, device)
        test_metrics = eval_model(model, test_loader, device, data["class_names"])
        print(f"  [TEST] macro_f1={test_metrics['macro_f1']:.4f}")

    # Classical RF test/val for same split (already have val from teacher)
    rf_val = teacher_pack["rf_val_macro_f1"]
    # RF test
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=args.seed)
    rf.fit(data["X_train"], data["y_train"])
    rf_test_pred = rf.predict(data["X_test"])
    rf_test_f1 = float(
        f1_score(data["y_test"], rf_test_pred, average="macro", zero_division=0)
    )
    rf_test_metrics = compute_classification_metrics(
        data["y_test"], rf_test_pred, data["class_names"]
    )

    historical = {
        "toniot_clean_cnn_macro_f1": 0.9526,
        "toniot_clean_rf_macro_f1": 0.9851,
        "note": "Historical clean used 26 features; this run uses processed_toniot 13-feat splits",
        "features_this_run": data["n_features"],
        "features_historical_clean": 26,
    }

    val_f1 = float(ft_metrics.get("macro_f1", 0.0))
    test_f1 = float(test_metrics["macro_f1"]) if test_metrics else None

    summary = {
        "experiment_id": "wp8_toniot_final_cad_cba_v1",
        "tracker": ["K2", "K3_partial"],
        "work_package": "WP8",
        "protocol_id": "toniot_processed_v1",
        "method": "CAD-CBA-v1 mapped (V3 + ensemble KD + focal + hpo_best HPs)",
        "seed": args.seed,
        "data": {
            "dir": str(TON_DIR.relative_to(PROJECT_ROOT)),
            "n_train": int(len(data["y_train"])),
            "n_val": int(len(data["y_val"])),
            "n_test": int(len(data["y_test"])),
            "n_features": data["n_features"],
            "n_classes": data["n_classes"],
            "class_names": data["class_names"],
            "feature_columns": data["cfg"]["data"].get("feature_columns"),
        },
        "hpo_params_used": {
            "kd_lr": kd_lr,
            "ft_lr": ft_lr,
            "batch_size": batch_size,
            "focal_gamma": focal_gamma,
            "weight_decay": weight_decay,
            "scheduler": scheduler_name,
            "dropout_rate": model_cfg["model"]["dropout_rate"],
            "attention_dropout": model_cfg["model"]["attention_dropout"],
            "alpha_kd": args.alpha,
            "temperature": args.temperature,
            "bot_hpo_best_lr_not_used_for_from_scratch": float(hpo.get("lr", 5.89e-5)),
            "note": (
                "Recipe transfer (V3+ensemble KD+focal+γ/dropouts/wd/cosine). "
                "BoT hpo_best lr is FT-scale on large BoT; ToN uses from-scratch kd_lr/ft_lr."
            ),
        },
        "teachers": teacher_pack["teachers"],
        "teacher_fit_wall_sec": teacher_wall,
        "stage_a_kd": {
            "best_val_macro_f1": float(kd_hist["best_val_macro_f1"]),
            "metrics": kd_metrics,
            "checkpoint": str(kd_path.relative_to(PROJECT_ROOT)),
            "history": kd_hist["history"],
        },
        "stage_b_ft": {
            "best_val_macro_f1": float(ft_hist["best_val_macro_f1"]),
            "metrics": ft_metrics,
            "checkpoint": str(ft_path.relative_to(PROJECT_ROOT)),
            "history": ft_hist["history"],
        },
        "test": {
            "sealed_skipped": bool(args.skip_test),
            "metrics": test_metrics,
            "macro_f1": test_f1,
        },
        "classical_rf_same_split": {
            "val_macro_f1": rf_val,
            "test_macro_f1": rf_test_f1,
            "test_metrics": rf_test_metrics,
        },
        "historical_reference": historical,
        "bot_champion_md5": bot_md5,
        "bot_champion_unchanged": True,
        "comparators": {
            "this_val_macro_f1": val_f1,
            "this_test_macro_f1": test_f1,
            "rf_test_macro_f1": rf_test_f1,
            "historical_clean_cnn": 0.9526,
            "historical_clean_rf": 0.9851,
        },
        "decision": "RUN_DOCUMENTED",
        "decision_note": (
            f"CAD-CBA-v1 on ToN processed (13-feat): val macro-F1={val_f1:.4f}"
            + (f", test macro-F1={test_f1:.4f}" if test_f1 is not None else ", test sealed")
            + f"; same-split RF test={rf_test_f1:.4f}. "
            "Not weight-transfer from BoT. Feature set differs from historical 26-feat clean "
            "(0.9526) — do not claim identity with that number. Multi-dataset support is "
            "evidence of recipe transferability under documented protocol."
        ),
        "wall_sec": float(time.time() - t0),
        "git_sha": git_sha(PROJECT_ROOT),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    }

    # Individual result files
    (OUT_RES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_RES / f"ft_seed{args.seed}.json").write_text(
        json.dumps(
            {
                "experiment_id": "toniot_final_ft",
                "metrics_val": ft_metrics,
                "metrics_test": test_metrics,
                "checkpoint": str(ft_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    (OUT_RES / "classical_rf.json").write_text(
        json.dumps(
            {
                "val_macro_f1": rf_val,
                "test_macro_f1": rf_test_f1,
                "test_metrics": rf_test_metrics,
            },
            indent=2,
        )
        + "\n"
    )

    md = [
        "# WP8 ToN-IoT final method (CAD-CBA-v1 mapped)\n\n",
        f"- Val macro-F1: **{val_f1:.4f}**\n",
        f"- Test macro-F1: **{test_f1}**\n" if test_f1 is not None else "- Test: sealed\n",
        f"- RF same-split test macro-F1: **{rf_test_f1:.4f}**\n",
        f"- Ensemble teacher val: **{teacher_pack['ensemble_val_macro_f1']:.4f}**\n",
        f"- Features: **{data['n_features']}** (historical clean used 26)\n",
        f"- Decision: **{summary['decision']}**\n\n",
        summary["decision_note"] + "\n",
    ]
    (OUT_RES / "table.md").write_text("".join(md))

    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "val_macro_f1": val_f1,
                "test_macro_f1": test_f1,
                "rf_test_macro_f1": rf_test_f1,
                "kd_val": kd_hist["best_val_macro_f1"],
                "wall_sec": summary["wall_sec"],
                "out": str(OUT_RES / "summary.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
