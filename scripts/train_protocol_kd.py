#!/usr/bin/env python3
"""
Protocol knowledge-distillation (WP4b): train CNN–BiLSTM student under botiot_v1.

Stage: stage_a_kd (SMOTE path; historical KD recipe).
Selection: validation macro-F1 only. Test sealed unless --allow-test.
Champion path is never written (save under model/teachers_kd/).

KD objective (historical name: legacy_teacher_smoothed_kl)
---------------------------------------------------------
Teacher probabilities are temperature-softened offline; the student KL term does
**not** apply the same temperature or the conventional T² factor. This is the
frozen champion recipe — see docs/KD_OBJECTIVES.md. Do not change the training
formula when renaming or documenting the objective.

Teachers (soft labels on train):
  none     — hard-label focal only (no KD control)
  rf       — sklearn RandomForest (n=200, historical)
  xgb      — XGBoost multi:softprob
  lgbm     — LightGBM
  ensemble — mean(RF, XGB, LGBM) predict_proba

Example:
  PYTHONPATH=. .venv/bin/python scripts/train_protocol_kd.py \\
    --teacher rf --alpha 0.6 --temperature 10 --focal-gamma 2 \\
    --seed 42 --epochs 15 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.losses import FocalLoss  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402

CHAMPION_PATH = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
TEACHERS = ("none", "rf", "xgb", "lgbm", "ensemble")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-7, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    if temperature != 1.0:
        p = np.exp(np.log(p) / temperature)
        p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-7, 1.0).astype(np.float32)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def fit_teacher(
    name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: list[str],
    seed: int,
    temperature: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return (train soft labels or None, teacher diagnostics)."""
    meta: dict[str, Any] = {"teacher": name}
    t0 = time.time()

    if name == "none":
        meta["fit_sec"] = 0.0
        meta["note"] = "no soft labels; hard-label focal only"
        return None, meta

    models: dict[str, Any] = {}

    def _fit_rf():
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)
        return clf

    def _fit_xgb():
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            n_jobs=-1,
            random_state=seed,
            tree_method="hist",
        )
        clf.fit(X_tr, y_tr)
        return clf

    def _fit_lgbm():
        from lightgbm import LGBMClassifier
        import pandas as pd

        cols = [f"f{i}" for i in range(X_tr.shape[1])]
        X_tr_df = pd.DataFrame(np.asarray(X_tr, dtype=np.float32), columns=cols)
        clf = LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
            force_col_wise=True,
        )
        clf.fit(X_tr_df, y_tr)
        # attach column names for predict
        clf._colide_cols = cols  # type: ignore[attr-defined]
        return clf

    def _proba(clf, X: np.ndarray, kind: str) -> np.ndarray:
        if kind == "lgbm":
            import pandas as pd

            cols = getattr(clf, "_colide_cols", [f"f{i}" for i in range(X.shape[1])])
            X_df = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=cols)
            return clf.predict_proba(X_df)
        return clf.predict_proba(X)

    if name == "rf":
        models["rf"] = _fit_rf()
        raw = _proba(models["rf"], X_tr, "rf")
        val_pred = models["rf"].predict(X_val)
    elif name == "xgb":
        models["xgb"] = _fit_xgb()
        raw = _proba(models["xgb"], X_tr, "xgb")
        val_pred = models["xgb"].predict(X_val)
    elif name == "lgbm":
        models["lgbm"] = _fit_lgbm()
        raw = _proba(models["lgbm"], X_tr, "lgbm")
        import pandas as pd

        cols = models["lgbm"]._colide_cols  # type: ignore[attr-defined]
        val_pred = models["lgbm"].predict(
            pd.DataFrame(np.asarray(X_val, dtype=np.float32), columns=cols)
        )
    elif name == "ensemble":
        models["rf"] = _fit_rf()
        models["xgb"] = _fit_xgb()
        models["lgbm"] = _fit_lgbm()
        raw = (
            _proba(models["rf"], X_tr, "rf")
            + _proba(models["xgb"], X_tr, "xgb")
            + _proba(models["lgbm"], X_tr, "lgbm")
        ) / 3.0
        val_probs = (
            _proba(models["rf"], X_val, "rf")
            + _proba(models["xgb"], X_val, "xgb")
            + _proba(models["lgbm"], X_val, "lgbm")
        ) / 3.0
        val_pred = np.argmax(val_probs, axis=1)
        # per-member val diagnostics
        solo = {}
        for k, kind in (("rf", "rf"), ("xgb", "xgb"), ("lgbm", "lgbm")):
            pred_k = (
                models[k].predict(X_val)
                if kind != "lgbm"
                else models[k].predict(
                    __import__("pandas").DataFrame(
                        np.asarray(X_val, dtype=np.float32),
                        columns=models[k]._colide_cols,  # type: ignore[attr-defined]
                    )
                )
            )
            solo[k] = compute_classification_metrics(y_val, pred_k, class_names)
        meta["solo_teacher_val"] = {
            k: {
                "macro_f1": solo[k]["macro_f1"],
                "min_per_class_f1": solo[k]["min_per_class_f1"],
                "theft_f1": solo[k].get("theft_f1"),
            }
            for k in solo
        }
    else:
        raise ValueError(f"unknown teacher: {name}")

    soft = _temperature_scale(raw, temperature)
    teacher_val = compute_classification_metrics(y_val, val_pred, class_names)
    meta["fit_sec"] = float(time.time() - t0)
    meta["teacher_val"] = {
        "macro_f1": teacher_val["macro_f1"],
        "balanced_accuracy": teacher_val["balanced_accuracy"],
        "min_per_class_f1": teacher_val["min_per_class_f1"],
        "theft_f1": teacher_val.get("theft_f1"),
        "normal_f1": teacher_val.get("normal_f1"),
        "per_class_f1": {k: v["f1"] for k, v in teacher_val["per_class"].items()},
    }
    meta["temperature"] = temperature
    meta["soft_label_shape"] = list(soft.shape)
    print(
        f"Teacher {name}: fit {meta['fit_sec']:.1f}s | "
        f"val_macro_f1={teacher_val['macro_f1']:.4f} | "
        f"min_cls={teacher_val['min_per_class_f1']:.4f} | "
        f"theft_f1={teacher_val.get('theft_f1', float('nan')):.4f}"
    )
    # free tree models before student training
    del models
    return soft, meta


def make_loader(
    X,
    y,
    soft: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    pin = device.type == "cuda"
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    if soft is not None:
        st = torch.from_numpy(np.asarray(soft, dtype=np.float32))
        ds = TensorDataset(xt, yt, st)
    else:
        # dummy soft column for uniform batch unpacking
        st = torch.zeros((len(y), 1), dtype=torch.float32)
        ds = TensorDataset(xt, yt, st)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=pin,
        persistent_workers=True,
    )


@torch.no_grad()
def eval_split(model, loader, device, class_names, has_soft: bool):
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
    return compute_classification_metrics(
        np.concatenate(targets), np.concatenate(preds), class_names
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=str, default="rf", choices=list(TEACHERS))
    p.add_argument("--alpha", type=float, default=0.6, help="KD mix: α*KL + (1-α)*hard")
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save-path", type=str, default="")
    p.add_argument("--results-path", type=str, default="")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="If >0, stratified subsample train (document in JSON). 0=full protocol.",
    )
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = Path(
        args.save_path
        or (
            PROJECT_ROOT
            / "model"
            / "teachers_kd"
            / f"kd_{args.teacher}_a{args.alpha}_T{args.temperature}_g{args.focal_gamma}_seed{args.seed}.pth"
        )
    )
    results_path = Path(
        args.results_path
        or (
            PROJECT_ROOT
            / "benchmarks"
            / "results"
            / "teachers_kd"
            / f"kd_{args.teacher}_seed{args.seed}.json"
        )
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.resolve() == CHAMPION_PATH.resolve():
        print("ERROR: refusing to write production champion path", file=sys.stderr)
        return 2

    print("=" * 60)
    print(
        f"PROTOCOL KD | teacher={args.teacher} α={args.alpha} T={args.temperature} "
        f"γ={args.focal_gamma} seed={args.seed} device={device}"
    )
    print("=" * 60)

    t_load = time.time()
    bundle = load_botiot(stage="stage_a_kd", seed=args.seed)
    print(f"Loaded stage_a_kd in {time.time() - t_load:.1f}s | n_train={len(bundle.y_train)}")

    X_tr, y_tr = bundle.X_train, bundle.y_train
    subsample_note = None
    if args.max_train and args.max_train < len(y_tr):
        from sklearn.model_selection import train_test_split

        # stratified subsample for compute-bound ranking (document clearly)
        X_tr, _, y_tr, _ = train_test_split(
            X_tr,
            y_tr,
            train_size=args.max_train,
            random_state=args.seed,
            stratify=y_tr,
        )
        subsample_note = {
            "max_train": args.max_train,
            "n_train_used": int(len(y_tr)),
            "note": "stratified subsample of stage_a_kd train; val/test full protocol",
        }
        print(f"Subsample train -> {len(y_tr)} (max_train={args.max_train})")

    soft, teacher_meta = fit_teacher(
        args.teacher,
        X_tr,
        y_tr,
        bundle.X_val,
        bundle.y_val,
        bundle.class_names,
        args.seed,
        args.temperature if args.teacher != "none" else 1.0,
    )
    use_kd = soft is not None and args.alpha > 0.0

    train_loader = make_loader(X_tr, y_tr, soft, args.batch_size, True, device)
    val_loader = make_loader(
        bundle.X_val, bundle.y_val, None, args.batch_size, False, device
    )
    test_loader = make_loader(
        bundle.X_test, bundle.y_test, None, args.batch_size, False, device
    )

    config = load_config(PROJECT_ROOT / args.config)
    sys.path.insert(0, str(PROJECT_ROOT / "model"))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    model = CNNBiLSTMAttention(config).to(device)
    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_val_f1 = -1.0
    patience_left = args.patience
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb, pb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                hard = criterion(logits, yb)
                if use_kd:
                    pb = pb.to(device, non_blocking=True)
                    student_log = F.log_softmax(logits, dim=1)
                    # -----------------------------------------------------------------
                    # KD objective name: legacy_teacher_smoothed_kl
                    # (see docs/KD_OBJECTIVES.md)
                    #
                    # Historical / champion recipe — DO NOT change the formula:
                    #   * teacher probs softened offline with T (prob-space p^(1/T))
                    #   * student uses log_softmax at T=1 (no student temperature)
                    #   * no conventional T^2 scaling on the KL term
                    #   * mix: alpha * KL(student || teacher_soft) + (1-alpha) * hard
                    # This is NOT canonical Hinton KD (T on both sides + T^2).
                    # -----------------------------------------------------------------
                    kd = F.kl_div(student_log, pb, reduction="batchmean")
                    loss = args.alpha * kd + (1.0 - args.alpha) * hard
                else:
                    loss = hard
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        scheduler.step()
        val_m = eval_split(model, val_loader, device, bundle.class_names, False)
        train_loss = running / max(n, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": val_m["macro_f1"],
                "val_balanced_accuracy": val_m["balanced_accuracy"],
                "val_min_per_class_f1": val_m["min_per_class_f1"],
                "val_theft_f1": val_m.get("theft_f1"),
                "val_normal_f1": val_m.get("normal_f1"),
            }
        )
        print(
            f"teacher={args.teacher} epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"val_macro_f1 {val_m['macro_f1']:.4f} | "
            f"min_cls {val_m['min_per_class_f1']:.4f} | "
            f"theft {val_m.get('theft_f1', float('nan')):.4f}"
        )

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            patience_left = args.patience
            torch.save(model.state_dict(), save_path)
            print(f"  >> saved best -> {save_path} (val_macro_f1={best_val_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    val_final = eval_split(model, val_loader, device, bundle.class_names, False)
    test_final = None
    if args.allow_test:
        test_final = eval_split(model, test_loader, device, bundle.class_names, False)

    elapsed = time.time() - t0
    payload = make_result_envelope(
        experiment_id=f"protocol_kd_{args.teacher}_seed{args.seed}",
        protocol_id=bundle.protocol_id,
        stage="stage_a_kd",
        seed=args.seed,
        config={
            "teacher": args.teacher,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "focal_gamma": args.focal_gamma,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "use_kd": use_kd,
            "save_path": str(save_path),
            "max_train": args.max_train,
            "subsample": subsample_note,
            "recipe": "CAD-CBA-v1 KD defaults (α=0.6,T=10,γ=2) unless overridden",
            # Historical objective name; formula unchanged (see docs/KD_OBJECTIVES.md)
            "kd_objective": "legacy_teacher_smoothed_kl",
            "temperature_applied_to": "teacher_probs_only",
            "t_squared_scaling": False,
            "teacher_probs_on": "teacher_training_rows",
            "loss_version": "legacy_pt_from_weighted_ce",
        },
        metrics={
            "best_val_macro_f1": float(best_val_f1),
            "val": val_final,
            "test": test_final,
            "teacher": teacher_meta.get("teacher_val"),
        },
        extra={
            "history": history,
            "elapsed_sec": elapsed,
            "teacher_meta": teacher_meta,
            "allow_test": bool(args.allow_test),
            "device": str(device),
            "data_summary": bundle.summary(),
            "n_train_used": int(len(y_tr)),
        },
        project_root=PROJECT_ROOT,
    )
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"DONE teacher={args.teacher} best_val_macro_f1={best_val_f1:.4f} "
        f"elapsed={elapsed:.1f}s -> {results_path}"
    )
    if not args.allow_test:
        print("test: SEALED (not evaluated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
