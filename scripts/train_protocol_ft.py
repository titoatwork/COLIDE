#!/usr/bin/env python3
"""
Fine-tune a distill checkpoint on BoT-IoT under the canonical protocol (stage_b_ft).

- Data: scripts.protocol.botiot (real data, no SMOTE)
- Selection: validation macro-F1 only
- Test: only with --allow-test (sealed)
- NEVER overwrites model/best_model_botiot_twostage.pth unless --allow-overwrite-champion

Example:
  PYTHONPATH=. .venv/bin/python scripts/train_protocol_ft.py \\
    --init-checkpoint model/best_model_botiot_distill_a0.6_T10.0_focal2.pth \\
    --seed 42 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.losses import FocalLoss  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402

CHAMPION_PATH = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(X, y, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
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
    )


@torch.no_grad()
def eval_split(model, loader, device, class_names):
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
    return compute_classification_metrics(y_true, y_pred, class_names)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default="model/best_model_botiot_distill_a0.6_T10.0_focal2.pth",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Default: model/multirun/ft_seed{seed}.pth",
    )
    p.add_argument(
        "--results-path",
        type=str,
        default="",
        help="Default: benchmarks/results/multirun/ft_seed{seed}.json",
    )
    p.add_argument("--allow-test", action="store_true")
    p.add_argument(
        "--allow-overwrite-champion",
        action="store_true",
        help="Required if --save-path points at the production champion file",
    )
    p.add_argument("--config", type=str, default="config/config.yaml")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = Path(
        args.save_path
        or (PROJECT_ROOT / "model" / "multirun" / f"ft_seed{args.seed}.pth")
    )
    results_path = Path(
        args.results_path
        or (
            PROJECT_ROOT
            / "benchmarks"
            / "results"
            / "multirun"
            / f"ft_seed{args.seed}.json"
        )
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.resolve() == CHAMPION_PATH.resolve() and not args.allow_overwrite_champion:
        print(
            "ERROR: refusing to overwrite production champion without "
            "--allow-overwrite-champion",
            file=sys.stderr,
        )
        return 2

    init_ckpt = Path(args.init_checkpoint)
    if not init_ckpt.is_file():
        print(f"ERROR: init checkpoint missing: {init_ckpt}", file=sys.stderr)
        return 1

    bundle = load_botiot(stage="stage_b_ft", seed=args.seed)
    train_loader = make_loader(
        bundle.X_train, bundle.y_train, args.batch_size, True, device
    )
    val_loader = make_loader(
        bundle.X_val, bundle.y_val, args.batch_size, False, device
    )
    test_loader = make_loader(
        bundle.X_test, bundle.y_test, args.batch_size, False, device
    )

    config = load_config(PROJECT_ROOT / args.config)
    sys.path.insert(0, str(PROJECT_ROOT / "model"))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    model = CNNBiLSTMAttention(config).to(device)
    model.load_state_dict(torch.load(init_ckpt, map_location=device, weights_only=True))

    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_val_f1 = -1.0
    patience_left = args.patience
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        val_m = eval_split(model, val_loader, device, bundle.class_names)
        train_loss = running / max(n, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": val_m["macro_f1"],
                "val_balanced_accuracy": val_m["balanced_accuracy"],
                "val_min_per_class_f1": val_m["min_per_class_f1"],
            }
        )
        print(
            f"seed={args.seed} epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"val_macro_f1 {val_m['macro_f1']:.4f} | "
            f"val_bal_acc {val_m['balanced_accuracy']:.4f} | "
            f"min_cls_f1 {val_m['min_per_class_f1']:.4f}"
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

    # Reload best for final val (+ optional test)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    val_final = eval_split(model, val_loader, device, bundle.class_names)
    test_final = None
    if args.allow_test:
        test_final = eval_split(model, test_loader, device, bundle.class_names)

    elapsed = time.time() - t0
    payload = make_result_envelope(
        experiment_id=f"protocol_ft_seed{args.seed}",
        protocol_id=bundle.protocol_id,
        stage="stage_b_ft",
        seed=args.seed,
        config={
            "init_checkpoint": str(init_ckpt),
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "focal_gamma": args.focal_gamma,
            "patience": args.patience,
            "save_path": str(save_path),
        },
        metrics={
            "best_val_macro_f1": float(best_val_f1),
            "val": val_final,
            "test": test_final,
        },
        extra={
            "history": history,
            "elapsed_sec": elapsed,
            "allow_test": bool(args.allow_test),
            "device": str(device),
            "data_summary": bundle.summary(),
        },
        project_root=PROJECT_ROOT,
    )
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"DONE seed={args.seed} best_val_macro_f1={best_val_f1:.4f} elapsed={elapsed:.1f}s")
    print(f"results -> {results_path}")
    if not args.allow_test:
        print("test: SEALED (not evaluated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
