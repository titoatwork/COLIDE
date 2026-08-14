#!/usr/bin/env python3
"""
Fine-tune a distill / KD checkpoint on BoT-IoT under the canonical protocol (stage_b_ft).

- Data: scripts.protocol.botiot (real data, no SMOTE)
- Selection: validation macro-F1 only
- Test: only with --allow-test (sealed)
- NEVER overwrites model/best_model_botiot_twostage.pth unless --allow-overwrite-champion

Supports CAD-CBA-v1 train HPs (Optuna WP3): dropout, attention_dropout, weight_decay,
scheduler (AdamW + cosine/step). Optional --hpo-config fills *unset* train HPs from
hpo_best.yaml.

Config precedence (highest wins):
  1. explicit CLI arguments
  2. --hpo-config file (best_params), only for fields left unset on the CLI
  3. program defaults

Use --print-effective-config to dump the resolved config (and exit with --dry-run).

Example:
  PYTHONPATH=. .venv/bin/python scripts/train_protocol_ft.py \\
    --init-checkpoint model/teachers_kd/kd_ensemble_a0.6_T10.0_g2.0_seed42.pth \\
    --hpo-config config/hpo_best.yaml --seed 42 --epochs 10
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.losses import (  # noqa: E402
    FocalLoss,
    LogitAdjustedCrossEntropy,
    class_balanced_weights,
    class_probs_from_counts,
)
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


def make_loader(
    X,
    y,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    drop_last: bool = False,
    sampler: str = "shuffle",
) -> DataLoader:
    """
    sampler:
      - shuffle: default random shuffle (no replacement within epoch)
      - stratified: WeightedRandomSampler with inverse class-frequency weights
        (D6 class-balanced / stratified batch sampling for imbalance)
    """
    pin = device.type == "cuda"
    y_arr = np.asarray(y, dtype=np.int64)
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(y_arr),
    )
    if sampler == "stratified" and shuffle:
        # Inverse-frequency weights → each class contributes roughly equally
        # expected samples per epoch (class-balanced stratified batches).
        classes, counts = np.unique(y_arr, return_counts=True)
        freq = np.zeros(int(classes.max()) + 1, dtype=np.float64)
        for c, n in zip(classes, counts):
            freq[int(c)] = float(n)
        w = np.array([1.0 / max(freq[int(yi)], 1.0) for yi in y_arr], dtype=np.float64)
        # Normalize so sum(weights) is stable; replacement=True required for
        # minority oversample to fill num_samples == N.
        w_t = torch.as_tensor(w, dtype=torch.double)
        ws = WeightedRandomSampler(
            weights=w_t,
            num_samples=len(y_arr),
            replacement=True,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=ws,
            shuffle=False,
            num_workers=2,
            pin_memory=pin,
            drop_last=drop_last,
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


def _load_hpo_best_params(path: Path) -> dict:
    with open(path) as f:
        doc = yaml.safe_load(f)
    hpo = doc.get("hpo", doc)
    params = hpo.get("best_params") or hpo.get("params") or {}
    if not params:
        raise ValueError(f"No best_params in {path}")
    return {k: params[k] for k in params}


# Program defaults for fields that HPO may also set. CLI uses None so we can
# distinguish "user set this" from "still unset → fill from HPO then defaults".
# Precedence: explicit CLI > hpo file > these defaults.
PROGRAM_DEFAULTS = {
    "lr": 1e-4,
    "batch_size": 256,
    "focal_gamma": 2.0,
    "weight_decay": 0.0,
    "scheduler": "none",
    # dropout_rate / attention_dropout stay None → fall through to config.yaml
}

# CLI dest → key in hpo best_params
HPO_FIELD_MAP = {
    "lr": ("lr", float),
    "batch_size": ("batch_size", int),
    "focal_gamma": ("focal_gamma", float),
    "dropout_rate": ("dropout_rate", float),
    "attention_dropout": ("attention_dropout", float),
    "weight_decay": ("weight_decay", float),
    "scheduler": ("scheduler", str),
}


def _resolve_train_hps(args: argparse.Namespace) -> tuple[argparse.Namespace, str | None]:
    """
    Apply precedence: CLI (non-None) > hpo file > PROGRAM_DEFAULTS.

    Returns (args, hpo_src_path_or_None).
    """
    hpo_src = None
    hp: dict = {}
    if args.hpo_config:
        hpo_path = Path(args.hpo_config)
        if not hpo_path.is_file():
            raise FileNotFoundError(f"hpo config missing: {hpo_path}")
        hp = _load_hpo_best_params(hpo_path)
        hpo_src = str(hpo_path)

    for dest, (hp_key, caster) in HPO_FIELD_MAP.items():
        cur = getattr(args, dest)
        if cur is not None:
            continue  # explicit CLI wins
        if hp_key in hp:
            setattr(args, dest, caster(hp[hp_key]))
        elif dest in PROGRAM_DEFAULTS:
            setattr(args, dest, PROGRAM_DEFAULTS[dest])
        # else leave None (dropout → config.yaml)

    # optimizer: only auto-promote to adamw when HPO is loaded and user left auto
    if args.optimizer == "auto" and hpo_src is not None and args.weight_decay and float(args.weight_decay) > 0:
        args.optimizer = "adamw"

    return args, hpo_src


def _effective_config_dict(args: argparse.Namespace, hpo_src: str | None) -> dict:
    return {
        "precedence": "CLI > hpo file > program defaults",
        "hpo_config": hpo_src,
        "init_checkpoint": args.init_checkpoint,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "focal_gamma": args.focal_gamma,
        "dropout_rate": args.dropout_rate,
        "attention_dropout": args.attention_dropout,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "logit_tau": args.logit_tau,
        "patience": args.patience,
        "save_path": args.save_path or None,
        "results_path": args.results_path or None,
        "allow_test": args.allow_test,
        "config": args.config,
        "drop_last_train": args.drop_last_train,
        "train_sampler": args.train_sampler,
        "command": " ".join(sys.argv),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default="model/best_model_botiot_distill_a0.6_T10.0_focal2.pth",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    # None = unset → may be filled from --hpo-config then PROGRAM_DEFAULTS
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default 1e-4; or from --hpo-config). Explicit CLI wins over HPO.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default 256; or from --hpo-config). Explicit CLI wins over HPO.",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Focal gamma (default 2.0; or from --hpo-config). Explicit CLI wins over HPO.",
    )
    p.add_argument(
        "--dropout-rate",
        type=float,
        default=None,
        help="Model dropout (default: config.yaml; or from --hpo-config)",
    )
    p.add_argument(
        "--attention-dropout",
        type=float,
        default=None,
        help="Attention dropout (default: config.yaml; or from --hpo-config)",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay (default 0.0; or from --hpo-config)",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["none", "cosine", "step"],
        help="LR schedule (default none; or from --hpo-config)",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["auto", "adam", "adamw"],
        help="auto: AdamW if weight_decay>0 else Adam (legacy multirun)",
    )
    p.add_argument(
        "--hpo-config",
        type=str,
        default="",
        help="If set, fill *unset* train HPs from best_params (CLI always wins)",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="focal",
        choices=["focal", "ce", "focal_cb", "logit_adj"],
        help="Train loss: focal | ce | class-balanced focal | logit-adjusted CE",
    )
    p.add_argument("--logit-tau", type=float, default=1.0)
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
    p.add_argument(
        "--drop-last-train",
        action="store_true",
        help="drop_last on train loader (recommended for large batch + BN; HPO used this)",
    )
    p.add_argument(
        "--train-sampler",
        type=str,
        default="shuffle",
        choices=["shuffle", "stratified"],
        help=(
            "Train loader sampling: shuffle (default) or stratified "
            "(WeightedRandomSampler inverse class-frequency; D6)"
        ),
    )
    p.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print resolved train config JSON (after CLI/HPO/default merge) to stdout",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config only; do not train (implies --print-effective-config)",
    )
    args = p.parse_args()

    try:
        args, hpo_src = _resolve_train_hps(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.print_effective_config or args.dry_run:
        print(json.dumps(_effective_config_dict(args, hpo_src), indent=2, default=str))
        if args.dry_run:
            return 0

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
    # Match WP3 HPO: drop_last on train for large batches / BN stability
    drop_last = bool(args.drop_last_train) or args.batch_size >= 512
    train_loader = make_loader(
        bundle.X_train,
        bundle.y_train,
        args.batch_size,
        True,
        device,
        drop_last=drop_last,
        sampler=args.train_sampler,
    )
    val_loader = make_loader(
        bundle.X_val,
        bundle.y_val,
        args.batch_size,
        False,
        device,
        drop_last=False,
        sampler="shuffle",
    )
    test_loader = make_loader(
        bundle.X_test,
        bundle.y_test,
        args.batch_size,
        False,
        device,
        drop_last=False,
        sampler="shuffle",
    )

    config = load_config(PROJECT_ROOT / args.config)
    cfg = copy.deepcopy(config)
    if args.dropout_rate is not None:
        cfg["model"]["dropout_rate"] = float(args.dropout_rate)
    if args.attention_dropout is not None:
        cfg["model"]["attention_dropout"] = float(args.attention_dropout)

    sys.path.insert(0, str(PROJECT_ROOT / "model"))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    model = CNNBiLSTMAttention(cfg).to(device)
    model.load_state_dict(torch.load(init_ckpt, map_location=device, weights_only=True))

    y_tr_t = torch.from_numpy(bundle.y_train)
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.loss == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == "focal_cb":
        w = class_balanced_weights(y_tr_t, bundle.num_classes).to(device)
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=w)
    elif args.loss == "logit_adj":
        pi = class_probs_from_counts(y_tr_t, bundle.num_classes).to(device)
        criterion = LogitAdjustedCrossEntropy(pi, tau=args.logit_tau)
    else:
        raise ValueError(args.loss)

    optim_name = args.optimizer
    if optim_name == "auto":
        optim_name = "adamw" if args.weight_decay > 0 else "adam"
    if optim_name == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay)
        )
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=max(args.epochs // 2, 1), gamma=0.5)
    else:
        scheduler = None

    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_val_f1 = -1.0
    patience_left = args.patience
    history = []
    t0 = time.time()

    print(
        f"FT start seed={args.seed} device={device} init={init_ckpt.name} "
        f"lr={args.lr} bs={args.batch_size} γ={args.focal_gamma} "
        f"drop={cfg['model'].get('dropout_rate')} att_drop={cfg['model'].get('attention_dropout')} "
        f"wd={args.weight_decay} sched={args.scheduler} optim={optim_name} "
        f"epochs≤{args.epochs} patience={args.patience} drop_last_train={drop_last} "
        f"train_sampler={args.train_sampler}",
        flush=True,
    )

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

        if scheduler is not None:
            scheduler.step()

        val_m = eval_split(model, val_loader, device, bundle.class_names)
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
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"seed={args.seed} epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"val_macro_f1 {val_m['macro_f1']:.4f} | "
            f"val_bal_acc {val_m['balanced_accuracy']:.4f} | "
            f"min_cls_f1 {val_m['min_per_class_f1']:.4f} | "
            f"theft {val_m.get('theft_f1', float('nan')):.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
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
    effective = _effective_config_dict(args, hpo_src)
    effective["dropout_rate"] = cfg["model"].get("dropout_rate")
    effective["attention_dropout"] = cfg["model"].get("attention_dropout")
    effective["optimizer"] = optim_name
    effective["drop_last_train"] = drop_last
    effective["save_path"] = str(save_path)
    effective["init_checkpoint"] = str(init_ckpt.resolve())

    payload = make_result_envelope(
        experiment_id=f"protocol_ft_seed{args.seed}",
        protocol_id=bundle.protocol_id,
        stage="stage_b_ft",
        seed=args.seed,
        config=effective,
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
            "loss_version": "legacy_pt_from_weighted_ce",
        },
        project_root=PROJECT_ROOT,
        command=" ".join(sys.argv),
        use_in_manuscript=True,
        valid=True,
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
