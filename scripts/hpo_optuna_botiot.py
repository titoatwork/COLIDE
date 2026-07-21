#!/usr/bin/env python3
"""
WP3 — Systematic Optuna HPO for CAD-CBA-v1 under protocol (val only).

Stage: botiot_v1 / stage_b_ft (real-data fine-tune; no SMOTE).
Architecture: fixed V3 CNN–BiLSTM–Attention (CAD-CBA-v1); search train HPs that
load cleanly from the distill / ensemble-KD init checkpoint.

Selection: maximize val macro-F1 (primary). Secondary metrics logged:
  val min-per-class F1, balanced accuracy, Theft/Normal F1.
Test: SEALED (never used for trial selection).

Design (compute-honest on RTX 3050):
  Stage A — exploratory study on full protocol train, reduced epochs/patience
  Stage B — optional full-epoch retrain of top-K configs (same seed)
  Winner → config/hpo_best.yaml + benchmarks/results/hpo/*

Example:
  PYTHONPATH=. .venv/bin/python scripts/hpo_optuna_botiot.py \\
    --n-trials 20 --epochs 5 --patience 2 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import yaml
from optuna.samplers import TPESampler
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.losses import FocalLoss  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402

CHAMPION_PATH = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
DEFAULT_INIT = "model/best_model_botiot_distill_a0.6_T10.0_focal2.pth"


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
    # drop_last only for train to avoid BN issues on tiny last batch
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # in-process HPO: avoid worker fork overhead / RAM spike
        pin_memory=pin,
        drop_last=shuffle,
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


def build_model(base_config: dict, dropout: float, attention_dropout: float, device):
    sys.path.insert(0, str(PROJECT_ROOT / "model"))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    cfg = copy.deepcopy(base_config)
    cfg["model"]["dropout_rate"] = float(dropout)
    cfg["model"]["attention_dropout"] = float(attention_dropout)
    model = CNNBiLSTMAttention(cfg).to(device)
    return model


def train_one(
    *,
    bundle,
    base_config: dict,
    init_ckpt: Path,
    device: torch.device,
    params: dict[str, Any],
    epochs: int,
    patience: int,
    seed: int,
    save_path: Path | None,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:
    """Fine-tune one config; return metrics + history. Test never evaluated."""
    set_seed(seed)
    lr = float(params["lr"])
    batch_size = int(params["batch_size"])
    focal_gamma = float(params["focal_gamma"])
    dropout = float(params["dropout_rate"])
    att_drop = float(params["attention_dropout"])
    weight_decay = float(params["weight_decay"])
    scheduler_name = str(params["scheduler"])

    train_loader = make_loader(bundle.X_train, bundle.y_train, batch_size, True, device)
    val_loader = make_loader(bundle.X_val, bundle.y_val, batch_size, False, device)

    model = build_model(base_config, dropout, att_drop, device)
    state = torch.load(init_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    criterion = FocalLoss(gamma=focal_gamma)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.5)
    else:
        scheduler = None

    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_val_f1 = -1.0
    best_val_metrics: dict[str, Any] | None = None
    best_state = None
    patience_left = patience
    history: list[dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
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
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_m["macro_f1"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_min_per_class_f1": val_m["min_per_class_f1"],
            "val_theft_f1": val_m.get("theft_f1"),
            "val_normal_f1": val_m.get("normal_f1"),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"  ep {epoch:02d} | loss {train_loss:.4f} | "
            f"val_macro_f1 {val_m['macro_f1']:.4f} | "
            f"min_cls {val_m['min_per_class_f1']:.4f} | "
            f"theft {val_m.get('theft_f1', float('nan')):.4f}",
            flush=True,
        )

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_m["macro_f1"])
            best_val_metrics = val_m
            patience_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

        if trial is not None:
            trial.report(best_val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    elapsed = time.time() - t0
    if best_state is not None and save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)

    # free GPU before next trial
    del model, optimizer, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    assert best_val_metrics is not None
    return {
        "best_val_macro_f1": best_val_f1,
        "val": best_val_metrics,
        "history": history,
        "elapsed_sec": elapsed,
        "params": params,
        "save_path": str(save_path) if save_path else None,
    }


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Prof §1 train HPs (arch fixed to CAD-CBA-v1 / V3 for weight transfer)."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "focal_gamma": trial.suggest_float("focal_gamma", 0.5, 3.5),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.10, 0.50),
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.30),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "scheduler": trial.suggest_categorical("scheduler", ["none", "cosine", "step"]),
    }


BASELINE_PARAMS = {
    "lr": 1e-4,
    "batch_size": 256,
    "focal_gamma": 2.0,
    "dropout_rate": 0.3,
    "attention_dropout": 0.1,
    "weight_decay": 1e-6,  # AdamW needs >0; near-zero ≈ prior Adam
    "scheduler": "none",
}


def write_hpo_best_yaml(
    path: Path,
    *,
    params: dict[str, Any],
    metrics: dict[str, Any],
    study_name: str,
    trial_number: int,
    seed: int,
    init_checkpoint: str,
    git_sha: str,
) -> None:
    doc = {
        "hpo": {
            "study_name": study_name,
            "trial_number": trial_number,
            "seed": seed,
            "protocol_id": "botiot_v1",
            "stage": "stage_b_ft",
            "init_checkpoint": init_checkpoint,
            "architecture": "cnn_bilstm_v3_attention (CAD-CBA-v1 fixed)",
            "objective": "maximize val_macro_f1 (test sealed)",
            "git_sha": git_sha,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "best_params": params,
            "best_val_metrics": {
                "macro_f1": metrics.get("best_val_macro_f1"),
                "min_per_class_f1": metrics.get("val", {}).get("min_per_class_f1"),
                "balanced_accuracy": metrics.get("val", {}).get("balanced_accuracy"),
                "theft_f1": metrics.get("val", {}).get("theft_f1"),
                "normal_f1": metrics.get("val", {}).get("normal_f1"),
            },
            "note": (
                "Architecture HPs (CNN filters/BiLSTM dims/kernel) not searched here: "
                "CAD-CBA-v1 freezes V3 for KD weight transfer. "
                "Arch search deferred to WP2c if package plateaus."
            ),
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=20, help="Stage-A Optuna trials (incl. baseline)")
    p.add_argument("--epochs", type=int, default=5, help="Max epochs per trial (Stage A)")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-checkpoint", type=str, default=DEFAULT_INIT)
    p.add_argument("--study-name", type=str, default="botiot_stage_b_ft_hpo_v1")
    p.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna storage URL; default sqlite under benchmarks/results/hpo/",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="benchmarks/results/hpo",
    )
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default="model/hpo",
    )
    p.add_argument(
        "--hpo-best-yaml",
        type=str,
        default="config/hpo_best.yaml",
    )
    p.add_argument(
        "--refine-top-k",
        type=int,
        default=3,
        help="Stage B: retrain top-K full epochs on full data (0=skip)",
    )
    p.add_argument("--refine-epochs", type=int, default=10)
    p.add_argument("--refine-patience", type=int, default=3)
    p.add_argument(
        "--enqueue-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enqueue known multirun baseline params as trial 0",
    )
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument(
        "--timeout-sec",
        type=float,
        default=0.0,
        help="Optional wall-clock budget for study.optimize (0=unlimited)",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=0,
        help=(
            "If >0, stratified subsample of train for Stage A only "
            "(val always full protocol). Stage B refine always uses full train. "
            "0 = full train for all stages."
        ),
    )
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    ckpt_dir = PROJECT_ROOT / args.ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///{(out_dir / 'study.db').resolve()}"
    init_ckpt = Path(args.init_checkpoint)
    if not init_ckpt.is_file():
        print(f"ERROR: init checkpoint missing: {init_ckpt}", file=sys.stderr)
        return 1
    if not CHAMPION_PATH.is_file():
        print(f"ERROR: champion missing: {CHAMPION_PATH}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} storage={storage}", flush=True)
    print(f"init={init_ckpt} n_trials={args.n_trials} epochs={args.epochs}", flush=True)

    # Champion md5 recorded for audit (never written)
    import hashlib

    champ_md5 = hashlib.md5(CHAMPION_PATH.read_bytes()).hexdigest()
    print(f"champion_md5={champ_md5} (must remain 80a90f7cc210276300eaa90173a5a385)", flush=True)

    print("Loading protocol data (once)...", flush=True)
    t_load = time.time()
    bundle_full = load_botiot(stage="stage_b_ft", seed=args.seed)
    base_config = load_config(PROJECT_ROOT / args.config)
    print(
        f"data loaded in {time.time() - t_load:.1f}s | "
        f"train={len(bundle_full.y_train)} val={len(bundle_full.y_val)} test=SEALED",
        flush=True,
    )

    # Stage-A train view (optional stratified subsample); val always full.
    subsample_note: dict[str, Any] | None = None
    if args.max_train and args.max_train < len(bundle_full.y_train):
        X_tr, _, y_tr, _ = train_test_split(
            bundle_full.X_train,
            bundle_full.y_train,
            train_size=args.max_train,
            stratify=bundle_full.y_train,
            random_state=args.seed,
        )
        # shallow copy of bundle fields with reduced train
        from dataclasses import replace

        bundle_a = replace(bundle_full, X_train=X_tr, y_train=y_tr)
        subsample_note = {
            "max_train": args.max_train,
            "n_train_stage_a": int(len(y_tr)),
            "n_train_full": int(len(bundle_full.y_train)),
            "note": "Stage A stratified subsample; Stage B refine uses full train; val full",
        }
        print(
            f"Stage A train subsampled -> {len(y_tr)} (max_train={args.max_train}); "
            f"val remains {len(bundle_full.y_val)}",
            flush=True,
        )
    else:
        bundle_a = bundle_full

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )

    if args.enqueue_baseline and len(study.trials) == 0:
        study.enqueue_trial(BASELINE_PARAMS)
        print("Enqueued baseline trial (multirun-like train HPs)", flush=True)

    completed_before = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining = max(args.n_trials - completed_before, 0)
    print(
        f"study has {completed_before} complete trials; running up to {remaining} more "
        f"(target n_trials={args.n_trials})",
        flush=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        # Baseline enqueued trials already have params fixed
        if trial.number == 0 and trial.user_attrs.get("is_baseline") is None:
            # detect if params match baseline within float tol
            pass
        trial.set_user_attr("params", params)
        save_path = ckpt_dir / f"trial_{trial.number:03d}_seed{args.seed}.pth"
        print("=" * 70, flush=True)
        print(f"TRIAL {trial.number} params={params}", flush=True)
        try:
            result = train_one(
                bundle=bundle_a,
                base_config=base_config,
                init_ckpt=init_ckpt,
                device=device,
                params=params,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                save_path=save_path,
                trial=trial,
            )
        except optuna.TrialPruned:
            print(f"TRIAL {trial.number} PRUNED", flush=True)
            raise
        except Exception as e:
            print(f"TRIAL {trial.number} FAILED: {e}", flush=True)
            raise

        val = result["val"]
        trial.set_user_attr("val_macro_f1", result["best_val_macro_f1"])
        trial.set_user_attr("val_min_per_class_f1", val["min_per_class_f1"])
        trial.set_user_attr("val_balanced_accuracy", val["balanced_accuracy"])
        trial.set_user_attr("val_theft_f1", val.get("theft_f1"))
        trial.set_user_attr("val_normal_f1", val.get("normal_f1"))
        trial.set_user_attr("elapsed_sec", result["elapsed_sec"])
        trial.set_user_attr("save_path", result["save_path"])
        trial.set_user_attr("history", result["history"])

        trial_json = out_dir / f"trial_{trial.number:03d}_seed{args.seed}.json"
        payload = make_result_envelope(
            experiment_id=f"hpo_stage_a_trial{trial.number}",
            protocol_id=bundle_full.protocol_id,
            stage="stage_b_ft",
            seed=args.seed,
            config={
                "study_name": args.study_name,
                "trial_number": trial.number,
                "init_checkpoint": str(init_ckpt),
                "epochs": args.epochs,
                "patience": args.patience,
                "params": params,
                "hpo_stage": "A_explore",
                "max_train": args.max_train,
                "subsample": subsample_note,
            },
            metrics={
                "best_val_macro_f1": result["best_val_macro_f1"],
                "val": val,
                "test": None,
            },
            extra={
                "history": result["history"],
                "elapsed_sec": result["elapsed_sec"],
                "device": str(device),
                "champion_md5": champ_md5,
                "save_path": result["save_path"],
                "n_train": int(len(bundle_a.y_train)),
                "n_val": int(len(bundle_a.y_val)),
            },
            project_root=PROJECT_ROOT,
        )
        with open(trial_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"TRIAL {trial.number} DONE val_macro_f1={result['best_val_macro_f1']:.4f} "
            f"-> {trial_json}",
            flush=True,
        )
        return float(result["best_val_macro_f1"])

    t_study = time.time()
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=args.timeout_sec if args.timeout_sec > 0 else None,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    study_elapsed = time.time() - t_study

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print("ERROR: no complete trials", file=sys.stderr)
        return 1

    best = study.best_trial
    print("=" * 70, flush=True)
    print(
        f"STAGE A best trial={best.number} val_macro_f1={best.value:.4f} params={best.params}",
        flush=True,
    )

    # ---- Stage B: refine top-K with full epochs ----
    refine_rows: list[dict[str, Any]] = []
    if args.refine_top_k > 0:
        ranked = sorted(complete, key=lambda t: t.value if t.value is not None else -1, reverse=True)
        top = ranked[: args.refine_top_k]
        print(
            f"STAGE B refine top-{len(top)} with epochs={args.refine_epochs} "
            f"patience={args.refine_patience}",
            flush=True,
        )
        for rank, t in enumerate(top, start=1):
            params = dict(t.params)
            save_path = ckpt_dir / f"refine_rank{rank}_trial{t.number:03d}_seed{args.seed}.pth"
            print("=" * 70, flush=True)
            print(f"REFINE rank={rank} from trial {t.number} params={params}", flush=True)
            result = train_one(
                bundle=bundle_full,  # always full train for refine
                base_config=base_config,
                init_ckpt=init_ckpt,
                device=device,
                params=params,
                epochs=args.refine_epochs,
                patience=args.refine_patience,
                seed=args.seed,
                save_path=save_path,
                trial=None,
            )
            row = {
                "rank": rank,
                "source_trial": t.number,
                "params": params,
                "stage_a_val_macro_f1": t.value,
                "best_val_macro_f1": result["best_val_macro_f1"],
                "val_min_per_class_f1": result["val"]["min_per_class_f1"],
                "val_balanced_accuracy": result["val"]["balanced_accuracy"],
                "val_theft_f1": result["val"].get("theft_f1"),
                "val_normal_f1": result["val"].get("normal_f1"),
                "elapsed_sec": result["elapsed_sec"],
                "save_path": result["save_path"],
            }
            refine_rows.append(row)
            refine_json = out_dir / f"refine_rank{rank}_trial{t.number:03d}_seed{args.seed}.json"
            payload = make_result_envelope(
                experiment_id=f"hpo_stage_b_refine_rank{rank}",
                protocol_id=bundle_full.protocol_id,
                stage="stage_b_ft",
                seed=args.seed,
                config={
                    "study_name": args.study_name,
                    "source_trial": t.number,
                    "rank": rank,
                    "init_checkpoint": str(init_ckpt),
                    "epochs": args.refine_epochs,
                    "patience": args.refine_patience,
                    "params": params,
                    "hpo_stage": "B_refine",
                },
                metrics={
                    "best_val_macro_f1": result["best_val_macro_f1"],
                    "val": result["val"],
                    "test": None,
                },
                extra={
                    "history": result["history"],
                    "elapsed_sec": result["elapsed_sec"],
                    "device": str(device),
                    "champion_md5": champ_md5,
                    "save_path": result["save_path"],
                    "stage_a_val_macro_f1": t.value,
                },
                project_root=PROJECT_ROOT,
            )
            with open(refine_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(
                f"REFINE rank={rank} DONE val_macro_f1={result['best_val_macro_f1']:.4f}",
                flush=True,
            )

    # Pick final winner: best Stage B refine if any, else Stage A best
    if refine_rows:
        winner_row = max(refine_rows, key=lambda r: r["best_val_macro_f1"])
        winner_source = "stage_b_refine"
        winner_params = winner_row["params"]
        winner_metrics = {
            "best_val_macro_f1": winner_row["best_val_macro_f1"],
            "val": {
                "min_per_class_f1": winner_row["val_min_per_class_f1"],
                "balanced_accuracy": winner_row["val_balanced_accuracy"],
                "theft_f1": winner_row["val_theft_f1"],
                "normal_f1": winner_row["val_normal_f1"],
            },
        }
        winner_trial = winner_row["source_trial"]
        winner_ckpt = winner_row["save_path"]
    else:
        winner_source = "stage_a"
        winner_params = dict(best.params)
        winner_metrics = {
            "best_val_macro_f1": best.value,
            "val": {
                "min_per_class_f1": best.user_attrs.get("val_min_per_class_f1"),
                "balanced_accuracy": best.user_attrs.get("val_balanced_accuracy"),
                "theft_f1": best.user_attrs.get("val_theft_f1"),
                "normal_f1": best.user_attrs.get("val_normal_f1"),
            },
        }
        winner_trial = best.number
        winner_ckpt = best.user_attrs.get("save_path")

    from scripts.protocol.result_schema import git_sha as _git_sha

    sha = _git_sha(PROJECT_ROOT)
    hpo_best_path = PROJECT_ROOT / args.hpo_best_yaml
    write_hpo_best_yaml(
        hpo_best_path,
        params=winner_params,
        metrics=winner_metrics,
        study_name=args.study_name,
        trial_number=int(winner_trial),
        seed=args.seed,
        init_checkpoint=str(init_ckpt),
        git_sha=sha,
    )

    # Summary table
    trials_table = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            trials_table.append(
                {
                    "trial": t.number,
                    "state": str(t.state),
                    "value": t.value,
                    "params": t.params,
                }
            )
            continue
        trials_table.append(
            {
                "trial": t.number,
                "state": "COMPLETE",
                "val_macro_f1": t.value,
                "val_min_per_class_f1": t.user_attrs.get("val_min_per_class_f1"),
                "val_balanced_accuracy": t.user_attrs.get("val_balanced_accuracy"),
                "val_theft_f1": t.user_attrs.get("val_theft_f1"),
                "val_normal_f1": t.user_attrs.get("val_normal_f1"),
                "elapsed_sec": t.user_attrs.get("elapsed_sec"),
                "params": t.params,
                "save_path": t.user_attrs.get("save_path"),
            }
        )

    trials_table_sorted = sorted(
        [r for r in trials_table if r.get("val_macro_f1") is not None],
        key=lambda r: r["val_macro_f1"],
        reverse=True,
    )

    # Decision vs multirun seed42 baseline 0.9780
    baseline_ref = 0.9780405138031053
    win_f1 = float(winner_metrics["best_val_macro_f1"])
    delta = win_f1 - baseline_ref
    if delta > 0.001:
        decision = "INCORPORATE"
        decision_note = (
            f"HPO winner val_macro_f1={win_f1:.4f} beats multirun seed42 baseline "
            f"{baseline_ref:.4f} by Δ={delta:+.4f}; adopt as CAD-CBA-v1 train HPs."
        )
    elif abs(delta) <= 0.001:
        decision = "RUN_DOCUMENTED"
        decision_note = (
            f"HPO winner val_macro_f1={win_f1:.4f} ≈ multirun seed42 baseline "
            f"{baseline_ref:.4f} (Δ={delta:+.4f}); keep prior defaults unless secondary metrics improve."
        )
    else:
        decision = "RUN_DOCUMENTED"
        decision_note = (
            f"HPO winner val_macro_f1={win_f1:.4f} below multirun seed42 baseline "
            f"{baseline_ref:.4f} (Δ={delta:+.4f}); keep baseline train HPs; study retained."
        )

    # Secondary: if macro flat but min-cls improves meaningfully, note for human
    win_min = winner_metrics["val"].get("min_per_class_f1")
    if win_min is not None and win_min > 0.9315 + 0.01 and decision == "RUN_DOCUMENTED":
        decision_note += (
            f" Secondary: min_per_class_f1={win_min:.4f} > baseline 0.9315 — "
            "consider minority-aware incorporation even if macro flat."
        )

    summary = {
        "experiment_id": "hpo_optuna_botiot_wp3",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": args.seed,
        "study_name": args.study_name,
        "storage": storage,
        "init_checkpoint": str(init_ckpt),
        "n_trials_target": args.n_trials,
        "n_trials_complete": len(complete),
        "stage_a_epochs": args.epochs,
        "stage_a_patience": args.patience,
        "stage_a_max_train": args.max_train,
        "stage_a_subsample": subsample_note,
        "refine_top_k": args.refine_top_k,
        "refine_epochs": args.refine_epochs,
        "refine_uses_full_train": True,
        "stage_a_study_elapsed_sec": study_elapsed,
        "champion_md5": champ_md5,
        "architecture": "cnn_bilstm_v3_attention fixed (CAD-CBA-v1)",
        "search_space": {
            "lr": "loguniform 1e-5..3e-3",
            "batch_size": [128, 256, 512, 1024],
            "focal_gamma": "uniform 0.5..3.5",
            "dropout_rate": "uniform 0.10..0.50",
            "attention_dropout": "uniform 0.0..0.30",
            "weight_decay": "loguniform 1e-6..1e-3",
            "scheduler": ["none", "cosine", "step"],
            "not_searched": (
                "cnn_filters, kernel, bilstm dims — fixed for KD init transfer; WP2c if plateau"
            ),
        },
        "baseline_ref_val_macro_f1_seed42": baseline_ref,
        "stage_a_best": {
            "trial": best.number,
            "val_macro_f1": best.value,
            "params": best.params,
            "val_min_per_class_f1": best.user_attrs.get("val_min_per_class_f1"),
            "val_theft_f1": best.user_attrs.get("val_theft_f1"),
        },
        "stage_b_refine": refine_rows,
        "winner": {
            "source": winner_source,
            "trial": winner_trial,
            "params": winner_params,
            "metrics": winner_metrics,
            "checkpoint": winner_ckpt,
            "hpo_best_yaml": str(hpo_best_path),
        },
        "decision": decision,
        "decision_note": decision_note,
        "trials_ranked": trials_table_sorted[:20],
        "all_trials": trials_table,
        "test": "SEALED",
        "note": (
            "Val-only HPO. Production champion not overwritten. "
            "Numbers only from this study JSON + trial JSONs."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Top-10 human table
    top10_path = out_dir / "top10_trials.json"
    with open(top10_path, "w") as f:
        json.dump(trials_table_sorted[:10], f, indent=2)

    print("=" * 70, flush=True)
    print(f"SUMMARY wrote {summary_path}", flush=True)
    print(f"WINNER source={winner_source} trial={winner_trial} "
          f"val_macro_f1={win_f1:.4f} decision={decision}", flush=True)
    print(f"hpo_best.yaml -> {hpo_best_path}", flush=True)
    print(f"decision_note: {decision_note}", flush=True)
    print("test: SEALED | champion not overwritten", flush=True)

    # Post-check champion unchanged
    champ_md5_after = hashlib.md5(CHAMPION_PATH.read_bytes()).hexdigest()
    if champ_md5_after != champ_md5:
        print("FATAL: champion md5 changed during HPO!", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
