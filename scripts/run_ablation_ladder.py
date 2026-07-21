#!/usr/bin/env python3
"""
WP5a — Protocol ablation ladder for CAD-CBA-v1 (val-only; test sealed).

Trains each ladder row under botiot_v1 / stage_b_ft with a controlled budget.
Does not clobber production champion or prior multirun trees.

Default ladder (seed 42, epochs≤8, patience=3, batch=512, lr=1e-3 Adam unless noted):

  A1 cnn_only          CE    scratch
  A2 bilstm_only       CE    scratch
  A3 cnn_bilstm        CE    scratch
  A4 cnn_bilstm_attn   CE    scratch
  A5 cnn_bilstm_attn   focal scratch          (+ imbalance)
  A6 cnn_bilstm_attn   focal ensemble KD init (+ KD)
  A7 full package      focal ensemble KD + hpo_best train HPs

Systems metrics logged: param count, state_dict bytes, forward latency (CUDA).

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_ablation_ladder.py
  PYTHONPATH=. .venv/bin/python scripts/run_ablation_ladder.py --rows A1,A3,A7
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.losses import FocalLoss  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402

CHAMPION_PATH = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
ENSEMBLE_KD = (
    PROJECT_ROOT
    / "model"
    / "teachers_kd"
    / "kd_ensemble_a0.6_T10.0_g2.0_seed42.pth"
)
HPO_YAML = PROJECT_ROOT / "config" / "hpo_best.yaml"

# Ladder definition (stable IDs for tracker F1–F7)
LADDER: list[dict[str, Any]] = [
    {
        "row": "A1",
        "name": "cnn_only",
        "variant": "cnn_only",
        "loss": "ce",
        "init": "scratch",
        "use_hpo": False,
        "tracker": "F1",
    },
    {
        "row": "A2",
        "name": "bilstm_only",
        "variant": "bilstm_only",
        "loss": "ce",
        "init": "scratch",
        "use_hpo": False,
        "tracker": "F2",
    },
    {
        "row": "A3",
        "name": "cnn_bilstm",
        "variant": "cnn_bilstm",
        "loss": "ce",
        "init": "scratch",
        "use_hpo": False,
        "tracker": "F3",
    },
    {
        "row": "A4",
        "name": "cnn_bilstm_attn_ce",
        "variant": "cnn_bilstm_attn",
        "loss": "ce",
        "init": "scratch",
        "use_hpo": False,
        "tracker": "F6",  # attention backbone without package extras
    },
    {
        "row": "A5",
        "name": "cnn_bilstm_attn_focal",
        "variant": "cnn_bilstm_attn",
        "loss": "focal",
        "init": "scratch",
        "use_hpo": False,
        "tracker": "F5",
    },
    {
        "row": "A6",
        "name": "attn_focal_ensemble_kd",
        "variant": "cnn_bilstm_attn",
        "loss": "focal",
        "init": "ensemble_kd",
        "use_hpo": False,
        "tracker": "F4",
    },
    {
        "row": "A7",
        "name": "full_cad_cba_v1",
        "variant": "cnn_bilstm_attn",
        "loss": "focal",
        "init": "ensemble_kd",
        "use_hpo": True,
        "tracker": "F7",
    },
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(X, y, batch_size: int, shuffle: bool, device: torch.device, drop_last: bool = False):
    pin = device.type == "cuda"
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
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
    return compute_classification_metrics(
        np.concatenate(targets), np.concatenate(preds), class_names
    )


@torch.no_grad()
def measure_latency_ms(
    model, sample: torch.Tensor, device: torch.device, warmup: int = 20, reps: int = 50
) -> dict[str, float]:
    model.eval()
    xb = sample.to(device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        for _ in range(warmup):
            _ = model(xb)
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = model(xb)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    else:
        for _ in range(warmup):
            _ = model(xb)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = model(xb)
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "batch_size": int(xb.size(0)),
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        "p50_ms": float(statistics.median(times)),
        "per_sample_us": float(statistics.mean(times) * 1000.0 / xb.size(0)),
    }


def load_hpo_params(path: Path) -> dict:
    with open(path) as f:
        doc = yaml.safe_load(f)
    return dict(doc["hpo"]["best_params"])


def train_row(
    *,
    row: dict[str, Any],
    bundle,
    base_config: dict,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    default_lr: float,
    default_batch: int,
    default_gamma: float,
    save_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    from model.ablation_variants import build_ablation_model

    set_seed(seed)
    cfg = copy.deepcopy(base_config)
    lr = default_lr
    batch_size = default_batch
    gamma = default_gamma
    weight_decay = 0.0
    scheduler_name = "none"
    optim_name = "adam"
    dropout = cfg["model"]["dropout_rate"]
    att_drop = cfg["model"].get("attention_dropout", 0.1)

    if row["use_hpo"]:
        hp = load_hpo_params(HPO_YAML)
        lr = float(hp["lr"])
        batch_size = int(hp["batch_size"])
        gamma = float(hp["focal_gamma"])
        dropout = float(hp["dropout_rate"])
        att_drop = float(hp["attention_dropout"])
        weight_decay = float(hp["weight_decay"])
        scheduler_name = str(hp["scheduler"])
        optim_name = "adamw"
        cfg["model"]["dropout_rate"] = dropout
        cfg["model"]["attention_dropout"] = att_drop

    drop_last = batch_size >= 512
    train_loader = make_loader(
        bundle.X_train, bundle.y_train, batch_size, True, device, drop_last=drop_last
    )
    val_loader = make_loader(
        bundle.X_val, bundle.y_val, batch_size, False, device, drop_last=False
    )

    model = build_ablation_model(row["variant"], cfg, device)
    init_path = None
    if row["init"] == "ensemble_kd":
        if row["variant"] != "cnn_bilstm_attn":
            raise ValueError("ensemble_kd init only valid for cnn_bilstm_attn")
        if not ENSEMBLE_KD.is_file():
            raise FileNotFoundError(ENSEMBLE_KD)
        state = torch.load(ENSEMBLE_KD, map_location=device, weights_only=True)
        model.load_state_dict(state)
        init_path = str(ENSEMBLE_KD)

    if row["loss"] == "focal":
        criterion = FocalLoss(gamma=gamma)
    elif row["loss"] == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(row["loss"])

    if optim_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr)

    scheduler = None
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    best_val_f1 = -1.0
    best_val_metrics = None
    patience_left = patience
    history: list[dict] = []
    t0 = time.time()

    print(
        f"\n=== {row['row']} {row['name']} variant={row['variant']} loss={row['loss']} "
        f"init={row['init']} hpo={row['use_hpo']} lr={lr} bs={batch_size} ===",
        flush=True,
    )

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
        row_h = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_macro_f1": val_m["macro_f1"],
            "val_min_per_class_f1": val_m["min_per_class_f1"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_theft_f1": val_m.get("theft_f1"),
        }
        history.append(row_h)
        print(
            f"  {row['row']} ep{epoch:02d} loss={row_h['train_loss']:.4f} "
            f"val_macro_f1={val_m['macro_f1']:.4f} min={val_m['min_per_class_f1']:.4f} "
            f"theft={val_m.get('theft_f1', float('nan')):.4f}",
            flush=True,
        )
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_m["macro_f1"])
            best_val_metrics = val_m
            patience_left = patience
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    # reload best
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    val_final = eval_split(model, val_loader, device, bundle.class_names)
    n_params = int(
        model.count_parameters()
        if hasattr(model, "count_parameters")
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    # latency on a fixed val batch
    sample = torch.from_numpy(np.asarray(bundle.X_val[:256], dtype=np.float32))
    latency = measure_latency_ms(model, sample, device)
    # peak cuda mem during a forward (approx)
    mem = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = model(sample.to(device))
        torch.cuda.synchronize()
        mem = {
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }

    elapsed = time.time() - t0
    ckpt_bytes = save_path.stat().st_size if save_path.is_file() else None
    payload = make_result_envelope(
        experiment_id=f"ablation_{row['row']}_{row['name']}_seed{seed}",
        protocol_id=bundle.protocol_id,
        stage="stage_b_ft",
        seed=seed,
        config={
            "row": row["row"],
            "name": row["name"],
            "variant": row["variant"],
            "loss": row["loss"],
            "init": row["init"],
            "init_checkpoint": init_path,
            "use_hpo": row["use_hpo"],
            "tracker": row["tracker"],
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "batch_size": batch_size,
            "focal_gamma": gamma if row["loss"] == "focal" else None,
            "dropout_rate": dropout,
            "attention_dropout": att_drop,
            "weight_decay": weight_decay,
            "scheduler": scheduler_name,
            "optimizer": optim_name,
            "save_path": str(save_path),
        },
        metrics={
            "best_val_macro_f1": float(best_val_f1),
            "val": val_final,
            "systems": {
                "n_params": n_params,
                "checkpoint_bytes": ckpt_bytes,
                "latency": latency,
                "cuda_memory": mem,
            },
        },
        extra={
            "history": history,
            "elapsed_sec": elapsed,
            "allow_test": False,
            "device": str(device),
            "best_val_snapshot": best_val_metrics,
            "data_summary": bundle.summary(),
        },
        project_root=PROJECT_ROOT,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"  DONE {row['row']} best_val_macro_f1={best_val_f1:.4f} "
        f"params={n_params} lat_ms={latency['mean_ms']:.3f} -> {results_path}",
        flush=True,
    )
    return {
        "row": row["row"],
        "name": row["name"],
        "tracker": row["tracker"],
        "best_val_macro_f1": float(best_val_f1),
        "val_min_per_class_f1": val_final.get("min_per_class_f1"),
        "val_theft_f1": val_final.get("theft_f1"),
        "val_balanced_accuracy": val_final.get("balanced_accuracy"),
        "n_params": n_params,
        "latency_mean_ms": latency["mean_ms"],
        "per_sample_us": latency["per_sample_us"],
        "checkpoint_bytes": ckpt_bytes,
        "elapsed_sec": elapsed,
        "results_path": str(results_path),
        "checkpoint_path": str(save_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument(
        "--rows",
        type=str,
        default="A1,A2,A3,A4,A5,A6,A7",
        help="Comma-separated ladder rows to run",
    )
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--tag", type=str, default="ablations")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse row JSON+ckpt if both exist (thermal resume / partial re-run)",
    )
    args = p.parse_args()

    want = {r.strip().upper() for r in args.rows.split(",") if r.strip()}
    rows = [r for r in LADDER if r["row"] in want]
    if not rows:
        print("ERROR: no matching rows", file=sys.stderr)
        return 1

    if CHAMPION_PATH.is_file():
        # touch-proof: we never write here
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = load_config(PROJECT_ROOT / args.config)
    bundle = load_botiot(stage="stage_b_ft", seed=args.seed)

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / args.tag
    ckpt_dir = PROJECT_ROOT / "model" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = datetime.now(timezone.utc)
    results = []
    for row in rows:
        save_path = ckpt_dir / f"{row['row']}_{row['name']}_seed{args.seed}.pth"
        results_path = out_dir / f"{row['row']}_{row['name']}_seed{args.seed}.json"
        if (
            args.skip_existing
            and results_path.is_file()
            and save_path.is_file()
        ):
            try:
                prev = json.loads(results_path.read_text())
                m = prev.get("metrics") or {}
                entry = {
                    "row": row["row"],
                    "name": row["name"],
                    "tracker": row["tracker"],
                    "best_val_macro_f1": m.get("best_val_macro_f1"),
                    "val_min_per_class_f1": (m.get("val") or {}).get("min_per_class_f1"),
                    "val_theft_f1": (m.get("val") or {}).get("theft_f1"),
                    "val_balanced_accuracy": (m.get("val") or {}).get(
                        "balanced_accuracy"
                    ),
                    "n_params": (m.get("systems") or {}).get("n_params"),
                    "latency_mean_ms": ((m.get("systems") or {}).get("latency") or {}).get(
                        "mean_ms"
                    ),
                    "per_sample_us": ((m.get("systems") or {}).get("latency") or {}).get(
                        "per_sample_us"
                    ),
                    "checkpoint_bytes": (m.get("systems") or {}).get("checkpoint_bytes"),
                    "elapsed_sec": (prev.get("extra") or {}).get("elapsed_sec"),
                    "results_path": str(results_path),
                    "checkpoint_path": str(save_path),
                    "returncode": 0,
                    "skipped_existing": True,
                }
                print(
                    f"SKIP existing {row['row']} best_val_macro_f1="
                    f"{entry.get('best_val_macro_f1')}",
                    flush=True,
                )
                results.append(entry)
                continue
            except Exception as e:
                print(f"WARN skip-existing failed for {row['row']}: {e}", flush=True)
        try:
            entry = train_row(
                row=row,
                bundle=bundle,
                base_config=base_config,
                device=device,
                seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
                default_lr=args.lr,
                default_batch=args.batch_size,
                default_gamma=args.focal_gamma,
                save_path=save_path,
                results_path=results_path,
            )
            entry["returncode"] = 0
        except Exception as e:
            print(f"ERROR {row['row']}: {e}", file=sys.stderr, flush=True)
            entry = {
                "row": row["row"],
                "name": row["name"],
                "returncode": 1,
                "error": repr(e),
            }
        results.append(entry)

    t1 = datetime.now(timezone.utc)
    # ranking
    ok = [r for r in results if r.get("best_val_macro_f1") is not None]
    ok_sorted = sorted(ok, key=lambda r: r["best_val_macro_f1"], reverse=True)
    summary = {
        "experiment_id": "wp5a_ablation_ladder",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "default_lr": args.lr,
        "default_batch_size": args.batch_size,
        "tag": args.tag,
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "wall_sec": (t1 - t0).total_seconds(),
        "rows_requested": [r["row"] for r in rows],
        "n_success": len(ok),
        "ranking_val_macro_f1": [
            {
                "row": r["row"],
                "name": r["name"],
                "best_val_macro_f1": r["best_val_macro_f1"],
                "val_min_per_class_f1": r.get("val_min_per_class_f1"),
                "val_theft_f1": r.get("val_theft_f1"),
                "n_params": r.get("n_params"),
                "per_sample_us": r.get("per_sample_us"),
            }
            for r in ok_sorted
        ],
        "results": results,
        "note": (
            "Val-only ablation ladder. Test sealed. Champion not overwritten. "
            "A7 = full CAD-CBA-v1 (ensemble KD + HPO HPs). "
            "Systems latency measured on batch=256 val slice, CUDA sync."
        ),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nABLATION SUMMARY", json.dumps(summary["ranking_val_macro_f1"], indent=2))
    print("wrote", summary_path)
    return 0 if summary["n_success"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
