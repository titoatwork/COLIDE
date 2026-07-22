#!/usr/bin/env python3
"""
WP5b — Protocol-fair neural baselines G6–G12 (val-only; test sealed).

Trains each baseline under botiot_v1 / stage_b_ft with a controlled shared budget
so architecture comparisons are honest vs CAD-CBA-v1 (not historical non-protocol
MLP numbers).

Default suite (seed 42, epochs≤8, patience=3, batch=512, lr=1e-3 Adam, CE, scratch):

  G6  mlp            Multi-layer perceptron
  G7  cnn1d          1D-CNN
  G8  lstm           Unidirectional LSTM
  G9  bilstm         Bidirectional LSTM
  G10 cnn_lstm       CNN + LSTM
  G11 cnn_bilstm     CNN + BiLSTM (no attention) — arch reference
  G12 transformer    Lightweight temporal transformer

Does not clobber production champion or prior multirun / ablation trees.
G15 note: shared fixed HPs (no per-baseline Optuna); document as equal budget.

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_neural_baselines.py
  PYTHONPATH=. .venv/bin/python scripts/run_neural_baselines.py --rows G6,G7,G12
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
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402

CHAMPION_PATH = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"

# Suite definition (stable IDs for tracker G6–G12)
SUITE: list[dict[str, Any]] = [
    {
        "row": "G6",
        "name": "mlp",
        "variant": "mlp",
        "tracker": "G6",
        "description": "Multi-layer perceptron (flat features)",
    },
    {
        "row": "G7",
        "name": "cnn1d",
        "variant": "cnn1d",
        "tracker": "G7",
        "description": "1D-CNN stack + GAP",
    },
    {
        "row": "G8",
        "name": "lstm",
        "variant": "lstm",
        "tracker": "G8",
        "description": "Unidirectional 2-layer LSTM",
    },
    {
        "row": "G9",
        "name": "bilstm",
        "variant": "bilstm",
        "tracker": "G9",
        "description": "Bidirectional 2-layer LSTM",
    },
    {
        "row": "G10",
        "name": "cnn_lstm",
        "variant": "cnn_lstm",
        "tracker": "G10",
        "description": "CNN + unidirectional LSTM",
    },
    {
        "row": "G11",
        "name": "cnn_bilstm",
        "variant": "cnn_bilstm",
        "tracker": "G11",
        "description": "CNN + BiLSTM mean-pool (no attention)",
    },
    {
        "row": "G12",
        "name": "transformer",
        "variant": "transformer",
        "tracker": "G12",
        "description": "Lightweight temporal transformer encoder",
    },
]

G15_HPO_NOTE = (
    "G15 comparable HPO effort: all WP5b neural baselines use the SAME fixed train "
    "budget (lr=1e-3 Adam, batch=512, epochs≤8, patience=3, CE, scratch, seed42). "
    "No per-baseline Optuna. CAD-CBA-v1 separately received WP3 Optuna train-HP search "
    "on fixed V3 arch (hpo_best.yaml) + multi-seed confirms — not applied to these "
    "baselines so the table is architecture/recipe-fair under equal compute, not "
    "HPO-maxed baselines. Classical trees (RF/XGB) use sklearn defaults under same "
    "protocol split (see baselines_classical/). Historical MLP macro_f1≈0.962 is "
    "non-protocol and must not be mixed with these numbers."
)


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


def train_row(
    *,
    row: dict[str, Any],
    bundle,
    base_config: dict,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    save_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    from model.neural_baselines import build_neural_baseline

    set_seed(seed)
    cfg = copy.deepcopy(base_config)
    drop_last = batch_size >= 512
    train_loader = make_loader(
        bundle.X_train, bundle.y_train, batch_size, True, device, drop_last=drop_last
    )
    val_loader = make_loader(
        bundle.X_val, bundle.y_val, batch_size, False, device, drop_last=False
    )

    model = build_neural_baseline(row["variant"], cfg, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_val_f1 = -1.0
    best_val_metrics = None
    patience_left = patience
    history: list[dict] = []
    t0 = time.time()

    print(
        f"\n=== {row['row']} {row['name']} variant={row['variant']} "
        f"loss=ce init=scratch lr={lr} bs={batch_size} ===",
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

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    val_final = eval_split(model, val_loader, device, bundle.class_names)
    n_params = int(
        model.count_parameters()
        if hasattr(model, "count_parameters")
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    sample = torch.from_numpy(np.asarray(bundle.X_val[:256], dtype=np.float32))
    latency = measure_latency_ms(model, sample, device)
    mem: dict[str, float] = {}
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
        experiment_id=f"neural_baseline_{row['row']}_{row['name']}_seed{seed}",
        protocol_id=bundle.protocol_id,
        stage="stage_b_ft",
        seed=seed,
        config={
            "row": row["row"],
            "name": row["name"],
            "variant": row["variant"],
            "tracker": row["tracker"],
            "description": row.get("description"),
            "loss": "ce",
            "init": "scratch",
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": "adam",
            "save_path": str(save_path),
            "hpo_budget_note": "fixed shared HPs; no per-baseline Optuna (G15)",
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
            "comparators_ref": {
                "wp5a_A3_cnn_bilstm_ce": 0.9493,
                "wp5a_A7_full_package": 0.9699,
                "wp1b_multirun_mean": 0.9714,
                "protocol_rf_val": 0.9778,
                "historical_mlp_non_protocol": 0.9624,
            },
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


def _entry_from_existing(row: dict, results_path: Path, save_path: Path) -> dict[str, Any]:
    prev = json.loads(results_path.read_text())
    m = prev.get("metrics") or {}
    return {
        "row": row["row"],
        "name": row["name"],
        "tracker": row["tracker"],
        "best_val_macro_f1": m.get("best_val_macro_f1"),
        "val_min_per_class_f1": (m.get("val") or {}).get("min_per_class_f1"),
        "val_theft_f1": (m.get("val") or {}).get("theft_f1"),
        "val_balanced_accuracy": (m.get("val") or {}).get("balanced_accuracy"),
        "n_params": (m.get("systems") or {}).get("n_params"),
        "latency_mean_ms": ((m.get("systems") or {}).get("latency") or {}).get("mean_ms"),
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--rows",
        type=str,
        default="G6,G7,G8,G9,G10,G11,G12",
        help="Comma-separated suite rows to run",
    )
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--tag", type=str, default="baselines_neural")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse row JSON+ckpt if both exist (thermal resume)",
    )
    args = p.parse_args()

    want = {r.strip().upper() for r in args.rows.split(",") if r.strip()}
    rows = [r for r in SUITE if r["row"] in want]
    if not rows:
        print("ERROR: no matching rows", file=sys.stderr)
        return 1

    if CHAMPION_PATH.is_file():
        pass  # never write champion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = PROJECT_ROOT / args.config
    if not cfg_path.is_file():
        # fallback: repo-root config.yaml
        alt = PROJECT_ROOT / "config.yaml"
        cfg_path = alt if alt.is_file() else cfg_path
    base_config = load_config(cfg_path)
    bundle = load_botiot(stage="stage_b_ft", seed=args.seed)

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / args.tag
    ckpt_dir = PROJECT_ROOT / "model" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    for row in rows:
        save_path = ckpt_dir / f"{row['row']}_{row['name']}_seed{args.seed}.pth"
        results_path = out_dir / f"{row['row']}_{row['name']}_seed{args.seed}.json"
        if args.skip_existing and results_path.is_file() and save_path.is_file():
            try:
                entry = _entry_from_existing(row, results_path, save_path)
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
                lr=args.lr,
                batch_size=args.batch_size,
                save_path=save_path,
                results_path=results_path,
            )
            entry["returncode"] = 0
        except Exception as e:
            print(f"ERROR {row['row']}: {e}", file=sys.stderr, flush=True)
            import traceback

            traceback.print_exc()
            entry = {
                "row": row["row"],
                "name": row["name"],
                "returncode": 1,
                "error": repr(e),
            }
        results.append(entry)

    t1 = datetime.now(timezone.utc)
    ok = [r for r in results if r.get("best_val_macro_f1") is not None]
    ok_sorted = sorted(ok, key=lambda r: r["best_val_macro_f1"], reverse=True)
    summary = {
        "experiment_id": "wp5b_neural_baselines",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "loss": "ce",
        "init": "scratch",
        "optimizer": "adam",
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
        "g15_hpo_budget_note": G15_HPO_NOTE,
        "comparators": {
            "wp5a_A3_cnn_bilstm_ce_seed42": 0.9493,
            "wp5a_A7_full_cad_cba_v1_seed42": 0.9699,
            "wp1b_multirun_mean_std": "0.9714 ± 0.0109",
            "protocol_rf_val": 0.9778,
            "protocol_xgb_val": 0.9762,
            "historical_mlp_non_protocol": 0.9624,
        },
        "note": (
            "Val-only protocol-fair neural baselines G6–G12. Test sealed. "
            "Champion not overwritten. Equal fixed HPs (no per-baseline HPO). "
            "Do not mix with historical ablation_mlp.json / mlp_twostage.json. "
            "G11 is architecture reference under same loop as G6–G10/G12; "
            "WP1b multirun is multi-seed package path with different init/loss."
        ),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nNEURAL BASELINES SUMMARY")
    print(json.dumps(summary["ranking_val_macro_f1"], indent=2))
    print("wrote", summary_path)
    return 0 if summary["n_success"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
