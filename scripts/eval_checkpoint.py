#!/usr/bin/env python3
"""
Evaluate a checkpoint under the canonical BoT-IoT protocol.

Default: validation metrics only (safe for selection / HPO).
Test metrics: only with --allow-test (sealed evaluation after config freeze).

Example:
  PYTHONPATH=. .venv/bin/python scripts/eval_checkpoint.py \\
    --checkpoint model/best_model_botiot_twostage.pth --stage stage_b_ft

  PYTHONPATH=. .venv/bin/python scripts/eval_checkpoint.py \\
    --checkpoint model/best_model_botiot_twostage.pth --stage stage_b_ft --allow-test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@torch.no_grad()
def predict(model, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    for (xb,) in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--stage",
        choices=["stage_a_kd", "stage_b_ft"],
        default="stage_b_ft",
        help="Data protocol stage (must match how the model was trained)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-test", action="store_true", help="Also evaluate on sealed test set")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: benchmarks/results/protocol/eval_<stem>.json)",
    )
    p.add_argument("--config", type=str, default="config/config.yaml")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_botiot(stage=args.stage, seed=args.seed)
    config = load_config(PROJECT_ROOT / args.config)

    sys.path.insert(0, str(PROJECT_ROOT / "model"))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    model = CNNBiLSTMAttention(config).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    val_pred = predict(model, bundle.X_val, device, args.batch_size)
    val_metrics = compute_classification_metrics(
        bundle.y_val, val_pred, bundle.class_names
    )

    payload = {
        "protocol_id": bundle.protocol_id,
        "stage": args.stage,
        "seed": args.seed,
        "checkpoint": str(ckpt),
        "git_sha": _git_sha(),
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "allow_test": bool(args.allow_test),
        "data_summary": bundle.summary(),
        "val": val_metrics,
        "test": None,
        "note": (
            "Test metrics included only because --allow-test was set. "
            "Do not use test for model selection."
            if args.allow_test
            else "Test sealed: not evaluated (pass --allow-test for final sealed eval)."
        ),
    }

    if args.allow_test:
        test_pred = predict(model, bundle.X_test, device, args.batch_size)
        payload["test"] = compute_classification_metrics(
            bundle.y_test, test_pred, bundle.class_names
        )

    out = Path(args.out) if args.out else (
        PROJECT_ROOT
        / "benchmarks"
        / "results"
        / "protocol"
        / f"eval_{ckpt.stem}_{args.stage}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"checkpoint: {ckpt}")
    print(f"stage:      {args.stage}  seed={args.seed}  device={device}")
    print(f"val macro-F1:           {val_metrics['macro_f1']:.4f}")
    print(f"val balanced accuracy:  {val_metrics['balanced_accuracy']:.4f}")
    print(f"val min per-class F1:   {val_metrics['min_per_class_f1']:.4f}")
    if args.allow_test and payload["test"]:
        print(f"test macro-F1:          {payload['test']['macro_f1']:.4f}")
        print(f"test balanced accuracy: {payload['test']['balanced_accuracy']:.4f}")
    else:
        print("test: SEALED (not evaluated)")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
