#!/usr/bin/env python3
"""
Phase 4 partial: compare train losses under protocol FT (val-only selection).

Runs train_protocol_ft for each loss with distinct save/results paths.
Skip-nothing: each loss gets JSON even if not selected later.

Example (after multirun GPU free):
  PYTHONPATH=. .venv/bin/python scripts/run_imbalance_loss_compare.py --epochs 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSSES = ["ce", "focal", "focal_cb", "logit_adj"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--init-checkpoint",
        default="model/best_model_botiot_distill_a0.6_T10.0_focal2.pth",
    )
    p.add_argument("--losses", type=str, default=",".join(LOSSES))
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    losses = [x.strip() for x in args.losses.split(",") if x.strip()]
    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "imbalance_loss"
    ckpt_dir = PROJECT_ROOT / "model" / "imbalance_loss"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    for loss in losses:
        save = ckpt_dir / f"ft_{loss}_seed{args.seed}.pth"
        res = out_dir / f"ft_{loss}_seed{args.seed}.json"
        cmd = [
            args.python,
            str(PROJECT_ROOT / "scripts" / "train_protocol_ft.py"),
            "--init-checkpoint",
            args.init_checkpoint,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--loss",
            loss,
            "--save-path",
            str(save),
            "--results-path",
            str(res),
        ]
        print("=" * 60, f"loss={loss}", flush=True)
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
        row = {"loss": loss, "returncode": rc, "results_path": str(res)}
        if res.is_file():
            with open(res) as f:
                data = json.load(f)
            row["best_val_macro_f1"] = data["metrics"]["best_val_macro_f1"]
            row["val_min_per_class_f1"] = data["metrics"]["val"]["min_per_class_f1"]
            row["val_balanced_accuracy"] = data["metrics"]["val"]["balanced_accuracy"]
        rows.append(row)

    ranked = sorted(
        [r for r in rows if "best_val_macro_f1" in r],
        key=lambda r: r["best_val_macro_f1"],
        reverse=True,
    )
    summary = {
        "experiment_id": "imbalance_loss_compare",
        "seed": args.seed,
        "epochs": args.epochs,
        "rows": rows,
        "best_by_val_macro_f1": ranked[0] if ranked else None,
        "note": "Val-only; test sealed. Incorporate best or RUN_DOCUMENTED others.",
    }
    sp = out_dir / "summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", sp)
    print("best", summary["best_by_val_macro_f1"])
    return 0 if all(r["returncode"] == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
