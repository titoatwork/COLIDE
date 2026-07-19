#!/usr/bin/env python3
"""
WP1b: multi-seed fine-tune baseline under canonical protocol (stage_b_ft).

Runs train_protocol_ft.py for each seed. Does not overwrite the production champion.
Aggregates val metrics into benchmarks/results/multirun/summary.json.

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_baseline_multirun.py --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default="model/best_model_botiot_distill_a0.6_T10.0_focal2.pth",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--python", type=str, default=sys.executable)
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "multirun"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in seeds:
        cmd = [
            args.python,
            str(PROJECT_ROOT / "scripts" / "train_protocol_ft.py"),
            "--init-checkpoint",
            args.init_checkpoint,
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--batch-size",
            str(args.batch_size),
            "--focal-gamma",
            str(args.focal_gamma),
            "--patience",
            str(args.patience),
        ]
        if args.allow_test:
            cmd.append("--allow-test")
        print("\n" + "=" * 70)
        print("MULTI-RUN seed", seed)
        print(" ".join(cmd))
        print("=" * 70, flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
        result_path = out_dir / f"ft_seed{seed}.json"
        entry = {"seed": seed, "returncode": rc, "result_path": str(result_path)}
        if result_path.is_file():
            with open(result_path) as f:
                data = json.load(f)
            entry["best_val_macro_f1"] = data["metrics"]["best_val_macro_f1"]
            entry["val_macro_f1"] = data["metrics"]["val"]["macro_f1"]
            entry["val_balanced_accuracy"] = data["metrics"]["val"]["balanced_accuracy"]
            entry["val_min_per_class_f1"] = data["metrics"]["val"]["min_per_class_f1"]
            if data["metrics"].get("test"):
                entry["test_macro_f1"] = data["metrics"]["test"]["macro_f1"]
        runs.append(entry)
        if rc != 0:
            print(f"WARNING: seed {seed} failed rc={rc}", flush=True)

    val_f1s = [r["best_val_macro_f1"] for r in runs if "best_val_macro_f1" in r]
    summary = {
        "experiment_id": "baseline_multirun_ft",
        "init_checkpoint": args.init_checkpoint,
        "seeds": seeds,
        "n_success": len(val_f1s),
        "val_macro_f1_mean": statistics.mean(val_f1s) if val_f1s else None,
        "val_macro_f1_std": statistics.stdev(val_f1s) if len(val_f1s) > 1 else 0.0,
        "val_macro_f1_min": min(val_f1s) if val_f1s else None,
        "val_macro_f1_max": max(val_f1s) if val_f1s else None,
        "runs": runs,
        "note": "Production champion not overwritten. Test sealed unless --allow-test.",
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSUMMARY", json.dumps({k: summary[k] for k in summary if k != "runs"}, indent=2))
    print("wrote", summary_path)
    return 0 if summary["n_success"] == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
