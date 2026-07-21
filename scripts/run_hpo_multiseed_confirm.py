#!/usr/bin/env python3
"""
Multi-seed confirm of WP3 HPO winner train HPs (n≥5, val-only, test sealed).

Init matches the HPO study (historical distill), NOT ensemble KD:
  model/best_model_botiot_distill_a0.6_T10.0_focal2.pth
HPs from config/hpo_best.yaml.

Writes to multirun_hpo_confirm/ (does not clobber WP1b multirun or package multirun).

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_hpo_multiseed_confirm.py --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INIT = "model/best_model_botiot_distill_a0.6_T10.0_focal2.pth"
DEFAULT_HPO = "config/hpo_best.yaml"
TAG = "multirun_hpo_confirm"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--init-checkpoint", type=str, default=DEFAULT_INIT)
    p.add_argument("--hpo-config", type=str, default=DEFAULT_HPO)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--tag", type=str, default=TAG)
    p.add_argument("--python", type=str, default=sys.executable)
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    init_ckpt = Path(args.init_checkpoint)
    hpo_cfg = Path(args.hpo_config)
    if not init_ckpt.is_file():
        print(f"ERROR: missing init {init_ckpt}", file=sys.stderr)
        return 1
    if not hpo_cfg.is_file():
        print(f"ERROR: missing hpo {hpo_cfg}", file=sys.stderr)
        return 1
    if args.tag in ("multirun", "imbalance_loss", "hpo", "teachers_kd", "multirun_ensemble_hpo"):
        print(f"ERROR: refusing locked tag {args.tag!r}", file=sys.stderr)
        return 2

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / args.tag
    ckpt_dir = PROJECT_ROOT / "model" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    t0 = datetime.now(timezone.utc)
    for seed in seeds:
        save_path = ckpt_dir / f"ft_seed{seed}.pth"
        results_path = out_dir / f"ft_seed{seed}.json"
        cmd = [
            args.python,
            str(PROJECT_ROOT / "scripts" / "train_protocol_ft.py"),
            "--init-checkpoint",
            str(init_ckpt),
            "--hpo-config",
            str(hpo_cfg),
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--save-path",
            str(save_path),
            "--results-path",
            str(results_path),
        ]
        print("\n" + "=" * 70)
        print(f"HPO MULTI-SEED CONFIRM seed={seed}")
        print(" ".join(cmd))
        print("=" * 70, flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
        entry: dict = {"seed": seed, "returncode": rc, "result_path": str(results_path)}
        if results_path.is_file():
            with open(results_path) as f:
                data = json.load(f)
            m = data["metrics"]
            entry["best_val_macro_f1"] = m["best_val_macro_f1"]
            entry["val_macro_f1"] = m["val"]["macro_f1"]
            entry["val_min_per_class_f1"] = m["val"]["min_per_class_f1"]
            entry["val_theft_f1"] = m["val"].get("theft_f1")
            entry["val_balanced_accuracy"] = m["val"]["balanced_accuracy"]
            entry["elapsed_sec"] = data.get("extra", {}).get("elapsed_sec")
        runs.append(entry)

    val_f1s = [r["best_val_macro_f1"] for r in runs if "best_val_macro_f1" in r]
    t1 = datetime.now(timezone.utc)
    summary = {
        "experiment_id": "hpo_winner_multiseed_confirm",
        "tag": args.tag,
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "init_checkpoint": str(init_ckpt),
        "hpo_config": str(hpo_cfg),
        "epochs": args.epochs,
        "patience": args.patience,
        "seeds": seeds,
        "n_success": len(val_f1s),
        "val_macro_f1_mean": statistics.mean(val_f1s) if val_f1s else None,
        "val_macro_f1_std": statistics.stdev(val_f1s) if len(val_f1s) > 1 else 0.0,
        "val_macro_f1_min": min(val_f1s) if val_f1s else None,
        "val_macro_f1_max": max(val_f1s) if val_f1s else None,
        "wall_sec": (t1 - t0).total_seconds(),
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "runs": runs,
        "comparators": {
            "wp3_hpo_winner_seed42_full_refine": 0.9791,
            "wp1b_multirun_default_hps_mean_std": "0.9714 ± 0.0109",
        },
        "note": (
            "Multi-seed confirm of HPO train HPs with the same init as the Optuna study. "
            "Test sealed. Champion not overwritten. Distinct from package multirun "
            "(ensemble KD init)."
        ),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSUMMARY", json.dumps({k: summary[k] for k in summary if k != "runs"}, indent=2))
    print("wrote", summary_path)
    return 0 if summary["n_success"] == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
