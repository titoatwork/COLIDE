#!/usr/bin/env python3
"""
CAD-CBA-v1 package multirun: stage_b_ft from ensemble KD init + WP3 HPO train HPs.

Writes to a dedicated tree so WP1b baseline multirun/ is NOT clobbered:
  model/multirun_ensemble_hpo/ft_seed{seed}.pth
  benchmarks/results/multirun_ensemble_hpo/ft_seed{seed}.json
  benchmarks/results/multirun_ensemble_hpo/summary.json

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_package_ft_multirun.py \\
    --seeds 42,43,44,45,46 --epochs 10 --patience 3
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

DEFAULT_INIT = "model/teachers_kd/kd_ensemble_a0.6_T10.0_g2.0_seed42.pth"
DEFAULT_HPO = "config/hpo_best.yaml"
TAG = "multirun_ensemble_hpo"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--init-checkpoint", type=str, default=DEFAULT_INIT)
    p.add_argument("--hpo-config", type=str, default=DEFAULT_HPO)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--tag", type=str, default=TAG)
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--python", type=str, default=sys.executable)
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    init_ckpt = Path(args.init_checkpoint)
    hpo_cfg = Path(args.hpo_config)
    if not init_ckpt.is_file():
        print(f"ERROR: init checkpoint missing: {init_ckpt}", file=sys.stderr)
        return 1
    if not hpo_cfg.is_file():
        print(f"ERROR: hpo config missing: {hpo_cfg}", file=sys.stderr)
        return 1

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / args.tag
    ckpt_dir = PROJECT_ROOT / "model" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Refuse accidental overwrite of WP1b baseline tree
    if args.tag in ("multirun", "imbalance_loss", "hpo", "teachers_kd"):
        print(
            f"ERROR: tag={args.tag!r} would collide with a locked result tree. "
            f"Use a dedicated tag (default {TAG!r}).",
            file=sys.stderr,
        )
        return 2

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
        if args.allow_test:
            cmd.append("--allow-test")
        print("\n" + "=" * 70)
        print(f"PACKAGE FT MULTI-RUN seed={seed} tag={args.tag}")
        print(" ".join(cmd))
        print("=" * 70, flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
        entry: dict = {
            "seed": seed,
            "returncode": rc,
            "result_path": str(results_path),
            "checkpoint_path": str(save_path),
        }
        if results_path.is_file():
            with open(results_path) as f:
                data = json.load(f)
            m = data.get("metrics", {})
            entry["best_val_macro_f1"] = m.get("best_val_macro_f1")
            val = m.get("val") or {}
            entry["val_macro_f1"] = val.get("macro_f1")
            entry["val_balanced_accuracy"] = val.get("balanced_accuracy")
            entry["val_min_per_class_f1"] = val.get("min_per_class_f1")
            entry["val_theft_f1"] = val.get("theft_f1")
            entry["val_normal_f1"] = val.get("normal_f1")
            entry["elapsed_sec"] = (data.get("extra") or {}).get("elapsed_sec")
            entry["config"] = data.get("config")
            if m.get("test"):
                entry["test_macro_f1"] = m["test"].get("macro_f1")
        runs.append(entry)
        if rc != 0:
            print(f"WARNING: seed {seed} failed rc={rc}", flush=True)

    val_f1s = [r["best_val_macro_f1"] for r in runs if r.get("best_val_macro_f1") is not None]
    min_cls = [r["val_min_per_class_f1"] for r in runs if r.get("val_min_per_class_f1") is not None]
    thefts = [r["val_theft_f1"] for r in runs if r.get("val_theft_f1") is not None]
    t1 = datetime.now(timezone.utc)

    summary = {
        "experiment_id": "package_ft_multirun_ensemble_hpo",
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
        "val_min_per_class_f1_mean": statistics.mean(min_cls) if min_cls else None,
        "val_theft_f1_mean": statistics.mean(thefts) if thefts else None,
        "wall_sec": (t1 - t0).total_seconds(),
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "runs": runs,
        "comparators": {
            "wp1b_multirun_mean_std": "0.9714 ± 0.0109 (old distill init, default HPs)",
            "wp3_hpo_winner_seed42": 0.9791,
            "wp4b_ensemble_kd_stage_a": 0.9401,
        },
        "note": (
            "CAD-CBA-v1 package path: ensemble KD stage_a init + WP3 hpo_best train HPs. "
            "WP1b multirun/ tree intentionally untouched. Production champion not overwritten. "
            "Test sealed unless --allow-test."
        ),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    head = {k: summary[k] for k in summary if k not in ("runs",)}
    print("\nSUMMARY", json.dumps(head, indent=2))
    print("wrote", summary_path)
    return 0 if summary["n_success"] == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
