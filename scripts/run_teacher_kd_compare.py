#!/usr/bin/env python3
"""
WP4b orchestrator: teacher/KD compare under botiot_v1 stage_a_kd.

Runs train_protocol_kd.py for each teacher, writes per-run JSON + summary.
Skip-nothing: every teacher gets a row (INCORPORATED or RUN_DOCUMENTED later).

Default recipe matches historical CAD-CBA KD: α=0.6, T=10, focal γ=2, seed=42.
Val-only selection; test sealed.

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_teacher_kd_compare.py \\
    --epochs 12 --patience 4 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHERS = ["none", "rf", "xgb", "lgbm", "ensemble"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--teachers", type=str, default=",".join(DEFAULT_TEACHERS))
    p.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Pass through to train_protocol_kd (0=full stage_a_kd train)",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip teacher if results JSON already has best_val_macro_f1",
    )
    args = p.parse_args()

    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "teachers_kd"
    ckpt_dir = PROJECT_ROOT / "model" / "teachers_kd"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    rows = []
    t_all = time.time()

    for teacher in teachers:
        save = ckpt_dir / (
            f"kd_{teacher}_a{args.alpha}_T{args.temperature}_g{args.focal_gamma}_seed{args.seed}.pth"
        )
        res = out_dir / f"kd_{teacher}_seed{args.seed}.json"

        if args.skip_existing and res.is_file():
            try:
                with open(res) as f:
                    prev = json.load(f)
                if "best_val_macro_f1" in prev.get("metrics", {}):
                    print(f"SKIP existing {res}", flush=True)
                    row = {
                        "teacher": teacher,
                        "returncode": 0,
                        "results_path": str(res),
                        "skipped_existing": True,
                        "best_val_macro_f1": prev["metrics"]["best_val_macro_f1"],
                        "val_min_per_class_f1": prev["metrics"]["val"]["min_per_class_f1"],
                        "val_theft_f1": prev["metrics"]["val"].get("theft_f1"),
                        "teacher_val_macro_f1": (prev["metrics"].get("teacher") or {}).get(
                            "macro_f1"
                        ),
                    }
                    rows.append(row)
                    continue
            except Exception:
                pass

        cmd = [
            args.python,
            "-u",
            str(PROJECT_ROOT / "scripts" / "train_protocol_kd.py"),
            "--teacher",
            teacher,
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
            "--alpha",
            str(args.alpha),
            "--temperature",
            str(args.temperature),
            "--focal-gamma",
            str(args.focal_gamma),
            "--save-path",
            str(save),
            "--results-path",
            str(res),
            "--max-train",
            str(args.max_train),
        ]
        print("=" * 60, f"teacher={teacher}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
        row: dict = {
            "teacher": teacher,
            "returncode": rc,
            "results_path": str(res),
            "elapsed_sec": time.time() - t0,
            "save_path": str(save),
        }
        if res.is_file():
            with open(res) as f:
                data = json.load(f)
            m = data["metrics"]
            row["best_val_macro_f1"] = m["best_val_macro_f1"]
            row["val_min_per_class_f1"] = m["val"]["min_per_class_f1"]
            row["val_balanced_accuracy"] = m["val"]["balanced_accuracy"]
            row["val_theft_f1"] = m["val"].get("theft_f1")
            row["val_normal_f1"] = m["val"].get("normal_f1")
            if m.get("teacher"):
                row["teacher_val_macro_f1"] = m["teacher"].get("macro_f1")
                row["teacher_val_theft_f1"] = m["teacher"].get("theft_f1")
        rows.append(row)
        # progressive summary so handoff survives partial runs
        _write_summary(out_dir, args, teachers, rows, t_all)

    summary = _write_summary(out_dir, args, teachers, rows, t_all)
    print("wrote", out_dir / "summary.json")
    print("best", summary.get("best_by_student_val_macro_f1"))
    return 0 if all(r["returncode"] == 0 for r in rows) else 1


def _write_summary(out_dir, args, teachers, rows, t_all):
    ranked = sorted(
        [r for r in rows if "best_val_macro_f1" in r],
        key=lambda r: r["best_val_macro_f1"],
        reverse=True,
    )
    summary = {
        "experiment_id": "teacher_kd_compare",
        "protocol_id": "botiot_v1",
        "stage": "stage_a_kd",
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "focal_gamma": args.focal_gamma,
        "max_train": args.max_train,
        "teachers_requested": teachers,
        "rows": rows,
        "best_by_student_val_macro_f1": ranked[0] if ranked else None,
        "ranking_student_val_macro_f1": [
            {
                "teacher": r["teacher"],
                "best_val_macro_f1": r["best_val_macro_f1"],
                "val_min_per_class_f1": r.get("val_min_per_class_f1"),
                "val_theft_f1": r.get("val_theft_f1"),
            }
            for r in ranked
        ],
        "wall_sec": time.time() - t_all,
        "note": (
            "Val-only student selection; test sealed. "
            "Compare student val macro-F1 across teachers under fixed α/T/γ. "
            "Incorporate best teacher into CAD-CBA-v1 or RUN_DOCUMENTED others."
        ),
    }
    sp = out_dir / "summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
