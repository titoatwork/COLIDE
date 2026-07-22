#!/usr/bin/env python3
"""
D6 — Stratified (class-balanced) batch sampling vs shuffle under protocol FT.

Compares train_sampler=shuffle vs train_sampler=stratified on the same
init / HPs / seed / val-only protocol. Does not clobber champion or multirun trees.

Stratified = WeightedRandomSampler with inverse class-frequency weights
(each class contributes ~equal expected samples per epoch).

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_stratified_batch_compare.py \\
    --epochs 8 --patience 3 --hpo-config config/hpo_best.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLERS = ["shuffle", "stratified"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument(
        "--init-checkpoint",
        default="model/best_model_botiot_distill_a0.6_T10.0_focal2.pth",
        help="Same init as WP1b/WP3 for fair FT compare",
    )
    p.add_argument(
        "--hpo-config",
        default="config/hpo_best.yaml",
        help="Empty string to use default FT HPs instead of HPO winner",
    )
    p.add_argument("--loss", default="focal")
    p.add_argument(
        "--samplers",
        default=",".join(SAMPLERS),
        help="Comma list: shuffle,stratified",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a sampler if its results JSON already has best_val_macro_f1",
    )
    args = p.parse_args()

    samplers = [x.strip() for x in args.samplers.split(",") if x.strip()]
    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "stratified_batch"
    ckpt_dir = PROJECT_ROOT / "model" / "stratified_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    rows = []

    for samp in samplers:
        save = ckpt_dir / f"ft_{samp}_seed{args.seed}.pth"
        res = out_dir / f"ft_{samp}_seed{args.seed}.json"
        if args.skip_existing and res.is_file():
            with open(res) as f:
                data = json.load(f)
            m = (data.get("metrics") or {}).get("best_val_macro_f1")
            if m is not None:
                print(f"SKIP existing {samp} best_val_macro_f1={m}", flush=True)
                rows.append(
                    {
                        "sampler": samp,
                        "returncode": 0,
                        "skipped": True,
                        "results_path": str(res),
                        "best_val_macro_f1": m,
                        "val_min_per_class_f1": (data.get("metrics") or {})
                        .get("val", {})
                        .get("min_per_class_f1"),
                        "val_theft_f1": (data.get("metrics") or {})
                        .get("val", {})
                        .get("theft_f1")
                        or (data.get("metrics") or {})
                        .get("val", {})
                        .get("per_class", {})
                        .get("Theft", {})
                        .get("f1"),
                    }
                )
                continue

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
            "--loss",
            args.loss,
            "--train-sampler",
            samp,
            "--save-path",
            str(save),
            "--results-path",
            str(res),
        ]
        if args.hpo_config:
            cmd.extend(["--hpo-config", args.hpo_config])

        print("=" * 60, f"sampler={samp}", flush=True)
        print("CMD:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
        row: dict = {
            "sampler": samp,
            "returncode": rc,
            "skipped": False,
            "results_path": str(res),
            "checkpoint": str(save),
        }
        if res.is_file():
            with open(res) as f:
                data = json.load(f)
            metrics = data.get("metrics") or {}
            val = metrics.get("val") or {}
            row["best_val_macro_f1"] = metrics.get("best_val_macro_f1")
            row["val_min_per_class_f1"] = val.get("min_per_class_f1")
            row["val_balanced_accuracy"] = val.get("balanced_accuracy")
            row["val_theft_f1"] = val.get("theft_f1") or (
                (val.get("per_class") or {}).get("Theft") or {}
            ).get("f1")
            row["val_normal_f1"] = val.get("normal_f1") or (
                (val.get("per_class") or {}).get("Normal") or {}
            ).get("f1")
            row["elapsed_sec"] = (data.get("extra") or {}).get("elapsed_sec")
            print(
                f"  -> best_val_macro_f1={row.get('best_val_macro_f1')} "
                f"min_cls={row.get('val_min_per_class_f1')} "
                f"theft={row.get('val_theft_f1')}",
                flush=True,
            )
        else:
            print(f"  -> FAILED rc={rc} (no results JSON)", flush=True)
        rows.append(row)

    # Ranking
    ok_rows = [r for r in rows if r.get("best_val_macro_f1") is not None]
    ranking = sorted(ok_rows, key=lambda r: -float(r["best_val_macro_f1"]))
    best = ranking[0] if ranking else None
    shuffle_f1 = next(
        (r.get("best_val_macro_f1") for r in rows if r["sampler"] == "shuffle"), None
    )
    strat_f1 = next(
        (r.get("best_val_macro_f1") for r in rows if r["sampler"] == "stratified"), None
    )
    delta = None
    if shuffle_f1 is not None and strat_f1 is not None:
        delta = float(strat_f1) - float(shuffle_f1)

    # Decision: INCORPORATE stratified only if clear val lift without collapse
    decision = "RUN_DOCUMENTED"
    decision_note = ""
    if best is None:
        decision = "FAILED"
        decision_note = "No successful runs"
    elif strat_f1 is not None and shuffle_f1 is not None:
        if delta is not None and delta > 0.002:
            decision = "INCORPORATE"
            decision_note = (
                f"stratified lifts val macro-F1 by {delta:+.4f} vs shuffle; "
                "consider CAD-CBA-v1 train_sampler=stratified"
            )
        else:
            decision = "RUN_DOCUMENTED"
            decision_note = (
                f"stratified Δmacro={delta:+.4f} vs shuffle does not clear +0.002 "
                "incorporate bar (or hurts); keep shuffle default"
            )

    summary = {
        "experiment_id": "stratified_batch_compare_d6",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "loss": args.loss,
        "init_checkpoint": args.init_checkpoint,
        "hpo_config": args.hpo_config or None,
        "samplers": samplers,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_sec": time.time() - t0,
        "rows": rows,
        "ranking_val_macro_f1": [
            {
                "sampler": r["sampler"],
                "best_val_macro_f1": r.get("best_val_macro_f1"),
                "val_min_per_class_f1": r.get("val_min_per_class_f1"),
                "val_theft_f1": r.get("val_theft_f1"),
            }
            for r in ranking
        ],
        "delta_stratified_minus_shuffle": delta,
        "decision": decision,
        "decision_note": decision_note,
        "note": (
            "D6: class-balanced stratified batch (WeightedRandomSampler inverse-freq) "
            "vs shuffle under same init/HPs/seed. Val-only; test sealed. "
            "Champion not overwritten."
        ),
    }
    sp = out_dir / "summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 60, flush=True)
    print(f"D6 summary -> {sp}", flush=True)
    print(f"decision={decision} delta={delta} ranking={summary['ranking_val_macro_f1']}", flush=True)
    print(decision_note, flush=True)
    return 0 if all(r.get("returncode", 1) == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
