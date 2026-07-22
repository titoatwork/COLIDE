#!/usr/bin/env python3
"""
B14 — Sealed multi-seed BoT TEST for CAD-CBA-v1 (FINAL_CONFIG_FREEZE_CARD gate).

HARD GATE: refuses to unseal BoT test unless the user-lock CLI flags match the
freeze card. Never overwrites production champion.

Init paths (from FINAL_CONFIG_FREEZE_CARD.md):
  A (recommended): distill init + hpo_best FT per seed, then test
  B: ensemble KD init + hpo_best FT per seed, then test
  C: evaluate existing checkpoints only (no retrain)

Example (only after user pastes lock in chat and operator copies flags):

  PYTHONPATH=. .venv/bin/python scripts/run_sealed_test_b14.py \\
    --i-lock-cad-cba-v1 \\
    --init-path A \\
    --confirm-champion-md5 80a90f7cc210276300eaa90173a5a385 \\
    --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CHAMPION_MD5 = "80a90f7cc210276300eaa90173a5a385"
CHAMPION_PATH = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
DEFAULT_HPO = "config/hpo_best.yaml"
DEFAULT_TAG = "sealed_test"

INIT_PATHS = {
    "A": {
        "name": "distill_hpo_best_ft",
        "init": "model/best_model_botiot_distill_a0.6_T10.0_focal2.pth",
        "mode": "train_then_test",
        "rationale": "Matches WP3 / HPO confirm protocol (recommended)",
    },
    "B": {
        "name": "ensemble_kd_hpo_best_ft",
        "init": "model/teachers_kd/kd_ensemble_a0.6_T10.0_g2.0_seed42.pth",
        "mode": "train_then_test",
        "rationale": "Matches package multirun (mean lower than WP1b on val)",
    },
    "C": {
        "name": "eval_existing_only",
        "init": None,
        "mode": "eval_only",
        "rationale": "Fast; weaker fresh multi-seed train story",
        "default_ckpts": [
            f"model/multirun_hpo_confirm/ft_seed{s}.pth" for s in (42, 43, 44, 45, 46)
        ],
    },
}

LOCKED_TAGS = {
    "multirun",
    "imbalance_loss",
    "hpo",
    "teachers_kd",
    "multirun_ensemble_hpo",
    "multirun_hpo_confirm",
    "ablation_ladder",
    "baselines_neural",
    "cstar_bounded",
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def enforce_lock(args: argparse.Namespace) -> int:
    """Return 0 if lock OK, else non-zero."""
    if not args.i_lock_cad_cba_v1:
        print(
            "ERROR: BoT test remains SEALED.\n"
            "Paste the freeze-card lock in chat, then re-run with:\n"
            "  --i-lock-cad-cba-v1 --init-path A|B|C "
            f"--confirm-champion-md5 {EXPECTED_CHAMPION_MD5}\n"
            "See docs/execution_plan/FINAL_CONFIG_FREEZE_CARD.md",
            file=sys.stderr,
        )
        return 10

    if args.init_path not in INIT_PATHS:
        print(f"ERROR: --init-path must be A, B, or C (got {args.init_path!r})", file=sys.stderr)
        return 11

    got = (args.confirm_champion_md5 or "").strip().lower()
    if got != EXPECTED_CHAMPION_MD5:
        print(
            f"ERROR: --confirm-champion-md5 must be exactly {EXPECTED_CHAMPION_MD5} "
            f"(got {args.confirm_champion_md5!r}). "
            "Champion must remain unchanged unless user says BACKUP+replace.",
            file=sys.stderr,
        )
        return 12

    if not CHAMPION_PATH.is_file():
        print(f"ERROR: missing champion {CHAMPION_PATH}", file=sys.stderr)
        return 13

    live = md5_file(CHAMPION_PATH)
    if live != EXPECTED_CHAMPION_MD5:
        print(
            f"ERROR: live champion md5 {live} != expected {EXPECTED_CHAMPION_MD5}. Abort.",
            file=sys.stderr,
        )
        return 14

    if args.tag in LOCKED_TAGS:
        print(f"ERROR: tag={args.tag!r} collides with locked result tree.", file=sys.stderr)
        return 15

    if args.allow_overwrite_champion:
        print(
            "ERROR: refusing --allow-overwrite-champion on B14 driver. "
            "Champion stays frozen.",
            file=sys.stderr,
        )
        return 16

    return 0


def run_train_seed(
    *,
    python: str,
    init_ckpt: Path,
    hpo_cfg: Path,
    seed: int,
    epochs: int,
    patience: int,
    save_path: Path,
    results_path: Path,
) -> int:
    cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "train_protocol_ft.py"),
        "--init-checkpoint",
        str(init_ckpt),
        "--hpo-config",
        str(hpo_cfg),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--save-path",
        str(save_path),
        "--results-path",
        str(results_path),
        "--allow-test",
    ]
    print("\n" + "=" * 70)
    print(f"B14 TRAIN+TEST seed={seed}")
    print(" ".join(cmd))
    print("=" * 70, flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)


def run_eval_seed(
    *,
    python: str,
    checkpoint: Path,
    seed: int,
    out_path: Path,
) -> int:
    cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "eval_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--stage",
        "stage_b_ft",
        "--seed",
        str(seed),
        "--allow-test",
        "--out",
        str(out_path),
    ]
    print("\n" + "=" * 70)
    print(f"B14 EVAL-ONLY seed={seed} ckpt={checkpoint}")
    print(" ".join(cmd))
    print("=" * 70, flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)


def extract_metrics(results_path: Path, mode: str) -> dict:
    entry: dict = {"result_path": str(results_path)}
    if not results_path.is_file():
        return entry
    with open(results_path) as f:
        data = json.load(f)
    if mode == "train_then_test":
        m = data.get("metrics") or {}
        val = m.get("val") or {}
        test = m.get("test") or {}
        entry["best_val_macro_f1"] = m.get("best_val_macro_f1")
        entry["val_macro_f1"] = val.get("macro_f1")
        entry["val_min_per_class_f1"] = val.get("min_per_class_f1")
        entry["val_theft_f1"] = val.get("theft_f1")
        entry["val_balanced_accuracy"] = val.get("balanced_accuracy")
        entry["test_macro_f1"] = test.get("macro_f1") if test else None
        entry["test_min_per_class_f1"] = test.get("min_per_class_f1") if test else None
        entry["test_theft_f1"] = test.get("theft_f1") if test else None
        entry["test_balanced_accuracy"] = test.get("balanced_accuracy") if test else None
        entry["test_per_class"] = test.get("per_class") if test else None
        entry["elapsed_sec"] = (data.get("extra") or {}).get("elapsed_sec")
        entry["allow_test"] = (data.get("extra") or {}).get("allow_test")
    else:
        # eval_checkpoint schema
        val = data.get("val") or {}
        test = data.get("test") or {}
        entry["val_macro_f1"] = val.get("macro_f1")
        entry["val_min_per_class_f1"] = val.get("min_per_class_f1")
        entry["val_theft_f1"] = val.get("theft_f1")
        entry["val_balanced_accuracy"] = val.get("balanced_accuracy")
        entry["test_macro_f1"] = test.get("macro_f1") if test else None
        entry["test_min_per_class_f1"] = test.get("min_per_class_f1") if test else None
        entry["test_theft_f1"] = test.get("theft_f1") if test else None
        entry["test_balanced_accuracy"] = test.get("balanced_accuracy") if test else None
        entry["test_per_class"] = test.get("per_class") if test else None
        entry["allow_test"] = data.get("allow_test")
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--i-lock-cad-cba-v1",
        action="store_true",
        help="Required: confirms user pasted FINAL_CONFIG_FREEZE_CARD lock in chat",
    )
    p.add_argument(
        "--init-path",
        type=str,
        choices=["A", "B", "C"],
        required=True,
        help="Freeze-card init path A|B|C",
    )
    p.add_argument(
        "--confirm-champion-md5",
        type=str,
        default="",
        help=f"Must equal {EXPECTED_CHAMPION_MD5}",
    )
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--hpo-config", type=str, default=DEFAULT_HPO)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--tag", type=str, default=DEFAULT_TAG)
    p.add_argument(
        "--ckpts",
        type=str,
        default="",
        help="Path C only: comma-separated checkpoints (default multirun_hpo_confirm seeds)",
    )
    p.add_argument(
        "--allow-overwrite-champion",
        action="store_true",
        help="Intentionally rejected by this driver",
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate lock + print plan only (does not unseal / train)",
    )
    args = p.parse_args()

    rc = enforce_lock(args)
    if rc != 0:
        return rc

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 5:
        print("ERROR: B14 requires ≥5 seeds (L5 / freeze card).", file=sys.stderr)
        return 17

    path_meta = INIT_PATHS[args.init_path]
    hpo_cfg = Path(args.hpo_config)
    if path_meta["mode"] == "train_then_test":
        if not hpo_cfg.is_file():
            print(f"ERROR: missing hpo config {hpo_cfg}", file=sys.stderr)
            return 18
        init_ckpt = Path(path_meta["init"])
        if not init_ckpt.is_file():
            print(f"ERROR: missing init {init_ckpt}", file=sys.stderr)
            return 19
    else:
        init_ckpt = None
        if args.ckpts:
            ckpt_list = [Path(x.strip()) for x in args.ckpts.split(",") if x.strip()]
        else:
            ckpt_list = [Path(x) for x in path_meta["default_ckpts"]]
        if len(ckpt_list) != len(seeds):
            print(
                f"ERROR: path C needs same count of ckpts and seeds "
                f"({len(ckpt_list)} vs {len(seeds)})",
                file=sys.stderr,
            )
            return 20
        for c in ckpt_list:
            if not c.is_file():
                print(f"ERROR: missing checkpoint {c}", file=sys.stderr)
                return 21

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / args.tag
    ckpt_dir = PROJECT_ROOT / "model" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if path_meta["mode"] == "train_then_test":
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "experiment_id": "b14_sealed_multiseed_bot_test",
        "tracker": "B14",
        "work_package": "WP3-final-test / sealed_test",
        "init_path": args.init_path,
        "init_path_meta": path_meta,
        "seeds": seeds,
        "hpo_config": str(hpo_cfg) if path_meta["mode"] == "train_then_test" else None,
        "tag": args.tag,
        "champion_md5_expected": EXPECTED_CHAMPION_MD5,
        "champion_md5_live": md5_file(CHAMPION_PATH),
        "git_sha": _git_sha(),
        "dry_run": bool(args.dry_run),
        "lock_flags_ok": True,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("DRY-RUN: lock OK; no train / no test unseal.")
        dry_path = out_dir / "dry_run_plan.json"
        with open(dry_path, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"wrote {dry_path}")
        return 0

    # Final pre-flight champion check
    if md5_file(CHAMPION_PATH) != EXPECTED_CHAMPION_MD5:
        print("ERROR: champion md5 changed mid-flight. Abort.", file=sys.stderr)
        return 14

    runs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    if path_meta["mode"] == "train_then_test":
        assert init_ckpt is not None
        for seed in seeds:
            save_path = ckpt_dir / f"ft_seed{seed}.pth"
            results_path = out_dir / f"ft_seed{seed}.json"
            rc_seed = run_train_seed(
                python=args.python,
                init_ckpt=init_ckpt,
                hpo_cfg=hpo_cfg,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                save_path=save_path,
                results_path=results_path,
            )
            entry = {
                "seed": seed,
                "returncode": rc_seed,
                "checkpoint_path": str(save_path),
                "mode": "train_then_test",
            }
            entry.update(extract_metrics(results_path, "train_then_test"))
            runs.append(entry)
    else:
        for seed, ckpt in zip(seeds, ckpt_list):
            results_path = out_dir / f"eval_seed{seed}.json"
            rc_seed = run_eval_seed(
                python=args.python,
                checkpoint=ckpt,
                seed=seed,
                out_path=results_path,
            )
            entry = {
                "seed": seed,
                "returncode": rc_seed,
                "checkpoint_path": str(ckpt),
                "mode": "eval_only",
            }
            entry.update(extract_metrics(results_path, "eval_only"))
            runs.append(entry)

    t1 = datetime.now(timezone.utc)
    champ_after = md5_file(CHAMPION_PATH)
    test_f1s = [r["test_macro_f1"] for r in runs if r.get("test_macro_f1") is not None]
    val_f1s = [r["val_macro_f1"] for r in runs if r.get("val_macro_f1") is not None]
    test_min = [r["test_min_per_class_f1"] for r in runs if r.get("test_min_per_class_f1") is not None]
    test_theft = [r["test_theft_f1"] for r in runs if r.get("test_theft_f1") is not None]

    test_mean, test_std = _mean_std(test_f1s)
    val_mean, val_std = _mean_std(val_f1s)
    min_mean, min_std = _mean_std(test_min)
    theft_mean, theft_std = _mean_std(test_theft)

    summary = {
        "experiment_id": "b14_sealed_multiseed_bot_test",
        "tracker": "B14",
        "work_package": "WP3-final-test / sealed_test",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "init_path": args.init_path,
        "init_path_meta": path_meta,
        "hpo_config": str(hpo_cfg) if path_meta["mode"] == "train_then_test" else None,
        "init_checkpoint": str(init_ckpt) if init_ckpt else None,
        "epochs": args.epochs if path_meta["mode"] == "train_then_test" else None,
        "patience": args.patience if path_meta["mode"] == "train_then_test" else None,
        "seeds": seeds,
        "tag": args.tag,
        "n_success": len(test_f1s),
        "n_requested": len(seeds),
        "test_macro_f1_mean": test_mean,
        "test_macro_f1_std": test_std,
        "test_macro_f1_min": min(test_f1s) if test_f1s else None,
        "test_macro_f1_max": max(test_f1s) if test_f1s else None,
        "test_min_per_class_f1_mean": min_mean,
        "test_min_per_class_f1_std": min_std,
        "test_theft_f1_mean": theft_mean,
        "test_theft_f1_std": theft_std,
        "val_macro_f1_mean": val_mean,
        "val_macro_f1_std": val_std,
        "wall_sec": (t1 - t0).total_seconds(),
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "git_sha": _git_sha(),
        "champion_md5_expected": EXPECTED_CHAMPION_MD5,
        "champion_md5_before": EXPECTED_CHAMPION_MD5,
        "champion_md5_after": champ_after,
        "champion_unchanged": champ_after == EXPECTED_CHAMPION_MD5,
        "allow_test": True,
        "user_lock_confirmed": True,
        "runs": runs,
        "comparators_val_only": {
            "wp1b_multirun_mean_std": "0.9714 ± 0.0109",
            "wp3_hpo_winner_seed42": 0.9791,
            "hpo_confirm_mean_std": "0.9689 ± 0.0145",
            "package_ensemble_hpo_mean_std": "0.9639 ± 0.0185",
            "protocol_rf_val": 0.9778,
            "protocol_lgbm_val": 0.9818,
        },
        "decision": "RUN_DOCUMENTED" if test_f1s else "FAILED",
        "decision_note": (
            "Sealed multi-seed BoT TEST after explicit freeze-card lock. "
            "Production champion not overwritten. Numbers from this summary only."
        ),
        "note": (
            "Do not mix these TEST numbers with val-only multirun claims without labeling. "
            "Rebuild claims package after this run."
        ),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Human table
    table_lines = [
        "# B14 Sealed multi-seed BoT TEST",
        "",
        f"- Init path: **{args.init_path}** ({path_meta['name']})",
        f"- Seeds: {seeds}",
        f"- Champion md5 unchanged: **{summary['champion_unchanged']}** (`{champ_after}`)",
        f"- Test macro-F1 mean±std: **{test_mean:.4f} ± {test_std:.4f}** (n={len(test_f1s)})"
        if test_mean is not None and test_std is not None
        else f"- Test macro-F1: incomplete (n={len(test_f1s)})",
        "",
        "| Seed | val macro-F1 | test macro-F1 | test min-cls | test Theft |",
        "|------|--------------|---------------|--------------|------------|",
    ]
    for r in runs:
        table_lines.append(
            f"| {r.get('seed')} | "
            f"{r.get('val_macro_f1') if r.get('val_macro_f1') is not None else '—'} | "
            f"{r.get('test_macro_f1') if r.get('test_macro_f1') is not None else '—'} | "
            f"{r.get('test_min_per_class_f1') if r.get('test_min_per_class_f1') is not None else '—'} | "
            f"{r.get('test_theft_f1') if r.get('test_theft_f1') is not None else '—'} |"
        )
    table_path = out_dir / "table.md"
    table_path.write_text("\n".join(table_lines) + "\n")

    print("\nSUMMARY", json.dumps({k: summary[k] for k in summary if k != "runs"}, indent=2))
    print("wrote", summary_path)
    print("wrote", table_path)

    if champ_after != EXPECTED_CHAMPION_MD5:
        print("CRITICAL: champion md5 changed — investigate immediately", file=sys.stderr)
        return 30
    if len(test_f1s) != len(seeds):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
