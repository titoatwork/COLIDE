#!/usr/bin/env python3
"""
Classical ML baselines under canonical BoT-IoT protocol (same split).

Fits on stage_b_ft train; selects nothing on test unless --allow-test.
Writes per-model JSON + summary (Prof §6 G1–G5).

G2 notes (SVM):
  - Pilot --max-train with *random* subsample can drop rare classes below CV k
    (historical ERROR: CalibratedClassifierCV cv=3 with <3 samples/class).
  - Full protocol uses LinearSVC (dual=False) hard labels — no probability
    calibration (metrics are hard-label macro-F1). class_weight=None for
    protocol-fair parity with RF/XGB defaults.
  - Optional --svm-balanced adds class_weight='balanced' (documented variant).

G5 notes (LGBM):
  - Prior full run (val macro 0.5512) used weak multi-class defaults under
    extreme imbalance (Theft F1=0). Fix pass uses explicit multiclass +
    class_weight='balanced' + lower min_child_samples so rare leaves can form.
  - Still protocol-fair same split; not Optuna per-baseline HPO (G15).

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_classical_baselines.py
  PYTHONPATH=. .venv/bin/python scripts/run_classical_baselines.py --models svm,lgbm
  PYTHONPATH=. .venv/bin/python scripts/run_classical_baselines.py --models svm --max-train 200000 --stratify-subsample
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, max_n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample preserving class presence (for pilots only)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    if max_n >= n:
        return X, y
    # Proportional allocation with at least min(count, 3) per class for CV safety
    props = counts / counts.sum()
    alloc = np.maximum(np.floor(props * max_n).astype(int), np.minimum(counts, 3))
    # Cap at available
    alloc = np.minimum(alloc, counts)
    # Redistribute remainder to majority classes
    deficit = max_n - int(alloc.sum())
    order = np.argsort(-counts)
    i = 0
    while deficit > 0 and i < 100000:
        c = order[i % len(order)]
        if alloc[c] < counts[c]:
            alloc[c] += 1
            deficit -= 1
        i += 1
    idx_parts = []
    for c, k in zip(classes, alloc):
        pool = np.where(y == c)[0]
        take = rng.choice(pool, size=int(k), replace=False)
        idx_parts.append(take)
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def fit_predict(
    name: str,
    X_tr,
    y_tr,
    X_te,
    seed: int,
    *,
    svm_balanced: bool = False,
    lgbm_mode: str = "fixed",
):
    """
    lgbm_mode:
      - legacy: prior weak config (for reference only)
      - fixed: multiclass + class_weight balanced + rare-class leaf support (G5 fix)
    """
    model_cfg: dict = {"name": name}

    if name == "lr":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
        model_cfg.update({"type": "LogisticRegression", "max_iter": 500, "solver": "lbfgs"})

    elif name == "svm":
        from sklearn.svm import LinearSVC

        # Hard-label LinearSVC: no CalibratedClassifierCV (avoids CV fold rare-class
        # failures and 3× cost). dual=False is correct for n_samples >> n_features.
        cw = "balanced" if svm_balanced else None
        clf = LinearSVC(
            random_state=seed,
            max_iter=5000,
            dual=False,
            class_weight=cw,
            tol=1e-4,
        )
        model_cfg.update(
            {
                "type": "LinearSVC",
                "dual": False,
                "max_iter": 5000,
                "class_weight": cw,
                "calibration": None,
                "note": "Hard labels only; probability calibration disabled for protocol metrics",
            }
        )

    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        )
        model_cfg.update({"type": "RandomForestClassifier", "n_estimators": 200})

    elif name == "xgb":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            n_jobs=-1,
            random_state=seed,
            tree_method="hist",
        )
        model_cfg.update(
            {
                "type": "XGBClassifier",
                "n_estimators": 200,
                "max_depth": 8,
                "objective": "multi:softprob",
            }
        )

    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        import pandas as pd

        cols = [f"f{i}" for i in range(X_tr.shape[1])]
        X_tr = pd.DataFrame(np.asarray(X_tr, dtype=np.float32), columns=cols)
        X_te = pd.DataFrame(np.asarray(X_te, dtype=np.float32), columns=cols)
        n_class = int(len(np.unique(y_tr)))

        if lgbm_mode == "legacy":
            clf = LGBMClassifier(
                n_estimators=300,
                num_leaves=63,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=seed,
                verbose=-1,
                force_col_wise=True,
            )
            model_cfg.update(
                {
                    "type": "LGBMClassifier",
                    "mode": "legacy",
                    "n_estimators": 300,
                    "num_leaves": 63,
                    "note": "Prior weak full-run config (macro~0.55)",
                }
            )
        else:
            # G5 fix: explicit multiclass + class-balanced sample weights + rare leaves
            clf = LGBMClassifier(
                n_estimators=300,
                num_leaves=63,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                num_class=n_class,
                class_weight="balanced",
                min_child_samples=5,
                min_split_gain=0.0,
                reg_lambda=0.0,
                n_jobs=-1,
                random_state=seed,
                verbose=-1,
                force_col_wise=True,
            )
            model_cfg.update(
                {
                    "type": "LGBMClassifier",
                    "mode": "fixed",
                    "n_estimators": 300,
                    "num_leaves": 63,
                    "max_depth": 8,
                    "objective": "multiclass",
                    "num_class": n_class,
                    "class_weight": "balanced",
                    "min_child_samples": 5,
                    "note": "G5 fix pass: multiclass + balanced weights + min_child_samples=5",
                }
            )
    else:
        raise ValueError(name)

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    train_sec = time.time() - t0
    t1 = time.time()
    pred = clf.predict(X_te)
    infer_sec = time.time() - t1
    return pred, train_sec, infer_sec, clf, model_cfg


def rebuild_handoff(out_dir: Path, seed: int) -> Path:
    """Rebuild summary_handoff.json from individual successful model JSONs."""
    rows = []
    for name in ["lr", "svm", "rf", "xgb", "lgbm"]:
        path = out_dir / f"{name}_seed{seed}.json"
        if not path.is_file():
            continue
        with open(path) as f:
            env = json.load(f)
        val = (env.get("metrics") or {}).get("val") or {}
        cfg = env.get("config") or {}
        per = val.get("per_class") or {}
        rows.append(
            {
                "model": name,
                "ok": True,
                "subsampled": bool(cfg.get("subsampled", False)),
                "max_train": cfg.get("max_train", 0),
                "val_macro_f1": val.get("macro_f1"),
                "val_balanced_accuracy": val.get("balanced_accuracy"),
                "val_min_per_class_f1": val.get("min_per_class_f1"),
                "theft_f1": (per.get("Theft") or {}).get("f1"),
                "normal_f1": (per.get("Normal") or {}).get("f1"),
                "path": str(path.relative_to(PROJECT_ROOT))
                if path.is_relative_to(PROJECT_ROOT)
                else str(path),
                "model_cfg": cfg.get("model_cfg"),
            }
        )
    handoff = {
        "experiment_id": "classical_baselines_protocol_stage_b_ft",
        "seed": seed,
        "allow_test": False,
        "note": (
            "Rebuilt from individual JSONs. Published RF 0.9864 is a different "
            "pipeline (rf_baseline_processed). Prefer this file over summary.json "
            "(which may only hold the last --models invocation)."
        ),
        "rows": rows,
    }
    hp = out_dir / "summary_handoff.json"
    with open(hp, "w") as f:
        json.dump(handoff, f, indent=2)
    return hp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models",
        type=str,
        default="lr,svm,rf,xgb,lgbm",
        help="Comma list: lr,svm,rf,xgb,lgbm",
    )
    p.add_argument("--allow-test", action="store_true")
    p.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="If >0, subsample train for speed (document as pilot if used)",
    )
    p.add_argument(
        "--stratify-subsample",
        action="store_true",
        help="When --max-train>0, use stratified subsample (preserves rare classes)",
    )
    p.add_argument(
        "--svm-balanced",
        action="store_true",
        help="LinearSVC class_weight='balanced' (default None for protocol-fair)",
    )
    p.add_argument(
        "--lgbm-mode",
        type=str,
        default="fixed",
        choices=["fixed", "legacy"],
        help="G5 fix (fixed) vs prior weak config (legacy)",
    )
    p.add_argument(
        "--rebuild-handoff-only",
        action="store_true",
        help="Only rebuild summary_handoff.json from existing per-model JSONs",
    )
    args = p.parse_args()

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "baselines_classical"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_handoff_only:
        hp = rebuild_handoff(out_dir, args.seed)
        print("wrote", hp)
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    bundle = load_botiot(stage="stage_b_ft", seed=args.seed)
    X_tr, y_tr = bundle.X_train, bundle.y_train
    if args.max_train and args.max_train < len(y_tr):
        if args.stratify_subsample:
            X_tr, y_tr = stratified_subsample(
                np.asarray(X_tr), np.asarray(y_tr), args.max_train, args.seed
            )
            subsample_mode = "stratified"
        else:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(y_tr), size=args.max_train, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
            subsample_mode = "random"
        subsampled = True
    else:
        subsampled = False
        subsample_mode = None

    # Class counts for documentation
    u, c = np.unique(y_tr, return_counts=True)
    train_counts = {bundle.class_names[int(i)]: int(n) for i, n in zip(u, c)}
    print(f"train n={len(y_tr)} counts={train_counts} subsampled={subsampled}", flush=True)

    summary_rows = []
    for name in models:
        print(f"\n=== baseline {name} ===", flush=True)
        try:
            pred_val, tr_sec, _, _, model_cfg = fit_predict(
                name,
                X_tr,
                y_tr,
                bundle.X_val,
                args.seed,
                svm_balanced=args.svm_balanced,
                lgbm_mode=args.lgbm_mode,
            )
            val_m = compute_classification_metrics(
                bundle.y_val, pred_val, bundle.class_names
            )
            test_m = None
            te_sec = None
            if args.allow_test:
                pred_te, _, te_sec, _, _ = fit_predict(
                    name,
                    X_tr,
                    y_tr,
                    bundle.X_test,
                    args.seed,
                    svm_balanced=args.svm_balanced,
                    lgbm_mode=args.lgbm_mode,
                )
                test_m = compute_classification_metrics(
                    bundle.y_test, pred_te, bundle.class_names
                )

            env = make_result_envelope(
                experiment_id=f"classical_{name}",
                protocol_id=bundle.protocol_id,
                stage="stage_b_ft",
                seed=args.seed,
                config={
                    "model": name,
                    "model_cfg": model_cfg,
                    "max_train": args.max_train,
                    "subsampled": subsampled,
                    "subsample_mode": subsample_mode,
                    "allow_test": bool(args.allow_test),
                    "train_class_counts": train_counts,
                    "lgbm_mode": args.lgbm_mode if name == "lgbm" else None,
                    "svm_balanced": bool(args.svm_balanced) if name == "svm" else None,
                },
                metrics={"val": val_m, "test": test_m},
                extra={
                    "train_sec": tr_sec,
                    "infer_sec_on_eval_split": te_sec,
                    "data_summary": bundle.summary(),
                },
                project_root=PROJECT_ROOT,
            )
            path = out_dir / f"{name}_seed{args.seed}.json"
            with open(path, "w") as f:
                json.dump(env, f, indent=2)
            row = {
                "model": name,
                "val_macro_f1": val_m["macro_f1"],
                "val_balanced_accuracy": val_m["balanced_accuracy"],
                "val_min_per_class_f1": val_m["min_per_class_f1"],
                "theft_f1": (val_m.get("per_class") or {}).get("Theft", {}).get("f1"),
                "normal_f1": (val_m.get("per_class") or {}).get("Normal", {}).get("f1"),
                "train_sec": tr_sec,
                "path": str(path),
                "ok": True,
                "model_cfg": model_cfg,
            }
            if test_m:
                row["test_macro_f1"] = test_m["macro_f1"]
            print(
                f"{name}: val_macro_f1={val_m['macro_f1']:.4f} "
                f"min_cls={val_m['min_per_class_f1']:.4f} "
                f"theft={row.get('theft_f1')} train_sec={tr_sec:.1f}",
                flush=True,
            )
        except Exception as e:
            import traceback

            row = {
                "model": name,
                "ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
            err_path = out_dir / f"{name}_seed{args.seed}_ERROR.json"
            with open(err_path, "w") as f:
                json.dump(row, f, indent=2)
            print(f"{name}: FAILED {e}", flush=True)
            traceback.print_exc()
        summary_rows.append(row)

    summary = {
        "experiment_id": "classical_baselines",
        "seed": args.seed,
        "subsampled": subsampled,
        "subsample_mode": subsample_mode,
        "max_train": args.max_train,
        "allow_test": bool(args.allow_test),
        "models_requested": models,
        "train_class_counts": train_counts,
        "rows": summary_rows,
        "note": (
            "summary.json only covers this invocation's --models. "
            "Use summary_handoff.json for the full G1–G5 table."
        ),
    }
    sp = out_dir / "summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", sp, flush=True)

    hp = rebuild_handoff(out_dir, args.seed)
    print("wrote", hp, flush=True)

    # TABLE_VAL convenience
    table = {
        "seed": args.seed,
        "stage": "stage_b_ft",
        "protocol_id": "botiot_v1",
        "val_macro_f1": {
            r["model"]: r.get("val_macro_f1")
            for r in (json.load(open(hp))["rows"])
            if r.get("ok")
        },
    }
    # merge prior models not in this run from handoff
    tp = out_dir / "TABLE_VAL.json"
    with open(tp, "w") as f:
        json.dump(table, f, indent=2)
    print("wrote", tp, flush=True)

    return 0 if all(r.get("ok") for r in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
