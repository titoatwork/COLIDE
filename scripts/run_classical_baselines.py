#!/usr/bin/env python3
"""
Classical ML baselines under canonical BoT-IoT protocol (same split).

Fits on stage_b_ft train; selects nothing on test unless --allow-test.
Writes per-model JSON + summary (Prof §6 G1–G5 partial).

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_classical_baselines.py
  PYTHONPATH=. .venv/bin/python scripts/run_classical_baselines.py --models rf,xgb --allow-test
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


def fit_predict(name: str, X_tr, y_tr, X_te, seed: int):
    if name == "lr":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=seed,
        )
    elif name == "svm":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        base = LinearSVC(random_state=seed, max_iter=2000, dual=False)
        clf = CalibratedClassifierCV(base, cv=3)
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        )
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
    elif name == "lgbm":
        from lightgbm import LGBMClassifier

        # Force numeric feature matrix (avoid feature-name mismatch warnings / bugs)
        X_tr = np.asarray(X_tr, dtype=np.float32)
        X_te = np.asarray(X_te, dtype=np.float32)
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
    else:
        raise ValueError(name)

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    train_sec = time.time() - t0
    t1 = time.time()
    pred = clf.predict(X_te)
    infer_sec = time.time() - t1
    return pred, train_sec, infer_sec, clf


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
        help="If >0, subsample train for speed (document as RUN_DOCUMENTED if used)",
    )
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "baselines_classical"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_botiot(stage="stage_b_ft", seed=args.seed)
    X_tr, y_tr = bundle.X_train, bundle.y_train
    if args.max_train and args.max_train < len(y_tr):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(y_tr), size=args.max_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        subsampled = True
    else:
        subsampled = False

    summary_rows = []
    for name in models:
        print(f"\n=== baseline {name} ===", flush=True)
        try:
            # Val predictions for selection narrative; test only if allowed
            pred_val, tr_sec, _, _ = fit_predict(
                name, X_tr, y_tr, bundle.X_val, args.seed
            )
            val_m = compute_classification_metrics(
                bundle.y_val, pred_val, bundle.class_names
            )
            test_m = None
            if args.allow_test:
                pred_te, _, te_sec, _ = fit_predict(
                    name, X_tr, y_tr, bundle.X_test, args.seed
                )
                # refit once already done — re-fit is wasteful; use second fit_predict
                test_m = compute_classification_metrics(
                    bundle.y_test, pred_te, bundle.class_names
                )
            else:
                te_sec = None

            env = make_result_envelope(
                experiment_id=f"classical_{name}",
                protocol_id=bundle.protocol_id,
                stage="stage_b_ft",
                seed=args.seed,
                config={
                    "model": name,
                    "max_train": args.max_train,
                    "subsampled": subsampled,
                    "allow_test": bool(args.allow_test),
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
                "path": str(path),
                "ok": True,
            }
            if test_m:
                row["test_macro_f1"] = test_m["macro_f1"]
            print(
                f"{name}: val_macro_f1={val_m['macro_f1']:.4f} "
                f"min_cls={val_m['min_per_class_f1']:.4f}"
            )
        except Exception as e:
            row = {"model": name, "ok": False, "error": repr(e)}
            err_path = out_dir / f"{name}_seed{args.seed}_ERROR.json"
            with open(err_path, "w") as f:
                json.dump(row, f, indent=2)
            print(f"{name}: FAILED {e}")
        summary_rows.append(row)

    summary = {
        "experiment_id": "classical_baselines",
        "seed": args.seed,
        "subsampled": subsampled,
        "max_train": args.max_train,
        "allow_test": bool(args.allow_test),
        "rows": summary_rows,
    }
    sp = out_dir / "summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", sp)
    return 0 if all(r.get("ok") for r in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
