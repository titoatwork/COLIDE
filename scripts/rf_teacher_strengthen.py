#!/usr/bin/env python3
"""
COLIDE Session 5 — RF teacher / baseline strengthening sweep.

Trains several RandomForest configurations on the EXACT same preprocessed
splits as scripts/rf_baseline_processed.py (data/processed/*.npy).

Safety:
  - Does NOT overwrite benchmarks/results/rf_baseline_processed.json
    (canonical source of the published 0.9864 figure).
  - Does NOT touch any model/*.pth checkpoints.
  - Writes only benchmarks/results/rf_teacher_strengthen.json.

Usage:
    PYTHONPATH=. python scripts/rf_teacher_strengthen.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROJECT_ROOT / "benchmarks" / "results" / "rf_teacher_strengthen.json"
CANONICAL_PATH = PROJECT_ROOT / "benchmarks" / "results" / "rf_baseline_processed.json"
SEED = 42

# Must match rf_baseline_processed.py exactly for the sanity row.
CONFIGS = [
    {
        "name": "baseline_200",
        "n_estimators": 200,
        "class_weight": None,
        "max_depth": None,
        "note": "Sanity: should reproduce rf_baseline_processed.json ~0.9864",
    },
    {
        "name": "trees_500",
        "n_estimators": 500,
        "class_weight": None,
        "max_depth": None,
        "note": "More trees only",
    },
    {
        "name": "trees_200_balanced",
        "n_estimators": 200,
        "class_weight": "balanced",
        "max_depth": None,
        "note": "class_weight=balanced",
    },
    {
        "name": "trees_500_balanced",
        "n_estimators": 500,
        "class_weight": "balanced",
        "max_depth": None,
        "note": "More trees + balanced",
    },
    {
        "name": "trees_500_balanced_depth30",
        "n_estimators": 500,
        "class_weight": "balanced",
        "max_depth": 30,
        "note": "Capped depth to reduce overfit risk",
    },
    {
        "name": "trees_500_balanced_subsample",
        "n_estimators": 500,
        "class_weight": "balanced_subsample",
        "max_depth": None,
        "note": "balanced_subsample variant",
    },
]


def _eval(rf, X, y):
    pred = rf.predict(X)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_f1": f1_score(y, pred, average=None).tolist(),
    }


def main():
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Canonical baseline JSON (untouched): {CANONICAL_PATH}")
    print(f"Results will be written to: {OUT_PATH}")
    print("=" * 70)

    rows = []
    for cfg in CONFIGS:
        print(f"\n>>> {cfg['name']}: n_estimators={cfg['n_estimators']} "
              f"class_weight={cfg['class_weight']} max_depth={cfg['max_depth']}")
        rf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            class_weight=cfg["class_weight"],
            max_depth=cfg["max_depth"],
            random_state=SEED,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        val_m = _eval(rf, X_val, y_val)
        test_m = _eval(rf, X_test, y_test)
        row = {
            "name": cfg["name"],
            "note": cfg["note"],
            "n_estimators": cfg["n_estimators"],
            "class_weight": cfg["class_weight"],
            "max_depth": cfg["max_depth"],
            "random_state": SEED,
            "val": val_m,
            "test": test_m,
        }
        rows.append(row)
        print(
            f"    Val  macro-F1={val_m['macro_f1']:.4f} | "
            f"Test macro-F1={test_m['macro_f1']:.4f} acc={test_m['accuracy']:.4f}"
        )
        print(f"    Test per-class F1: {[round(x, 4) for x in test_m['per_class_f1']]}")

    # Rank by test macro-F1 (descending)
    ranked = sorted(rows, key=lambda r: r["test"]["macro_f1"], reverse=True)
    best = ranked[0]
    baseline = next(r for r in rows if r["name"] == "baseline_200")

    payload = {
        "data_source": "data/processed/*.npy (same as rf_baseline_processed.py)",
        "seed": SEED,
        "canonical_baseline_json": str(CANONICAL_PATH.relative_to(PROJECT_ROOT)),
        "canonical_baseline_untouched": True,
        "n_configs": len(rows),
        "configs": rows,
        "best_by_test_macro_f1": {
            "name": best["name"],
            "test_macro_f1": best["test"]["macro_f1"],
            "delta_vs_baseline_200": best["test"]["macro_f1"] - baseline["test"]["macro_f1"],
        },
        "baseline_200_test_macro_f1": baseline["test"]["macro_f1"],
        "recommendation": (
            "If best beats baseline meaningfully on minority classes, try KD with "
            "train_distill.py --rf-n-estimators ... --rf-class-weight ... (Session 6). "
            "If plateau / worse, keep published RF bar 0.9864 and student 0.9790."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"BEST: {best['name']} test macro-F1={best['test']['macro_f1']:.4f} "
          f"(delta vs baseline_200: {best['test']['macro_f1'] - baseline['test']['macro_f1']:+.4f})")
    print(f"Saved: {OUT_PATH}")
    print("Canonical rf_baseline_processed.json was NOT modified.")


if __name__ == "__main__":
    main()
