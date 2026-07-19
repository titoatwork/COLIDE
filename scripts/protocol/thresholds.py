"""
Per-class decision threshold calibration on **validation only**.

Never fit thresholds on the test set (Prof / Phase 1 seal rule).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    probs: (N, C)
    thresholds: (C,) — pick class i if p_i >= t_i; if multiple, highest p_i;
                if none, argmax p.
    """
    probs = np.asarray(probs, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    n, c = probs.shape
    assert thresholds.shape == (c,)
    preds = np.empty(n, dtype=np.int64)
    for i in range(n):
        eligible = np.where(probs[i] >= thresholds)[0]
        if len(eligible) == 0:
            preds[i] = int(np.argmax(probs[i]))
        else:
            preds[i] = int(eligible[np.argmax(probs[i, eligible])])
    return preds


def search_class_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray | None = None,
    objective: str = "macro_f1",
) -> dict[str, Any]:
    """
    Coordinate-wise grid search on validation probabilities.

    Starts from 0.5 vector; optimises one class at a time for macro-F1 (default).
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    n, c = probs.shape
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)

    thresholds = np.full(c, 0.5, dtype=np.float64)
    base_pred = apply_thresholds(probs, thresholds)
    best_score = float(f1_score(y_true, base_pred, average="macro", zero_division=0))

    for cls in range(c):
        local_best = thresholds[cls]
        local_score = best_score
        for t in grid:
            trial = thresholds.copy()
            trial[cls] = float(t)
            pred = apply_thresholds(probs, trial)
            score = float(f1_score(y_true, pred, average="macro", zero_division=0))
            if score > local_score:
                local_score = score
                local_best = float(t)
        thresholds[cls] = local_best
        best_score = local_score

    final_pred = apply_thresholds(probs, thresholds)
    return {
        "thresholds": thresholds.tolist(),
        "val_macro_f1": float(
            f1_score(y_true, final_pred, average="macro", zero_division=0)
        ),
        "objective": objective,
        "grid": grid.tolist(),
        "n_val": int(n),
        "n_classes": int(c),
    }
