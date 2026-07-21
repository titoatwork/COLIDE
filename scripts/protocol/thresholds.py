"""
Per-class decision threshold calibration on **validation only**.

Never fit thresholds on the test set (Prof / Phase 1 seal rule).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    probs: (N, C)
    thresholds: (C,) — pick class i if p_i >= t_i; if multiple, highest p_i;
                if none, argmax p.

    Vectorised: mask ineligible classes with -inf then argmax.
    """
    probs = np.asarray(probs, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    n, c = probs.shape
    assert thresholds.shape == (c,)
    eligible = probs >= thresholds  # (N, C)
    masked = np.where(eligible, probs, -np.inf)
    # rows with no eligible class → fall back to plain argmax
    none_ok = ~eligible.any(axis=1)
    if none_ok.any():
        masked = masked.copy()
        masked[none_ok] = probs[none_ok]
    return np.argmax(masked, axis=1).astype(np.int64)


def _score_from_preds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    objective: str,
) -> float:
    """Scalar objective for threshold search (higher is better)."""
    if objective == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if objective == "min_per_class_f1":
        per = f1_score(
            y_true,
            y_pred,
            average=None,
            labels=list(range(int(y_true.max()) + 1)),
            zero_division=0,
        )
        return float(np.min(per)) if len(per) else 0.0
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if objective.startswith("class_f1:"):
        # e.g. class_f1:3 → F1 of class index 3
        cls = int(objective.split(":", 1)[1])
        per = f1_score(
            y_true,
            y_pred,
            average=None,
            labels=list(range(int(max(y_true.max(), y_pred.max())) + 1)),
            zero_division=0,
        )
        if cls < 0 or cls >= len(per):
            return 0.0
        return float(per[cls])
    raise ValueError(
        f"Unknown objective {objective!r}; "
        "use macro_f1 | min_per_class_f1 | balanced_accuracy | class_f1:<idx>"
    )


def search_class_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray | None = None,
    objective: str = "macro_f1",
    init_thresholds: np.ndarray | None = None,
    n_passes: int = 2,
) -> dict[str, Any]:
    """
    Coordinate-wise grid search on validation probabilities.

    Starts from ``init_thresholds`` (default 0.5 vector); optimises one class
    at a time for ``objective``. Multiple passes refine after other classes move.

    Parameters
    ----------
    objective:
        macro_f1 | min_per_class_f1 | balanced_accuracy | class_f1:<idx>
    n_passes:
        Full sweeps over all classes (default 2 for better local optima).
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    n, c = probs.shape
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)

    if init_thresholds is None:
        thresholds = np.full(c, 0.5, dtype=np.float64)
    else:
        thresholds = np.asarray(init_thresholds, dtype=np.float64).copy()
        assert thresholds.shape == (c,)

    base_pred = apply_thresholds(probs, thresholds)
    best_score = _score_from_preds(y_true, base_pred, objective)

    history: list[dict[str, Any]] = []
    for pass_i in range(int(n_passes)):
        for cls in range(c):
            local_best = float(thresholds[cls])
            local_score = best_score
            for t in grid:
                trial = thresholds.copy()
                trial[cls] = float(t)
                pred = apply_thresholds(probs, trial)
                score = _score_from_preds(y_true, pred, objective)
                if score > local_score + 1e-12:
                    local_score = score
                    local_best = float(t)
            thresholds[cls] = local_best
            best_score = local_score
            history.append(
                {
                    "pass": pass_i,
                    "class": int(cls),
                    "threshold": local_best,
                    "score": float(local_score),
                }
            )

    final_pred = apply_thresholds(probs, thresholds)
    return {
        "thresholds": thresholds.tolist(),
        "val_objective_score": float(
            _score_from_preds(y_true, final_pred, objective)
        ),
        "val_macro_f1": float(
            f1_score(y_true, final_pred, average="macro", zero_division=0)
        ),
        "objective": objective,
        "grid": np.asarray(grid).tolist(),
        "n_val": int(n),
        "n_classes": int(c),
        "n_passes": int(n_passes),
        "search_history_tail": history[-min(len(history), 20) :],
    }


def search_joint_macro_then_min(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray | None = None,
    n_passes: int = 2,
) -> dict[str, Any]:
    """
    Two-stage search for minority-aware thresholds:

    1. Optimise macro-F1 (primary selection metric under protocol).
    2. From that point, re-optimise for min_per_class_F1 **without** allowing
       macro-F1 to drop (accept only steps that keep macro ≥ stage-1 macro
       and improve min per-class F1).
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    n, c = probs.shape
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)

    stage1 = search_class_thresholds(
        y_true, probs, grid=grid, objective="macro_f1", n_passes=n_passes
    )
    thresholds = np.asarray(stage1["thresholds"], dtype=np.float64)
    macro_floor = float(stage1["val_macro_f1"])
    pred0 = apply_thresholds(probs, thresholds)
    min_f1 = _score_from_preds(y_true, pred0, "min_per_class_f1")

    for _ in range(int(n_passes)):
        for cls in range(c):
            local_best_t = float(thresholds[cls])
            local_best_min = min_f1
            for t in grid:
                trial = thresholds.copy()
                trial[cls] = float(t)
                pred = apply_thresholds(probs, trial)
                macro = _score_from_preds(y_true, pred, "macro_f1")
                if macro + 1e-12 < macro_floor:
                    continue
                mn = _score_from_preds(y_true, pred, "min_per_class_f1")
                if mn > local_best_min + 1e-12:
                    local_best_min = mn
                    local_best_t = float(t)
            thresholds[cls] = local_best_t
            min_f1 = local_best_min

    final_pred = apply_thresholds(probs, thresholds)
    return {
        "thresholds": thresholds.tolist(),
        "objective": "joint_macro_then_min_per_class_f1",
        "macro_floor": macro_floor,
        "val_macro_f1": float(
            f1_score(y_true, final_pred, average="macro", zero_division=0)
        ),
        "val_min_per_class_f1": float(
            _score_from_preds(y_true, final_pred, "min_per_class_f1")
        ),
        "stage1_macro_search": stage1,
        "grid": np.asarray(grid).tolist(),
        "n_val": int(n),
        "n_classes": int(c),
        "n_passes": int(n_passes),
    }
