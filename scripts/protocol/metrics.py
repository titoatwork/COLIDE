"""Shared classification metrics (val always; test only when explicitly allowed)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable metrics dict for BoT-IoT-style multi-class tasks."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = int(max(y_true.max(), y_pred.max()) + 1) if len(y_true) else 0
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    per_p, per_r, per_f1, per_s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
            "support": int(per_s[i]),
        }

    minority_f1s = [per_class[n]["f1"] for n in class_names]
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "min_per_class_f1": float(min(minority_f1s)) if minority_f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }
    # Explicit rare-class hooks used as HPO secondary objectives
    for rare in ("Theft", "Normal"):
        if rare in per_class:
            out[f"{rare.lower()}_f1"] = per_class[rare]["f1"]
            out[f"{rare.lower()}_recall"] = per_class[rare]["recall"]
    return out
