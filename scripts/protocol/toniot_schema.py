"""
ToN-IoT leakage-safe feature schema utilities (pure functions).

Target-derived columns must never appear in the feature matrix when predicting
``type`` (or any multiclass target). Historical ``train_toniot_clean`` kept
numeric ``label`` in X while predicting ``type`` — these helpers exist so
loaders and unit tests can reject that class of leakage.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# Columns that encode the prediction target or a direct function of it.
# Checked after normalize_column_name.
TARGET_BLACKLIST: frozenset[str] = frozenset(
    {
        "label",
        "type",
        "attack",
        "category",
        "class",
        "target",
        "y",
        "gt",
        "ground_truth",
        "groundtruth",
        "attack_cat",
        "attackcat",
        "attack_type",
        "attacktype",
        "subclass",
        "sub_class",
    }
)


def normalize_column_name(name: str) -> str:
    """
    Normalize a column name for blacklist / allowlist comparison.

    - strip whitespace
    - lowercase
    - replace non-alphanumeric runs with a single underscore
    - strip leading/trailing underscores
    """
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


def normalize_columns(columns: Iterable[str]) -> list[str]:
    """Return normalized names preserving order (duplicates kept for diagnostics)."""
    return [normalize_column_name(c) for c in columns]


def is_blacklisted(name: str) -> bool:
    """True if the (possibly raw) column name normalizes into TARGET_BLACKLIST."""
    return normalize_column_name(name) in TARGET_BLACKLIST


def blacklisted_in_features(feature_columns: Sequence[str]) -> list[str]:
    """
    Return the subset of *feature_columns* whose normalized form is blacklisted.

    Original spellings are returned (not the normalized forms) so callers can
    report exact column names in error messages.
    """
    hits: list[str] = []
    for col in feature_columns:
        if is_blacklisted(col):
            hits.append(col)
    return hits


def assert_features_leakage_free(
    feature_columns: Sequence[str],
    *,
    target_column: str | None = "type",
) -> list[str]:
    """
    Fatal check: feature list must not intersect the target blacklist.

    Also rejects the explicit ``target_column`` if provided (even if someone
    removes it from the blacklist by mistake).

    Returns the ordered feature list on success.
    Raises ``ValueError`` if any blacklisted / target column is present.
    """
    hits = blacklisted_in_features(feature_columns)
    if target_column is not None:
        tnorm = normalize_column_name(target_column)
        for col in feature_columns:
            if normalize_column_name(col) == tnorm and col not in hits:
                hits.append(col)
    if hits:
        raise ValueError(
            "Target-derived column(s) present in feature list (label leakage): "
            f"{hits}. Blacklist (normalized): {sorted(TARGET_BLACKLIST)}"
        )
    return list(feature_columns)
