"""Canonical data and metric protocols for COLIDE experiments (Phase 1+)."""

from .botiot import (
    FEATURE_COLUMNS,
    BotIoTBundle,
    load_botiot,
    load_config,
)
from .metrics import compute_classification_metrics

__all__ = [
    "FEATURE_COLUMNS",
    "BotIoTBundle",
    "load_botiot",
    "load_config",
    "compute_classification_metrics",
]
