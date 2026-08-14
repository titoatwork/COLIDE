#!/usr/bin/env python3
"""
Entry point for the corrected leakage-safe ToN-IoT minimal experiment.

Delegates to scripts/protocol/toniot_leakage_safe.py (checklist §4).

Usage:
  PYTHONPATH=. python scripts/run_toniot_corrected_simple.py
  PYTHONPATH=. python scripts/run_toniot_corrected_simple.py --epochs 40
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.toniot_leakage_safe import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
