#!/usr/bin/env python3
"""
Entry point for the corrected leakage-safe ToN-IoT minimal experiment.

Delegates to scripts/protocol/toniot_leakage_safe.py (checklist §4).

The pipeline:
  - 13-feature allowlist; blacklist label/type/attack/category
  - Stratified 60/20/20 split before fit; train-only encoders + MinMaxScaler
  - Categorical missing: fillna(UNKNOWN) then astype(str) (no literal "nan" class)
  - Numeric missing: fixed zero imputation (documented, not train-fitted stats)
  - RF (class_weight=balanced) + hard-label CNN (class-weighted CE, no KD/SMOTE)
  - Provenance: git_sha, source_dirty, command, checkpoint_sha256, environment
  - use_in_manuscript=true only when protocol succeeds and source_dirty=false

Does NOT retrain or touch BoT champion weights.

Usage:
  PYTHONPATH=. python scripts/run_toniot_corrected_simple.py
  PYTHONPATH=. python scripts/run_toniot_corrected_simple.py --epochs 40
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.toniot_leakage_safe import (  # noqa: E402
    OUT_DIR,
    PROTOCOL_ID,
    main,
)


if __name__ == "__main__":
    print(f"[run_toniot_corrected_simple] protocol={PROTOCOL_ID}")
    print(f"[run_toniot_corrected_simple] results dir: {OUT_DIR}")
    code = main()
    summary = OUT_DIR / "summary.json"
    table = OUT_DIR / "table.md"
    if summary.is_file():
        print(f"[run_toniot_corrected_simple] artifact: {summary}")
    if table.is_file():
        print(f"[run_toniot_corrected_simple] table: {table}")
    raise SystemExit(code)
