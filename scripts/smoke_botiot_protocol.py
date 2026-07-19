#!/usr/bin/env python3
"""Smoke-test canonical BoT-IoT protocol (no training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot  # noqa: E402


def main() -> int:
    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for stage in ("stage_a_kd", "stage_b_ft"):
        b = load_botiot(stage=stage, seed=42)
        s = b.summary()
        reports[stage] = s
        print(
            f"{stage}: train={s['n_train']:,} val={s['n_val']:,} test={s['n_test']:,} "
            f"feats={s['n_features']} classes={s['class_names']}"
        )
        assert s["n_features"] == 10
        assert s["n_test"] > 0 and s["n_val"] > 0 and s["n_train"] > 0
        assert len(s["class_names"]) == 5

    # Stage A must SMOTE-expand train vs Stage B
    assert reports["stage_a_kd"]["n_train"] > reports["stage_b_ft"]["n_train"]
    # Test size identical across stages (same official test, same scaler fit policy per stage
    # — test *count* must match; values differ because scaler is train-fit)
    assert reports["stage_a_kd"]["n_test"] == reports["stage_b_ft"]["n_test"]

    path = out_dir / "botiot_protocol_smoke.json"
    with open(path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"OK wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
