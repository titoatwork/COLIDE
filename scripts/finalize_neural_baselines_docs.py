#!/usr/bin/env python3
"""
Post-run helper for WP5b neural baselines.
Reads benchmarks/results/<tag>/summary.json + row JSONs, prints md5/manifest
block and ranking table for docs (does not invent numbers; disk-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="baselines_neural")
    args = p.parse_args()
    out_dir = PROJECT_ROOT / "benchmarks" / "results" / args.tag
    ckpt_dir = PROJECT_ROOT / "model" / args.tag
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        print(f"MISSING {summary_path}")
        return 1
    s = json.loads(summary_path.read_text())
    print("=== SUMMARY ===")
    print(f"n_success={s.get('n_success')} wall_sec={s.get('wall_sec')}")
    print(f"summary_md5={md5(summary_path)} bytes={summary_path.stat().st_size}")
    print("\n=== RANKING ===")
    for r in s.get("ranking_val_macro_f1") or []:
        print(
            f"{r.get('row')} {r.get('name')}: "
            f"macro={r.get('best_val_macro_f1')} "
            f"min={r.get('val_min_per_class_f1')} "
            f"theft={r.get('val_theft_f1')} "
            f"params={r.get('n_params')} "
            f"us/sample={r.get('per_sample_us')}"
        )
    print("\n=== MANIFEST ROWS (JSON) ===")
    for path in sorted(out_dir.glob("G*_seed*.json")):
        print(
            f"| `{path.relative_to(PROJECT_ROOT)}` | yes | `{md5(path)}` | "
            f"{path.stat().st_size} |"
        )
    print(
        f"| `{summary_path.relative_to(PROJECT_ROOT)}` | yes | `{md5(summary_path)}` | "
        f"{summary_path.stat().st_size} |"
    )
    print("\n=== CHECKPOINTS ===")
    for path in sorted(ckpt_dir.glob("G*_seed*.pth")):
        print(
            f"| `{path.relative_to(PROJECT_ROOT)}` | yes | `{md5(path)}` | "
            f"{path.stat().st_size} |"
        )
    champ = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
    if champ.is_file():
        print(f"\nchampion_md5={md5(champ)}")
    if s.get("g15_hpo_budget_note"):
        print("\n=== G15 NOTE ===")
        print(s["g15_hpo_budget_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
