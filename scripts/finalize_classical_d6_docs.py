#!/usr/bin/env python3
"""
Rebuild classical handoff + print D6/G2/G5 headlines for docs (no invent).
Run after classical + stratified_batch jobs finish.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"


def main() -> int:
    # Rebuild handoff via classical script
    from scripts.run_classical_baselines import rebuild_handoff

    out = ROOT / "benchmarks" / "results" / "baselines_classical"
    hp = rebuild_handoff(out, 42)
    handoff = json.loads(hp.read_text())
    print("=== classical summary_handoff ===")
    for r in handoff.get("rows", []):
        print(
            f"  {r['model']}: macro={r.get('val_macro_f1')} "
            f"min={r.get('val_min_per_class_f1')} theft={r.get('theft_f1')}"
        )
    print(f"  md5 {hp.name}={md5(hp)}")

    for name in ["svm", "lgbm", "lr", "rf", "xgb"]:
        p = out / f"{name}_seed42.json"
        if p.is_file():
            env = json.loads(p.read_text())
            val = (env.get("metrics") or {}).get("val") or {}
            print(
                f"  file {name}: macro={val.get('macro_f1')} "
                f"md5={md5(p)} bytes={p.stat().st_size}"
            )

    d6 = ROOT / "benchmarks" / "results" / "stratified_batch" / "summary.json"
    print("=== D6 stratified_batch ===")
    if d6.is_file():
        s = json.loads(d6.read_text())
        print(f"  decision={s.get('decision')} delta={s.get('delta_stratified_minus_shuffle')}")
        print(f"  ranking={s.get('ranking_val_macro_f1')}")
        print(f"  note={s.get('decision_note')}")
        print(f"  md5={md5(d6)} wall={s.get('wall_sec')}")
    else:
        print("  summary MISSING")

    champ = ROOT / "model" / "best_model_botiot_twostage.pth"
    print(f"champion md5={md5(champ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
