#!/usr/bin/env python3
"""
Verify the production champion checkpoint identity (MD5).

Expected MD5 (frozen): 80a90f7cc210276300eaa90173a5a385
Path: model/best_model_botiot_twostage.pth  (config/paths.py / config/champion.json)

Exit codes:
  0 — file present and MD5 matches
  1 — file missing
  2 — MD5 mismatch
  3 — other error

Usage:
  PYTHONPATH=. python scripts/verify_champion.py
  PYTHONPATH=. python scripts/verify_champion.py --path model/best_model_botiot_twostage.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.paths import CHAMPION_MD5, CHAMPION_PATH, CHAMPION_RELPATH
except ImportError:  # minimal fallback
    CHAMPION_RELPATH = "model/best_model_botiot_twostage.pth"
    CHAMPION_PATH = PROJECT_ROOT / CHAMPION_RELPATH
    CHAMPION_MD5 = "80a90f7cc210276300eaa90173a5a385"

EXPECTED_MD5 = CHAMPION_MD5


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        type=str,
        default=str(CHAMPION_PATH),
        help=f"Champion checkpoint (default: {CHAMPION_RELPATH})",
    )
    p.add_argument(
        "--expected-md5",
        type=str,
        default=EXPECTED_MD5,
        help="Expected MD5 hex digest",
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable result")
    args = p.parse_args()

    path = Path(args.path)
    if not path.is_file():
        msg = f"MISSING: champion checkpoint not found: {path}"
        if args.json:
            print(json.dumps({"ok": False, "error": "missing", "path": str(path)}))
        else:
            print(msg, file=sys.stderr)
        return 1

    digest = md5_file(path)
    ok = digest.lower() == args.expected_md5.lower()
    payload = {
        "ok": ok,
        "path": str(path.resolve()),
        "md5": digest,
        "expected_md5": args.expected_md5.lower(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"path:     {path.resolve()}")
        print(f"md5:      {digest}")
        print(f"expected: {args.expected_md5.lower()}")
        print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
