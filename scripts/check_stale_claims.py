#!/usr/bin/env python3
"""Fail if forbidden stale claims appear in active documentation surfaces.

Exit 0: clean
Exit 1: one or more violations

A line is exempt when it contains any of:
  HISTORICAL, INVALID, tombstone, archive
(case-insensitive), or (for 0.9790 only) historical / legacy nearby.

Active files (Phase 1):
  README.md
  docs/CLAIM_MAP_PREWRITE.md
  docs/PRE_MANUSCRIPT_CLOSURE.md
  docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_FILES = [
    ROOT / "README.md",
    ROOT / "docs" / "CLAIM_MAP_PREWRITE.md",
    ROOT / "docs" / "PRE_MANUSCRIPT_CLOSURE.md",
    ROOT / "docs" / "manuscript" / "CAD_CBA_v1_MANUSCRIPT.md",
]

EXEMPT_MARKERS = re.compile(
    r"historical|invalid|tombstone|archive|legacy|withdrawn|quarantine|forbidden",
    re.IGNORECASE,
)

# Hard-forbidden numeric / improvement claim strings (ToN clean invalid path).
HARD_FORBIDDEN = [
    ("0.9526", re.compile(r"0\.9526")),
    ("0.9851", re.compile(r"0\.9851")),
    ("+15.4%", re.compile(r"\+15\.4\s*%")),
    ("15.4% improvement claim", re.compile(r"(?<!\+)\b15\.4\s*%")),
]

# Bare principal use of 0.9790 without historical/legacy context (best effort).
F9790 = re.compile(r"0\.9790")


def line_exempt(line: str) -> bool:
    return bool(EXEMPT_MARKERS.search(line))


def check_file(path: Path) -> list[str]:
    violations: list[str] = []
    if not path.is_file():
        violations.append(f"MISSING active file: {path.relative_to(ROOT)}")
        return violations

    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line_exempt(line):
            # Still forbid improvement claim if it presents +15.4% as a live win
            # without invalidation language — exempt markers already cover that.
            continue

        for label, pattern in HARD_FORBIDDEN:
            if pattern.search(line):
                # 15.4% alone may appear in other contexts; require improvement-ish
                # framing for the bare 15.4% pattern (not +15.4% which is always bad).
                if label == "15.4% improvement claim":
                    if not re.search(
                        r"(improv|lift|gain|increase|better|\+|CNN|RF|ToN|clean)",
                        line,
                        re.IGNORECASE,
                    ):
                        continue
                rel = path.relative_to(ROOT)
                violations.append(
                    f"{rel}:{lineno}: forbidden '{label}': {line.strip()[:160]}"
                )

        if F9790.search(line) and not line_exempt(line):
            # Window: same line must not already be exempt; require nearby
            # historical/legacy within ±1 line for best-effort principal check.
            # (Already not exempt on this line.)
            rel = path.relative_to(ROOT)
            violations.append(
                f"{rel}:{lineno}: bare 0.9790 without historical/legacy/INVALID "
                f"context: {line.strip()[:160]}"
            )

    return violations


def main() -> int:
    all_v: list[str] = []
    for path in ACTIVE_FILES:
        all_v.extend(check_file(path))

    if all_v:
        print("check_stale_claims: FAILED", file=sys.stderr)
        for v in all_v:
            print(f"  {v}", file=sys.stderr)
        print(f"\n{len(all_v)} violation(s).", file=sys.stderr)
        return 1

    print("check_stale_claims: OK (no forbidden strings in active surfaces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
