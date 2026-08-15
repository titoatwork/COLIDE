#!/usr/bin/env python3
"""Fail if forbidden stale claims appear in active documentation surfaces.

Exit 0: clean
Exit 1: one or more violations

A line is exempt when it contains any of:
  HISTORICAL, INVALID, tombstone, archive
(case-insensitive), or (for 0.9790 only) historical / legacy nearby.

Active files (publication surfaces):
  README.md
  docs/CLAIM_MAP_PREWRITE.md
  docs/PRE_MANUSCRIPT_CLOSURE.md
  docs/paper_text_blocks.md
  docs/manuscript/**/* (all files under manuscript/)
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
    ROOT / "docs" / "paper_text_blocks.md",
    ROOT / "docs" / "manuscript" / "CAD_CBA_v1_MANUSCRIPT.md",
]


def _expand_active_files() -> list[Path]:
    """Include every file under docs/manuscript/ in addition to explicit paths."""
    seen: set[Path] = set()
    out: list[Path] = []
    for p in ACTIVE_FILES:
        if p not in seen:
            seen.add(p)
            out.append(p)
    ms_dir = ROOT / "docs" / "manuscript"
    if ms_dir.is_dir():
        for p in sorted(ms_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".md", ".txt", ".rst", ".tex"}:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    return out

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

# Risky: treat DICC/server B3 latency as post_fix without historical/pre_fix caveat.
# Option B (2026-08-15): active post_fix DICC B3 speed claims forbidden without new SUCCESS tree.
# See docs/B3_SERVER_LATENCY_DECISION.md.
B3_POSTFIX_EXEMPT = re.compile(
    r"historical|pre[_-]?fix|"
    r"not\s+\**post[_-]?fix|"  # allow markdown **post_fix**
    r"until\s+rebench|without\s+a\s+new|forbidden|option\s+b|"
    r"remains\s+open|\bOPEN\b|pending|"
    r"not\s+a\s+\**post[_-]?fix|dropped\s+from\s+active|"
    r"or\s+drop|drop\s+comparative|do\s+not\s+invent|"
    r"≠\s*DICC|!=\s*DICC|local\s+only|per-block\s+\*\*local\*\*|"
    r"local\s+(production-weight\s+)?parity|kernel_status|"
    r"latency\s+decision|rebench\s+path",
    re.IGNORECASE,
)

# Line must mention B3 (or Block[- ]3) AND post_fix AND a server/DICC cue.
RISKY_B3_POSTFIX_SERVER = re.compile(
    r"(?is)"
    r"(?=.*\b(?:B3|Block[\s-]?3)\b)"
    r"(?=.*post[_-]?fix)"
    r"(?=.*\b(?:DICC|server|V100S?|A100|multi-session\s+latency|SUCCESS\s+tree)\b)"
)


def line_exempt(line: str) -> bool:
    return bool(EXEMPT_MARKERS.search(line))


def b3_postfix_risky(line: str) -> bool:
    """True if line looks like a live post_fix DICC/server B3 latency claim."""
    if not RISKY_B3_POSTFIX_SERVER.search(line):
        return False
    if B3_POSTFIX_EXEMPT.search(line):
        return False
    if EXEMPT_MARKERS.search(line):
        return False
    return True


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
            # B3 post_fix risk still checked below only when not exempt.
            pass
        else:
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

            if F9790.search(line):
                # Window: same line must not already be exempt; require nearby
                # historical/legacy within ±1 line for best-effort principal check.
                # (Already not exempt on this line.)
                rel = path.relative_to(ROOT)
                violations.append(
                    f"{rel}:{lineno}: bare 0.9790 without historical/legacy/INVALID "
                    f"context: {line.strip()[:160]}"
                )

        # B3 + post_fix + server/DICC without historical/pre_fix caveat (Option B).
        if b3_postfix_risky(line):
            rel = path.relative_to(ROOT)
            violations.append(
                f"{rel}:{lineno}: risky post_fix DICC/server B3 claim without "
                f"historical/pre_fix caveat: {line.strip()[:160]}"
            )

    return violations


def main() -> int:
    all_v: list[str] = []
    for path in _expand_active_files():
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
