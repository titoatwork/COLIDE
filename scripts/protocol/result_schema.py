"""Minimal result envelope for new experiments (reproducibility)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_sha(cwd: Path | None = None) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# Paths that may change during a claim-eligible experiment without implying
# *source* dirty (generated results, weights, logs, OS junk).
_SOURCE_DIRTY_IGNORE_PREFIXES = (
    "benchmarks/results/",
    "logs/",
    "model/",  # checkpoints; code lives under model/*.py still counted if tracked
    ".pytest_cache/",
    "__pycache__/",
)
_SOURCE_DIRTY_IGNORE_SUFFIXES = (
    ".pth",
    ".onnx",
    ".npz",
    ".log",
    ".bak",
    ":Zone.Identifier",
)
_SOURCE_DIRTY_ALWAYS_IGNORE_UNTRACKED = True


def _is_ignored_for_source_dirty(path: str) -> bool:
    p = path.replace("\\", "/").lstrip("./")
    # model/*.py is source; model/**/*.pth is not
    if p.startswith("model/") and p.endswith(".py"):
        return False
    if p.startswith("model/"):
        return True
    for pref in _SOURCE_DIRTY_IGNORE_PREFIXES:
        if p.startswith(pref):
            return True
    for suf in _SOURCE_DIRTY_IGNORE_SUFFIXES:
        if p.endswith(suf) or suf in p:
            return True
    if p.endswith(".bak_before_review") or "Zone.Identifier" in p:
        return True
    return False


def git_dirty(cwd: Path | None = None) -> bool | None:
    """True if *source* working tree has uncommitted changes; None if git unavailable.

    Ignores untracked noise (logs, OS junk) and generated result/checkpoint
    paths so experiment runs can emit claim-eligible JSON without self-dirtying
    the tree. Tracked edits under scripts/, inference/, docs/, tests/, README,
    and model/*.py still count as dirty.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "-uall"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode(errors="replace")
    except Exception:
        return None
    for line in out.splitlines():
        if not line.strip():
            continue
        # porcelain: XY path  OR  XY origin -> path
        status = line[:2]
        rest = line[3:].strip() if len(line) > 3 else ""
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[-1]
        # untracked (??) under ignore prefixes: skip
        if status == "??" and _SOURCE_DIRTY_ALWAYS_IGNORE_UNTRACKED:
            if _is_ignored_for_source_dirty(rest):
                continue
            # untracked source files still dirty
            if not _is_ignored_for_source_dirty(rest):
                # only treat as dirty if looks like project source
                if rest.startswith(
                    ("scripts/", "inference/", "docs/", "tests/", "config/")
                ) or rest in ("README.md", "LICENSE", ".gitignore"):
                    return True
            continue
        if _is_ignored_for_source_dirty(rest):
            continue
        # any other modified/added/deleted tracked path
        return True
    return False


def make_result_envelope(
    *,
    experiment_id: str,
    protocol_id: str = "botiot_v1",
    stage: str | None = None,
    seed: int | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    project_root: Path | None = None,
    valid: bool | None = None,
    invalid_reason: str | None = None,
    use_in_manuscript: bool | None = None,
    source_dirty: bool | None = None,
    command: str | list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a standard result dict.

    Optional provenance / claim-gate fields (checklist §19):
      valid, invalid_reason, use_in_manuscript, source_dirty, command
    When ``source_dirty`` is omitted, it is inferred from ``git status`` if possible.
    """
    env: dict[str, Any] = {
        "experiment_id": experiment_id,
        "protocol_id": protocol_id,
        "stage": stage,
        "seed": seed,
        "git_sha": git_sha(project_root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config or {},
        "metrics": metrics or {},
    }
    if valid is not None:
        env["valid"] = bool(valid)
    if invalid_reason is not None:
        env["invalid_reason"] = invalid_reason
    if use_in_manuscript is not None:
        env["use_in_manuscript"] = bool(use_in_manuscript)
    if source_dirty is not None:
        env["source_dirty"] = bool(source_dirty)
    else:
        dirty = git_dirty(project_root)
        if dirty is not None:
            env["source_dirty"] = dirty
    if command is not None:
        env["command"] = command
    if extra:
        env["extra"] = extra
    return env
