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


def git_dirty(cwd: Path | None = None) -> bool | None:
    """True if working tree has uncommitted changes; None if git unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return None


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
