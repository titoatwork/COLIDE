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
) -> dict[str, Any]:
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
    if extra:
        env["extra"] = extra
    return env
