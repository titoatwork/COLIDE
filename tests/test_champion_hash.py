"""Champion checkpoint MD5 identity check (skip if file absent)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
EXPECTED_MD5 = "80a90f7cc210276300eaa90173a5a385"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.skipif(not CHAMPION.is_file(), reason="champion checkpoint not present")
def test_champion_md5():
    digest = _md5(CHAMPION)
    assert digest == EXPECTED_MD5, f"got {digest}, expected {EXPECTED_MD5}"


def test_paths_module_exports():
    from config.paths import CHAMPION_MD5, CHAMPION_PATH, CHAMPION_RELPATH

    assert CHAMPION_MD5 == EXPECTED_MD5
    assert CHAMPION_RELPATH.endswith("best_model_botiot_twostage.pth")
    assert CHAMPION_PATH.name == "best_model_botiot_twostage.pth"
