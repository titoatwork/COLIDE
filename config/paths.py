"""
Central checkpoint and artifact paths for COLIDE.

Canonical production champion is the two-stage BoT-IoT CAD-CBA-v1 checkpoint.
Historical MD5 is retained for frozen identity checks; prefer SHA-256 for new
integrity work.
"""

from __future__ import annotations

from pathlib import Path

# Repository root (parent of config/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Canonical production champion (do not overwrite without backup + user OK)
CHAMPION_RELPATH = "model/best_model_botiot_twostage.pth"
CHAMPION_PATH = PROJECT_ROOT / CHAMPION_RELPATH

# Historical frozen identity (README / freeze card). MD5 retained for compatibility.
CHAMPION_MD5 = "80a90f7cc210276300eaa90173a5a385"

# Stage-1 KD init commonly used before FT (not the production champion)
STAGE1_KD_RELPATH = "model/best_model_botiot_distill_a0.6_T10.0_focal2.pth"
STAGE1_KD_PATH = PROJECT_ROOT / STAGE1_KD_RELPATH
