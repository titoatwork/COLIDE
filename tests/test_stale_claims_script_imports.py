"""Smoke: stale-claim guard module imports and exposes expected API."""

from __future__ import annotations


def test_check_stale_claims_import():
    import scripts.check_stale_claims as mod

    assert hasattr(mod, "main")
    assert callable(mod.main)
    # Support either historical or current constant names
    has_patterns = (
        hasattr(mod, "FORBIDDEN_PATTERNS")
        or hasattr(mod, "HARD_FORBIDDEN")
        or hasattr(mod, "ACTIVE_FILES")
    )
    assert has_patterns, "expected FORBIDDEN_PATTERNS or HARD_FORBIDDEN/ACTIVE_FILES"


def test_check_stale_claims_main_callable():
    import scripts.check_stale_claims as mod

    # Module-level helpers used by the guard
    assert callable(getattr(mod, "check_file", None) or getattr(mod, "scan_file", None) or mod.main)
