"""Result envelope optional fields."""

from __future__ import annotations

from scripts.protocol.result_schema import make_result_envelope


def test_envelope_optional_fields():
    env = make_result_envelope(
        experiment_id="unit_test",
        protocol_id="botiot_v1",
        seed=42,
        valid=False,
        invalid_reason="unit test",
        use_in_manuscript=False,
        source_dirty=True,
        command="pytest tests/test_result_schema.py",
    )
    assert env["valid"] is False
    assert env["invalid_reason"] == "unit test"
    assert env["use_in_manuscript"] is False
    assert env["source_dirty"] is True
    assert env["command"] == "pytest tests/test_result_schema.py"
    assert "git_sha" in env
    assert "timestamp_utc" in env


def test_envelope_defaults_omit_optional():
    env = make_result_envelope(experiment_id="min")
    # valid not forced; source_dirty may be auto-filled from git
    assert env["experiment_id"] == "min"
    assert "invalid_reason" not in env
