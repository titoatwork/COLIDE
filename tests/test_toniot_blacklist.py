"""Unit tests for ToN-IoT leakage-safe schema helpers."""

from __future__ import annotations

import pytest

from scripts.protocol.toniot_schema import (
    TARGET_BLACKLIST,
    assert_features_leakage_free,
    blacklisted_in_features,
    is_blacklisted,
    normalize_column_name,
    normalize_columns,
)


def test_normalize_column_name_basic():
    assert normalize_column_name("  Label ") == "label"
    assert normalize_column_name("TYPE") == "type"
    assert normalize_column_name("Attack-Cat") == "attack_cat"
    assert normalize_column_name("ground.truth") == "ground_truth"
    assert normalize_column_name("src_bytes") == "src_bytes"


def test_normalize_columns_order():
    assert normalize_columns(["A", "b_c", " Label "]) == ["a", "b_c", "label"]


def test_blacklist_capitalization_variants():
    for raw in ("label", "Label", "LABEL", " type ", "Category", "ATTACK", "attack_type"):
        assert is_blacklisted(raw), raw


def test_blacklisted_in_features_detects_label():
    feats = ["duration", "src_bytes", "label", "dst_port"]
    hits = blacklisted_in_features(feats)
    assert hits == ["label"]


def test_assert_rejects_label_in_features():
    with pytest.raises(ValueError, match="label leakage"):
        assert_features_leakage_free(["duration", "label", "src_bytes"], target_column="type")


def test_assert_rejects_target_column():
    with pytest.raises(ValueError, match="label leakage"):
        assert_features_leakage_free(["duration", "type"], target_column="type")


def test_assert_accepts_clean_allowlist():
    feats = ["duration", "src_bytes", "dst_bytes", "proto"]
    out = assert_features_leakage_free(feats, target_column="type")
    assert out == feats


def test_blacklist_contains_required_names():
    for name in ("label", "type", "attack", "category"):
        assert name in TARGET_BLACKLIST
