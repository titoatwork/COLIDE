"""Unit tests for ToN-IoT leakage-safe schema helpers and categorical NaN handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.protocol.toniot_leakage_safe import (
    UNKNOWN_CAT_TOKEN,
    TrainOnlyCatEncoder,
    series_to_cat,
)
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


# ---------------------------------------------------------------------------
# Categorical missing-value order (fillna before astype(str))
# ---------------------------------------------------------------------------


def test_series_to_cat_maps_nan_to_unknown_token():
    """Real NaN must become UNKNOWN_CAT_TOKEN, not the literal string 'nan'."""
    s = pd.Series(["tcp", np.nan, "udp", None, "tcp"])
    out = series_to_cat(s)
    assert list(out) == ["tcp", UNKNOWN_CAT_TOKEN, "udp", UNKNOWN_CAT_TOKEN, "tcp"]
    assert "nan" not in set(out)
    assert UNKNOWN_CAT_TOKEN in set(out)


def test_series_to_cat_normalizes_string_nan_edge_cases():
    """Literal 'nan'/'None'/'NaN' from prior str conversion map to unknown."""
    s = pd.Series(["http", "nan", "None", "NaN", "dns"])
    out = series_to_cat(s)
    assert list(out) == [
        "http",
        UNKNOWN_CAT_TOKEN,
        UNKNOWN_CAT_TOKEN,
        UNKNOWN_CAT_TOKEN,
        "dns",
    ]
    assert "nan" not in set(out)
    assert "None" not in set(out)
    assert "NaN" not in set(out)


def test_train_only_cat_encoder_nan_not_separate_class():
    """Fit/transform must not invent a 'nan' string class for missing values."""
    train = pd.Series(["tcp", "udp", np.nan, "tcp"])
    enc = TrainOnlyCatEncoder()
    enc.fit(train)
    assert "nan" not in enc.classes_
    assert UNKNOWN_CAT_TOKEN in enc.classes_
    # Missing and unseen map to unknown path (UNK gets a train code; truly unseen → unknown_code)
    codes = enc.transform(pd.Series([np.nan, "tcp", "sctp", None]))
    unk_code = enc.map_[UNKNOWN_CAT_TOKEN]
    assert codes[0] == unk_code  # NaN → UNK class code seen in train
    assert codes[1] == enc.map_["tcp"]
    assert codes[2] == enc.unknown_code  # unseen category
    assert codes[3] == unk_code  # None → UNK
