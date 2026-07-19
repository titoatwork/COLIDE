"""
Canonical BoT-IoT load protocol (Phase 1 freeze foundation).

Two official stages used in COLIDE:
  - stage_a_kd: stratified val split + SMOTE on train only (KD / distill path)
  - stage_b_ft: stratified val split + NO SMOTE (two-stage real-data fine-tune)

Test set is ALWAYS the official Testing CSV, scaled with the train-fit scaler.
Never use test for model selection (HPO / thresholds / early-stopping choice).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS = [
    "N_IN_Conn_P_DstIP",
    "N_IN_Conn_P_SrcIP",
    "drate",
    "max",
    "mean",
    "min",
    "seq",
    "srate",
    "state_number",
    "stddev",
]

# Historical KD SMOTE targets (train_distill.py / ensemble) — freeze for Stage A
DEFAULT_SMOTE_TARGETS = {
    "DDoS": 100_000,
    "DoS": 100_000,
    "Normal": 2_000,
    "Reconnaissance": 50_000,  # distill path; config resampling uses 65627 for preprocess_v2
    "Theft": 1_000,
}

StageName = Literal["stage_a_kd", "stage_b_ft"]


@dataclass
class BotIoTBundle:
    """Arrays ready for training / evaluation under the frozen protocol."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    stage: StageName
    seed: int
    protocol_id: str = "botiot_v1"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def summary(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "stage": self.stage,
            "seed": self.seed,
            "class_names": self.class_names,
            "n_train": int(len(self.y_train)),
            "n_val": int(len(self.y_val)),
            "n_test": int(len(self.y_test)),
            "n_features": int(self.X_train.shape[1]),
            "train_class_counts": {
                self.class_names[i]: int(c)
                for i, c in sorted(Counter(self.y_train).items())
            },
            "val_class_counts": {
                self.class_names[i]: int(c)
                for i, c in sorted(Counter(self.y_val).items())
            },
            "test_class_counts": {
                self.class_names[i]: int(c)
                for i, c in sorted(Counter(self.y_test).items())
            },
            "meta": self.meta,
        }


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _raw_paths(root: Path) -> tuple[Path, Path]:
    train = root / "data" / "raw" / "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
    test = root / "data" / "raw" / "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"
    if not train.is_file() or not test.is_file():
        raise FileNotFoundError(
            f"BoT-IoT CSVs missing under {root / 'data' / 'raw'}. "
            "Expected UNSW_2018_IoT_Botnet_Final_10_best_{Training,Testing}.csv"
        )
    return train, test


def load_botiot(
    stage: StageName = "stage_b_ft",
    seed: int = 42,
    val_size: float = 0.1,
    smote_scale: float = 1.0,
    smote_targets: dict[str, int] | None = None,
    project_root: str | Path | None = None,
    feature_columns: list[str] | None = None,
) -> BotIoTBundle:
    """
    Load BoT-IoT under the canonical protocol.

    Parameters
    ----------
    stage:
        stage_a_kd — SMOTE on train (KD).
        stage_b_ft — real counts only (two-stage fine-tune).
    seed:
        Controls train/val split and SMOTE.
    val_size:
        Fraction of official Training CSV held out as validation (stratified).
    smote_scale:
        Multiplier for DDoS/DoS SMOTE targets (stage_a only); 1.0 = historical.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    feats = feature_columns or FEATURE_COLUMNS
    train_csv, test_csv = _raw_paths(root)

    df_train = pd.read_csv(train_csv, sep=",")
    df_test = pd.read_csv(test_csv, sep=",")

    X_train_raw = df_train[feats].values.astype(np.float32)
    X_test_raw = df_test[feats].values.astype(np.float32)
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y_all_train = le.fit_transform(df_train["category"])
    y_test = le.transform(df_test["category"])
    class_names = list(le.classes_)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw,
        y_all_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_all_train,
    )

    meta: dict[str, Any] = {
        "val_size": val_size,
        "feature_columns": feats,
        "train_csv": str(train_csv.relative_to(root)),
        "test_csv": str(test_csv.relative_to(root)),
        "smote_applied": False,
    }

    if stage == "stage_a_kd":
        targets = dict(smote_targets or DEFAULT_SMOTE_TARGETS)
        strat: dict[int, int] = {}
        counts = Counter(y_tr)
        for i, name in enumerate(class_names):
            base = targets.get(name, int(counts[i]))
            if smote_scale != 1.0 and name in ("DDoS", "DoS"):
                base = int(base * smote_scale)
            strat[i] = base
        valid_strat = {k: v for k, v in strat.items() if v > counts[k]}
        smote = SMOTE(
            sampling_strategy=valid_strat,
            random_state=seed,
            k_neighbors=3,
        )
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        meta["smote_applied"] = True
        meta["smote_targets"] = {class_names[k]: int(v) for k, v in valid_strat.items()}
        meta["smote_scale"] = smote_scale
        meta["smote_k_neighbors"] = 3
    elif stage == "stage_b_ft":
        meta["smote_applied"] = False
        meta["note"] = "Real-data fine-tune path; no SMOTE (matches train_twostage.py)"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return BotIoTBundle(
        X_train=X_tr,
        y_train=y_tr.astype(np.int64),
        X_val=X_val,
        y_val=y_val.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        class_names=class_names,
        stage=stage,
        seed=seed,
        meta=meta,
    )
