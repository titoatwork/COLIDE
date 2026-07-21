"""
Protocol ablation architecture variants for WP5a (CAD-CBA ladder).

All variants accept the same flat feature tensor (batch, input_features)
and return class logits (batch, num_classes). Dimensional choices match
config.yaml V3 CAD-CBA backbone where applicable so comparisons are fair.

Variants
--------
cnn_only          — projection + CNN stack + GAP + dense head
bilstm_only       — projection as sequence + BiLSTM stack + GAP + head
cnn_bilstm        — CNN + BiLSTM, mean-pool (no attention)
cnn_bilstm_attn   — full V3 CNN–BiLSTM–Attention (package backbone)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _mget(cfg: dict, key: str, default: Any = None) -> Any:
    model = cfg["model"] if "model" in cfg else cfg
    return model.get(key, default)


class _Head(nn.Module):
    def __init__(self, in_dim: int, dense_units: int, num_classes: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, dense_units)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        return self.fc2(x)


class CNNOnly(nn.Module):
    """A1 — CNN stack only (no recurrent / attention)."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.input_features = m["input_features"]
        self.projection_dim = m["projection_dim"]
        self.reshape_channels = m["reshape"][0]
        self.reshape_length = m["reshape"][1]
        self.dropout_rate = m["dropout_rate"]
        self.num_classes = m["num_classes"]

        self.input_projection = nn.Linear(self.input_features, self.projection_dim)
        self.conv1 = nn.Conv1d(
            self.reshape_channels, m["cnn_filters_1"], m["cnn_kernel_size"], padding=1
        )
        self.bn1 = nn.BatchNorm1d(m["cnn_filters_1"])
        self.conv2 = nn.Conv1d(
            m["cnn_filters_1"], m["cnn_filters_2"], m["cnn_kernel_size"], padding=1
        )
        self.bn2 = nn.BatchNorm1d(m["cnn_filters_2"])
        self.pool = nn.MaxPool1d(m["pool_size"])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.head = _Head(
            m["cnn_filters_2"], m["dense_units"], self.num_classes, self.dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.view(x.size(0), self.reshape_channels, self.reshape_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=2)  # GAP over length
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTMOnly(nn.Module):
    """A2 — BiLSTM on projected feature sequence (no CNN / attention)."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.input_features = m["input_features"]
        # Treat each scalar feature as a timestep after a shared embed.
        self.embed = nn.Linear(1, m["projection_dim"])
        self.dropout_rate = m["dropout_rate"]
        self.num_classes = m["num_classes"]
        self.bilstm1 = nn.LSTM(
            input_size=m["projection_dim"],
            hidden_size=m["bilstm_units_1"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm2 = nn.LSTM(
            input_size=m["bilstm_units_1"] * 2,
            hidden_size=m["bilstm_units_2"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.head = _Head(
            m["bilstm_units_2"] * 2,
            m["dense_units"],
            self.num_classes,
            self.dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F) -> (B, F, 1) -> (B, F, P)
        x = self.embed(x.unsqueeze(-1))
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNBiLSTM(nn.Module):
    """A3 — CNN + BiLSTM mean-pool (no self-attention). Matches V3 backbone dims."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.input_features = m["input_features"]
        self.projection_dim = m["projection_dim"]
        self.reshape_channels = m["reshape"][0]
        self.reshape_length = m["reshape"][1]
        self.dropout_rate = m["dropout_rate"]
        self.num_classes = m["num_classes"]

        self.input_projection = nn.Linear(self.input_features, self.projection_dim)
        self.conv1 = nn.Conv1d(
            self.reshape_channels, m["cnn_filters_1"], m["cnn_kernel_size"], padding=1
        )
        self.bn1 = nn.BatchNorm1d(m["cnn_filters_1"])
        self.conv2 = nn.Conv1d(
            m["cnn_filters_1"], m["cnn_filters_2"], m["cnn_kernel_size"], padding=1
        )
        self.bn2 = nn.BatchNorm1d(m["cnn_filters_2"])
        self.pool = nn.MaxPool1d(m["pool_size"])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.bilstm1 = nn.LSTM(
            input_size=m["cnn_filters_2"],
            hidden_size=m["bilstm_units_1"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm2 = nn.LSTM(
            input_size=m["bilstm_units_1"] * 2,
            hidden_size=m["bilstm_units_2"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = _Head(
            m["bilstm_units_2"] * 2,
            m["dense_units"],
            self.num_classes,
            self.dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.view(x.size(0), self.reshape_channels, self.reshape_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_ablation_model(variant: str, config: dict, device: torch.device) -> nn.Module:
    """
    variant:
      cnn_only | bilstm_only | cnn_bilstm | cnn_bilstm_attn
    """
    v = variant.lower().strip()
    if v == "cnn_only":
        model = CNNOnly(config)
    elif v == "bilstm_only":
        model = BiLSTMOnly(config)
    elif v in ("cnn_bilstm", "cnn_bilstm_no_attn"):
        model = CNNBiLSTM(config)
    elif v in ("cnn_bilstm_attn", "v3", "full_backbone"):
        # Import sibling module under model/
        import sys
        from pathlib import Path

        model_dir = Path(__file__).resolve().parent
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from cnn_bilstm_v3_attention import CNNBiLSTMAttention

        model = CNNBiLSTMAttention(config)
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
    return model.to(device)


VARIANT_CHOICES = (
    "cnn_only",
    "bilstm_only",
    "cnn_bilstm",
    "cnn_bilstm_attn",
)
