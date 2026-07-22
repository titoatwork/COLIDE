"""
Protocol-fair neural baseline architectures for WP5b (G6–G12).

All variants accept flat features (batch, input_features) and return logits
(batch, num_classes). Designed for botiot_v1 / stage_b_ft fair comparison
against CAD-CBA-v1 — not for KD weight transfer.

Variants
--------
mlp              — G6 multi-layer perceptron (~matched capacity band)
cnn1d            — G7 1D-CNN stack + GAP
lstm             — G8 unidirectional LSTM on feature sequence
bilstm           — G9 bidirectional LSTM on feature sequence
cnn_lstm         — G10 CNN + unidirectional LSTM
cnn_bilstm       — G11 CNN + BiLSTM mean-pool (arch baseline; no attention)
transformer      — G12 lightweight temporal transformer encoder
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


class MLPBaseline(nn.Module):
    """G6 — deep MLP on flat features (historical ablation_mlp capacity band)."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        d_in = m["input_features"]
        n_cls = m["num_classes"]
        drop = float(m["dropout_rate"])
        # ~530k band like historical MLP ablation (512-512-256)
        self.net = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN1DBaseline(nn.Module):
    """G7 — 1D-CNN only (project → reshape → conv stack → GAP → head)."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.input_features = m["input_features"]
        self.projection_dim = m["projection_dim"]
        self.reshape_channels = m["reshape"][0]
        self.reshape_length = m["reshape"][1]
        drop = float(m["dropout_rate"])

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
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.head = _Head(
            m["cnn_filters_2"], m["dense_units"], m["num_classes"], drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.view(x.size(0), self.reshape_channels, self.reshape_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=2)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMBaseline(nn.Module):
    """G8 — unidirectional 2-layer LSTM on per-feature embeddings."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        drop = float(m["dropout_rate"])
        self.embed = nn.Linear(1, m["projection_dim"])
        self.lstm1 = nn.LSTM(
            input_size=m["projection_dim"],
            hidden_size=m["bilstm_units_1"],
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=m["bilstm_units_1"],
            hidden_size=m["bilstm_units_2"],
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(drop)
        self.head = _Head(
            m["bilstm_units_2"], m["dense_units"], m["num_classes"], drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.unsqueeze(-1))  # (B, F, P)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTMBaseline(nn.Module):
    """G9 — bidirectional 2-layer LSTM on per-feature embeddings."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        drop = float(m["dropout_rate"])
        self.embed = nn.Linear(1, m["projection_dim"])
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
        self.dropout = nn.Dropout(drop)
        self.head = _Head(
            m["bilstm_units_2"] * 2, m["dense_units"], m["num_classes"], drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.unsqueeze(-1))
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNLSTMBaseline(nn.Module):
    """G10 — CNN front-end + unidirectional LSTM (no attention)."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.input_features = m["input_features"]
        self.projection_dim = m["projection_dim"]
        self.reshape_channels = m["reshape"][0]
        self.reshape_length = m["reshape"][1]
        drop = float(m["dropout_rate"])

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
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(
            input_size=m["cnn_filters_2"],
            hidden_size=m["bilstm_units_1"],
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=m["bilstm_units_1"],
            hidden_size=m["bilstm_units_2"],
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = _Head(
            m["bilstm_units_2"], m["dense_units"], m["num_classes"], drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.view(x.size(0), self.reshape_channels, self.reshape_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNBiLSTMBaseline(nn.Module):
    """G11 — CNN + BiLSTM mean-pool (protocol arch baseline; no attention)."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.input_features = m["input_features"]
        self.projection_dim = m["projection_dim"]
        self.reshape_channels = m["reshape"][0]
        self.reshape_length = m["reshape"][1]
        drop = float(m["dropout_rate"])

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
        self.dropout = nn.Dropout(drop)
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
            m["bilstm_units_2"] * 2, m["dense_units"], m["num_classes"], drop
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


class TemporalTransformerBaseline(nn.Module):
    """G12 — lightweight transformer encoder on projected feature tokens."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        d_model = int(m.get("transformer_d_model", m["projection_dim"]))  # default 64
        nhead = int(m.get("transformer_nhead", m.get("attention_heads", 4)))
        nlayers = int(m.get("transformer_layers", 2))
        dim_ff = int(m.get("transformer_ff", 256))
        drop = float(m["dropout_rate"])
        # ensure d_model divisible by nhead
        if d_model % nhead != 0:
            d_model = nhead * max(1, d_model // nhead)

        self.d_model = d_model
        self.token_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, m["input_features"], d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(drop)
        self.head = _Head(d_model, m["dense_units"], m["num_classes"], drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F) -> (B, F, 1) -> (B, F, d)
        x = self.token_proj(x.unsqueeze(-1))
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_neural_baseline(variant: str, config: dict, device: torch.device) -> nn.Module:
    v = variant.lower().strip()
    table = {
        "mlp": MLPBaseline,
        "cnn1d": CNN1DBaseline,
        "cnn_1d": CNN1DBaseline,
        "1d_cnn": CNN1DBaseline,
        "lstm": LSTMBaseline,
        "bilstm": BiLSTMBaseline,
        "cnn_lstm": CNNLSTMBaseline,
        "cnn_bilstm": CNNBiLSTMBaseline,
        "transformer": TemporalTransformerBaseline,
        "temporal_transformer": TemporalTransformerBaseline,
    }
    if v not in table:
        raise ValueError(f"Unknown neural baseline variant: {variant}")
    return table[v](config).to(device)


VARIANT_CHOICES = (
    "mlp",
    "cnn1d",
    "lstm",
    "bilstm",
    "cnn_lstm",
    "cnn_bilstm",
    "transformer",
)
