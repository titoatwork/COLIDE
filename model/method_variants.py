"""
Bounded architecture variants for C* playlist (multi-scale CNN, gated fusion).

Used by scripts/run_bounded_cstar.py. Keeps CAD-CBA-v1 V3 dims where possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


class _Head(nn.Module):
    def __init__(self, in_dim: int, dense: int, n_cls: int, drop: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, dense)
        self.fc2 = nn.Linear(dense, n_cls)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc1(x))


class MultiScaleCNNBiLSTM(nn.Module):
    """
    C4 — Multi-scale temporal convolution: parallel Conv1d k∈{3,5,7}, concat channels,
    then BiLSTM + mean pool + head. Same reshape as V3 [2,32].
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.input_features = m["input_features"]
        self.projection_dim = m["projection_dim"]
        self.reshape_channels = m["reshape"][0]
        self.reshape_length = m["reshape"][1]
        self.dropout_rate = m["dropout_rate"]
        self.num_classes = m["num_classes"]
        f1 = m["cnn_filters_1"]
        f2 = m["cnn_filters_2"]

        self.input_projection = nn.Linear(self.input_features, self.projection_dim)
        # parallel multi-scale towers (pad to keep length)
        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.reshape_channels, f1 // 2, k, padding=k // 2),
                    nn.BatchNorm1d(f1 // 2),
                    nn.ReLU(),
                    nn.Conv1d(f1 // 2, f2 // 3, k, padding=k // 2),
                    nn.BatchNorm1d(f2 // 3),
                    nn.ReLU(),
                )
                for k in (3, 5, 7)
            ]
        )
        fused_ch = 3 * (f2 // 3)
        self.fuse = nn.Conv1d(fused_ch, f2, kernel_size=1)
        self.bn_fuse = nn.BatchNorm1d(f2)
        self.pool = nn.MaxPool1d(m["pool_size"])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.bilstm1 = nn.LSTM(
            input_size=f2,
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
            m["bilstm_units_2"] * 2, m["dense_units"], self.num_classes, self.dropout_rate
        )

    def forward(self, x: torch.Tensor, return_embed: bool = False):
        x = self.input_projection(x)
        x = x.view(x.size(0), self.reshape_channels, self.reshape_length)
        scales = [conv(x) for conv in self.scale_convs]
        # align lengths (padding may differ by 1 for even kernels on odd L)
        L = min(t.size(-1) for t in scales)
        scales = [t[..., :L] for t in scales]
        x = torch.cat(scales, dim=1)
        x = self.relu(self.bn_fuse(self.fuse(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        if return_embed:
            emb = self.head.embed(x)
            logits = self.head.fc2(self.head.drop(emb))
            return logits, emb
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GatedCNNBiLSTM(nn.Module):
    """
    C5 — Gated CNN–BiLSTM fusion: CNN path + projected BiLSTM path with sigmoid gate.
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.input_features = m["input_features"]
        self.projection_dim = m["projection_dim"]
        self.reshape_channels = m["reshape"][0]
        self.reshape_length = m["reshape"][1]
        self.dropout_rate = m["dropout_rate"]
        self.num_classes = m["num_classes"]
        f1, f2 = m["cnn_filters_1"], m["cnn_filters_2"]

        self.input_projection = nn.Linear(self.input_features, self.projection_dim)
        self.conv1 = nn.Conv1d(self.reshape_channels, f1, m["cnn_kernel_size"], padding=1)
        self.bn1 = nn.BatchNorm1d(f1)
        self.conv2 = nn.Conv1d(f1, f2, m["cnn_kernel_size"], padding=1)
        self.bn2 = nn.BatchNorm1d(f2)
        self.pool = nn.MaxPool1d(m["pool_size"])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.bilstm1 = nn.LSTM(
            f2, m["bilstm_units_1"], 1, batch_first=True, bidirectional=True
        )
        self.bilstm2 = nn.LSTM(
            m["bilstm_units_1"] * 2,
            m["bilstm_units_2"],
            1,
            batch_first=True,
            bidirectional=True,
        )
        lstm_dim = m["bilstm_units_2"] * 2
        # CNN global pool projects to lstm_dim; gate blends
        self.cnn_proj = nn.Linear(f2, lstm_dim)
        self.gate = nn.Linear(lstm_dim * 2, lstm_dim)
        self.head = _Head(lstm_dim, m["dense_units"], self.num_classes, self.dropout_rate)

    def forward(self, x: torch.Tensor, return_embed: bool = False):
        x = self.input_projection(x)
        x = x.view(x.size(0), self.reshape_channels, self.reshape_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        cnn_g = torch.mean(x, dim=-1)  # (B, f2)
        cnn_h = self.cnn_proj(cnn_g)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        lstm_h = torch.mean(x, dim=1)
        g = torch.sigmoid(self.gate(torch.cat([cnn_h, lstm_h], dim=-1)))
        fused = g * cnn_h + (1.0 - g) * lstm_h
        if return_embed:
            emb = self.head.embed(fused)
            logits = self.head.fc2(self.head.drop(emb))
            return logits, emb
        return self.head(fused)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class V3WithEmbed(nn.Module):
    """Thin wrapper around CNNBiLSTMAttention exposing penultimate embedding."""

    def __init__(self, config: dict):
        super().__init__()
        from cnn_bilstm_v3_attention import CNNBiLSTMAttention

        self.backbone = CNNBiLSTMAttention(config)

    def forward(self, x: torch.Tensor, return_embed: bool = False):
        # replicate forward until fc1 for embed
        b = self.backbone
        x = b.input_projection(x)
        x = x.view(x.size(0), b.reshape_channels, b.reshape_length)
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.relu(b.bn2(b.conv2(x)))
        x = b.pool(x)
        x = b.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = b.bilstm1(x)
        x = b.dropout(x)
        x, _ = b.bilstm2(x)
        x = b.dropout(x)
        attn_out, _ = b.attention(x, x, x, need_weights=False)
        x = b.attention_norm(x + attn_out)
        x = torch.mean(x, dim=1)
        emb = b.relu(b.fc1(x))
        logits = b.fc2(b.dropout(emb))
        if return_embed:
            return logits, emb
        return logits

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        # accept both raw V3 keys and backbone.* keys
        if any(k.startswith("backbone.") for k in state_dict):
            return super().load_state_dict(state_dict, strict=strict)
        remapped = {f"backbone.{k}": v for k, v in state_dict.items()}
        return super().load_state_dict(remapped, strict=strict)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        full = super().state_dict(*args, **kwargs)
        # also allow saving as raw V3 for interoperability
        return full

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_method_variant(name: str, config: dict, device: torch.device) -> nn.Module:
    n = name.lower().strip()
    if n in ("multi_scale", "multiscale", "c4"):
        model: nn.Module = MultiScaleCNNBiLSTM(config)
    elif n in ("gated", "gated_fusion", "c5"):
        model = GatedCNNBiLSTM(config)
    elif n in ("v3", "v3_embed", "cad_cba", "attn"):
        model = V3WithEmbed(config)
    else:
        raise ValueError(f"Unknown method variant: {name}")
    return model.to(device)
