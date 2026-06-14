import os
from pathlib import Path

import yaml
import numpy as np

import torch
import torch.nn as nn


class CNNBiLSTMAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        model_cfg = config["model"]

        # ====================================================
        # Config
        # ====================================================

        self.input_features = model_cfg["input_features"]

        self.projection_dim = model_cfg["projection_dim"]

        self.reshape_channels = (
            model_cfg["reshape"][0]
        )

        self.reshape_length = (
            model_cfg["reshape"][1]
        )

        self.cnn_filters_1 = (
            model_cfg["cnn_filters_1"]
        )

        self.cnn_filters_2 = (
            model_cfg["cnn_filters_2"]
        )

        self.kernel_size = (
            model_cfg["cnn_kernel_size"]
        )

        self.pool_size = (
            model_cfg["pool_size"]
        )

        self.bilstm_units_1 = (
            model_cfg["bilstm_units_1"]
        )

        self.bilstm_units_2 = (
            model_cfg["bilstm_units_2"]
        )

        self.dense_units = (
            model_cfg["dense_units"]
        )

        self.dropout_rate = (
            model_cfg["dropout_rate"]
        )

        self.num_classes = (
            model_cfg["num_classes"]
        )

        self.attention_heads = (
            model_cfg["attention_heads"]
        )

        self.attention_dropout = (
            model_cfg["attention_dropout"]
        )

        # ====================================================
        # Input Projection
        # ====================================================

        self.input_projection = nn.Linear(

            self.input_features,

            self.projection_dim

        )

        # ====================================================
        # CNN Block 1
        # ====================================================

        self.conv1 = nn.Conv1d(

            in_channels=self.reshape_channels,

            out_channels=self.cnn_filters_1,

            kernel_size=self.kernel_size,

            padding=1

        )

        self.bn1 = nn.BatchNorm1d(

            self.cnn_filters_1

        )

        # ====================================================
        # CNN Block 2
        # ====================================================

        self.conv2 = nn.Conv1d(

            in_channels=self.cnn_filters_1,

            out_channels=self.cnn_filters_2,

            kernel_size=self.kernel_size,

            padding=1

        )

        self.bn2 = nn.BatchNorm1d(

            self.cnn_filters_2

        )

        self.pool = nn.MaxPool1d(

            kernel_size=self.pool_size

        )

        self.dropout = nn.Dropout(

            self.dropout_rate

        )

        self.relu = nn.ReLU()

        # ====================================================
        # BiLSTM #1
        # ====================================================

        self.bilstm1 = nn.LSTM(

            input_size=self.cnn_filters_2,

            hidden_size=self.bilstm_units_1,

            num_layers=1,

            batch_first=True,

            bidirectional=True

        )

        # ====================================================
        # BiLSTM #2
        # ====================================================

        self.bilstm2 = nn.LSTM(

            input_size=(
                self.bilstm_units_1 * 2
            ),

            hidden_size=self.bilstm_units_2,

            num_layers=1,

            batch_first=True,

            bidirectional=True

        )

        # ====================================================
        # Self-Attention
        # ====================================================

        self.attention = nn.MultiheadAttention(

            embed_dim=(
                self.bilstm_units_2 * 2
            ),

            num_heads=self.attention_heads,

            dropout=self.attention_dropout,

            batch_first=True

        )

        self.attention_norm = nn.LayerNorm(

            self.bilstm_units_2 * 2

        )

        # ====================================================
        # Dense Head
        # ====================================================

        self.fc1 = nn.Linear(

            self.bilstm_units_2 * 2,

            self.dense_units

        )

        self.fc2 = nn.Linear(

            self.dense_units,

            self.num_classes

        )

    # ====================================================
    # Forward
    # ====================================================

    def forward(self, x):

        # Input:
        # (batch, 10)

        x = self.input_projection(
            x
        )

        # (batch, 64)

        x = x.view(

            x.size(0),

            self.reshape_channels,

            self.reshape_length

        )

        # (batch, 2, 32)

        x = self.conv1(
            x
        )

        x = self.bn1(
            x
        )

        x = self.relu(
            x
        )

        x = self.conv2(
            x
        )

        x = self.bn2(
            x
        )

        x = self.relu(
            x
        )

        x = self.pool(
            x
        )

        x = self.dropout(
            x
        )

        # Conv output:
        # (batch, channels, length)

        x = x.permute(
            0,
            2,
            1
        )

        # (batch, length, channels)

        x, _ = self.bilstm1(
            x
        )

        x = self.dropout(
            x
        )

        x, _ = self.bilstm2(
            x
        )

        x = self.dropout(
            x
        )

        # ====================================================
        # Self-Attention Block
        # ====================================================

        attn_out, _ = self.attention(

            x,

            x,

            x,

            need_weights=False

        )

        x = self.attention_norm(

            x + attn_out

        )

        # Global Average Pooling

        x = torch.mean(

            x,

            dim=1

        )

        # (batch, 128)

        x = self.fc1(
            x
        )

        x = self.relu(
            x
        )

        x = self.dropout(
            x
        )

        logits = self.fc2(
            x
        )

        return logits

    # ====================================================
    # Parameter Count
    # ====================================================

    def count_parameters(self):

        return sum(

            p.numel()

            for p in self.parameters()

            if p.requires_grad

        )

    # ====================================================
    # Model Summary
    # ====================================================

    def get_model_summary(self):

        params = (
            self.count_parameters()
        )

        size_mb = (

            params * 4

        ) / (1024 ** 2)

        print(
            "\n" +
            "=" * 70
        )

        print(
            "MODEL SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"Total Parameters    : "
            f"{params:,}"
        )

        print(
            f"FP32 Size (MB)      : "
            f"{size_mb:.2f}"
        )

    # ====================================================
    # Export Weights
    # ====================================================

    def export_weights(self):

        config_path = (
            Path(__file__)
            .resolve()
            .parent
            .parent
            / "config"
            / "config.yaml"
        )

        with open(
            config_path,
            "r"
        ) as f:

            config = yaml.safe_load(f)

        export_dir = (
            Path(__file__)
            .resolve()
            .parent
            .parent
            / config["model"]["weight_export_path"]
        )

        export_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        print(
            "\nExporting Weights..."
        )

        for name, param in self.named_parameters():

            safe_name = (
                name.replace(".", "_")
            )

            fp32_path = (
                export_dir /
                f"{safe_name}_fp32.npy"
            )

            fp16_path = (
                export_dir /
                f"{safe_name}_fp16.npy"
            )

            weights = (

                param.detach()

                .cpu()

                .numpy()

            )

            np.save(

                fp32_path,

                weights.astype(
                    np.float32
                )

            )

            np.save(

                fp16_path,

                weights.astype(
                    np.float16
                )

            )

        print(
            f"Saved weights to: "
            f"{export_dir}"
        )


# ========================================================
# Config Loader
# ========================================================

def load_config():

    config_path = (
        Path(__file__)
        .resolve()
        .parent
        .parent
        / "config"
        / "config.yaml"
    )

    with open(
        config_path,
        "r"
    ) as f:

        config = yaml.safe_load(f)

    return config


# ========================================================
# Sanity Check
# ========================================================

if __name__ == "__main__":

    config = load_config()

    model = CNNBiLSTMAttention(
        config
    )

    dummy_input = torch.randn(

        4,

        config["model"]["input_features"]

    )

    output = model(
        dummy_input
    )

    print(
        f"Output Shape: "
        f"{output.shape}"
    )

    assert output.shape == (

        4,

        config["model"]["num_classes"]

    )

    model.get_model_summary()

    print(
        "\nSanity check passed."
    )