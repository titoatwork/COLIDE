"""
COLIDE
CNN-BiLSTM Architecture

Input:
    (batch_size, window_size, num_features)

Example:
    (128, 20, 10)

Author:
    COLIDE Data Science Team
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM Hybrid Network
    """

    def __init__(self, config):

        super().__init__()

        model_cfg = config["model"]

        self.input_features = model_cfg["input_features"]

        self.window_size = model_cfg["window_size"]

        self.cnn_filters_1 = model_cfg["cnn_filters_1"]
        self.cnn_filters_2 = model_cfg["cnn_filters_2"]

        self.cnn_kernel_size = model_cfg["cnn_kernel_size"]

        self.pool_size = model_cfg["pool_size"]

        self.bilstm_units_1 = model_cfg["bilstm_units_1"]
        self.bilstm_units_2 = model_cfg["bilstm_units_2"]

        self.dense_units = model_cfg["dense_units"]

        self.dropout_rate = model_cfg["dropout_rate"]

        self.num_classes = model_cfg["num_classes"]

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=self.input_features,
            out_channels=self.cnn_filters_1,
            kernel_size=self.cnn_kernel_size,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.cnn_filters_1,
            out_channels=self.cnn_filters_2,
            kernel_size=self.cnn_kernel_size,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(
            kernel_size=self.pool_size
        )

        self.dropout = nn.Dropout(
            self.dropout_rate
        )

        # ----------------------------------------------------
        # BiLSTM 1
        # ----------------------------------------------------

        self.bilstm1 = nn.LSTM(
            input_size=self.cnn_filters_2,
            hidden_size=self.bilstm_units_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ----------------------------------------------------
        # BiLSTM 2
        # ----------------------------------------------------

        self.bilstm2 = nn.LSTM(
            input_size=self.bilstm_units_1 * 2,
            hidden_size=self.bilstm_units_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ----------------------------------------------------
        # Dense Layers
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            self.bilstm_units_2 * 2,
            self.dense_units
        )

        self.fc2 = nn.Linear(
            self.dense_units,
            self.num_classes
        )

    def forward(self, x):

        # --------------------------------------------
        # Input:
        # (B, 20, 10)
        # --------------------------------------------

        x = x.permute(
            0,
            2,
            1
        )

        # --------------------------------------------
        # CNN
        # --------------------------------------------

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        x = self.dropout(x)

        # --------------------------------------------
        # Convert for LSTM
        #
        # (B, C, L)
        # ->
        # (B, L, C)
        # --------------------------------------------

        x = x.permute(
            0,
            2,
            1
        )

        # --------------------------------------------
        # BiLSTM 1
        # --------------------------------------------

        x, _ = self.bilstm1(x)

        x = self.dropout(x)

        # --------------------------------------------
        # BiLSTM 2
        # --------------------------------------------

        x, _ = self.bilstm2(x)

        x = self.dropout(x)

        # --------------------------------------------
        # Last Timestep
        # --------------------------------------------

        x = x[:, -1, :]

        # --------------------------------------------
        # Dense
        # --------------------------------------------

        x = self.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        logits = self.fc2(x)

        return logits

    # ==================================================
    # Utilities
    # ==================================================

    def get_model_summary(self):

        total_params = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        model_size_mb = (
            total_params * 4
        ) / (1024 ** 2)

        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)

        print(
            f"Total Parameters    : "
            f"{total_params:,}"
        )

        print(
            f"Trainable Parameters: "
            f"{trainable_params:,}"
        )

        print(
            f"FP32 Size (MB)      : "
            f"{model_size_mb:.2f}"
        )
    def count_parameters(self):
        """
        Return number of trainable parameters.
        """

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
    def export_weights(
        self,
        export_dir
    ):

        export_dir = Path(export_dir)

        fp32_dir = (
            export_dir /
            "fp32"
        )

        fp16_dir = (
            export_dir /
            "fp16"
        )

        fp32_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        fp16_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "\nExporting weights..."
        )

        for name, param in self.state_dict().items():

            clean_name = (
                name.replace(".", "_")
            )

            fp32_path = (
                fp32_dir /
                f"{clean_name}.npy"
            )

            fp16_path = (
                fp16_dir /
                f"{clean_name}.npy"
            )

            np.save(
                fp32_path,
                param.cpu()
                .numpy()
                .astype(np.float32)
            )

            np.save(
                fp16_path,
                param.cpu()
                .numpy()
                .astype(np.float16)
            )

        print(
            "Weight export complete."
        )


# ==========================================================
# Sanity Check
# ==========================================================

if __name__ == "__main__":

    cfg = {
        "model": {
            "input_features": 10,
            "window_size": 20,
            "cnn_filters_1": 64,
            "cnn_filters_2": 128,
            "cnn_kernel_size": 3,
            "pool_size": 2,
            "bilstm_units_1": 128,
            "bilstm_units_2": 64,
            "dense_units": 64,
            "dropout_rate": 0.3,
            "num_classes": 5
        }
    }

    model = CNNBiLSTM(cfg)

    x = torch.randn(
        4,
        20,
        10
    )

    y = model(x)

    print(
        "Output Shape:",
        y.shape
    )

    assert y.shape == (
        4,
        5
    )

    model.get_model_summary()

    print(
        "\nSanity check passed."
    )

