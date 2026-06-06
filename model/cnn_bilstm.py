import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class CNNBiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]

        self.input_features = model_cfg["input_features"]
        self.projection_dim = model_cfg["projection_dim"]
        self.reshape_channels = model_cfg["reshape"][0]
        self.reshape_length = model_cfg["reshape"][1]
        self.cnn_filters_1 = model_cfg["cnn_filters_1"]
        self.cnn_filters_2 = model_cfg["cnn_filters_2"]
        self.kernel_size = model_cfg["cnn_kernel_size"]
        self.pool_size = model_cfg["pool_size"]
        self.bilstm_units_1 = model_cfg["bilstm_units_1"]
        self.bilstm_units_2 = model_cfg["bilstm_units_2"]
        self.dense_units = model_cfg["dense_units"]
        self.dropout_rate = model_cfg["dropout_rate"]
        self.num_classes = model_cfg["num_classes"]

        self.input_projection = nn.Linear(
            self.input_features,
            self.projection_dim,
        )

        self.conv1 = nn.Conv1d(
            in_channels=self.reshape_channels,
            out_channels=self.cnn_filters_1,
            kernel_size=self.kernel_size,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(self.cnn_filters_1)

        self.conv2 = nn.Conv1d(
            in_channels=self.cnn_filters_1,
            out_channels=self.cnn_filters_2,
            kernel_size=self.kernel_size,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(self.cnn_filters_2)

        self.pool = nn.MaxPool1d(kernel_size=self.pool_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.bilstm1 = nn.LSTM(
            input_size=self.cnn_filters_2,
            hidden_size=self.bilstm_units_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.bilstm2 = nn.LSTM(
            input_size=self.bilstm_units_1 * 2,
            hidden_size=self.bilstm_units_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(
            self.bilstm_units_2 * 2,
            self.dense_units,
        )
        self.fc2 = nn.Linear(
            self.dense_units,
            self.num_classes,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_projection(x)
        x = x.view(
            x.size(0),
            self.reshape_channels,
            self.reshape_length,
        )

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        logits = self.fc2(x)
        return logits

    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def get_model_summary(self):
        params = self.count_parameters()
        size_mb = (params * 4) / (1024 ** 2)

        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        print(f"Total Parameters    : {params:,}")
        print(f"FP32 Size (MB)      : {size_mb:.2f}")

    def export_weights(self):
        config_path = (
            Path(__file__)
            .resolve()
            .parent
            .parent
            / "config"
            / "config.yaml"
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        export_dir = (
            Path(__file__)
            .resolve()
            .parent
            .parent
            / config["model"]["weight_export_path"]
        )

        export_dir.mkdir(parents=True, exist_ok=True)

        print("\nExporting Weights...")

        for name, param in self.named_parameters():
            safe_name = name.replace(".", "_")
            fp32_path = export_dir / f"{safe_name}_fp32.npy"
            fp16_path = export_dir / f"{safe_name}_fp16.npy"
            weights = param.detach().cpu().numpy()

            np.save(fp32_path, weights.astype(np.float32))
            np.save(fp16_path, weights.astype(np.float16))

        print(f"Saved weights to: {export_dir}")


def load_config():
    config_path = (
        Path(__file__)
        .resolve()
        .parent
        .parent
        / "config"
        / "config.yaml"
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    config = load_config()
    model = CNNBiLSTM(config)

    dummy_input = torch.randn(
        4,
        config["model"]["input_features"],
    )

    output = model(dummy_input)

    print(f"Output Shape: {output.shape}")
    assert output.shape == (
        4,
        config["model"]["num_classes"],
    )

    model.get_model_summary()
    print("\nSanity check passed.")
