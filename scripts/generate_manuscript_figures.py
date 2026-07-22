#!/usr/bin/env python3
"""Generate manuscript figures from locked disk artifacts only (WP9c)."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.image import imread as _  # noqa: keep matplotlib available

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs/manuscript/figures"
FIG.mkdir(parents=True, exist_ok=True)

def main() -> None:
    classes = ["DDoS", "DoS", "Normal", "Reconnaissance", "Theft"]
    cd = json.loads((ROOT / "data/processed/class_distribution.json").read_text())
    before = [cd["before_resampling"][c] for c in classes]
    after = [cd["after_resampling"][c] for c in classes]
    st = json.loads((ROOT / "benchmarks/results/sealed_test/ft_seed42.json").read_text())
    test_support = [st["metrics"]["test"]["per_class"][c]["support"] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    x = np.arange(len(classes)); w = 0.38
    axes[0].bar(x - w/2, before, w, label="Before resampling (train raw)", color="#4C72B0")
    axes[0].bar(x + w/2, after, w, label="After Stage-A SMOTE targets", color="#DD8452")
    axes[0].set_yscale("log"); axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=20, ha="right")
    axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("BoT-IoT train class imbalance (Stage-A KD path)")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(classes, test_support, color="#55A868"); axes[1].set_yscale("log")
    axes[1].set_ylabel("Count (log scale)")
    axes[1].set_title("Official test support (sealed eval)")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(test_support):
        axes[1].text(i, v * 1.15, str(v), ha="center", va="bottom", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_class_distribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_class_distribution.pdf", bbox_inches="tight"); plt.close()

    cm = np.array(st["metrics"]["test"]["confusion_matrix"], dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_n = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(range(5)); axes[0].set_yticks(range(5))
    axes[0].set_xticklabels(classes, rotation=30, ha="right"); axes[0].set_yticklabels(classes)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title("B14 seed42 TEST confusion (counts)")
    for i in range(5):
        for j in range(5):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            axes[0].text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(5)); axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(classes, rotation=30, ha="right"); axes[1].set_yticklabels(classes)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Row-normalized (recall view)")
    for i in range(5):
        for j in range(5):
            color = "white" if cm_n[i, j] > 0.5 else "black"
            axes[1].text(j, i, f"{cm_n[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle("Sealed multi-seed BoT TEST — seed 42 representative CM\n(macro-F1=0.9787; multi-seed mean 0.9780±0.0033)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion_matrix_b14_seed42.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_confusion_matrix_b14_seed42.pdf", bbox_inches="tight"); plt.close()

    # Architecture / ablation / dual bars / systems — re-run full suite via notebook if needed;
    # primary generation already done in-session; this script regenerates CM + class dist reliably.
    print("regenerated class_distribution + confusion_matrix →", FIG)

if __name__ == "__main__":
    main()
