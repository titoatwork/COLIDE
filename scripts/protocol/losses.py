"""Imbalance-aware losses for Phase 4 (class-aware package)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Standard focal loss with optional per-class alpha weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LogitAdjustedCrossEntropy(nn.Module):
    """
    Logit adjustment for long-tail classification (Menon et al.).
    logits' = logits + tau * log(pi), pi = class frequency on train.
    """

    def __init__(self, class_probs: torch.Tensor, tau: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.tau = tau
        self.reduction = reduction
        # class_probs: (C,) positive, sum to 1
        log_pi = torch.log(class_probs.clamp(min=1e-12))
        self.register_buffer("log_pi", log_pi.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_pi.unsqueeze(0)
        return F.cross_entropy(adjusted, targets, reduction=self.reduction)


def class_probs_from_counts(y: torch.Tensor | list, num_classes: int) -> torch.Tensor:
    """Empirical class frequencies from label vector."""
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    counts = torch.bincount(y.long(), minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    return counts / counts.sum()


def class_balanced_weights(
    y: torch.Tensor | list,
    num_classes: int,
    beta: float = 0.9999,
) -> torch.Tensor:
    """Class-balanced weights (Cui et al.) from effective number of samples."""
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    counts = torch.bincount(y.long(), minlength=num_classes).float().clamp(min=1.0)
    effective = 1.0 - torch.pow(torch.tensor(beta), counts)
    weights = (1.0 - beta) / effective
    weights = weights / weights.sum() * num_classes
    return weights
