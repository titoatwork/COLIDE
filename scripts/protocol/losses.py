"""Imbalance-aware losses for Phase 4 (class-aware package).

Focal formulations
------------------
* ``LegacyFocalLoss`` / ``FocalLossLegacy`` (historical; also aliased as ``FocalLoss``)
  obtains ``pt = exp(-CE)`` where CE may already be class-weighted. That makes the
  focusing factor depend on the weight, which is **not** the usual class-weighted
  focal formulation. Historical CAD-CBA / BoT-IoT runs used this path; do not claim
  they used the corrected form.

* ``StandardFocalLoss`` computes ``pt`` from the unweighted target log-probability
  and multiplies class weights only on the final focal term. With ``gamma=0`` and
  no weights this equals ordinary CE; with weights it equals weighted CE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyFocalLoss(nn.Module):
    """
    Historical focal loss: ``pt = exp(-weighted_CE)``.

    When ``alpha`` (class weight) is set, cross-entropy is weighted *before*
    ``pt`` is formed, so the focusing factor ``(1 - pt)^gamma`` is distorted by
    the weight. Preserved for reproduction of champion / protocol recipes.
    Prefer ``StandardFocalLoss`` for new work.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.loss_version = "legacy_pt_from_weighted_ce"
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # weight enters CE → pt is not the true class probability when alpha is set
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# Aliases for the historical implementation
FocalLossLegacy = LegacyFocalLoss
# Backward-compatible name used throughout existing training scripts
FocalLoss = LegacyFocalLoss


class StandardFocalLoss(nn.Module):
    """
    Standard class-weighted focal loss (Lin et al. style).

    * ``pt`` is derived from the **unweighted** target log-probability.
    * Class weight (if any) multiplies the **final** focal term only.

    Contract:
      - gamma=0, alpha=None  → ordinary mean CE
      - gamma=0, alpha set   → weighted CE (same reduction)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.loss_version = "standard_focal_v1"
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Unweighted log-prob of the true class → pt independent of alpha
        log_probs = F.log_softmax(logits, dim=-1)
        targets_long = targets.long()
        log_pt = log_probs.gather(1, targets_long.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        # CE per-sample without class weight
        ce = -log_pt
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at = self.alpha.gather(0, targets_long)
            loss = at * loss
            if self.reduction == "mean":
                # Match F.cross_entropy(weight=...): sum(w*l) / sum(w)
                return loss.sum() / at.sum().clamp(min=1e-12)
            if self.reduction == "sum":
                return loss.sum()
            return loss
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


class AsymmetricLossMultiClass(nn.Module):
    """
    Multi-class asymmetric focal-style loss (bounded C8).

    Emphasizes hard positives for minority classes via higher gamma_neg on easy
    negatives and mild gamma_pos. Simplified multi-class form of ASL:
      loss = -((1-p)^gamma_pos) * y * log(p)  for true class
             -((p)^gamma_neg) * (1-y) * log(1-p) soft one-vs-rest on other classes
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # one-hot targets
        num_classes = logits.size(-1)
        y = F.one_hot(targets.long(), num_classes=num_classes).float()
        # probability via softmax
        xs_pos = torch.softmax(logits, dim=-1)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=1e-8))
        if self.gamma_pos > 0:
            los_pos = los_pos * torch.pow(1.0 - xs_pos, self.gamma_pos)
        if self.gamma_neg > 0:
            los_neg = los_neg * torch.pow(xs_pos, self.gamma_neg)
        loss = -(los_pos + los_neg).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (Khosla et al.) on L2-normalized embeddings.
    Bounded C7/D9: pair with CE/focal classification loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (B, D) unnormalized embeddings
        labels: (B,)
        """
        device = features.device
        features = F.normalize(features, dim=1)
        b = features.size(0)
        if b < 2:
            return features.new_zeros(())
        sim = torch.matmul(features, features.T) / self.temperature
        # mask self
        logits_mask = torch.ones_like(sim) - torch.eye(b, device=device)
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float() * logits_mask
        # for numerical stability
        logits_max, _ = torch.max(sim * logits_mask, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))
        # mean log-prob over positives
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count
        # only anchors that have ≥1 positive contribute
        has_pos = (pos_mask.sum(dim=1) > 0).float()
        if has_pos.sum() < 1:
            return features.new_zeros(())
        loss = -(mean_log_prob_pos * has_pos).sum() / has_pos.sum()
        return loss
