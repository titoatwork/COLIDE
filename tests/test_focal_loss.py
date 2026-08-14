"""Focal loss contracts: gamma=0 equals (weighted) CE; standard vs legacy."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.protocol.losses import LegacyFocalLoss, StandardFocalLoss


def test_standard_gamma_zero_equals_ce():
    torch.manual_seed(0)
    logits = torch.randn(16, 5, requires_grad=False)
    targets = torch.randint(0, 5, (16,))
    fl = StandardFocalLoss(gamma=0.0, alpha=None, reduction="mean")
    ce = F.cross_entropy(logits, targets, reduction="mean")
    assert torch.allclose(fl(logits, targets), ce, atol=1e-6, rtol=1e-5)


def test_standard_gamma_zero_weighted_equals_weighted_ce():
    torch.manual_seed(1)
    logits = torch.randn(16, 4)
    targets = torch.randint(0, 4, (16,))
    w = torch.tensor([1.0, 2.0, 0.5, 1.5])
    fl = StandardFocalLoss(gamma=0.0, alpha=w, reduction="mean")
    ce = F.cross_entropy(logits, targets, weight=w, reduction="mean")
    assert torch.allclose(fl(logits, targets), ce, atol=1e-6, rtol=1e-5)


def test_legacy_gamma_zero_no_weight_equals_ce():
    torch.manual_seed(2)
    logits = torch.randn(12, 3)
    targets = torch.randint(0, 3, (12,))
    fl = LegacyFocalLoss(gamma=0.0, alpha=None, reduction="mean")
    ce = F.cross_entropy(logits, targets, reduction="mean")
    assert torch.allclose(fl(logits, targets), ce, atol=1e-6, rtol=1e-5)


def test_standard_higher_true_prob_lower_loss():
    # Same target; boost true-class logit → lower focal loss
    targets = torch.tensor([1, 1, 1])
    hard = torch.tensor(
        [[0.0, 0.1, 0.0], [0.0, 0.1, 0.0], [0.0, 0.1, 0.0]],
        dtype=torch.float32,
    )
    easy = torch.tensor(
        [[0.0, 5.0, 0.0], [0.0, 5.0, 0.0], [0.0, 5.0, 0.0]],
        dtype=torch.float32,
    )
    fl = StandardFocalLoss(gamma=2.0, reduction="mean")
    assert fl(easy, targets).item() < fl(hard, targets).item()


def test_standard_no_nan_extreme_logits():
    targets = torch.tensor([0, 1])
    logits = torch.tensor(
        [[1e4, -1e4, 0.0], [-1e4, 1e4, 0.0]],
        dtype=torch.float32,
    )
    fl = StandardFocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 2.0, 0.5]))
    loss = fl(logits, targets)
    assert torch.isfinite(loss).all()
