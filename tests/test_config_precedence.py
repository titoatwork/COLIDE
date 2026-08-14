"""CLI > HPO file > program defaults for train_protocol_ft."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.train_protocol_ft import PROGRAM_DEFAULTS, _resolve_train_hps

HPO = Path(__file__).resolve().parents[1] / "config" / "hpo_best.yaml"


def _ns(**kwargs) -> argparse.Namespace:
    base = dict(
        lr=None,
        batch_size=None,
        focal_gamma=None,
        dropout_rate=None,
        attention_dropout=None,
        weight_decay=None,
        scheduler=None,
        optimizer="auto",
        hpo_config="",
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_cli_overrides_hpo_lr():
    args = _ns(hpo_config=str(HPO), lr=1e-3)
    out, src = _resolve_train_hps(args)
    assert src is not None
    assert out.lr == 1e-3
    # HPO still fills unset fields
    assert out.batch_size == 1024


def test_hpo_fills_when_cli_unset():
    args = _ns(hpo_config=str(HPO))
    out, _ = _resolve_train_hps(args)
    assert abs(out.lr - 5.89306076111462e-05) < 1e-12
    assert out.scheduler == "cosine"


def test_program_defaults_without_hpo():
    args = _ns()
    out, src = _resolve_train_hps(args)
    assert src is None
    assert out.lr == PROGRAM_DEFAULTS["lr"]
    assert out.batch_size == PROGRAM_DEFAULTS["batch_size"]
    assert out.scheduler == PROGRAM_DEFAULTS["scheduler"]
