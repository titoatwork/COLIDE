#!/usr/bin/env python3
"""
WP2d — Val-only per-class decision threshold search (never fit on test).

Default checkpoint: model/imbalance_loss/ft_focal_seed42.pth (focal winner).

Runs:
  1) argmax baseline (standard decode)
  2) fixed t=0.5 per class (threshold rule baseline)
  3) coordinate search → max val macro-F1
  4) coordinate search → max val min_per_class_F1
  5) joint: max macro then improve min-class without dropping macro
  6) optional class_f1 search for named minority (Theft / Normal)

Writes JSON under benchmarks/results/imbalance_loss/ by default.
Test remains sealed unless --allow-test (metrics only; still never *fit* on test).

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_val_thresholds.py \\
    --checkpoint model/imbalance_loss/ft_focal_seed42.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402
from scripts.protocol.thresholds import (  # noqa: E402
    apply_thresholds,
    search_class_thresholds,
    search_joint_macro_then_min,
)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def predict_proba(
    model, X: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks = []
    use_cuda = device.type == "cuda"
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=use_cuda)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
        chunks.append(probs)
    return np.concatenate(chunks, axis=0)


def pack_metrics(y_true, y_pred, class_names) -> dict:
    return compute_classification_metrics(y_true, y_pred, class_names)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="model/imbalance_loss/ft_focal_seed42.pth",
    )
    p.add_argument(
        "--stage",
        choices=["stage_a_kd", "stage_b_ft"],
        default="stage_b_ft",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--grid-min",
        type=float,
        default=0.05,
        help="Lower bound of threshold grid",
    )
    p.add_argument("--grid-max", type=float, default=0.95)
    p.add_argument(
        "--grid-steps",
        type=int,
        default=19,
        help="Number of grid points (default 19 → step 0.05 on [0.05,0.95])",
    )
    p.add_argument("--n-passes", type=int, default=2)
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Default: benchmarks/results/imbalance_loss/thresholds_focal_seed42.json",
    )
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument(
        "--allow-test",
        action="store_true",
        help="Also *evaluate* sealed test under selected thresholds (never fit on test)",
    )
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        # allow relative to project root
        ckpt = PROJECT_ROOT / args.checkpoint
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  checkpoint={ckpt}", flush=True)

    bundle = load_botiot(stage=args.stage, seed=args.seed)
    config = load_config(PROJECT_ROOT / args.config)

    sys.path.insert(0, str(PROJECT_ROOT / "model"))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    model = CNNBiLSTMAttention(config).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    print(
        f"protocol={bundle.protocol_id} stage={args.stage} "
        f"n_val={len(bundle.y_val)} classes={bundle.class_names}",
        flush=True,
    )
    print("computing val softmax probabilities…", flush=True)
    val_probs = predict_proba(model, bundle.X_val, device, args.batch_size)
    y_val = np.asarray(bundle.y_val)

    grid = np.linspace(args.grid_min, args.grid_max, args.grid_steps)

    # --- baselines ---
    pred_argmax = np.argmax(val_probs, axis=1)
    m_argmax = pack_metrics(y_val, pred_argmax, bundle.class_names)

    t05 = np.full(val_probs.shape[1], 0.5, dtype=np.float64)
    pred_t05 = apply_thresholds(val_probs, t05)
    m_t05 = pack_metrics(y_val, pred_t05, bundle.class_names)

    print(
        f"argmax  macro-F1={m_argmax['macro_f1']:.4f}  "
        f"minF1={m_argmax['min_per_class_f1']:.4f}  "
        f"theft={m_argmax.get('theft_f1', float('nan')):.4f}",
        flush=True,
    )

    # --- searches (val only) ---
    print("search: macro_f1 …", flush=True)
    s_macro = search_class_thresholds(
        y_val, val_probs, grid=grid, objective="macro_f1", n_passes=args.n_passes
    )
    pred_macro = apply_thresholds(val_probs, np.asarray(s_macro["thresholds"]))
    m_macro = pack_metrics(y_val, pred_macro, bundle.class_names)

    print("search: min_per_class_f1 …", flush=True)
    s_min = search_class_thresholds(
        y_val,
        val_probs,
        grid=grid,
        objective="min_per_class_f1",
        n_passes=args.n_passes,
    )
    pred_min = apply_thresholds(val_probs, np.asarray(s_min["thresholds"]))
    m_min = pack_metrics(y_val, pred_min, bundle.class_names)

    print("search: joint macro→min …", flush=True)
    s_joint = search_joint_macro_then_min(
        y_val, val_probs, grid=grid, n_passes=args.n_passes
    )
    pred_joint = apply_thresholds(val_probs, np.asarray(s_joint["thresholds"]))
    m_joint = pack_metrics(y_val, pred_joint, bundle.class_names)

    minority_searches: dict[str, Any] = {}
    for rare_name in ("Theft", "Normal"):
        if rare_name not in bundle.class_names:
            continue
        idx = bundle.class_names.index(rare_name)
        obj = f"class_f1:{idx}"
        print(f"search: {obj} ({rare_name}) …", flush=True)
        s_r = search_class_thresholds(
            y_val, val_probs, grid=grid, objective=obj, n_passes=args.n_passes
        )
        pred_r = apply_thresholds(val_probs, np.asarray(s_r["thresholds"]))
        m_r = pack_metrics(y_val, pred_r, bundle.class_names)
        minority_searches[rare_name] = {
            "search": s_r,
            "val": m_r,
            "class_index": idx,
        }

    variants = {
        "argmax": {
            "decode": "argmax",
            "thresholds": None,
            "val": m_argmax,
        },
        "fixed_0.5": {
            "decode": "per_class_threshold",
            "thresholds": t05.tolist(),
            "val": m_t05,
        },
        "search_macro_f1": {
            "decode": "per_class_threshold",
            "thresholds": s_macro["thresholds"],
            "search": {
                k: s_macro[k]
                for k in (
                    "objective",
                    "val_objective_score",
                    "val_macro_f1",
                    "grid",
                    "n_passes",
                    "n_val",
                    "n_classes",
                )
            },
            "val": m_macro,
        },
        "search_min_per_class_f1": {
            "decode": "per_class_threshold",
            "thresholds": s_min["thresholds"],
            "search": {
                k: s_min[k]
                for k in (
                    "objective",
                    "val_objective_score",
                    "val_macro_f1",
                    "grid",
                    "n_passes",
                    "n_val",
                    "n_classes",
                )
            },
            "val": m_min,
        },
        "search_joint_macro_then_min": {
            "decode": "per_class_threshold",
            "thresholds": s_joint["thresholds"],
            "search": {
                "objective": s_joint["objective"],
                "macro_floor": s_joint["macro_floor"],
                "val_macro_f1": s_joint["val_macro_f1"],
                "val_min_per_class_f1": s_joint["val_min_per_class_f1"],
                "grid": s_joint["grid"],
                "n_passes": s_joint["n_passes"],
            },
            "val": m_joint,
        },
    }
    for name, blob in minority_searches.items():
        variants[f"search_class_f1_{name.lower()}"] = {
            "decode": "per_class_threshold",
            "thresholds": blob["search"]["thresholds"],
            "search": {
                "objective": blob["search"]["objective"],
                "val_objective_score": blob["search"]["val_objective_score"],
                "val_macro_f1": blob["search"]["val_macro_f1"],
                "class_name": name,
                "class_index": blob["class_index"],
            },
            "val": blob["val"],
        }

    # Select best by primary protocol metric (val macro-F1), tie-break min_per_class
    def rank_key(item):
        name, v = item
        m = v["val"]
        return (m["macro_f1"], m["min_per_class_f1"], m.get("theft_f1", 0.0))

    ranked = sorted(variants.items(), key=rank_key, reverse=True)
    best_name, best_var = ranked[0]
    delta_macro = best_var["val"]["macro_f1"] - m_argmax["macro_f1"]
    delta_min = best_var["val"]["min_per_class_f1"] - m_argmax["min_per_class_f1"]

    # Decision policy for CAD-CBA-v1
    # Incorporate only if clear non-trivial gain; else RUN_DOCUMENTED keep argmax
    EPS_MACRO = 0.0005  # 5e-4 absolute macro-F1
    EPS_MIN = 0.005
    if delta_macro >= EPS_MACRO or (
        abs(delta_macro) < EPS_MACRO and delta_min >= EPS_MIN
    ):
        decision = "INCORPORATE"
        decision_note = (
            f"Best variant '{best_name}' improves over argmax "
            f"(Δmacro={delta_macro:+.4f}, Δmin_cls={delta_min:+.4f}). "
            "Lock thresholds for this checkpoint family; re-fit on val after any retrain."
        )
        selected_decode = best_name
    else:
        decision = "RUN_DOCUMENTED"
        decision_note = (
            f"Best variant '{best_name}' Δmacro={delta_macro:+.4f}, "
            f"Δmin_cls={delta_min:+.4f} vs argmax — below incorporation thresholds "
            f"(macro≥{EPS_MACRO} or min≥{EPS_MIN} with flat macro). "
            "Keep default argmax decode; thresholds available if minority ops need them."
        )
        selected_decode = "argmax"

    comparison_table = []
    for name, v in variants.items():
        m = v["val"]
        comparison_table.append(
            {
                "variant": name,
                "val_macro_f1": m["macro_f1"],
                "val_balanced_accuracy": m["balanced_accuracy"],
                "val_min_per_class_f1": m["min_per_class_f1"],
                "val_theft_f1": m.get("theft_f1"),
                "val_normal_f1": m.get("normal_f1"),
                "thresholds": v.get("thresholds"),
            }
        )
    comparison_table.sort(key=lambda r: r["val_macro_f1"], reverse=True)

    test_block = None
    if args.allow_test:
        print("evaluating sealed test under selected + argmax (NOT used for fit)…", flush=True)
        test_probs = predict_proba(model, bundle.X_test, device, args.batch_size)
        y_test = np.asarray(bundle.y_test)
        test_block = {
            "note": "Test evaluated only; thresholds were fit on val only.",
            "argmax": pack_metrics(
                y_test, np.argmax(test_probs, axis=1), bundle.class_names
            ),
        }
        if best_var.get("thresholds") is not None:
            test_block[best_name] = pack_metrics(
                y_test,
                apply_thresholds(test_probs, np.asarray(best_var["thresholds"])),
                bundle.class_names,
            )

    ckpt_md5 = md5_file(ckpt)
    envelope = make_result_envelope(
        experiment_id="val_thresholds_focal_seed42",
        protocol_id=bundle.protocol_id,
        stage=args.stage,
        seed=args.seed,
        config={
            "checkpoint": str(ckpt.relative_to(PROJECT_ROOT) if ckpt.is_relative_to(PROJECT_ROOT) else ckpt),
            "checkpoint_md5": ckpt_md5,
            "grid_min": args.grid_min,
            "grid_max": args.grid_max,
            "grid_steps": args.grid_steps,
            "n_passes": args.n_passes,
            "batch_size": args.batch_size,
            "device": str(device),
            "allow_test": bool(args.allow_test),
            "incorporation_eps_macro": EPS_MACRO,
            "incorporation_eps_min_per_class": EPS_MIN,
        },
        metrics={
            "argmax_val_macro_f1": m_argmax["macro_f1"],
            "argmax_val_min_per_class_f1": m_argmax["min_per_class_f1"],
            "best_variant": best_name,
            "best_val_macro_f1": best_var["val"]["macro_f1"],
            "best_val_min_per_class_f1": best_var["val"]["min_per_class_f1"],
            "delta_macro_vs_argmax": delta_macro,
            "delta_min_per_class_vs_argmax": delta_min,
            "selected_decode": selected_decode,
            "decision": decision,
        },
        extra={
            "class_names": bundle.class_names,
            "data_summary": bundle.summary(),
            "comparison_table": comparison_table,
            "variants": variants,
            "decision_note": decision_note,
            "test": test_block,
            "note": (
                "Val-only threshold fit. Test sealed unless --allow-test "
                "(eval only, never fit)."
            ),
        },
        project_root=PROJECT_ROOT,
    )

    out = Path(args.out) if args.out else (
        PROJECT_ROOT
        / "benchmarks"
        / "results"
        / "imbalance_loss"
        / f"thresholds_focal_seed{args.seed}.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(envelope, f, indent=2)

    # Human summary
    print("=" * 60)
    print(f"checkpoint md5: {ckpt_md5}")
    print(f"{'variant':32s}  macroF1   minF1    theftF1  normalF1")
    for row in comparison_table:
        print(
            f"{row['variant']:32s}  "
            f"{row['val_macro_f1']:.4f}   "
            f"{row['val_min_per_class_f1']:.4f}   "
            f"{(row['val_theft_f1'] or 0):.4f}   "
            f"{(row['val_normal_f1'] or 0):.4f}"
        )
    print("-" * 60)
    print(f"BEST by macro-F1: {best_name}")
    print(f"  Δmacro={delta_macro:+.4f}  Δmin_cls={delta_min:+.4f}")
    print(f"DECISION: {decision}")
    print(f"  {decision_note}")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
