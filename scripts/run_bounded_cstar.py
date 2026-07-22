#!/usr/bin/env python3
"""
Bounded C* playlist experiments (val-only; test sealed).

Covers tracker C4/C5/C7/C8/C10 and D9 under a fixed seed-42 budget:

  C4  multi_scale     multi-scale CNN–BiLSTM (scratch, focal)
  C5  gated           gated CNN–BiLSTM fusion (scratch, focal)
  C7  supcon          V3 + focal + supervised contrastive (distill init)
  C8  asymmetric      V3 + asymmetric multi-class loss (distill init)
  C10 uncertainty     post-hoc MC-dropout + entropy abstain on HPO confirm ckpt
  CTRL control_v3     V3 focal hpo-ish budget control (distill init) for fair Δ

Does not clobber champion. Writes benchmarks/results/cstar_bounded/.

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_bounded_cstar.py
  PYTHONPATH=. .venv/bin/python scripts/run_bounded_cstar.py --rows C4,C5,C10
  PYTHONPATH=. .venv/bin/python scripts/run_bounded_cstar.py --skip-existing
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.method_variants import build_method_variant  # noqa: E402
from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.losses import (  # noqa: E402
    AsymmetricLossMultiClass,
    FocalLoss,
    SupConLoss,
)
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402

CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
DISTILL = PROJECT_ROOT / "model" / "best_model_botiot_distill_a0.6_T10.0_focal2.pth"
HPO_YAML = PROJECT_ROOT / "config" / "hpo_best.yaml"
HPO_CONFIRM_CKPT = PROJECT_ROOT / "model" / "multirun_hpo_confirm" / "ft_seed42.pth"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "cstar_bounded"
CKPT_DIR = PROJECT_ROOT / "model" / "cstar_bounded"

ROWS: list[dict[str, Any]] = [
    {
        "row": "CTRL",
        "name": "control_v3_focal",
        "tracker": "control",
        "variant": "v3_embed",
        "loss": "focal",
        "init": "distill",
        "use_hpo": True,
        "epochs_default": 8,
    },
    {
        "row": "C4",
        "name": "multi_scale",
        "tracker": "C4",
        "variant": "multi_scale",
        "loss": "focal",
        "init": "scratch",
        "use_hpo": False,
        "epochs_default": 8,
    },
    {
        "row": "C5",
        "name": "gated_fusion",
        "tracker": "C5",
        "variant": "gated",
        "loss": "focal",
        "init": "scratch",
        "use_hpo": False,
        "epochs_default": 8,
    },
    {
        "row": "C7",
        "name": "supcon_focal",
        "tracker": "C7/D9",
        "variant": "v3_embed",
        "loss": "supcon_focal",
        "init": "distill",
        "use_hpo": True,
        "epochs_default": 8,
    },
    {
        "row": "C8",
        "name": "asymmetric",
        "tracker": "C8",
        "variant": "v3_embed",
        "loss": "asymmetric",
        "init": "distill",
        "use_hpo": True,
        "epochs_default": 8,
    },
    {
        "row": "C10",
        "name": "uncertainty_mc_dropout",
        "tracker": "C10",
        "variant": "posthoc",
        "loss": None,
        "init": "hpo_confirm",
        "use_hpo": False,
        "epochs_default": 0,
    },
]


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_hpo_params(path: Path) -> dict:
    with open(path) as f:
        doc = yaml.safe_load(f)
    return dict(doc["hpo"]["best_params"])


def make_loader(X, y, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    pin = device.type == "cuda"
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )


@torch.no_grad()
def eval_split(model, loader, device, class_names, *, mc_dropout: bool = False, mc_passes: int = 1):
    if mc_dropout:
        model.train()  # enable dropout
    else:
        model.eval()
    preds, targets = [], []
    entropies = []
    use_amp = device.type == "cuda"
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if mc_dropout and mc_passes > 1:
            probs_stack = []
            for _ in range(mc_passes):
                with autocast("cuda", enabled=use_amp):
                    out = model(xb)
                    logits = out[0] if isinstance(out, tuple) else out
                probs_stack.append(torch.softmax(logits.float(), dim=-1))
            probs = torch.stack(probs_stack, dim=0).mean(0)
            pred = probs.argmax(dim=1)
            ent = -(probs * (probs.clamp(min=1e-8).log())).sum(dim=-1)
            entropies.append(ent.cpu().numpy())
        else:
            with autocast("cuda", enabled=use_amp):
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
            pred = logits.argmax(dim=1)
            probs = torch.softmax(logits.float(), dim=-1)
            ent = -(probs * (probs.clamp(min=1e-8).log())).sum(dim=-1)
            entropies.append(ent.cpu().numpy())
        preds.append(pred.cpu().numpy())
        targets.append(yb.numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    ent_all = np.concatenate(entropies)
    m = compute_classification_metrics(y_true, y_pred, class_names)
    m["mean_entropy"] = float(ent_all.mean())
    return m, y_true, y_pred, ent_all


@torch.no_grad()
def measure_latency(model, sample, device, warmup=15, reps=30) -> dict:
    model.eval()
    xb = sample.to(device)
    if device.type == "cuda":
        for _ in range(warmup):
            _ = model(xb)
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = model(xb)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    else:
        for _ in range(warmup):
            _ = model(xb)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = model(xb)
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "batch_size": int(xb.size(0)),
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        "per_sample_us": float(statistics.mean(times) * 1000.0 / xb.size(0)),
    }


def try_load_v3_weights(model, path: Path, device) -> bool:
    if not path.is_file():
        return False
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    cleaned = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        nk = k[len("module.") :] if k.startswith("module.") else k
        cleaned[nk] = v
    try:
        model.load_state_dict(cleaned, strict=True)
        return True
    except Exception:
        try:
            model.load_state_dict(cleaned, strict=False)
            return True
        except Exception:
            return False


def train_row(
    *,
    row: dict[str, Any],
    bundle,
    base_cfg: dict,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    supcon_weight: float,
    skip_existing: bool,
) -> dict[str, Any]:
    rid = row["row"]
    name = row["name"]
    results_path = OUT_DIR / f"{rid}_{name}_seed{seed}.json"
    save_path = CKPT_DIR / f"{rid}_{name}_seed{seed}.pth"

    if skip_existing and results_path.is_file():
        print(f"  skip-existing {rid}", flush=True)
        with open(results_path) as f:
            env = json.load(f)
        m = env.get("metrics") or {}
        return {
            "row": rid,
            "name": name,
            "tracker": row["tracker"],
            "best_val_macro_f1": m.get("best_val_macro_f1"),
            "val_min_per_class_f1": (m.get("val") or {}).get("min_per_class_f1"),
            "val_theft_f1": (m.get("val") or {}).get("theft_f1"),
            "results_path": str(results_path),
            "checkpoint_path": str(save_path) if save_path.is_file() else None,
            "returncode": 0,
            "skipped": True,
        }

    t0 = time.time()
    cfg = copy.deepcopy(base_cfg)
    hpo = load_hpo_params(HPO_YAML) if row.get("use_hpo") else {}
    if row.get("use_hpo"):
        cfg["model"]["dropout_rate"] = float(hpo.get("dropout_rate", cfg["model"]["dropout_rate"]))
        cfg["model"]["attention_dropout"] = float(
            hpo.get("attention_dropout", cfg["model"].get("attention_dropout", 0.1))
        )
        lr = float(hpo.get("lr", 1e-3))
        batch_size = int(hpo.get("batch_size", 512))
        weight_decay = float(hpo.get("weight_decay", 0.0))
        gamma = float(hpo.get("focal_gamma", 2.0))
        optim_name = "adamw"
    else:
        lr = 1e-3
        batch_size = 512
        weight_decay = 0.0
        gamma = 2.0
        optim_name = "adam"

    set_seed(seed)
    model = build_method_variant(row["variant"], cfg, device)
    init_note = row["init"]
    if row["init"] == "distill":
        ok = try_load_v3_weights(model, DISTILL, device)
        init_note = f"distill:{DISTILL.name}:{'ok' if ok else 'FAIL'}"
        if not ok:
            print(f"  WARN {rid}: distill load failed; training from scratch", flush=True)

    train_loader = make_loader(bundle.X_train, bundle.y_train, batch_size, True, device)
    val_loader = make_loader(bundle.X_val, bundle.y_val, batch_size, False, device)

    if row["loss"] == "focal":
        criterion_cls: Any = FocalLoss(gamma=gamma)
    elif row["loss"] == "asymmetric":
        criterion_cls = AsymmetricLossMultiClass(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    elif row["loss"] == "supcon_focal":
        criterion_cls = FocalLoss(gamma=gamma)
    else:
        criterion_cls = FocalLoss(gamma=gamma)
    criterion_sup = SupConLoss(temperature=0.07) if row["loss"] == "supcon_focal" else None

    if optim_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1)) if row.get("use_hpo") else None
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    best_val = None
    patience_left = patience
    history = []
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                if criterion_sup is not None:
                    logits, emb = model(xb, return_embed=True)
                    loss_cls = criterion_cls(logits, yb)
                    loss_sc = criterion_sup(emb.float(), yb)
                    loss = loss_cls + supcon_weight * loss_sc
                else:
                    out = model(xb)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = criterion_cls(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        if scheduler is not None:
            scheduler.step()

        val_m, _, _, _ = eval_split(model, val_loader, device, bundle.class_names)
        row_h = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_macro_f1": val_m["macro_f1"],
            "val_min_per_class_f1": val_m["min_per_class_f1"],
            "val_theft_f1": val_m.get("theft_f1"),
        }
        history.append(row_h)
        print(
            f"  {rid} ep{epoch:02d} loss={row_h['train_loss']:.4f} "
            f"val_macro_f1={val_m['macro_f1']:.4f} min={val_m['min_per_class_f1']:.4f} "
            f"theft={val_m.get('theft_f1', float('nan')):.4f}",
            flush=True,
        )
        if val_m["macro_f1"] > best_f1:
            best_f1 = float(val_m["macro_f1"])
            best_val = val_m
            patience_left = patience
            torch.save(model.state_dict(), save_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    val_final, _, _, _ = eval_split(model, val_loader, device, bundle.class_names)
    sample = torch.from_numpy(np.asarray(bundle.X_val[:256], dtype=np.float32))
    latency = measure_latency(model, sample, device)
    mem = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = model(sample.to(device))
        torch.cuda.synchronize()
        mem = {
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }
    n_params = int(
        model.count_parameters()
        if hasattr(model, "count_parameters")
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    elapsed = time.time() - t0
    env = make_result_envelope(
        experiment_id=f"cstar_{rid}_{name}_seed{seed}",
        protocol_id=bundle.protocol_id,
        stage="stage_b_ft",
        seed=seed,
        config={
            "row": rid,
            "name": name,
            "tracker": row["tracker"],
            "variant": row["variant"],
            "loss": row["loss"],
            "init": init_note,
            "use_hpo": row.get("use_hpo"),
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "batch_size": batch_size,
            "focal_gamma": gamma if row["loss"] in ("focal", "supcon_focal") else None,
            "supcon_weight": supcon_weight if row["loss"] == "supcon_focal" else None,
            "weight_decay": weight_decay,
            "optimizer": optim_name,
            "save_path": str(save_path),
        },
        metrics={
            "best_val_macro_f1": float(best_f1),
            "val": val_final,
            "systems": {
                "n_params": n_params,
                "checkpoint_bytes": save_path.stat().st_size if save_path.is_file() else None,
                "latency": latency,
                "cuda_memory": mem,
            },
        },
        extra={
            "history": history,
            "elapsed_sec": elapsed,
            "allow_test": False,
            "device": str(device),
            "best_val_snapshot": best_val,
            "data_summary": bundle.summary(),
            "git_sha": git_sha(),
        },
        project_root=PROJECT_ROOT,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(env, f, indent=2)
    print(
        f"  {rid} DONE best_f1={best_f1:.4f} params={n_params} "
        f"lat={latency['per_sample_us']:.2f}µs -> {results_path}",
        flush=True,
    )
    return {
        "row": rid,
        "name": name,
        "tracker": row["tracker"],
        "best_val_macro_f1": float(best_f1),
        "val_min_per_class_f1": float(val_final.get("min_per_class_f1") or 0.0),
        "val_theft_f1": float(val_final.get("theft_f1") or 0.0),
        "n_params": n_params,
        "per_sample_us": latency["per_sample_us"],
        "results_path": str(results_path),
        "checkpoint_path": str(save_path),
        "returncode": 0,
        "elapsed_sec": elapsed,
    }


def run_c10_uncertainty(
    *,
    bundle,
    base_cfg: dict,
    device: torch.device,
    seed: int,
    mc_passes: int = 10,
) -> dict[str, Any]:
    """C10 post-hoc MC-dropout uncertainty on HPO confirm checkpoint."""
    rid, name = "C10", "uncertainty_mc_dropout"
    results_path = OUT_DIR / f"{rid}_{name}_seed{seed}.json"
    t0 = time.time()
    set_seed(seed)
    cfg = copy.deepcopy(base_cfg)
    if HPO_YAML.is_file():
        hpo = load_hpo_params(HPO_YAML)
        cfg["model"]["dropout_rate"] = float(hpo.get("dropout_rate", cfg["model"]["dropout_rate"]))
        cfg["model"]["attention_dropout"] = float(
            hpo.get("attention_dropout", cfg["model"].get("attention_dropout", 0.1))
        )
    model = build_method_variant("v3_embed", cfg, device)
    ckpt = HPO_CONFIRM_CKPT if HPO_CONFIRM_CKPT.is_file() else DISTILL
    ok = try_load_v3_weights(model, ckpt, device)
    print(f"  C10 load {ckpt.name}: {'ok' if ok else 'FAIL'}", flush=True)

    val_loader = make_loader(bundle.X_val, bundle.y_val, 512, False, device)
    # baseline argmax (eval mode)
    m_det, y_true, y_pred_det, ent_det = eval_split(
        model, val_loader, device, bundle.class_names, mc_dropout=False
    )
    # MC dropout mean
    m_mc, _, y_pred_mc, ent_mc = eval_split(
        model, val_loader, device, bundle.class_names, mc_dropout=True, mc_passes=mc_passes
    )

    # entropy-abstain: reject top-q% most uncertain; among kept, recompute macro-F1
    # (abstain counted as wrong for macro fairness? better: evaluate on kept subset only
    # and also report coverage — standard selective classification)
    def selective_f1(y_t, y_p, entropy, keep_frac: float) -> dict:
        thr = np.quantile(entropy, keep_frac)
        keep = entropy <= thr
        if keep.sum() < 10:
            return {"keep_frac": keep_frac, "coverage": 0.0, "macro_f1": None}
        # Subset may drop rare classes; use labels=full class range so report stays 5-way.
        from sklearn.metrics import f1_score, balanced_accuracy_score

        yt, yp = y_t[keep], y_p[keep]
        labels = list(range(len(bundle.class_names)))
        macro = float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0))
        per = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
        min_cls = float(np.min(per)) if len(per) else 0.0
        # Theft index from class_names if present
        theft_idx = None
        for i, n in enumerate(bundle.class_names):
            if str(n).lower() == "theft":
                theft_idx = i
                break
        theft_f1 = float(per[theft_idx]) if theft_idx is not None else 0.0
        return {
            "keep_frac_target": keep_frac,
            "coverage": float(keep.mean()),
            "entropy_threshold": float(thr),
            "macro_f1": macro,
            "min_per_class_f1": min_cls,
            "theft_f1": theft_f1,
            "n_kept": int(keep.sum()),
            "n_classes_present": int(len(np.unique(yt))),
            "balanced_accuracy": float(
                balanced_accuracy_score(yt, yp) if len(np.unique(yt)) > 1 else 0.0
            ),
        }

    selective = {
        "deterministic": [selective_f1(y_true, y_pred_det, ent_det, q) for q in (1.0, 0.99, 0.95, 0.90)],
        "mc_dropout": [selective_f1(y_true, y_pred_mc, ent_mc, q) for q in (1.0, 0.99, 0.95, 0.90)],
    }

    # Decision: does selective classification improve macro on kept set meaningfully?
    # For package: only incorporate if coverage≥0.99 and macro lift vs full-set det.
    det_full = m_det["macro_f1"]
    best_sel = max(
        selective["mc_dropout"] + selective["deterministic"],
        key=lambda d: (d["macro_f1"] or 0.0),
    )
    decision = "RUN_DOCUMENTED"
    decision_note = (
        "Post-hoc MC-dropout / entropy selective classification. "
        "Package keeps full-coverage argmax unless selective shows clear lift at high coverage."
    )
    # Prefer MC 0.99 coverage if available
    mc99 = next((d for d in selective["mc_dropout"] if abs(d["keep_frac_target"] - 0.99) < 1e-9), None)
    if mc99 and mc99["macro_f1"] is not None and mc99["macro_f1"] > det_full + 0.005:
        decision_note += f" MC@0.99 coverage macro {mc99['macro_f1']:.4f} > det {det_full:.4f}."
    else:
        decision_note += f" No robust high-coverage lift over det full-set {det_full:.4f}; keep argmax."

    elapsed = time.time() - t0
    env = make_result_envelope(
        experiment_id=f"cstar_{rid}_{name}_seed{seed}",
        protocol_id=bundle.protocol_id,
        stage="stage_b_ft",
        seed=seed,
        config={
            "row": rid,
            "name": name,
            "tracker": "C10",
            "mode": "posthoc_mc_dropout_entropy",
            "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
            "mc_passes": mc_passes,
            "load_ok": ok,
        },
        metrics={
            "best_val_macro_f1": float(m_det["macro_f1"]),
            "val_deterministic": m_det,
            "val_mc_dropout": m_mc,
            "selective_classification": selective,
            "best_selective": best_sel,
        },
        extra={
            "elapsed_sec": elapsed,
            "allow_test": False,
            "device": str(device),
            "decision": decision,
            "decision_note": decision_note,
            "data_summary": bundle.summary(),
            "git_sha": git_sha(),
        },
        project_root=PROJECT_ROOT,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(env, f, indent=2)
    print(
        f"  C10 DONE det={m_det['macro_f1']:.4f} mc={m_mc['macro_f1']:.4f} "
        f"best_sel={best_sel.get('macro_f1')} cov={best_sel.get('coverage')}",
        flush=True,
    )
    return {
        "row": rid,
        "name": name,
        "tracker": "C10",
        "best_val_macro_f1": float(m_det["macro_f1"]),
        "val_mc_macro_f1": float(m_mc["macro_f1"]),
        "best_selective_macro_f1": best_sel.get("macro_f1"),
        "best_selective_coverage": best_sel.get("coverage"),
        "decision": decision,
        "results_path": str(results_path),
        "returncode": 0,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Bounded C* playlist")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--rows", type=str, default="CTRL,C4,C5,C7,C8,C10")
    p.add_argument("--supcon-weight", type=float, default=0.1)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    wanted = {x.strip().upper() for x in args.rows.split(",") if x.strip()}
    plan = [r for r in ROWS if r["row"] in wanted]
    if not plan:
        print("No rows selected", file=sys.stderr)
        return 2

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device} | rows={[r['row'] for r in plan]}", flush=True)
    # champion guard
    import hashlib

    if CHAMPION.is_file():
        h = hashlib.md5(CHAMPION.read_bytes()).hexdigest()
        print(f"Champion md5: {h}", flush=True)

    started = datetime.now(timezone.utc)
    print("Loading botiot stage_b_ft…", flush=True)
    bundle = load_botiot(stage="stage_b_ft", seed=args.seed)
    base_cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for row in plan:
        print("=" * 60, f"{row['row']} {row['name']}", flush=True)
        if row["row"] == "C10":
            results.append(
                run_c10_uncertainty(
                    bundle=bundle, base_cfg=base_cfg, device=device, seed=args.seed
                )
            )
        else:
            results.append(
                train_row(
                    row=row,
                    bundle=bundle,
                    base_cfg=base_cfg,
                    device=device,
                    seed=args.seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    supcon_weight=args.supcon_weight,
                    skip_existing=args.skip_existing,
                )
            )

    finished = datetime.now(timezone.utc)
    # ranking for trainable rows with F1
    ranked = sorted(
        [r for r in results if r.get("best_val_macro_f1") is not None and r["row"] != "C10"],
        key=lambda r: r["best_val_macro_f1"],
        reverse=True,
    )
    ctrl = next((r for r in results if r["row"] == "CTRL"), None)
    ctrl_f1 = ctrl.get("best_val_macro_f1") if ctrl else None

    decisions = {}
    for r in results:
        rid = r["row"]
        if rid == "C10":
            decisions[rid] = {
                "status": r.get("decision", "RUN_DOCUMENTED"),
                "note": "see JSON decision_note",
            }
            continue
        f1 = r.get("best_val_macro_f1")
        if f1 is None:
            decisions[rid] = {"status": "BLOCKED", "note": "missing F1"}
            continue
        # incorporate only if beats control by ≥0.005 macro (bounded claim bar)
        if ctrl_f1 is not None and f1 >= ctrl_f1 + 0.005 and rid != "CTRL":
            decisions[rid] = {
                "status": "INCORPORATED_CANDIDATE",
                "note": f"Δ={f1 - ctrl_f1:+.4f} vs CTRL {ctrl_f1:.4f} — review for package",
            }
        elif rid == "CTRL":
            decisions[rid] = {"status": "CONTROL", "note": f"CTRL F1={f1:.4f}"}
        else:
            decisions[rid] = {
                "status": "RUN_DOCUMENTED",
                "note": (
                    f"F1={f1:.4f}"
                    + (f" vs CTRL {ctrl_f1:.4f} (Δ={f1 - ctrl_f1:+.4f})" if ctrl_f1 else "")
                    + " — no package incorporation"
                ),
            }

    summary = {
        "experiment_id": "bounded_cstar_playlist",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "supcon_weight": args.supcon_weight,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_sec": (finished - started).total_seconds(),
        "git_sha": git_sha(),
        "rows_requested": [r["row"] for r in plan],
        "n_success": sum(1 for r in results if r.get("returncode", 1) == 0),
        "ranking_val_macro_f1": ranked,
        "control_f1": ctrl_f1,
        "decisions": decisions,
        "results": results,
        "comparators": {
            "wp1b_mean": 0.9714,
            "hpo_seed42": 0.9791,
            "a7_ladder": 0.9699,
            "g11_cnn_bilstm": 0.9493,
        },
        "note": (
            "Bounded C* playlist. Test sealed. Champion not overwritten. "
            "Arch variants (C4/C5) train from scratch (no V3 weight transfer). "
            "C7/C8 use distill init + HPO-ish HPs when applicable. "
            "C10 is post-hoc selective classification only."
        ),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== C* BOUNDED DONE ===", flush=True)
    print(json.dumps({"decisions": decisions, "ranking": [
        (r["row"], r.get("best_val_macro_f1")) for r in ranked
    ]}, indent=2), flush=True)
    print(f"summary → {OUT_DIR / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
