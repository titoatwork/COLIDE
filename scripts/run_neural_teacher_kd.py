#!/usr/bin/env python3
"""
E6 — Bounded neural teacher KD (val-only; test sealed).

Teacher: frozen protocol neural checkpoint (default G11 cnn_bilstm CE 0.9493, or
optional V3/HPO ckpt). Soft labels via temperature-scaled teacher softmax on
stage_a_kd train. Student: V3 CNN–BiLSTM–Attention trained with α·KL + (1-α)·focal.

Compares against WP4b classical teachers (ensemble student 0.9401).

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_neural_teacher_kd.py
  PYTHONPATH=. .venv/bin/python scripts/run_neural_teacher_kd.py --teacher-ckpt model/baselines_neural/G11_cnn_bilstm_seed42.pth --teacher-variant cnn_bilstm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.losses import FocalLoss  # noqa: E402
from scripts.protocol.metrics import compute_classification_metrics  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402

CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "teachers_kd_neural"
CKPT_DIR = PROJECT_ROOT / "model" / "teachers_kd_neural"


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


def build_teacher(variant: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    if variant in ("cnn_bilstm", "g11"):
        from model.neural_baselines import build_neural_baseline

        return build_neural_baseline("cnn_bilstm", cfg, device)
    if variant in ("v3", "attn"):
        import sys as _sys
        from pathlib import Path as _P

        md = _P(PROJECT_ROOT) / "model"
        if str(md) not in _sys.path:
            _sys.path.insert(0, str(md))
        from cnn_bilstm_v3_attention import CNNBiLSTMAttention

        return CNNBiLSTMAttention(cfg).to(device)
    raise ValueError(variant)


def build_student(cfg: dict, device: torch.device) -> torch.nn.Module:
    import sys as _sys
    from pathlib import Path as _P

    md = _P(PROJECT_ROOT) / "model"
    if str(md) not in _sys.path:
        _sys.path.insert(0, str(md))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    return CNNBiLSTMAttention(cfg).to(device)


def load_sd(model: torch.nn.Module, path: Path, device: torch.device) -> bool:
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and any(torch.is_tensor(v) for v in raw.values()):
        state = {k: v for k, v in raw.items() if torch.is_tensor(v)}
    else:
        return False
    cleaned = {}
    for k, v in state.items():
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


@torch.no_grad()
def teacher_soft_labels(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32)).to(device)
        logits = model(xb).float()
        # temperature on logits then softmax
        probs = torch.softmax(logits / temperature, dim=-1)
        outs.append(probs.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


@torch.no_grad()
def eval_model(model, loader, device, class_names):
    model.eval()
    preds, targets = [], []
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(yb.numpy())
    return compute_classification_metrics(
        np.concatenate(targets), np.concatenate(preds), class_names
    )


def main() -> int:
    p = argparse.ArgumentParser(description="E6 neural teacher KD")
    p.add_argument(
        "--teacher-ckpt",
        type=str,
        default="model/baselines_neural/G11_cnn_bilstm_seed42.pth",
    )
    p.add_argument("--teacher-variant", type=str, default="cnn_bilstm")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    started = datetime.now(timezone.utc)
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    teacher_ckpt = PROJECT_ROOT / args.teacher_ckpt
    if not teacher_ckpt.is_file():
        print(f"Missing teacher ckpt: {teacher_ckpt}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CKPT_DIR / (
        f"kd_neural_{args.teacher_variant}_a{args.alpha}_T{args.temperature}"
        f"_g{args.focal_gamma}_seed{args.seed}.pth"
    )
    results_path = OUT_DIR / f"kd_neural_{args.teacher_variant}_seed{args.seed}.json"

    print(f"E6 neural teacher KD | device={device}", flush=True)
    print(f"Teacher ckpt: {teacher_ckpt}", flush=True)
    if CHAMPION.is_file():
        print(f"Champion md5: {hashlib.md5(CHAMPION.read_bytes()).hexdigest()}", flush=True)

    bundle = load_botiot(stage="stage_a_kd", seed=args.seed)
    cfg = load_config()

    # Teacher
    teacher = build_teacher(args.teacher_variant, cfg, device)
    ok = load_sd(teacher, teacher_ckpt, device)
    print(f"Teacher load: {'ok' if ok else 'FAIL'}", flush=True)
    if not ok:
        return 3
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    # Teacher val (hard argmax on stage_a val for reference)
    val_loader_t = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(bundle.X_val, dtype=np.float32)),
            torch.from_numpy(np.asarray(bundle.y_val, dtype=np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    t_preds, t_tgts = [], []
    with torch.no_grad():
        for xb, yb in val_loader_t:
            xb = xb.to(device)
            t_preds.append(teacher(xb).argmax(1).cpu().numpy())
            t_tgts.append(yb.numpy())
    teacher_val = compute_classification_metrics(
        np.concatenate(t_tgts), np.concatenate(t_preds), bundle.class_names
    )
    print(
        f"Teacher val macro-F1={teacher_val['macro_f1']:.4f} "
        f"min={teacher_val['min_per_class_f1']:.4f}",
        flush=True,
    )

    soft = teacher_soft_labels(
        teacher, bundle.X_train, device, args.batch_size, args.temperature
    )
    print(f"Soft labels shape={soft.shape}", flush=True)

    # Student train
    student = build_student(cfg, device)
    train_ds = TensorDataset(
        torch.from_numpy(np.asarray(bundle.X_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.y_train, dtype=np.int64)),
        torch.from_numpy(soft),
    )
    val_ds = TensorDataset(
        torch.from_numpy(np.asarray(bundle.X_val, dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.y_val, dtype=np.int64)),
        torch.zeros((len(bundle.y_val), soft.shape[1]), dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"
    )

    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = Adam(student.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    T = args.temperature
    alpha = args.alpha

    best_f1 = -1.0
    best_val = None
    patience_left = args.patience
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        student.train()
        running = 0.0
        n = 0
        for xb, yb, sb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = student(xb)
                hard = criterion(logits, yb)
                log_p = F.log_softmax(logits.float() / T, dim=-1)
                soft_loss = F.kl_div(log_p, sb.float(), reduction="batchmean") * (T * T)
                loss = alpha * soft_loss + (1.0 - alpha) * hard
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        scheduler.step()
        val_m = eval_model(student, val_loader, device, bundle.class_names)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(n, 1),
                "val_macro_f1": val_m["macro_f1"],
                "val_min_per_class_f1": val_m["min_per_class_f1"],
                "val_theft_f1": val_m.get("theft_f1"),
            }
        )
        print(
            f"  ep{epoch:02d} loss={history[-1]['train_loss']:.4f} "
            f"val_macro_f1={val_m['macro_f1']:.4f} min={val_m['min_per_class_f1']:.4f} "
            f"theft={val_m.get('theft_f1', float('nan')):.4f}",
            flush=True,
        )
        if val_m["macro_f1"] > best_f1:
            best_f1 = float(val_m["macro_f1"])
            best_val = val_m
            patience_left = args.patience
            torch.save(student.state_dict(), save_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    student.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    val_final = eval_model(student, val_loader, device, bundle.class_names)
    elapsed = time.time() - t0

    # Compare to WP4b ensemble
    wp4b_ensemble = 0.9401
    decision = "RUN_DOCUMENTED"
    decision_note = (
        f"Neural teacher ({args.teacher_variant}) student val macro-F1={best_f1:.4f}. "
        f"WP4b ensemble student={wp4b_ensemble:.4f}. "
    )
    if best_f1 >= wp4b_ensemble + 0.005:
        decision = "INCORPORATED_CANDIDATE"
        decision_note += "Beats ensemble by ≥0.005 — review for package KD teacher."
    else:
        decision_note += "Does not beat ensemble teacher path; keep ensemble INCORPORATED."

    finished = datetime.now(timezone.utc)
    env = make_result_envelope(
        experiment_id=f"e6_neural_teacher_kd_{args.teacher_variant}_seed{args.seed}",
        protocol_id=bundle.protocol_id,
        stage="stage_a_kd",
        seed=args.seed,
        config={
            "teacher_ckpt": str(teacher_ckpt.relative_to(PROJECT_ROOT)),
            "teacher_variant": args.teacher_variant,
            "teacher_ckpt_md5": hashlib.md5(teacher_ckpt.read_bytes()).hexdigest(),
            "alpha": args.alpha,
            "temperature": args.temperature,
            "focal_gamma": args.focal_gamma,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "student": "cnn_bilstm_v3_attention",
            "save_path": str(save_path),
        },
        metrics={
            "best_val_macro_f1": float(best_f1),
            "val": val_final,
            "teacher_val": {
                "macro_f1": teacher_val["macro_f1"],
                "min_per_class_f1": teacher_val["min_per_class_f1"],
                "theft_f1": teacher_val.get("theft_f1"),
            },
        },
        extra={
            "history": history,
            "elapsed_sec": elapsed,
            "allow_test": False,
            "device": str(device),
            "best_val_snapshot": best_val,
            "decision": decision,
            "decision_note": decision_note,
            "comparators": {
                "wp4b_ensemble_student": wp4b_ensemble,
                "wp4b_rf_student": 0.9346,
                "wp4b_none_student": 0.9326,
            },
            "data_summary": bundle.summary(),
            "git_sha": git_sha(),
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
        },
        project_root=PROJECT_ROOT,
    )
    with open(results_path, "w") as f:
        json.dump(env, f, indent=2)

    summary = {
        "experiment_id": "e6_neural_teacher_kd",
        "tracker": "E6",
        "student_val_macro_f1": float(best_f1),
        "teacher_val_macro_f1": float(teacher_val["macro_f1"]),
        "decision": decision,
        "decision_note": decision_note,
        "results_path": str(results_path.relative_to(PROJECT_ROOT)),
        "checkpoint_path": str(save_path.relative_to(PROJECT_ROOT)),
        "wp4b_ensemble_student": wp4b_ensemble,
        "wall_sec": elapsed,
        "champion_md5": hashlib.md5(CHAMPION.read_bytes()).hexdigest()
        if CHAMPION.is_file()
        else None,
        "note": "stage_a_kd student numbers — do not mix with stage_b_ft multirun means",
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== E6 DONE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
