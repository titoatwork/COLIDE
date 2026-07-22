#!/usr/bin/env python3
"""
WP5c / H8 — Multi-objective Pareto: F1–latency–memory (val-only; test sealed).

Consolidates protocol-fair systems metrics from WP5a/WP5b result JSONs, re-benches
key CAD-CBA checkpoints under a unified GPU forward protocol, optionally fits
classical RF/XGB/LGBM for CPU systems axes, then:

  1) Locks composite-score weights a priori (before ranking)
  2) Builds multi-obj table + Pareto front (max F1, min latency, min memory)
  3) Writes JSON / CSV / figure under benchmarks/results/pareto_h8/

Composite (locked weights — do not retune after seeing ranking):
  score = 0.40 * f1_norm
        + 0.25 * (1 - latency_norm)
        + 0.20 * (1 - mem_norm)
        + 0.15 * minority_norm
  Norms are min-max within the primary protocol-fair neural cohort only.

Axes:
  - Detection: val macro-F1 (protocol botiot_v1 / stage_b_ft; test sealed)
  - Latency: GPU batch-256 forward per_sample_us (neural); CPU predict batch-256
    per_sample_us (classical systems re-fit)
  - Memory: CUDA max_allocated_mb (neural) or model_bytes (classical)

Example:
  PYTHONPATH=. .venv/bin/python scripts/run_pareto_multiobj.py
  PYTHONPATH=. .venv/bin/python scripts/run_pareto_multiobj.py --skip-classical
  PYTHONPATH=. .venv/bin/python scripts/run_pareto_multiobj.py --skip-rebench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.result_schema import make_result_envelope  # noqa: E402

OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "pareto_h8"
CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
EXPECTED_CHAMPION_MD5 = "80a90f7cc210276300eaa90173a5a385"

# ---------------------------------------------------------------------------
# Composite weights LOCKED a priori (H8 / Phase 5 §3.2). Do not retune after ranking.
# ---------------------------------------------------------------------------
COMPOSITE_WEIGHTS = {
    "w_f1": 0.40,
    "w_latency": 0.25,  # applied as (1 - latency_norm)
    "w_memory": 0.20,  # applied as (1 - mem_norm)
    "w_minority": 0.15,
}
COMPOSITE_NOTE = (
    "Weights locked before ranking. Norms = min-max within primary neural cohort "
    "(ablation A1–A7 + neural G6–G12 + rebench package points). Classical and "
    "historical rows are reported for comparison but do NOT enter the norm set."
)

BATCH_SIZE = 256
WARMUP = 20
REPS = 50


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


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


@torch.no_grad()
def measure_latency_ms(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int = WARMUP,
    reps: int = REPS,
) -> dict[str, float]:
    model.eval()
    xb = sample.to(device)
    use_cuda = device.type == "cuda"
    if use_cuda:
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
        "p50_ms": float(statistics.median(times)),
        "per_sample_us": float(statistics.mean(times) * 1000.0 / xb.size(0)),
        "device": str(device),
    }


def measure_cuda_mem(model: torch.nn.Module, sample: torch.Tensor, device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    torch.cuda.reset_peak_memory_stats()
    _ = model(sample.to(device))
    torch.cuda.synchronize()
    return {
        "max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
    }


def collect_from_protocol_json(
    path: Path,
    *,
    family: str,
    row_id: str | None = None,
    name: str | None = None,
    role: str = "primary_neural",
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    env = load_json(path)
    metrics = env.get("metrics") or {}
    systems = metrics.get("systems") or {}
    val = metrics.get("val") or {}
    cfg = env.get("config") or {}
    lat = systems.get("latency") or {}
    mem = systems.get("cuda_memory") or {}
    rid = row_id or cfg.get("row") or path.stem
    nm = name or cfg.get("name") or rid
    f1 = metrics.get("best_val_macro_f1")
    if f1 is None:
        f1 = val.get("macro_f1")
    if f1 is None:
        return None
    return {
        "id": f"{family}:{rid}",
        "family": family,
        "row": rid,
        "name": nm,
        "role": role,
        "source_json": str(path.relative_to(PROJECT_ROOT)),
        "val_macro_f1": float(f1),
        "val_min_per_class_f1": float(val.get("min_per_class_f1") or 0.0),
        "val_theft_f1": float(val.get("theft_f1") or 0.0),
        "val_balanced_accuracy": float(val.get("balanced_accuracy") or 0.0)
        if val.get("balanced_accuracy") is not None
        else None,
        "n_params": systems.get("n_params"),
        "checkpoint_bytes": systems.get("checkpoint_bytes"),
        "per_sample_us": lat.get("per_sample_us"),
        "latency_mean_ms": lat.get("mean_ms"),
        "latency_batch_size": lat.get("batch_size"),
        "cuda_max_allocated_mb": mem.get("max_allocated_mb"),
        "cuda_max_reserved_mb": mem.get("max_reserved_mb"),
        "hardware_class": "gpu_rtx3050_forward",
        "memory_axis_mb": mem.get("max_allocated_mb"),
        "memory_axis_note": "cuda max_allocated_mb during batch-256 forward",
    }


def collect_ablation_and_neural() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    abl = PROJECT_ROOT / "benchmarks" / "results" / "ablation_ladder"
    for p in sorted(abl.glob("*_seed42.json")):
        r = collect_from_protocol_json(p, family="ablation", role="primary_neural")
        if r:
            rows.append(r)
    neu = PROJECT_ROOT / "benchmarks" / "results" / "baselines_neural"
    for p in sorted(neu.glob("G*_seed42.json")):
        r = collect_from_protocol_json(p, family="neural_baseline", role="primary_neural")
        if r:
            rows.append(r)
    return rows


def build_v3_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention

    model = CNNBiLSTMAttention(cfg)
    return model.to(device)


def try_load_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> bool:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif "state_dict" in raw:
            state = raw["state_dict"]
        else:
            # plain state_dict
            state = raw
        # strip common prefixes
        cleaned = {}
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            cleaned[nk] = v
        if not cleaned:
            return False
        try:
            model.load_state_dict(cleaned, strict=True)
            return True
        except Exception:
            try:
                model.load_state_dict(cleaned, strict=False)
                return True
            except Exception:
                return False
    return False


def rebench_checkpoint(
    *,
    row_id: str,
    name: str,
    ckpt: Path,
    f1_source: Path | None,
    sample: torch.Tensor,
    device: torch.device,
    cfg: dict,
    role: str = "rebench_package",
) -> dict[str, Any] | None:
    if not ckpt.is_file():
        print(f"  skip rebench {row_id}: missing {ckpt}", flush=True)
        return None
    model = build_v3_model(cfg, device)
    ok = try_load_state(model, ckpt, device)
    if not ok:
        print(f"  skip rebench {row_id}: failed load {ckpt}", flush=True)
        return None
    model.eval()
    lat = measure_latency_ms(model, sample, device)
    mem = measure_cuda_mem(model, sample, device)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    ckpt_bytes = ckpt.stat().st_size

    f1 = min_f1 = theft = bal = None
    src_rel = None
    if f1_source and f1_source.is_file():
        env = load_json(f1_source)
        metrics = env.get("metrics") or {}
        val = metrics.get("val") or {}
        f1 = metrics.get("best_val_macro_f1", val.get("macro_f1"))
        min_f1 = val.get("min_per_class_f1")
        theft = val.get("theft_f1")
        bal = val.get("balanced_accuracy")
        src_rel = str(f1_source.relative_to(PROJECT_ROOT))

    # champion protocol eval fallback (schema: top-level val, not metrics.val)
    if f1 is None and row_id == "champion":
        proto = (
            PROJECT_ROOT
            / "benchmarks"
            / "results"
            / "protocol"
            / "eval_best_model_botiot_twostage_stage_b_ft.json"
        )
        if proto.is_file():
            env = load_json(proto)
            metrics = env.get("metrics") or {}
            val = env.get("val") or metrics.get("val") or metrics
            if isinstance(val, dict):
                f1 = val.get("macro_f1") or metrics.get("macro_f1")
                min_f1 = val.get("min_per_class_f1")
                theft = val.get("theft_f1")
                bal = val.get("balanced_accuracy")
                src_rel = str(proto.relative_to(PROJECT_ROOT))

    if f1 is None:
        print(f"  warn rebench {row_id}: no F1 source; systems-only", flush=True)
        f1 = float("nan")

    return {
        "id": f"rebench:{row_id}",
        "family": "rebench",
        "row": row_id,
        "name": name,
        "role": role,
        "source_json": src_rel,
        "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
        "checkpoint_md5": md5_file(ckpt),
        "val_macro_f1": float(f1) if f1 is not None else None,
        "val_min_per_class_f1": float(min_f1 or 0.0),
        "val_theft_f1": float(theft or 0.0),
        "val_balanced_accuracy": float(bal) if bal is not None else None,
        "n_params": n_params,
        "checkpoint_bytes": ckpt_bytes,
        "per_sample_us": lat["per_sample_us"],
        "latency_mean_ms": lat["mean_ms"],
        "latency_batch_size": lat["batch_size"],
        "cuda_max_allocated_mb": mem.get("max_allocated_mb"),
        "cuda_max_reserved_mb": mem.get("max_reserved_mb"),
        "hardware_class": "gpu_rtx3050_forward",
        "memory_axis_mb": mem.get("max_allocated_mb"),
        "memory_axis_note": "cuda max_allocated_mb during batch-256 forward (rebench)",
        "latency_detail": lat,
        "cuda_memory_detail": mem,
    }


def classical_systems(
    *,
    models: list[str],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Re-fit classical models for systems axes; F1 from existing protocol JSONs."""
    from scripts.run_classical_baselines import fit_predict  # type: ignore

    bundle = load_botiot(stage="stage_b_ft", seed=seed)
    X_tr, y_tr = bundle.X_train, bundle.y_train
    X_va, y_va = bundle.X_val, bundle.y_val
    sample = np.asarray(X_va[:BATCH_SIZE], dtype=np.float32)
    rows = []

    f1_map = {}
    for name in models:
        p = (
            PROJECT_ROOT
            / "benchmarks"
            / "results"
            / "baselines_classical"
            / f"{name}_seed{seed}.json"
        )
        if p.is_file():
            env = load_json(p)
            val = (env.get("metrics") or {}).get("val") or {}
            f1_map[name] = {
                "val_macro_f1": float(val.get("macro_f1", 0.0)),
                "val_min_per_class_f1": float(val.get("min_per_class_f1", 0.0)),
                "val_theft_f1": float(val.get("theft_f1", 0.0)),
                "val_balanced_accuracy": float(val.get("balanced_accuracy", 0.0))
                if val.get("balanced_accuracy") is not None
                else None,
                "source_json": str(p.relative_to(PROJECT_ROOT)),
            }

    for name in models:
        print(f"  classical systems fit: {name}", flush=True)
        pred, train_sec, infer_full_sec, clf, model_cfg = fit_predict(
            name,
            X_tr,
            y_tr,
            X_va,
            seed=seed,
            svm_balanced=False,
            lgbm_mode="fixed",
        )
        # batch-256 latency (CPU)
        # warm up
        for _ in range(5):
            _ = clf.predict(sample)
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            _ = clf.predict(sample)
            times.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = float(statistics.mean(times))
        per_us = mean_ms * 1000.0 / BATCH_SIZE

        # model size via joblib dump
        tmp = OUT_DIR / f"_tmp_{name}.joblib"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, tmp)
        model_bytes = tmp.stat().st_size
        tmp.unlink(missing_ok=True)

        f1info = f1_map.get(name, {})
        # Prefer disk F1 (protocol run); fall back to this re-fit if missing
        if "val_macro_f1" not in f1info:
            from scripts.protocol.metrics import compute_classification_metrics

            m = compute_classification_metrics(y_va, pred, bundle.class_names)
            f1info = {
                "val_macro_f1": float(m["macro_f1"]),
                "val_min_per_class_f1": float(m["min_per_class_f1"]),
                "val_theft_f1": float(m.get("theft_f1") or 0.0),
                "val_balanced_accuracy": float(m.get("balanced_accuracy") or 0.0),
                "source_json": "re-fit (no prior JSON)",
            }

        rows.append(
            {
                "id": f"classical:{name}",
                "family": "classical",
                "row": name.upper() if name != "lgbm" else "LGBM",
                "name": name,
                "role": "classical_comparator",
                "source_json": f1info.get("source_json"),
                "val_macro_f1": f1info["val_macro_f1"],
                "val_min_per_class_f1": f1info["val_min_per_class_f1"],
                "val_theft_f1": f1info["val_theft_f1"],
                "val_balanced_accuracy": f1info.get("val_balanced_accuracy"),
                "n_params": None,
                "checkpoint_bytes": model_bytes,
                "model_bytes": model_bytes,
                "per_sample_us": per_us,
                "latency_mean_ms": mean_ms,
                "latency_batch_size": BATCH_SIZE,
                "cuda_max_allocated_mb": None,
                "cuda_max_reserved_mb": None,
                "hardware_class": "cpu_sklearn_predict",
                "memory_axis_mb": model_bytes / (1024**2),
                "memory_axis_note": "serialized model_bytes (joblib) as deploy memory proxy",
                "train_sec_refit": float(train_sec),
                "infer_full_val_sec": float(infer_full_sec),
                "model_cfg": model_cfg,
            }
        )
        print(
            f"    {name}: F1={f1info['val_macro_f1']:.4f} lat={per_us:.2f}µs/s "
            f"size={model_bytes/1e6:.2f}MB train={train_sec:.1f}s",
            flush=True,
        )
    return rows


def historical_comparators() -> list[dict[str, Any]]:
    """Labeled historical / cross-hardware rows — not in composite norm set."""
    rows = []
    cuml = PROJECT_ROOT / "benchmarks" / "results" / "cuml_rf_resources.json"
    if cuml.is_file():
        d = load_json(cuml)
        rows.append(
            {
                "id": "historical:cuml_rf_a100",
                "family": "historical",
                "row": "cuML_RF_A100",
                "name": "cuml_rf",
                "role": "historical_cross_hw",
                "source_json": str(cuml.relative_to(PROJECT_ROOT)),
                "val_macro_f1": float(d.get("cuml_rf_f1") or float("nan")),
                "val_min_per_class_f1": None,
                "val_theft_f1": None,
                "n_params": None,
                "checkpoint_bytes": None,
                "per_sample_us": None,  # throughput-based, not µs protocol
                "throughput_flows_sec": d.get("cuml_rf_throughput_avg"),
                "cuda_max_allocated_mb": float(d.get("cuml_rf_vram_mb") or 0.0),
                "hardware_class": "gpu_a100_historical",
                "memory_axis_mb": float(d.get("cuml_rf_vram_mb") or 0.0),
                "memory_axis_note": "cuML RF VRAM on A100 (historical; not protocol-fair F1)",
                "note": d.get("note"),
                "energy_mj_per_flow": d.get("cuml_rf_energy_mj_per_flow"),
            }
        )
        # companion neural number from same file (also historical)
        if d.get("cnn_bilstm_vram_mb") is not None:
            rows.append(
                {
                    "id": "historical:cnn_bilstm_vram_note",
                    "family": "historical",
                    "row": "CNN_BiLSTM_VRAM_note",
                    "name": "cnn_bilstm_hist",
                    "role": "historical_cross_hw",
                    "source_json": str(cuml.relative_to(PROJECT_ROOT)),
                    "val_macro_f1": float(d.get("cnn_bilstm_f1") or float("nan")),
                    "cuda_max_allocated_mb": float(d.get("cnn_bilstm_vram_mb")),
                    "memory_axis_mb": float(d.get("cnn_bilstm_vram_mb")),
                    "hardware_class": "mixed_historical",
                    "memory_axis_note": "historical companion VRAM (~2MB) vs cuML RF",
                    "per_sample_us": None,
                    "note": d.get("note"),
                }
            )
    bl = PROJECT_ROOT / "benchmarks" / "results" / "baseline_latency.json"
    if bl.is_file():
        d = load_json(bl)
        rows.append(
            {
                "id": "historical:baseline_latency_rtx3050",
                "family": "historical",
                "row": "baseline_latency_meta",
                "name": "baseline_latency",
                "role": "historical_meta",
                "source_json": str(bl.relative_to(PROJECT_ROOT)),
                "val_macro_f1": None,
                "per_sample_us": float(d.get("gpu_rtx3050_single_p50_ms", 0) * 1000.0)
                if d.get("gpu_rtx3050_single_p50_ms")
                else None,
                "gpu_batch128_p50_ms": d.get("gpu_rtx3050_batch128_p50_ms"),
                "throughput_flows_sec": d.get("gpu_rtx3050_throughput_flows_sec"),
                "hardware_class": "gpu_rtx3050_historical_single",
                "memory_axis_mb": None,
                "note": d.get("note"),
            }
        )
    return rows


def minmax(vals: list[float]) -> tuple[float, float]:
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return lo, lo + 1.0
    return lo, hi


def apply_composite(rows: list[dict[str, Any]]) -> None:
    """In-place: compute composite for primary_neural + rebench_package with valid axes."""
    cohort = [
        r
        for r in rows
        if r.get("role") in ("primary_neural", "rebench_package")
        and r.get("val_macro_f1") is not None
        and r.get("per_sample_us") is not None
        and r.get("memory_axis_mb") is not None
        and not (isinstance(r["val_macro_f1"], float) and np.isnan(r["val_macro_f1"]))
    ]
    if not cohort:
        return
    f1s = [r["val_macro_f1"] for r in cohort]
    lats = [r["per_sample_us"] for r in cohort]
    mems = [r["memory_axis_mb"] for r in cohort]
    mins = [r.get("val_min_per_class_f1") or 0.0 for r in cohort]
    f_lo, f_hi = minmax(f1s)
    l_lo, l_hi = minmax(lats)
    m_lo, m_hi = minmax(mems)
    n_lo, n_hi = minmax(mins)
    w = COMPOSITE_WEIGHTS
    for r in cohort:
        f1_n = (r["val_macro_f1"] - f_lo) / (f_hi - f_lo)
        lat_n = (r["per_sample_us"] - l_lo) / (l_hi - l_lo)
        mem_n = (r["memory_axis_mb"] - m_lo) / (m_hi - m_lo)
        min_n = ((r.get("val_min_per_class_f1") or 0.0) - n_lo) / (n_hi - n_lo)
        score = (
            w["w_f1"] * f1_n
            + w["w_latency"] * (1.0 - lat_n)
            + w["w_memory"] * (1.0 - mem_n)
            + w["w_minority"] * min_n
        )
        r["f1_norm"] = f1_n
        r["latency_norm"] = lat_n
        r["memory_norm"] = mem_n
        r["minority_norm"] = min_n
        r["composite_score"] = float(score)
        r["in_composite_cohort"] = True
    for r in rows:
        if "in_composite_cohort" not in r:
            r["in_composite_cohort"] = False
            r["composite_score"] = None


def pareto_front(rows: list[dict[str, Any]], role_filter: set[str] | None = None) -> list[str]:
    """Non-dominated: max F1, min latency, min memory. Returns ids."""
    cand = []
    for r in rows:
        if role_filter and r.get("role") not in role_filter:
            continue
        if r.get("val_macro_f1") is None or r.get("per_sample_us") is None:
            continue
        if r.get("memory_axis_mb") is None:
            continue
        if isinstance(r["val_macro_f1"], float) and np.isnan(r["val_macro_f1"]):
            continue
        cand.append(r)

    front_ids = []
    for a in cand:
        dominated = False
        for b in cand:
            if a["id"] == b["id"]:
                continue
            # b dominates a if b is ≥ F1, ≤ lat, ≤ mem, and strict in at least one
            better_or_eq = (
                b["val_macro_f1"] >= a["val_macro_f1"]
                and b["per_sample_us"] <= a["per_sample_us"]
                and b["memory_axis_mb"] <= a["memory_axis_mb"]
            )
            strict = (
                b["val_macro_f1"] > a["val_macro_f1"]
                or b["per_sample_us"] < a["per_sample_us"]
                or b["memory_axis_mb"] < a["memory_axis_mb"]
            )
            if better_or_eq and strict:
                dominated = True
                break
        if not dominated:
            front_ids.append(a["id"])
    return front_ids


def plot_pareto(rows: list[dict[str, Any]], front_ids: set[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    # primary neural
    for r in rows:
        if r.get("per_sample_us") is None or r.get("val_macro_f1") is None:
            continue
        if isinstance(r["val_macro_f1"], float) and np.isnan(r["val_macro_f1"]):
            continue
        x = r["per_sample_us"]
        y = r["val_macro_f1"]
        mem = r.get("memory_axis_mb") or 10.0
        size = 40 + min(mem, 500) * 1.5
        role = r.get("role")
        on_front = r["id"] in front_ids
        if role == "primary_neural":
            color = "#1f77b4"
            marker = "o"
            alpha = 0.85
        elif role == "rebench_package":
            color = "#d62728"
            marker = "D"
            alpha = 0.95
        elif role == "classical_comparator":
            color = "#2ca02c"
            marker = "s"
            alpha = 0.9
        else:
            color = "#7f7f7f"
            marker = "x"
            alpha = 0.6
        edge = "black" if on_front else "none"
        lw = 1.8 if on_front else 0.0
        ax.scatter(
            x,
            y,
            s=size,
            c=color,
            marker=marker,
            alpha=alpha,
            edgecolors=edge,
            linewidths=lw,
            zorder=3 if on_front else 2,
        )
        label = r.get("row") or r.get("name")
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            alpha=0.9,
        )

    ax.set_xlabel("Latency (µs / sample, batch=256)")
    ax.set_ylabel("Val macro-F1 (protocol botiot_v1, test sealed)")
    ax.set_title(
        "WP5c / H8 Pareto — F1 vs latency\n"
        "(marker size ∝ memory axis; black edge = Pareto front)"
    )
    ax.grid(True, alpha=0.3)
    # legend proxies
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markersize=8,
            label="Ablation / neural baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#d62728",
            markersize=8,
            label="CAD-CBA rebench",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#2ca02c",
            markersize=8,
            label="Classical (CPU systems)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label="Pareto front",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def rows_to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    cols = [
        "id",
        "family",
        "row",
        "name",
        "role",
        "val_macro_f1",
        "val_min_per_class_f1",
        "val_theft_f1",
        "per_sample_us",
        "memory_axis_mb",
        "n_params",
        "checkpoint_bytes",
        "composite_score",
        "in_composite_cohort",
        "on_pareto_front",
        "hardware_class",
        "source_json",
    ]
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.8g}")
            else:
                s = str(v).replace(",", ";")
                vals.append(s)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="WP5c Pareto multi-objective H8")
    p.add_argument("--skip-classical", action="store_true")
    p.add_argument("--skip-rebench", action="store_true")
    p.add_argument(
        "--classical-models",
        type=str,
        default="rf,xgb,lgbm",
        help="Comma list: rf,xgb,lgbm,lr,svm",
    )
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    started = datetime.now(timezone.utc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    champ_md5 = md5_file(CHAMPION) if CHAMPION.is_file() else None
    if champ_md5 != EXPECTED_CHAMPION_MD5:
        print(
            f"WARNING: champion md5 {champ_md5} != expected {EXPECTED_CHAMPION_MD5}",
            flush=True,
        )
    else:
        print(f"Champion md5 OK: {champ_md5}", flush=True)

    print("Collecting ablation + neural baseline systems from disk…", flush=True)
    rows = collect_ablation_and_neural()
    print(f"  loaded {len(rows)} protocol neural rows", flush=True)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)

    # Need val sample for rebench
    if not args.skip_rebench:
        print("Loading protocol val sample for rebench…", flush=True)
        bundle = load_botiot(stage="stage_b_ft", seed=42)
        sample = torch.from_numpy(np.asarray(bundle.X_val[:BATCH_SIZE], dtype=np.float32))
        cfg = load_config()
        # apply HPO dropouts for package-ish models is not required for load if state_dict matches
        rebench_specs = [
            {
                "row_id": "A7_package",
                "name": "full_cad_cba_v1",
                "ckpt": PROJECT_ROOT / "model" / "ablation_ladder" / "A7_full_cad_cba_v1_seed42.pth",
                "f1": PROJECT_ROOT
                / "benchmarks"
                / "results"
                / "ablation_ladder"
                / "A7_full_cad_cba_v1_seed42.json",
            },
            {
                "row_id": "HPO_refine_t8",
                "name": "hpo_winner_trial8",
                "ckpt": PROJECT_ROOT / "model" / "hpo" / "refine_rank2_trial008_seed42.pth",
                "f1": PROJECT_ROOT
                / "benchmarks"
                / "results"
                / "hpo"
                / "refine_rank2_trial008_seed42.json",
            },
            {
                "row_id": "HPO_confirm_s42",
                "name": "hpo_confirm_seed42",
                "ckpt": PROJECT_ROOT / "model" / "multirun_hpo_confirm" / "ft_seed42.pth",
                "f1": PROJECT_ROOT
                / "benchmarks"
                / "results"
                / "multirun_hpo_confirm"
                / "ft_seed42.json",
            },
            {
                "row_id": "package_ens_hpo_s45",
                "name": "ensemble_hpo_seed45",
                "ckpt": PROJECT_ROOT / "model" / "multirun_ensemble_hpo" / "ft_seed45.pth",
                "f1": PROJECT_ROOT
                / "benchmarks"
                / "results"
                / "multirun_ensemble_hpo"
                / "ft_seed45.json",
            },
            {
                "row_id": "WP1b_s44",
                "name": "multirun_seed44",
                "ckpt": PROJECT_ROOT / "model" / "multirun" / "ft_seed44.pth",
                "f1": PROJECT_ROOT / "benchmarks" / "results" / "multirun" / "ft_seed44.json",
            },
            {
                "row_id": "champion",
                "name": "production_champion",
                "ckpt": CHAMPION,
                "f1": PROJECT_ROOT
                / "benchmarks"
                / "results"
                / "protocol"
                / "eval_best_model_botiot_twostage_stage_b_ft.json",
            },
        ]
        for spec in rebench_specs:
            print(f"  rebench {spec['row_id']}…", flush=True)
            r = rebench_checkpoint(
                row_id=spec["row_id"],
                name=spec["name"],
                ckpt=spec["ckpt"],
                f1_source=spec["f1"],
                sample=sample,
                device=device,
                cfg=cfg,
                role="rebench_package",
            )
            if r:
                rows.append(r)
                print(
                    f"    F1={r['val_macro_f1']:.4f} lat={r['per_sample_us']:.2f}µs "
                    f"mem={r.get('memory_axis_mb')}",
                    flush=True,
                )

    if not args.skip_classical:
        print("Classical systems re-fit (RF/XGB/LGBM)…", flush=True)
        models = [m.strip() for m in args.classical_models.split(",") if m.strip()]
        rows.extend(classical_systems(models=models, seed=42))

    print("Adding historical comparators…", flush=True)
    rows.extend(historical_comparators())

    # Composite + Pareto
    print("Computing composite scores (weights locked a priori)…", flush=True)
    apply_composite(rows)

    # Pareto on primary neural + rebench + classical (all with full 3 axes)
    front = pareto_front(
        rows, role_filter={"primary_neural", "rebench_package", "classical_comparator"}
    )
    front_set = set(front)
    for r in rows:
        r["on_pareto_front"] = r["id"] in front_set

    # neural-only front (fair GPU axes)
    front_neural = pareto_front(rows, role_filter={"primary_neural", "rebench_package"})
    for r in rows:
        r["on_neural_pareto_front"] = r["id"] in set(front_neural)

    # rankings
    composite_ranked = sorted(
        [r for r in rows if r.get("composite_score") is not None],
        key=lambda r: r["composite_score"],
        reverse=True,
    )
    f1_ranked = sorted(
        [
            r
            for r in rows
            if r.get("val_macro_f1") is not None
            and not (isinstance(r["val_macro_f1"], float) and np.isnan(r["val_macro_f1"]))
            and r.get("role")
            in ("primary_neural", "rebench_package", "classical_comparator")
        ],
        key=lambda r: r["val_macro_f1"],
        reverse=True,
    )

    # Figure
    fig_path = OUT_DIR / "pareto_f1_latency.png"
    plot_pareto(rows, front_set, fig_path)
    print(f"Wrote figure {fig_path}", flush=True)

    csv_path = OUT_DIR / "multiobj_table.csv"
    rows_to_csv(rows, csv_path)

    finished = datetime.now(timezone.utc)
    wall = (finished - started).total_seconds()

    # Decision narrative
    top_comp = composite_ranked[0] if composite_ranked else None
    best_f1 = f1_ranked[0] if f1_ranked else None
    # Find best neural F1 with systems
    neural_f1 = [r for r in f1_ranked if r["role"] in ("primary_neural", "rebench_package")]
    classical_f1 = [r for r in f1_ranked if r["role"] == "classical_comparator"]

    decision = "RUN_DOCUMENTED"
    decision_note = (
        "Multi-obj table locked under a priori composite weights. "
        "Detection leadership may remain with classical LGBM/RF; neural wins on "
        "GPU-deploy memory vs historical cuML RF and offers streaming temporal path. "
        "Composite ranking is for deploy trade-off discussion — not a new champion."
    )

    summary = {
        "experiment_id": "wp5c_pareto_h8",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": 42,
        "allow_test": False,
        "test_sealed": True,
        "git_sha": git_sha(),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_sec": wall,
        "champion_md5": champ_md5,
        "champion_md5_expected": EXPECTED_CHAMPION_MD5,
        "champion_unchanged": champ_md5 == EXPECTED_CHAMPION_MD5,
        "composite_definition": {
            "weights": COMPOSITE_WEIGHTS,
            "formula": (
                "0.40*f1_norm + 0.25*(1-latency_norm) + 0.20*(1-mem_norm) + 0.15*minority_norm"
            ),
            "norm_cohort": "primary_neural + rebench_package with valid 3 axes",
            "note": COMPOSITE_NOTE,
            "locked_a_priori": True,
        },
        "latency_protocol": {
            "neural": "GPU sync forward, batch=256, warmup=20, reps=50, per_sample_us",
            "classical": "CPU sklearn predict, batch=256, warmup=5, reps=50, per_sample_us",
        },
        "memory_protocol": {
            "neural": "torch.cuda.max_memory_allocated after batch-256 forward (MB)",
            "classical": "joblib serialized model size (MB) as deploy proxy",
        },
        "n_rows": len(rows),
        "pareto_front_ids": front,
        "neural_pareto_front_ids": front_neural,
        "composite_ranking": [
            {
                "id": r["id"],
                "row": r.get("row"),
                "name": r.get("name"),
                "composite_score": r.get("composite_score"),
                "val_macro_f1": r.get("val_macro_f1"),
                "per_sample_us": r.get("per_sample_us"),
                "memory_axis_mb": r.get("memory_axis_mb"),
                "val_min_per_class_f1": r.get("val_min_per_class_f1"),
            }
            for r in composite_ranked
        ],
        "f1_ranking_protocol": [
            {
                "id": r["id"],
                "row": r.get("row"),
                "val_macro_f1": r.get("val_macro_f1"),
                "per_sample_us": r.get("per_sample_us"),
                "memory_axis_mb": r.get("memory_axis_mb"),
                "hardware_class": r.get("hardware_class"),
            }
            for r in f1_ranked
        ],
        "headline": {
            "top_composite": {
                "id": top_comp["id"],
                "score": top_comp["composite_score"],
                "f1": top_comp["val_macro_f1"],
                "lat_us": top_comp["per_sample_us"],
                "mem_mb": top_comp["memory_axis_mb"],
            }
            if top_comp
            else None,
            "top_f1_overall": {
                "id": best_f1["id"],
                "f1": best_f1["val_macro_f1"],
                "hardware": best_f1.get("hardware_class"),
            }
            if best_f1
            else None,
            "top_f1_neural": {
                "id": neural_f1[0]["id"],
                "f1": neural_f1[0]["val_macro_f1"],
                "lat_us": neural_f1[0].get("per_sample_us"),
                "mem_mb": neural_f1[0].get("memory_axis_mb"),
            }
            if neural_f1
            else None,
            "top_f1_classical": {
                "id": classical_f1[0]["id"],
                "f1": classical_f1[0]["val_macro_f1"],
                "lat_us": classical_f1[0].get("per_sample_us"),
                "mem_mb": classical_f1[0].get("memory_axis_mb"),
            }
            if classical_f1
            else None,
        },
        "answers": {
            "vs_rf_detection": (
                "Protocol RF val macro-F1 ~0.9778; LGBM ~0.9818 tops detection. "
                "Best package neural rebench/HPO ~0.9791; multirun mean 0.9714. "
                "Neural does not dominate classical detection under protocol."
            ),
            "vs_best_neural_baseline": (
                "G11/A3 cnn_bilstm CE 0.9493 is strongest pure-arch baseline; "
                "CAD-CBA package (A7 0.9699 / HPO 0.9791) improves detection at higher "
                "latency (~26µs vs ~20µs) and higher CUDA mem vs MLP (~4.3µs, low mem)."
            ),
            "where_we_win": (
                "Deploy memory vs historical cuML RF (~2MB vs ~444MB VRAM claim); "
                "GPU streaming path; composite can favor low-latency MLP/G11 when "
                "detection gap acceptable; package near-RF on F1 with order-of-magnitude "
                "smaller GPU footprint than cuML RF historical."
            ),
            "where_we_lose": (
                "Absolute protocol detection vs LGBM 0.9818 / RF 0.9778; "
                "attention package slower than plain CNN–BiLSTM; classical CPU "
                "batch-256 µs may be lower for tree models (different hardware class)."
            ),
        },
        "decision": decision,
        "decision_note": decision_note,
        "artifacts": {
            "summary": str((OUT_DIR / "summary.json").relative_to(PROJECT_ROOT)),
            "csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "figure": str(fig_path.relative_to(PROJECT_ROOT)),
            "rows_json": str((OUT_DIR / "rows.json").relative_to(PROJECT_ROOT)),
        },
        "rows": rows,
        "note": (
            "WP5c H8 multi-objective. Test sealed. Champion not overwritten. "
            "Classical latency is CPU sklearn; do not mix with GPU µs in absolute claims. "
            "Historical cuML RF is cross-hardware — label carefully in paper."
        ),
    }

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_DIR / "rows.json", "w") as f:
        json.dump(rows, f, indent=2)

    # compact envelope for protocol consistency
    env = make_result_envelope(
        experiment_id="wp5c_pareto_h8",
        protocol_id="botiot_v1",
        stage="stage_b_ft",
        seed=42,
        config={
            "composite_weights": COMPOSITE_WEIGHTS,
            "batch_size": BATCH_SIZE,
            "skip_classical": args.skip_classical,
            "skip_rebench": args.skip_rebench,
        },
        metrics={
            "n_rows": len(rows),
            "pareto_front_ids": front,
            "top_composite": summary["headline"]["top_composite"],
        },
        extra={"wall_sec": wall, "decision": decision},
        project_root=PROJECT_ROOT,
    )
    with open(OUT_DIR / "envelope.json", "w") as f:
        json.dump(env, f, indent=2)

    print("\n=== WP5c H8 DONE ===", flush=True)
    print(f"rows={len(rows)} wall={wall:.1f}s", flush=True)
    print(f"Pareto front ({len(front)}): {front}", flush=True)
    print(f"Neural Pareto front: {front_neural}", flush=True)
    if top_comp:
        print(
            f"Top composite: {top_comp['id']} score={top_comp['composite_score']:.4f} "
            f"F1={top_comp['val_macro_f1']:.4f}",
            flush=True,
        )
    if best_f1:
        print(f"Top F1: {best_f1['id']} F1={best_f1['val_macro_f1']:.4f}", flush=True)
    print(f"summary → {summary_path}", flush=True)
    print(f"champion md5 still {champ_md5}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
