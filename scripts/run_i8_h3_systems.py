#!/usr/bin/env python3
"""
I8 batch-size sensitivity + H3 peak VRAM (local RTX, val-only systems).

Does NOT unseal BoT test. Uses production champion (frozen md5).
Warm-up protocol: 20 sync forwards discarded, then timed runs.

Outputs:
  benchmarks/results/systems_i8_h3/summary.json
  benchmarks/results/systems_i8_h3/table.md
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402

EXPECTED_CHAMPION_MD5 = "80a90f7cc210276300eaa90173a5a385"
CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512, 1024]
WARMUP = 20
ITERS = 50


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


@torch.no_grad()
def measure_batch(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict:
    n = X.shape[0]
    # Fixed synthetic-like slice from real val features (no labels needed for latency)
    # Use first min(n, max batch * 4) rows for stable batches
    need = min(n, max(batch_size * 4, batch_size))
    xb_np = np.asarray(X[:need], dtype=np.float32)
    # pad if needed
    if xb_np.shape[0] < batch_size:
        reps = int(np.ceil(batch_size / xb_np.shape[0]))
        xb_np = np.tile(xb_np, (reps, 1))[:batch_size]

    xb = torch.from_numpy(xb_np[:batch_size]).to(device)

    # Peak VRAM for this batch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(device)

    # Warm-up
    for _ in range(WARMUP):
        _ = model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(ITERS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    p95_ms = times_ms[int(0.95 * (len(times_ms) - 1))]
    mean_ms = float(np.mean(times_ms))
    per_sample_us = (median_ms * 1000.0) / batch_size

    peak_alloc_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    return {
        "batch_size": batch_size,
        "warmup": WARMUP,
        "iters": ITERS,
        "median_ms": float(median_ms),
        "mean_ms": float(mean_ms),
        "p95_ms": float(p95_ms),
        "per_sample_us_median": float(per_sample_us),
        "throughput_samples_per_s": float(batch_size / (median_ms / 1000.0)),
        "peak_alloc_mb": float(peak_alloc_mb) if peak_alloc_mb is not None else None,
        "peak_reserved_mb": float(peak_reserved_mb) if peak_reserved_mb is not None else None,
    }


def main() -> int:
    if not CHAMPION.is_file():
        print(f"ERROR: missing champion {CHAMPION}", file=sys.stderr)
        return 1
    champ_md5 = md5_file(CHAMPION)
    if champ_md5 != EXPECTED_CHAMPION_MD5:
        print(
            f"ERROR: champion md5 {champ_md5} != {EXPECTED_CHAMPION_MD5}",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARN: CUDA not available; I8/H3 will be CPU-only (disclose).", file=sys.stderr)

    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    bundle = load_botiot(stage="stage_b_ft", seed=42)

    sys.path.insert(0, str(PROJECT_ROOT / "model"))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    model = CNNBiLSTMAttention(config).to(device)
    state = torch.load(CHAMPION, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # n_params + ckpt bytes (memory proxy)
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_bytes = CHAMPION.stat().st_size

    t0 = datetime.now(timezone.utc)
    rows = []
    for bs in BATCH_SIZES:
        try:
            row = measure_batch(model, bundle.X_val, bs, device)
            rows.append(row)
            print(
                f"bs={bs:4d}  median={row['median_ms']:.3f} ms  "
                f"per_sample={row['per_sample_us_median']:.2f} µs  "
                f"peak_alloc={row['peak_alloc_mb']}"
            )
        except RuntimeError as e:
            rows.append(
                {
                    "batch_size": bs,
                    "error": str(e),
                    "note": "OOM or runtime failure",
                }
            )
            print(f"bs={bs:4d}  ERROR: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    t1 = datetime.now(timezone.utc)

    ok_rows = [r for r in rows if "error" not in r]
    peak_vram = max((r["peak_alloc_mb"] for r in ok_rows if r.get("peak_alloc_mb") is not None), default=None)
    best_thr = max(ok_rows, key=lambda r: r["throughput_samples_per_s"]) if ok_rows else None
    bs256 = next((r for r in ok_rows if r["batch_size"] == 256), None)

    # cuML historical contrast (do not invent; load if present)
    cuml_vram = None
    cuml_path = PROJECT_ROOT / "benchmarks" / "results" / "cuml_rf_resources.json"
    if not cuml_path.is_file():
        # try alternate locations
        for cand in PROJECT_ROOT.glob("**/cuml_rf_resources.json"):
            cuml_path = cand
            break
    if cuml_path.is_file():
        try:
            cuml = json.loads(cuml_path.read_text())
            # flexible keys
            for k in ("rf_vram_mb", "cuml_vram_mb", "vram_mb", "peak_vram_mb"):
                if k in cuml:
                    cuml_vram = cuml[k]
                    break
            if cuml_vram is None and isinstance(cuml.get("rf"), dict):
                cuml_vram = cuml["rf"].get("vram_mb") or cuml["rf"].get("memory_mb")
        except Exception:
            pass

    summary = {
        "experiment_id": "i8_batch_sensitivity_h3_peak_vram",
        "tracker": ["I8", "H3", "I7"],
        "work_package": "systems local (pre-B14 ok)",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": 42,
        "allow_test": False,
        "test_sealed": True,
        "checkpoint": str(CHAMPION.relative_to(PROJECT_ROOT)),
        "champion_md5": champ_md5,
        "champion_md5_expected": EXPECTED_CHAMPION_MD5,
        "champion_unchanged": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "warmup_forwards": WARMUP,
        "timed_iters": ITERS,
        "batch_sizes": BATCH_SIZES,
        "n_params": int(n_params),
        "checkpoint_bytes": int(ckpt_bytes),
        "checkpoint_mb": ckpt_bytes / (1024**2),
        "param_memory_proxy_mb_fp32": n_params * 4 / (1024**2),
        "rows": rows,
        "headline": {
            "peak_alloc_mb_across_batches": peak_vram,
            "batch256_per_sample_us": bs256["per_sample_us_median"] if bs256 else None,
            "batch256_peak_alloc_mb": bs256["peak_alloc_mb"] if bs256 else None,
            "best_throughput_batch": best_thr["batch_size"] if best_thr else None,
            "best_throughput_samples_per_s": best_thr["throughput_samples_per_s"] if best_thr else None,
            "historical_cuml_rf_vram_mb": cuml_vram,
        },
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "wall_sec": (t1 - t0).total_seconds(),
        "git_sha": git_sha(),
        "decision": "RUN_DOCUMENTED",
        "decision_note": (
            "I8 batch-size sensitivity + H3 peak alloc VRAM on frozen champion, val features only. "
            "Test sealed. Warm-up I7: 20 discarded sync forwards. "
            "Does not replace DICC multi-day / multi-GPU (I1–I6 still BLOCKED)."
        ),
    }

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "systems_i8_h3"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# I8 batch-size sensitivity + H3 peak VRAM",
        "",
        f"- Checkpoint: `{CHAMPION.name}` md5 `{champ_md5}`",
        f"- Device: {device} ({summary['gpu_name']})",
        f"- Warm-up: {WARMUP} sync forwards; timed iters: {ITERS}",
        f"- Peak alloc across batches: **{peak_vram:.2f} MiB**" if peak_vram is not None else "- Peak alloc: n/a",
        f"- Batch-256 per-sample: **{bs256['per_sample_us_median']:.2f} µs**" if bs256 else "",
        f"- n_params: {n_params} · ckpt: {ckpt_bytes/1024**2:.2f} MiB",
        "",
        "| batch | median ms | µs/sample | thrput /s | peak alloc MiB | peak reserved MiB |",
        "|------:|----------:|----------:|----------:|---------------:|------------------:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['batch_size']} | ERROR | — | — | — | — |")
        else:
            lines.append(
                f"| {r['batch_size']} | {r['median_ms']:.3f} | {r['per_sample_us_median']:.2f} | "
                f"{r['throughput_samples_per_s']:.0f} | {r['peak_alloc_mb']:.2f} | {r['peak_reserved_mb']:.2f} |"
            )
    table_path = out_dir / "table.md"
    table_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(summary["headline"], indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {table_path}")
    # post champion check
    if md5_file(CHAMPION) != EXPECTED_CHAMPION_MD5:
        print("CRITICAL: champion md5 changed", file=sys.stderr)
        return 30
    return 0 if ok_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
