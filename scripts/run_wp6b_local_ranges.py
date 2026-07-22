#!/usr/bin/env python3
"""
WP6b — Local multi-session latency / energy ranges (Option A).

After B14 freeze: re-measure laptop systems numbers as **ranges** across
independent sessions (not single lucky points). Covers:

  - Full V3 PyTorch champion absolute latency (batch 1 / 128 / 256)
  - Energy mJ/flow @ batch 128 (power-sampled)
  - I7 warm-up protocol (discarded sync forwards)
  - I8 batch-size sensitivity table
  - H3 peak allocated VRAM across batches
  - Option A CUDA block multi-session ranges (block3 FP16 + pipeline sum)
    — never claims full custom pipeline vs full V3 parity

Does NOT unseal BoT test. Does NOT clobber champion. Does NOT invent DICC
multi-day numbers.

Outputs:
  benchmarks/results/wp6b_local_ranges/summary.json
  benchmarks/results/wp6b_local_ranges/table.md
  benchmarks/results/wp6b_local_ranges/sessions/session_{k}.json
  benchmarks/results/systems_i8_h3/summary.json  (I8/H3 mirror from session 0)

Usage:
  PYTHONPATH=. .venv/bin/python3 scripts/run_wp6b_local_ranges.py
  PYTHONPATH=. .venv/bin/python3 scripts/run_wp6b_local_ranges.py --n-sessions 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.botiot import load_botiot, load_config  # noqa: E402
from scripts.protocol.result_schema import git_sha  # noqa: E402

EXPECTED_CHAMPION_MD5 = "80a90f7cc210276300eaa90173a5a385"
CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "wp6b_local_ranges"
I8_DIR = PROJECT_ROOT / "benchmarks" / "results" / "systems_i8_h3"
KERNELS_DIR = PROJECT_ROOT / "inference" / "kernels"

BATCH_HEADLINE = [1, 128, 256]
BATCH_SENSITIVITY = [1, 8, 32, 64, 128, 256, 512, 1024]
DEFAULT_SESSIONS = 5
DEFAULT_WARMUP = 50
DEFAULT_ITERS = 100
DEFAULT_ENERGY_ITERS = 400
SESSION_COOLDOWN_S = 3.0
CUDA_TRIALS_PER_SESSION = 20


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_power_w() -> float:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return float(r.stdout.strip().split("\n")[0])
    except Exception:
        return float("nan")


def sample_temp_c() -> float | None:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return float(r.stdout.strip().split("\n")[0])
    except Exception:
        return None


def power_monitor(readings: list[tuple[float, float]], stop: threading.Event, interval: float = 0.05) -> None:
    while not stop.is_set():
        readings.append((time.time(), sample_power_w()))
        time.sleep(interval)


def t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.96


def aggregate(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "cv_pct": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
            "range_low": None,
            "range_high": None,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    se = std / math.sqrt(arr.size) if arr.size > 0 else float("nan")
    tcrit = t_crit_95(arr.size - 1) if arr.size > 1 else float("nan")
    return {
        "n": int(arr.size),
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        "cv_pct": float(100.0 * std / mean) if mean != 0 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": float(mean - tcrit * se) if arr.size > 1 else mean,
        "ci95_high": float(mean + tcrit * se) if arr.size > 1 else mean,
        "range_low": float(arr.min()),
        "range_high": float(arr.max()),
    }


def _batch_tensor(X: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    need = min(len(X), max(batch_size * 4, batch_size))
    xb = np.asarray(X[:need], dtype=np.float32)
    if xb.shape[0] < batch_size:
        reps = int(math.ceil(batch_size / xb.shape[0]))
        xb = np.tile(xb, (reps, 1))[:batch_size]
    return torch.from_numpy(xb[:batch_size]).to(device)


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    xb = _batch_tensor(X, batch_size, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    for _ in range(warmup):
        _ = model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms_sorted = sorted(times_ms)
    median_ms = times_ms_sorted[len(times_ms_sorted) // 2]
    mean_ms = float(np.mean(times_ms))
    p95_ms = times_ms_sorted[int(0.95 * (len(times_ms_sorted) - 1))]
    per_sample_us_median = (median_ms * 1000.0) / batch_size
    per_sample_us_mean = (mean_ms * 1000.0) / batch_size
    thr = batch_size / (median_ms / 1000.0)

    peak_alloc_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "median_ms": float(median_ms),
        "mean_ms": float(mean_ms),
        "p95_ms": float(p95_ms),
        "per_sample_us_median": float(per_sample_us_median),
        "per_sample_us_mean": float(per_sample_us_mean),
        "throughput_samples_per_s": float(thr),
        "peak_alloc_mb": float(peak_alloc_mb) if peak_alloc_mb is not None else None,
        "peak_reserved_mb": float(peak_reserved_mb) if peak_reserved_mb is not None else None,
    }


@torch.no_grad()
def measure_energy_batch128(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    batch_size = 128
    xb = _batch_tensor(X, batch_size, device)

    for _ in range(warmup):
        _ = model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()

    readings: list[tuple[float, float]] = []
    stop = threading.Event()
    mon = threading.Thread(target=power_monitor, args=(readings, stop, 0.05), daemon=True)
    mon.start()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    stop.set()
    mon.join(timeout=2.0)

    powers = [p for _, p in readings if p == p]  # drop nan
    mean_power = float(np.mean(powers)) if powers else float("nan")
    batch_latency_s = elapsed / iters
    thr_flows = (iters * batch_size) / elapsed
    mj_per_flow = mean_power * batch_latency_s / batch_size * 1000.0  # mJ

    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "elapsed_s": float(elapsed),
        "mean_power_w": mean_power,
        "n_power_samples": len(powers),
        "throughput_flows_s": float(thr_flows),
        "mj_per_flow": float(mj_per_flow),
        "batch_latency_s": float(batch_latency_s),
    }


def run_cuda_binary_trials(
    binary: Path,
    pattern: str,
    n_trials: int,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run a kernel binary n_trials times; extract one latency figure per run."""
    if not binary.is_file():
        return {"error": f"missing binary {binary}", "samples_us": []}

    samples: list[float] = []
    failures = 0
    rx = re.compile(pattern)
    for i in range(n_trials):
        try:
            r = subprocess.run(
                [str(binary)],
                cwd=str(binary.parent),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if r.returncode != 0:
                failures += 1
                continue
            m = rx.search(r.stdout)
            if not m:
                failures += 1
                continue
            samples.append(float(m.group(1)))
        except Exception:
            failures += 1

    agg = aggregate(samples)
    return {
        "binary": str(binary.relative_to(PROJECT_ROOT)) if binary.is_relative_to(PROJECT_ROOT) else str(binary),
        "n_trials_requested": n_trials,
        "n_ok": len(samples),
        "failures": failures,
        "samples_us": samples,
        "stats": agg,
    }


def measure_cuda_session(n_trials: int) -> dict[str, Any]:
    """Option A: per-block CUDA absolute latencies (not full V3 parity claim)."""
    targets = {
        "fused_block3_fp16": (
            KERNELS_DIR / "fused_block3_fp16",
            r"Block3 FP16 half2:\s*([\d.]+)",
        ),
        "fused_pipeline_b124": (
            KERNELS_DIR / "fused_pipeline",
            r"Blocks 1\+2\+4 chained.*?:\s*([\d.]+)",
        ),
    }
    out: dict[str, Any] = {}
    for name, (binary, pat) in targets.items():
        out[name] = run_cuda_binary_trials(binary, pat, n_trials)

    # Derived pipeline total = block3_fp16 + b124 chained (same composition as historical README)
    b3 = out["fused_block3_fp16"]["stats"].get("mean")
    b124 = out["fused_pipeline_b124"]["stats"].get("mean")
    derived = None
    if b3 is not None and b124 is not None:
        derived = float(b3) + float(b124)
    out["derived_pipeline_total_us_session_mean"] = derived
    return out


def fmt_range(agg: dict[str, Any], unit: str = "", decimals: int = 2) -> str:
    lo, hi = agg.get("range_low"), agg.get("range_high")
    if lo is None or hi is None:
        return "n/a"
    if unit:
        return f"{lo:.{decimals}f}–{hi:.{decimals}f} {unit}"
    return f"{lo:.{decimals}f}–{hi:.{decimals}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="WP6b local multi-session latency/energy ranges")
    ap.add_argument("--n-sessions", type=int, default=DEFAULT_SESSIONS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--energy-iters", type=int, default=DEFAULT_ENERGY_ITERS)
    ap.add_argument("--cuda-trials", type=int, default=CUDA_TRIALS_PER_SESSION)
    ap.add_argument("--skip-cuda", action="store_true", help="Skip CUDA binary multi-session")
    ap.add_argument("--cooldown", type=float, default=SESSION_COOLDOWN_S)
    args = ap.parse_args()

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

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for WP6b local ranges", file=sys.stderr)
        return 3

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    bundle = load_botiot(stage="stage_b_ft", seed=42)
    X_val = np.asarray(bundle.X_val, dtype=np.float32)

    sys.path.insert(0, str(PROJECT_ROOT / "model"))
    from cnn_bilstm_v3_attention import CNNBiLSTMAttention

    model = CNNBiLSTMAttention(config).to(device)
    state = torch.load(CHAMPION, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_params = int(sum(p.numel() for p in model.parameters()))
    ckpt_bytes = CHAMPION.stat().st_size

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sess_dir = OUT_DIR / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    sessions: list[dict[str, Any]] = []

    print("=" * 72)
    print("WP6b LOCAL MULTI-SESSION RANGES (Option A)")
    print(f"champion md5={champ_md5}")
    print(f"gpu={gpu_name}  sessions={args.n_sessions}  warmup={args.warmup}  iters={args.iters}")
    print("=" * 72)

    for s in range(args.n_sessions):
        temp0 = sample_temp_c()
        print(f"\n--- session {s + 1}/{args.n_sessions}  temp={temp0}°C ---")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Headline latencies
        lat: dict[str, Any] = {}
        for bs in BATCH_HEADLINE:
            row = measure_latency(model, X_val, bs, device, args.warmup, args.iters)
            lat[f"batch{bs}"] = row
            print(
                f"  PT bs={bs:4d}  med={row['median_ms']:.3f} ms  "
                f"µs/sample={row['per_sample_us_median']:.2f}  thr={row['throughput_samples_per_s']:.0f}/s"
            )

        # Full batch sensitivity every session (feeds I8 multi-session ranges)
        sens_rows = []
        for bs in BATCH_SENSITIVITY:
            try:
                row = measure_latency(model, X_val, bs, device, args.warmup, max(30, args.iters // 2))
                sens_rows.append(row)
            except RuntimeError as e:
                sens_rows.append({"batch_size": bs, "error": str(e)})
                print(f"  sens bs={bs}: ERROR {e}")
                torch.cuda.empty_cache()

        # Energy @ batch 128
        energy = measure_energy_batch128(model, X_val, device, args.warmup, args.energy_iters)
        print(
            f"  energy bs128  power={energy['mean_power_w']:.2f} W  "
            f"mJ/flow={energy['mj_per_flow']:.4f}  thr={energy['throughput_flows_s']:.0f}/s"
        )

        # CUDA Option A block session
        cuda_sess = None
        if not args.skip_cuda:
            cuda_sess = measure_cuda_session(args.cuda_trials)
            d = cuda_sess.get("derived_pipeline_total_us_session_mean")
            b3m = cuda_sess["fused_block3_fp16"]["stats"].get("mean")
            print(f"  CUDA block3_fp16 mean={b3m}  derived_pipeline_total={d}")

        temp1 = sample_temp_c()
        sess = {
            "session": s,
            "temp_c_start": temp0,
            "temp_c_end": temp1,
            "pytorch_latency": lat,
            "batch_sensitivity": sens_rows,
            "energy_batch128": energy,
            "cuda_option_a": cuda_sess,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        sessions.append(sess)
        (sess_dir / f"session_{s}.json").write_text(json.dumps(sess, indent=2))

        if s + 1 < args.n_sessions:
            time.sleep(args.cooldown)

    finished = datetime.now(timezone.utc)

    # ---- Aggregate across sessions ----
    def collect(path_fn) -> list[float]:
        vals = []
        for sess in sessions:
            v = path_fn(sess)
            if v is not None:
                vals.append(float(v))
        return vals

    agg_pt: dict[str, Any] = {}
    for bs in BATCH_HEADLINE:
        key = f"batch{bs}"
        agg_pt[key] = {
            "per_sample_us_median": aggregate(
                collect(lambda s, k=key: s["pytorch_latency"][k]["per_sample_us_median"])
            ),
            "per_sample_us_mean": aggregate(
                collect(lambda s, k=key: s["pytorch_latency"][k]["per_sample_us_mean"])
            ),
            "median_ms": aggregate(collect(lambda s, k=key: s["pytorch_latency"][k]["median_ms"])),
            "throughput_samples_per_s": aggregate(
                collect(lambda s, k=key: s["pytorch_latency"][k]["throughput_samples_per_s"])
            ),
            "peak_alloc_mb": aggregate(
                collect(lambda s, k=key: s["pytorch_latency"][k].get("peak_alloc_mb"))
            ),
        }

    agg_energy = {
        "mj_per_flow": aggregate(collect(lambda s: s["energy_batch128"]["mj_per_flow"])),
        "mean_power_w": aggregate(collect(lambda s: s["energy_batch128"]["mean_power_w"])),
        "throughput_flows_s": aggregate(collect(lambda s: s["energy_batch128"]["throughput_flows_s"])),
    }

    # I8: per batch_size multi-session aggregate of per_sample_us_median
    i8_by_bs: dict[str, Any] = {}
    peak_vram_all: list[float] = []
    for bs in BATCH_SENSITIVITY:
        vals = []
        thr_vals = []
        peak_vals = []
        for sess in sessions:
            for row in sess["batch_sensitivity"]:
                if row.get("batch_size") == bs and "error" not in row:
                    vals.append(row["per_sample_us_median"])
                    thr_vals.append(row["throughput_samples_per_s"])
                    if row.get("peak_alloc_mb") is not None:
                        peak_vals.append(row["peak_alloc_mb"])
                        peak_vram_all.append(row["peak_alloc_mb"])
        i8_by_bs[str(bs)] = {
            "per_sample_us_median": aggregate(vals),
            "throughput_samples_per_s": aggregate(thr_vals),
            "peak_alloc_mb": aggregate(peak_vals),
        }

    # CUDA aggregates
    agg_cuda: dict[str, Any] | None = None
    if not args.skip_cuda and sessions[0].get("cuda_option_a"):
        agg_cuda = {
            "fused_block3_fp16_us": aggregate(
                collect(
                    lambda s: (s.get("cuda_option_a") or {})
                    .get("fused_block3_fp16", {})
                    .get("stats", {})
                    .get("mean")
                )
            ),
            "fused_pipeline_b124_us": aggregate(
                collect(
                    lambda s: (s.get("cuda_option_a") or {})
                    .get("fused_pipeline_b124", {})
                    .get("stats", {})
                    .get("mean")
                )
            ),
            "derived_pipeline_total_us": aggregate(
                collect(
                    lambda s: (s.get("cuda_option_a") or {}).get(
                        "derived_pipeline_total_us_session_mean"
                    )
                )
            ),
            "note": (
                "Option A: block/pipeline CUDA absolutes from independent sessions. "
                "derived_pipeline_total = session-mean(block3_fp16) + session-mean(b124_chained). "
                "Do NOT claim full custom pipeline vs full V3 PT speedup as parity."
            ),
        }

    peak_vram_global = max(peak_vram_all) if peak_vram_all else None
    bs256 = agg_pt.get("batch256", {})
    bs128 = agg_pt.get("batch128", {})

    headline = {
        "pt_batch256_us_per_sample_range": {
            "low": bs256.get("per_sample_us_median", {}).get("range_low"),
            "high": bs256.get("per_sample_us_median", {}).get("range_high"),
            "mean": bs256.get("per_sample_us_median", {}).get("mean"),
            "std": bs256.get("per_sample_us_median", {}).get("std"),
            "cv_pct": bs256.get("per_sample_us_median", {}).get("cv_pct"),
            "ci95": [
                bs256.get("per_sample_us_median", {}).get("ci95_low"),
                bs256.get("per_sample_us_median", {}).get("ci95_high"),
            ],
        },
        "pt_batch128_us_per_sample_range": {
            "low": bs128.get("per_sample_us_median", {}).get("range_low"),
            "high": bs128.get("per_sample_us_median", {}).get("range_high"),
            "mean": bs128.get("per_sample_us_median", {}).get("mean"),
        },
        "pt_batch1_us_per_sample_range": {
            "low": agg_pt["batch1"]["per_sample_us_median"].get("range_low"),
            "high": agg_pt["batch1"]["per_sample_us_median"].get("range_high"),
            "mean": agg_pt["batch1"]["per_sample_us_median"].get("mean"),
        },
        "energy_mj_per_flow_range": {
            "low": agg_energy["mj_per_flow"].get("range_low"),
            "high": agg_energy["mj_per_flow"].get("range_high"),
            "mean": agg_energy["mj_per_flow"].get("mean"),
            "std": agg_energy["mj_per_flow"].get("std"),
            "cv_pct": agg_energy["mj_per_flow"].get("cv_pct"),
            "ci95": [
                agg_energy["mj_per_flow"].get("ci95_low"),
                agg_energy["mj_per_flow"].get("ci95_high"),
            ],
        },
        "peak_alloc_mb_global_max": peak_vram_global,
        "cuda_derived_pipeline_total_us_range": (
            {
                "low": agg_cuda["derived_pipeline_total_us"].get("range_low"),
                "high": agg_cuda["derived_pipeline_total_us"].get("range_high"),
                "mean": agg_cuda["derived_pipeline_total_us"].get("mean"),
            }
            if agg_cuda
            else None
        ),
        "cuda_block3_fp16_us_range": (
            {
                "low": agg_cuda["fused_block3_fp16_us"].get("range_low"),
                "high": agg_cuda["fused_block3_fp16_us"].get("range_high"),
                "mean": agg_cuda["fused_block3_fp16_us"].get("mean"),
            }
            if agg_cuda
            else None
        ),
        "n_sessions": args.n_sessions,
        "platform": f"local {gpu_name}",
        "option_a": True,
        "historical_single_shot_energy_mj_flow": 0.7860839754130158,
        "note": (
            "All headlines are multi-session ranges (min–max of session means) with "
            "mean±std / CV / 95% CI. Local RTX only — not DICC multi-day. "
            "Full V3 PT absolute latency is allowed under Option A; CUDA is per-block."
        ),
    }

    summary: dict[str, Any] = {
        "experiment_id": "wp6b_local_multisession_ranges",
        "tracker": ["WP6b", "I6_local", "I7", "I8", "H3", "H4", "F9_ranges", "L8", "K6"],
        "work_package": "WP6b",
        "protocol_id": "botiot_v1",
        "stage": "stage_b_ft",
        "seed": 42,
        "allow_test": False,
        "test_sealed": True,
        "checkpoint": str(CHAMPION.relative_to(PROJECT_ROOT)),
        "champion_md5": champ_md5,
        "champion_md5_expected": EXPECTED_CHAMPION_MD5,
        "champion_unchanged": champ_md5 == EXPECTED_CHAMPION_MD5,
        "device": str(device),
        "gpu_name": gpu_name,
        "n_sessions": args.n_sessions,
        "warmup_forwards": args.warmup,
        "timed_iters": args.iters,
        "energy_iters": args.energy_iters,
        "cuda_trials_per_session": args.cuda_trials if not args.skip_cuda else 0,
        "session_cooldown_s": args.cooldown,
        "n_params": n_params,
        "checkpoint_bytes": ckpt_bytes,
        "checkpoint_mb": ckpt_bytes / (1024**2),
        "param_memory_proxy_mb_fp32": n_params * 4 / (1024**2),
        "headline": headline,
        "aggregate_pytorch": agg_pt,
        "aggregate_energy": agg_energy,
        "aggregate_i8_batch_sensitivity": i8_by_bs,
        "aggregate_cuda_option_a": agg_cuda,
        "sessions": [
            {
                "session": s["session"],
                "temp_c_start": s["temp_c_start"],
                "temp_c_end": s["temp_c_end"],
                "pt_batch1_us": s["pytorch_latency"]["batch1"]["per_sample_us_median"],
                "pt_batch128_us": s["pytorch_latency"]["batch128"]["per_sample_us_median"],
                "pt_batch256_us": s["pytorch_latency"]["batch256"]["per_sample_us_median"],
                "energy_mj_per_flow": s["energy_batch128"]["mj_per_flow"],
                "cuda_derived_pipeline_us": (s.get("cuda_option_a") or {}).get(
                    "derived_pipeline_total_us_session_mean"
                ),
            }
            for s in sessions
        ],
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_sec": (finished - started).total_seconds(),
        "git_sha": git_sha(PROJECT_ROOT),
        "decision": "DONE",
        "decision_note": (
            "WP6b local multi-session latency/energy ranges on frozen CAD-CBA-v1 champion. "
            "I7 warm-up applied; I8 batch sensitivity multi-session; H3 peak VRAM measured. "
            "Option A CUDA block ranges included when binaries present. "
            "DICC multi-day (I1–I6 cluster) remains BLOCKED until dedicated session. "
            "Do not mix these local ranges with legacy single-shot or invented multi-day CIs."
        ),
    }

    # Champion post-check
    champ_after = md5_file(CHAMPION)
    summary["champion_md5_after"] = champ_after
    summary["champion_unchanged"] = champ_after == EXPECTED_CHAMPION_MD5
    if champ_after != EXPECTED_CHAMPION_MD5:
        print("CRITICAL: champion md5 changed during WP6b", file=sys.stderr)
        summary["decision"] = "ERROR_CHAMPION_CHANGED"

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Markdown table
    lines = [
        "# WP6b — Local multi-session latency / energy ranges (Option A)",
        "",
        f"- Checkpoint: `{CHAMPION.name}` md5 `{champ_md5}` (unchanged={summary['champion_unchanged']})",
        f"- Device: `{gpu_name}`",
        f"- Sessions: **{args.n_sessions}** · warm-up **{args.warmup}** · timed iters **{args.iters}**",
        f"- Wall: {summary['wall_sec']:.1f} s",
        "",
        "## Headlines (session-mean ranges)",
        "",
        f"| Metric | Range (min–max session means) | Mean ± std | CV% | 95% CI |",
        f"|--------|-------------------------------|------------|-----|--------|",
    ]

    def row(name: str, agg: dict) -> str:
        if not agg or agg.get("mean") is None:
            return f"| {name} | n/a | n/a | n/a | n/a |"
        return (
            f"| {name} | {agg['range_low']:.4g}–{agg['range_high']:.4g} | "
            f"{agg['mean']:.4g} ± {agg['std']:.4g} | "
            f"{agg['cv_pct']:.2f} | "
            f"[{agg['ci95_low']:.4g}, {agg['ci95_high']:.4g}] |"
        )

    lines.append(row("PT µs/sample @bs=256", agg_pt["batch256"]["per_sample_us_median"]))
    lines.append(row("PT µs/sample @bs=128", agg_pt["batch128"]["per_sample_us_median"]))
    lines.append(row("PT µs/sample @bs=1", agg_pt["batch1"]["per_sample_us_median"]))
    lines.append(row("Energy mJ/flow @bs=128", agg_energy["mj_per_flow"]))
    lines.append(row("Energy throughput flows/s @bs=128", agg_energy["throughput_flows_s"]))
    if agg_cuda:
        lines.append(row("CUDA block3 FP16 µs (session mean)", agg_cuda["fused_block3_fp16_us"]))
        lines.append(row("CUDA derived pipeline total µs", agg_cuda["derived_pipeline_total_us"]))
    lines.append(
        f"| Peak alloc VRAM (global max across sessions/batches) | **{peak_vram_global:.2f} MiB** | — | — | — |"
        if peak_vram_global is not None
        else "| Peak alloc VRAM | n/a | — | — | — |"
    )
    lines += [
        "",
        "## I8 batch-size sensitivity (multi-session µs/sample median)",
        "",
        "| batch | range µs/sample | mean ± std | thrput range /s | peak alloc mean MiB |",
        "|------:|----------------:|------------|----------------:|--------------------:|",
    ]
    for bs in BATCH_SENSITIVITY:
        a = i8_by_bs[str(bs)]["per_sample_us_median"]
        t = i8_by_bs[str(bs)]["throughput_samples_per_s"]
        p = i8_by_bs[str(bs)]["peak_alloc_mb"]
        if a.get("mean") is None:
            lines.append(f"| {bs} | ERROR | — | — | — |")
        else:
            thr_s = (
                f"{t['range_low']:.0f}–{t['range_high']:.0f}"
                if t.get("mean") is not None
                else "—"
            )
            peak_s = f"{p['mean']:.2f}" if p.get("mean") is not None else "—"
            lines.append(
                f"| {bs} | {a['range_low']:.2f}–{a['range_high']:.2f} | "
                f"{a['mean']:.2f} ± {a['std']:.2f} | {thr_s} | {peak_s} |"
            )

    lines += [
        "",
        "## Per-session raw session means",
        "",
        "| session | temp°C | PT@1 µs | PT@128 µs | PT@256 µs | mJ/flow | CUDA pipeline µs |",
        "|--------:|-------:|--------:|----------:|----------:|--------:|-----------------:|",
    ]
    for s in summary["sessions"]:
        lines.append(
            f"| {s['session']} | {s['temp_c_start']} | "
            f"{s['pt_batch1_us']:.2f} | {s['pt_batch128_us']:.2f} | {s['pt_batch256_us']:.2f} | "
            f"{s['energy_mj_per_flow']:.4f} | "
            f"{s['cuda_derived_pipeline_us'] if s['cuda_derived_pipeline_us'] is not None else '—'} |"
        )

    lines += [
        "",
        "## Claim rules",
        "",
        "- Report **ranges** (and mean±std / CI), not single points, for laptop systems headlines.",
        "- Option A: CUDA figures are **per-block / derived pipeline sum**, not full V3 parity.",
        "- Local RTX only; DICC multi-day still **BLOCKED** until dedicated session.",
        f"- Historical single-shot energy 0.786 mJ/flow remains labeled historical; new range supersedes for WP6b.",
        "",
        f"**Decision:** `{summary['decision']}` — {summary['decision_note']}",
        "",
    ]
    table_path = OUT_DIR / "table.md"
    table_path.write_text("\n".join(lines) + "\n")

    # Mirror I8/H3 summary for tracker compatibility (from multi-session aggregates)
    I8_DIR.mkdir(parents=True, exist_ok=True)
    i8_rows = []
    for bs in BATCH_SENSITIVITY:
        a = i8_by_bs[str(bs)]
        i8_rows.append(
            {
                "batch_size": bs,
                "n_sessions": args.n_sessions,
                "per_sample_us_median": a["per_sample_us_median"],
                "throughput_samples_per_s": a["throughput_samples_per_s"],
                "peak_alloc_mb": a["peak_alloc_mb"],
            }
        )
    i8_summary = {
        "experiment_id": "i8_batch_sensitivity_h3_peak_vram_multisession",
        "tracker": ["I8", "H3", "I7", "WP6b"],
        "work_package": "WP6b / systems_i8_h3",
        "source": "wp6b_local_ranges (multi-session)",
        "protocol_id": "botiot_v1",
        "checkpoint": str(CHAMPION.relative_to(PROJECT_ROOT)),
        "champion_md5": champ_md5,
        "champion_unchanged": True,
        "device": str(device),
        "gpu_name": gpu_name,
        "warmup_forwards": args.warmup,
        "n_sessions": args.n_sessions,
        "n_params": n_params,
        "checkpoint_bytes": ckpt_bytes,
        "rows": i8_rows,
        "headline": {
            "peak_alloc_mb_across_batches": peak_vram_global,
            "batch256_per_sample_us_mean": bs256.get("per_sample_us_median", {}).get("mean"),
            "batch256_per_sample_us_range": [
                bs256.get("per_sample_us_median", {}).get("range_low"),
                bs256.get("per_sample_us_median", {}).get("range_high"),
            ],
            "batch256_peak_alloc_mb_mean": bs256.get("peak_alloc_mb", {}).get("mean"),
        },
        "timestamp_utc": finished.isoformat(),
        "git_sha": git_sha(PROJECT_ROOT),
        "decision": "DONE",
        "decision_note": (
            "I8 batch-size sensitivity + H3 peak VRAM measured as multi-session ranges under WP6b. "
            "Warm-up I7 applied. DICC multi-day still BLOCKED."
        ),
    }
    (I8_DIR / "summary.json").write_text(json.dumps(i8_summary, indent=2))
    # simple I8 table
    i8_lines = [
        "# I8 batch-size sensitivity + H3 peak VRAM (from WP6b multi-session)",
        "",
        f"- Sessions: {args.n_sessions} · warm-up: {args.warmup}",
        f"- Peak alloc global max: **{peak_vram_global:.2f} MiB**" if peak_vram_global else "",
        "",
        "| batch | µs/sample range | mean ± std | thrput range /s | peak MiB mean |",
        "|------:|----------------:|------------|----------------:|--------------:|",
    ]
    for bs in BATCH_SENSITIVITY:
        a = i8_by_bs[str(bs)]["per_sample_us_median"]
        t = i8_by_bs[str(bs)]["throughput_samples_per_s"]
        p = i8_by_bs[str(bs)]["peak_alloc_mb"]
        if a.get("mean") is None:
            i8_lines.append(f"| {bs} | ERROR | — | — | — |")
        else:
            i8_lines.append(
                f"| {bs} | {a['range_low']:.2f}–{a['range_high']:.2f} | "
                f"{a['mean']:.2f} ± {a['std']:.2f} | "
                f"{t['range_low']:.0f}–{t['range_high']:.0f} | "
                f"{p['mean']:.2f} |"
            )
    (I8_DIR / "table.md").write_text("\n".join(i8_lines) + "\n")

    print("\n" + "=" * 72)
    print("WP6b HEADLINES")
    print(json.dumps(headline, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {table_path}")
    print(f"wrote {I8_DIR / 'summary.json'}")
    print(f"champion unchanged: {summary['champion_unchanged']}")
    return 0 if summary["champion_unchanged"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
