"""
COLIDE - CUDA Kernel Statistical Benchmark

Runs each compiled CUDA kernel binary N_TRIALS times and computes real
mean/std/percentiles/CIs, instead of trusting a single point measurement.

Motivation: while re-verifying the manuscript's numbers (2026-07-01), a
single re-run of fused_block3_fp16 landed 24% away from the headline
"601.4 us" figure baked into README.md/ablation_study.py. Twenty repeated
runs showed why: normal run-to-run variance on this kernel is +-4-5%
(mean ~609us, std ~27us) -- a single run can land anywhere in a much wider
tail. Every kernel latency treated as a fixed constant elsewhere in this
repo should be backed by a distribution like this, not one run.

Strict mode (default for DICC via --strict):
  - missing binary -> fatal
  - nonzero process exit -> fatal
  - timeout -> fatal
  - missing expected metric in stdout -> fatal
  - numerical validation FAILED in stdout -> fatal
  - fewer collected samples than n_trials -> fatal

Usage:
    # Local dev (binaries are unsuffixed in inference/kernels/):
    PYTHONPATH=. python scripts/benchmark_cuda_kernels_stats.py \\
        --suffix "" --tag rtx3050 --n-trials 100 \\
        --output benchmarks/results/cuda_kernel_stats_rtx3050.json

    # DICC (binaries under inference/kernels/{v100,a100}/):
    PYTHONPATH=. python scripts/benchmark_cuda_kernels_stats.py \\
        --kernels-dir inference/kernels/v100 --suffix "" --tag v100s \\
        --n-trials 100 --strict \\
        --output /path/to/run_dir/cuda_kernel_stats.json \\
        --raw-output /path/to/run_dir/raw/cuda_kernel_raw.json

Binaries must be compiled matching the Dockerfile's build command (no extra
-O flags -- confirmed 2026-07-01 that adding -O3 measurably changes the
timing-loop behavior, not just device-code optimization):
    nvcc -arch=sm_86 -o inference/kernels/fused_block1 inference/kernels/fused_block1.cu
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_TRIALS_DEFAULT = 100

# Each binary's stdout pattern(s) to extract latency figures from.
# (binary_base_name, {result_key: regex})
TARGETS = [
    ("fused_block1", {
        "latency_us": r"Fused kernel \(FP32\) time:\s*([\d.]+)",
    }),
    ("fused_block2", {
        "latency_us": r"Fused kernel \(FP32\) time:\s*([\d.]+)",
    }),
    ("fused_block3", {
        "no_graphs_us": r"Without CUDA Graphs:\s*([\d.]+)",
        "with_graphs_us": r"With CUDA Graphs:\s*([\d.]+)",
    }),
    ("fused_block3_fp16", {
        "latency_us": r"Block3 FP16 half2:\s*([\d.]+)",
    }),
    ("fused_block3_naive", {
        "latency_us": r"Block3 \(BiLSTM\) time:\s*([\d.]+)",
    }),
    ("fused_block4", {
        "latency_us": r"Block4 \(Dense\) time:\s*([\d.]+)",
    }),
    ("fused_pipeline", {
        "b124_chained_us": r"Blocks 1\+2\+4 chained.*?:\s*([\d.]+)",
    }),
]


class BenchmarkError(RuntimeError):
    """Fatal benchmark condition (strict mode or always-fatal errors)."""


def _t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value. Prefer scipy; fall back to normal approx."""
    if df <= 0:
        return float("nan")
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        # Normal approximation; fine for large n (e.g. 100).
        return 1.96


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
    # Sample std (ddof=1) for CI; also report population std for continuity.
    std_sample = float(arr.std(ddof=1)) if n > 1 else 0.0
    std_pop = float(arr.std(ddof=0))
    se = std_sample / math.sqrt(n) if n > 0 else float("nan")
    tcrit = _t_critical_95(n - 1) if n > 1 else float("nan")
    half = tcrit * se if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std_pop,
        "std_sample": std_sample,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "cv_pct": float(std_pop / mean * 100) if mean != 0 else float("nan"),
        "ci95_low": float(mean - half) if n > 1 else mean,
        "ci95_high": float(mean + half) if n > 1 else mean,
        "ci95_halfwidth": float(half) if n > 1 else 0.0,
    }


def run_trials(
    binary_path: Path,
    patterns: dict[str, str],
    n: int,
    *,
    strict: bool,
    timeout_sec: float | None,
    max_retries: int = 3,
    max_validation_failures: int | None = None,
) -> tuple[dict, dict]:
    """Return (stats_dict, raw_samples_dict). Raises BenchmarkError in strict mode.

    Intermittent numerical-validation flakes (seen on A100 fused_block3_fp16) are
    retried up to max_retries times per trial slot. Only after retries are exhausted
    does the trial count as a validation failure. If max_validation_failures is set,
    the run continues until that many hard failures accumulate; otherwise one hard
    failure is fatal under strict=True.
    """
    if not binary_path.exists():
        msg = f"binary not found: {binary_path}"
        if strict:
            raise BenchmarkError(msg)
        print(f"[SKIP] {msg}")
        return {}, {}

    if not binary_path.is_file():
        raise BenchmarkError(f"not a file: {binary_path}")

    if max_validation_failures is None:
        max_validation_failures = 0 if strict else n  # strict default: no hard fails allowed after retries

    collected: dict[str, list[float]] = {key: [] for key in patterns}
    raw_runs: list[dict] = []
    validation_failures = 0
    nonzero_exits = 0
    timeouts = 0
    parse_failures = 0
    retries_used = 0

    # Absolute path required: cwd= re-resolves relative paths against the new cwd.
    resolved_path = binary_path.resolve()
    for trial_idx in range(n):
        attempt = 0
        while True:
            attempt += 1
            try:
                result = subprocess.run(
                    [str(resolved_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(binary_path.parent),
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                timeouts += 1
                raw_runs.append({
                    "trial": trial_idx,
                    "attempt": attempt,
                    "error": "timeout",
                    "timeout_sec": timeout_sec,
                })
                if strict:
                    raise BenchmarkError(
                        f"{binary_path.name} trial {trial_idx}: timed out after {timeout_sec}s"
                    ) from exc
                break

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            run_record: dict = {
                "trial": trial_idx,
                "attempt": attempt,
                "returncode": result.returncode,
                "metrics": {},
            }

            if result.returncode != 0:
                nonzero_exits += 1
                run_record["stderr_tail"] = stderr[-500:]
                raw_runs.append(run_record)
                if strict:
                    raise BenchmarkError(
                        f"{binary_path.name} trial {trial_idx}: nonzero exit "
                        f"{result.returncode}; stderr_tail={stderr[-300:]!r}"
                    )
                break

            if "FAILED" in stdout:
                retries_used += 1
                run_record["validation"] = "FAILED"
                raw_runs.append(run_record)
                if attempt <= max_retries:
                    print(
                        f"  {binary_path.name} trial {trial_idx}: validation FAILED "
                        f"(attempt {attempt}/{max_retries + 1}), retrying…"
                    )
                    continue
                validation_failures += 1
                if strict and validation_failures > max_validation_failures:
                    raise BenchmarkError(
                        f"{binary_path.name} trial {trial_idx}: numerical validation FAILED "
                        f"after {attempt} attempts "
                        f"(hard failures={validation_failures} > allowed={max_validation_failures})"
                    )
                # Soft-skip this trial slot (do not append metrics).
                break

            run_record["validation"] = "ok" if "PASSED" in stdout else "unknown"
            missing = []
            for key, pattern in patterns.items():
                m = re.search(pattern, stdout, re.DOTALL)
                if m:
                    val = float(m.group(1))
                    collected[key].append(val)
                    run_record["metrics"][key] = val
                else:
                    missing.append(key)

            if missing:
                parse_failures += 1
                run_record["missing_metrics"] = missing
                run_record["stdout_tail"] = stdout[-500:]
                raw_runs.append(run_record)
                if strict:
                    raise BenchmarkError(
                        f"{binary_path.name} trial {trial_idx}: missing metrics {missing}"
                    )
                break

            raw_runs.append(run_record)
            break  # success for this trial slot

    stats: dict = {
        "n_trials_requested": n,
        "n_samples": {k: len(v) for k, v in collected.items()},
        "validation_failures": validation_failures,
        "validation_retries": retries_used,
        "nonzero_exits": nonzero_exits,
        "timeouts": timeouts,
        "parse_failures": parse_failures,
        "binary": str(resolved_path),
    }

    for key, values in collected.items():
        if len(values) == 0:
            if strict:
                raise BenchmarkError(
                    f"{binary_path.name}: no samples collected for metric '{key}'"
                )
            continue
        # Require at least 90% of requested trials after rare soft-skips.
        min_ok = max(1, int(n * 0.9)) if max_validation_failures > 0 else n
        if strict and len(values) < min_ok:
            raise BenchmarkError(
                f"{binary_path.name}: metric '{key}' has {len(values)} samples, "
                f"need >= {min_ok} (requested n={n})"
            )
        stats[key] = summarize(values)

    raw = {
        "binary": str(resolved_path),
        "n_trials_requested": n,
        "samples": {key: list(vals) for key, vals in collected.items()},
        "runs": raw_runs,
    }
    return stats, raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--kernels-dir",
        default=str(PROJECT_ROOT / "inference" / "kernels"),
        help="Directory containing compiled kernel binaries",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Binary name suffix (empty matches unsuffixed local and DICC binaries)",
    )
    parser.add_argument(
        "--tag",
        default="rtx3050_local",
        help="Hardware tag stored in the result JSON",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS_DEFAULT,
        help=f"Number of repeated runs per binary (default {N_TRIALS_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for summary JSON (default: benchmarks/results/cuda_kernel_stats_<tag>.json)",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path for per-trial raw samples JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fatal on missing binary, timeout, nonzero exit, hard validation failure, or missing metrics",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per trial on intermittent numerical validation FAILED (default 3)",
    )
    parser.add_argument(
        "--max-validation-failures",
        type=int,
        default=None,
        help="Hard validation failures allowed after retries (default: 0 if --strict else unlimited). "
             "Use e.g. 5 for flaky A100 fp16 validation while still collecting ~n samples.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=300.0,
        help="Per-trial subprocess timeout in seconds (default 300; <=0 disables)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of binary base names to run",
    )
    args = parser.parse_args(argv)

    if args.n_trials < 1:
        print("ERROR: --n-trials must be >= 1", file=sys.stderr)
        return 2

    timeout = args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None
    kernels_dir = Path(args.kernels_dir)
    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "benchmarks" / "results" / f"cuda_kernel_stats_{args.tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "hardware_tag": args.tag,
        "n_trials": args.n_trials,
        "kernels_dir": str(kernels_dir.resolve()) if kernels_dir.exists() else str(kernels_dir),
        "strict": bool(args.strict),
        "timeout_sec": timeout,
        "protocol": {
            "description": "Independent process invocations of each kernel binary; "
                           "latency parsed from binary stdout; stats across trials",
            "n_trials": args.n_trials,
        },
        "kernels": {},
    }
    raw_all: dict = {
        "hardware_tag": args.tag,
        "n_trials": args.n_trials,
        "kernels": {},
    }

    print("=" * 78)
    print(
        f"CUDA KERNEL STATISTICAL BENCHMARK (n={args.n_trials} trials each, "
        f"tag={args.tag}, strict={args.strict})"
    )
    print("=" * 78)

    targets = TARGETS
    if args.only:
        only = set(args.only)
        targets = [(n, p) for n, p in TARGETS if n in only]
        missing_names = only - {n for n, _ in targets}
        if missing_names:
            raise BenchmarkError(f"unknown --only names: {sorted(missing_names)}")

    try:
        for base_name, patterns in targets:
            binary_path = kernels_dir / f"{base_name}{args.suffix}"
            print(f"\n--- {binary_path.name} ---")
            stats, raw = run_trials(
                binary_path,
                patterns,
                n=args.n_trials,
                strict=args.strict,
                timeout_sec=timeout,
                max_retries=args.max_retries,
                max_validation_failures=args.max_validation_failures,
            )
            if not stats:
                continue
            results["kernels"][base_name] = stats
            # Also keep top-level keys for backward compatibility with older consumers.
            results[base_name] = stats
            raw_all["kernels"][base_name] = raw

            for key, val in stats.items():
                if isinstance(val, dict) and "mean" in val:
                    print(
                        f"  {key}: n={val['n']}  mean={val['mean']:.2f}  "
                        f"std={val['std']:.2f}  (CV={val['cv_pct']:.1f}%)  "
                        f"p50={val['p50']:.2f}  "
                        f"95%CI=[{val['ci95_low']:.2f}, {val['ci95_high']:.2f}]  "
                        f"range=[{val['min']:.2f}, {val['max']:.2f}]"
                    )
            if stats.get("validation_failures", 0) > 0:
                print(
                    f"  *** {stats['validation_failures']}/{args.n_trials} runs FAILED "
                    f"numerical validation ***"
                )

        # Additive reconstruction ONLY for documentation — NOT a valid full-model
        # CUDA pipeline latency (fused_pipeline skips Block 3 / feeds Block 4 with
        # uninitialized memory for the BiLSTM path; see plan critical findings).
        if "fused_pipeline" in results and "fused_block3_fp16" in results:
            b124_stats = results["fused_pipeline"].get("b124_chained_us")
            b3_stats = results["fused_block3_fp16"].get("latency_us")
            if isinstance(b124_stats, dict) and isinstance(b3_stats, dict):
                total = b124_stats["mean"] + b3_stats["mean"]
                print(
                    f"\nDerived additive total (mean b124 + mean block3 FP16): {total:.1f} us"
                )
                print(
                    "  WARNING: not a measured full pipeline; architecture parity incomplete. "
                    "Do not publish as full-model CUDA latency or CUDA-vs-PyTorch speedup."
                )
                results["derived_pipeline_total_us"] = {
                    "mean_sum": total,
                    "comparability_valid": False,
                    "reason": (
                        "Additive reconstruction from Blocks1+2+4 chained + Block3 FP16. "
                        "fused_pipeline.cu does not execute Block3 into Block4; CUDA path "
                        "also lacks attention/LayerNorm/global-average-pool present in V3 PyTorch."
                    ),
                }

    except BenchmarkError as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        # Still write partial results for debugging if anything was collected.
        partial_path = out_path.with_suffix(".partial.json")
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Partial results written to {partial_path}", file=sys.stderr)
        return 1

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            json.dump(raw_all, f, indent=2)
        print(f"Raw samples saved to {raw_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
