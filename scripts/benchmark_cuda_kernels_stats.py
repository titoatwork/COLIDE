"""
COLIDE - CUDA Kernel Statistical Benchmark

Runs each compiled CUDA kernel binary N_TRIALS times and computes real
mean/std/percentiles, instead of trusting a single point measurement.

Motivation: while re-verifying the manuscript's numbers (2026-07-01), a
single re-run of fused_block3_fp16 landed 24% away from the headline
"601.4 us" figure baked into README.md/ablation_study.py. Twenty repeated
runs showed why: normal run-to-run variance on this kernel is +-4-5%
(mean ~609us, std ~27us) -- a single run can land anywhere in a much wider
tail. Every kernel latency treated as a fixed constant elsewhere in this
repo should be backed by a distribution like this, not one run.

Usage:
    # Local dev (binaries are unsuffixed in inference/kernels/, matching the
    # Dockerfile's build command exactly -- the --suffix "_official" default
    # below describes a convention that was never actually adopted; confirmed
    # 2026-07-02, session 3, those binaries don't exist on disk):
    PYTHONPATH=. python scripts/benchmark_cuda_kernels_stats.py --suffix "" --tag rtx3050

    # DICC (binaries compiled by dicc_scripts/01_setup.sh into
    # inference/kernels/v100/ or inference/kernels/a100/, unsuffixed):
    PYTHONPATH=. python scripts/benchmark_cuda_kernels_stats.py --kernels-dir inference/kernels/v100 --suffix "" --tag v100s
    PYTHONPATH=. python scripts/benchmark_cuda_kernels_stats.py --kernels-dir inference/kernels/a100 --suffix "" --tag a100

Binaries must be compiled matching the Dockerfile's build command (no extra
-O flags -- confirmed 2026-07-01 that adding -O3 measurably changes the
timing-loop behavior, not just device-code optimization):
    nvcc -arch=sm_86 -o inference/kernels/fused_block1 inference/kernels/fused_block1.cu
    (repeat for block2, block3, block3_fp16, block4, pipeline)
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_TRIALS_DEFAULT = 20

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


def run_trials(binary_path, patterns, n=N_TRIALS_DEFAULT):
    if not binary_path.exists():
        print(f"[SKIP] {binary_path.name} not found -- compile it first")
        return None

    collected = {key: [] for key in patterns}
    failures = 0
    # Must be an absolute path: cwd= below re-resolves a relative binary_path
    # against the *new* cwd (binary_path.parent), not the original cwd -- a
    # relative --kernels-dir would otherwise silently look for
    # "<kernels-dir>/<kernels-dir>/<binary>" and crash with FileNotFoundError
    # (hit this 2026-07-02, session 3, re-running the RTX3050 stability check).
    resolved_path = binary_path.resolve()
    for _ in range(n):
        result = subprocess.run([str(resolved_path)], capture_output=True, text=True, cwd=binary_path.parent)
        stdout = result.stdout
        if "FAILED" in stdout:
            failures += 1
        for key, pattern in patterns.items():
            m = re.search(pattern, stdout, re.DOTALL)
            if m:
                collected[key].append(float(m.group(1)))

    stats = {"n_trials": n, "validation_failures": failures}
    for key, values in collected.items():
        if not values:
            continue
        arr = np.array(values)
        stats[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "cv_pct": float(arr.std() / arr.mean() * 100),
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels-dir", default=str(PROJECT_ROOT / "inference" / "kernels"),
                         help="Directory containing compiled kernel binaries")
    parser.add_argument("--suffix", default="",
                         help='Binary name suffix. Empty by default -- matches the actual '
                              'unsuffixed binaries in inference/kernels/ (local and DICC); '
                              'the "_official" convention this used to default to was never '
                              'actually adopted (confirmed 2026-07-02, session 3).')
    parser.add_argument("--tag", default="rtx3050_local",
                         help="Hardware tag used in the output filename, e.g. v100s, a100")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                         help="Number of repeated runs per binary")
    args = parser.parse_args()

    kernels_dir = Path(args.kernels_dir)
    results = {"hardware_tag": args.tag, "n_trials": args.n_trials}
    print("=" * 78)
    print(f"CUDA KERNEL STATISTICAL BENCHMARK (n={args.n_trials} trials each, "
          f"tag={args.tag})")
    print("=" * 78)
    for base_name, patterns in TARGETS:
        binary_path = kernels_dir / f"{base_name}{args.suffix}"
        print(f"\n--- {binary_path.name} ---")
        stats = run_trials(binary_path, patterns, n=args.n_trials)
        if stats is None:
            continue
        results[base_name] = stats
        for key, val in stats.items():
            if isinstance(val, dict):
                print(f"  {key}: mean={val['mean']:.2f}  std={val['std']:.2f}  "
                      f"(CV={val['cv_pct']:.1f}%)  p50={val['p50']:.2f}  "
                      f"range=[{val['min']:.2f}, {val['max']:.2f}]")
        if stats.get("validation_failures", 0) > 0:
            print(f"  *** {stats['validation_failures']}/{args.n_trials} runs FAILED "
                  f"numerical validation ***")

    # Derive the pipeline total the same additive way as fused_pipeline.cu /
    # the DICC summary txt files: measured(1+2+4) + measured(block3_fp16).
    if "fused_pipeline" in results and "fused_block3_fp16" in results:
        b124 = results["fused_pipeline"]["b124_chained_us"]["mean"]
        b3fp16 = results["fused_block3_fp16"]["latency_us"]["mean"]
        total = b124 + b3fp16
        print(f"\nDerived chained pipeline total (mean b124 + mean block3 FP16): "
              f"{total:.1f} us")
        results["derived_pipeline_total_us"] = total

    out_path = PROJECT_ROOT / "benchmarks" / "results" / f"cuda_kernel_stats_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
