"""
COLIDE - PyTorch Block 3 (BiLSTM) Statistical Benchmark

Resolves HANDOFF.md open item #1: two conflicting single-run PyTorch cuDNN
baseline numbers existed for Block 3 alone (740.7us historical vs 943.6us
from a fresh benchmark_pipeline.py run). Phase 2.7's CUDA kernel statistical
work already showed this dev box has real run-to-run CV of 5-20%+ (worse for
some blocks) -- a single script invocation is just one draw from that
distribution, not a stable number.

Mirrors scripts/benchmark_cuda_kernels_stats.py's approach: run N independent
trials as separate subprocesses (capturing real process/GPU-state jitter, not
just intra-process noise) and report mean/std/percentiles instead of trusting
one run. Each trial internally times 200 iterations and takes the median
(same methodology benchmark_pipeline.py already uses per-invocation) -- the
statistics here are across trials, not within one.

Uses model/best_model.pth (same checkpoint benchmark_pipeline.py uses) --
latency is shape-, not weight-, dependent, so the stale-vs-twostage
checkpoint distinction that matters for accuracy claims doesn't apply here.

Usage:
    PYTHONPATH=. python scripts/benchmark_pytorch_block3_stats.py --n-trials 50
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_worker(runs):
    import torch
    import yaml

    from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda not available"}))
        return

    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    model = CNNBiLSTM(config)
    model.load_state_dict(
        torch.load(PROJECT_ROOT / "model" / "best_model.pth", map_location="cpu", weights_only=True)
    )
    model.eval()

    class Block3(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.bilstm1 = m.bilstm1
            self.bilstm2 = m.bilstm2
            self.dropout = m.dropout

        def forward(self, x):
            x, _ = self.bilstm1(x)
            x = self.dropout(x)
            x, _ = self.bilstm2(x)
            x = self.dropout(x)
            x = x[:, -1, :]  # last timestep, matching CUDA block 3
            return x

    block_gpu = copy.deepcopy(Block3(model)).eval().cuda()
    inp_gpu = torch.randn(1, 16, 128).cuda()

    for _ in range(20):
        with torch.no_grad():
            _ = block_gpu(inp_gpu)
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = block_gpu(inp_gpu)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    print(json.dumps({
        "gpu_p50_us": times[runs // 2],
        "gpu_p95_us": times[int(runs * 0.95)],
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--n-trials", type=int, default=50,
                         help="Number of independent subprocess trials")
    parser.add_argument("--inner-runs", type=int, default=200,
                         help="Timed iterations within each trial (matches benchmark_pipeline.py's runs=200)")
    parser.add_argument("--tag", default="rtx3050",
                         help="Hardware tag used in the output filename")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.inner_runs)
        return

    print("=" * 78)
    print(f"PYTORCH BLOCK 3 (BiLSTM) STATISTICAL BENCHMARK "
          f"(n={args.n_trials} independent-process trials, "
          f"{args.inner_runs} inner timed iterations each, tag={args.tag})")
    print("=" * 78)

    import subprocess

    p50s, p95s = [], []
    for i in range(args.n_trials):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             "--inner-runs", str(args.inner_runs)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            print(f"  trial {i + 1}/{args.n_trials}: FAILED to parse worker output")
            print(f"    stdout: {result.stdout!r}")
            print(f"    stderr: {result.stderr[-500:]!r}")
            continue
        if "error" in data:
            print(f"  trial {i + 1}/{args.n_trials}: {data['error']}")
            continue
        p50s.append(data["gpu_p50_us"])
        p95s.append(data["gpu_p95_us"])
        print(f"  trial {i + 1}/{args.n_trials}: p50={data['gpu_p50_us']:.1f} us")

    if not p50s:
        print("\nNo successful trials -- CUDA not available or all workers failed.")
        sys.exit(1)

    arr = np.array(p50s)
    stats = {
        "hardware_tag": args.tag,
        "n_trials": len(p50s),
        "inner_runs": args.inner_runs,
        "gpu_p50_us": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "cv_pct": float(arr.std() / arr.mean() * 100),
        },
    }
    print(f"\n{'=' * 78}")
    print(f"Block3 PyTorch GPU across {len(p50s)} independent trials: "
          f"mean={stats['gpu_p50_us']['mean']:.1f}us  "
          f"std={stats['gpu_p50_us']['std']:.1f}us "
          f"(CV={stats['gpu_p50_us']['cv_pct']:.1f}%)  "
          f"range=[{stats['gpu_p50_us']['min']:.1f}, {stats['gpu_p50_us']['max']:.1f}]")
    print(f"{'=' * 78}")

    out_path = PROJECT_ROOT / "benchmarks" / "results" / f"pytorch_block3_stats_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
