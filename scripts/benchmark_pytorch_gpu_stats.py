"""
COLIDE - PyTorch GPU statistical harness (full V3 model + Blocks 1–4)

Designed for DICC multi-day campaigns. Collects independent trial medians of
timed forward passes and reports mean/std/CI plus hardware/protocol metadata.

Protocol (defaults match the DICC plan):
  - 20 independent trial batches
  - 1,000 timed forwards per trial (after warmup)
  - Production checkpoint: model/best_model_botiot_twostage.pth
  - Full V3 model (CNN-BiLSTM-Attention) plus isolated Blocks 1–4

Comparability notes (encoded in the JSON):
  - Full CUDA-vs-PyTorch pipeline speedup is NOT valid:
      * PyTorch V3 runs attention + LayerNorm + global average pooling
      * CUDA kernels implement none of those; fused_pipeline skips Block 3
  - Block 3 (BiLSTM only, last-timestep reduce) is a latency head-to-head vs
    fused_block3 / fused_block3_fp16 only when CUDA reverse outputs are stored
    at original sequence positions (aligned) so extract at seq_len-1 matches
    PyTorch output[:, -1, :]. Full sequence remains preferred for V3 attention.

Usage:
    PYTHONPATH=. python scripts/benchmark_pytorch_gpu_stats.py \\
        --checkpoint model/best_model_botiot_twostage.pth \\
        --n-trials 20 --inner-forwards 1000 --tag a100 \\
        --output /path/to/run_dir/pytorch_gpu_stats.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
N_TRIALS_DEFAULT = 20
INNER_DEFAULT = 1000
WARMUP_DEFAULT = 50


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.96


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
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
        "values": [float(x) for x in arr.tolist()],
    }


def run_worker(checkpoint: str, inner: int, warmup: int, seed: int) -> None:
    """Single-process worker: load model once, time full + blocks, print JSON."""
    import torch
    import yaml

    from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda not available"}))
        return

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Torch 2.13+cu130 cuDNN refuses SM < 7.5 (V100 is 7.0). Prefer reinstalling
    # a cu121 wheel; as a last resort disable cuDNN so benches still complete.
    cudnn_mode = "enabled"
    props = torch.cuda.get_device_properties(0)
    sm75_or_newer = (props.major > 7) or (props.major == 7 and props.minor >= 5)
    if not sm75_or_newer:
        # Proactively avoid the hard crash path on V100 with modern cuDNN.
        try:
            torch.backends.cudnn.enabled = True
            x = torch.randn(1, 8, 16, device="cuda")
            w = torch.randn(8, 8, 3, device="cuda")
            with torch.no_grad():
                _ = torch.nn.functional.conv1d(x, w, padding=1)
        except RuntimeError as exc:
            msg = str(exc)
            torch.backends.cudnn.enabled = False
            cudnn_mode = f"disabled_for_sm_{props.major}.{props.minor}:{msg[:100]}"

    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    model = CNNBiLSTM(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    try:
        model_gpu = model.cuda()
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}))
        return

    class Block1(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.proj = m.input_projection
            self.conv1 = m.conv1
            self.bn1 = m.bn1
            self.relu = m.relu
            self.reshape_channels = m.reshape_channels
            self.reshape_length = m.reshape_length

        def forward(self, x):
            x = self.proj(x)
            x = x.view(x.size(0), self.reshape_channels, self.reshape_length)
            return self.relu(self.bn1(self.conv1(x)))

    class Block2(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.conv2 = m.conv2
            self.bn2 = m.bn2
            self.relu = m.relu
            self.pool = m.pool

        def forward(self, x):
            x = self.relu(self.bn2(self.conv2(x)))
            return self.pool(x)

    class Block3(torch.nn.Module):
        """BiLSTM stack + last-timestep.

        Matches CUDA Block 3 only after reverse sequence alignment (store at
        original pos). Prefer full sequence for V3 attention parity.
        """

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
            return x[:, -1, :]

    class Block4(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.fc1 = m.fc1
            self.fc2 = m.fc2
            self.relu = m.relu
            self.dropout = m.dropout

        def forward(self, x):
            x = self.dropout(self.relu(self.fc1(x)))
            return self.fc2(x)

    def time_module(mod, inp, n_inner: int, n_warmup: int) -> float:
        mod = mod.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = mod(inp)
        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(n_inner):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = mod(inp)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)
        times.sort()
        return times[len(times) // 2]

    # Shapes matching CUDA block contracts / model input
    inp_full = torch.randn(1, 10, device="cuda")
    inp_b1 = torch.randn(1, 10, device="cuda")
    inp_b2 = torch.randn(1, 64, 32, device="cuda")
    inp_b3 = torch.randn(1, 16, 128, device="cuda")
    inp_b4 = torch.randn(1, 128, device="cuda")

    b1 = copy.deepcopy(Block1(model_gpu)).cuda()
    b2 = copy.deepcopy(Block2(model_gpu)).cuda()
    b3 = copy.deepcopy(Block3(model_gpu)).cuda()
    b4 = copy.deepcopy(Block4(model_gpu)).cuda()

    props = torch.cuda.get_device_properties(0)
    payload = {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": int(props.total_memory),
        "checkpoint": str(checkpoint),
        "inner_forwards": inner,
        "warmup": warmup,
        "seed": seed,
        "torch_version": torch.__version__,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_mode": cudnn_mode,
        "full_model_us": time_module(model_gpu, inp_full, inner, warmup),
        "block1_us": time_module(b1, inp_b1, inner, warmup),
        "block2_us": time_module(b2, inp_b2, inner, warmup),
        "block3_us": time_module(b3, inp_b3, inner, warmup),
        "block4_us": time_module(b4, inp_b4, inner, warmup),
    }
    print(json.dumps(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Production checkpoint path (default: best_model_botiot_twostage.pth)",
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--inner-forwards", type=int, default=INNER_DEFAULT)
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    parser.add_argument("--tag", default="unknown_gpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--raw-output", default=None)
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run trials in-process instead of subprocesses (faster debug; less isolation)",
    )
    args = parser.parse_args(argv)

    if args.worker:
        run_worker(args.checkpoint, args.inner_forwards, args.warmup, args.worker_seed)
        return 0

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    if args.n_trials < 1 or args.inner_forwards < 1:
        print("ERROR: n-trials and inner-forwards must be >= 1", file=sys.stderr)
        return 2

    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "benchmarks" / "results" / f"pytorch_gpu_stats_{args.tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(
        f"PYTORCH GPU STATISTICAL HARNESS "
        f"(n={args.n_trials} trials × {args.inner_forwards} forwards, tag={args.tag})"
    )
    print(f"checkpoint={checkpoint}")
    print("=" * 78)

    keys = ["full_model_us", "block1_us", "block2_us", "block3_us", "block4_us"]
    series: dict[str, list[float]] = {k: [] for k in keys}
    raw_trials: list[dict] = []
    hw_meta: dict = {}

    for i in range(args.n_trials):
        seed = 42 + i
        if args.in_process:
            # Capture worker stdout by running inline via subprocess still is safer
            # for CUDA context hygiene; in-process is for unit tests without GPU.
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                run_worker(str(checkpoint.resolve()), args.inner_forwards, args.warmup, seed)
            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
            if not lines:
                print(f"  trial {i + 1}/{args.n_trials}: FAILED (empty worker output)")
                return 1
            data = json.loads(lines[-1])
            rc = 0
            stderr = ""
        else:
            child_env = os.environ.copy()
            child_env["PYTHONPATH"] = (
                str(PROJECT_ROOT)
                + (os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--checkpoint", str(checkpoint.resolve()),
                    "--inner-forwards", str(args.inner_forwards),
                    "--warmup", str(args.warmup),
                    "--worker-seed", str(seed),
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                env=child_env,
                check=False,
            )
            rc = result.returncode
            stderr = result.stderr or ""
            try:
                lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
                data = json.loads(lines[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                print(f"  trial {i + 1}/{args.n_trials}: FAILED to parse worker output")
                print(f"    returncode={rc}")
                print(f"    stdout_tail={(result.stdout or '')[-400:]!r}")
                print(f"    stderr_tail={stderr[-400:]!r}")
                print(f"    parse_error={exc}")
                return 1

        if rc != 0 and "error" not in data:
            print(f"  trial {i + 1}/{args.n_trials}: worker exit {rc}")
            print(f"    stderr_tail={stderr[-400:]!r}")
            return 1
        if "error" in data:
            print(f"  trial {i + 1}/{args.n_trials}: {data['error']}")
            return 1

        for k in keys:
            if k not in data:
                print(f"  trial {i + 1}: missing key {k}", file=sys.stderr)
                return 1
            series[k].append(float(data[k]))

        if not hw_meta:
            hw_meta = {
                "gpu_name": data.get("gpu_name"),
                "compute_capability": data.get("compute_capability"),
                "total_memory_bytes": data.get("total_memory_bytes"),
            }

        raw_trials.append(data)
        print(
            f"  trial {i + 1:>2}/{args.n_trials}: "
            f"full={data['full_model_us']:.1f}  "
            f"b1={data['block1_us']:.1f}  b2={data['block2_us']:.1f}  "
            f"b3={data['block3_us']:.1f}  b4={data['block4_us']:.1f} us"
        )

    stats = {k: summarize(v) for k, v in series.items()}

    # Sanitize GPU tag for any fallback naming
    gpu_tag = args.tag
    if hw_meta.get("gpu_name") and args.tag in ("unknown_gpu", "rtx3050_local"):
        pass  # keep user tag

    result = {
        "hardware_tag": gpu_tag,
        "hardware": hw_meta,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "protocol": {
            "n_trials": args.n_trials,
            "inner_forwards": args.inner_forwards,
            "warmup": args.warmup,
            "metric_per_trial": "median of timed CUDA-synchronized forwards",
            "isolation": "subprocess" if not args.in_process else "in_process",
            "model": "CNNBiLSTMAttention (V3)",
            "seed_base": 42,
        },
        "metrics": stats,
        # Convenience aliases
        "full_model_us": stats["full_model_us"],
        "block1_us": stats["block1_us"],
        "block2_us": stats["block2_us"],
        "block3_us": stats["block3_us"],
        "block4_us": stats["block4_us"],
        "comparability": {
            "full_pipeline_cuda_vs_pytorch": {
                "valid": False,
                "reason": (
                    "PyTorch V3 forward includes multi-head attention, LayerNorm residual, "
                    "and global average pooling after BiLSTM. CUDA kernels implement none of "
                    "those stages. fused_pipeline.cu additionally skips Block 3 and does not "
                    "feed a real BiLSTM output into Block 4 (additive reconstruction only). "
                    "Collect full-model PyTorch latency for absolute numbers, but do not "
                    "publish a full-pipeline CUDA/PyTorch speedup ratio until architecture "
                    "parity is fixed."
                ),
            },
            "block3_cuda_vs_pytorch": {
                "valid": True,
                "reason": (
                    "Block 3 isolates the 2-layer BiLSTM + last-timestep reduce. Latency "
                    "comparability holds when CUDA reverse outputs are position-aligned "
                    "(store at original pos) so seq_len-1 extract matches PyTorch "
                    "output[:, -1, :]. Full sequence is preferred for V3 attention; "
                    "last-timestep is the current harness contract."
                ),
                "pytorch_metric": "block3_us",
                "cuda_metrics": [
                    "fused_block3.with_graphs_us",
                    "fused_block3.no_graphs_us",
                    "fused_block3_fp16.latency_us",
                    "fused_block3_naive.latency_us",
                ],
            },
            "block1_block2_block4": {
                "valid": True,
                "note": "Per-block shapes match CUDA contracts; still report as per-block only.",
            },
        },
    }

    print(f"\n{'=' * 78}")
    for name in keys:
        s = stats[name]
        print(
            f"{name:<16} mean={s['mean']:.1f} us  std={s['std']:.1f}  "
            f"CV={s['cv_pct']:.1f}%  95%CI=[{s['ci95_low']:.1f}, {s['ci95_high']:.1f}]  "
            f"n={s['n']}"
        )
    print("comparability.full_pipeline_cuda_vs_pytorch.valid = false")
    print("comparability.block3_cuda_vs_pytorch.valid = true")
    print(f"{'=' * 78}")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            json.dump({"trials": raw_trials, "hardware_tag": gpu_tag}, f, indent=2)
        print(f"Raw trials saved to {raw_path}")

    return 0


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
