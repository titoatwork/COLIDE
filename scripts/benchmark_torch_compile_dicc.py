"""torch.compile vs eager full V3 on current GPU; Option A: report absolute full-model only."""
from __future__ import annotations
import argparse, json, time, math
from pathlib import Path
import numpy as np
import torch
import yaml
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention

def summarize(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "p50": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "cv_pct": float(100*std/mean) if mean else float("nan")}

def trial_median_us(fn, inner, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(inner):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(samples))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "model/best_model_botiot_twostage.pth"))
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--tag", default="gpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    model = CNNBiLSTMAttention(cfg).cuda().eval()
    sd = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(sd)
    x = torch.randn(1, 10, device="cuda")

    def eager():
        with torch.no_grad():
            model(x)

    eager_trials = []
    for i in range(args.n_trials):
        eager_trials.append(trial_median_us(eager, args.inner, args.warmup))
        print(f"eager trial {i+1}/{args.n_trials}: {eager_trials[-1]:.2f} us", flush=True)

    # reduce-overhead may fall back; catch failures
    compile_err = None
    compile_trials = []
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        def cfn():
            with torch.no_grad():
                compiled(x)
        # first call may compile
        with torch.no_grad():
            compiled(x)
        torch.cuda.synchronize()
        for i in range(args.n_trials):
            compile_trials.append(trial_median_us(cfn, args.inner, args.warmup))
            print(f"compile trial {i+1}/{args.n_trials}: {compile_trials[-1]:.2f} us", flush=True)
    except Exception as e:
        compile_err = repr(e)
        print("COMPILE_FAILED", compile_err, flush=True)

    out = {
        "tag": args.tag,
        "hardware": torch.cuda.get_device_name(0),
        "checkpoint": args.checkpoint,
        "protocol": {"n_trials": args.n_trials, "inner": args.inner, "warmup": args.warmup, "batch": 1},
        "eager_full_model_us": summarize(eager_trials),
        "torch_compile_full_model_us": summarize(compile_trials) if compile_trials else None,
        "torch_compile_error": compile_err,
        "notes": [
            "Full-model absolute latencies only; not Option A block parity.",
            "Do not ratio against incomplete Custom CUDA pipeline without caveat.",
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print("WROTE", args.output)
    print("eager", out["eager_full_model_us"])
    print("compile", out["torch_compile_full_model_us"])

if __name__ == "__main__":
    main()
