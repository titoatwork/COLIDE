#!/usr/bin/env python3
"""Full multi-compiler matrix on DICC (absolute full-model latencies).

Measures (as available):
  - eager PyTorch full V3
  - torch.compile (reduce-overhead)
  - ONNX Runtime CUDA EP
  - ONNX Runtime CPU EP
  - TensorRT native (if tensorrt+pycuda importable)
  - ORT TensorrtExecutionProvider (if available)

Not Option A: full-model framework absolutes only.
Champion: model/best_model_botiot_twostage.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention  # noqa: E402


def summarize(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    if n == 0:
        return None
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "p50": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "cv_pct": float(100 * std / mean) if mean else float("nan"),
    }


def trial_median_us(fn, inner, warmup, cuda_sync=True):
    for _ in range(warmup):
        fn()
    if cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(inner):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(samples))


def multi_trial(fn, n_trials, inner, warmup, label, cuda_sync=True):
    trials = []
    for i in range(n_trials):
        v = trial_median_us(fn, inner, warmup, cuda_sync=cuda_sync)
        trials.append(v)
        print(f"{label} trial {i+1}/{n_trials}: {v:.2f} us", flush=True)
    return trials


def load_model(checkpoint: Path, cfg):
    model = CNNBiLSTMAttention(cfg).cuda().eval()
    sd = torch.load(str(checkpoint), map_location="cuda", weights_only=True)
    model.load_state_dict(sd)
    return model


def export_onnx(model, onnx_path: Path, batch: int = 1, feat: int = 10):
    dummy = torch.randn(batch, feat, device="cuda")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer dynamo=False for broader opset compatibility on older ORT/TRT
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
        )
    print(f"ONNX exported: {onnx_path}", flush=True)
    return onnx_path


def bench_ort(onnx_path: Path, provider: str, n_trials, inner, warmup, label):
    # TRT EP needs libnvinfer; prepare after torch has already used cuDNN9
    if provider == "TensorrtExecutionProvider":
        _prepare_trt_ld_path()
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider not in available:
        return None, f"provider {provider} not in {available}"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # CentOS7/SLURM: avoid pthread_setaffinity_np spam / invalid affinity masks
    try:
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
    except Exception:
        pass
    providers = [provider]
    # fallback chain for CUDA
    if provider == "CUDAExecutionProvider":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "TensorrtExecutionProvider":
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    except Exception as e:
        return None, repr(e)

    active = sess.get_providers()
    print(f"ORT session providers={active} requested={provider}", flush=True)
    # If requested GPU provider silently fell back, still report but flag
    got = active[0] if active else None
    x = np.random.randn(1, 10).astype(np.float32)

    def run():
        sess.run(None, {"input": x})

    # ORT may not need torch.cuda.synchronize for pure ORT path
    trials = multi_trial(run, n_trials, inner, warmup, label, cuda_sync=False)
    stats = summarize(trials)
    if stats is not None:
        stats["active_provider"] = got
        stats["requested_provider"] = provider
    return stats, None


def _prepare_trt_ld_path():
    """Prepend TRT8+cuDNN8 snapshot so libnvinfer loads on CentOS7 GPU nodes.

    Must run *after* torch eager/compile so torch keeps cuDNN9, while TRT gets cuDNN8.
    """
    import ctypes
    import os

    candidates = [
        Path.home() / "colide/third_party/trt8_libs",
        ROOT / "third_party/trt8_libs",
        Path(os.environ.get("COLIDE_TRT8_LIBS", "")),
    ]
    tp = next((p for p in candidates if p and p.is_dir() and (p / "libnvinfer.so.8").exists()), None)
    if tp is None:
        return None
    os.environ["LD_LIBRARY_PATH"] = f"{tp}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    for lib in (
        "libcudnn.so.8",
        "libnvinfer.so.8",
        "libnvinfer_plugin.so.8",
        "libnvonnxparser.so.8",
        "libnvparsers.so.8",
    ):
        p = tp / lib
        if p.exists():
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"preload {lib}: {e}", flush=True)
    print(f"TRT LD prepared: {tp}", flush=True)
    return tp


def _import_tensorrt():
    """Import TensorRT Python API (pip meta or tensorrt_bindings)."""
    _prepare_trt_ld_path()
    try:
        import tensorrt as trt  # type: ignore

        return trt, None
    except Exception as e1:
        try:
            import tensorrt_bindings.tensorrt as trt  # type: ignore

            return trt, None
        except Exception as e2:
            return None, f"import_failed: tensorrt={e1!r}; bindings={e2!r}"


def bench_tensorrt_native(onnx_path: Path, n_trials, inner, warmup, fp16: bool = True):
    """Native TensorRT builder path using torch CUDA buffers (no pycuda)."""
    trt, err = _import_tensorrt()
    if trt is None:
        return None, err

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            return None, f"parse_failed: {errs}"

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    use_fp16 = bool(fp16 and builder.platform_has_fast_fp16)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Building TensorRT engine (fp16={use_fp16})...", flush=True)
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except Exception as e:
        return None, f"build_failed: {e!r}"
    if engine_bytes is None:
        return None, "build_failed: engine_bytes is None"

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # Torch device buffers + tensor-address API (TRT 10+/11)
    d_input = torch.randn(1, 10, device="cuda", dtype=torch.float32)
    d_output = torch.empty(1, 5, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()

    input_name = output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
            context.set_input_shape(name, tuple(d_input.shape))
        else:
            output_name = name
    if input_name is None or output_name is None:
        return None, f"io_names_failed: in={input_name} out={output_name}"

    context.set_tensor_address(input_name, int(d_input.data_ptr()))
    context.set_tensor_address(output_name, int(d_output.data_ptr()))

    def run():
        with torch.cuda.stream(stream):
            ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v3 returned False")
        stream.synchronize()

    # Warmup once outside multi_trial to catch hard failures early
    try:
        run()
    except Exception as e:
        return None, f"execute_failed: {e!r}"

    trials = multi_trial(run, n_trials, inner, warmup, "tensorrt_native", cuda_sync=False)
    stats = summarize(trials)
    if stats is not None:
        stats["tensorrt_version"] = getattr(trt, "__version__", "tensorrt_bindings-8.6")
        stats["fp16"] = use_fp16
        stats["io"] = {"input": input_name, "output": output_name}
    return stats, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "model/best_model_botiot_twostage.pth"))
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--tag", default="gpu")
    ap.add_argument("--output", required=True)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-ort", action="store_true")
    ap.add_argument("--skip-trt", action="store_true")
    ap.add_argument(
        "--onnx",
        default="",
        help="Pre-exported ONNX path (recommended on DICC). If set, skip live export.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available", flush=True)
        sys.exit(2)

    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    ckpt = Path(args.checkpoint)
    model = load_model(ckpt, cfg)
    x = torch.randn(1, 10, device="cuda")

    out = {
        "tag": args.tag,
        "hardware": torch.cuda.get_device_name(0),
        "checkpoint": str(ckpt),
        "protocol": {
            "n_trials": args.n_trials,
            "inner": args.inner,
            "warmup": args.warmup,
            "batch": 1,
        },
        "notes": [
            "Full multi-compiler absolute latencies; not Option A block parity.",
            "Do not ratio against incomplete Custom CUDA pipeline without caveat.",
        ],
        "errors": {},
    }

    # --- Eager ---
    def eager():
        with torch.no_grad():
            model(x)

    out["eager_full_model_us"] = summarize(
        multi_trial(eager, args.n_trials, args.inner, args.warmup, "eager")
    )

    # --- torch.compile ---
    if not args.skip_compile:
        try:
            compiled = torch.compile(model, mode="reduce-overhead")

            def cfn():
                with torch.no_grad():
                    compiled(x)

            with torch.no_grad():
                compiled(x)
            torch.cuda.synchronize()
            out["torch_compile_full_model_us"] = summarize(
                multi_trial(cfn, args.n_trials, args.inner, args.warmup, "torch_compile")
            )
            out["torch_compile_error"] = None
        except Exception as e:
            out["torch_compile_full_model_us"] = None
            out["torch_compile_error"] = repr(e)
            out["errors"]["torch_compile"] = traceback.format_exc()
            print("COMPILE_FAILED", e, flush=True)
    else:
        out["torch_compile_full_model_us"] = None
        out["torch_compile_error"] = "skipped"

    # ONNX for ORT/TRT: prefer pre-exported file (login node export is more reliable)
    onnx_path = None
    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.is_file():
            out["errors"]["onnx_preexport"] = f"missing {onnx_path}"
            print("ONNX_MISSING", onnx_path, flush=True)
            onnx_path = None
        else:
            out["onnx_path"] = str(onnx_path)
            print(f"Using pre-exported ONNX: {onnx_path}", flush=True)
    else:
        onnx_path = Path(args.output).parent / f"colide_{args.tag}.onnx"
        try:
            # Ensure onnx package present (torch 2.5 exporter requires it)
            import onnx  # noqa: F401

            export_onnx(model, onnx_path)
            out["onnx_path"] = str(onnx_path)
        except Exception as e:
            out["onnx_path"] = None
            out["errors"]["onnx_export"] = repr(e)
            print("ONNX_EXPORT_FAILED", e, flush=True)
            onnx_path = None

    # Free some PyTorch state before heavy ORT/TRT if needed — keep model for now

    # --- ORT ---
    if not args.skip_ort and onnx_path is not None:
        try:
            import onnxruntime as ort

            out["ort_available_providers"] = ort.get_available_providers()
            out["ort_version"] = ort.__version__
        except Exception as e:
            out["ort_available_providers"] = []
            out["errors"]["ort_import"] = repr(e)
            ort = None

        if "ort_available_providers" in out and out["ort_available_providers"]:
            for key, prov, label in [
                ("ort_cuda_full_model_us", "CUDAExecutionProvider", "ort_cuda"),
                ("ort_cpu_full_model_us", "CPUExecutionProvider", "ort_cpu"),
                ("ort_tensorrt_ep_full_model_us", "TensorrtExecutionProvider", "ort_trt_ep"),
            ]:
                stats, err = bench_ort(
                    onnx_path, prov, args.n_trials, args.inner, args.warmup, label
                )
                out[key] = stats
                if err:
                    out["errors"][key] = err
                    print(f"{label}_SKIP/FAIL: {err}", flush=True)
    else:
        out["ort_cuda_full_model_us"] = None
        out["ort_cpu_full_model_us"] = None
        out["ort_tensorrt_ep_full_model_us"] = None

    # --- TensorRT native ---
    if not args.skip_trt and onnx_path is not None:
        stats, err = bench_tensorrt_native(
            onnx_path, args.n_trials, args.inner, args.warmup, fp16=True
        )
        out["tensorrt_native_full_model_us"] = stats
        if err:
            out["errors"]["tensorrt_native"] = err
            print("TRT_NATIVE_FAIL", err, flush=True)
            # try FP32 if FP16 path failed hard on build
            if "build_failed" in err or "parse_failed" in err:
                stats2, err2 = bench_tensorrt_native(
                    onnx_path, args.n_trials, args.inner, args.warmup, fp16=False
                )
                out["tensorrt_native_fp32_full_model_us"] = stats2
                if err2:
                    out["errors"]["tensorrt_native_fp32"] = err2
    else:
        out["tensorrt_native_full_model_us"] = None

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", args.output, flush=True)
    for k in [
        "eager_full_model_us",
        "torch_compile_full_model_us",
        "ort_cuda_full_model_us",
        "ort_cpu_full_model_us",
        "ort_tensorrt_ep_full_model_us",
        "tensorrt_native_full_model_us",
    ]:
        print(k, out.get(k), flush=True)
    if out.get("errors"):
        print("ERRORS", json.dumps(out["errors"], indent=2), flush=True)


if __name__ == "__main__":
    main()
