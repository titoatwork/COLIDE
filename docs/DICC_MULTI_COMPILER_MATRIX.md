# DICC full multi-compiler matrix (server)

**Date (UTC):** 2026-08-12  
**Scope:** Full-model absolute latencies on UM DICC (not Option A block parity).  
**Champion:** `model/best_model_botiot_twostage.pth` md5 `80a90f7…`  
**Harness:** `scripts/benchmark_multi_compiler_dicc.py`  
**ONNX:** pre-exported `benchmarks/results/dicc/framework/colide_champion.onnx` (login node; GPU nodes are CentOS 7 / glibc 2.17)  
**Protocol:** n_trials=20 · inner=200 · warmup=50 · batch=1  

---

## Jobs

| GPU | Job | State | JSON |
|-----|-----|-------|------|
| V100S | **395433** | COMPLETED | `framework/multi_compiler_v100s.json` |
| A100 | **395417** | COMPLETED | `framework/multi_compiler_a100.json` |

---

## Results (mean µs)

| Method | V100S | A100 | Notes |
|--------|------:|-----:|-------|
| Eager full V3 | **1041.3** | **931.5** | PyTorch 2.5.1+cu121 |
| torch.compile (`reduce-overhead`) | **865.0** | **770.2** | ~1.20–1.21× vs eager |
| ORT CUDA EP | **894.6** | **864.9** | onnxruntime-gpu **1.16.3** (manylinux2014) |
| ORT CPU EP | **500.1** | **460.7** | CPU EP on compute node (not laptop) |
| ORT TensorRT EP | **766.1** | **2032.7** | active=`TensorrtExecutionProvider` |
| TensorRT native FP16 | **528.3** | **587.7** | TensorRT **8.6** bindings + libs |

### Rank (fastest → slowest, this protocol)

| Rank | V100S | A100 |
|------|-------|------|
| 1 | **ORT CPU** 500 | **ORT CPU** 461 |
| 2 | **TRT native** 528 | **TRT native** 588 |
| 3 | **ORT TRT EP** 766 | **torch.compile** 770 |
| 4 | **torch.compile** 865 | **ORT CUDA** 865 |
| 5 | **ORT CUDA** 895 | **eager** 931 |
| 6 | **eager** 1041 | **ORT TRT EP** 2033 |

**Important:** ORT CPU winning absolute µs on server nodes is **not** a deployment claim for GPU inference; it is a valid measurement under this batch-1 protocol. ORT TensorRT EP on A100 is much slower than native TRT (engine path / EP overhead / fallback behavior differs).

---

## Environment notes (why these versions)

| Constraint | Resolution |
|------------|------------|
| GPU nodes: **CentOS 7 / glibc 2.17** | Cannot use modern manylinux_2_27/28 wheels (ORT 1.23, TRT 11 pip meta) |
| ORT | **1.16.3** manylinux2014 + **numpy 1.26.x** |
| TensorRT | **tensorrt_libs/bindings 8.6.1** + cuDNN8 snapshot in `~/colide/third_party/trt8_libs` |
| Torch | **2.5.1+cu121** with matching nvidia-* 12.1 packages (restored after cuDNN mess) |
| cuDNN clash | Torch needs cuDNN **9**; TRT 8 needs cuDNN **8** → TRT path preloads TRT8 libs **after** eager/compile |

---

## Paper-facing wording (approved shape)

> On UM DICC (CentOS 7 GPU nodes), a full-model multi-compiler matrix under a fixed multi-trial protocol shows: TensorRT 8.6 native FP16 means of **~528 µs (V100S)** and **~588 µs (A100)**; torch.compile **~865 / ~770 µs**; ORT-CUDA **~895 / ~865 µs**; eager **~1041 / ~932 µs**. ORT TensorRT EP is active on both GPUs but is not uniformly competitive with native TRT (notably slower on A100). These are absolute framework latencies, not Option A Custom CUDA parity. Laptop multi-compiler ranges remain a separate local story.

---

## Files

```text
scripts/benchmark_multi_compiler_dicc.py
logs/job_multi_compiler_{v100s,a100}.sh
benchmarks/results/dicc/framework/multi_compiler_{v100s,a100}.json
benchmarks/results/dicc/framework/colide_champion.onnx
third_party/trt8_libs/   # on cluster only (large; not git)
```

*End full multi-compiler pack.*
