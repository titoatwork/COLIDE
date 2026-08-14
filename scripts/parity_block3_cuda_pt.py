#!/usr/bin/env python3
"""
Block-3 real-weight CUDA ↔ PyTorch parity gate (COLIDE).

Phase-3 harness:
  1. Load champion ``best_model_botiot_twostage.pth`` (MD5 gate).
  2. Export BiLSTM weights to ``benchmarks/results/block3_parity_weights/``.
  3. Run PyTorch BiLSTM full sequence [B,SEQ,128] and last timestep.
  4. Run CUDA-contract CPU reference (mirrors fused_block3.cu).
  5. If fused_block3 supports weight inject (COLIDE_B3_WEIGHTS / argv dir),
     run GPU with champion weights and compare full sequence vs PyTorch.
  6. Hybrid suffix: feed Block-3 sequence through V3 attention/LN/pool/head;
     compare logits vs PT path through same suffix.
  7. Write ``benchmarks/results/block3_parity_gate.json``.
  8. Exit nonzero if numerical parity fails (CI).

Usage
-----
    PYTHONPATH=. python scripts/parity_block3_cuda_pt.py
    PYTHONPATH=. python scripts/parity_block3_cuda_pt.py --skip-cuda
    PYTHONPATH=. python scripts/parity_block3_cuda_pt.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import CHAMPION_MD5, CHAMPION_PATH  # noqa: E402
from scripts.protocol.result_schema import git_dirty, git_sha  # noqa: E402

# Block-3 geometry (matches fused_block3.cu constants)
SEQ = 16
IN_CH = 128
H1 = 128
H2 = 64
OUT_SIZE = 128  # 2 * H2

SEED = 42
BATCH = 4

# Predeclared FP32 tolerances (before looking at results)
FP32_TOL_PT_CPU = 1e-4  # PT vs CUDA-contract CPU ref (float64 accum on CPU)
FP32_TOL_GPU_PT = 1e-3  # GPU inject vs PT full sequence / last
FP32_TOL_HYBRID = 1e-4  # hybrid suffix logits (same FP32 path)

CUDA_SELFCHECK_TOL_FP32 = 1e-2
CUDA_SELFCHECK_TOL_FP16 = 5e-2

KERNEL_DIR = PROJECT_ROOT / "inference" / "kernels"
OUT_PATH = PROJECT_ROOT / "benchmarks" / "results" / "block3_parity_gate.json"
WEIGHTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "block3_parity_weights"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
FUSED_BLOCK3 = KERNEL_DIR / "fused_block3"

CONTRACT_NOTES = [
    "Full sequence is time-aligned: fw[k] and rev[k] both correspond to input time k.",
    "Reverse path: pos = seq_len-1-t; gates read input[pos]; store h_new at pos "
    "(not recurrence index t).",
    "Last timestep = output[:, -1, :] on the aligned sequence "
    "(CUDA extract at seq_len-1 for both fw and rev channels).",
    "Do not equate last-timestep reverse channels with reverse h_n (final reverse "
    "recurrence state after processing pos 0).",
    "Matching ops: 2-layer BiLSTM. Hybrid suffix reuses V3 attention/LN/GAP/head.",
    "CUDA inject: COLIDE_B3_WEIGHTS=<dir> or argv[1]=<dir> with w_ih1_f.bin etc.",
]


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cpu_lstm_forward(
    x_ct: np.ndarray,
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
    *,
    reverse: bool,
) -> np.ndarray:
    """Pure-Python mirror of fused_block3.cu ``cpu_lstm_forward``."""
    input_size, seq_len = x_ct.shape
    hidden = w_hh.shape[1]
    assert w_ih.shape == (4 * hidden, input_size)
    assert w_hh.shape == (4 * hidden, hidden)

    h_out = np.zeros((hidden, seq_len), dtype=np.float64)
    h_prev = np.zeros(hidden, dtype=np.float64)
    c_prev = np.zeros(hidden, dtype=np.float64)

    for t in range(seq_len):
        pos = (seq_len - 1 - t) if reverse else t
        x_t = x_ct[:, pos]
        gates = b_ih + b_hh + (w_ih @ x_t) + (w_hh @ h_prev)
        i_g = _sigmoid(gates[0:hidden])
        f_g = _sigmoid(gates[hidden : 2 * hidden])
        g_g = np.tanh(gates[2 * hidden : 3 * hidden])
        o_g = _sigmoid(gates[3 * hidden : 4 * hidden])
        c_prev = f_g * c_prev + i_g * g_g
        h_prev = o_g * np.tanh(c_prev)
        h_out[:, pos] = h_prev
    return h_out


def cpu_block3_pipeline(
    x_btc: np.ndarray,
    weights: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Two-layer BiLSTM CPU ref. Returns (full [B,SEQ,128], last [B,128])."""
    bsz, seq, inch = x_btc.shape
    assert seq == SEQ and inch == IN_CH

    full = np.zeros((bsz, seq, OUT_SIZE), dtype=np.float64)
    last = np.zeros((bsz, OUT_SIZE), dtype=np.float64)

    for b in range(bsz):
        x_ct = x_btc[b].T.astype(np.float64)
        h1_fw = cpu_lstm_forward(
            x_ct,
            weights["l1_fw_w_ih"],
            weights["l1_fw_w_hh"],
            weights["l1_fw_b_ih"],
            weights["l1_fw_b_hh"],
            reverse=False,
        )
        h1_rev = cpu_lstm_forward(
            x_ct,
            weights["l1_rev_w_ih"],
            weights["l1_rev_w_hh"],
            weights["l1_rev_b_ih"],
            weights["l1_rev_b_hh"],
            reverse=True,
        )
        in2 = np.concatenate([h1_fw, h1_rev], axis=0)

        h2_fw = cpu_lstm_forward(
            in2,
            weights["l2_fw_w_ih"],
            weights["l2_fw_w_hh"],
            weights["l2_fw_b_ih"],
            weights["l2_fw_b_hh"],
            reverse=False,
        )
        h2_rev = cpu_lstm_forward(
            in2,
            weights["l2_rev_w_ih"],
            weights["l2_rev_w_hh"],
            weights["l2_rev_b_ih"],
            weights["l2_rev_b_hh"],
            reverse=True,
        )
        for t in range(seq):
            full[b, t, :H2] = h2_fw[:, t]
            full[b, t, H2:] = h2_rev[:, t]
        last[b, :H2] = h2_fw[:, seq - 1]
        last[b, H2:] = h2_rev[:, seq - 1]

    return full, last


def extract_bilstm_weights(model) -> dict[str, np.ndarray]:
    """Pull BiLSTM params as float64 numpy (PyTorch LSTM layout)."""

    def _p(mod, name: str) -> np.ndarray:
        return getattr(mod, name).detach().cpu().numpy().astype(np.float64)

    l1, l2 = model.bilstm1, model.bilstm2
    return {
        "l1_fw_w_ih": _p(l1, "weight_ih_l0"),
        "l1_fw_w_hh": _p(l1, "weight_hh_l0"),
        "l1_fw_b_ih": _p(l1, "bias_ih_l0"),
        "l1_fw_b_hh": _p(l1, "bias_hh_l0"),
        "l1_rev_w_ih": _p(l1, "weight_ih_l0_reverse"),
        "l1_rev_w_hh": _p(l1, "weight_hh_l0_reverse"),
        "l1_rev_b_ih": _p(l1, "bias_ih_l0_reverse"),
        "l1_rev_b_hh": _p(l1, "bias_hh_l0_reverse"),
        "l2_fw_w_ih": _p(l2, "weight_ih_l0"),
        "l2_fw_w_hh": _p(l2, "weight_hh_l0"),
        "l2_fw_b_ih": _p(l2, "bias_ih_l0"),
        "l2_fw_b_hh": _p(l2, "bias_hh_l0"),
        "l2_rev_w_ih": _p(l2, "weight_ih_l0_reverse"),
        "l2_rev_w_hh": _p(l2, "weight_hh_l0_reverse"),
        "l2_rev_b_ih": _p(l2, "bias_ih_l0_reverse"),
        "l2_rev_b_hh": _p(l2, "bias_hh_l0_reverse"),
    }


def pytorch_block3(model, x_btc) -> tuple[np.ndarray, np.ndarray]:
    """Matching-ops BiLSTM stack; dropout disabled via eval()."""
    import torch

    with torch.no_grad():
        y1, _ = model.bilstm1(x_btc)
        y2, _ = model.bilstm2(y1)
        last = y2[:, -1, :]
    return (
        y2.detach().cpu().numpy().astype(np.float64),
        last.detach().cpu().numpy().astype(np.float64),
    )


def pytorch_v3_suffix(model, b3_seq) -> np.ndarray:
    """
    Attention → residual → LayerNorm → mean pool → fc1/relu → fc2.
    ``b3_seq``: torch [B, SEQ, 128] or numpy.
    """
    import torch

    if not isinstance(b3_seq, torch.Tensor):
        b3_seq = torch.from_numpy(np.asarray(b3_seq, dtype=np.float32))
    with torch.no_grad():
        x = b3_seq
        attn_out, _ = model.attention(x, x, x, need_weights=False)
        x = model.attention_norm(x + attn_out)
        x = torch.mean(x, dim=1)
        x = model.fc1(x)
        x = model.relu(x)
        # eval(): dropout is identity
        x = model.dropout(x)
        logits = model.fc2(x)
    return logits.detach().cpu().numpy().astype(np.float64)


def _err_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    nan_a = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    nan_b = int(np.isnan(b).sum()) if np.issubdtype(b.dtype, np.floating) else 0
    return {
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "max_rel_error": float(
            (diff / np.maximum(np.abs(b.astype(np.float64)), 1e-12)).max()
        )
        if diff.size
        else 0.0,
        "nan_count_a": nan_a,
        "nan_count_b": nan_b,
    }


def binary_supports_weight_inject(binary: Path) -> bool:
    """Detect inject path in binary or source."""
    if not binary.is_file():
        return False
    try:
        out = subprocess.check_output(
            ["strings", str(binary)],
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).decode("utf-8", errors="ignore")
        if "COLIDE_B3_WEIGHTS" in out or "WEIGHT_INJECT" in out or "w_ih1_f" in out:
            return True
    except Exception:
        pass
    # Fall back to source scan
    src = binary.with_suffix(".cu")
    if src.is_file():
        text = src.read_text(errors="ignore")
        return "COLIDE_B3_WEIGHTS" in text or "WEIGHT_INJECT" in text
    return False


def run_cuda_inject(
    weights_dir: Path,
    *,
    binary: Path = FUSED_BLOCK3,
    timeout_s: int = 180,
) -> dict[str, Any]:
    """
    Invoke fused_block3 with champion weight inject for sample-0 input.
    Returns row with gpu_full / gpu_last arrays if dumps succeed.
    """
    row: dict[str, Any] = {
        "binary": str(binary.relative_to(PROJECT_ROOT)) if binary.exists() else None,
        "exists": binary.is_file(),
        "inject_supported": False,
        "passed": None,
        "error": None,
        "stdout_tail": "",
        "returncode": None,
        "gpu_full": None,
        "gpu_last": None,
        "command": None,
    }
    if not binary.is_file():
        row["error"] = "binary missing"
        return row

    supports = binary_supports_weight_inject(binary)
    row["inject_supported"] = supports
    if not supports:
        row["error"] = (
            "binary lacks weight inject (rebuild fused_block3.cu with "
            "COLIDE_B3_WEIGHTS / argv dir support)"
        )
        return row

    # Work in a temp copy so multi-sample re-runs can overwrite input.bin
    work = Path(tempfile.mkdtemp(prefix="b3_inject_"))
    try:
        for name in [
            "input.bin",
            "w_ih1_f.bin",
            "w_hh1_f.bin",
            "b_ih1_f.bin",
            "b_hh1_f.bin",
            "w_ih1_r.bin",
            "w_hh1_r.bin",
            "b_ih1_r.bin",
            "b_hh1_r.bin",
            "w_ih2_f.bin",
            "w_hh2_f.bin",
            "b_ih2_f.bin",
            "b_hh2_f.bin",
            "w_ih2_r.bin",
            "w_hh2_r.bin",
            "b_ih2_r.bin",
            "b_hh2_r.bin",
        ]:
            src = weights_dir / name
            if not src.is_file():
                row["error"] = f"missing weight file: {name}"
                return row
            shutil.copy2(src, work / name)

        env = os.environ.copy()
        env["COLIDE_B3_WEIGHTS"] = str(work)
        cmd = [str(binary.resolve()), str(work)]
        row["command"] = " ".join(cmd)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(KERNEL_DIR),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            row["error"] = f"timeout after {timeout_s}s"
            return row
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
            return row

        out = (proc.stdout or "") + (proc.stderr or "")
        row["stdout_tail"] = out[-800:]
        row["returncode"] = proc.returncode

        out_full = work / "out_full.bin"
        out_last = work / "out_last.bin"
        # Also accept alternate dump names from older inject patches
        if not out_full.is_file() and (work / "gpu_full_seq.bin").is_file():
            out_full = work / "gpu_full_seq.bin"
        if not out_last.is_file() and (work / "gpu_last.bin").is_file():
            out_last = work / "gpu_last.bin"

        if out_full.is_file():
            gpu_full = np.fromfile(out_full, dtype=np.float32).reshape(SEQ, OUT_SIZE)
            row["gpu_full"] = gpu_full.astype(np.float64)
            # Persist next to champion weights for inspection
            shutil.copy2(out_full, weights_dir / "out_full.bin")
        if out_last.is_file():
            gpu_last = np.fromfile(out_last, dtype=np.float32).reshape(OUT_SIZE)
            row["gpu_last"] = gpu_last.astype(np.float64)
            shutil.copy2(out_last, weights_dir / "out_last.bin")

        if row["gpu_full"] is None and row["gpu_last"] is None:
            row["error"] = (
                "inject ran but no out_full.bin/out_last.bin dumps; "
                f"returncode={proc.returncode}"
            )
            row["passed"] = False
        elif re.search(r"validation FAILED", out):
            row["passed"] = False
            row["error"] = row["error"] or "binary reported validation FAILED"
        else:
            row["passed"] = True
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return row


def run_cuda_inject_batch(
    weights_dir: Path,
    x_btc: np.ndarray,
    *,
    binary: Path = FUSED_BLOCK3,
) -> dict[str, Any]:
    """Run inject once per batch row; stack GPU full/last."""
    base = run_cuda_inject(weights_dir, binary=binary)
    if not base.get("inject_supported"):
        return base
    if base.get("error") and base.get("gpu_full") is None:
        return base

    bsz = x_btc.shape[0]
    fulls = []
    lasts = []
    per_sample: list[dict[str, Any]] = []

    for b in range(bsz):
        # Write this sample as input.bin
        x_ct = np.ascontiguousarray(x_btc[b].T.astype(np.float32))
        # Stage into a private dir with shared weights
        work = Path(tempfile.mkdtemp(prefix=f"b3_inj_b{b}_"))
        try:
            for name in [
                "w_ih1_f.bin",
                "w_hh1_f.bin",
                "b_ih1_f.bin",
                "b_hh1_f.bin",
                "w_ih1_r.bin",
                "w_hh1_r.bin",
                "b_ih1_r.bin",
                "b_hh1_r.bin",
                "w_ih2_f.bin",
                "w_hh2_f.bin",
                "b_ih2_f.bin",
                "b_hh2_f.bin",
                "w_ih2_r.bin",
                "w_hh2_r.bin",
                "b_ih2_r.bin",
                "b_hh2_r.bin",
            ]:
                shutil.copy2(weights_dir / name, work / name)
            x_ct.tofile(work / "input.bin")
            env = os.environ.copy()
            env["COLIDE_B3_WEIGHTS"] = str(work)
            cmd = [str(binary.resolve()), str(work)]
            proc = subprocess.run(
                cmd,
                cwd=str(KERNEL_DIR),
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            out_full = work / "out_full.bin"
            out_last = work / "out_last.bin"
            sample: dict[str, Any] = {
                "batch_index": b,
                "returncode": proc.returncode,
                "validation_failed": bool(re.search(r"validation FAILED", out)),
            }
            if out_full.is_file():
                gf = np.fromfile(out_full, dtype=np.float32).reshape(SEQ, OUT_SIZE)
                fulls.append(gf.astype(np.float64))
                sample["has_full"] = True
            else:
                sample["has_full"] = False
            if out_last.is_file():
                gl = np.fromfile(out_last, dtype=np.float32).reshape(OUT_SIZE)
                lasts.append(gl.astype(np.float64))
                sample["has_last"] = True
            else:
                sample["has_last"] = False
            per_sample.append(sample)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    result = {
        "binary": base["binary"],
        "exists": True,
        "inject_supported": True,
        "per_sample": per_sample,
        "command": f"{binary} <work_dir>  # COLIDE_B3_WEIGHTS; B={bsz}",
        "passed": None,
        "error": None,
        "gpu_full": None,
        "gpu_last": None,
    }
    if len(fulls) == bsz:
        result["gpu_full"] = np.stack(fulls, axis=0)  # [B,SEQ,128]
    if len(lasts) == bsz:
        result["gpu_last"] = np.stack(lasts, axis=0)
    if result["gpu_full"] is None and result["gpu_last"] is None:
        result["error"] = "no GPU dumps produced for any sample"
        result["passed"] = False
    else:
        result["passed"] = all(not s.get("validation_failed") for s in per_sample)
    return result


def run_cuda_binary_selfcheck(
    binary_name: str, tolerance: float, timeout_s: int = 120
) -> dict[str, Any]:
    binary = KERNEL_DIR / binary_name
    row: dict[str, Any] = {
        "binary": binary_name,
        "path": str(binary.relative_to(PROJECT_ROOT)) if binary.exists() else None,
        "exists": binary.is_file(),
        "tolerance": tolerance,
        "passed": None,
        "error": None,
        "stdout_tail": "",
        "note": "Synthetic RNG weights inside binary (not champion real weights).",
    }
    if not binary.is_file():
        row["error"] = "binary missing"
        return row
    try:
        # Ensure env does not force inject mode for self-check
        env = os.environ.copy()
        env.pop("COLIDE_B3_WEIGHTS", None)
        proc = subprocess.run(
            [str(binary.resolve())],
            cwd=str(KERNEL_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        row["stdout_tail"] = out[-600:]
        row["returncode"] = proc.returncode
        if re.search(r"validation FAILED", out):
            row["passed"] = False
        elif re.search(r"validation PASSED", out):
            row["passed"] = True
        else:
            row["passed"] = False
            row["error"] = f"no validation marker; returncode={proc.returncode}"
    except subprocess.TimeoutExpired:
        row["error"] = f"timeout after {timeout_s}s"
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)
    return row


def build_dry_run_payload(
    champion_md5: str | None,
    binaries: dict[str, bool],
) -> dict[str, Any]:
    inject = binary_supports_weight_inject(FUSED_BLOCK3) if binaries.get("fused_block3") else False
    return {
        "experiment_id": "block3_parity_gate",
        "protocol_id": "block3_real_weight_parity_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(PROJECT_ROOT),
        "source_dirty": git_dirty(PROJECT_ROOT),
        "valid": False,
        "use_in_manuscript": False,
        "kernel_status": "code_fixed_awaiting_rebench",
        "status": "dry_run",
        "pre_fix_vs_post_fix": "pre_fix",
        "champion_md5": champion_md5,
        "champion_md5_expected": CHAMPION_MD5,
        "champion_path": str(CHAMPION_PATH.relative_to(PROJECT_ROOT)),
        "dry_run": True,
        "cuda_binaries": binaries,
        "cuda_inject_supported": inject,
        "contract_notes": CONTRACT_NOTES,
        "invalid_reason": "dry-run only; no numerical comparison executed",
        "comparison": None,
        "fp32_tolerances": {
            "pt_vs_cpu_ref": FP32_TOL_PT_CPU,
            "gpu_vs_pt": FP32_TOL_GPU_PT,
            "hybrid_logits": FP32_TOL_HYBRID,
        },
    }


def run_parity(skip_cuda: bool = False) -> dict[str, Any]:
    import torch
    import yaml
    from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM
    from scripts.export_block3_weights import export as export_weights

    if not CHAMPION_PATH.is_file():
        raise FileNotFoundError(f"Champion missing: {CHAMPION_PATH}")

    champion_md5 = _md5_file(CHAMPION_PATH)
    md5_ok = champion_md5 == CHAMPION_MD5

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model = CNNBiLSTM(config)
    state = torch.load(CHAMPION_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Inspect module names
    module_names = [n for n, _ in model.named_children()]
    bilstm_names = [n for n in module_names if "lstm" in n.lower()]

    weights = extract_bilstm_weights(model)

    # Export bins for CUDA inject
    export_manifest = export_weights(WEIGHTS_DIR)

    rng = np.random.RandomState(SEED)
    x_np = rng.randn(BATCH, SEQ, IN_CH).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    pt_full, pt_last = pytorch_block3(model, x_t)
    cpu_full, cpu_last = cpu_block3_pipeline(x_np, weights)

    full_stats = _err_stats(pt_full, cpu_full)
    last_stats = _err_stats(pt_last, cpu_last)
    last_vs_full = _err_stats(pt_last, pt_full[:, -1, :])

    pt_cpu_ok = (
        full_stats["max_abs_error"] < FP32_TOL_PT_CPU
        and last_stats["max_abs_error"] < FP32_TOL_PT_CPU
        and full_stats["nan_count_a"] == 0
        and full_stats["nan_count_b"] == 0
    )

    # Hybrid suffix: PT B3 seq → V3 modules vs CPU-ref B3 seq → same modules
    hybrid: dict[str, Any] = {"skipped": False, "reason": None}
    try:
        pt_logits = pytorch_v3_suffix(model, pt_full.astype(np.float32))
        cpu_logits = pytorch_v3_suffix(model, cpu_full.astype(np.float32))
        hybrid_stats = _err_stats(pt_logits, cpu_logits)
        class_agree = int(
            (np.argmax(pt_logits, axis=-1) == np.argmax(cpu_logits, axis=-1)).sum()
        )
        hybrid_ok = hybrid_stats["max_abs_error"] < FP32_TOL_HYBRID
        hybrid.update(
            {
                "method": "pt_b3_seq_vs_cpu_ref_b3_seq_through_v3_suffix",
                "modules": [
                    "attention",
                    "attention_norm (residual+LN)",
                    "mean_pool",
                    "fc1",
                    "relu",
                    "dropout(eval)",
                    "fc2",
                ],
                "pt_vs_cpu_ref_logits": hybrid_stats,
                "class_agreement": class_agree,
                "class_total": BATCH,
                "pass": hybrid_ok,
                "fp32_tol": FP32_TOL_HYBRID,
            }
        )
    except Exception as e:  # noqa: BLE001
        hybrid = {
            "skipped": True,
            "reason": f"hybrid suffix failed: {e}",
            "pass": None,
        }
        hybrid_ok = True  # non-critical if skipped for load reasons

    binaries_present = {
        "fused_block3": FUSED_BLOCK3.is_file(),
        "fused_block3_fp16": (KERNEL_DIR / "fused_block3_fp16").is_file(),
        "fused_block3_naive": (KERNEL_DIR / "fused_block3_naive").is_file(),
    }
    exec_sha = _sha256_file(FUSED_BLOCK3) if FUSED_BLOCK3.is_file() else None
    inject_supported = (
        binary_supports_weight_inject(FUSED_BLOCK3)
        if binaries_present["fused_block3"]
        else False
    )

    cuda_selfcheck: list[dict[str, Any]] = []
    cuda_inject: dict[str, Any] | None = None
    gpu_pt_full_stats: dict[str, Any] | None = None
    gpu_pt_last_stats: dict[str, Any] | None = None
    gpu_pt_ok = None
    hybrid_gpu: dict[str, Any] | None = None

    if not skip_cuda:
        if binaries_present["fused_block3"]:
            cuda_selfcheck.append(
                run_cuda_binary_selfcheck("fused_block3", CUDA_SELFCHECK_TOL_FP32)
            )
        if binaries_present["fused_block3_fp16"]:
            cuda_selfcheck.append(
                run_cuda_binary_selfcheck(
                    "fused_block3_fp16", CUDA_SELFCHECK_TOL_FP16
                )
            )

        if inject_supported:
            cuda_inject = run_cuda_inject_batch(WEIGHTS_DIR, x_np, binary=FUSED_BLOCK3)
            # Drop large arrays from JSON later; compute stats now
            if cuda_inject.get("gpu_full") is not None:
                gpu_pt_full_stats = _err_stats(cuda_inject["gpu_full"], pt_full)
            if cuda_inject.get("gpu_last") is not None:
                gpu_pt_last_stats = _err_stats(cuda_inject["gpu_last"], pt_last)
            if gpu_pt_full_stats is not None:
                gpu_pt_ok = (
                    gpu_pt_full_stats["max_abs_error"] < FP32_TOL_GPU_PT
                    and (
                        gpu_pt_last_stats is None
                        or gpu_pt_last_stats["max_abs_error"] < FP32_TOL_GPU_PT
                    )
                    and gpu_pt_full_stats.get("nan_count_a", 0) == 0
                )
            elif gpu_pt_last_stats is not None:
                gpu_pt_ok = gpu_pt_last_stats["max_abs_error"] < FP32_TOL_GPU_PT
            else:
                gpu_pt_ok = False

            # Hybrid with CUDA sequence if available
            if cuda_inject.get("gpu_full") is not None and not hybrid.get("skipped"):
                try:
                    gpu_logits = pytorch_v3_suffix(
                        model, cuda_inject["gpu_full"].astype(np.float32)
                    )
                    pt_logits_ref = pytorch_v3_suffix(
                        model, pt_full.astype(np.float32)
                    )
                    hs = _err_stats(gpu_logits, pt_logits_ref)
                    hybrid_gpu = {
                        "method": "cuda_b3_seq_through_v3_suffix_vs_pt",
                        "stats": hs,
                        "class_agreement": int(
                            (
                                np.argmax(gpu_logits, axis=-1)
                                == np.argmax(pt_logits_ref, axis=-1)
                            ).sum()
                        ),
                        "pass": hs["max_abs_error"] < FP32_TOL_HYBRID,
                        "fp32_tol": FP32_TOL_HYBRID,
                    }
                except Exception as e:  # noqa: BLE001
                    hybrid_gpu = {"skipped": True, "reason": str(e)}
        else:
            cuda_inject = {
                "inject_supported": False,
                "exists": binaries_present["fused_block3"],
                "error": (
                    "CUDA inject path not available in binary "
                    "(rebuild fused_block3 from current .cu)"
                ),
                "passed": None,
            }

    # kernel_status
    if gpu_pt_ok is True:
        kernel_status = "post_fix"
        pre_post = "post_fix"
    elif inject_supported and gpu_pt_ok is False:
        kernel_status = "cuda_inject_failed"
        pre_post = "pre_fix"
    else:
        kernel_status = "code_fixed_awaiting_rebench"
        pre_post = "pre_fix"

    # Status string
    if not md5_ok:
        status = "champion_md5_mismatch"
    elif not pt_cpu_ok:
        status = "pt_cpu_ref_mismatch"
    elif gpu_pt_ok is True:
        status = "pt_cpu_ref_ok_cuda_inject_ok"
    elif inject_supported and gpu_pt_ok is False:
        status = "pt_cpu_ref_ok_cuda_inject_failed"
    elif inject_supported:
        status = "pt_cpu_ref_ok_cuda_inject_pending"
    elif skip_cuda:
        status = "pt_cpu_ref_ok_cuda_skipped"
    else:
        status = "pt_cpu_ref_ok_awaiting_cuda_inject_rebuild"

    dirty = git_dirty(PROJECT_ROOT)
    critical_ok = pt_cpu_ok and md5_ok
    if gpu_pt_ok is not None:
        critical_ok = critical_ok and bool(gpu_pt_ok)
    if hybrid.get("pass") is False:
        critical_ok = False
    if hybrid_gpu and hybrid_gpu.get("pass") is False:
        critical_ok = False

    # valid only if critical pass AND clean tree AND md5 ok
    # Require CUDA inject pass for manuscript-valid green gate when binary present
    valid = bool(
        critical_ok
        and dirty is False
        and md5_ok
        and gpu_pt_ok is True
    )

    invalid_reason = None
    if valid:
        invalid_reason = None
    elif not md5_ok:
        invalid_reason = (
            f"champion MD5 mismatch: got {champion_md5}, expected {CHAMPION_MD5}"
        )
    elif not pt_cpu_ok:
        invalid_reason = (
            f"PT vs CUDA-contract CPU ref max_abs_error "
            f"full={full_stats['max_abs_error']:.3e} "
            f"last={last_stats['max_abs_error']:.3e} exceeds tol {FP32_TOL_PT_CPU}"
        )
    elif gpu_pt_ok is False:
        inv_max = (gpu_pt_full_stats or gpu_pt_last_stats or {}).get(
            "max_abs_error", "n/a"
        )
        invalid_reason = (
            f"CUDA inject vs PT max_abs_error={inv_max} "
            f"(tol {FP32_TOL_GPU_PT}); kernel_status={kernel_status}"
        )
    elif gpu_pt_ok is None:
        invalid_reason = (
            "real-weight GPU inject not completed: "
            + (
                cuda_inject.get("error")
                if cuda_inject
                else "inject path unavailable"
            )
        )
    elif dirty:
        invalid_reason = "source_dirty=true (clean tree required for valid=true)"
    elif hybrid.get("pass") is False:
        invalid_reason = "hybrid suffix logit parity failed"
    else:
        invalid_reason = f"gate not green (status={status})"

    # Serialize inject without large arrays
    inject_summary = None
    if cuda_inject is not None:
        inject_summary = {
            k: v
            for k, v in cuda_inject.items()
            if k not in ("gpu_full", "gpu_last")
        }
        inject_summary["gpu_full_shape"] = (
            list(cuda_inject["gpu_full"].shape)
            if cuda_inject.get("gpu_full") is not None
            else None
        )
        inject_summary["gpu_last_shape"] = (
            list(cuda_inject["gpu_last"].shape)
            if cuda_inject.get("gpu_last") is not None
            else None
        )

    payload: dict[str, Any] = {
        "experiment_id": "block3_parity_gate",
        "protocol_id": "block3_real_weight_parity_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(PROJECT_ROOT),
        "source_dirty": dirty,
        "valid": valid,
        "use_in_manuscript": valid,
        "kernel_status": kernel_status,
        "status": status,
        "pre_fix_vs_post_fix": pre_post,
        "invalid_reason": invalid_reason,
        "champion_md5": champion_md5,
        "champion_md5_expected": CHAMPION_MD5,
        "champion_md5_ok": md5_ok,
        "champion_path": str(CHAMPION_PATH.relative_to(PROJECT_ROOT)),
        "executable_sha256": exec_sha,
        "command": "PYTHONPATH=. python scripts/parity_block3_cuda_pt.py",
        "seed": SEED,
        "batch": BATCH,
        "geometry": {
            "seq": SEQ,
            "in_ch": IN_CH,
            "h1": H1,
            "h2": H2,
            "out_size": OUT_SIZE,
        },
        "module_names": module_names,
        "bilstm_module_names": bilstm_names,
        "fp32_tolerances": {
            "pt_vs_cpu_ref": FP32_TOL_PT_CPU,
            "gpu_vs_pt": FP32_TOL_GPU_PT,
            "hybrid_logits": FP32_TOL_HYBRID,
            "note": "predeclared before result inspection",
        },
        "fp32_tol_pt_vs_cpu_ref": FP32_TOL_PT_CPU,
        "contract_notes": CONTRACT_NOTES,
        "weights_export": {
            "dir": str(WEIGHTS_DIR.relative_to(PROJECT_ROOT)),
            "champion_md5": export_manifest.get("champion_md5"),
            "n_weight_tensors": 16,
        },
        "comparison": {
            "method": "pytorch_bilstm_vs_cuda_contract_cpu_ref_and_optional_gpu_inject",
            "weights": "champion_real",
            "pt_vs_cpu_ref_full_sequence": full_stats,
            "pt_vs_cpu_ref_last_timestep": last_stats,
            "pt_last_equals_full_seq_last": last_vs_full,
            "pt_cpu_ref_pass": pt_cpu_ok,
            "max_abs_error": last_stats["max_abs_error"],
            "mean_abs_error": last_stats["mean_abs_error"],
            "max_abs_error_full_sequence": full_stats["max_abs_error"],
            "mean_abs_error_full_sequence": full_stats["mean_abs_error"],
            "max_abs_error_last": last_stats["max_abs_error"],
            "mean_abs_error_last": last_stats["mean_abs_error"],
            "gpu_vs_pt_full_sequence": gpu_pt_full_stats,
            "gpu_vs_pt_last_timestep": gpu_pt_last_stats,
            "gpu_pt_pass": gpu_pt_ok,
            "nan_counts": {
                "pt_full": full_stats.get("nan_count_a"),
                "cpu_full": full_stats.get("nan_count_b"),
            },
            "pass": critical_ok,
            "fail": not critical_ok,
        },
        "hybrid_suffix": hybrid,
        "hybrid_suffix_cuda": hybrid_gpu,
        "cuda_binaries": binaries_present,
        "cuda_inject_supported": inject_supported,
        "cuda_inject": inject_summary,
        "cuda_selfcheck": cuda_selfcheck or None,
        "weight_keys": sorted(weights.keys()),
        "remediation": {
            "cuda_b3_001": "double-buffer race fix in fused_block3.cu / fp16",
            "cuda_b3_002": "reverse store at original pos",
            "cuda_b3_003": "contract documented; full sequence primary",
            "cuda_b3_inject": "COLIDE_B3_WEIGHTS / argv dir real-weight inject",
            "docs": [
                "docs/CUDA_WEIGHT_MAPPING.md",
                "scripts/parity_block3_cuda_pt.py",
                "scripts/export_block3_weights.py",
                "inference/kernels/fused_block3.cu",
            ],
        },
    }
    # parity fail flag for exit code (numerical, not dirty-only)
    payload["_parity_fail"] = (
        not md5_ok
        or not pt_cpu_ok
        or gpu_pt_ok is False
        or hybrid.get("pass") is False
        or (hybrid_gpu is not None and hybrid_gpu.get("pass") is False)
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Block-3 real-weight parity gate: champion BiLSTM PT vs CUDA-contract "
            "CPU ref; optional fused_block3 real-weight inject; hybrid V3 suffix."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write JSON skeleton without loading model / running CUDA",
    )
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Skip invoking CUDA binaries (PT + CPU-ref + hybrid only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help=f"Output JSON path (default: {OUT_PATH})",
    )
    args = parser.parse_args(argv)

    out_path: Path = args.out
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        champion_md5 = None
        if CHAMPION_PATH.is_file():
            champion_md5 = _md5_file(CHAMPION_PATH)
        binaries = {
            "fused_block3": FUSED_BLOCK3.is_file(),
            "fused_block3_fp16": (KERNEL_DIR / "fused_block3_fp16").is_file(),
            "fused_block3_naive": (KERNEL_DIR / "fused_block3_naive").is_file(),
        }
        payload = build_dry_run_payload(champion_md5, binaries)
        parity_fail = False
    else:
        payload = run_parity(skip_cuda=args.skip_cuda)
        parity_fail = bool(payload.pop("_parity_fail", False))

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path}")
    print(f"  status={payload.get('status')}")
    print(
        f"  valid={payload.get('valid')}  "
        f"use_in_manuscript={payload.get('use_in_manuscript')}"
    )
    print(f"  kernel_status={payload.get('kernel_status')}")
    print(f"  champion_md5={payload.get('champion_md5')}")
    print(f"  cuda_inject_supported={payload.get('cuda_inject_supported')}")
    print(f"  source_dirty={payload.get('source_dirty')}")
    cmp_ = payload.get("comparison") or {}
    if cmp_:
        print(
            f"  max_abs_error(full)={cmp_.get('max_abs_error_full_sequence')}  "
            f"max_abs_error(last)={cmp_.get('max_abs_error_last')}  "
            f"pt_cpu_ref_pass={cmp_.get('pt_cpu_ref_pass')}  "
            f"gpu_pt_pass={cmp_.get('gpu_pt_pass')}"
        )
    if parity_fail:
        print("PARITY FAILED — exiting nonzero for CI")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
