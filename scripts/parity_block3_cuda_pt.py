#!/usr/bin/env python3
"""
Block-3 real-weight CUDA ↔ PyTorch parity gate (COLIDE).

Purpose
-------
Compare champion BiLSTM Block 3 under the documented matching-op contract
against a pure-Python CPU reference that mirrors ``fused_block3.cu``'s
``cpu_lstm_forward`` (gate order i,f,g,o; reverse store at original pos;
last-timestep extract at seq_len-1 on the aligned sequence).

Standalone CUDA binaries under ``inference/kernels/`` currently self-check
with *synthetic* RNG weights (no real-weight inject path). When present they
are still invoked for GPU-vs-in-binary-CPU PASS/FAIL. Full GPU real-weight
parity remains gated on a rebench after the 2026-08-14 race+align source fix.

Contract (canonical for this harness)
-------------------------------------
* Input (matching ops): ``[B, SEQ=16, IN_CH=128]`` time-major (PyTorch
  ``batch_first`` after Block-2 permute).
* Full sequence: ``[B, SEQ, 2*H2=128]`` with channels ``fw | rev`` at each
  time index *k*; both directions' outputs at *k* correspond to input time *k*
  (reverse path stores at original ``pos``, not recurrence index ``t``).
* Last-timestep extract: ``output[:, -1, :]`` on that aligned sequence —
  shape ``[B, 128]``. This matches CUDA ``extract_last_timestep_kernel`` after
  alignment (not reverse recurrence final state at pos 0 / ``h_n`` reverse).
* Attention / LN / GAP are **out of scope** (Option A; CLAIM-PIPE-001).

Usage
-----
    PYTHONPATH=. python scripts/parity_block3_cuda_pt.py
    PYTHONPATH=. python scripts/parity_block3_cuda_pt.py --dry-run
    PYTHONPATH=. python scripts/parity_block3_cuda_pt.py --help

Safety: read-only on champion weights; writes only
``benchmarks/results/block3_parity_gate.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
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
FP32_TOL = 1e-4  # PT vs CPU-ref on real weights (float64 accum)
CUDA_SELFCHECK_TOL_FP32 = 1e-2
CUDA_SELFCHECK_TOL_FP16 = 5e-2

KERNEL_DIR = PROJECT_ROOT / "inference" / "kernels"
OUT_PATH = PROJECT_ROOT / "benchmarks" / "results" / "block3_parity_gate.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

CONTRACT_NOTES = [
    "Full sequence is time-aligned: fw[k] and rev[k] both correspond to input time k.",
    "Reverse path: pos = seq_len-1-t; gates read input[pos]; store h_new at pos "
    "(not recurrence index t).",
    "Last timestep = output[:, -1, :] on the aligned sequence "
    "(CUDA extract at seq_len-1 for both fw and rev channels).",
    "Do not equate last-timestep reverse channels with reverse h_n (final reverse "
    "recurrence state after processing pos 0).",
    "Matching ops only (2-layer BiLSTM). V3 attention/LN/GAP not in CUDA Block 3.",
    "kernel_status=code_fixed_awaiting_rebench: race+align fixed in source 2026-08-14; "
    "DICC/laptop wall-clock numbers remain pre_fix until rebench + this gate green.",
]


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
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
    """
    Pure-Python mirror of fused_block3.cu ``cpu_lstm_forward``.

    Parameters
    ----------
    x_ct : (input_size, seq_len) channel-major float64
    w_ih : (4H, input_size)
    w_hh : (4H, H)  — host / PyTorch layout (not transposed)
    b_ih, b_hh : (4H,)
    reverse : if True, process pos = seq_len-1-t and store at pos

    Returns
    -------
    h_out : (H, seq_len) channel-major, time-aligned
    """
    input_size, seq_len = x_ct.shape
    hidden = w_hh.shape[1]
    assert w_ih.shape == (4 * hidden, input_size)
    assert w_hh.shape == (4 * hidden, hidden)
    assert b_ih.shape == (4 * hidden,)
    assert b_hh.shape == (4 * hidden,)

    h_out = np.zeros((hidden, seq_len), dtype=np.float64)
    h_prev = np.zeros(hidden, dtype=np.float64)
    c_prev = np.zeros(hidden, dtype=np.float64)

    for t in range(seq_len):
        pos = (seq_len - 1 - t) if reverse else t
        x_t = x_ct[:, pos]  # (input_size,)
        # gates: i, f, g, o  — each (H,)
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
    """
    Two-layer BiLSTM CPU ref matching CUDA combine + last extract.

    x_btc : (B, SEQ, IN_CH) time-major
    Returns (full_seq [B,SEQ,128], last [B,128])
    """
    bsz, seq, inch = x_btc.shape
    assert seq == SEQ and inch == IN_CH

    full = np.zeros((bsz, seq, OUT_SIZE), dtype=np.float64)
    last = np.zeros((bsz, OUT_SIZE), dtype=np.float64)

    for b in range(bsz):
        # channel-major for layer 1
        x_ct = x_btc[b].T.astype(np.float64)  # (IN_CH, SEQ)
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
        # combine: (H1x2, SEQ) = fw | rev (CUDA combine_kernel)
        in2 = np.concatenate([h1_fw, h1_rev], axis=0)  # (256, SEQ)

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
        # time-major full sequence: (SEQ, 128) = fw|rev per timestep
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


def pytorch_block3(
    model, x_btc
) -> tuple[np.ndarray, np.ndarray]:
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


def _err_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "max_rel_error": float(
            (diff / np.maximum(np.abs(b.astype(np.float64)), 1e-12)).max()
        )
        if diff.size
        else 0.0,
    }


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
        proc = subprocess.run(
            [str(binary.resolve())],
            cwd=str(KERNEL_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_s,
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
    status = "needs_cuda_binary"
    if binaries.get("fused_block3") or binaries.get("fused_block3_fp16"):
        status = "dry_run_binaries_present"
    return {
        "experiment_id": "block3_parity_gate",
        "protocol_id": "block3_real_weight_parity_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(PROJECT_ROOT),
        "source_dirty": git_dirty(PROJECT_ROOT),
        "valid": False,
        "use_in_manuscript": False,
        "kernel_status": "code_fixed_awaiting_rebench",
        "status": status,
        "pre_fix_vs_post_fix": "pre_fix",
        "champion_md5": champion_md5,
        "champion_md5_expected": CHAMPION_MD5,
        "champion_path": str(CHAMPION_PATH.relative_to(PROJECT_ROOT)),
        "dry_run": True,
        "cuda_binaries": binaries,
        "contract_notes": CONTRACT_NOTES,
        "invalid_reason": (
            "dry-run only; no numerical comparison executed"
        ),
        "comparison": None,
        "cuda_selfcheck": None,
    }


def run_parity(skip_cuda: bool = False) -> dict[str, Any]:
    import torch
    import yaml
    from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

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

    weights = extract_bilstm_weights(model)

    rng = np.random.RandomState(SEED)
    x_np = rng.randn(BATCH, SEQ, IN_CH).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    pt_full, pt_last = pytorch_block3(model, x_t)
    cpu_full, cpu_last = cpu_block3_pipeline(x_np, weights)

    full_stats = _err_stats(pt_full, cpu_full)
    last_stats = _err_stats(pt_last, cpu_last)

    pt_cpu_ok = (
        full_stats["max_abs_error"] < FP32_TOL
        and last_stats["max_abs_error"] < FP32_TOL
    )

    # Optional: also check last-timestep identity vs full[:, -1, :]
    last_vs_full = _err_stats(pt_last, pt_full[:, -1, :])

    bin_fp32 = KERNEL_DIR / "fused_block3"
    bin_fp16 = KERNEL_DIR / "fused_block3_fp16"
    binaries_present = {
        "fused_block3": bin_fp32.is_file(),
        "fused_block3_fp16": bin_fp16.is_file(),
        "fused_block3_naive": (KERNEL_DIR / "fused_block3_naive").is_file(),
    }

    cuda_selfcheck: list[dict[str, Any]] = []
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

    if not any(binaries_present.values()):
        status = "needs_cuda_binary"
    elif not cuda_selfcheck and skip_cuda:
        status = "pt_cpu_ref_only_cuda_skipped"
    else:
        # Real-weight path is PT vs CUDA-contract CPU ref; GPU self-check is synthetic
        any_fail = any(r.get("passed") is False for r in cuda_selfcheck)
        any_pass = any(r.get("passed") is True for r in cuda_selfcheck)
        if pt_cpu_ok and any_pass and not any_fail:
            status = "pt_cpu_ref_ok_cuda_selfcheck_ok_awaiting_real_weight_gpu"
        elif pt_cpu_ok:
            status = "pt_cpu_ref_ok_awaiting_cuda_real_weight_or_selfcheck"
        else:
            status = "pt_cpu_ref_mismatch"

    # Gate remains non-manuscript until post_fix rebench with real-weight GPU inject
    valid = False
    invalid_reason = (
        "kernel_status=code_fixed_awaiting_rebench: source race+align fixed; "
        "wall-clock DICC/laptop B3 numbers stay pre_fix until rebench; "
        "real-weight GPU inject into fused_block3 binaries not yet wired — "
        "PT vs CUDA-contract CPU ref is intermediate evidence only"
    )
    if not md5_ok:
        invalid_reason = (
            f"champion MD5 mismatch: got {champion_md5}, expected {CHAMPION_MD5}"
        )
        status = "champion_md5_mismatch"
    elif not pt_cpu_ok:
        invalid_reason = (
            f"PT vs CUDA-contract CPU ref max_abs_error "
            f"full={full_stats['max_abs_error']:.3e} "
            f"last={last_stats['max_abs_error']:.3e} exceeds tol {FP32_TOL}"
        )

    payload: dict[str, Any] = {
        "experiment_id": "block3_parity_gate",
        "protocol_id": "block3_real_weight_parity_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(PROJECT_ROOT),
        "source_dirty": git_dirty(PROJECT_ROOT),
        "valid": valid,
        "use_in_manuscript": False,
        "kernel_status": "code_fixed_awaiting_rebench",
        "status": status,
        "pre_fix_vs_post_fix": "pre_fix",
        "invalid_reason": invalid_reason,
        "champion_md5": champion_md5,
        "champion_md5_expected": CHAMPION_MD5,
        "champion_md5_ok": md5_ok,
        "champion_path": str(CHAMPION_PATH.relative_to(PROJECT_ROOT)),
        "seed": SEED,
        "batch": BATCH,
        "geometry": {
            "seq": SEQ,
            "in_ch": IN_CH,
            "h1": H1,
            "h2": H2,
            "out_size": OUT_SIZE,
        },
        "fp32_tol_pt_vs_cpu_ref": FP32_TOL,
        "contract_notes": CONTRACT_NOTES,
        "comparison": {
            "method": "pytorch_bilstm_vs_cuda_contract_cpu_ref",
            "weights": "champion_real",
            "pt_vs_cpu_ref_full_sequence": full_stats,
            "pt_vs_cpu_ref_last_timestep": last_stats,
            "pt_last_equals_full_seq_last": last_vs_full,
            "pt_cpu_ref_pass": pt_cpu_ok,
            "max_abs_error": last_stats["max_abs_error"],
            "mean_abs_error": last_stats["mean_abs_error"],
            "max_abs_error_full_sequence": full_stats["max_abs_error"],
            "mean_abs_error_full_sequence": full_stats["mean_abs_error"],
        },
        "cuda_binaries": binaries_present,
        "cuda_selfcheck": cuda_selfcheck or None,
        "weight_keys": sorted(weights.keys()),
        "remediation": {
            "cuda_b3_001": "double-buffer race fix in fused_block3.cu / fp16 (2026-08-14)",
            "cuda_b3_002": "reverse store at original pos (2026-08-14)",
            "cuda_b3_003": "contract documented; last-timestep = aligned output[:, -1, :]",
            "docs": [
                "docs/CUDA_WEIGHT_MAPPING.md",
                "scripts/parity_block3_cuda_pt.py",
                "inference/kernels/fused_block3.cu",
                "inference/kernels/fused_block3_fp16.cu",
            ],
        },
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Block-3 real-weight parity gate: champion BiLSTM PT vs CUDA-contract "
            "CPU ref; optional fused_block3 binary self-check."
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
        help="Skip invoking CUDA binaries (PT + CPU-ref only)",
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
            "fused_block3": (KERNEL_DIR / "fused_block3").is_file(),
            "fused_block3_fp16": (KERNEL_DIR / "fused_block3_fp16").is_file(),
            "fused_block3_naive": (KERNEL_DIR / "fused_block3_naive").is_file(),
        }
        payload = build_dry_run_payload(champion_md5, binaries)
    else:
        payload = run_parity(skip_cuda=args.skip_cuda)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path}")
    print(f"  status={payload.get('status')}")
    print(f"  valid={payload.get('valid')}  use_in_manuscript={payload.get('use_in_manuscript')}")
    print(f"  kernel_status={payload.get('kernel_status')}")
    print(f"  champion_md5={payload.get('champion_md5')}")
    cmp_ = payload.get("comparison") or {}
    if cmp_:
        print(
            f"  max_abs_error(last)={cmp_.get('max_abs_error')}  "
            f"mean_abs_error(last)={cmp_.get('mean_abs_error')}  "
            f"pt_cpu_ref_pass={cmp_.get('pt_cpu_ref_pass')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
