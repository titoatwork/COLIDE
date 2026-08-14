#!/usr/bin/env python3
"""
Export champion BiLSTM (Block-3) weights for CUDA real-weight inject.

Writes float32 little-endian ``.bin`` files (row-major, PyTorch LSTM layout)
using the filenames expected by ``inference/kernels/fused_block3.cu`` inject mode
(``COLIDE_B3_WEIGHTS`` / argv[1]), plus a fixed seed-42 input and a JSON manifest.

Default output: ``benchmarks/results/block3_parity_weights/``

Usage
-----
    PYTHONPATH=. python scripts/export_block3_weights.py
    PYTHONPATH=. python scripts/export_block3_weights.py --out benchmarks/results/block3_parity_weights
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import CHAMPION_MD5, CHAMPION_PATH  # noqa: E402

# Block-3 geometry (matches fused_block3.cu)
SEQ = 16
IN_CH = 128
H1 = 128
H2 = 64
OUT_SIZE = 128
SEED = 42
BATCH = 4

DEFAULT_OUT = PROJECT_ROOT / "benchmarks" / "results" / "block3_parity_weights"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# CUDA inject basenames (fused_block3.cu WEIGHT_INJECT_MODE) → (state_dict key, shape)
# Gate order i,f,g,o; row-major float32.
CUDA_WEIGHT_FILES: dict[str, tuple[str, tuple[int, ...]]] = {
    "w_ih1_f": ("bilstm1.weight_ih_l0", (4 * H1, IN_CH)),
    "w_hh1_f": ("bilstm1.weight_hh_l0", (4 * H1, H1)),
    "b_ih1_f": ("bilstm1.bias_ih_l0", (4 * H1,)),
    "b_hh1_f": ("bilstm1.bias_hh_l0", (4 * H1,)),
    "w_ih1_r": ("bilstm1.weight_ih_l0_reverse", (4 * H1, IN_CH)),
    "w_hh1_r": ("bilstm1.weight_hh_l0_reverse", (4 * H1, H1)),
    "b_ih1_r": ("bilstm1.bias_ih_l0_reverse", (4 * H1,)),
    "b_hh1_r": ("bilstm1.bias_hh_l0_reverse", (4 * H1,)),
    "w_ih2_f": ("bilstm2.weight_ih_l0", (4 * H2, 2 * H1)),
    "w_hh2_f": ("bilstm2.weight_hh_l0", (4 * H2, H2)),
    "b_ih2_f": ("bilstm2.bias_ih_l0", (4 * H2,)),
    "b_hh2_f": ("bilstm2.bias_hh_l0", (4 * H2,)),
    "w_ih2_r": ("bilstm2.weight_ih_l0_reverse", (4 * H2, 2 * H1)),
    "w_hh2_r": ("bilstm2.weight_hh_l0_reverse", (4 * H2, H2)),
    "b_ih2_r": ("bilstm2.bias_ih_l0_reverse", (4 * H2,)),
    "b_hh2_r": ("bilstm2.bias_hh_l0_reverse", (4 * H2,)),
}

# Friendly aliases used by the Python parity harness (same bytes as CUDA names).
ALIAS_MAP: dict[str, str] = {
    "l1_fw_w_ih": "w_ih1_f",
    "l1_fw_w_hh": "w_hh1_f",
    "l1_fw_b_ih": "b_ih1_f",
    "l1_fw_b_hh": "b_hh1_f",
    "l1_rev_w_ih": "w_ih1_r",
    "l1_rev_w_hh": "w_hh1_r",
    "l1_rev_b_ih": "b_ih1_r",
    "l1_rev_b_hh": "b_hh1_r",
    "l2_fw_w_ih": "w_ih2_f",
    "l2_fw_w_hh": "w_hh2_f",
    "l2_fw_b_ih": "b_ih2_f",
    "l2_fw_b_hh": "b_hh2_f",
    "l2_rev_w_ih": "w_ih2_r",
    "l2_rev_w_hh": "w_hh2_r",
    "l2_rev_b_ih": "b_ih2_r",
    "l2_rev_b_hh": "b_hh2_r",
}


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_bilstm_weights_from_model(model) -> dict[str, np.ndarray]:
    """Return CUDA-basename → float32 ndarray from live bilstm1/bilstm2 modules."""
    def _p(mod, name: str) -> np.ndarray:
        return getattr(mod, name).detach().cpu().numpy().astype(np.float32)

    l1, l2 = model.bilstm1, model.bilstm2
    raw = {
        "w_ih1_f": _p(l1, "weight_ih_l0"),
        "w_hh1_f": _p(l1, "weight_hh_l0"),
        "b_ih1_f": _p(l1, "bias_ih_l0"),
        "b_hh1_f": _p(l1, "bias_hh_l0"),
        "w_ih1_r": _p(l1, "weight_ih_l0_reverse"),
        "w_hh1_r": _p(l1, "weight_hh_l0_reverse"),
        "b_ih1_r": _p(l1, "bias_ih_l0_reverse"),
        "b_hh1_r": _p(l1, "bias_hh_l0_reverse"),
        "w_ih2_f": _p(l2, "weight_ih_l0"),
        "w_hh2_f": _p(l2, "weight_hh_l0"),
        "b_ih2_f": _p(l2, "bias_ih_l0"),
        "b_hh2_f": _p(l2, "bias_hh_l0"),
        "w_ih2_r": _p(l2, "weight_ih_l0_reverse"),
        "w_hh2_r": _p(l2, "weight_hh_l0_reverse"),
        "b_ih2_r": _p(l2, "bias_ih_l0_reverse"),
        "b_hh2_r": _p(l2, "bias_hh_l0_reverse"),
    }
    for basename, (_, shape) in CUDA_WEIGHT_FILES.items():
        if tuple(raw[basename].shape) != shape:
            raise ValueError(
                f"{basename}: shape {raw[basename].shape} != expected {shape}"
            )
    return raw


def weights_for_cpu_ref(cuda_weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map CUDA basenames → parity harness l1_fw_* keys (float64 for CPU ref)."""
    inv = {v: k for k, v in ALIAS_MAP.items()}
    return {inv[k]: v.astype(np.float64) for k, v in cuda_weights.items()}


def write_weight_bins(
    weights: dict[str, np.ndarray],
    out_dir: Path,
    *,
    x_btc: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write CUDA inject bins + aliases + input; return manifest dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}

    for name, arr in weights.items():
        path = out_dir / f"{name}.bin"
        flat = np.ascontiguousarray(arr.astype(np.float32).ravel())
        flat.tofile(path)
        raw = flat.tobytes()
        files[f"{name}.bin"] = {
            "shape": list(arr.shape),
            "dtype": "float32",
            "nbytes": len(raw),
            "sha256": _sha256_bytes(raw),
            "byte_order": "little",
            "layout": "row_major_pytorch_lstm",
        }

    # Friendly aliases (hardlink/copy same content) for Python-side naming
    for alias, cuda_name in ALIAS_MAP.items():
        src = out_dir / f"{cuda_name}.bin"
        dst = out_dir / f"{alias}.bin"
        data = np.fromfile(src, dtype=np.float32)
        data.tofile(dst)
        files[f"{alias}.bin"] = {
            "alias_of": f"{cuda_name}.bin",
            "sha256": _sha256_file(dst),
        }

    if x_btc is None:
        rng = np.random.RandomState(SEED)
        x_btc = rng.randn(BATCH, SEQ, IN_CH).astype(np.float32)
    else:
        x_btc = np.ascontiguousarray(x_btc.astype(np.float32))

    x_path = out_dir / "input_btc.bin"
    x_btc.tofile(x_path)
    files["input_btc.bin"] = {
        "shape": list(x_btc.shape),
        "dtype": "float32",
        "nbytes": x_btc.nbytes,
        "sha256": _sha256_file(x_path),
        "layout": "batch_time_channel",
    }

    # Sample-0 channel-major for CUDA inject
    x0_ct = np.ascontiguousarray(x_btc[0].T.astype(np.float32))  # (IN_CH, SEQ)
    x0_path = out_dir / "input.bin"
    x0_ct.tofile(x0_path)
    files["input.bin"] = {
        "shape": [IN_CH, SEQ],
        "dtype": "float32",
        "nbytes": IN_CH * SEQ * 4,
        "sha256": _sha256_file(x0_path),
        "layout": "channel_major",
        "note": "sample 0; fused_block3.cu inject layout",
    }

    for b in range(x_btc.shape[0]):
        xb = np.ascontiguousarray(x_btc[b].T.astype(np.float32))
        p = out_dir / f"input_b{b}.bin"
        xb.tofile(p)
        files[f"input_b{b}.bin"] = {
            "shape": [IN_CH, SEQ],
            "dtype": "float32",
            "sha256": _sha256_file(p),
            "layout": "channel_major",
        }

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": {
            "seq": SEQ,
            "in_ch": IN_CH,
            "h1": H1,
            "h2": H2,
            "out_size": OUT_SIZE,
            "batch": int(x_btc.shape[0]),
        },
        "seed": SEED,
        "gate_order": ["i", "f", "g", "o"],
        "weight_files": files,
        "state_dict_map": {
            basename: sd_key for basename, (sd_key, _) in CUDA_WEIGHT_FILES.items()
        },
        "alias_map": ALIAS_MAP,
        "cuda_inject": {
            "env": "COLIDE_B3_WEIGHTS=<dir>",
            "argv": "fused_block3 <dir>",
            "required_bins": sorted(
                [f"{k}.bin" for k in CUDA_WEIGHT_FILES] + ["input.bin"]
            ),
            "outputs": ["out_last.bin", "out_full.bin"],
            "out_full_layout": f"time_major [{SEQ}, {OUT_SIZE}] fw|rev",
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def export(out_dir: Path) -> dict[str, Any]:
    import torch
    import yaml
    from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM

    if not CHAMPION_PATH.is_file():
        raise FileNotFoundError(f"Champion missing: {CHAMPION_PATH}")

    champion_md5 = _md5_file(CHAMPION_PATH)
    if champion_md5 != CHAMPION_MD5:
        raise RuntimeError(
            f"Champion MD5 mismatch: got {champion_md5}, expected {CHAMPION_MD5}"
        )

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model = CNNBiLSTM(config)
    state = torch.load(CHAMPION_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Verify module names bilstm1 / bilstm2 exist
    if not hasattr(model, "bilstm1") or not hasattr(model, "bilstm2"):
        raise RuntimeError(
            f"Expected bilstm1/bilstm2; got {[n for n,_ in model.named_children()]}"
        )

    weights = extract_bilstm_weights_from_model(model)
    rng = np.random.RandomState(SEED)
    x_btc = rng.randn(BATCH, SEQ, IN_CH).astype(np.float32)
    manifest = write_weight_bins(weights, out_dir, x_btc=x_btc)
    manifest["champion_path"] = str(CHAMPION_PATH.relative_to(PROJECT_ROOT))
    manifest["champion_md5"] = champion_md5
    manifest["champion_md5_ok"] = True
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Block-3 BiLSTM weight bins")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    manifest = export(out_dir)
    print(f"Exported {len(CUDA_WEIGHT_FILES)} weight tensors + input to {out_dir}")
    print(f"  champion_md5={manifest.get('champion_md5')}")
    print(f"  manifest={out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
