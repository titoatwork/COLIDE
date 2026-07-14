#!/usr/bin/env python3
"""
COLIDE Session 7 — Numerical fidelity table (PyTorch live vs exported refs + CUDA binary self-checks).

Produces benchmarks/results/numerical_fidelity.json and prints a manuscript-ready table.

Two complementary checks (do not conflate them):

1. **Real-weight block fidelity** — live PyTorch intermediates from
   model/best_model_botiot_twostage.pth vs model/weights_bin/reference/*.bin
   for the 10 indices in validation_metadata.json. This is the same block
   decomposition the CUDA kernels implement (last BiLSTM timestep; see
   metadata model_version note about attention).

2. **CUDA kernel self-validation** — each standalone binary under
   inference/kernels/ re-runs GPU vs its internal CPU reference at a fixed
   tolerance (often on synthetic rand weights for unit tests). We record
   PASS/FAIL + the disclosed tolerance from source. This is *not* the same
   as (1); together they support the paper's correctness section.

Safety: read-only w.r.t. model checkpoints; only writes numerical_fidelity.json.

Usage:
    PYTHONPATH=. python scripts/numerical_fidelity.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model.cnn_bilstm_v3_attention import CNNBiLSTMAttention as CNNBiLSTM  # noqa: E402

SEED = 42
CHECKPOINT = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
WEIGHT_DIR = PROJECT_ROOT / "model" / "weights_bin"
REF_DIR = WEIGHT_DIR / "reference"
META_PATH = WEIGHT_DIR / "validation_metadata.json"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
KERNEL_DIR = PROJECT_ROOT / "inference" / "kernels"
OUT_PATH = PROJECT_ROOT / "benchmarks" / "results" / "numerical_fidelity.json"

# Disclosed GPU-vs-CPU tolerances from inference/kernels/*.cu validation loops
CUDA_SELFCHECK = [
    {"name": "fused_block1", "tolerance": 1e-3, "dtype": "fp32", "binary": "fused_block1"},
    {"name": "fused_block2", "tolerance": 1e-3, "dtype": "fp32", "binary": "fused_block2"},
    {"name": "fused_block3", "tolerance": 1e-2, "dtype": "fp32", "binary": "fused_block3"},
    {"name": "fused_block3_fp16", "tolerance": 5e-2, "dtype": "fp16", "binary": "fused_block3_fp16"},
    {"name": "fused_block3_naive", "tolerance": 1e-2, "dtype": "fp32", "binary": "fused_block3_naive"},
    {"name": "fused_block4", "tolerance": 1e-3, "dtype": "fp32", "binary": "fused_block4"},
]


def _max_abs_rel(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(b), 1e-12)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    return max_abs, max_rel, mean_abs


@torch.no_grad()
def _block_outputs(model, x: torch.Tensor):
    """Match validate_real_weights.py Test 3 / CUDA block split (no attention)."""
    b1 = model.input_projection(x)
    b1 = b1.view(1, 2, 32)
    b1 = model.relu(model.bn1(model.conv1(b1)))

    b2 = model.relu(model.bn2(model.conv2(b1)))
    b2 = model.pool(b2)

    b3_in = b2.permute(0, 2, 1)
    b3_l1, _ = model.bilstm1(b3_in)
    b3_l2, _ = model.bilstm2(b3_l1)
    b3 = b3_l2[:, -1, :]

    b4 = model.fc2(model.relu(model.fc1(b3)))
    return {
        "block1": b1.cpu().numpy().ravel(),
        "block2": b2.cpu().numpy().ravel(),
        "block3": b3.cpu().numpy().ravel(),
        "block4": b4.cpu().numpy().ravel(),
    }


def real_weight_fidelity(model, X_test, val_indices):
    blocks = ["block1", "block2", "block3", "block4", "full"]
    per_block = {b: {"max_abs": [], "max_rel": [], "mean_abs": []} for b in blocks}
    pred_agree = 0

    for i, idx in enumerate(val_indices):
        x = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)
        live = _block_outputs(model, x)
        with torch.no_grad():
            full_live = model(x).detach().cpu().numpy().ravel()
        live["full"] = full_live

        for bname in ["block1", "block2", "block3", "block4"]:
            ref = np.fromfile(REF_DIR / f"{bname}_out_{i}.bin", dtype=np.float32)
            ma, mr, me = _max_abs_rel(live[bname], ref)
            per_block[bname]["max_abs"].append(ma)
            per_block[bname]["max_rel"].append(mr)
            per_block[bname]["mean_abs"].append(me)

        ref_full = np.fromfile(REF_DIR / f"full_out_{i}.bin", dtype=np.float32)
        ma, mr, me = _max_abs_rel(full_live, ref_full)
        per_block["full"]["max_abs"].append(ma)
        per_block["full"]["max_rel"].append(mr)
        per_block["full"]["mean_abs"].append(me)
        if int(np.argmax(full_live)) == int(np.argmax(ref_full)):
            pred_agree += 1

    summary = {}
    for bname, stats in per_block.items():
        summary[bname] = {
            "n_samples": len(stats["max_abs"]),
            "max_abs_error": float(np.max(stats["max_abs"])),
            "max_rel_error": float(np.max(stats["max_rel"])),
            "mean_of_per_sample_mean_abs": float(np.mean(stats["mean_abs"])),
            "mean_of_per_sample_max_abs": float(np.mean(stats["max_abs"])),
        }
    summary["prediction_agreement"] = {
        "agree": pred_agree,
        "total": len(val_indices),
        "rate": pred_agree / max(len(val_indices), 1),
    }
    return summary


def run_cuda_selfchecks(timeout_s: int = 120):
    results = []
    for spec in CUDA_SELFCHECK:
        binary = KERNEL_DIR / spec["binary"]
        row = {
            "name": spec["name"],
            "tolerance": spec["tolerance"],
            "dtype": spec["dtype"],
            "binary_exists": binary.is_file(),
            "passed": None,
            "stdout_tail": "",
            "error": None,
        }
        if not binary.is_file():
            row["error"] = "binary missing"
            results.append(row)
            continue
        try:
            proc = subprocess.run(
                [str(binary.resolve())],
                cwd=str(KERNEL_DIR),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            row["stdout_tail"] = out[-500:]
            if re.search(r"validation FAILED", out):
                row["passed"] = False
            elif re.search(r"validation PASSED", out):
                row["passed"] = True
            elif "Full Pipeline" in out or "Pipeline chained" in out:
                row["passed"] = None
                row["error"] = "pipeline binary has no numerical self-check (timing only)"
            else:
                row["passed"] = False
                row["error"] = f"no validation marker; returncode={proc.returncode}"
        except subprocess.TimeoutExpired:
            row["error"] = f"timeout after {timeout_s}s"
        except Exception as e:
            row["error"] = str(e)
        results.append(row)
    return results


def main():
    print("=" * 70)
    print("COLIDE numerical fidelity (Session 7)")
    print("=" * 70)

    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    meta = json.loads(META_PATH.read_text())
    val_indices = meta["val_indices"]
    X_test = np.load(DATA_DIR / "X_test.npy")

    model = CNNBiLSTM(config)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"Checkpoint: {CHECKPOINT.name}")
    print(f"Reference samples: {len(val_indices)}")
    print("\n--- Real-weight block fidelity (PyTorch live vs weights_bin/reference) ---")
    rw = real_weight_fidelity(model, X_test, val_indices)
    for bname in ["block1", "block2", "block3", "block4", "full"]:
        s = rw[bname]
        print(
            f"  {bname:8s}  max|Δ|={s['max_abs_error']:.3e}  "
            f"max rel={s['max_rel_error']:.3e}  "
            f"mean max|Δ|={s['mean_of_per_sample_max_abs']:.3e}"
        )
    pa = rw["prediction_agreement"]
    print(f"  pred agreement: {pa['agree']}/{pa['total']} ({pa['rate']:.0%})")

    print("\n--- CUDA binary self-checks (GPU vs in-binary CPU ref) ---")
    cuda_rows = run_cuda_selfchecks()
    for r in cuda_rows:
        status = (
            "PASS" if r["passed"] is True
            else "FAIL" if r["passed"] is False
            else "N/A"
        )
        print(
            f"  {r['name']:22s} tol={r['tolerance']:.0e}  {status}"
            + (f"  ({r['error']})" if r.get("error") else "")
        )

    payload = {
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "seed": SEED,
        "model_version_note": meta.get("model_version"),
        "n_reference_samples": len(val_indices),
        "real_weight_fidelity": rw,
        "cuda_selfcheck": [
            {
                "name": r["name"],
                "tolerance": r["tolerance"],
                "dtype": r["dtype"],
                "binary_exists": r["binary_exists"],
                "passed": r["passed"],
                "error": r["error"],
            }
            for r in cuda_rows
        ],
        "table_notes": [
            "Real-weight max abs/rel errors compare live PyTorch block outputs to "
            "exported reference bins (same decomposition as CUDA blocks).",
            "CUDA self-check uses each binary's internal GPU-vs-CPU test at the "
            "disclosed tolerance; some kernels use synthetic weights for unit test.",
            "Block 3 FP16 uses looser tolerance (5e-2) due to half precision.",
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_PATH}")

    # Manuscript-ready markdown snippet to stdout
    print("\n### Manuscript table (copy source: paper_text_blocks §16)\n")
    print("| Path | Max abs error | Max rel error | Notes |")
    print("|------|---------------|---------------|-------|")
    for bname, label in [
        ("block1", "Block 1 (proj+conv+BN+ReLU)"),
        ("block2", "Block 2 (conv+BN+ReLU+pool)"),
        ("block3", "Block 3 (BiLSTM last step)"),
        ("block4", "Block 4 (dense head)"),
        ("full", "Full logits (export path)"),
    ]:
        s = rw[bname]
        print(
            f"| {label} | {s['max_abs_error']:.2e} | {s['max_rel_error']:.2e} | "
            f"n={s['n_samples']} ref samples |"
        )


if __name__ == "__main__":
    main()
