"""
Cross-day DICC session comparator.

Requires two successful run directories (or date labels under a campaign) for
the same GPU label. Rejects mismatched commits, kernel checksums, checkpoints,
protocols, or GPU identity. Reports session means, CVs, CIs, relative spread,
effect sizes (Cohen's d), and Welch t-tests.

Stable DICC results are described as "consistent with WSL2-specific drift"
when session-to-session relative spread is small — this does NOT prove the
local WSL2 machine is the sole source of historical drift; it only supports
that interpretation.

Usage:
    PYTHONPATH=. python scripts/compare_dicc_sessions.py \\
        --gpu v100s --date-a 20260711 --date-b 20260712

    PYTHONPATH=. python scripts/compare_dicc_sessions.py \\
        --run-a benchmarks/results/dicc/core/v100s/20260711_job123 \\
        --run-b benchmarks/results/dicc/core/v100s/20260712_job456
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CompareError(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _find_run(campaign: str, gpu: str, date_label: str) -> Path:
    base = PROJECT_ROOT / "benchmarks" / "results" / "dicc" / campaign / gpu
    if not base.is_dir():
        raise CompareError(f"no runs under {base}")
    matches = sorted(base.glob(f"{date_label}_job*"))
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        raise CompareError(f"no run dir matching {base}/{date_label}_job*")
    # Prefer SUCCESS-marked runs
    ok = [m for m in matches if (m / "SUCCESS").is_file()]
    if len(ok) == 1:
        return ok[0]
    if len(ok) > 1:
        raise CompareError(
            f"multiple SUCCESS runs for {gpu} date={date_label}: {ok}. "
            f"Pass --run-a/--run-b explicitly."
        )
    raise CompareError(
        f"runs exist for {gpu} date={date_label} but none marked SUCCESS: {matches}"
    )


def _require_success(run_dir: Path) -> None:
    if not (run_dir / "SUCCESS").is_file():
        raise CompareError(f"run incomplete (no SUCCESS): {run_dir}")
    status = (run_dir / "exit_status").read_text().strip() if (run_dir / "exit_status").is_file() else "?"
    if status not in ("0", "?"):
        # "?" tolerated only if SUCCESS exists (older fixtures); prefer 0
        if status != "0":
            raise CompareError(f"run exit_status={status} at {run_dir}")


def _manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise CompareError(f"missing manifest.json in {run_dir}")
    return _load_json(path)


def _checksums(run_dir: Path) -> str:
    path = run_dir / "kernel_SHA256SUMS"
    if not path.is_file():
        raise CompareError(f"missing kernel_SHA256SUMS in {run_dir}")
    return path.read_text()


def _welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, 1.0
    t = (m1 - m2) / se
    # Welch–Satterthwaite df
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else float("nan")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:
        # Normal approximation for p
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    return float(t), p


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _extract_cuda_metric(cuda: dict, kernel: str, metric: str) -> dict | None:
    block = cuda.get("kernels", {}).get(kernel) or cuda.get(kernel)
    if not isinstance(block, dict):
        return None
    val = block.get(metric)
    if isinstance(val, dict) and "mean" in val:
        return val
    return None


def _raw_samples(run_dir: Path, kernel: str, metric: str) -> list[float] | None:
    raw_path = run_dir / "raw" / "cuda_kernel_raw.json"
    if not raw_path.is_file():
        return None
    raw = _load_json(raw_path)
    k = raw.get("kernels", {}).get(kernel, {})
    samples = k.get("samples", {}).get(metric)
    if samples is None:
        return None
    return [float(x) for x in samples]


def _pytorch_values(pt: dict, key: str) -> list[float] | None:
    metrics = pt.get("metrics", {})
    block = metrics.get(key) or pt.get(key)
    if not isinstance(block, dict):
        return None
    if "values" in block:
        return [float(x) for x in block["values"]]
    return None


def compare_runs(run_a: Path, run_b: Path) -> dict:
    _require_success(run_a)
    _require_success(run_b)

    man_a = _manifest(run_a)
    man_b = _manifest(run_b)

    date_a = man_a.get("date_label") or run_a.name.split("_job")[0]
    date_b = man_b.get("date_label") or run_b.name.split("_job")[0]
    if date_a == date_b:
        raise CompareError(
            f"dates must differ for a cross-day comparison (both {date_a}). "
            f"Same-day re-runs are not accepted as day-1/day-2 evidence."
        )

    def _field(name: str):
        va, vb = man_a.get(name), man_b.get(name)
        if va != vb:
            raise CompareError(f"manifest field '{name}' mismatch: {va!r} vs {vb!r}")
        return va

    gpu_label = _field("gpu_label")
    _field("gpu_name")
    _field("gpu_compute_capability")
    _field("git_sha")
    _field("checkpoint")
    _field("n_trials_cuda")
    _field("n_trials_pytorch")
    _field("pytorch_inner_forwards")

    # Dirty trees are not comparable provenance
    if man_a.get("git_dirty") or man_b.get("git_dirty"):
        raise CompareError("one or both runs recorded git_dirty=true")

    sum_a = _checksums(run_a)
    sum_b = _checksums(run_b)
    if sum_a != sum_b:
        raise CompareError(
            "kernel SHA256SUMS differ between runs — binaries are not the same build"
        )

    cuda_a = _load_json(run_a / "cuda_kernel_stats.json")
    cuda_b = _load_json(run_b / "cuda_kernel_stats.json")
    pt_a = _load_json(run_a / "pytorch_gpu_stats.json")
    pt_b = _load_json(run_b / "pytorch_gpu_stats.json")

    # Checkpoint hash if present
    cpa = pt_a.get("checkpoint_sha256")
    cpb = pt_b.get("checkpoint_sha256")
    if cpa and cpb and cpa != cpb:
        raise CompareError(f"checkpoint sha256 mismatch: {cpa} vs {cpb}")

    # Protocol equality for pytorch
    proto_keys = ("n_trials", "inner_forwards", "warmup", "model")
    for k in proto_keys:
        va = (pt_a.get("protocol") or {}).get(k)
        vb = (pt_b.get("protocol") or {}).get(k)
        if va != vb:
            raise CompareError(f"pytorch protocol '{k}' mismatch: {va!r} vs {vb!r}")

    comparisons = []

    cuda_metrics = [
        ("fused_block1", "latency_us"),
        ("fused_block2", "latency_us"),
        ("fused_block3", "with_graphs_us"),
        ("fused_block3", "no_graphs_us"),
        ("fused_block3_fp16", "latency_us"),
        ("fused_block3_naive", "latency_us"),
        ("fused_block4", "latency_us"),
        ("fused_pipeline", "b124_chained_us"),
    ]

    for kernel, metric in cuda_metrics:
        sa = _extract_cuda_metric(cuda_a, kernel, metric)
        sb = _extract_cuda_metric(cuda_b, kernel, metric)
        if sa is None or sb is None:
            continue
        raw_a = _raw_samples(run_a, kernel, metric)
        raw_b = _raw_samples(run_b, kernel, metric)
        # Fall back to means-only if raw missing (still report spread of means)
        arr_a = np.array(raw_a if raw_a is not None else [sa["mean"]], dtype=np.float64)
        arr_b = np.array(raw_b if raw_b is not None else [sb["mean"]], dtype=np.float64)
        t, p = _welch_ttest(arr_a, arr_b) if raw_a and raw_b else (float("nan"), float("nan"))
        d = _cohens_d(arr_a, arr_b) if raw_a and raw_b else float("nan")
        mean_a, mean_b = sa["mean"], sb["mean"]
        mid = 0.5 * (mean_a + mean_b)
        rel_spread_pct = abs(mean_a - mean_b) / mid * 100 if mid else float("nan")
        comparisons.append({
            "source": "cuda",
            "name": f"{kernel}.{metric}",
            "day_a": {"date": date_a, "mean": mean_a, "std": sa.get("std"), "cv_pct": sa.get("cv_pct"),
                      "ci95": [sa.get("ci95_low"), sa.get("ci95_high")], "n": sa.get("n")},
            "day_b": {"date": date_b, "mean": mean_b, "std": sb.get("std"), "cv_pct": sb.get("cv_pct"),
                      "ci95": [sb.get("ci95_low"), sb.get("ci95_high")], "n": sb.get("n")},
            "relative_spread_pct": rel_spread_pct,
            "welch_t": t,
            "welch_p": p,
            "cohens_d": d,
        })

    for key in ("full_model_us", "block1_us", "block2_us", "block3_us", "block4_us"):
        sa = (pt_a.get("metrics") or {}).get(key) or pt_a.get(key)
        sb = (pt_b.get("metrics") or {}).get(key) or pt_b.get(key)
        if not isinstance(sa, dict) or not isinstance(sb, dict):
            continue
        raw_a = _pytorch_values(pt_a, key)
        raw_b = _pytorch_values(pt_b, key)
        arr_a = np.array(raw_a if raw_a else [sa["mean"]], dtype=np.float64)
        arr_b = np.array(raw_b if raw_b else [sb["mean"]], dtype=np.float64)
        t, p = _welch_ttest(arr_a, arr_b) if raw_a and raw_b else (float("nan"), float("nan"))
        d = _cohens_d(arr_a, arr_b) if raw_a and raw_b else float("nan")
        mean_a, mean_b = sa["mean"], sb["mean"]
        mid = 0.5 * (mean_a + mean_b)
        rel_spread_pct = abs(mean_a - mean_b) / mid * 100 if mid else float("nan")
        comparisons.append({
            "source": "pytorch",
            "name": key,
            "day_a": {"date": date_a, "mean": mean_a, "std": sa.get("std"), "cv_pct": sa.get("cv_pct"),
                      "ci95": [sa.get("ci95_low"), sa.get("ci95_high")], "n": sa.get("n")},
            "day_b": {"date": date_b, "mean": mean_b, "std": sb.get("std"), "cv_pct": sb.get("cv_pct"),
                      "ci95": [sb.get("ci95_low"), sb.get("ci95_high")], "n": sb.get("n")},
            "relative_spread_pct": rel_spread_pct,
            "welch_t": t,
            "welch_p": p,
            "cohens_d": d,
        })

    spreads = [c["relative_spread_pct"] for c in comparisons if c["relative_spread_pct"] == c["relative_spread_pct"]]
    max_spread = max(spreads) if spreads else float("nan")
    # Heuristic threshold: <5% relative session mean spread across all metrics
    stable = bool(spreads) and max_spread < 5.0

    interpretation = (
        "DICC cross-day session means are stable (max relative spread "
        f"{max_spread:.2f}% < 5%). This is consistent with WSL2-specific drift "
        "on the local RTX 3050 development machine, but does not by itself prove "
        "that WSL2 is the sole cause of historical session-to-session variance."
        if stable else
        "DICC cross-day session means show non-trivial spread "
        f"(max relative spread {max_spread:.2f}%). Do not treat DICC as "
        "session-stable; investigate before updating paper numbers. "
        "This neither confirms nor rules out WSL2-specific drift locally."
    )

    return {
        "gpu_label": gpu_label,
        "date_a": date_a,
        "date_b": date_b,
        "run_a": str(run_a),
        "run_b": str(run_b),
        "git_sha": man_a.get("git_sha"),
        "checkpoint": man_a.get("checkpoint"),
        "stable_cross_day": stable,
        "max_relative_spread_pct": max_spread,
        "interpretation": interpretation,
        "comparisons": comparisons,
        "notes": [
            "Full CUDA-vs-PyTorch pipeline speedup remains invalid until architecture parity.",
            "Block 3 comparisons are the preferred CUDA/PyTorch head-to-head.",
            "Update README/paper claims only from accepted cross-day artifacts.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", default=None, help="GPU label (v100s or a100) when using --date-a/b")
    parser.add_argument("--campaign", default="core")
    parser.add_argument("--date-a", default=None)
    parser.add_argument("--date-b", default=None)
    parser.add_argument("--run-a", default=None, help="Explicit run directory A")
    parser.add_argument("--run-b", default=None, help="Explicit run directory B")
    parser.add_argument("--output", default=None, help="Write comparison JSON here")
    args = parser.parse_args(argv)

    try:
        if args.run_a and args.run_b:
            run_a = Path(args.run_a)
            run_b = Path(args.run_b)
        elif args.gpu and args.date_a and args.date_b:
            run_a = _find_run(args.campaign, args.gpu, args.date_a)
            run_b = _find_run(args.campaign, args.gpu, args.date_b)
        else:
            print("ERROR: provide --run-a/--run-b or --gpu with --date-a/--date-b", file=sys.stderr)
            return 2

        report = compare_runs(run_a, run_b)
    except CompareError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"DICC cross-day comparison: {report['gpu_label']}  {report['date_a']} vs {report['date_b']}")
    print(f"git_sha={report['git_sha']}")
    print(f"run_a={report['run_a']}")
    print(f"run_b={report['run_b']}")
    print("-" * 78)
    for c in report["comparisons"]:
        print(
            f"{c['source']:<8} {c['name']:<36} "
            f"A={c['day_a']['mean']:.2f}  B={c['day_b']['mean']:.2f}  "
            f"spread={c['relative_spread_pct']:.2f}%  "
            f"d={c['cohens_d']:.3f}  p={c['welch_p']:.4g}"
        )
    print("-" * 78)
    print(f"stable_cross_day={report['stable_cross_day']}  "
          f"max_relative_spread_pct={report['max_relative_spread_pct']:.2f}")
    print(report["interpretation"])
    print("=" * 78)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {out}")
    else:
        # Default write next to campaign
        out = Path(report["run_b"]).parent / f"compare_{report['date_a']}_vs_{report['date_b']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
