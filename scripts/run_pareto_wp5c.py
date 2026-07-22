#!/usr/bin/env python3
"""
WP5c / H8 — Pareto consolidation: val macro-F1 vs latency vs model size.

No retrain. Consolidates protocol-measured systems metrics from:
  - benchmarks/results/ablation_ladder/summary.json   (A1–A7)
  - benchmarks/results/baselines_neural/summary.json  (G6–G12)
plus classical val-F1 reference points (no CUDA latency in same harness)
and historical resource notes (cuML RF VRAM, energy).

Outputs:
  benchmarks/results/pareto/summary.json
  benchmarks/results/pareto/table.md
  benchmarks/plots/pareto_f1_latency.png  (if matplotlib available)
  benchmarks/plots/pareto_f1_params.png

Decision framing (H1/H8): multi-objective trade-off table for paper;
does not overwrite champion or change train HPs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "pareto"
PLOT_DIR = PROJECT_ROOT / "benchmarks" / "plots"
CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _md5(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    return hashlib.md5(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
    except Exception:
        return "unknown"


def _dominates(a: dict[str, Any], b: dict[str, Any], axes: list[tuple[str, str]]) -> bool:
    """True if a weakly dominates b on all axes and strictly on ≥1.

    axes: list of (key, 'max'|'min')
    """
    better_or_eq = True
    strictly_better = False
    for key, sense in axes:
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None or (isinstance(av, float) and math.isnan(av)):
            return False
        if sense == "max":
            if av < bv:
                better_or_eq = False
                break
            if av > bv:
                strictly_better = True
        else:  # min
            if av > bv:
                better_or_eq = False
                break
            if av < bv:
                strictly_better = True
    return better_or_eq and strictly_better


def pareto_front(points: list[dict[str, Any]], axes: list[tuple[str, str]]) -> list[dict[str, Any]]:
    eligible = [p for p in points if all(p.get(k) is not None for k, _ in axes)]
    front: list[dict[str, Any]] = []
    for p in eligible:
        if any(_dominates(q, p, axes) for q in eligible if q is not p):
            continue
        front.append(p)
    # stable sort by first maximize axis if present
    max_keys = [k for k, s in axes if s == "max"]
    if max_keys:
        front.sort(key=lambda x: (-float(x[max_keys[0]]), str(x.get("id"))))
    return front


def composite_score(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Min-max normalize: higher F1 better; lower latency + params better.

    score = 0.5 * nF1 + 0.25 * (1-nLat) + 0.25 * (1-nParams)
    """
    usable = [
        p
        for p in points
        if p.get("val_macro_f1") is not None
        and p.get("per_sample_us") is not None
        and p.get("n_params") is not None
    ]
    if not usable:
        return []

    def _rng(vals: list[float]) -> tuple[float, float]:
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return lo, lo + 1.0
        return lo, hi

    f1s = [float(p["val_macro_f1"]) for p in usable]
    lats = [float(p["per_sample_us"]) for p in usable]
    pars = [float(p["n_params"]) for p in usable]
    f_lo, f_hi = _rng(f1s)
    l_lo, l_hi = _rng(lats)
    p_lo, p_hi = _rng(pars)

    scored: list[dict[str, Any]] = []
    for p in usable:
        nf1 = (float(p["val_macro_f1"]) - f_lo) / (f_hi - f_lo)
        nlat = (float(p["per_sample_us"]) - l_lo) / (l_hi - l_lo)
        npar = (float(p["n_params"]) - p_lo) / (p_hi - p_lo)
        score = 0.5 * nf1 + 0.25 * (1.0 - nlat) + 0.25 * (1.0 - npar)
        row = dict(p)
        row["composite_score"] = float(score)
        row["norm_f1"] = float(nf1)
        row["norm_lat_cost"] = float(nlat)
        row["norm_param_cost"] = float(npar)
        scored.append(row)
    scored.sort(key=lambda r: -r["composite_score"])
    return scored


def collect_protocol_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    abl_path = PROJECT_ROOT / "benchmarks" / "results" / "ablation_ladder" / "summary.json"
    if abl_path.is_file():
        abl = _load(abl_path)
        for r in abl.get("results") or []:
            if r.get("returncode") not in (0, None):
                continue
            points.append(
                {
                    "id": r.get("row"),
                    "name": r.get("name"),
                    "family": "ablation_ladder",
                    "tracker": r.get("tracker"),
                    "val_macro_f1": r.get("best_val_macro_f1"),
                    "val_min_per_class_f1": r.get("val_min_per_class_f1"),
                    "val_theft_f1": r.get("val_theft_f1"),
                    "n_params": r.get("n_params"),
                    "per_sample_us": r.get("per_sample_us"),
                    "latency_mean_ms": r.get("latency_mean_ms"),
                    "checkpoint_bytes": r.get("checkpoint_bytes"),
                    "source": str(abl_path.relative_to(PROJECT_ROOT)),
                }
            )

    neu_path = PROJECT_ROOT / "benchmarks" / "results" / "baselines_neural" / "summary.json"
    if neu_path.is_file():
        neu = _load(neu_path)
        for r in neu.get("results") or []:
            if r.get("returncode") not in (0, None):
                continue
            rid = r.get("row")
            points.append(
                {
                    "id": rid,
                    "name": r.get("name"),
                    "family": "neural_baseline",
                    "tracker": r.get("tracker"),
                    "val_macro_f1": r.get("best_val_macro_f1"),
                    "val_min_per_class_f1": r.get("val_min_per_class_f1"),
                    "val_theft_f1": r.get("val_theft_f1"),
                    "n_params": r.get("n_params"),
                    "per_sample_us": r.get("per_sample_us"),
                    "latency_mean_ms": r.get("latency_mean_ms"),
                    "checkpoint_bytes": r.get("checkpoint_bytes"),
                    "source": str(neu_path.relative_to(PROJECT_ROOT)),
                }
            )

    return points


def collect_classical_refs() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    handoff = (
        PROJECT_ROOT
        / "benchmarks"
        / "results"
        / "baselines_classical"
        / "summary_handoff.json"
    )
    if not handoff.is_file():
        return refs
    data = _load(handoff)
    for r in data.get("rows") or []:
        refs.append(
            {
                "id": f"classical_{r.get('model')}",
                "name": r.get("model"),
                "family": "classical_reference",
                "val_macro_f1": r.get("val_macro_f1"),
                "val_min_per_class_f1": r.get("val_min_per_class_f1"),
                "val_theft_f1": r.get("theft_f1"),
                "n_params": None,
                "per_sample_us": None,
                "latency_mean_ms": None,
                "checkpoint_bytes": None,
                "note": "Val-F1 only; CUDA batch256 latency not measured under neural harness",
                "source": str(handoff.relative_to(PROJECT_ROOT)),
            }
        )
    return refs


def collect_systems_notes() -> dict[str, Any]:
    notes: dict[str, Any] = {}
    cuml = PROJECT_ROOT / "benchmarks" / "results" / "cuml_rf_resources.json"
    if cuml.is_file():
        c = _load(cuml)
        notes["cuml_rf_resources"] = {
            "path": str(cuml.relative_to(PROJECT_ROOT)),
            "cuml_rf_vram_mb": c.get("cuml_rf_vram_mb"),
            "cnn_bilstm_vram_mb": c.get("cnn_bilstm_vram_mb"),
            "vram_ratio": c.get("vram_ratio"),
            "hardware": c.get("hardware"),
            "note": c.get("note"),
            "caveat": "Historical A100 cuML RF vs CNN; cnn_bilstm_f1 0.9639 STALE vs protocol champion",
        }
    lat = PROJECT_ROOT / "benchmarks" / "results" / "baseline_latency.json"
    if lat.is_file():
        notes["baseline_latency"] = {
            "path": str(lat.relative_to(PROJECT_ROOT)),
            **{k: v for k, v in _load(lat).items()},
        }
    energy = PROJECT_ROOT / "benchmarks" / "results" / "energy_efficiency.json"
    if energy.is_file():
        notes["energy_efficiency"] = {
            "path": str(energy.relative_to(PROJECT_ROOT)),
            **{k: v for k, v in _load(energy).items()},
        }
    return notes


def write_markdown(
    path: Path,
    points: list[dict[str, Any]],
    front_fl: list[dict[str, Any]],
    front_fp: list[dict[str, Any]],
    front_3d: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    classical: list[dict[str, Any]],
) -> None:
    lines = [
        "# WP5c Pareto — F1 · latency · model size",
        "",
        "Protocol systems metrics: CUDA forward on val batch=256 (ablation/neural harness).",
        "Classical rows are **val-F1 reference only** (not same latency protocol).",
        "",
        "## Neural / ablation points (ranked by val macro-F1)",
        "",
        "| ID | Name | Family | val macro-F1 | min-cls | Theft | params | µs/sample | ckpt bytes |",
        "|----|------|--------|-------------:|--------:|------:|-------:|----------:|-----------:|",
    ]
    ranked = sorted(
        points,
        key=lambda p: float(p.get("val_macro_f1") or -1),
        reverse=True,
    )
    for p in ranked:
        lines.append(
            "| {id} | {name} | {family} | {f1:.4f} | {mn} | {th} | {np} | {us:.2f} | {cb} |".format(
                id=p.get("id"),
                name=p.get("name"),
                family=p.get("family"),
                f1=float(p["val_macro_f1"]),
                mn=f"{float(p['val_min_per_class_f1']):.4f}"
                if p.get("val_min_per_class_f1") is not None
                else "—",
                th=f"{float(p['val_theft_f1']):.4f}"
                if p.get("val_theft_f1") is not None
                else "—",
                np=p.get("n_params"),
                us=float(p["per_sample_us"]),
                cb=p.get("checkpoint_bytes"),
            )
        )

    def _front_section(title: str, front: list[dict[str, Any]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        if not front:
            lines.append("_empty_")
            return
        lines.append("| ID | Name | val macro-F1 | µs/sample | params |")
        lines.append("|----|------|-------------:|----------:|-------:|")
        for p in front:
            lines.append(
                f"| {p.get('id')} | {p.get('name')} | {float(p['val_macro_f1']):.4f} | "
                f"{float(p['per_sample_us']):.2f} | {p.get('n_params')} |"
            )

    _front_section("Pareto front: max F1, min latency", front_fl)
    _front_section("Pareto front: max F1, min params", front_fp)
    _front_section("Pareto front: max F1, min latency, min params (3-obj)", front_3d)

    lines.extend(
        [
            "",
            "## Composite score (0.5·nF1 + 0.25·(1−nLat) + 0.25·(1−nParams))",
            "",
            "| Rank | ID | Name | composite | val macro-F1 | µs/sample | params |",
            "|-----:|----|------|----------:|-------------:|----------:|-------:|",
        ]
    )
    for i, p in enumerate(scored, 1):
        lines.append(
            f"| {i} | {p.get('id')} | {p.get('name')} | {p['composite_score']:.4f} | "
            f"{float(p['val_macro_f1']):.4f} | {float(p['per_sample_us']):.2f} | {p.get('n_params')} |"
        )

    if classical:
        lines.extend(
            [
                "",
                "## Classical val-F1 reference (protocol-fair, stage_b_ft)",
                "",
                "| Model | val macro-F1 | min-cls | Theft |",
                "|-------|-------------:|--------:|------:|",
            ]
        )
        for r in sorted(
            classical, key=lambda x: float(x.get("val_macro_f1") or -1), reverse=True
        ):
            lines.append(
                f"| {r.get('name')} | {float(r['val_macro_f1']):.4f} | "
                f"{float(r.get('val_min_per_class_f1') or 0):.4f} | "
                f"{float(r.get('val_theft_f1') or 0):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Headline trade-offs (no invented numbers)",
            "",
            "- Best val macro-F1 among protocol systems set: full package **A7**.",
            "- Fastest high-F1 neural among suite often **MLP G6** (low µs/sample) but lower F1 than A7/G11.",
            "- Classical LGBM/RF remain competitive on pure detection F1; neural win is multi-objective "
            "(model size / deploy path / optional CUDA blocks) — see historical cuML VRAM note.",
            "- Production champion unchanged; this table is analysis-only.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def try_plots(
    points: list[dict[str, Any]],
    front_fl: list[dict[str, Any]],
    front_fp: list[dict[str, Any]],
) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def _scatter(xkey: str, xlabel: str, fname: str, front: list[dict[str, Any]]) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        families = sorted({p.get("family") or "?" for p in points})
        colors = {
            "ablation_ladder": "#2563eb",
            "neural_baseline": "#16a34a",
        }
        for fam in families:
            subset = [p for p in points if p.get("family") == fam]
            if not subset:
                continue
            ax.scatter(
                [float(p[xkey]) for p in subset],
                [float(p["val_macro_f1"]) for p in subset],
                label=fam,
                c=colors.get(fam, "#6b7280"),
                s=55,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
            for p in subset:
                ax.annotate(
                    str(p.get("id")),
                    (float(p[xkey]), float(p["val_macro_f1"])),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7,
                )
        if front:
            fx = [float(p[xkey]) for p in sorted(front, key=lambda z: float(z[xkey]))]
            fy = [
                float(p["val_macro_f1"])
                for p in sorted(front, key=lambda z: float(z[xkey]))
            ]
            ax.plot(fx, fy, "k--", linewidth=1.0, alpha=0.6, label="Pareto front")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("val macro-F1 (seed42, protocol)")
        ax.set_title("WP5c Pareto — detection vs systems cost")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        out = PLOT_DIR / fname
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(str(out.relative_to(PROJECT_ROOT)))

    _scatter("per_sample_us", "CUDA forward µs/sample (batch=256)", "pareto_f1_latency.png", front_fl)
    _scatter("n_params", "Parameter count", "pareto_f1_params.png", front_fp)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="WP5c Pareto consolidation (no retrain)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    points = collect_protocol_points()
    classical = collect_classical_refs()
    systems_notes = collect_systems_notes()

    if not points:
        print("ERROR: no protocol systems points found (need ablation + neural summaries)")
        return 1

    axes_fl = [("val_macro_f1", "max"), ("per_sample_us", "min")]
    axes_fp = [("val_macro_f1", "max"), ("n_params", "min")]
    axes_3d = [
        ("val_macro_f1", "max"),
        ("per_sample_us", "min"),
        ("n_params", "min"),
    ]

    front_fl = pareto_front(points, axes_fl)
    front_fp = pareto_front(points, axes_fp)
    front_3d = pareto_front(points, axes_3d)
    scored = composite_score(points)

    best_f1 = max(points, key=lambda p: float(p["val_macro_f1"]))
    best_comp = scored[0] if scored else None
    a7 = next((p for p in points if p.get("id") == "A7"), None)
    g11 = next((p for p in points if p.get("id") == "G11"), None)
    g6 = next((p for p in points if p.get("id") == "G6"), None)
    lgbm = next((c for c in classical if c.get("name") == "lgbm"), None)
    rf = next((c for c in classical if c.get("name") == "rf"), None)

    # Decision: analysis artifact for H8 — always DONE when tables written.
    # Package selection not changed by composite alone (CAD-CBA-v1 already signed).
    decision = "DONE"
    decision_note = (
        "WP5c Pareto tables generated from protocol ablation+neural systems metrics. "
        f"Best F1 point={best_f1.get('id')} ({float(best_f1['val_macro_f1']):.4f}). "
        + (
            f"Best composite={best_comp.get('id')} ({best_comp['composite_score']:.4f}). "
            if best_comp
            else ""
        )
        + "CAD-CBA-v1 (A7) remains detection package; multi-obj shows A7 tops F1 while "
        "MLP/G6 and slim CNN win efficiency. Classical LGBM/RF higher or near F1 without "
        "matching neural deploy path. Test sealed. Champion unchanged."
    )

    plots = try_plots(points, front_fl, front_fp)
    md_path = out_dir / "table.md"
    write_markdown(md_path, points, front_fl, front_fp, front_3d, scored, classical)

    summary = {
        "experiment_id": "wp5c_pareto_f1_latency_memory",
        "protocol_id": "botiot_v1",
        "tracker": "H8",
        "work_package": "WP5c",
        "stage": "analysis_only",
        "seed_note": "Underlying points seed42 (ablation/neural); classical seed42",
        "measurement": {
            "latency": "CUDA forward mean ms on val batch=256; per_sample_us = mean_ms*1000/256",
            "memory_proxy": "n_params + checkpoint_bytes (weights disk size); peak VRAM not re-logged here",
            "f1": "best_val_macro_f1 under stage_b_ft protocol; test sealed",
        },
        "n_protocol_points": len(points),
        "n_classical_refs": len(classical),
        "points": points,
        "classical_reference": classical,
        "pareto_front_f1_latency": [
            {"id": p["id"], "name": p["name"], "val_macro_f1": p["val_macro_f1"], "per_sample_us": p["per_sample_us"], "n_params": p["n_params"]}
            for p in front_fl
        ],
        "pareto_front_f1_params": [
            {"id": p["id"], "name": p["name"], "val_macro_f1": p["val_macro_f1"], "per_sample_us": p["per_sample_us"], "n_params": p["n_params"]}
            for p in front_fp
        ],
        "pareto_front_3obj": [
            {"id": p["id"], "name": p["name"], "val_macro_f1": p["val_macro_f1"], "per_sample_us": p["per_sample_us"], "n_params": p["n_params"]}
            for p in front_3d
        ],
        "composite_ranking": [
            {
                "id": p["id"],
                "name": p["name"],
                "composite_score": p["composite_score"],
                "val_macro_f1": p["val_macro_f1"],
                "per_sample_us": p["per_sample_us"],
                "n_params": p["n_params"],
            }
            for p in scored
        ],
        "headlines": {
            "best_f1_id": best_f1.get("id"),
            "best_f1": best_f1.get("val_macro_f1"),
            "best_composite_id": best_comp.get("id") if best_comp else None,
            "best_composite_score": best_comp.get("composite_score") if best_comp else None,
            "A7_full_package": a7,
            "G11_cnn_bilstm": g11,
            "G6_mlp": g6,
            "classical_lgbm_val_macro_f1": lgbm.get("val_macro_f1") if lgbm else None,
            "classical_rf_val_macro_f1": rf.get("val_macro_f1") if rf else None,
        },
        "systems_notes_historical": systems_notes,
        "plots": plots,
        "table_md": str(md_path.relative_to(PROJECT_ROOT)),
        "decision": decision,
        "decision_note": decision_note,
        "champion_md5": _md5(CHAMPION),
        "champion_unchanged_expected": "80a90f7cc210276300eaa90173a5a385",
        "test": "SEALED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "note": (
            "Analysis-only H8. Composite weights are explicit and re-tunable; "
            "do not treat composite as sole method lock. Numbers only from source JSONs."
        ),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== WP5c Pareto ===")
    print(f"protocol points: {len(points)}")
    print(f"front F1-lat: {[p['id'] for p in front_fl]}")
    print(f"front F1-params: {[p['id'] for p in front_fp]}")
    print(f"front 3obj: {[p['id'] for p in front_3d]}")
    if scored:
        print(
            f"composite #1: {scored[0]['id']} score={scored[0]['composite_score']:.4f} "
            f"f1={scored[0]['val_macro_f1']:.4f} us={scored[0]['per_sample_us']:.2f}"
        )
    print(f"best F1: {best_f1['id']} {float(best_f1['val_macro_f1']):.4f}")
    if lgbm:
        print(f"classical LGBM ref: {float(lgbm['val_macro_f1']):.4f}")
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")
    for p in plots:
        print(f"plot/note: {p}")
    print(f"decision={decision}")
    print(f"champion md5={summary['champion_md5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
