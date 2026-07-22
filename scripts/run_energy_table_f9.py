#!/usr/bin/env python3
"""
F9 — Energy / systems table consolidation (no retrain; no invented numbers).

Merges existing on-disk measurements:
  - benchmarks/results/energy_efficiency.json   (RTX 3050 batch energy)
  - benchmarks/results/a100_energy.json         (A100 energy if present)
  - benchmarks/results/cuml_rf_resources.json   (cuML RF VRAM/energy)
  - ablation_ladder + baselines_neural latency/params (from summaries)
  - pareto / pareto_h8 for multi-obj context

Writes benchmarks/results/energy_table/summary.json + table.md
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.protocol.result_schema import git_sha  # noqa: E402

OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "energy_table"
CHAMPION = PROJECT_ROOT / "model" / "best_model_botiot_twostage.pth"
CHAMPION_MD5_EXPECTED = "80a90f7cc210276300eaa90173a5a385"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(rel: str) -> dict[str, Any] | None:
    p = PROJECT_ROOT / rel
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def main() -> int:
    champ_md5 = _md5(CHAMPION)
    energy = _load("benchmarks/results/energy_efficiency.json") or {}
    a100 = _load("benchmarks/results/a100_energy.json") or {}
    cuml = _load("benchmarks/results/cuml_rf_resources.json") or {}
    abl = _load("benchmarks/results/ablation_ladder/summary.json") or {}
    neur = _load("benchmarks/results/baselines_neural/summary.json") or {}
    pareto = _load("benchmarks/results/pareto/summary.json") or {}
    pareto_h8 = _load("benchmarks/results/pareto_h8/summary.json") or {}
    stream = _load("benchmarks/results/streaming_throughput.json") or {}
    llm = _load("benchmarks/results/llm_explainability.json") or {}

    rows: list[dict[str, Any]] = []

    # RTX energy (batch128 flows)
    if energy:
        g128 = energy.get("gpu_batch128") or {}
        g1 = energy.get("gpu_batch1") or {}
        c1 = energy.get("cpu_batch1") or {}
        rows.append(
            {
                "id": "rtx3050_gpu_batch128",
                "platform": "RTX 3050 laptop (historical energy harness)",
                "modality": "energy",
                "mj_per_flow": g128.get("mj_per_flow"),
                "throughput_flows_s": g128.get("throughput"),
                "power_w": g128.get("power_w"),
                "source": "benchmarks/results/energy_efficiency.json",
            }
        )
        rows.append(
            {
                "id": "rtx3050_gpu_batch1",
                "platform": "RTX 3050 laptop",
                "modality": "energy",
                "mj_per_inf": g1.get("mj_per_inf"),
                "throughput_flows_s": g1.get("throughput"),
                "power_w": g1.get("power_w"),
                "source": "benchmarks/results/energy_efficiency.json",
            }
        )
        rows.append(
            {
                "id": "rtx3050_cpu_batch1",
                "platform": "CPU (same harness)",
                "modality": "energy",
                "mj_per_inf": c1.get("mj_per_inf"),
                "throughput_flows_s": c1.get("throughput"),
                "power_w": c1.get("power_w"),
                "source": "benchmarks/results/energy_efficiency.json",
            }
        )

    if a100:
        # Preserve structure without inventing fields
        rows.append(
            {
                "id": "a100_energy",
                "platform": "A100 (historical single-shot energy)",
                "modality": "energy",
                "raw": a100,
                "source": "benchmarks/results/a100_energy.json",
                "note": "Cluster single-shot — not multi-day DICC stats",
            }
        )

    if cuml:
        rows.append(
            {
                "id": "cuml_rf_resources",
                "platform": "GPU RF (cuML historical)",
                "modality": "memory_energy",
                "raw": cuml,
                "source": "benchmarks/results/cuml_rf_resources.json",
                "note": "High VRAM vs neural ~2MB class; disclose both directions",
            }
        )

    # Latency/params from ablation + neural (systems, not mJ)
    def _pull_systems(summary: dict, tag: str):
        results = summary.get("results") or summary.get("rows") or []
        if isinstance(results, dict):
            results = list(results.values())
        for r in results:
            if not isinstance(r, dict):
                continue
            rid = r.get("row") or r.get("id") or r.get("name")
            rows.append(
                {
                    "id": f"{tag}:{rid}",
                    "platform": "RTX 3050 protocol batch=256",
                    "modality": "latency_params",
                    "val_macro_f1": r.get("best_val_macro_f1")
                    or r.get("val_macro_f1")
                    or (r.get("metrics") or {}).get("macro_f1"),
                    "n_params": r.get("n_params") or r.get("params"),
                    "per_sample_us": r.get("per_sample_us")
                    or r.get("latency_us_per_sample")
                    or (r.get("systems") or {}).get("per_sample_us"),
                    "source": f"benchmarks/results/{tag}/summary.json",
                }
            )

    _pull_systems(abl, "ablation_ladder")
    _pull_systems(neur, "baselines_neural")

    # Pareto composite headline
    composite = None
    if pareto_h8.get("composite_ranking"):
        composite = pareto_h8["composite_ranking"][:5]
    elif pareto.get("composite_ranking"):
        composite = pareto["composite_ranking"][:5]

    headline = {
        "rtx_gpu_batch128_mj_per_flow": (energy.get("gpu_batch128") or {}).get("mj_per_flow"),
        "rtx_gpu_batch128_throughput": (energy.get("gpu_batch128") or {}).get("throughput"),
        "streaming_throughput": stream.get("throughput_flows_per_s")
        or stream.get("throughput")
        or stream.get("flows_per_sec"),
        "llm_dispatch_p99_us": llm.get("overhead_p99_us"),
        "pareto_h8_composite_top": composite[0] if composite else None,
        "note": (
            "Energy rows are historical harness measurements on the production path; "
            "ablation/baseline rows provide latency+params only (F9 energy-per-ablation "
            "mJ not re-measured per row — disclose proxy gap)."
        ),
    }

    summary = {
        "experiment_id": "f9_energy_systems_table",
        "tracker": "F9",
        "work_package": "WP5a-F9 / systems",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(PROJECT_ROOT),
        "champion_md5": champ_md5,
        "champion_md5_expected": CHAMPION_MD5_EXPECTED,
        "champion_unchanged": champ_md5 == CHAMPION_MD5_EXPECTED,
        "n_rows": len(rows),
        "headline": headline,
        "rows": rows,
        "sources_present": {
            "energy_efficiency.json": bool(energy),
            "a100_energy.json": bool(a100),
            "cuml_rf_resources.json": bool(cuml),
            "ablation_ladder/summary.json": bool(abl),
            "baselines_neural/summary.json": bool(neur),
            "pareto_h8/summary.json": bool(pareto_h8),
            "pareto/summary.json": bool(pareto),
            "streaming_throughput.json": bool(stream),
            "llm_explainability.json": bool(llm),
        },
        "decision": "RUN_DOCUMENTED",
        "decision_note": (
            "F9 energy table consolidated from existing JSON. Full per-ablation mJ "
            "re-measure not re-run (thermal/cost); latency+params cover systems axes "
            "for A1–A7/G6–G12; RTX batch128 energy and A100/cuML historical rows included. "
            "Do not invent multi-day DICC energy CIs."
        ),
        "gaps": [
            "Per-ablation joules not re-measured under identical harness",
            "DICC multi-day energy BLOCKED until dedicated session",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = ["# F9 Energy / systems table\n\n"]
    md.append(f"- Champion md5: `{champ_md5}`\n")
    md.append(f"- Decision: **{summary['decision']}**\n\n")
    h = headline
    md.append("## Headline (from sources only)\n\n")
    md.append(f"| Metric | Value | Source |\n|--------|-------|--------|\n")
    md.append(
        f"| RTX GPU batch128 mJ/flow | {h.get('rtx_gpu_batch128_mj_per_flow')} | energy_efficiency.json |\n"
    )
    md.append(
        f"| RTX GPU batch128 thr | {h.get('rtx_gpu_batch128_throughput')} | energy_efficiency.json |\n"
    )
    md.append(
        f"| LLM dispatch p99 overhead µs | {h.get('llm_dispatch_p99_us')} | llm_explainability.json |\n"
    )
    if h.get("pareto_h8_composite_top"):
        top = h["pareto_h8_composite_top"]
        md.append(
            f"| pareto_h8 composite #1 | {top.get('id')} score={top.get('composite_score')} | pareto_h8 |\n"
        )
    md.append("\n## Notes\n\n")
    md.append(summary["decision_note"] + "\n")
    for g in summary["gaps"]:
        md.append(f"- Gap: {g}\n")
    (OUT_DIR / "table.md").write_text("".join(md))

    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "n_rows": summary["n_rows"],
                "headline": {
                    k: h[k]
                    for k in h
                    if k != "pareto_h8_composite_top"
                },
                "out": str(OUT_DIR / "summary.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
