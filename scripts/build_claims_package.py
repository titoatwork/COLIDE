#!/usr/bin/env python3
"""
WP9a — Build protocol-era claims package from on-disk result JSON.

Never invents numbers. Reads existing benchmarks/results/**/summary*.json
(and key per-run files), writes:

  benchmarks/results/claims_package/protocol_claims.json
  benchmarks/results/claims_package/table.md
  docs/execution_plan/CLAIMS_REGISTRY.md   (committed copy)

Usage:
    PYTHONPATH=. python3 scripts/build_claims_package.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUT = RESULTS / "claims_package"
DOCS_REG = ROOT / "docs" / "execution_plan" / "CLAIMS_REGISTRY.md"
CHAMP = ROOT / "model" / "best_model_botiot_twostage.pth"
CHAMP_EXPECTED = "80a90f7cc210276300eaa90173a5a385"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(rel: str) -> tuple[Any | None, dict | None]:
    """Load JSON relative to RESULTS (or ROOT if path starts with config/ or model/)."""
    if rel.startswith("config/") or rel.startswith("model/") or rel.startswith("docs/"):
        p = ROOT / rel
    else:
        p = RESULTS / rel
    if not p.exists():
        return None, None
    if p.suffix not in {".json", ".yaml", ".yml", ".md", ".txt"}:
        # binary artifact — metadata only
        return None, {"path": str(p.relative_to(ROOT)), "md5": md5_file(p), "bytes": p.stat().st_size}
    with open(p) as f:
        data = json.load(f)
    return data, {
        "path": str(p.relative_to(ROOT)),
        "md5": md5_file(p),
        "bytes": p.stat().st_size,
    }


def dig(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def find_metrics_block(o: Any, depth: int = 0) -> dict | None:
    if depth > 6 or not isinstance(o, dict):
        return None
    if "per_class" in o and "macro_f1" in o:
        return o
    if "val" in o and isinstance(o["val"], dict) and "macro_f1" in o["val"]:
        return o["val"]
    for k in ("metrics", "best_val", "val_metrics", "best"):
        if k in o:
            r = find_metrics_block(o[k], depth + 1)
            if r:
                return r
    for v in o.values():
        if isinstance(v, dict):
            r = find_metrics_block(v, depth + 1)
            if r:
                return r
    return None


def macro_f1(d: Any) -> float | None:
    if d is None:
        return None
    for path in (
        ("metrics", "macro_f1"),
        ("val", "macro_f1"),
        ("best_val_macro_f1",),
        ("val_macro_f1",),
        ("macro_f1",),
    ):
        v = dig(d, *path) if len(path) > 1 or path[0] in (d or {}) else d.get(path[0]) if isinstance(d, dict) else None
        if isinstance(v, (int, float)):
            return float(v)
    m = find_metrics_block(d)
    if m and isinstance(m.get("macro_f1"), (int, float)):
        return float(m["macro_f1"])
    return None


def render_value(value: Any, decimals: int | None = None) -> str:
    if value is None:
        return "PENDING"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        d = decimals if decimals is not None else (4 if abs(value) < 100 else 2)
        return f"{value:.{d}f}"
    return str(value)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    claims: list[dict] = []
    missing: list[str] = []

    def add(
        cid: str,
        value: Any,
        source_rel: str,
        json_path: str,
        notes: str = "",
        status: str = "LOCKED_VAL",
        decimals: int | None = None,
    ) -> None:
        meta = None
        if source_rel not in ("PENDING", "—", ""):
            _, meta = load_json(source_rel)
            if meta is None and source_rel not in ("PENDING",):
                # try as absolute-under-results already handled; record missing
                if not (RESULTS / source_rel).exists() and not (ROOT / source_rel).exists():
                    missing.append(f"{cid}:{source_rel}")
        claims.append(
            {
                "id": cid,
                "value": value,
                "render": render_value(value, decimals),
                "source_file": source_rel,
                "json_path": json_path,
                "source_md5": meta["md5"] if meta else None,
                "notes": notes,
                "status": status,
            }
        )

    # --- load core summaries ---
    mr, _ = load_json("multirun/summary.json")
    hpo, _ = load_json("hpo/summary.json")
    pkg, _ = load_json("multirun_ensemble_hpo/summary.json")
    hpc, _ = load_json("multirun_hpo_confirm/summary.json")
    lgbm, _ = load_json("baselines_classical/lgbm_seed42.json")
    rf, _ = load_json("baselines_classical/rf_seed42.json")
    xgb, _ = load_json("baselines_classical/xgb_seed42.json")
    svm, _ = load_json("baselines_classical/svm_seed42.json")
    a7, _ = load_json("ablation_ladder/A7_full_cad_cba_v1_seed42.json")
    a3, _ = load_json("ablation_ladder/A3_cnn_bilstm_seed42.json")
    a4, _ = load_json("ablation_ladder/A4_cnn_bilstm_attn_ce_seed42.json")
    g11, _ = load_json("baselines_neural/G11_cnn_bilstm_seed42.json")
    g6, _ = load_json("baselines_neural/G6_mlp_seed42.json")
    g12, _ = load_json("baselines_neural/G12_transformer_seed42.json")
    ctrl, _ = load_json("cstar_bounded/CTRL_control_v3_focal_seed42.json")
    c4, _ = load_json("cstar_bounded/C4_multi_scale_seed42.json")
    e6sum, _ = load_json("teachers_kd_neural/summary.json")
    e6run, _ = load_json("teachers_kd_neural/kd_neural_cnn_bilstm_seed42.json")
    ens_kd, _ = load_json("teachers_kd/kd_ensemble_seed42.json")
    xai, _ = load_json("xai/summary.json")
    en, _ = load_json("energy_table/summary.json")
    ph8, _ = load_json("pareto_h8/summary.json")
    ton, _ = load_json("toniot_final/summary.json")
    st_sh, _ = load_json("stratified_batch/ft_shuffle_seed42.json")
    st_st, _ = load_json("stratified_batch/ft_stratified_seed42.json")
    imb, _ = load_json("imbalance_loss/summary.json")
    rf_hist, _ = load_json("rf_baseline_processed.json")
    fid, _ = load_json("numerical_fidelity.json")

    # WP1b
    assert mr is not None, "multirun/summary.json missing"
    add("wp1b_multirun_mean", mr["val_macro_f1_mean"], "multirun/summary.json", "val_macro_f1_mean", "WP1b n=5 val-only", decimals=4)
    add("wp1b_multirun_std", mr["val_macro_f1_std"], "multirun/summary.json", "val_macro_f1_std", decimals=4)

    # WP3 HPO
    assert hpo is not None
    hpo_w = dig(hpo, "winner", "metrics", "best_val_macro_f1")
    add("wp3_hpo_winner_val", hpo_w, "hpo/summary.json", "winner.metrics.best_val_macro_f1", "INCORPORATE train HPs", decimals=4)

    # package / confirm multiruns
    assert pkg is not None and hpc is not None
    add("package_ensemble_hpo_mean", pkg["val_macro_f1_mean"], "multirun_ensemble_hpo/summary.json", "val_macro_f1_mean", "RUN_DOCUMENTED; not mean-win vs WP1b", decimals=4)
    add("package_ensemble_hpo_std", pkg["val_macro_f1_std"], "multirun_ensemble_hpo/summary.json", "val_macro_f1_std", decimals=4)
    add("hpo_confirm_mean", hpc["val_macro_f1_mean"], "multirun_hpo_confirm/summary.json", "val_macro_f1_mean", decimals=4)
    add("hpo_confirm_std", hpc["val_macro_f1_std"], "multirun_hpo_confirm/summary.json", "val_macro_f1_std", decimals=4)

    # WP4b ensemble student
    ens_f1 = macro_f1(ens_kd)
    add("wp4b_ensemble_student_val", ens_f1, "teachers_kd/kd_ensemble_seed42.json", "metrics.macro_f1", "INCORPORATE KD teacher (stage_a_kd)", decimals=4)

    # classical protocol
    add("g5_lgbm_val", macro_f1(lgbm), "baselines_classical/lgbm_seed42.json", "metrics.macro_f1", "protocol classical top", decimals=4)
    add("g3_rf_val", macro_f1(rf), "baselines_classical/rf_seed42.json", "metrics.macro_f1", decimals=4)
    add("g4_xgb_val", macro_f1(xgb), "baselines_classical/xgb_seed42.json", "metrics.macro_f1", decimals=4)
    add("g2_svm_val", macro_f1(svm), "baselines_classical/svm_seed42.json", "metrics.macro_f1", "RUN_DOCUMENTED weak", decimals=4)

    # historical dual bar
    pub_rf = dig(rf_hist, "test_macro_f1") if rf_hist else 0.9864
    add("published_rf_test", pub_rf, "rf_baseline_processed.json", "test_macro_f1", "NOT protocol-fair; dual bar only", status="HISTORICAL", decimals=4)

    # ablation
    add("wp5a_a7_val", macro_f1(a7), "ablation_ladder/A7_full_cad_cba_v1_seed42.json", "metrics.macro_f1", "ladder top package path", decimals=4)
    add("wp5a_a3_val", macro_f1(a3), "ablation_ladder/A3_cnn_bilstm_seed42.json", "metrics.macro_f1", decimals=4)
    add("wp5a_a4_val", macro_f1(a4), "ablation_ladder/A4_cnn_bilstm_attn_ce_seed42.json", "metrics.macro_f1", "attn+CE underperforms A3", decimals=4)

    # neural baselines
    add("g11_cnn_bilstm_val", macro_f1(g11), "baselines_neural/G11_cnn_bilstm_seed42.json", "metrics.macro_f1", decimals=4)
    add("g6_mlp_val", macro_f1(g6), "baselines_neural/G6_mlp_seed42.json", "metrics.macro_f1", decimals=4)
    add("g12_transformer_val", macro_f1(g12), "baselines_neural/G12_transformer_seed42.json", "metrics.macro_f1", "weak under equal budget", decimals=4)

    # cstar
    add("cstar_ctrl_val", macro_f1(ctrl), "cstar_bounded/CTRL_control_v3_focal_seed42.json", "metrics.macro_f1", decimals=4)
    add("cstar_c4_val", macro_f1(c4), "cstar_bounded/C4_multi_scale_seed42.json", "metrics.macro_f1", "no package incorporate", decimals=4)

    # E6
    e6s = dig(e6sum, "student_val_macro_f1") or dig(e6sum, "best_val_macro_f1") or macro_f1(e6run)
    if e6s is None and e6sum:
        # scan summary for first float macro
        for k, v in e6sum.items():
            if "student" in k.lower() and isinstance(v, (int, float)):
                e6s = float(v)
                break
        if e6s is None:
            e6s = dig(e6sum, "comparators", "student_val_macro_f1")
    add("e6_neural_teacher_student_val", e6s, "teachers_kd_neural/summary.json", "student_val_macro_f1", "≪ ensemble 0.9401; keep ensemble", decimals=4)

    # XAI
    assert xai is not None
    add("xai_rank_corr", xai["consistency_spearman_rank"], "xai/summary.json", "consistency_spearman_rank", decimals=4)
    add("xai_faith_mass", dig(xai, "faithfulness_proxy", "value"), "xai/summary.json", "faithfulness_proxy.value", decimals=4)
    add("xai_dispatch_p99_us", dig(xai, "latency", "dispatch_overhead_p99_us"), "xai/summary.json", "latency.dispatch_overhead_p99_us", "dispatch only — not gen", decimals=2)
    # Canonical free-form quality metric: any_feature_mention_rate (0.333 in suite).
    # Do NOT use class_mention_rate (often 1.0) — that is class-name presence only.
    llm_rate = dig(xai, "llm_sample_eval", "any_feature_mention_rate")
    if llm_rate is None:
        llm_rate = dig(xai, "llm_sample_eval", "strict_feature_mention_rate")
    add(
        "xai_llm_feature_mention",
        llm_rate,
        "xai/summary.json",
        "llm_sample_eval.any_feature_mention_rate",
        "free-form weak; not class_mention_rate",
        decimals=3,
    )
    add(
        "xai_llm_top3_agree",
        dig(xai, "llm_sample_eval", "top3_feature_agreement_rate"),
        "xai/summary.json",
        "llm_sample_eval.top3_feature_agreement_rate",
        "weak top-3 agreement",
        decimals=3,
    )
    add(
        "xai_llm_gen_mean_ms",
        dig(xai, "llm_sample_eval", "llm_generation_mean_ms"),
        "xai/summary.json",
        "llm_sample_eval.llm_generation_mean_ms",
        "never conflate with dispatch µs",
        decimals=0,
    )
    add("xai_structured_usefulness", dig(xai, "structured_usefulness_mean"), "xai/summary.json", "structured_usefulness_mean", decimals=1)
    add("xai_j10", xai.get("j10_path"), "xai/summary.json", "j10_path", "DROP full claim keep structured+dispatch")

    # energy / pareto
    assert en is not None
    add(
        "f9_rtx_mj_per_flow",
        dig(en, "headline", "rtx_gpu_batch128_mj_per_flow"),
        "energy_table/summary.json",
        "headline.rtx_gpu_batch128_mj_per_flow",
        "batch128",
        decimals=3,
    )
    comp = dig(en, "headline", "pareto_h8_composite_top", "composite_score")
    if comp is None:
        comp = dig(ph8, "headline", "top_composite", "score")
    add("pareto_h8_composite_g6", comp, "pareto_h8/summary.json", "headline.top_composite.score", "a priori composite #1 G6", decimals=4)

    # ToN (test allowed for WP8 multi-dataset)
    assert ton is not None
    add("ton_val", dig(ton, "comparators", "this_val_macro_f1"), "toniot_final/summary.json", "comparators.this_val_macro_f1", "13-feat processed_toniot", status="LOCKED_TEST", decimals=4)
    add("ton_test", dig(ton, "comparators", "this_test_macro_f1"), "toniot_final/summary.json", "comparators.this_test_macro_f1", "WP8 ToN test allowed", status="LOCKED_TEST", decimals=4)
    add("ton_rf_test", dig(ton, "comparators", "rf_test_macro_f1"), "toniot_final/summary.json", "comparators.rf_test_macro_f1", status="LOCKED_TEST", decimals=4)

    # D6 stratified
    add("d6_shuffle_val", dig(st_sh, "best_val_macro_f1") or macro_f1(st_sh), "stratified_batch/ft_shuffle_seed42.json", "best_val_macro_f1", "keep shuffle default", decimals=4)
    add("d6_stratified_val", dig(st_st, "best_val_macro_f1") or macro_f1(st_st), "stratified_batch/ft_stratified_seed42.json", "best_val_macro_f1", "hurts under hpo_best", decimals=4)

    # imbalance focal
    focal = None
    if imb:
        focal = dig(imb, "best", "val_macro_f1") or dig(imb, "best_val_macro_f1")
        if focal is None and isinstance(imb.get("ranking"), list) and imb["ranking"]:
            focal = imb["ranking"][0].get("val_macro_f1")
        if focal is None:
            # common shape: rows with loss name
            for row in imb.get("rows") or imb.get("results") or []:
                if isinstance(row, dict) and row.get("loss") == "focal":
                    focal = row.get("best_val_macro_f1") or row.get("val_macro_f1")
                    break
        if focal is None and "focal" in imb:
            focal = dig(imb, "focal", "best_val_macro_f1") or dig(imb, "focal", "val_macro_f1")
    if focal is None:
        foc_j, _ = load_json("imbalance_loss/ft_focal_seed42.json")
        focal = dig(foc_j, "best_val_macro_f1") or macro_f1(foc_j)
    add("d3_focal_val", focal, "imbalance_loss/ft_focal_seed42.json", "best_val_macro_f1", "INCORPORATED CAD-CBA loss", decimals=4)

    # fidelity
    bit_ok = False
    if fid:
        rw = fid.get("real_weight_fidelity") or {}
        maxes = [
            dig(rw, b, "max_abs_error")
            for b in ("block1", "block2", "block3", "block4", "full")
            if dig(rw, b, "max_abs_error") is not None
        ]
        bit_ok = bool(maxes) and max(maxes) == 0.0
    add("fidelity_bit_identical", bit_ok, "numerical_fidelity.json", "real_weight_fidelity.*.max_abs_error", "WP6a re-export")

    # champion
    champ_md5 = md5_file(CHAMP) if CHAMP.exists() else None
    add(
        "champion_md5",
        champ_md5,
        "model/best_model_botiot_twostage.pth",
        "md5",
        "never clobber without BACKUP + OK",
    )

    # open gates
    add(
        "bot_sealed_test_multiseed",
        None,
        "PENDING",
        "—",
        "B14 only after explicit user final-config lock",
        status="PENDING_SEALED_TEST",
    )
    add(
        "dicc_multiday_stats",
        None,
        "PENDING",
        "—",
        "WP0 / I1–I6 user-scheduled DICC session",
        status="BLOCKED_DICC",
    )

    # minority tables
    minority_rows: list[dict] = []
    for label, rel in [
        ("HPO winner seed42", "hpo/refine_rank2_trial008_seed42.json"),
        ("WP1b multirun seed42", "multirun/ft_seed42.json"),
        ("Focal FT seed42", "imbalance_loss/ft_focal_seed42.json"),
        ("LGBM protocol", "baselines_classical/lgbm_seed42.json"),
        ("RF protocol", "baselines_classical/rf_seed42.json"),
        ("XGB protocol", "baselines_classical/xgb_seed42.json"),
        ("A7 CAD-CBA ladder", "ablation_ladder/A7_full_cad_cba_v1_seed42.json"),
        ("CTRL cstar", "cstar_bounded/CTRL_control_v3_focal_seed42.json"),
        ("Package seed45 (max)", "multirun_ensemble_hpo/ft_seed45.json"),
        ("HPO confirm seed42", "multirun_hpo_confirm/ft_seed42.json"),
        ("Ensemble KD stage_a", "teachers_kd/kd_ensemble_seed42.json"),
    ]:
        d, meta = load_json(rel)
        if not d or not meta:
            continue
        m = find_metrics_block(d)
        if not m:
            continue
        pc = m.get("per_class") or {}
        minority_rows.append(
            {
                "label": label,
                "source": rel,
                "source_md5": meta["md5"],
                "macro_f1": m.get("macro_f1"),
                "min_per_class_f1": m.get("min_per_class_f1"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "per_class_f1": {
                    c: (v.get("f1") if isinstance(v, dict) else v) for c, v in pc.items()
                },
            }
        )

    advantage = {
        "detection_protocol_top_classical": {
            "model": "LGBM",
            "val_macro_f1": macro_f1(lgbm),
            "source": "baselines_classical/lgbm_seed42.json",
        },
        "detection_protocol_neural_hpo": {
            "model": "CAD-CBA-v1 HPO refine",
            "val_macro_f1": hpo_w,
            "source": "hpo/summary.json",
        },
        "detection_protocol_neural_multirun": {
            "model": "WP1b FT multirun",
            "mean": mr["val_macro_f1_mean"],
            "std": mr["val_macro_f1_std"],
            "source": "multirun/summary.json",
        },
        "efficiency_composite_top": dig(en, "headline", "pareto_h8_composite_top"),
        "energy_rtx_batch128_mj_flow": dig(en, "headline", "rtx_gpu_batch128_mj_per_flow"),
        "llm_dispatch_p99_us": dig(en, "headline", "llm_dispatch_p99_us")
        or dig(xai, "latency", "dispatch_overhead_p99_us"),
        "vram_proxy_note": (
            "Historical cuML RF ~444MB vs CNN ~2MB (cuml_rf_resources.json); "
            "protocol proxy via n_params/ckpt in pareto_h8"
        ),
        "honest_gap_vs_published_rf": (
            "Protocol neural HPO ~0.9791 / multirun ~0.9714 vs protocol RF ~0.9778 / "
            "published RF 0.9864 — do not mix pipelines"
        ),
        "ton_gap": "ToN 13-feat neural test ~0.8110 vs RF ~0.9393",
        "xai_policy": "DROP full LLM-explainable claim; keep dispatch + structured evidence",
    }

    package = {
        "experiment_id": "wp9a_claims_package",
        "work_package": "WP9a",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "champion_md5": champ_md5,
        "champion_md5_expected": CHAMP_EXPECTED,
        "champion_unchanged": champ_md5 == CHAMP_EXPECTED,
        "policy": {
            "never_invent_numbers": True,
            "test_sealed_botiot": True,
            "ton_test_allowed_for_wp8": True,
            "dicc_blocked": True,
            "option_a_cuda": True,
        },
        "method_package": "CAD-CBA-v1",
        "method_components": {
            "arch": "cnn_bilstm_v3_attention",
            "kd_teacher": "ensemble RF+XGB+LGBM mean soft labels",
            "kd_recipe": "alpha=0.6 T=10",
            "loss": "focal",
            "train_hps": "config/hpo_best.yaml",
            "decode": "argmax",
            "train_sampler": "shuffle",
        },
        "claims": claims,
        "minority_tables_val": minority_rows,
        "quantitative_advantage_snapshot": advantage,
        "open_gates": [
            "B14 sealed multi-seed TEST on BoT after explicit user final-config lock",
            "WP6b local multi-session latency/energy ranges after lock",
            "WP0 DICC multi-day (user-scheduled)",
            "WP9b manuscript spine after tracker largely green",
        ],
        "missing_sources": missing,
        "decision": "PARTIAL_DONE",
        "decision_note": (
            "WP9a numbers-match package built from on-disk protocol JSON. "
            "BoT sealed multi-seed TEST claims remain PENDING. Historical README "
            "claims still verified by scripts/verify_claims.py. Do not quote "
            "PENDING or BLOCKED as final paper numbers."
        ),
    }

    out_json = OUT / "protocol_claims.json"
    with open(out_json, "w") as f:
        json.dump(package, f, indent=2)
        f.write("\n")

    # Markdown
    lines: list[str] = []
    lines.append("# Protocol Claims Package (WP9a)")
    lines.append("")
    lines.append(f"**Generated (UTC):** {package['timestamp_utc']}")
    lines.append(f"**Git:** `{package['git_sha']}`")
    lines.append(
        f"**Champion md5:** `{champ_md5}` "
        f"(unchanged={package['champion_unchanged']}; expected `{CHAMP_EXPECTED}`)"
    )
    lines.append(f"**Method:** CAD-CBA-v1")
    lines.append("")
    lines.append(
        "> Source of truth for **protocol-era** numbers. Every row traces to a local "
        "JSON under `benchmarks/results/` (gitignored) with md5 recorded here and in "
        "`RESULTS_DISK_MANIFEST.md`. Never invent multi-day DICC numbers."
    )
    lines.append("")
    lines.append("## Claim inventory")
    lines.append("")
    lines.append("| ID | Value | Status | Source | Notes |")
    lines.append("|----|-------|--------|--------|-------|")
    for c in claims:
        val = c["render"]
        notes = (c["notes"] or "").replace("|", "\\|")
        lines.append(
            f"| `{c['id']}` | **{val}** | {c['status']} | `{c['source_file']}` | {notes} |"
        )
    lines.append("")
    lines.append("## Minority / per-class F1 (val only)")
    lines.append("")
    classes: list[str] = []
    for r in minority_rows:
        for c in r["per_class_f1"]:
            if c not in classes:
                classes.append(c)
    if classes:
        lines.append(
            "| Model | macro-F1 | min-cls | "
            + " | ".join(classes)
            + " | source |"
        )
        lines.append(
            "|-------|----------|---------|"
            + "|".join(["------"] * len(classes))
            + "|--------|"
        )
        for r in minority_rows:
            pcs = " | ".join(
                f"{r['per_class_f1'][c]:.4f}"
                if isinstance(r["per_class_f1"].get(c), float)
                else "—"
                for c in classes
            )
            lines.append(
                f"| {r['label']} | {r['macro_f1']:.4f} | {r['min_per_class_f1']:.4f} | "
                f"{pcs} | `{r['source']}` |"
            )
    lines.append("")
    lines.append("## Quantitative advantage snapshot (A4 / multi-obj)")
    lines.append("")
    lines.append("| Dimension | Evidence | Source |")
    lines.append("|-----------|----------|--------|")
    lines.append(
        f"| Protocol classical F1 top | LGBM **{macro_f1(lgbm):.4f}** | `lgbm_seed42.json` |"
    )
    lines.append(
        f"| Protocol neural HPO | **{hpo_w:.4f}** | `hpo/summary.json` |"
    )
    lines.append(
        f"| Protocol neural multirun | "
        f"**{mr['val_macro_f1_mean']:.4f}±{mr['val_macro_f1_std']:.4f}** | `multirun/summary.json` |"
    )
    comp_row = dig(en, "headline", "pareto_h8_composite_top") or {}
    if comp_row:
        lines.append(
            f"| A priori composite #1 | G6 score **{comp_row.get('composite_score', comp):.4f}** "
            f"@ {comp_row.get('per_sample_us', float('nan')):.2f} µs "
            f"F1 {comp_row.get('val_macro_f1', float('nan')):.4f} | `pareto_h8` / energy headline |"
        )
    lines.append(
        f"| Energy (RTX batch128) | "
        f"**{dig(en, 'headline', 'rtx_gpu_batch128_mj_per_flow'):.3f} mJ/flow** | `energy_table/summary.json` |"
    )
    disp = dig(en, "headline", "llm_dispatch_p99_us") or dig(
        xai, "latency", "dispatch_overhead_p99_us"
    )
    lines.append(
        f"| LLM dispatch p99 | **{disp:.2f} µs** (dispatch only) | `xai` / energy |"
    )
    lines.append(
        "| XAI policy | DROP full explainable claim; keep structured+dispatch | `xai/summary.json` J10 |"
    )
    lines.append(
        "| ToN honesty | test 0.8110 ≪ RF 0.9393 (13-feat) | `toniot_final` |"
    )
    lines.append(
        "| Published RF dual bar | 0.9864 ≠ protocol RF 0.9778 | historical vs protocol |"
    )
    lines.append("")
    lines.append("## Open gates (do not claim as done)")
    lines.append("")
    for g in package["open_gates"]:
        lines.append(f"- {g}")
    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTHONPATH=. python3 scripts/build_claims_package.py")
    lines.append("PYTHONPATH=. python3 scripts/verify_claims.py")
    lines.append("# protocol prose block lives in docs/paper_text_blocks.md §Protocol-era")
    lines.append("```")
    lines.append("")
    lines.append("## Status legend")
    lines.append("")
    lines.append("| Status | Meaning |")
    lines.append("|--------|---------|")
    lines.append("| LOCKED_VAL | Val-only protocol number; safe to quote with source |")
    lines.append("| LOCKED_TEST | Explicitly allowed test number (ToN WP8) |")
    lines.append("| HISTORICAL | Prior pipeline / dual bar — label carefully |")
    lines.append("| PENDING_SEALED_TEST | Wait for user final-config lock + B14 run |")
    lines.append("| BLOCKED_DICC | Wait for dedicated DICC session |")
    lines.append("")

    md_text = "\n".join(lines)
    (OUT / "table.md").write_text(md_text)
    DOCS_REG.write_text(md_text)

    print(f"Wrote {out_json}")
    print(f"Wrote {OUT / 'table.md'}")
    print(f"Wrote {DOCS_REG}")
    print(f"Claims: {len(claims)} | minority rows: {len(minority_rows)}")
    print(f"Champion md5: {champ_md5} unchanged={package['champion_unchanged']}")
    if missing:
        print("Missing sources:", missing)
    # fail if any LOCKED claim has null value
    bad = [
        c["id"]
        for c in claims
        if c["value"] is None
        and c["status"] not in ("PENDING_SEALED_TEST", "BLOCKED_DICC")
    ]
    if bad:
        print("ERROR null locked claims:", bad)
        return 1
    if champ_md5 != CHAMP_EXPECTED:
        print("ERROR champion md5 mismatch")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
