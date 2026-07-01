"""
COLIDE - Claim Verifier

Cross-checks every headline number in README.md and docs/paper_text_blocks.md
against the benchmarks/results/*.json file that is supposed to have produced it.

Every number quoted in the manuscript should trace to exactly one entry in
CLAIMS below. If a fix changes a source JSON, this script is how you confirm
the prose was updated to match -- instead of re-checking every table by hand.

Usage:
    PYTHONPATH=. python scripts/verify_claims.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
DOC_FILES = {
    "README.md": PROJECT_ROOT / "README.md",
    "paper_text_blocks.md": PROJECT_ROOT / "docs" / "paper_text_blocks.md",
}


def load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_docs():
    texts = {}
    for label, path in DOC_FILES.items():
        texts[label] = path.read_text() if path.exists() else ""
    return texts


def fmt(value, decimals=4):
    return f"{value:.{decimals}f}"


def fmt_pct(value, decimals=2):
    return f"{value * 100:.{decimals}f}"


def fmt_ratio(value, decimals=2):
    return f"{value:.{decimals}f}x"


# ================================================================
# Claim manifest: (id, description, source(s), expected string variants)
# `check` returns a list of acceptable rendered strings; if ANY of them
# appears in ANY doc file, the claim passes.
# ================================================================

def build_claims():
    claims = []

    def add(claim_id, description, source, variants):
        claims.append({
            "id": claim_id,
            "description": description,
            "source": source,
            "variants": [v for v in variants if v],
        })

    stats = load_json("statistical_significance_v2.json")
    if stats:
        custom = stats["Custom CUDA FP16"]["mean_us"]
        for name, key in [
            ("TensorRT", "vs_tensorrt"),
            ("torch.compile", "vs_compile"),
            ("Eager PyTorch", "vs_eager"),
        ]:
            mean = stats[name]["mean_us"]
            ratio = mean / custom
            add(
                f"framework_speedup_{key}",
                f"Custom CUDA speedup vs {name} (statistical_significance_v2.json)",
                "statistical_significance_v2.json",
                [fmt_ratio(ratio, 2)],
            )
        # Raw per-framework latencies (20-trial means) -- these should be the
        # ONLY RTX3050 framework-latency numbers quoted anywhere in the
        # manuscript. Any other single-run number for these frameworks
        # (there were several superseded ones in git history) is stale.
        for name in ["Eager PyTorch", "torch.compile", "TensorRT", "ORT GPU", "ORT CPU"]:
            mean = stats[name]["mean_us"]
            add(
                f"framework_latency_{name.replace(' ', '_').replace('.', '')}",
                f"{name} 20-trial mean latency (statistical_significance_v2.json)",
                "statistical_significance_v2.json",
                [f"{round(mean):,}", f"{mean:.0f}"],
            )
        # The chained-pipeline "vs PyTorch GPU" ratio IS the eager-PyTorch
        # speedup ratio -- they're the same comparison. Track it explicitly
        # so it can't silently drift back to an unsourced constant.
        custom = stats["Custom CUDA FP16"]["mean_us"]  # now ~674.7, n=100-trial derived
        pipeline_ratio = stats["Eager PyTorch"]["mean_us"] / custom
        add(
            "pipeline_speedup_vs_pytorch_rtx3050",
            "Chained pipeline speedup vs eager PyTorch -- must equal "
            "framework_speedup_vs_eager, not an independently-sourced number",
            "statistical_significance_v2.json (Custom CUDA now n=100-trial "
            "derived, see cuda_kernel_stats_rtx3050.json)",
            [fmt_ratio(pipeline_ratio, 2)],
        )
        # Two-sample Welch's p-values (fixed 2026-07-01, was one-sample vs a
        # bare constant). Track a couple of the exact values so they can't
        # silently drift back to the old one-sample numbers.
        for name, expected_p in [
            ("torch.compile", "3.55e-19"),
            ("ORT CPU", "0.483"),
        ]:
            add(
                f"two_sample_pvalue_{name.replace(' ', '_').replace('.', '')}",
                f"{name} two-sample Welch p-value vs Custom CUDA "
                "(statistical_significance_v2.json)",
                "statistical_significance_v2.json",
                [expected_p],
            )

    # LLM dispatch overhead -- fixed 2026-07-01 (Phase 1.1). Real p99 over
    # 5,000 trials of the classify+construct+push code path.
    llm = load_json("llm_explainability.json")
    if llm:
        add(
            "llm_dispatch_overhead_p99",
            "Async LLM dispatch overhead, p99 (llm_explainability.json dispatch_p99_us)",
            "llm_explainability.json",
            [f"{llm['overhead_p99_us']:.2f}"],
        )

    # RF accuracy gap
    add(
        "rf_gap_botiot_final",
        "RF gap: 0.9864 (CPU RF) - 0.9639 (two-stage CNN-BiLSTM)",
        "arithmetic on README's own two headline numbers",
        [fmt_pct(0.9864 - 0.9639, 2)],
    )
    add(
        "rf_gap_botiot_baseline",
        "RF gap before distillation: 0.9864 - 0.9352 (Original V3)",
        "arithmetic on README's own two headline numbers",
        [fmt_pct(0.9864 - 0.9352, 2)],
    )

    # KD sweep table
    kd_sweep = [
        ("distill_botiot_a0.3_T1.0.json", None),
        ("distill_botiot.json", None),
        ("distill_botiot_a0.7_T1.0.json", None),
        ("distill_botiot_a0.9_T1.0.json", None),
        ("distill_botiot_a0.5_T3.0.json", None),
        ("distill_botiot_a0.7_T5.0.json", None),
        ("distill_botiot_focal_T5.json", None),
    ]
    for fname, _ in kd_sweep:
        d = load_json(fname)
        if not d:
            continue
        add(
            f"kd_sweep_valf1_{fname}",
            f"KD sweep Val F1 (alpha={d.get('alpha')}, T={d.get('temperature')}) "
            f"from {fname}",
            fname,
            [fmt(d["best_val_f1"], 4)],
        )
        add(
            f"kd_sweep_testf1_{fname}",
            f"KD sweep Test F1 (alpha={d.get('alpha')}, T={d.get('temperature')}) "
            f"from {fname}",
            fname,
            [fmt(d["macro_f1"], 4)],
        )

    # Main BoT-IoT results table
    for claim_id, fname, key, desc in [
        ("mlp_distilled_f1", "ablation_mlp.json", "macro_f1", "MLP (distilled) macro-F1"),
        ("mlp_twostage_f1", "mlp_twostage.json", "macro_f1", "MLP (two-stage) macro-F1"),
        ("ensemble_kd_f1", "ensemble_distill.json", "macro_f1", "Ensemble KD macro-F1"),
    ]:
        d = load_json(fname)
        if d:
            add(claim_id, desc, fname, [fmt(d[key], 4)])

    # ToN-IoT
    toniot_clean = load_json("toniot_clean_comparison.json")
    if toniot_clean:
        add(
            "toniot_clean_gap",
            "ToN-IoT clean gap percent",
            "toniot_clean_comparison.json",
            [f"{toniot_clean['gap_percent']:.1f}", f"{toniot_clean['gap_percent']:.2f}"],
        )
    distill_v2 = load_json("distill_toniot_v2.json")
    if distill_v2:
        add(
            "toniot_original_f1",
            "CNN-BiLSTM (original, 13-feature) ToN-IoT macro-F1",
            "distill_toniot_v2.json",
            [fmt(distill_v2["macro_f1"], 4)],
        )

    # MLP ablation latency
    mlp_lat = load_json("mlp_latency.json")
    if mlp_lat:
        add(
            "mlp_latency_us",
            "MLP distilled latency (A100)",
            "mlp_latency.json",
            [f"{mlp_lat['avg_latency_us']:.0f} us", f"{round(mlp_lat['avg_latency_us'])} us"],
        )

    # Energy
    energy = load_json("energy_efficiency.json")
    if energy:
        add(
            "rtx3050_energy_mj_per_flow",
            "RTX 3050 energy per flow, batch=128",
            "energy_efficiency.json",
            [f"{energy['gpu_batch128']['mj_per_flow']:.2f}"],
        )
    a100_energy = load_json("a100_energy.json")
    if a100_energy:
        add(
            "a100_energy_mj_per_flow",
            "A100 energy per flow",
            "a100_energy.json",
            [f"{a100_energy['energy_per_flow_mj']:.3f}"],
        )
        # A100 CNN-BiLSTM throughput derived from batch time -- used to check
        # the cuML "(A100)" table isn't silently mixing in the RTX3050 number.
        a100_throughput = 128 / (a100_energy["avg_batch_time_ms"] / 1000.0)
        add(
            "a100_cnn_bilstm_throughput",
            "A100 CNN-BiLSTM throughput derived from a100_energy.json "
            "avg_batch_time_ms (128 / (avg_batch_time_ms/1000)) -- Phase 2.5 "
            "should make the cuML(A100) table use THIS, not the RTX3050 number",
            "a100_energy.json (derived)",
            [f"{a100_throughput:,.0f}", f"{round(a100_throughput):,}"],
        )

    # cuML comparison
    cuml_res = load_json("cuml_rf_resources.json")
    if cuml_res:
        add(
            "cuml_vram_mb",
            "cuML RF VRAM (A100)",
            "cuml_rf_resources.json",
            [f"{cuml_res['cuml_rf_vram_mb']}"],
        )
        add(
            "cuml_throughput",
            "cuML RF throughput (A100)",
            "cuml_rf_resources.json",
            [f"{cuml_res['cuml_rf_throughput_avg']:,}"],
        )

    # PyTorch cuDNN baseline for Block 3 alone -- fixed 2026-07-01, resolves
    # the 740.7us-vs-943.6us single-run ambiguity with a real n=50-trial mean
    # (scripts/benchmark_pytorch_block3_stats.py).
    pytorch_block3_stats = load_json("pytorch_block3_stats_rtx3050.json")
    if pytorch_block3_stats:
        pt_b3_mean = pytorch_block3_stats["gpu_p50_us"]["mean"]
        add(
            "pytorch_block3_cudnn_baseline",
            "PyTorch cuDNN baseline for Block 3 alone (fresh n=50-trial mean)",
            "pytorch_block3_stats_rtx3050.json",
            [f"{round(pt_b3_mean)}", f"{pt_b3_mean:.0f}"],
        )

    # Block 3 optimization progression. MEASUREMENT STABILITY finding
    # (2026-07-01): re-running the full n=100-trial harness later the same
    # day (needed to safely add the newly-fixed naive kernel without
    # clobbering the file) gave meaningfully different means for these
    # three configs than the SAME harness gave earlier that day. Rather than
    # picking one, these are RANGE claims -- both session's rounded bounds
    # must appear together in doc text, hyphenated as written in README.
    # Session 1 values captured before the file was regenerated in session 2.
    SESSION1_TRANSPOSED_NO_GRAPHS = 803.91257
    SESSION1_TRANSPOSED_WITH_GRAPHS = 788.51646
    SESSION1_FP16 = 601.65285

    def fmt_range(lo, hi, decimals=0, suffix="", thousands=False):
        if thousands:
            return f"{round(lo):,}–{round(hi):,}{suffix}"
        if decimals == 0:
            return f"{round(lo)}–{round(hi)}{suffix}"
        return f"{lo:.{decimals}f}{suffix}–{hi:.{decimals}f}{suffix}"

    cuda_stats = load_json("cuda_kernel_stats_rtx3050.json")
    if cuda_stats:
        no_graphs_lo, no_graphs_hi = sorted([
            SESSION1_TRANSPOSED_NO_GRAPHS,
            cuda_stats["fused_block3"]["no_graphs_us"]["mean"],
        ])
        with_graphs_lo, with_graphs_hi = sorted([
            SESSION1_TRANSPOSED_WITH_GRAPHS,
            cuda_stats["fused_block3"]["with_graphs_us"]["mean"],
        ])
        fp16_lo, fp16_hi = sorted([
            SESSION1_FP16,
            cuda_stats["fused_block3_fp16"]["latency_us"]["mean"],
        ])
        add(
            "block3_transposed_no_graphs",
            "Block 3 transposed W_hh, no graphs (range across 2 independent n=100 sessions)",
            "cuda_kernel_stats_rtx3050.json (2 sessions)",
            [fmt_range(no_graphs_lo, no_graphs_hi, thousands=True)],
        )
        add(
            "block3_transposed_with_graphs",
            "Block 3 transposed W_hh + CUDA graphs (range across 2 independent n=100 sessions)",
            "cuda_kernel_stats_rtx3050.json (2 sessions)",
            [fmt_range(with_graphs_lo, with_graphs_hi)],
        )
        add(
            "block3_fp16_range",
            "Block 3 FP16 half2 (range across 2 independent n=100 sessions)",
            "cuda_kernel_stats_rtx3050.json (2 sessions)",
            [fmt_range(fp16_lo, fp16_hi)],
        )
        naive = (
            cuda_stats["fused_block3_naive"]["latency_us"]["mean"]
            if "fused_block3_naive" in cuda_stats else 5698.0
        )
        if "fused_block3_naive" in cuda_stats:
            add(
                "block3_naive_latency",
                "Block 3 naive kernel latency, race-condition FIXED and reverified "
                "(real n=100-trial mean, replaces the old 5,698us historical single run)",
                "cuda_kernel_stats_rtx3050.json",
                [f"{round(naive):,}"],
            )
        add(
            "block3_progression_total_ratio",
            "Naive-to-FP16 total progression ratio range (naive n=100 mean / fp16 range)",
            "cuda_kernel_stats_rtx3050.json",
            [fmt_range(naive / fp16_hi, naive / fp16_lo, decimals=2, suffix="x")],
        )
        if pytorch_block3_stats:
            add(
                "block3_beats_cudnn_ratio",
                "FP16 Block 3 vs PyTorch cuDNN, range across 2 sessions (fixed 2026-07-01)",
                "pytorch_block3_stats_rtx3050.json + cuda_kernel_stats_rtx3050.json (2 sessions)",
                [fmt_range(pt_b3_mean / fp16_hi, pt_b3_mean / fp16_lo, decimals=2, suffix="x")],
            )

    # Cross-hardware pipeline totals -- these ARE trustworthy (see dicc summary
    # txt files), unlike the "x over PyTorch GPU" ratios derived from them.
    add(
        "pipeline_total_v100s",
        "V100S chained pipeline total (dicc_v100_summary.txt)",
        "dicc_v100_summary.txt",
        ["550.7", "550.664", "551"],
    )
    add(
        "pipeline_total_a100",
        "A100 chained pipeline total (dicc_a100_summary.txt)",
        "dicc_a100_summary.txt",
        ["592.0", "592.044", "592"],
    )

    # Streaming throughput -- streaming_throughput.json is the dedicated,
    # purpose-built benchmark for this claim, chosen as canonical over
    # cuml_rf_native.json's incidental custom_cuda_flows_sec field (25,410,
    # a ~2% lower secondary measurement from a script whose primary purpose
    # was the RF comparison, not streaming throughput). Fixed 2026-07-01.
    stream = load_json("streaming_throughput.json")
    if stream:
        max_gpu = stream["max_throughput"]["gpu_batched"]
        add(
            "streaming_throughput",
            "Max streaming throughput (streaming_throughput.json gpu_batched, "
            "the canonical source for this claim)",
            "streaming_throughput.json",
            [f"{max_gpu:,.0f}"],
        )

    return claims


def check_claims():
    docs = load_docs()
    claims = build_claims()
    passed, failed = [], []

    for claim in claims:
        found_in = []
        for doc_name, text in docs.items():
            for variant in claim["variants"]:
                if variant in text:
                    found_in.append((doc_name, variant))
                    break
        if found_in:
            passed.append((claim, found_in))
        else:
            failed.append(claim)

    return passed, failed


def scan_orphan_numbers(covered_variants):
    """Flag bolded/table numbers in README.md not covered by any claim.
    Heuristic only -- meant to surface candidates for the manifest, not a
    complete/precise parse of the document."""
    text = DOC_FILES["README.md"].read_text()
    numbers = set(re.findall(r"\*\*([\d.]+x?%?)\*\*", text))
    covered = set()
    for variants in covered_variants:
        for v in variants:
            covered.add(v.rstrip("x%").rstrip())
    orphans = sorted(
        n for n in numbers
        if n.rstrip("x%").rstrip() not in covered and len(n) > 1
    )
    return orphans


# Strings from fixed bugs that must never silently reappear (e.g. from a
# copy-pasted stale draft or a reverted edit). Each entry: (banned string,
# what it was, when/why it was fixed).
REGRESSION_GUARDS = [
    ("5.19 us", "fabricated LLM dispatch p99 (no computation ever backed it)", "2026-07-01"),
    ("0.9487", "KD sweep transcription error for alpha=0.5,T=3.0 (should be 0.9541)", "2026-07-01"),
    ("2.76x", "pipeline-vs-PyTorch ratio computed from an unsourced 1864.0 constant", "2026-07-01"),
    ("3.39x", "V100S vs-PyTorch ratio computed using the RTX3050 PyTorch baseline (cross-hardware mixing)", "2026-07-01"),
    ("3.15x", "A100 vs-PyTorch ratio computed using the RTX3050 PyTorch baseline (cross-hardware mixing)", "2026-07-01"),
    ("beating cuDNN by 1.23x", "Block3-vs-cuDNN ratio computed from an ambiguous single-run baseline (740.7 vs 943.6us); superseded by a real n=50-trial mean (784us), giving 1.30x", "2026-07-01"),
    ("5,698", "naive Block3 kernel latency: historical single-run figure from a pre-fix (racy) binary, superseded by a real n=100-trial mean of the race-condition-fixed kernel (5,050us)", "2026-07-01"),
    ("9.47x", "naive-to-FP16 progression ratio computed from the old 5,698us naive figure and a single-session FP16 mean; superseded by a range (8.39x-9.21x) reflecting both the naive-kernel fix and the measurement-stability finding", "2026-07-01"),
]


def check_regressions():
    docs = load_docs()
    hits = []
    for banned, why, fixed_date in REGRESSION_GUARDS:
        for doc_name, text in docs.items():
            if banned in text:
                hits.append((banned, why, fixed_date, doc_name))
    return hits


def main():
    passed, failed = check_claims()
    regressions = check_regressions()

    print("=" * 78)
    print("COLIDE CLAIM VERIFIER")
    print("=" * 78)

    print(f"\n{len(passed)} claim(s) PASSED (source JSON value found in doc text):\n")
    for claim, found_in in passed:
        loc = ", ".join(f"{doc}" for doc, _ in found_in)
        print(f"  [OK] {claim['id']:<35} <- {claim['source']}  (in {loc})")

    print(f"\n{len(failed)} claim(s) FAILED (source value NOT found in README/docs):\n")
    for claim in failed:
        print(f"  [FAIL] {claim['id']}")
        print(f"         {claim['description']}")
        print(f"         source: {claim['source']}")
        print(f"         expected one of: {claim['variants']}")
        print()

    all_claims = build_claims()
    orphans = scan_orphan_numbers([c["variants"] for c in all_claims])
    if orphans:
        print("-" * 78)
        print(f"{len(orphans)} bolded number(s) in README.md not covered by any "
              f"claim in this manifest (not necessarily wrong -- just unverified; "
              f"add them to CLAIMS if they're load-bearing for the manuscript):\n")
        print("  " + ", ".join(orphans))

    if regressions:
        print("-" * 78)
        print(f"REGRESSION ALERT: {len(regressions)} previously-fixed bad number(s) "
              f"have reappeared:\n")
        for banned, why, fixed_date, doc_name in regressions:
            print(f"  [REGRESSION] '{banned}' found in {doc_name}")
            print(f"               was: {why} (fixed {fixed_date})")

    print("\n" + "=" * 78)
    if failed or regressions:
        print(f"RESULT: {len(failed)} claim(s) need fixing, "
              f"{len(regressions)} regression(s) detected. Not safe to lock.")
    else:
        print("RESULT: all tracked claims verified against source JSON, "
              "no regressions detected.")
    print("=" * 78)

    return 1 if (failed or regressions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
