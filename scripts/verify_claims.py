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

    def fmt_range(lo, hi, decimals=0, suffix="", thousands=False):
        if thousands:
            return f"{round(lo):,}–{round(hi):,}{suffix}"
        if decimals == 0:
            return f"{round(lo)}–{round(hi)}{suffix}"
        return f"{lo:.{decimals}f}{suffix}–{hi:.{decimals}f}{suffix}"

    # Framework comparison. MEASUREMENT STABILITY finding, extended
    # 2026-07-02 (session 3): re-running benchmark_stats_v2.py -- even
    # TWICE, back-to-back, minutes apart, same session -- gave meaningfully
    # different framework latencies (torch.compile and TensorRT both swung
    # 14-17% run to run). This is the same phenomenon already documented for
    # Block 3 alone, now shown to affect the headline framework-comparison
    # ratios too. Historical and session-3a values are hardcoded (captured
    # before the file was regenerated again); the live statistical_
    # significance_v2.json provides the latest session, so the range widens
    # automatically as more sessions are measured instead of silently
    # dropping older ones when the file is overwritten.
    HIST_LATENCY = {
        "Eager PyTorch": 2246.8, "torch.compile": 1777.0, "ORT CPU": 699.2,
        "ORT GPU": 4651.9, "TensorRT": 2965.6, "Custom CUDA FP16": 674.7,
    }
    SESSION3A_LATENCY = {
        "Eager PyTorch": 2050.4, "torch.compile": 1518.9, "ORT CPU": 487.1,
        "ORT GPU": 3861.5, "TensorRT": 2427.1, "Custom CUDA FP16": 652.4,
    }
    # Custom CUDA FP16 (== derived pipeline total) has MORE independent
    # measurements than the framework side, because it's re-derived every
    # time cuda_kernel_stats_rtx3050.json is regenerated for the Block 1-4
    # re-checks, not just when benchmark_stats_v2.py itself is re-run. Found
    # 2026-07-02 while re-verifying all 4 blocks: the "3-session" range above
    # only used HIST/3A/live-file and silently missed two known intermediate
    # derived totals (614.5, 594.0) sitting in already-captured backups --
    # widening the true range from the previously-reported 652-675 to 594-675.
    CUSTOM_CUDA_EXTRA_TOTALS = [614.5, 594.0]

    stats = load_json("statistical_significance_v2.json")
    if stats:
        def latency_range(name):
            vals = [HIST_LATENCY[name], SESSION3A_LATENCY[name], stats[name]["mean_us"]]
            if name == "Custom CUDA FP16":
                vals += CUSTOM_CUDA_EXTRA_TOTALS
            return min(vals), max(vals)

        cc_lo, cc_hi = latency_range("Custom CUDA FP16")
        for name, key in [
            ("TensorRT", "vs_tensorrt"),
            ("torch.compile", "vs_compile"),
            ("Eager PyTorch", "vs_eager"),
        ]:
            fw_lo, fw_hi = latency_range(name)
            add(
                f"framework_speedup_{key}",
                f"Custom CUDA speedup vs {name} (range across 3 independent n=20/n=100 sessions)",
                "statistical_significance_v2.json (3 sessions)",
                [fmt_range(fw_lo / cc_hi, fw_hi / cc_lo, decimals=2, suffix="x")],
            )
        # Raw per-framework latencies -- these should be the ONLY RTX3050
        # framework-latency numbers quoted anywhere in the manuscript. Any
        # other single-run number for these frameworks (there were several
        # superseded ones in git history) is stale.
        for name in ["Eager PyTorch", "torch.compile", "TensorRT", "ORT GPU", "ORT CPU"]:
            lo, hi = latency_range(name)
            add(
                f"framework_latency_{name.replace(' ', '_').replace('.', '')}",
                f"{name} latency (range across 3 independent sessions)",
                "statistical_significance_v2.json (3 sessions)",
                [fmt_range(lo, hi, thousands=True)],
            )
        # The chained-pipeline "vs PyTorch GPU" ratio IS the eager-PyTorch
        # speedup ratio -- they're the same comparison. Track it explicitly
        # so it can't silently drift back to an unsourced constant.
        eager_lo, eager_hi = latency_range("Eager PyTorch")
        add(
            "pipeline_speedup_vs_pytorch_rtx3050",
            "Chained pipeline speedup vs eager PyTorch -- must equal "
            "framework_speedup_vs_eager, not an independently-sourced number",
            "statistical_significance_v2.json (Custom CUDA now n=100-trial "
            "derived, see cuda_kernel_stats_rtx3050.json; 3 sessions)",
            [fmt_range(eager_lo / cc_hi, eager_hi / cc_lo, decimals=2, suffix="x")],
        )
        # Two-sample Welch's significance (fixed 2026-07-01, was one-sample
        # vs a bare constant). The four headline frameworks (Eager PyTorch,
        # torch.compile, TensorRT, ORT GPU) are p<0.001 significant in ALL
        # THREE independently measured sessions -- a robust finding, unlike
        # the exact ratios. ORT CPU is NOT robust: p=0.483 (not significant)
        # historically, but p<0.001 (highly significant, ORT CPU faster) in
        # both fresh 2026-07-02 sessions -- tracked as its own claim rather
        # than pinning one session's now-unrepresentative exact p-value.
        add(
            "two_sample_pvalue_ORT_CPU_unstable",
            "ORT CPU vs Custom CUDA significance is NOT stable across "
            "sessions (not significant historically, highly significant in "
            "both fresh 2026-07-02 sessions) -- a separate, real finding "
            "from the ratio-range instability",
            "statistical_significance_v2.json (3 sessions)",
            ["not consistently", "is not robust"],
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

    # RF accuracy gap. 0.9864 had NO traceable source anywhere in the repo
    # until 2026-07-01 (not in DAILY_LOG.md, not reproduced by
    # scripts/rf_baseline.py's 100-tree/independent-resampling recipe [gives
    # 0.9768], not by a 200-tree variant of that recipe [0.9730], not by
    # train_distill.py's inline 200-tree RF teacher [~0.975 val]). Traced to
    # its actual source: a 200-tree RF trained/evaluated directly on
    # data/processed/*.npy -- the SAME preprocessed splits the CNN-BiLSTM
    # itself uses (apples-to-apples), now saved as a reproducible script
    # (scripts/rf_baseline_processed.py) instead of an ad-hoc terminal
    # command. Confirmed byte-for-byte reproducible: 0.9864 exactly.
    rf_processed = load_json("rf_baseline_processed.json")
    rf_ceiling = rf_processed["test_macro_f1"] if rf_processed else 0.9864
    if rf_processed:
        add(
            "rf_baseline_processed_test_f1",
            "RF baseline test macro-F1, trained/evaluated on data/processed/*.npy (apples-to-apples with CNN-BiLSTM)",
            "rf_baseline_processed.json",
            [fmt(rf_ceiling, 4)],
        )
    # The two-stage headline number itself: fixed 2026-07-01, was ALSO a
    # hand-typed literal here with no JSON source (train_twostage.py never
    # saved one) -- same class of provenance gap as every other fix this
    # session. train_twostage.py now saves benchmarks/results/twostage_botiot.json.
    twostage = load_json("twostage_botiot.json")
    twostage_f1 = twostage["macro_f1"] if twostage else 0.9790
    if twostage:
        add(
            "twostage_final_test_f1",
            "Two-stage CNN-BiLSTM final test macro-F1 (KD a=0.6,T=10.0 + focal + real-data FT)",
            "twostage_botiot.json",
            [fmt(twostage_f1, 4)],
        )
    add(
        "rf_gap_botiot_final",
        "RF gap: 0.9864 (CPU RF, data/processed/*.npy) - twostage_botiot.json's macro_f1",
        "rf_baseline_processed.json + twostage_botiot.json",
        [fmt_pct(rf_ceiling - twostage_f1, 2)],
    )
    add(
        "rf_gap_botiot_baseline",
        "RF gap before distillation: 0.9864 - 0.9352 (Original V3)",
        "rf_baseline_processed.json + README's own headline number",
        [fmt_pct(rf_ceiling - 0.9352, 2)],
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
        # Round 2 (2026-07-01, session 2): extended temperature past 5.0 with
        # focal_gamma=2.0 fixed. a0.6_T10.0 is the new best (feeds Phase B).
        ("distill_botiot_a0.6_T7.0_focal2.json", None),
        ("distill_botiot_a0.7_T7.0_focal2.json", None),
        ("distill_botiot_a0.8_T7.0_focal2.json", None),
        ("distill_botiot_a0.7_T10.0_focal2.json", None),
        ("distill_botiot_a0.8_T10.0_focal2.json", None),
        ("distill_botiot_a0.6_T10.0_focal2.json", None),
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
        add(
            "toniot_clean_cnn_bilstm_f1",
            "CNN-BiLSTM (clean, 26-feature) ToN-IoT macro-F1 -- was only checked "
            "indirectly via the derived gap_percent claim above; added directly "
            "2026-07-02 (session 3)",
            "toniot_clean_comparison.json",
            [fmt(toniot_clean["cnn_bilstm_clean_f1"], 4)],
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
    # (2026-07-01, extended 2026-07-02): re-running the full n=100-trial
    # harness across separate sessions gives meaningfully different means
    # for these configs each time. Rather than picking one, these are RANGE
    # claims -- the full min/max across ALL sessions measured so far must
    # appear together in doc text, hyphenated as written in README. Session
    # 1/2 values are hardcoded (files were overwritten before being frozen
    # here); session 3 (2026-07-02) and beyond read live from the current
    # cuda_kernel_stats_rtx3050.json, so the range widens automatically as
    # more sessions are measured instead of a fixed 2-point comparison
    # silently dropping older sessions when the file is regenerated again.
    SESSION1_TRANSPOSED_NO_GRAPHS = 803.91257
    SESSION1_TRANSPOSED_WITH_GRAPHS = 788.51646
    SESSION1_FP16 = 601.65285
    SESSION2_TRANSPOSED_NO_GRAPHS = 1022.61992
    SESSION2_TRANSPOSED_WITH_GRAPHS = 904.91548
    SESSION2_FP16 = 548.34398
    SESSION2_NAIVE = 5050.103
    # Sessions B/C (2026-07-02): two more independent n=100 runs captured
    # while re-verifying all 4 blocks from a fresh recompile -- previously
    # missed here (this section only ever compared session1/session2 vs
    # "whatever's live now", same class of bug as the framework-comparison
    # one above, same fix: widen to a true min/max over every known session).
    SESSION_B_NO_GRAPHS = 781.4228
    SESSION_B_WITH_GRAPHS = 817.6772
    SESSION_B_FP16 = 580.56564
    SESSION_C_NO_GRAPHS = 732.06563
    SESSION_C_WITH_GRAPHS = 724.37511
    SESSION_C_FP16 = 531.96116
    SESSION_C_NAIVE = 4544.1591

    cuda_stats = load_json("cuda_kernel_stats_rtx3050.json")
    if cuda_stats:
        no_graphs_lo, no_graphs_hi = (
            min(SESSION1_TRANSPOSED_NO_GRAPHS, SESSION2_TRANSPOSED_NO_GRAPHS,
                SESSION_B_NO_GRAPHS, SESSION_C_NO_GRAPHS,
                cuda_stats["fused_block3"]["no_graphs_us"]["mean"]),
            max(SESSION1_TRANSPOSED_NO_GRAPHS, SESSION2_TRANSPOSED_NO_GRAPHS,
                SESSION_B_NO_GRAPHS, SESSION_C_NO_GRAPHS,
                cuda_stats["fused_block3"]["no_graphs_us"]["mean"]),
        )
        with_graphs_lo, with_graphs_hi = (
            min(SESSION1_TRANSPOSED_WITH_GRAPHS, SESSION2_TRANSPOSED_WITH_GRAPHS,
                SESSION_B_WITH_GRAPHS, SESSION_C_WITH_GRAPHS,
                cuda_stats["fused_block3"]["with_graphs_us"]["mean"]),
            max(SESSION1_TRANSPOSED_WITH_GRAPHS, SESSION2_TRANSPOSED_WITH_GRAPHS,
                SESSION_B_WITH_GRAPHS, SESSION_C_WITH_GRAPHS,
                cuda_stats["fused_block3"]["with_graphs_us"]["mean"]),
        )
        fp16_lo, fp16_hi = (
            min(SESSION1_FP16, SESSION2_FP16, SESSION_B_FP16, SESSION_C_FP16,
                cuda_stats["fused_block3_fp16"]["latency_us"]["mean"]),
            max(SESSION1_FP16, SESSION2_FP16, SESSION_B_FP16, SESSION_C_FP16,
                cuda_stats["fused_block3_fp16"]["latency_us"]["mean"]),
        )
        add(
            "block3_transposed_no_graphs",
            "Block 3 transposed W_hh, no graphs (range across 5 independent n=100 sessions)",
            "cuda_kernel_stats_rtx3050.json (5 sessions)",
            [fmt_range(no_graphs_lo, no_graphs_hi, thousands=True)],
        )
        add(
            "block3_transposed_with_graphs",
            "Block 3 transposed W_hh + CUDA graphs (range across 5 independent n=100 sessions)",
            "cuda_kernel_stats_rtx3050.json (5 sessions)",
            [fmt_range(with_graphs_lo, with_graphs_hi)],
        )
        add(
            "block3_fp16_range",
            "Block 3 FP16 half2 (range across 5 independent n=100 sessions)",
            "cuda_kernel_stats_rtx3050.json (5 sessions)",
            [fmt_range(fp16_lo, fp16_hi)],
        )
        naive = (
            cuda_stats["fused_block3_naive"]["latency_us"]["mean"]
            if "fused_block3_naive" in cuda_stats else 5698.0
        )
        if "fused_block3_naive" in cuda_stats:
            naive_lo, naive_hi = (
                min(SESSION2_NAIVE, SESSION_C_NAIVE, naive),
                max(SESSION2_NAIVE, SESSION_C_NAIVE, naive),
            )
            add(
                "block3_naive_latency",
                "Block 3 naive kernel latency, race-condition FIXED and reverified "
                "(range across 3 independent n=100 sessions -- the fix landed in "
                "session 2, so no session-1 measurement of the fixed kernel exists; "
                "replaces the old 5,698us pre-fix historical single run)",
                "cuda_kernel_stats_rtx3050.json (3 sessions)",
                [fmt_range(naive_lo, naive_hi, thousands=True)],
            )
        add(
            "block3_progression_total_ratio",
            "Naive-to-FP16 total progression ratio range (naive range / fp16 range)",
            "cuda_kernel_stats_rtx3050.json",
            [fmt_range(naive_lo / fp16_hi, naive_hi / fp16_lo, decimals=2, suffix="x")],
        )
        if pytorch_block3_stats:
            add(
                "block3_beats_cudnn_ratio",
                "FP16 Block 3 vs PyTorch cuDNN, range across 3 sessions (fixed 2026-07-01)",
                "pytorch_block3_stats_rtx3050.json + cuda_kernel_stats_rtx3050.json (3 sessions)",
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
    ("0.9639 vs 0.9864", "RF gap headline computed against the round-1 KD recipe (a=0.7,T=5.0); superseded by round-2's a=0.6,T=10.0 recipe reaching 0.9790, gap now 0.74% not 2.25%", "2026-07-01"),
    ("closes the accuracy gap to **2.25%** on BoT-IoT (**0.9639**", "abstract's old headline gap/accuracy figures for BoT-IoT, superseded by the round-2 KD sweep (0.74% / 0.9790)", "2026-07-01"),
    ("covers CNN only with 2.7x", "fabricated closest-prior-work claim attributing a 2.7x CUDA speedup to 'Sophimatics Phase 3' (DOI 10.3390/app152211876), a paper that actually has no CUDA/CNN/speedup content; retracted pending a real replacement", "2026-07-02"),
    ("Must be cited in the manuscript as the closest prior work", "instruction text from the same fabricated Sophimatics citation note, never independently verified before being written into the manuscript text blocks", "2026-07-02"),
    ("4.40x over TensorRT", "single-point framework-comparison headline; superseded by a 3.60x-4.55x range once a 2nd and 3rd measurement session showed the original 674.7us/4.40x figure was near the favorable end, not representative", "2026-07-02"),
    ("2.63x over torch.compile", "single-point framework-comparison headline; superseded by a 2.25x-2.72x range (same session-to-session drift finding as the TensorRT/eager-PyTorch ratios)", "2026-07-02"),
    ("3.33x over eager PyTorch", "single-point framework-comparison headline; superseded by a 3.04x-3.44x range (same session-to-session drift finding)", "2026-07-02"),
    ("ORT GPU (6.89x)", "single-point framework-comparison headline; superseded by a 5.72x-7.13x range (same session-to-session drift finding)", "2026-07-02"),
    ("mean 674.7us, std 87.1us", "single-session Custom CUDA FP16 pipeline-total figure treated as fixed; superseded by a 594-675us range across 5 independent sessions", "2026-07-02"),
    ("3.60x–4.55x", "intermediate framework-ratio range (2 sessions' worth of Custom CUDA data); superseded by 3.60x-4.99x once 2 more Custom CUDA sessions (594, 614.5us) were folded in during the full 4-block re-verification pass", "2026-07-02"),
    ("2.25x–2.72x", "intermediate framework-ratio range; superseded by 2.25x-2.99x, same widening", "2026-07-02"),
    ("3.04x–3.44x", "intermediate framework-ratio range; superseded by 3.04x-3.78x, same widening", "2026-07-02"),
    ("5.72x–7.13x", "intermediate framework-ratio range; superseded by 5.72x-7.83x, same widening", "2026-07-02"),
    ("652–675", "intermediate Custom CUDA FP16 range (only used Historical+2 back-to-back framework runs, missed 2 already-captured intermediate derived-total measurements); superseded by 594-675", "2026-07-02"),
    ("8.08x–9.21x", "intermediate Block 3 progression range (3 sessions); superseded by 7.55x-9.50x once 2 more sessions were folded in", "2026-07-02"),
    ("4,860–5,050", "intermediate Block 3 naive-kernel range (2 sessions); superseded by 4,544-5,050 (3 sessions)", "2026-07-02"),
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
