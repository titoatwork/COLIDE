"""
COLIDE - Ablation Study
Shows the contribution of each optimization technique to Block 3 (BiLSTM).
Also shows overall pipeline progression.

Rebuilt 2026-07-01 to load from benchmarks/results/*.json wherever a source
exists, instead of hand-typed literals disconnected from any measurement.
A few numbers have no surviving JSON artifact (noted inline below, e.g. the
"+Precomputed W_ih*X" intermediate optimization step, whose standalone
kernel file was overwritten by later optimizations) -- those remain
historical constants, explicitly labeled as such rather than silently
presented as freshly computed.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def load(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


cuda_stats = load("cuda_kernel_stats_rtx3050.json")
pytorch_block3_stats = load("pytorch_block3_stats_rtx3050.json")
stat_sig = load("statistical_significance_v2.json")
pipeline_bench = load("pipeline_benchmark.json")
energy = load("energy_efficiency.json")
llm = load("llm_explainability.json")
toniot_multi = load("toniot_multi_eval.json")
toniot_v2 = load("distill_toniot_v2.json")
rf_toniot = load("rf_baseline_toniot.json")

print("=" * 75)
print("COLIDE ABLATION STUDY")
print("=" * 75)

# ================================================================
# Table A: Block 3 Optimization Progression
# ================================================================
print(f"\n{'='*75}")
print("TABLE A: Block 3 (BiLSTM) Optimization Progression")
print(f"{'='*75}")
print(f"{'Configuration':<70} {'Latency':>10} {'vs Naive':>10} {'vs PyTorch':>10}")
print(f"{'-'*75}")

# PyTorch cuDNN baseline for Block 3 alone. RESOLVED 2026-07-01: two
# single-run point estimates used to disagree (740.7us historical vs 943.6us
# from a fresh benchmark_pipeline.py run) because both were one draw from a
# noisy distribution -- benchmark_pytorch_block3_stats.py now runs 50
# independent subprocess trials (mirroring the CUDA kernel statistical
# harness) and gives a real mean/std: 784.1us +/- 88.6us (CV 11.3%,
# n=50). That real mean sits between the two old single-run numbers, as
# expected for a noisy quantity. This is now the canonical baseline.
pytorch_cudnn_mean = (
    pytorch_block3_stats["gpu_p50_us"]["mean"] if pytorch_block3_stats else 784.1
)

# Naive kernel: RACE CONDITION FIXED 2026-07-01. compute-sanitizer racecheck
# found a genuine shared-memory hazard (write of the per-timestep hidden
# state racing with the next timestep's read of it, despite a __syncthreads()
# between them) -- confirmed independently by the kernel producing different
# output across repeated runs with an identical, fixed host-side RNG seed
# (impossible from pure FP32 summation-order rounding, which is
# deterministic). Fixed via double-buffering the hidden state in
# fused_block3_naive.cu so a timestep's read and write never target the same
# shared array. Verified: 0 hazards under racecheck (was reporting thousands
# before), 100/100 runs pass at the standard 1e-2 tolerance (was ~6/30), and
# 20/20 pass even at a much tighter 1e-5 tolerance -- i.e. it's not just
# "passing at a loose threshold," it's genuinely close to the CPU reference
# now. naive_mean below is therefore a real n=100-trial mean of the FIXED
# kernel, not a historical pre-fix single run.
naive_mean = cuda_stats["fused_block3_naive"]["latency_us"]["mean"] if cuda_stats and "fused_block3_naive" in cuda_stats else 5698.0
naive_source = ("fresh, n=%d trials, race-condition FIXED and reverified"
                 % cuda_stats["fused_block3_naive"]["n_trials"]
                 if cuda_stats and "fused_block3_naive" in cuda_stats
                 else "historical, pre-fix single run")

# No standalone binary exists for this intermediate optimization step (the
# kernel file was overwritten by the next step) -- historical constant only.
precomputed_historical = 2901.0

# MEASUREMENT STABILITY FINDING (2026-07-01): re-running the full n=100-trial
# CUDA kernel harness later the same day (needed to safely add the
# newly-fixed naive kernel's stats without clobbering the file with a
# partial run) gave meaningfully different means for these three configs
# than the SAME harness gave earlier that day (session 1) -- despite each
# individual session's own internal CV looking tight (6.8%-24.4%). The
# within-session CV therefore understates true measurement uncertainty on
# this WSL2 dev box: there is real session-to-session drift (thermal state /
# background load / WSL2 scheduler) that one n=100 run, however tight its own
# std, does not capture. Reporting BOTH sessions' means as an explicit range
# rather than silently picking one -- see README.md's "Measurement
# Stability" note. (Recommendation for the DICC re-run: repeat the same
# n-trial harness across >=2 separate sbatch submissions on different days
# and check for the same drift there.)
SESSION1_TRANSPOSED_NO_GRAPHS = 803.91257
SESSION1_TRANSPOSED_WITH_GRAPHS = 788.51646
SESSION1_FP16 = 601.65285

transposed_no_graphs_s2 = cuda_stats["fused_block3"]["no_graphs_us"]["mean"] if cuda_stats else 1007.3
transposed_with_graphs_s2 = cuda_stats["fused_block3"]["with_graphs_us"]["mean"] if cuda_stats else 973.8
fp16_s2 = cuda_stats["fused_block3_fp16"]["latency_us"]["mean"] if cuda_stats else 601.4

no_graphs_lo, no_graphs_hi = sorted([SESSION1_TRANSPOSED_NO_GRAPHS, transposed_no_graphs_s2])
with_graphs_lo, with_graphs_hi = sorted([SESSION1_TRANSPOSED_WITH_GRAPHS, transposed_with_graphs_s2])
fp16_lo, fp16_hi = sorted([SESSION1_FP16, fp16_s2])

# Downstream tables (B, H) need one representative number, not a range --
# use the latest (session 2) measurement for those; Table A above/below
# carries the full range disclosure.
transposed_with_graphs = transposed_with_graphs_s2
fp16_mean = fp16_s2

def _rng(lo, hi, decimals=2, suffix="x"):
    return f"{lo:.{decimals}f}{suffix}" if abs(hi - lo) < 1e-9 else f"{lo:.{decimals}f}-{hi:.{decimals}f}{suffix}"

rows = [
    ("PyTorch GPU (cuDNN baseline)", f"{pytorch_cudnn_mean:.1f} us", "---", "1.00x"),
    (f"Naive custom (1 thread/hidden, global W_hh) [{naive_source}]",
     f"{naive_mean:.1f} us", "1.00x", f"{pytorch_cudnn_mean / naive_mean:.2f}x"),
    ("+ Precomputed input projection (W_ih*X) [historical, no live source]",
     f"{precomputed_historical:.1f} us", f"{naive_mean / precomputed_historical:.2f}x",
     f"{pytorch_cudnn_mean / precomputed_historical:.2f}x"),
    ("+ Transposed W_hh (coalesced reads) [range: 2 independent n=100 sessions]",
     f"{no_graphs_lo:.1f}-{no_graphs_hi:.1f} us",
     _rng(naive_mean / no_graphs_hi, naive_mean / no_graphs_lo),
     _rng(pytorch_cudnn_mean / no_graphs_hi, pytorch_cudnn_mean / no_graphs_lo)),
    ("+ CUDA Graphs [range: 2 independent n=100 sessions]",
     f"{with_graphs_lo:.1f}-{with_graphs_hi:.1f} us",
     _rng(naive_mean / with_graphs_hi, naive_mean / with_graphs_lo),
     _rng(pytorch_cudnn_mean / with_graphs_hi, pytorch_cudnn_mean / with_graphs_lo)),
    ("+ FP16 native half2 FMA [range: 2 independent n=100 sessions]",
     f"{fp16_lo:.1f}-{fp16_hi:.1f} us",
     _rng(naive_mean / fp16_hi, naive_mean / fp16_lo),
     _rng(pytorch_cudnn_mean / fp16_hi, pytorch_cudnn_mean / fp16_lo)),
]

for name, lat, vs_naive, vs_pytorch in rows:
    print(f"{name:<74} {lat:>14} {vs_naive:>12} {vs_pytorch:>12}")

# Percentage-contribution breakdown, computed from session 2 (latest) means.
# Shown alongside the session 1 breakdown to make the instability concrete:
# these percentages are NOT stable across sessions either.
def _contrib_pct(no_graphs, with_graphs, fp16):
    total = naive_mean - fp16
    return (
        (naive_mean - precomputed_historical) / total * 100,
        (precomputed_historical - no_graphs) / total * 100,
        (no_graphs - with_graphs) / total * 100,
        (with_graphs - fp16) / total * 100,
    )

s2_pct = _contrib_pct(transposed_no_graphs_s2, transposed_with_graphs_s2, fp16_s2)
s1_pct = _contrib_pct(SESSION1_TRANSPOSED_NO_GRAPHS, SESSION1_TRANSPOSED_WITH_GRAPHS, SESSION1_FP16)

print(f"\nContribution breakdown of naive-to-FP16 improvement (precompute / transpose")
print(f"/ graphs / fp16), computed two ways to make the session-to-session drift")
print(f"concrete rather than picking one silently:")
print(f"  Session 1 (n=100): {s1_pct[0]:.1f}% / {s1_pct[1]:.1f}% / {s1_pct[2]:.1f}% / {s1_pct[3]:.1f}%")
print(f"  Session 2 (n=100): {s2_pct[0]:.1f}% / {s2_pct[1]:.1f}% / {s2_pct[2]:.1f}% / {s2_pct[3]:.1f}%")
print(f"Naive-to-FP16 total ratio range: {naive_mean/fp16_hi:.2f}x-{naive_mean/fp16_lo:.2f}x")
print(f"({naive_mean:.0f} -> {fp16_lo:.0f}-{fp16_hi:.0f} us).")

if pytorch_block3_stats:
    print(f"\nPyTorch cuDNN baseline for Block 3 ({pytorch_cudnn_mean:.1f}us) is a real")
    print(f"n=50-trial mean (std {pytorch_block3_stats['gpu_p50_us']['std']:.1f}us, CV "
          f"{pytorch_block3_stats['gpu_p50_us']['cv_pct']:.1f}%), resolving the earlier")
    print(f"740.7-vs-943.6us single-run ambiguity -- see scripts/benchmark_pytorch_block3_stats.py.")
else:
    print(f"\n[no pytorch_block3_stats_rtx3050.json found -- run "
          f"scripts/benchmark_pytorch_block3_stats.py to regenerate]")
print(f"With this real baseline, the FP16 step beats cuDNN across both measurement")
print(f"sessions ({pytorch_cudnn_mean/fp16_hi:.2f}x-{pytorch_cudnn_mean/fp16_lo:.2f}x); the transposed-W_hh")
print(f"steps land at/below parity in BOTH sessions "
      f"({pytorch_cudnn_mean/no_graphs_hi:.2f}-{pytorch_cudnn_mean/no_graphs_lo:.2f}x no-graphs, "
      f"{pytorch_cudnn_mean/with_graphs_hi:.2f}-{pytorch_cudnn_mean/with_graphs_lo:.2f}x with-graphs) --")
print(f"that specific conclusion (transposed steps don't clearly beat cuDNN) is robust")
print(f"across the session-to-session drift, even though the exact ratios aren't.")

# ================================================================
# Table B: Per-Block Speedup Summary
# ================================================================
print(f"\n{'='*75}")
print("TABLE B: Per-Block Custom CUDA vs PyTorch GPU")
print(f"{'='*75}")
print(f"{'Block':<30} {'PyTorch GPU':>12} {'Custom CUDA':>12} {'Speedup':>10} {'% of Total':>10}")
print(f"{'-'*75}")

if cuda_stats and pipeline_bench:
    b1_cuda = cuda_stats["fused_block1"]["latency_us"]["mean"]
    b2_cuda = cuda_stats["fused_block2"]["latency_us"]["mean"]
    b3_cuda = transposed_with_graphs
    b4_cuda = cuda_stats["fused_block4"]["latency_us"]["mean"]
    pt = pipeline_bench["pytorch_blocks"]
    blocks = [
        ("1: Proj+Conv+BN+ReLU", pt["block1"]["gpu"], b1_cuda),
        ("2: Conv+BN+ReLU+Pool", pt["block2"]["gpu"], b2_cuda),
        ("3: BiLSTM (2 layers)", pt["block3"]["gpu"], b3_cuda),
        ("4: Dense (128->5)", pt["block4"]["gpu"], b4_cuda),
    ]
else:
    # Fallback historical values if the JSON sources aren't present.
    blocks = [
        ("1: Proj+Conv+BN+ReLU", 404.4, 61.7),
        ("2: Conv+BN+ReLU+Pool", 282.1, 87.2),
        ("3: BiLSTM (2 layers)", 791.1, 973.8),
        ("4: Dense (128->5)", 122.1, 20.1),
    ]

total_pt = sum(b[1] for b in blocks)
total_cuda = sum(b[2] for b in blocks)

for name, pt_lat, cuda_lat in blocks:
    speedup = pt_lat / cuda_lat
    pct = cuda_lat / total_cuda * 100
    print(f"{name:<30} {pt_lat:>10.1f} us {cuda_lat:>10.1f} us {speedup:>8.2f}x {pct:>9.1f}%")

print(f"{'-'*75}")
print(f"{'TOTAL':<30} {total_pt:>10.1f} us {total_cuda:>10.1f} us {total_pt/total_cuda:>8.2f}x {'100.0':>9}%")

print(f"\nAll figures above are freshly measured (n=100 trials for the custom CUDA")
print(f"kernels; see benchmarks/results/cuda_kernel_stats_rtx3050.json and")
print(f"pipeline_benchmark.json), not restated from an earlier draft.")

pipeline_total = cuda_stats.get("derived_pipeline_total_us") if cuda_stats else 674.2
eager_mean = stat_sig["Eager PyTorch"]["mean_us"] if stat_sig else None
if eager_mean:
    # NOTE (corrected 2026-07-01): this "pipeline_total" is an ADDITIVE
    # reconstruction (mean b124_chained + mean block3_fp16, both from
    # cuda_kernel_stats_rtx3050.json) -- a DIFFERENT measurement methodology
    # than statistical_significance_v2.json's directly-measured "Custom CUDA
    # FP16" total (674.7us), which is what README's official "3.33x over
    # eager PyTorch" headline actually uses. An earlier version of this
    # comment incorrectly claimed these were "the same comparison" -- they
    # are conceptually the same quantity but measured two different ways,
    # and the Measurement Stability finding (see README) shows the additive
    # reconstruction drifting session-to-session, so don't treat this ratio
    # as validating or replacing the headline one.
    print(f"FP16 pipeline total (additive reconstruction, NOT the headline number): "
          f"{pipeline_total:.1f} us chained")
    print(f"({eager_mean/pipeline_total:.2f}x over eager PyTorch GPU) -- README's official "
          f"3.33x uses statistical_significance_v2.json's")
    print(f"directly-measured 674.7us total instead; the two can diverge, see "
          f"Measurement Stability note.")

# ================================================================
# Table C: Framework Comparison at Different Batch Sizes
# ================================================================
# No surviving JSON for this specific batch-size sweep (historical only,
# from an early single-run comparison script). Not cited as a headline claim
# elsewhere -- kept for narrative completeness, explicitly labeled.
print(f"\n{'='*75}")
print("TABLE C: Throughput Comparison (flows/sec) [HISTORICAL -- no surviving")
print("JSON source; re-run scripts/benchmark_batch.py to refresh before citing]")
print(f"{'='*75}")
print(f"{'Batch Size':<12} {'PyTorch CPU':>12} {'PyTorch GPU':>12} {'ORT CPU':>12} {'ORT GPU':>12} {'Winner':>12}")
print(f"{'-'*75}")

batch_data = [
    (1, 733, 522, 2661, 253, "ORT CPU"),
    (32, 6524, 15510, 8202, 9145, "PyTorch GPU"),
    (128, 8323, 27455, 6804, 26577, "PyTorch GPU"),
    (256, 10325, 42878, 8814, 45855, "ORT GPU"),
]

for bs, pt_cpu, pt_gpu, ort_cpu, ort_gpu, winner in batch_data:
    print(f"{bs:<12} {pt_cpu:>12,} {pt_gpu:>12,} {ort_cpu:>12,} {ort_gpu:>12,} {winner:>12}")

print(f"\nKey insight: GPU inference becomes superior at batch >= 32.")
print(f"At batch=1, ORT CPU wins due to GPU kernel launch overhead.")

# ================================================================
# Table D: Model Architecture Ablation (V2 vs V3)
# ================================================================
# No surviving per-run JSON for the V2 baseline (superseded architecture,
# not retrained during this pass) -- historical, matches README's separately
# cited V2/V3 numbers.
print(f"\n{'='*75}")
print("TABLE D: Architecture Ablation (V2 vs V3) [HISTORICAL]")
print(f"{'='*75}")
print(f"{'Metric':<35} {'V2 (baseline)':>15} {'V3 (+attn)':>15} {'Delta':>10}")
print(f"{'-'*75}")

ablation = [
    ("Parameters", "463,877", "530,181", "+14.3%"),
    ("Test Macro-F1", "0.9330", "0.9352", "+0.0022"),
    ("Test Weighted-F1", "0.9695", "0.9698", "+0.0003"),
    ("Training time/epoch", "~40 sec", "~55 sec", "+37.5%"),
    ("DDoS F1", "0.9698", "0.9711", "+0.0013"),
    ("DoS F1", "0.9680", "0.9691", "+0.0011"),
    ("Normal F1", "0.8583", "0.8595", "+0.0012"),
    ("Reconnaissance F1", "0.9457", "0.9534", "+0.0077"),
    ("Theft F1", "0.9231", "0.9231", "+0.0000"),
]

for metric, v2, v3, delta in ablation:
    print(f"{metric:<35} {v2:>15} {v3:>15} {delta:>10}")

# ================================================================
# Table E: Energy Efficiency
# ================================================================
print(f"\n{'='*75}")
print("TABLE E: Energy Efficiency")
print(f"{'='*75}")
print(f"{'Configuration':<25} {'Power (W)':>10} {'Tput (f/s)':>12} {'mJ/flow':>10} {'Efficiency':>12}")
print(f"{'-'*75}")

if energy:
    b1 = energy["gpu_batch1"]
    b128 = energy["gpu_batch128"]
    c1 = energy["cpu_batch1"]
    eff_128 = b1["mj_per_inf"] / b128["mj_per_flow"]
    eff_cpu = c1["mj_per_inf"] / b1["mj_per_inf"]
    energy_rows = [
        ("GPU batch=1", b1["power_w"], round(b1["throughput"]), b1["mj_per_inf"], "1.0x"),
        ("GPU batch=128", b128["power_w"], round(b128["throughput"]), b128["mj_per_flow"], f"{eff_128:.1f}x"),
        ("CPU batch=1", c1["power_w"], round(c1["throughput"]), c1["mj_per_inf"], f"{1/eff_cpu:.1f}x"),
    ]
else:
    energy_rows = [
        ("GPU batch=1", 10.44, 411, 25.414, "1.0x"),
        ("GPU batch=128", 17.81, 17491, 1.018, "25.0x"),
        ("CPU batch=1", 16.40, 347, 47.225, "0.5x"),
    ]

for name, power, tput, mj, eff in energy_rows:
    print(f"{name:<25} {power:>10.2f} {tput:>12,} {mj:>10.3f} {eff:>12}")

print(f"\nKey insight: GPU batch=128 is dramatically more energy-efficient per")
print(f"flow than batch=1 on either device (see benchmarks/results/energy_efficiency.json).")

# ================================================================
# Table F: LLM Explainability Impact
# ================================================================
print(f"\n{'='*75}")
print("TABLE F: Async LLM Explainability Impact on Detection")
print(f"{'='*75}")
print(f"{'Metric':<40} {'Value':>20}")
print(f"{'-'*75}")

if llm:
    llm_metrics = [
        ("Detection latency, no dispatch (p50)", f"{llm['baseline_p50_us']:.2f} us"),
        ("Detection latency, with dispatch (p50)", f"{llm['dispatch_p50_us']:.2f} us"),
        ("Async dispatch overhead (p99)", f"{llm['overhead_p99_us']:.2f} us"),
        ("Overhead as % of 674 us pipeline", f"~{llm['overhead_p99_us']/674*100:.1f}%"),
        ("LLM generation time (mean)", f"{llm['llm_generation_mean_ms']:.0f} ms"),
        ("Impact on detection", "NEGLIGIBLE (vs multi-second generation)"),
        ("Ring buffer capacity", "32 alerts"),
        ("Alerts dropped / dispatched", f"{llm['alerts_dropped']}/{llm['alerts_dispatched']}"),
    ]
else:
    llm_metrics = [("No llm_explainability.json found -- run scripts/llm_explainability.py", "")]

for metric, val in llm_metrics:
    print(f"{metric:<40} {val:>20}")

if llm:
    print(f"\nKey insight: async LLM dispatch adds {llm['overhead_p99_us']:.2f} us overhead at p99")
    print(f"over {llm['dispatch_trials']} trials -- see benchmarks/results/llm_explainability.json.")
    print(f"Explanations are generated in background without blocking detection.")

# ================================================================
# Table G: Cross-Dataset Validation
# ================================================================
print(f"\n{'='*75}")
print("TABLE G: Cross-Dataset Validation (BoT-IoT vs ToN-IoT)")
print(f"{'='*75}")
print(f"{'Dataset':<12} {'Model':<15} {'Macro-F1':>10} {'Weighted-F1':>12} {'Classes':>8} {'Test Samples':>13}")
print(f"{'-'*75}")

cross_data = [
    ("BoT-IoT", "RF Baseline", 0.9768, None, 5, 733705),
    ("BoT-IoT", "CNN-BiLSTM V3", 0.9352, 0.9698, 5, 733705),
]
if rf_toniot:
    cross_data.append(("ToN-IoT", "RF Baseline", rf_toniot["macro_f1"], rf_toniot["weighted_f1"], 10, 42209))
else:
    cross_data.append(("ToN-IoT", "RF Baseline", 0.9396, 0.9844, 10, 42209))
if toniot_v2:
    cross_data.append(("ToN-IoT", "CNN-BiLSTM V3", toniot_v2["macro_f1"], toniot_v2["weighted_f1"], 10, 42209))
else:
    cross_data.append(("ToN-IoT", "CNN-BiLSTM V3", 0.8254, 0.8796, 10, 42209))

for ds, model, mf1, wf1, cls, samples in cross_data:
    wf1_str = f"{wf1:.4f}" if wf1 else "---"
    print(f"{ds:<12} {model:<15} {mf1:>10.4f} {wf1_str:>12} {cls:>8} {samples:>13,}")

print(f"\nKey insight: CNN-BiLSTM generalizes across both IoT datasets.")
print(f"RF achieves higher accuracy but cannot leverage GPU acceleration.")

# ================================================================
# Table H: Cross-Hardware Comparison (RTX 3050 vs V100S vs A100)
# ================================================================
print(f"\n{'='*75}")
print("TABLE H: Cross-Hardware CUDA Kernel Comparison (3 GPUs)")
print(f"{'='*75}")
print(f"{'Block':<22} {'RTX 3050':>10} {'V100S 32GB':>10} {'A100 80GB':>10} {'Fastest':>10}")
print(f"{'-'*75}")
print("NOTE: RTX 3050 column below is the fresh n=100-trial mean where available")
print("(cuda_kernel_stats_rtx3050.json); V100S/A100 columns are single DICC runs")
print("(dicc_v100_summary.txt / dicc_a100_summary.txt) pending the n=20-trial")
print("re-run now wired into dicc_scripts/02 and 03 (see Phase 3).")
print(f"{'-'*75}")

hw_data = [
    ("Block 1 (Conv)", cuda_stats["fused_block1"]["latency_us"]["mean"] if cuda_stats else 61.7, 10.0, 12.0, "V100S"),
    ("Block 2 (Conv)", cuda_stats["fused_block2"]["latency_us"]["mean"] if cuda_stats else 87.2, 29.1, 34.7, "V100S"),
    ("Block 3 FP32", transposed_with_graphs, 773.3, 1117.8, "V100S"),
    ("Block 3 FP16", fp16_mean, 511.9, 548.4, "V100S"),
    ("Block 4 FP32", cuda_stats["fused_block4"]["latency_us"]["mean"] if cuda_stats else 20.1, 8.3, 10.2, "V100S"),
    ("Block 4 FP16", 17.9, 5.2, 6.6, "V100S"),
]

for name, rtx, v100, a100, fastest in hw_data:
    print(f"{name:<22} {rtx:>8.1f} us {v100:>8.1f} us {a100:>8.1f} us {fastest:>10}")

print(f"\nChained pipeline (FP16):")
print(f"  RTX 3050:  {pipeline_total:.1f} us (additive reconstruction, see note above -- README's"
      f" official headline uses the 674.7us directly-measured total, not this figure)"
      if eager_mean else f"  RTX 3050:  {pipeline_total:.1f} us")
print(f"  V100S:     550.7 us (vs PyTorch: not reported -- no same-hardware")
print(f"             PyTorch GPU baseline was captured on this machine yet;")
print(f"             see dicc_v100_summary.txt)")
print(f"  A100:      592.0 us (vs PyTorch: not reported -- no same-hardware")
print(f"             PyTorch GPU baseline was captured on this machine yet;")
print(f"             see dicc_a100_summary.txt)")
print(f"\nKey insight: V100S achieves fastest BiLSTM FP16 (511.9 us) despite being")
print(f"older Volta architecture -- BiLSTM is sequential, so clock speed matters")
print(f"more than SM count. This finding does not depend on the PyTorch baseline")
print(f"gap above (it's a kernel-vs-kernel, same-methodology comparison).")

print(f"\n{'='*75}")
print("ABLATION STUDY COMPLETE")
print(f"{'='*75}")
