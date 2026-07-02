# COLIDE — Session Handoff

**Last session:** 2026-07-02 (Claude Sonnet 5, session 3, in progress). **Read this whole file
before doing anything else** — it has the full context needed to continue without re-deriving
what's already been established. **Open item #0 (fabricated Sophimatics citation) is now RESOLVED
— see "Session 3 progress" below before assuming it's still open.**

## Session 3 progress (2026-07-02)

**Open item #0 — fabricated Sophimatics citation — RESOLVED.** Removed the false claim from
`README.md` and `docs/paper_text_blocks.md` §14 immediately (before searching for a replacement,
so the repo was never left citing something known-false). Searched for a real closest-prior-work
replacement (8 WebSearch queries covering CNN-LSTM-IDS, FPGA-IDS, and general CUDA-vs-framework
angles) and found one: **Ibrahim, Paolini, Cugini, Paolucci, "From packets to predictions on GPU:
Accelerated graph-based intrusion detection system," *Computer Networks*, vol. 275, 2026, DOI
`10.1016/j.comnet.2025.111954`.** Verified two ways this time (unlike the original, which was
never checked at all): metadata confirmed via the Crossref API and cross-checked by resolving the
DOI to the same ScienceDirect article ID (`S1389128625009193`) found independently via topical
search; content (custom CUDA kernels for GNN graph construction/inference, memory coalescing,
shared memory, 1.22x-1.48x vs. a CPU baseline) corroborated by two independent search queries.
**Caveat, flagged in the manuscript text itself:** the actual abstract PDF could not be fetched
(ScienceDirect blocks automated retrieval, 403), so this rests on corroborated secondary
characterization, not a verbatim primary-source quote — worth a manual read of the real PDF before
final submission if full certainty is wanted. The citation is honestly differentiated from our
work rather than oversold: it's GNN (not CNN-BiLSTM) and benchmarks vs. CPU (not production
frameworks) — see `docs/paper_text_blocks.md` §14 and README's Verified Research Gaps #1 for the
exact framing used. Added a `verify_claims.py` regression guard on the fabricated claim's exact
phrasing so it can't silently reappear. `scripts/verify_claims.py` passes all 63 claims, 0
regressions, after this change.

**Next up in this session:** working through the remaining open items (below) one by one with the
user, in whatever order they prioritize.

## Mandate (set by Ibteshamul, applies to all future sessions)

- **Target: Q1 journal publication.** PI/supervisor is Prof. Dr. Por Lip Yee (FCSIT, Universiti
  Malaya). README currently cites FGCS as the realistic target, IoT-J as a stretch.
- **We are aiming for perfection, even knowing it's not fully reachable.** There is a lot of time
  before submission — take it. Micro-gains matter and are worth pursuing (e.g. the naive-kernel
  numerical issue below is low-stakes but still worth eventually fixing properly, not just
  disclosing and moving on forever).
- **No room for error.** Every number that goes into the manuscript must trace to a real,
  reproducible measurement. Don't restate a number because it "looks right" — check it against
  `benchmarks/results/*.json` or run `scripts/verify_claims.py`.
- **Future sessions will also try to improve substantively weak numbers**, not just verify
  existing ones — e.g. the accuracy gap vs the Random Forest baseline (was 2.25% on BoT-IoT,
  session 2 got it to 0.74% — see open item #5) is a candidate for actual improvement (better
  distillation recipe, more KD sweep points, etc.), not just accurate reporting.
- **Check in with the user after each phase/major fix**, don't run long stretches silently —
  this was explicitly requested and worked well last session.

## Why this work exists

A full-repo audit (including every gitignored file: `benchmarks/results/*.json`,
`data/`, `model/*.pth`) found that several headline numbers in `README.md` and
`docs/paper_text_blocks.md` were fabricated, stale, or had inconsistent provenance. Two were
critical (would not survive a competent review): a "5.19us p99" LLM dispatch overhead that had
never actually been computed anywhere (hardcoded placeholder, 25x off from the one real
measurement that existed), and a "674us / 2.76x" pipeline speedup ratio built on an unsourced
constant that didn't match any other PyTorch-GPU measurement in the repo. A phased roadmap was
written and approved to fix these before manuscript writing begins. The roadmap plan file is at:

`/home/titoisalive/.claude/plans/floating-coalescing-feigenbaum.md`

(That plan file still has the original Phase 1-5 breakdown, useful as a reference, but this
HANDOFF.md is the current source of truth for what's actually done vs. pending — some things
were resolved differently than originally planned, described below.)

## What's done (Phases 0-2, all complete)

**Phase 0 — `scripts/verify_claims.py` (built, working, use this constantly).**
Loads every `benchmarks/results/*.json`, defines a manifest of claims with their source and
expected value, and checks README.md/docs/paper_text_blocks.md against them. Also has
`REGRESSION_GUARDS` (a list of banned strings from fixed bugs, e.g. "5.19 us", "2.76x") so a
fixed number can never silently reappear. Run it after ANY edit to README/docs/benchmark
scripts:
```
source .venv/bin/activate && PYTHONPATH=. python3 scripts/verify_claims.py
```
Current status (end of session 2): **all 63 tracked claims pass, 0 regressions.** It also prints
"orphan numbers" (bolded figures in README not yet covered by a manifest entry) — currently:
`0.6, 0.9526, 1.00x, 10.0, 2.0, 3.3%, 675, 87`. These aren't necessarily wrong, just not yet
cross-checked; add manifest entries for them opportunistically.

**Phase 1.1 — Fixed the fabricated LLM dispatch overhead.**
`scripts/llm_explainability.py` was rewritten with a proper `benchmark_dispatch_overhead()`
function: 5,000 trials of the real classify+construct+push code path, computing actual
p50/p95/p99 via `np.percentile`, decoupled from LLM generation timing (which is a separate,
multi-second-scale measurement). Real result: **16.60us p99** (was fabricated as "5.19us p99").
Propagated everywhere: README.md (abstract, key contributions, results table, verified-gaps
section), `docs/paper_text_blocks.md`, `scripts/ablation_study.py`, `CLAUDE.md`.

**Phase 1.2 — Fixed the pipeline-total/speedup-ratio provenance, and found a bigger bug while
doing it.**
The "674us total" itself turned out to be legitimate (real methodology, matches the same
additive convention used in the DICC summary files — see below). But the "PyTorch GPU baseline"
used to compute the "2.76x" ratio was a hardcoded, unsourced constant (`1864.0` in
`inference/kernels/fused_pipeline.cu`) that matched none of the three other "PyTorch GPU total"
numbers recorded elsewhere in the repo. **Bigger finding:** the V100S (3.39x) and A100 (3.15x)
"vs PyTorch" ratios all divided out to that SAME ~1864-1867 constant — meaning they were
comparing V100S/A100 custom-CUDA kernels against the RTX 3050's PyTorch baseline, a
cross-hardware apples-to-oranges comparison. Fixed:
- RTX 3050: realized "674us vs PyTorch" and "Custom CUDA vs eager PyTorch" (the framework
  comparison table's headline number) are literally the same comparison. Consolidated to
  **3.33x** (was inconsistently 2.76x in one place, 3.33x in another).
- V100S/A100: removed the false ratio. README's cross-hardware table now reports real absolute
  latencies with an explicit footnote that no same-hardware PyTorch baseline exists yet for
  those machines (see Phase 3, below — this is exactly what the DICC re-run needs to produce).
- Fixed `scripts/benchmark_pipeline.py` to tag its output JSON with the GPU name
  (`pipeline_benchmark_<gpu_tag>.json`) instead of a fixed filename, so running it on V100S/A100
  via DICC won't silently overwrite the RTX 3050 result the way it apparently did before (this
  is *why* no per-hardware PyTorch baseline existed).

**Phase 2.3 — Fixed a real transcription error.**
KD sweep table, row alpha=0.5/T=3.0: Val F1 was listed as 0.9487, source JSON
(`distill_botiot_a0.5_T3.0.json`) says `best_val_f1 = 0.9541`. Fixed in both README.md and
`docs/paper_text_blocks.md` (it was duplicated in a second table there too).

**Phase 2.4 — Rebuilt `scripts/ablation_study.py` to load from JSON instead of hardcoded
literals.**
Also caught another silent bug this way: Table G's ToN-IoT V3 numbers were stale
(0.8029/0.8622) against the canonical 0.8254/0.8796 already used elsewhere — now consistent.
The rebuilt script clearly labels which numbers are freshly loaded vs. historical-only (see
"Open items" below for what's still historical).

**Phase 2.5 — Fixed cuML "(A100)" table hardware-mixing.**
The table's CNN-BiLSTM throughput (25,410 f/s) was actually the RTX 3050 number, sitting next
to A100 cuML-RF numbers under an "(A100)" header. Now uses the real A100 CNN-BiLSTM throughput
(87,791 f/s, derived from `a100_energy.json`'s batch timing). Also picked a canonical streaming
throughput source (25,899 from `streaming_throughput.json`, the purpose-built benchmark) over a
~2%-different incidental measurement from a different script (`cuml_rf_native.json`, 25,410) —
both are real, just picked one as canonical everywhere.

**Phase 2.6 — Re-exported CUDA kernel weights from the actual final model.**
`scripts/validate_weights.py` and `scripts/validate_real_weights.py` were pointed at
`model/best_model.pth` — a stale June 13 checkpoint (the pre-distillation "Original V3" model,
0.9352 macro-F1), not the actual final published model
(`model/best_model_botiot_twostage.pth`, 0.9639 macro-F1). Fixed both scripts, re-ran the
export (`model/weights_bin/` and `model/weights/` are now current), and re-ran the correctness
validation: classification accuracy on 1000 samples is now 0.979 (up from a stale 0.968 tied to
the old checkpoint, and consistent with the real model's known ~0.979 accuracy).

**Phase 2.7 — Fixed the statistical methodology, not just disclosed it.**
`benchmark_stats_v2.py` was doing a one-sample t-test against a fixed Custom-CUDA constant
(674.0, no variance). Built `scripts/benchmark_cuda_kernels_stats.py` — a proper N-trial harness
for the compiled CUDA kernel binaries (parses stdout, computes mean/std/percentiles, saves
JSON). Ran it at n=100 for the RTX 3050 locally (see "Key finding" below), then patched
`statistical_significance_v2.json`'s "Custom CUDA FP16" entry to have a real mean/std/n derived
from that data, and switched the significance tests to two-sample Welch's t-tests
(`scipy.stats.ttest_ind_from_stats`). **The corrected mean (674.70) was within 0.1% of the old
674.0 constant**, so no headline ratios needed to change — only exact p-values shifted slightly
(same conclusions throughout). Two ratios crossed a rounding boundary and got corrected:
torch.compile 2.64x→2.63x, ORT GPU 6.90x→6.89x.

### Key finding from the n=100 re-benchmark (important context, don't re-derive this)

Running every CUDA kernel binary 100 times (not just once) showed real run-to-run variance of
5-20%+ CV on this WSL2/RTX 3050 dev box (some blocks — block2, block4, the b124-chained
measurement — show 40-100%+ CV, almost certainly WSL2 scheduling/virtualization jitter, not real
kernel behavior; visible as rare extreme outliers in the max values). At n=20 this looked
alarming (one run showed the pipeline total 5% below the headline number). **At n=100 the mean
converged back almost exactly to the original historical numbers** (601.65us vs 601.4us headline
for Block 3 FP16; 674.70us vs 674us for the derived pipeline total). Conclusion: the original
headline numbers were legitimate; a single run or small-N sample on this hardware is just noisy.
Lesson for future sessions: **any new CUDA kernel benchmark should use n>=50-100 trials**, not a
single run or n=20, because of this environment's noise floor.

`scripts/benchmark_cuda_kernels_stats.py` supports both local dev use and DICC use:
```bash
# Local (binaries suffixed _official, must be compiled without -O3 to match the Dockerfile):
PYTHONPATH=. python scripts/benchmark_cuda_kernels_stats.py --tag rtx3050 --n-trials 100

# DICC (already wired into dicc_scripts/02 and 03, see Phase 3 below):
PYTHONPATH=. python scripts/benchmark_cuda_kernels_stats.py --kernels-dir inference/kernels/v100 --suffix "" --tag v100s
```

## Open items — decisions not yet made, flagged rather than silently resolved

0. **CRITICAL, NOT YET FIXED — a citation in the manuscript text blocks describes a real paper's
   content as completely fabricated.** `docs/paper_text_blocks.md` §14 ("Sophimatics Phase 3
   Citation Note") and README.md's "Verified Research Gaps" §1 cite "Sophimatics Phase 3"
   (Applied Sciences 2025) as prior work claiming "custom CUDA kernels for a CNN-based IDS
   achieving 2.7x speedup," used as the closest-prior-work comparator (4.40x vs their claimed
   2.7x, etc.). **Verified via WebSearch + WebFetch this session: the paper is real and correctly
   titled/dated (DOI `10.3390/app152211876`), but it has NOTHING to do with CUDA, CNNs, or
   speedup benchmarking.** The actual paper — "Super Time-Cognitive Neural Networks (Phase 3 of
   Sophimatics): Temporal-Philosophical Reasoning for Security-Critical AI Applications" — is
   about a philosophical/temporal-cognitive AI architecture (complex-valued "time" representing
   memory/present/imagination), evaluated across five unrelated security domains using
   detection-rate/false-positive metrics (96.3% detection, 2.1% FP for its IDS use case specifically
   — no latency, no CUDA, no CNN). The citation note's own text says "ChatGPT identified a
   comparable paper... Search for full citation" — meaning this was **never actually verified**,
   just carried forward as fact. This is more severe than any numeric provenance issue fixed this
   session: a reviewer who looks up this citation (trivial, it's a real indexed paper) would find
   it describes something unrelated, which reads as fabrication or serious carelessness either way.
   **User explicitly deferred fixing this to session 3** (to avoid overloading session 2's context
   window with a tangential fix while the KD sweep was running) — **not skipped, just deferred.**
   Three options were on the table, undecided: (a) remove the comparison now, find a real
   closest-prior-work citation later; (b) do a proper literature search now for an actual paper
   about custom CUDA kernels for CNN/RNN-based IDS; (c) remove entirely, no replacement — the
   paper's contribution (statistically validated speedups, torch.compile crash finding) stands on
   its own without a named comparator. Recommend starting session 3 with a decision on this, then
   executing it — it blocks manuscript submission regardless of which option is chosen.

1. **RESOLVED 2026-07-01 (session 2).** PyTorch cuDNN baseline for Block 3 used to be
   inconsistent (740.7us historical single-run vs 943.6us from a fresh `benchmark_pipeline.py`
   run). Built `scripts/benchmark_pytorch_block3_stats.py` — mirrors the CUDA kernel statistical
   harness's approach (N independent subprocess trials, not intra-process repeats, so it captures
   the same cross-process/GPU-state jitter) — and ran n=50 trials locally. Real result:
   **mean 784.1us, std 88.6us, CV 11.3%** (`benchmarks/results/pytorch_block3_stats_rtx3050.json`),
   sitting between the two old single-run numbers as expected for a noisy quantity now properly
   characterized. This is the stable half of the ratio; see item 2 below for how the OTHER half
   (the CUDA kernel side) turned out to need a range, not a point estimate.

2. **RESOLVED 2026-07-01 (session 2) — the naive Block 3 kernel's failure was a genuine data
   race, not FP32 rounding, and it's now fixed.** The original diagnosis ("accumulated FP32
   rounding error over its unoptimized summation order") was wrong: re-running the SAME
   `srand(42)`-seeded binary repeatedly produced *different* GPU output each time (e.g. index 3
   read -0.0502, -0.0847, and -0.0774 across three separate runs) — impossible for deterministic
   rounding-order error, only possible with an actual race. `compute-sanitizer --tool racecheck`
   confirmed it: the per-timestep hidden-state write (`s_h_prev[h] = h_val`) raced against the next
   timestep's read of it (`s_h_prev[j]`), despite an intervening `__syncthreads()` — thousands of
   hazards reported. `synccheck` found no barrier misuse (so it wasn't a missing-sync bug); fixed
   by double-buffering the hidden state in `fused_block3_naive.cu` (alternate shared arrays by
   `t%2` so a timestep's read and write never target the same location). Verified: **0 hazards
   under racecheck** (was thousands), **100/100 runs pass** at the standard 1e-2 tolerance (was
   ~6/30), and **20/20 pass at a much tighter 1e-5 tolerance** — genuinely close to the CPU
   reference, not just clearing a loose bar. The naive kernel's latency is now a real n=100-trial
   mean of the fixed kernel (**5,050us**, replacing the old 5,698us historical single-run figure).
   Recompiled and **committed the tracked binary** for it for the first time (previously
   source-only, kept unbuilt because it was known-broken; now built like the other 6 kernels).
   Propagated to `README.md` ("Naive Kernel Fix" subsection, was a disclosed limitation, now
   marked resolved), `scripts/ablation_study.py`, and `scripts/verify_claims.py` (new
   `block3_naive_latency` claim, regression guards on the superseded `5,698` and `9.47x` figures).

   **Getting this required a full re-run of `benchmark_cuda_kernels_stats.py` for ALL 7 kernels**
   (not just the naive one) — otherwise re-running with only the naive binary present would have
   silently overwritten `cuda_kernel_stats_rtx3050.json` and lost the other 6 kernels' data. That
   fresh n=100 run surfaced a bigger, unplanned finding:

   **NEW — Measurement Stability finding.** The fresh re-run's means for the transposed-W_hh and
   FP16 configs disagreed meaningfully with the SAME n=100-trial harness run earlier the SAME day
   (session 1) — despite each individual session's own internal CV looking tight (6.8%-24.4%):
   Block3 FP16 601.65us -> 548.34us (-9%), transposed no-graphs 803.91us -> 1022.62us (+27%),
   transposed with-graphs 788.52us -> 904.92us (+15%). This means within-session CV understates
   true measurement uncertainty on this WSL2 dev box — there's real session-to-session drift
   (thermal state / background load / WSL2 scheduler) that no single n=100 run captures, however
   tight its own std looks. **Decision (user, 2026-07-01): report both sessions' means as an
   explicit range rather than picking one.** Done throughout `README.md` (Key Contributions #2,
   Per-Block Performance table, Block 3 Optimization Progression section — now
   **8.39x-9.21x** progression, **1.30x-1.43x** beating cuDNN, was a single "9.47x"/"1.30x"),
   `docs/paper_text_blocks.md` (same range, was a stale "9.48x" that had already drifted from
   README's old "9.47x" even before this finding), `CLAUDE.md`, `scripts/ablation_study.py` (prints
   both sessions' contribution-percentage breakdowns side by side to make the instability
   concrete), and `scripts/verify_claims.py` (range-based manifest claims, regression guards on
   the superseded point estimates). Also fixed a related latent bug this surfaced:
   `ablation_study.py`'s Table B/H printed an "FP16 pipeline total... same comparison as the
   framework-comparison table" claim that was **not actually the same comparison** (it's an
   additive reconstruction — mean b124_chained + mean block3_fp16 — vs. README's directly-measured
   674.7us total from `statistical_significance_v2.json`; these are two different measurement
   methodologies for conceptually the same quantity and can diverge, confirmed this session:
   614.5us derived vs. 674.7us directly-measured). README's official 674.7us/3.33x headline was
   NOT affected (it never used the derived total), but the script's comment was corrected so it
   doesn't mislead a future reader into thinking the two numbers are interchangeable.
   `scripts/verify_claims.py` passes all 49 claims, 0 regressions.

   **Open follow-on question, not investigated this session:** is `statistical_significance_v2.json`'s
   674.7us full-pipeline figure (the one actually backing README's "3.33x over eager PyTorch"
   headline) ALSO subject to this same session-to-session drift? It wasn't re-measured this
   session (only the per-block breakdown was), so there's no evidence either way — flagging as an
   open question rather than assuming it's fine or assuming it's broken.

3. **Phase 3 (DICC re-run) needs the user to actually execute it** — no SSH/cluster access from
   this dev environment. `dicc_scripts/01_setup.sh`, `02_benchmark_v100.sh`, and
   `03_benchmark_a100.sh` are all updated and ready:
   - `01_setup.sh` now also compiles `fused_block3_naive` and `fused_pipeline` for V100/A100
     (previously only 5 of 7 binaries were compiled there — `fused_pipeline` was apparently
     compiled manually at some point since `dicc_v100_summary.txt`/`dicc_a100_summary.txt`
     already have pipeline-chained numbers, but the setup script itself had a gap, now closed).
   - `02_benchmark_v100.sh` / `03_benchmark_a100.sh` now run
     `scripts/benchmark_cuda_kernels_stats.py` (n=20 trials) for real statistical backing on
     those platforms, saving to `cuda_kernel_stats_v100s.json` / `cuda_kernel_stats_a100.json`.
   - They also still run `scripts/benchmark_pipeline.py`, which (thanks to the Phase 1.2 fix)
     will now save a hardware-tagged `pipeline_benchmark_<gpu>.json` instead of clobbering the
     RTX 3050 result — **this is what will finally give V100S/A100 a real same-hardware
     PyTorch-GPU baseline**, resolving the Phase 1.2 "n/a**" footnote in README's cross-hardware
     table.
   - **NEW instruction from the user (2026-07-01), given the local Measurement Stability finding
     above:** don't run the DICC n-trial harness as a single sitting. Submit the same benchmark
     via `sbatch` across **at least two separate submissions on different days** and compare —
     check whether V100S/A100 show the same kind of session-to-session drift found locally on
     this WSL2 box. If DICC (native Linux, no WSL2 passthrough) is stable across sessions, that's
     good evidence the local variance is a WSL2-specific artifact rather than a fundamental limit
     of the methodology — worth stating explicitly in the paper either way. If DICC shows the same
     instability, that's an even bigger methodology finding deserving its own disclosure section.
   - **Next session, once the user has run these on DICC:** pull the resulting JSON files back
     into `benchmarks/results/`, add manifest entries to `verify_claims.py` for the V100S/A100
     PyTorch baselines, compute the real "vs PyTorch" ratios for those platforms, update
     README's cross-hardware table to replace the "n/a**" footnote with real numbers, and update
     `dicc_v100_summary.txt`/`dicc_a100_summary.txt`-style summaries if useful.

4. **Phase 4 (optional ceiling-raising items, not started, no urgency)** — from the original
   plan, still relevant: batch-size sweep for TensorRT/torch.compile/ORT (currently batch=1
   only), explicit TensorRT fairness documentation (partially already done — see
   `docs/paper_text_blocks.md` section 9, "TensorRT Build Configuration," which already
   discloses no INT8 calibration / no manual CUDA graph capture), a numerical-fidelity table
   (max abs/relative error per block vs. PyTorch reference, pairs with item 2 above), narrative
   reframing to lead with the torch.compile crash finding rather than raw speedups, and a
   threats-to-validity section. **The Measurement Stability finding in item 2 above is a
   ready-made, concrete candidate for that threats-to-validity section** — it's a real,
   quantified (6-27% session-to-session drift) hardware/environment limitation, not a
   hypothetical one.

5. **SUBSTANTIAL PROGRESS 2026-07-01/02 (session 2): RF accuracy gap cut from 2.25% to 0.74%.**
   Plan file (approved, followed): `/home/titoisalive/.claude/plans/nested-dancing-kazoo.md`.

   **Step 1 — fixed two provenance gaps in the numbers the whole task is framed around, before
   touching the actual ML work** (found while double-checking the "apples-to-apples with RF"
   framing at the user's prompting):
   - **README's 0.9864 RF figure had no traceable source.** Checked `DAILY_LOG.md` (no mention),
     `scripts/rf_baseline.py` as-is (100 trees, independent undersample+SMOTE recipe -> 0.9768
     test, which IS the well-documented Day-3 windowing-decision number, just a different RF), a
     200-tree variant of that recipe (-> 0.9730, *lower* — ruled out "just add more trees"), and
     `train_distill.py`'s inline 200-tree RF teacher with full-SMOTE (-> 0.9750 validation). None
     matched. **User supplied the original ad-hoc terminal command**: a 200-tree RF
     trained/evaluated directly on `data/processed/*.npy` — the SAME preprocessed splits the
     CNN-BiLSTM itself trains/evaluates on (the methodologically correct apples-to-apples
     comparison, unlike the other two scripts' independent resampling). Reproduced byte-for-byte:
     0.9864 exactly. Saved permanently as `scripts/rf_baseline_processed.py` ->
     `benchmarks/results/rf_baseline_processed.json` (was an unrepeatable terminal one-liner).
   - **The final two-stage headline number (then 0.9639) ALSO had no JSON source** —
     `train_twostage.py` never saved one, `verify_claims.py` had it as a second hand-typed
     literal. Fixed: `train_twostage.py` now saves `benchmarks/results/twostage_botiot.json`;
     `verify_claims.py`'s `rf_gap_botiot_final` claim now loads both sides from JSON instead of
     two hardcoded numbers.
   - **Verified the apples-to-apples framing itself is sound** (user asked directly, "is there a
     catch hiding?"): confirmed RF and CNN-BiLSTM are scored on byte-identical test rows (same
     733,705-row CSV, zero NaN/inf, no filtering differences) with near-identical MinMax scaling
     (compared the two scalers' actual fitted min/max directly — differences are noise-level, e.g.
     "mean" feature 4.9804 vs 4.9819, ~0.03%, from undersampling removing a handful of rows before
     fitting). The gap is real, not a measurement artifact.

   **Step 2 — Phase A: extended the KD temperature/alpha sweep.** Round 1 (existing, sparse: 6
   points) had shown T=1->3->5 monotonically helping at alpha=0.7, never pushed past T=5. Ran 6
   new configs (`scripts/train_distill.py --focal-gamma 2.0`, suffix per config), T in {7.0,
   10.0} x alpha in {0.6, 0.7, 0.8}, sequentially in the background (~35-70 min each: RF teacher
   +~204-212s/epoch, up to 50 epochs with patience=10):

   | Config | Val F1 | Test F1 | Note |
   |---|---|---|---|
   | a=0.6, T=7.0 | 0.9780 | 0.9702 | |
   | a=0.7, T=7.0 | 0.9728 | 0.9687 | |
   | a=0.8, T=7.0 | 0.9751 | 0.9757 | |
   | a=0.7, T=10.0 | 0.9482 | **0.9033** | Outlier — Normal/Theft precision collapsed (0.75/0.67) despite DDoS/DoS/Recon all >=0.97 F1. Real finding, not noise: macro-F1 weights all 5 classes equally, so 2 tiny classes (107 + 14 test samples) can swing the average 6-7 points even when 99%+ of traffic is classified excellently. Kept in the sweep table as a useful negative result. |
   | a=0.8, T=10.0 | 0.9672 | 0.9745 | |
   | **a=0.6, T=10.0** | **0.9757** | **0.9763** | **Winner — fed into Phase B** |

   All 6 result JSONs saved to `benchmarks/results/distill_botiot_a<X>_T<Y>_focal2.json`, all 6
   checkpoints saved to `model/best_model_botiot_distill_a<X>_T<Y>_focal2.pth` (both committed).

   **Step 3 — Phase B: two-stage fine-tuned the winner.** `scripts/train_twostage.py
   --checkpoint model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` (same recipe as before:
   real data, no SMOTE, focal_gamma=2.0, up to 10 epochs, patience=3). Result: **early stopped at
   epoch 6 (best val F1 0.9780 at epoch 3), final test macro-F1 = 0.9790**
   (`benchmarks/results/twostage_botiot.json`). Per-class: DDoS 0.9832, DoS 0.9805, Normal 0.9358,
   Reconnaissance 0.9955, **Theft 1.0000** (was the weakest class before, now perfect).

   **IMPORTANT — `train_twostage.py` hardcodes its save path to
   `model/best_model_botiot_twostage.pth` (no suffix flag)**, i.e. it overwrites the production
   checkpoint on every run. Backed up the pre-session champion before running Phase B:
   `model/best_model_botiot_twostage_BACKUP_0.9639.pth` (verified byte-identical via md5sum before
   the overwrite, and confirmed it still loads as a valid state dict). **If a future session runs
   `train_twostage.py` again, back up the current best first** — this script will silently
   clobber whatever's there.

   **Result: RF gap 2.25% -> 0.74%** (0.9864 - 0.9790), a 67% relative reduction. Propagated
   everywhere: README.md (abstract, Key Contributions #3, Detection Accuracy table, KD Sweep
   table — now 14 configs, MLP Ablation table, cuML comparison table, Limitations),
   `docs/paper_text_blocks.md` (MLP ablation para, Summary Table x8 rows, KD Sweep Documentation
   table — now 14 rows, RF Defense para), `scripts/verify_claims.py` (new `twostage_final_test_f1`
   claim + 6 new `kd_sweep_*` claims for the round-2 configs, regression guards on the superseded
   "0.9639 vs 0.9864" and old abstract phrasing). `scripts/verify_claims.py` passes all 63 claims,
   0 regressions.

   **Not yet re-run after the model change (do this before citing anything CUDA-kernel-accuracy
   related):** `model/weights/` and `model/weights_bin/` (exported via
   `CNNBiLSTM.export_weights()`) and the validation numbers in Phase 2.6 above (0.979 accuracy on
   1000 samples) all still reflect the OLD 0.9639 checkpoint. Per this repo's own established rule
   ("if you retrain the model, re-export weights before re-benchmarking the kernels"), re-run
   `scripts/validate_weights.py`/`scripts/validate_real_weights.py` against the new
   `model/best_model_botiot_twostage.pth` before any future claim ties CUDA kernel correctness to
   "the" model's accuracy.

   **RESOLVED — reproducibility check confirmed exact.** To address a fair question raised at
   end of session ("is this result stable, or could it drift like the CUDA latency numbers
   did?"): `train_distill.py` has `cudnn.deterministic=True`, `cudnn.benchmark=False`, and full
   seeding (`random`/`numpy`/`torch`/`cuda`, all `SEED=42`) — a materially different reliability
   story than the CUDA kernels' wall-clock timing (no seed controls a physical clock/OS
   scheduler; a deterministic, seeded training computation should reproduce, and empirically
   should be checked rather than just asserted). Re-ran the winning config with a new suffix
   (`distill_botiot_a0.6_T10.0_focal2_repro.json`) to verify: **all 35 epochs matched the
   original run's train loss/train F1/val F1 to full printed precision** (only wall-clock
   per-epoch timings differed, as expected), early stopping triggered at the identical epoch
   (35), and the final test evaluation JSON is **byte-for-byte identical** to the original --
   `macro_f1=0.9763373489452476`, `weighted_f1=0.9814373060928923`,
   `accuracy=0.9814298662268895`, `best_val_f1=0.9756881295269905`, all matching exactly. Went
   one level deeper than the metrics: the saved `.pth` files' raw byte checksums differed (a red
   herring -- that's just `torch.save`'s zip-container metadata, e.g. timestamps), but loading
   both and comparing every parameter tensor with `torch.equal()` showed **all weights are
   bit-identical**, not merely close. This is real, checked evidence (not just "the flags are
   present so it should be fine") that the KD sweep winner is a fully deterministic result down
   to the last bit, not a lucky run. Deleted the duplicate repro checkpoint after confirming
   identity (redundant ~2MB, no information lost -- the JSON result stays as the evidence
   record). **Scope note: this
   confirms Stage 1 (KD) reproducibility only.** `train_twostage.py` has the same determinism
   flags so Stage 2 (fine-tuning) is *likely* similarly reproducible, but that specific claim
   hasn't been empirically tested — if it matters before submission, re-run it the same way
   (back up whatever's current in `model/best_model_botiot_twostage.pth` first, see above).

   **Next steps for further improvement (not started, Phase C in the plan file), roughly in order
   of expected payoff:**
   - **Target the minority classes directly**, since the outlier run proved the gap concentrates
     there, not in majority-class performance. Try focal_gamma sweep (1.0, 3.0, 4.0 — only 2.0 has
     been tried), or explicit per-class loss weighting.
   - **Strengthen the RF teacher on the same training-data pipeline** (more trees — currently only
     200, untuned; `class_weight='balanced'`; depth tuning) so the student distills from better
     soft labels, without confounding the ablation by also changing the data pipeline.
   - **Fix and re-tune the ensemble teacher.** `train_ensemble_distill.py` has a diagnostic-only
     bug (line ~105: compares `len(probs)` — the *training*-set size — against `len(y_val)`, so
     the printed "Ensemble teacher Val F1" is actually solo-RF's score, not the ensemble's;
     harmless to the actual KD loop, which correctly uses train-set probs, but misleading to read).
     The ensemble (RF+XGB+LGB, equal 1/3 weighting) underperformed the single RF teacher (0.9529
     vs 0.9601 in round 1) — untuned, worth a weighted-combination retry favoring RF.
   - Consider whether Round 2's winning region (T=10.0, alpha~0.6) should be swept *further*
     (T=12, 15?) given the trend wasn't obviously plateauing, balanced against the a=0.7/T=10.0
     outlier showing this region isn't uniformly safe either.

## Git state — session 2 committed locally, NOT pushed yet

Session 1 ended at `4dad75f` (committed and pushed). Session 2 added 6 more commits on top,
**committed locally but not yet pushed to `origin/master`** (user explicitly held the push at
end of session) — push is a user decision, not made automatically. Working tree is clean.
Commit range `4dad75f..547e895`, oldest to newest:

```
d85271c fix: resolve ambiguous PyTorch cuDNN baseline for Block 3 with real n=50-trial benchmark
3eb773a fix: naive Block3 kernel had a genuine data race, not FP32 rounding error
acdcba5 fix: trace README's uncorroborated 0.9864 RF figure to a real, reproducible script
f98bf33 feat: extend KD temperature sweep, cut RF accuracy gap from 2.25% to 0.74%
f5d090f docs: update HANDOFF.md for session 2 close-out
547e895 fix: update stale 0.9639/0.9601 comparison strings, fix pre-existing syntax error
```

(`547e895` also fixed three unrelated, pre-existing syntax errors in
`scripts/benchmark_cuml_rf_native.py` — literal unescaped newlines inside string literals that
would have crashed the script immediately on any run, predating this session; found during a
final repo-wide `py_compile` sweep of every script in `scripts/`, `preprocessing/`, and `model/`.)

Each commit message has full detail on that piece of work — read `git show <hash>` if you need
the exact reasoning instead of re-deriving it. Session 1's note about `benchmarks/results/*.json`
still applies: new result JSONs (e.g. `cuda_kernel_stats_rtx3050.json`, `twostage_botiot.json`,
the 6 new `distill_botiot_a*_focal2.json` sweep results) are NOT committed — gitignored by policy
("keep plots, not raw data"). **Model checkpoints (`model/*.pth`) ARE tracked and were
committed** in `f98bf33` (7 new/changed `.pth` files, ~15 MB) — this repo's established
convention, consistent with all prior checkpoint commits.

**Next session should ask the user whether to push** these 4 commits (and any new ones) before
or after fixing open item #0 (Sophimatics citation) — the user's call, not a default action.

**Carried over from session 1, still not done:** the *tracked* binary `inference/kernels/fused_pipeline`
was already recompiled+committed in session 2's `d85271c`/`3eb773a` work along with
`fused_block3`/`fused_block3_fp16` (all three needed recompiling after their `.cu` sources
changed) and a new tracked binary was added for `fused_block3_naive` (previously source-only,
now fixed and verified) — this item from session 1 is now fully resolved, no longer carried over.

## Quick orientation for a fresh session

- Read `CLAUDE.md` first (architecture overview, mostly still accurate — updated this session for
  the Block 3 naive-kernel fix and the measurement-stability range finding).
- **Open item #0 (fabricated Sophimatics citation) is RESOLVED as of session 3** — see "Session 3
  progress" near the top of this file. Move on to the remaining open items below (1-5).
- The audit findings and all fixes are described in full above — you shouldn't need to re-audit
  from scratch. If in doubt about a specific number, run `scripts/verify_claims.py` rather than
  manually re-deriving.
- Model checkpoints: **`model/best_model_botiot_twostage.pth` is the final, correct model
  (0.9790 macro-F1, updated session 2 — was 0.9639)**. The pre-session-2 version is preserved at
  `model/best_model_botiot_twostage_BACKUP_0.9639.pth` if ever needed for comparison.
  `model/best_model.pth` is a stale pre-distillation checkpoint (0.9352) — don't use it for
  anything claiming to represent "the" model; several scripts still default to it for pure
  latency benchmarking (harmless, latency is shape- not weight-dependent) but any
  correctness/accuracy claim must use the twostage checkpoint. **`train_twostage.py` has no
  suffix flag and will silently overwrite `model/best_model_botiot_twostage.pth` on every run —
  back up the current best before running it again.**
- **CUDA kernel weight exports are now stale** relative to the new 0.9790 checkpoint (they still
  reflect the old 0.9639 one) — re-run `scripts/validate_weights.py`/`validate_real_weights.py`
  before any claim ties CUDA kernel correctness to "the" model's accuracy. Not done this session
  (out of scope for the RF-gap work, but flagged so it isn't missed).
- To resume beyond item #0: (a) open item #5's "Next steps for further improvement" list
  (minority-class targeting, stronger RF teacher, ensemble re-tuning), (b) wait for the user to
  run the DICC jobs (open item #3) and pick up there, or (c) ask the user which they'd like to
  prioritize given "we have a lot of time" — all are legitimate, none blocks the others.
