# COLIDE — Session Handoff

**Last session:** 2026-07-01 (Claude Sonnet 5, high effort). **Read this whole file before doing
anything else** — it has the full context needed to continue without re-deriving what's already
been established.

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
  existing ones — e.g. the accuracy gap vs the Random Forest baseline (currently 2.25% on
  BoT-IoT) is a candidate for actual improvement (better distillation recipe, more KD sweep
  points, etc.), not just accurate reporting. That work has not started yet.
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
Current status: **all tracked claims pass, 0 regressions.** It also prints "orphan numbers"
(bolded figures in README not yet covered by a manifest entry) — currently: `0.7, 0.9526,
0.9639, 1.00x, 2.0, 3.3%, 5.0, 675, 87`. These aren't necessarily wrong, just not yet
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

5. **IN PROGRESS 2026-07-01 (session 2): substantively improve the RF accuracy gap** (not just
   report it accurately). Before starting the actual KD-recipe work, found and fixed a real
   provenance gap in the number the whole task is framed around:

   **RESOLVED — README's 0.9864 RF figure had no traceable source.** It's the "CPU RF (sklearn),
   200 trees" headline used for the "2.25% gap" claim. Checked `DAILY_LOG.md` (no mention),
   `scripts/rf_baseline.py` as-is (100 trees, independent undersample+SMOTE recipe -> 0.9768 test,
   which IS the well-documented Day-3 windowing-decision number, just a different RF than the
   0.9864 one), a 200-tree variant of that same recipe (-> 0.9730 test, *lower* — ruled out "just
   add more trees"), and `train_distill.py`'s inline 200-tree RF teacher with full-SMOTE (-> 0.9750
   validation). None matched. Git history showed 0.9864 present since the line was first added,
   never explained, only the gap arithmetic around it later corrected (1.29%->2.25%) without ever
   verifying the source value itself. **User supplied the original ad-hoc terminal command**: a
   200-tree RF trained/evaluated directly on `data/processed/*.npy` — i.e. the SAME preprocessed
   splits (268,627 train / 293,482 val / 733,705 test) the CNN-BiLSTM itself trains and evaluates
   on, which is actually the methodologically correct "gap" comparison (apples-to-apples same
   data), unlike the other two scripts' independent resampling. Reproduced byte-for-byte: **0.9864
   exactly**. Saved as a permanent, reproducible script — `scripts/rf_baseline_processed.py` ->
   `benchmarks/results/rf_baseline_processed.json` — instead of leaving it as an unrepeatable
   terminal one-liner. Added `verify_claims.py` manifest entries (`rf_baseline_processed_test_f1`,
   and rewired `rf_gap_botiot_final`/`rf_gap_botiot_baseline` to load from this JSON instead of a
   hardcoded `0.9864` literal). Added a footnote to README's Detection Accuracy table explaining
   why this number differs from the other two (legitimate, differently-configured) RF baselines
   in the repo. `scripts/verify_claims.py` passes all 51 claims, 0 regressions.

   **Not yet started:** the actual accuracy-gap-closing work (better KD recipe, finer
   alpha/temperature sweep, different teacher ensemble, etc.). A quick timing probe found ~204-212s
   per CNN-BiLSTM epoch (up to 50 epochs, patience=10) plus one-time RF-teacher training per run —
   worth factoring into how many new sweep points are feasible before proposing a concrete plan.

## Git state — everything committed and pushed

Update (end of session, after this file was first written): all fixes above are committed as 9
logical commits and pushed to `origin/master`. Working tree is clean. Commit range
`9c8d86f..dedf158`, oldest to newest:

```
9c8d86f tools: add claim verifier and CUDA kernel statistical benchmark harness
09b509b fix: LLM dispatch overhead was fabricated, replace with real percentile benchmark
a0ff1a8 fix: pipeline speedup ratio used an unsourced PyTorch baseline
96dfc58 fix: weight export/validation scripts pointed at a stale pre-distillation checkpoint
99b7f80 fix: significance tests were one-sample against a bare constant, not two-sample
4e09ca1 feat: rebuild ablation_study.py to load from JSON instead of hardcoded literals
9acbc15 chore: wire statistical CUDA kernel benchmark into DICC setup/run scripts
e928d8e docs: correct fabricated/stale numbers across README and paper text blocks
dedf158 docs: add CLAUDE.md architecture guide and session handoff
```

Each commit message has full detail on that phase's fix — read `git show <hash>` if you need the
exact reasoning for a specific change instead of re-deriving it. `benchmarks/results/*.json` new
files (e.g. `cuda_kernel_stats_rtx3050.json`) are NOT committed — that directory is gitignored
for new files (existing tracked result JSONs still update fine; only new ones are blocked without
`-f`). Consistent with this repo's stated policy ("keep plots, not raw data") — don't force-add
new benchmark JSONs unless the user asks for that policy to change.

**Still not done** (unrelated to the commit/push step, carried over from before): the *tracked*
binary `inference/kernels/fused_pipeline` and the source-only `fused_block3_naive.cu` (no tracked
binary exists for it) should probably be recompiled from their now-fixed `.cu` sources and
committed, so the binary artifacts match the corrected source before final lock. Check `git log`
on `inference/kernels/` to confirm this repo's convention of committing compiled binaries, then
decide.

## Quick orientation for a fresh session

- Read `CLAUDE.md` first (architecture overview, written this session, should already be
  accurate).
- The audit findings and all fixes are described in full above — you shouldn't need to re-audit
  from scratch. If in doubt about a specific number, run `scripts/verify_claims.py` rather than
  manually re-deriving.
- Model checkpoints: **`model/best_model_botiot_twostage.pth` is the final, correct model
  (0.9639 macro-F1)**. `model/best_model.pth` is a stale pre-distillation checkpoint (0.9352) —
  don't use it for anything claiming to represent "the" model going forward; several scripts
  still default to it for pure latency benchmarking (harmless, since latency is shape- not
  weight-dependent) but any correctness/accuracy claim must use the twostage checkpoint.
- To resume: either (a) address open item #1 (cuDNN baseline) or #2 (naive kernel) as
  self-contained local tasks, or (b) wait for the user to run the DICC jobs (open item #3) and
  pick up there, or (c) ask the user which they'd like to prioritize given "we have a lot of
  time" — all three are legitimate next steps, none is blocking the others.
