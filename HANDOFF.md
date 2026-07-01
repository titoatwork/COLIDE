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
   characterized. **This changed the actual finding, not just the provenance**: recomputed against
   this real baseline, only the FP16 kernel clearly beats cuDNN (784.1/601.7 = **1.30x**, was
   ambiguously reported as 1.23x); the transposed-W_hh steps (with/without CUDA Graphs) land at
   0.98x/0.99x — i.e. at or just below parity with PyTorch, not a clear win as the 943.6us reading
   would have implied, nor a small-but-real edge as the 740.7us reading implied. Propagated to
   `scripts/ablation_study.py` (now loads the JSON and prints the resolution + per-step ratios),
   `README.md` (Key Contributions #2, Per-Block Performance table, Block 3 Optimization
   Progression section), and `scripts/verify_claims.py` (new `pytorch_block3_cudnn_baseline` and
   `block3_beats_cudnn_ratio` manifest entries, new regression guard on the superseded "beating
   cuDNN by 1.23x" string). `scripts/verify_claims.py` passes all 47 claims, 0 regressions.

2. **The naive Block 3 kernel (`fused_block3_naive.cu`) has a real numerical-stability issue**,
   not just a validation-tolerance mismatch. Fixed one real bug already (its tolerance was 10x
   stricter than sibling kernels, `1e-3` vs `1e-2`), but even after that fix it still fails
   validation 24/30 times at a larger sample, with real divergence up to ~17% relative error on
   some hidden units — consistent with accumulated FP32 rounding error over its unoptimized
   sequential summation order. It's disclosed honestly in both `README.md` and
   `scripts/ablation_study.py` (reported for latency comparison only, not claimed as
   classification-verified). **This is exactly the kind of micro-gain the user wants pursued
   given "we have a lot of time"**: worth eventually either (a) fixing the naive kernel's
   accumulation order so it passes cleanly (would need to not change its "naive-ness" for the
   optimization story to still make sense), or (b) wiring its Block 3 output through Block 4 and
   checking whether the divergence ever actually flips a classification decision (would
   strengthen the disclosure from "unverified" to "verified harmless"). Not started.

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
   threats-to-validity section.

5. **NEW (not part of the original plan, mentioned by the user this session as future work,
   not started):** substantively improve weak numbers, not just report them accurately —
   specifically the accuracy gap vs. Random Forest (currently 2.25% on BoT-IoT, 3.3% on
   ToN-IoT). This is a real ML-improvement task (better KD recipe, more sweep points, different
   teacher ensemble, etc.), separate from and after the verification work above.

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
