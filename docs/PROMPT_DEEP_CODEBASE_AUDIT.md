# Paste-ready prompt — COLIDE deep codebase evidence audit

**How to use:** open a **new** agent chat in the COLIDE repo root  
(`/home/titoisalive/colide` or clone of `titoatwork/COLIDE`).  
Paste **everything under the line** as the first message.  
Do **not** paste only the title — the full block is the task.

---

```text
================================================================================
COLIDE — EXTREME DEEP CODEBASE EVIDENCE AUDIT (NEW CHAT, ONE PACKAGE ONLY)
================================================================================

You are NOT writing the professor report. You are NOT drafting email copy.
You are NOT inventing numbers. You are NOT training models. You are NOT
clobbering checkpoints. You are NOT SSHing to DICC.

You ARE a forensic research-codebase auditor. Your ONLY job this entire
chat is to exhaustively mine THIS repository (all history + all files) for
EVIDENCE: what numbers exist, where they live, how they were produced,
whether they were re-verified, what is stale/legacy/superseded/invalid,
what work is planned next, and what unplanned improvements would strengthen
the project.

A DIFFERENT LLM (or a later chat) will turn your deliverables into a report.
Your output is the FEEDSTOCK for that LLM. Incomplete, shallow, or
hand-wavy analysis is a failure. This session is expected to be LONG.
Prefer depth over speed. One major package = this audit only. Do not stack
manuscript writing or Prof email drafting.

--------------------------------------------------------------------------------
0. SITUATION (read carefully — this is the human context)
--------------------------------------------------------------------------------

Project: COLIDE (CUDA-Optimized CNN-BiLSTM + on-device LLM explainability
for IoT intrusion detection). Academic systems/performance research aiming
at FGCS-class venues. Contribution is systems/measurement, NOT model novelty.

Locked strategy (do not reopen):
  - Option A (docs/DESIGN_PLAN.md): valid claims are PER-BLOCK Custom CUDA
    vs PyTorch only. FORBIDDEN: full-pipeline Custom CUDA vs full PyTorch V3
    speedup (architecture parity gap: attention / LayerNorm / GAP; fused
    pipeline skips Block 3 in places).
  - Official cluster for multi-day: UM DICC only (not Rostam as official).
  - June 2026 DICC ~551 µs (V100S) / ~592 µs (A100) = LEGACY SINGLE-SHOT
    CUDA-only; not multi-day; no same-GPU PyTorch baseline that day.
  - Multi-day Day1+Day2 campaign with PyTorch baselines is SCRIPTED but NOT
    COMPLETED (ops/access; waiting on Prof re possible Cheran help).
  - Hard gate before any "final" Prof numbers email: codebase-wide numbers
    match + scripts/verify_claims.py green (docs/FINAL_PLAN.md P2).
  - Champion: model/best_model_botiot_twostage.pth
    md5 80a90f7cc210276300eaa90173a5a385 — do not retrain or overwrite.

Why this audit exists NOW:
  User decided NOT to wait for multi-day DICC before gathering evidence for
  a report. A previous chat wrongly DRAFTED a Prof status report
  (docs/PROF_POR_STATUS_REPORT.md). That is OUT OF SCOPE for you. Treat any
  pre-drafted report text as NON-authoritative. Your job is deeper: rebuild
  the evidence base from code, JSON, git history, and docs — so a later
  process can write an accurate report without guessing.

Human will feed YOUR markdown inventory into another LLM for report writing.
You deliver raw, structured, citable FACTS with paths, commits, and
confidence labels — not polished prose for the professor.

--------------------------------------------------------------------------------
1. INTENSITY CONTRACT (non-negotiable)
--------------------------------------------------------------------------------

This is an EXTREME analysis. Work like a hostile external auditor who
assumes numbers may be stale, dual-sourced, or once-fabricated-and-fixed.

You MUST:

1. Touch EVERY meaningful surface of the repo, not just README.
2. Inventory EVERY numeric claim that could appear in a paper/report.
3. Trace each claim to: source file → producing script → git introduction
   → subsequent re-verification or range widening commits when possible.
4. Use git history heavily: log, blame, show, pickaxe search (-S / -G) for
   key numbers and filenames under benchmarks/results/.
5. Distinguish: CURRENT / SUPERSEDED / LEGACY / UNVERIFIED / INVALID CLAIM /
   PLANNED BUT NOT RUN / TOOLING-ONLY (e.g. Rostam).
6. Never invent cluster multi-day numbers. If multi-day SUCCESS dirs are
   absent, say ABSENT with path you checked.
7. Never treat docs prose as stronger than JSON/scripts unless the doc is
   the only artifact (and then flag it as doc-only).
8. Run scripts/verify_claims.py and record full outcome; also read the
   CLAIMS manifest inside that script — it is incomplete relative to all
   bold numbers; find the GAPS.
9. Compare README.md, docs/paper_text_blocks.md, HANDOFF.md,
   docs/PROF_POR_3DAY.md, docs/FINAL_PLAN.md, docs/DESIGN_PLAN.md,
   colide_review_brief.md (home if present), DAILY_LOG.md for
   contradictions.
10. Map model variants (v1 windowed, v2, v3 attention, MLP, ensemble, KD
    sweeps, two-stage) and which checkpoint is production.
11. Map every CUDA kernel .cu and what each binary measures.
12. Map every benchmarks/results/*.json and *.txt — parse them, extract
    means/ranges/p-values/metadata, note methodology fields.
13. Map scripts/* to results files (producer graph).
14. Map dicc_scripts/* and what SUCCESS artifacts SHOULD look like vs what
    is on disk today.
15. Document threats-to-validity already acknowledged in-repo (measurement
    drift, WSL2, RF gap, Option A, energy vs VRAM, etc.).
16. List planned remaining work FROM THE REPO PLANS (FINAL_PLAN, PROF_POR,
    HANDOFF, DESIGN_PLAN) — quote, do not invent schedule.
17. Separately list UNPLANNED improvements that would make evidence/report
    stronger (your engineering judgment, clearly labeled as UNPLANNED /
    OPTIONAL — never as if already approved).
18. Spend the time. Multi-hour depth is expected. Do not stop after a
    surface README pass. If context is long, write intermediate markdown
    files as you go and keep expanding them.

If you finish "quickly," you failed the intensity bar — go deeper:
re-check commit ranges, re-scan JSON keys, re-diff docs vs results.

--------------------------------------------------------------------------------
2. HARD RULES
--------------------------------------------------------------------------------

- Repo root: work inside COLIDE (pwd should contain README.md, model/,
  inference/, benchmarks/, scripts/, docs/, HANDOFF.md).
- Read HANDOFF.md header + Session lifecycle first, then this prompt.
- Agents have NO DICC SSH. Do not pretend you ran cluster jobs.
- No training. No overwriting best_model_botiot_twostage.pth.
- No "Prof email" drafting as primary deliverable.
- No full-pipeline CUDA vs full V3 speedup as a recommended claim.
- Prefer exact quotes of numbers with file:line or JSON path.
- When uncertain, label UNCERTAIN and say what would resolve it.
- End session per HANDOFF lifecycle: update HANDOFF, commit+push the
  audit markdown artifacts (unless user forbids network), paste-ready
  next prompt for "report writer LLM" or residual gaps.

--------------------------------------------------------------------------------
3. METHOD (execute in order; do not skip)
--------------------------------------------------------------------------------

### Phase A — Orientation (still thorough)

- git status, git log --oneline (deep history, not just 10 commits),
  branches, tags if any.
- Tree of top-level dirs; count scripts, kernels, result JSONs.
- Read: HANDOFF.md, docs/FINAL_PLAN.md, docs/DESIGN_PLAN.md,
  docs/PROF_POR_3DAY.md, AGENTS.md, environment.md, README.md (full),
  DAILY_LOG.md structure, docs/paper_text_blocks.md (full if large —
  chunk it).
- Note that docs/PROF_POR_STATUS_REPORT.md may exist from a mistaken prior
  draft — inventory it as a DOC ARTIFACT only, not as ground truth.

### Phase B — Results corpus (every file)

For each file under benchmarks/results/ (and nested dirs if any, including
benchmarks/results/dicc/ if present):

  path | type | size | mtime if useful | schema/keys | headline metrics |
  producing script (guess then confirm via git log -- path and script
  references) | platform (RTX 3050 / V100 / A100 / unknown) | n trials |
  multi-session? | status: CURRENT / LEGACY / SUPERSEDED / PARTIAL

Parse JSON properly (python). Do not summarize only filenames.

Especially critical:
  - twostage_botiot.json, rf_baseline_processed.json
  - cuda_kernel_stats_rtx3050.json (multi-session)
  - pytorch_block3_stats_rtx3050.json
  - statistical_significance_v2.json / statistical_confidence.json
  - llm_explainability.json, streaming_throughput.json
  - numerical_fidelity.json, real_weight_validation.json
  - pipeline_benchmark*.json
  - tensorrt_native.json, torch_compile_native.json, energy_*, a100_*,
    cuml_*, ablation_*, distill_* sweeps, toniot_*, ensemble_*,
    dicc_v100_summary.txt, dicc_a100_summary.txt
  - Any framework comparison session files not listed above

### Phase C — Producer / consumer graph

Build a graph:

  script → config → checkpoint → metrics JSON → README/doc claim

Include:
  scripts/train*.py, benchmark_*.py, verify_claims.py,
  numerical_fidelity.py, compare_dicc_sessions.py, benchmark.sh,
  dicc_scripts/run_campaign.sh, submit_session.sh, profiles, etc.

Mark scripts that are dead / experimental / ablation-only.

### Phase D — Model & CUDA architecture map

- model/*.py: architectures, param counts if stated, which is production.
- Checkpoint inventory under model/ (md5 if present; never delete).
- inference/kernels/*.cu: blocks 1–4, naive, fp16, pipeline; what each
  validates; known Option A issues (fused_pipeline skip B3, V3 gaps).
- Export / weights_bin path and validation_metadata.json.

### Phase E — Claim surface extraction

Extract EVERY load-bearing number from:

  - README.md (all tables, bold figures, footnotes)
  - docs/paper_text_blocks.md
  - HANDOFF frozen numbers if any
  - docs/PROF_POR_* 
  - scripts/verify_claims.py CLAIMS list
  - Any abstract drafts

For each claim produce a row:

  claim_id | text snippet | number(s) | doc location | linked JSON/source |
  verify_claims covered? Y/N | git first introduced (commit) |
  last modified (commit) | re-verified? evidence | confidence
  (HIGH/MED/LOW) | notes (range vs point, WSL2, legacy, invalid Option A)

### Phase F — Git re-verification archaeology (CRITICAL)

This is the part users care about most: which numbers were REVERIFIED.

You MUST mine history for:

  - Fabrication/audit language (search commits and docs for: fabricat,
    unsourced, re-verif, session 3, range, multi-session, verify_claims,
    measurement stability, widened, corrected, placeholder, LLM latency)
  - Commits that changed benchmarks/results/*.json
  - Commits that changed README tables / ranges (e.g. d9e1f79-style range
    widenings — confirm real hashes yourself)
  - Bit-identical training repro claims and their artifacts
  - KD sweep + two-stage selection path (how 0.9790 became champion)
  - Fidelity self-check introduction
  - DICC summary files introduction vs campaign hardening commits

Deliver a dedicated section:

  "NUMBERS RE-VERIFIED VIA HISTORY"

with sub-bullets: what was wrong before, what commit fixed it, what the
current source of truth is, residual risk.

Also:

  "NUMBERS NEVER RE-RUN / SINGLE-SHOT ONLY"
  "NUMBERS WITH MULTI-SESSION RANGES"
  "NUMBERS DOC-ONLY OR UNSOURCED (if any remain)"

### Phase G — Consistency / contradiction matrix

Cross-check same metric across surfaces. Flag mismatches (even 1 ulp or
rounding). Flag README bold numbers not in verify_claims manifest.
Flag Option A violations if any text still implies full-pipeline speedup.

### Phase H — Planned work left (from repo plans ONLY)

From FINAL_PLAN P0–P5, PROF_POR_3DAY, HANDOFF, DESIGN_PLAN:

  - What is DONE (with evidence)
  - What is IN PROGRESS / BLOCKED (Prof reply, DICC SSH)
  - What is NEXT after unblock
  - What is EXPLICITLY DEFERRED (manuscript, stretch)

Do not invent deadlines beyond what docs say.

### Phase I — Unplanned improvements (clearly labeled OPTIONAL)

Propose improvements that would strengthen evidence/reportability, e.g.:

  - expand verify_claims coverage
  - automated README↔JSON drift CI
  - multi-day DICC artifact schema checklist
  - missing same-GPU PT baselines
  - better session tagging
  - energy methodology
  - dead script archival
  - claim language linter for Option A

Each item: problem → proposed change → effort → risk → whether it
blocks a status report or only helps manuscript quality.

### Phase J — Write deliverables (markdown files ON DISK)

Create under docs/audit/ (create directory):

  00_INDEX.md
      Master index, how to read the pack, confidence legend,
      "for report-writer LLM: start here"

  01_REPO_MAP.md
      Structure, scripts, kernels, configs, data paths

  02_RESULTS_INVENTORY.md
      Every results file parsed (tables)

  03_PRODUCER_GRAPH.md
      Script → artifact → claim surfaces

  04_CLAIMS_REGISTER.md
      Full claim register with sources + confidence

  05_GIT_REVERIFICATION.md
      History archaeology; re-verified vs single-shot vs stale

  06_CONTRADICTIONS.md
      Cross-doc mismatches and Option A risks

  07_PLANNED_WORK_LEFT.md
      From FINAL_PLAN / HANDOFF / PROF_POR only

  08_UNPLANNED_IMPROVEMENTS.md
      Optional strengthening ideas

  09_RAW_NUMBER_TABLES.md
      Dense tables a report LLM can copy (metric | value | source path |
      platform | n | date/commit | label CURRENT/LEGACY/…)

  10_EVIDENCE_GAPS.md
      What is missing for a complete honest status (esp. multi-day DICC)

  11_REPORT_WRITER_BRIEF.md
      NOT a professor letter. A machine-oriented brief:
      "Facts you may state", "Facts you must NOT state",
      "Required caveats", "Sources for each paragraph topic",
      "Open questions for human".
      Explicitly: multi-day DICC cells empty; use legacy only if labeled.

If files get huge, that is GOOD. Split further rather than truncate.

--------------------------------------------------------------------------------
4. QUALITY BAR FOR "DONE"
--------------------------------------------------------------------------------

You may only declare the audit complete when ALL are true:

  [ ] Every benchmarks/results file listed and parsed (or marked unreadable)
  [ ] verify_claims.py run + manifest gap analysis vs README bold numbers
  [ ] Claim register covers README + paper_text_blocks load-bearing numbers
  [ ] Git archaeology section has real commit hashes you inspected
  [ ] Champion md5 confirmed on disk if file present
  [ ] Multi-day DICC path checked; absence documented
  [ ] Planned work section cites FINAL_PLAN/HANDOFF
  [ ] Unplanned improvements clearly optional
  [ ] 00_INDEX.md tells a report-writer LLM the reading order
  [ ] No invented DICC multi-day numbers
  [ ] HANDOFF updated; audit docs committed (+ pushed if allowed)
  [ ] Closing message has status table + next-session prompt for the
      REPORT WRITER chat (separate package)

Partial work is fine mid-session if saved to disk; final close needs the
checklist above or an explicit residual list of what was not finished.

--------------------------------------------------------------------------------
5. TOOLING HINTS (use them)
--------------------------------------------------------------------------------

```bash
cd /path/to/colide
git log --oneline --all | head -200
git log --oneline -- benchmarks/results/ README.md scripts/verify_claims.py
git log -S'0.9790' --oneline --all
git log -S'fabricat' --oneline --all -i
git log -G'measurement stability|multi-session|verify_claims' --oneline -i
git blame -L 1,80 README.md   # as needed
find benchmarks/results -type f | sort
PYTHONPATH=. python scripts/verify_claims.py
md5sum model/best_model_botiot_twostage.pth
ls -la benchmarks/results/dicc 2>/dev/null || echo "no multi-day dicc tree"
rg -n "3\\.60|0\\.9790|16\\.60|25899|551|592|Option A|full.pipeline" \
  README.md docs/ HANDOFF.md scripts/verify_claims.py
```

Parse JSON with Python, not eyeballing alone.

--------------------------------------------------------------------------------
6. TONE OF DELIVERABLES
--------------------------------------------------------------------------------

- Neutral, forensic, evidence-first.
- Tables > essays.
- Every strong number needs a path.
- Explicit labels: CURRENT / LEGACY / SUPERSEDED / INVALID / ABSENT /
  UNCERTAIN / PLANNED / OPTIONAL.
- Never market; never hide RF gap or measurement drift.
- Never fill multi-day DICC blanks with guesses.

--------------------------------------------------------------------------------
7. START NOW
--------------------------------------------------------------------------------

1) Confirm cwd and git HEAD.
2) Create docs/audit/ and 00_INDEX.md skeleton.
3) Run Phase A→J without waiting for further permission.
4) Work until the quality bar is met or you hit a hard blocker (then
   document blocker in 10_EVIDENCE_GAPS.md).

Remember: a prior chat already wrongly drafted a professor report.
Your redemption metric is DEPTH and TRACEABILITY of evidence, not pretty
status prose. Another model will write the report from YOUR files.

GO. BE EXHAUSTIVE.
================================================================================
```

---

## Notes for the human (not part of the paste)

| Item | Detail |
|------|--------|
| **This file** | `docs/PROMPT_DEEP_CODEBASE_AUDIT.md` |
| **Your job next** | New chat → paste the fenced `text` block only (or whole file) |
| **That chat’s job** | Extreme audit → `docs/audit/*.md` feedstock |
| **Not that chat’s job** | Final Prof report / email polish |
| **After audit** | Separate chat/LLM: report writer using `docs/audit/11_REPORT_WRITER_BRIEF.md` + number tables |
| **Ignore as ground truth** | `docs/PROF_POR_STATUS_REPORT.md` from the mistaken draft chat (audit may catalog it) |
