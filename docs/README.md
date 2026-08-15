# COLIDE documentation map

**Date:** 2026-08-15  
**Purpose:** Public index of what to read now vs what is kept for audit.  
**Rule:** Stale files are **kept and labeled**, never deleted.

If a number in an older file conflicts with [`RESULTS_INDEX.md`](RESULTS_INDEX.md) or [`CLAIM_MAP_PREWRITE.md`](CLAIM_MAP_PREWRITE.md), **those two win**.

---

## Current (use these)

Start here if you are writing or checking claims.

| Path | Role |
|------|------|
| [`CHERAN_MANUSCRIPT_HANDOFF.md`](CHERAN_MANUSCRIPT_HANDOFF.md) | **Manuscript-lead start** — read order, locked numbers, hard rules |
| [`CLAIM_MAP_PREWRITE.md`](CLAIM_MAP_PREWRITE.md) | OK / FORBIDDEN claims for drafting |
| [`RESULTS_INDEX.md`](RESULTS_INDEX.md) | Claim → artifact map (principal authority for numbers) |
| [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) | Discussion-safe limitation language |
| [`ISSUE_REGISTER.md`](ISSUE_REGISTER.md) | Stable issue IDs (DATA-TON-001, CUDA-B3-*, CLAIM-PIPE-001, …) |
| [`B3_SERVER_LATENCY_DECISION.md`](B3_SERVER_LATENCY_DECISION.md) | **Option B (2026-08-15):** DICC B3 latency is historical **pre_fix** only |
| [`PRE_MANUSCRIPT_CLOSURE.md`](PRE_MANUSCRIPT_CLOSURE.md) | Data closed; remaining publication notes |
| [`manuscript/CAD_CBA_v1_MANUSCRIPT.md`](manuscript/CAD_CBA_v1_MANUSCRIPT.md) | Current draft spine |
| [`manuscript/figures/`](manuscript/figures/) | Manuscript figures (synced copies also under `benchmarks/results/figures_current/`) |
| [`manuscript/TABLES_FROM_ARTIFACTS.md`](manuscript/TABLES_FROM_ARTIFACTS.md) | Tables pulled from locked JSON |
| [`FIGURE_STATUS.md`](FIGURE_STATUS.md) | Figure inventory (CURRENT / STALE / INVALIDATED) |

Repo-root [`README.md`](../README.md) is the public landing page. It is current, but detailed claims still resolve through the claim map and results index.

### DICC_* current extraction (B3 = pre_fix)

These extraction / results docs are the current DICC pack. **Block-3 server wall-clock numbers in them are historical pre_fix only** — not a post_fix rebench, not parity-gated server latency. See [`B3_SERVER_LATENCY_DECISION.md`](B3_SERVER_LATENCY_DECISION.md).

| Path | Role |
|------|------|
| [`DICC_EXTRACTION_TABLES.md`](DICC_EXTRACTION_TABLES.md) | Per-block means from SUCCESS JSON |
| [`DICC_B3_CUDA_VS_PT_REPORT.md`](DICC_B3_CUDA_VS_PT_REPORT.md) | B3 CUDA FP16 vs PT (pre_fix) |
| [`DICC_COMPARE_OUTCOMES.md`](DICC_COMPARE_OUTCOMES.md) | Cross-session compare |
| [`DICC_RESULTS_AND_FLAGS.md`](DICC_RESULTS_AND_FLAGS.md) | Campaign flags + session map |
| [`DICC_MULTI_COMPILER_MATRIX.md`](DICC_MULTI_COMPILER_MATRIX.md) | Full-model eager / compile / ORT / TRT absolutes |
| [`DICC_TORCH_COMPILE_STRETCH.md`](DICC_TORCH_COMPILE_STRETCH.md) | torch.compile stretch notes |
| [`DICC_OPS_METHOD.md`](DICC_OPS_METHOD.md) | How the campaign was run |
| [`DICC_RUNBOOK.md`](DICC_RUNBOOK.md) | Operator runbook |
| [`DICC_D0_PREFLIGHT_CHECKLIST.md`](DICC_D0_PREFLIGHT_CHECKLIST.md) | Preflight (ops) |

B1/B2/B4 matched-op CUDA wins remain OK. Full Custom CUDA vs full V3 is **FORBIDDEN**. Local B3 correctness is separate (parity + sanitizers below) and is **not** a DICC latency claim.

Remaining pub notes (not claim tables): [`PUBLICATION_REMAINING_2026-08-15.md`](PUBLICATION_REMAINING_2026-08-15.md), [`REMEDIATION_STATUS.md`](REMEDIATION_STATUS.md).

---

## Evidence JSON

Do not invent numbers. Prefer these locked artifacts:

| Path | What it is |
|------|------------|
| `benchmarks/results/toniot_corrected/` | Leakage-safe ToN (`toniot_leakage_safe_v1`; `valid: true`) |
| `benchmarks/results/block3_parity_gate.json` | Local production-weight B3 CUDA↔PT parity (`kernel_status=post_fix`; **not** DICC latency) |
| `benchmarks/results/framework_parity_gate.json` | Framework logit parity (eager / ORT / compile; TRT native skipped) |
| `benchmarks/results/sanitizer_b3/` | Local B3 sanitizers (0 errors FP32+FP16) |
| `benchmarks/results/dicc/` | DICC SUCCESS trees + framework stretch JSON |

Principal BoT sealed multi-seed lives under `benchmarks/results/sealed_test/` (see results index). Historical “clean” ToN JSON is a **tombstone** (`toniot_clean_comparison*.json`) — DATA-TON-001.

---

## Historical / audit (kept, not current authority)

Useful for provenance. **Not** paper authority if they disagree with the current set.

| Path | Note |
|------|------|
| [`execution_plan/`](execution_plan/) | Phase plans, work packages, freeze cards |
| [`audit/`](audit/) | Deep audit pack (`00_INDEX.md` …) |
| [`STATUS_REPORT_DRAFT.md`](STATUS_REPORT_DRAFT.md) | Draft status — may lag later gates |
| [`PROF_POR_3DAY.md`](PROF_POR_3DAY.md), [`PROF_POR_STATUS_REPORT.md`](PROF_POR_STATUS_REPORT.md) | Supervisor packs (`PROF_POR_*`) |
| [`DESIGN_PLAN.md`](DESIGN_PLAN.md) | Option A design lock (historical plan) |
| [`FINAL_PLAN.md`](FINAL_PLAN.md) | Execution plan snapshot |
| [`../COLIDE Remediation and Limited-Scope Improvement Checklist.md`](../COLIDE%20Remediation%20and%20Limited-Scope%20Improvement%20Checklist.md) | Root checklist; some boxes lag later gate JSON |
| [`../COLIDE_Remediation_Update_Review.md`](../COLIDE_Remediation_Update_Review.md) | **Review status — still useful** as readiness review, not a number table |
| [`../DAILY_LOG.md`](../DAILY_LOG.md) | Session diary |
| [`../HANDOFF.md`](../HANDOFF.md) | Agent session handoff (Aug 12 “pre-manuscript closed” is stale vs later Option B / pub notes) |

Also historical / internal coding notes: [`../AGENTS.md`](../AGENTS.md), [`../CLAUDE.md`](../CLAUDE.md), [`PROF_FEEDBACK_ROADMAP.md`](PROF_FEEDBACK_ROADMAP.md), [`KD_OBJECTIVES.md`](KD_OBJECTIVES.md), [`CHECKLIST_DEEP_AUDIT_REPORT.md`](CHECKLIST_DEEP_AUDIT_REPORT.md).

---

## Internal correspondence (not paper evidence)

Do not cite emails as experimental evidence.

| Path |
|------|
| [`EMAIL_CHERAN_MATERIALS_HANDOFF.md`](EMAIL_CHERAN_MATERIALS_HANDOFF.md) |
| [`EMAIL_FINAL_STATUS_PROF_POR_INTERNSHIP.md`](EMAIL_FINAL_STATUS_PROF_POR_INTERNSHIP.md) |
| [`EMAIL_REPLY_PROF_POR_FEEDBACK.md`](EMAIL_REPLY_PROF_POR_FEEDBACK.md) |
| [`EMAIL_STATUS_PROF_POR_POST_FEEDBACK.md`](EMAIL_STATUS_PROF_POR_POST_FEEDBACK.md) |
| [`EMAIL_STATUS_PROF_POR_SHORT_SENT.md`](EMAIL_STATUS_PROF_POR_SHORT_SENT.md) |
| [`EMAILS_POR_REPLY_FOLLOWUP.md`](EMAILS_POR_REPLY_FOLLOWUP.md) |

---

## How to read stale docs

1. **Never delete.** Quarantine or label instead.
2. **Conflict rule:** [`RESULTS_INDEX.md`](RESULTS_INDEX.md) and [`CLAIM_MAP_PREWRITE.md`](CLAIM_MAP_PREWRITE.md) win over older prose, emails, HANDOFF, PROF_POR packs, execution-plan notes, and root checklist boxes.
3. **ToN “clean” 0.9526 / 0.9851 / +15.4%** is **INVALID** (DATA-TON-001) everywhere it still appears.
4. **Historical BoT 0.9790** is development/legacy only; principal is sealed **0.9780 ± 0.0033**.
5. **DICC B3 µs** are **pre_fix / historical** unless a new post_fix SUCCESS tree exists (it does not).
6. Guard: `python scripts/check_stale_claims.py`.
