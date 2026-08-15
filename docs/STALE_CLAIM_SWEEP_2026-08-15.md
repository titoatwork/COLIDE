# Stale-claim sweep — 2026-08-15

**Scope:** Final publication-surface sweep beyond the original `scripts/check_stale_claims.py` limited list.  
**Authority numbers (active):** Principal BoT sealed **0.9780 ± 0.0033**; ToN corrected CNN **0.8075** / RF **0.9626** (`toniot_leakage_safe_v1`); local B3 production-weight parity **closed** (`block3_parity_gate.json` `valid=true`, `kernel_status=post_fix`); DICC B3 latency remains **pre_fix** historical until rebench.  
**Guard:** `python scripts/check_stale_claims.py` → **OK** after this sweep.

---

## Checker expansion

| Change | Detail |
|--------|--------|
| Added to `ACTIVE_FILES` | `docs/paper_text_blocks.md` |
| Manuscript expansion | `_expand_active_files()` also includes every `.md`/`.txt`/`.rst`/`.tex` under `docs/manuscript/` |
| Kept | Exempt markers: historical / invalid / tombstone / archive / legacy / withdrawn / quarantine / forbidden |
| Kept | Hard forbidden: bare `0.9526`, `0.9851`, `+15.4%` / improvement-framed `15.4%`, bare `0.9790` |

---

## Files scanned

### Active checker surfaces
- `/home/titoisalive/colide/README.md`
- `/home/titoisalive/colide/docs/CLAIM_MAP_PREWRITE.md`
- `/home/titoisalive/colide/docs/PRE_MANUSCRIPT_CLOSURE.md`
- `/home/titoisalive/colide/docs/paper_text_blocks.md`
- `/home/titoisalive/colide/docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md`
- `/home/titoisalive/colide/docs/manuscript/figures/*` (image assets; captions live in manuscript prose)

### Additional publication / status surfaces (manual sweep)
- `/home/titoisalive/colide/docs/STATUS_REPORT_DRAFT.md`
- `/home/titoisalive/colide/docs/PROF_POR_STATUS_REPORT.md`
- `/home/titoisalive/colide/docs/PROF_POR_3DAY.md`
- `/home/titoisalive/colide/docs/PROF_PLAIN_NUMBERS_CARD.md`
- `/home/titoisalive/colide/docs/PROF_FEEDBACK_ROADMAP.md`
- `/home/titoisalive/colide/docs/KNOWN_LIMITATIONS.md`
- `/home/titoisalive/colide/docs/REMEDIATION_STATUS.md`

---

## Hits fixed (by file)

### `scripts/check_stale_claims.py`
- Added `docs/paper_text_blocks.md` to `ACTIVE_FILES`.
- Added `_expand_active_files()` so all manuscript text files are checked; `main()` uses the expander.

### `README.md`
- Reconciled internal contradiction: local production-weight B3 parity is **CLOSED** (not open).
- B3 note, per-block footnote, Block-3 progression, Limitations: local parity closed + DICC post_fix **latency** rebench still open.
- Numerical fidelity note: local B3 parity closed; native TRT numerical equivalence **not** claimed.

### `docs/paper_text_blocks.md`
- Bare **0.9790** labeled **historical / legacy** everywhere it appears (MLP note, summary table rows, KD two-stage v2, fidelity source, accuracy baseline).
- Summary table: incomplete Custom CUDA scope; bulk batched (not streaming); exploratory energy; CLAIM-PIPE-001 no cross-table ratios.
- GPU profiling + golden narrative + prior-work citation: tombstoned historical partial-vs-full “3.04×–3.78× / 3.60×–4.99×” headlines.
- RF defense: active ToN **0.8075 / 0.9626**; clean **0.9526 / 0.9851** INVALID.
- Threats / forbidden list: TRT numerical equivalence skipped; streaming SLA forbidden; post_fix DICC latency not rebench.

### `docs/PRE_MANUSCRIPT_CLOSURE.md`
- B3 DICC quarantine: pre_fix until **latency** rebench (local parity already closed).
- Gate table: production-weight parity **DONE** local; sanitizers **DONE** local; server rebench OPEN.
- Contribution spine + next-phase checklist: parity/sanitizer items marked DONE; remaining DICC post_fix latency + TRT numerical honesty.

### `docs/CLAIM_MAP_PREWRITE.md`
- Post_fix B3 language: local parity closed; **DICC multi-session post_fix latency** still FORBIDDEN until rebench.
- Welch caveat paragraph: production-weight parity open → **closed**.

### `docs/KNOWN_LIMITATIONS.md`
- §6 retitled and rewritten: local parity closed; DICC latency pre_fix; do not conflate.
- Numerical fidelity bullet: local B3 parity closed; TRT equivalence not claimed.

### `docs/REMEDIATION_STATUS.md`
- CUDA evidence: local parity closed; DICC rebench open.
- CUDA-B3-001/002/003 issue rows: **CLOSED (local)** with gate evidence (no longer “parity not established” / `valid=false`).

### `docs/STATUS_REPORT_DRAFT.md` (superseded feedstock)
- Banner claim-hygiene tombstone for 0.9790 / ToN clean / +15.4%.
- Executive, inventory, frozen tables, appendix anchors: **historical / legacy** for 0.9790; **INVALID / tombstone** for ToN clean; streaming → **bulk batched**.
- Energy anchors labeled **EXPLORATORY**.

### `docs/PROF_POR_STATUS_REPORT.md` (superseded)
- Banner + accuracy table: historical 0.9790; INVALID ToN 0.9526; active ToN numbers.
- Framework table: removed active partial-vs-full ratio column; CLAIM-PIPE-001 tombstone note.
- Streaming artifact row → bulk batched.

### `docs/PROF_POR_3DAY.md` (superseded historical run plan)
- Banner + ready-now table + draft Prof email: historical/INVALID labels; bulk batched; FORBIDDEN partial-vs-full ratios.

### `docs/PROF_PLAIN_NUMBERS_CARD.md`
- ToN: active 0.8075/0.9626 first; clean INVALID; optional older path labeled prior.
- DICC B3: pre_fix wall-clock; local parity closed; Option A no full-pipeline claim.

### `docs/PROF_FEEDBACK_ROADMAP.md`
- Champion 0.9790 → historical/legacy; principal sealed 0.9780±0.0033.

### `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md`
- **Already aligned** at scan time (or concurrent pass): active ToN 0.8075/0.9626; INVALID clean path; local B3 parity post_fix; DICC B3 PRE_FIX; bulk batched; Option A; TRT backend skipped. No further numeric invention. Figure captions already honest (Figure 6 bulk/exploratory/Option A).

---

## Residual historical-only mentions left intentionally

These remain on purpose as **labeled history / tombstones**, not active claims:

| Number / claim | Where it may still appear | Required label |
|----------------|---------------------------|----------------|
| **0.9790** | README dual-bar, paper_text_blocks KD table, STATUS/PROF superseded packs, KNOWN_LIMITATIONS | historical / legacy / development |
| **0.9526 / 0.9851 / +15.4%** | Tombstone callouts (README, manuscript §5.12, claim map, PRE_MANUSCRIPT, paper_text_blocks) | INVALID / tombstone / DATA-TON-001 |
| Partial-vs-full CUDA ratios (e.g. 3.60×–4.99× TRT) | Tombstone footnotes in paper_text_blocks / PROF packs | historical / FORBIDDEN / CLAIM-PIPE-001 |
| ~25,899 f/s | README, manuscript systems table, STATUS anchors | bulk batched only — not streaming SLA |
| DICC B3 ~513 vs ~363 µs (V100S) etc. | README, claim map, manuscript §5.10/§5.13 | pre_fix / PRE_FIX historical wall-clock |
| June single-shot ~551 / ~592 µs | README legacy line, STATUS LEGACY | legacy single-shot |
| Superseded “DICC ABSENT / blocker” narrative | STATUS_REPORT_DRAFT, PROF_* bodies | SUPERSEDED banner + historical |

---

## Explicit non-claims confirmed (no new numbers)

- No full custom CUDA pipeline vs full V3 speedup as active parity.
- No partial CUDA block-sum ratioed to full-model TRT/eager/compile as headline.
- No true streaming / offered-load saturation from the bulk harness.
- No native TensorRT numerical equivalence (engine skipped in framework gate).
- No post_fix DICC B3 latency rebench claimed as done.
- No “production-weight parity open” as active truth (local gate is valid=true).

---

## Verification

```text
$ python scripts/check_stale_claims.py
check_stale_claims: OK (no forbidden strings in active surfaces)
```

No retrain. No push (parent commit responsibility).
