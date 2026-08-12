# Progress log (results as they land)

**Policy:** document everything; label pilot vs full; test sealed unless noted.  
**Handoff snapshot:** 2026-08-12 — **PRE-MANUSCRIPT CLOSED** + default stretch largely done (S1c/S1d/S1a-V100); next = manuscript multi-GPU write (+ finish A100 torch.compile if still pending).

---

## 2026-08-12 — Pre-manuscript closed + stretch package

**Mode:** DICC evidence + stretch · Option A · no invent · champion frozen  
**Champion:** `80a90f7cc210276300eaa90173a5a385`

| Item | Status |
|------|--------|
| 6 SUCCESS S1/S2/Day2 × V100S+A100 on laptop | **DONE** (git force-added) |
| Extraction + formal compares | **DONE** |
| Fork: PT wins B3 on servers | **DONE** (`DICC_B3_CUDA_VS_PT_REPORT.md`) |
| S1c README hygiene | **DONE** |
| S1a torch.compile V100S job 395338 | **DONE** (~818 vs ~1033 µs) |
| S1a torch.compile A100 job 395339 | **DONE** (~761 vs ~957 µs) |
| S1b clean A100 | **DEFERRED** optional |
| Manuscript multi-GPU prose | **NEXT** |

**Authority:** `docs/PRE_MANUSCRIPT_CLOSURE.md`, `docs/DICC_TORCH_COMPILE_STRETCH.md`, `HANDOFF.md`

---

## 2026-07-29 — Return from pause: local wrap + D0 laptop preflight

**Mode:** packaging / preflight · no train · no multi-day numbers invented  
**Champion:** unchanged `80a90f7cc210276300eaa90173a5a385`

### Local wrap (A)
| Item | Status |
|------|--------|
| Prof short summary email | **SENT** (recorded `docs/EMAIL_STATUS_PROF_POR_SHORT_SENT.md`) |
| Plain-English numbers card | **DONE** `docs/PROF_PLAIN_NUMBERS_CARD.md` (no repo jargon for future Prof mail) |
| Claims green | **PASS** |
| DICC SUCCESS tree | still **ABSENT** |

### D0 laptop preflight
| Item | Status |
|------|--------|
| `run_campaign.sh --dry-run` | **OK** (local; no SLURM partitions — expected) |
| `local_validate.sh` | 26 pass / 2 fail (mock spool profile path); compare core OK |
| Ops + checklist | `docs/DICC_OPS_METHOD.md` + `docs/DICC_D0_PREFLIGHT_CHECKLIST.md` |
| User-only next | OnDemand VNC + screen + env + Day1 submit (**D1**) |

**Decision:** Local wrap closed. D0 **laptop** portion DONE. Full D0 needs user VNC login. Next session: **D1 Day1** after user completes VNC env steps in checklist §2.

---

## 2026-07-22 — Git branching policy lock (docs only)

**Mode:** documentation · no train · no DICC jobs  
**Primary artifact:** `docs/BRANCHING_POLICY.md`

### Delivered
| Item | Detail |
|------|--------|
| Final line | **`master` always final** (handoff, claims, manuscript tip) |
| When to branch | **Must** when work is a **true alternative option** (could be discarded / isolated fork) |
| Strict budget | Prefer 0–2 open feature branches; soft-cap ≤3 remote non-master; no vanity branches |
| After merge | Delete local + remote feature branch |
| Wired into | HANDOFF, SESSION_CONTINUITY, `16_SAFETY_AND_RULES` §5b, 00_INDEX, FINAL_PLAN pointers |

**Decision: DONE (policy)** — historical single-branch arc OK; future option-forks use short-lived branches then merge to master.

---

## 2026-07-22 — DICC ops method lock (docs only; no cluster run)

**Mode:** documentation hygiene · **no train** · no DICC jobs · champion frozen  
**Primary artifact:** `docs/DICC_OPS_METHOD.md`

### Delivered
| Item | Detail |
|------|--------|
| Authoritative ops | DICC **OnDemand → VNC Desktop** for interactive setup; **`screen`** for long tasks; prefer **batch** `run_campaign.sh` |
| Superseded (removed as primary plan) | Campus-stable runner; Cheran-as-default cluster operator; long interactive `srun`/`salloc` over VPN |
| Updated surfaces | `FINAL_PLAN`, `04_PHASE0_DICC`, `PROF_POR_STATUS_REPORT`, `STATUS_REPORT_DRAFT`, `PROF_POR_3DAY`, `DICC_RUNBOOK`, `dicc_scripts/README`, WP0, audit 07, HANDOFF/continuity, tracker A3/L1 |
| Scientific DICC goals | Unchanged (V100S/A100, Day1+Day2, Block3 CUDA vs PT, full V3 PT absolute, Option A) |
| SUCCESS tree | Still **ABSENT** — method locked; execution pending dedicated session |

**Decision: DONE (docs)** — next dedicated DICC session executes under OnDemand VNC method only.

---

## 2026-07-22 — Playlist closure audit + claims hygiene (no train)

**Mode:** audit / packaging only · **no train** · GPU idle · champion frozen  
**Primary artifacts:**
- `docs/execution_plan/PLAYLIST_CLOSURE_AUDIT.md`
- `scripts/build_claims_package.py` (Table 1b per-class claims + open_gates/advantage hygiene)
- `docs/execution_plan/CLAIMS_REGISTRY.md` (rebuilt, **64** claims)
- `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.{md,pdf}` (claims count 64; PDF rebuild ~797 KB)

### Delivered
| Item | Detail |
|------|--------|
| Tracker parse | **133/133 terminal** (DONE 66 · RUN_DOCUMENTED 52 · INCORPORATED 5 · BLOCKED 10 · TODO/PARTIAL **0**) |
| Table 1b re-verify | DDoS 0.9838 / DoS 0.9813 / Normal 0.9292 / Recon 0.9958 / Theft 1.0 — match seed JSON means |
| Claims | **59 → 64** (`bot_sealed_test_pc_*` LOCKED_TEST) |
| Open gates | DICC + PI venue/BibTeX only (WP9b removed as stale) |
| WP board | WP0/WP0b **BLOCKED(ops)**; WP6c **BLOCKED(N/A)** champion unchanged |
| Tracker hygiene | L7 playlist-open note closed; B8→B9 class-weights closed; L11 notes 64 |
| Phase docs | 04 Phase0 BLOCKED; 13 Phase9 sections DONE (PI polish) |
| Claims | **64** — `verify_claims` green |
| Champion | **unchanged** `80a90f7cc210276300eaa90173a5a385` |
| Still open | DICC ops BLOCKED; PI authors/venue/BibTeX |

**Decision: DONE (local playlist closed)** — no further science rows open under full playlist law.  
**Next:** DICC when user opens dedicated session **or** PI fills author/venue/BibTeX.

---

## 2026-07-22 — PI venue polish of camera-ready draft (no train)

**Mode:** writing / packaging only · **no train** · GPU idle · champion frozen  
**Primary artifacts:**
- `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` (venue-polished prose)
- `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf` (~797 KB rebuild)
- `scripts/build_manuscript_pdf.py` (reproducible ReportLab builder)

### Delivered
| Item | Detail |
|------|--------|
| Abstract | Continuous journal-style (5-part content preserved; labels removed) |
| Front matter | Author / affiliation / correspondence / venue placeholders for PI |
| Table 1b | Multi-seed **test** per-class F1 means from sealed_test seeds 42–46 (DDoS 0.9838 … Theft 1.0) |
| Table 5b | HPO Stage-B full-train refine ranking (trial 8 **0.9791** selected; trial 13 collapses 0.8656) |
| Table 6 | Systems ranges + mean±std / CV% / 95% CI from WP6b summary |
| Process jargon | Softened INCORPORATED/RUN_DOCUMENTED/Trap language for reader-facing prose |
| Repro / ethics | Data-availability + ethics stubs; App D PI checklist |
| Tracker | A6/L10 → DONE (PI polish); L11 notes **59** claims |
| Claims | **59** — `verify_claims` green (no number invention) |
| Champion | **unchanged** `80a90f7cc210276300eaa90173a5a385` |
| Still open | Final journal class file + BibTeX after PI venue choice; DICC ops BLOCKED |

**Decision: DONE (PI venue polish pass)** — not “publisher typeset final” until PI picks venue template.  
**Next:** DICC when user opens dedicated session **or** PI fills author/venue/BibTeX.

---

## 2026-07-22 — WP9c camera-ready manuscript draft + remaining figures

**Mode:** writing / packaging only · **no train** · GPU idle · champion frozen  
**Primary artifacts:**
- `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md` (~26 KB full prose)
- `docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf` (~850 KB camera-ready local-complete draft)
- `docs/manuscript/figures/` — architecture, class dist, dual bars, ablation, B14 CM seed42, WP6b ranges, Pareto copies

### Delivered
| Item | Detail |
|------|--------|
| Write order | Results tables → methods → intro/abstract (spine-locked numbers only) |
| Title | T1 multi-obj + Option A CUDA; J10 full LLM-XAI rejected |
| Figures | arch, class-dist, dual bars, ablation, CM, WP6b, Pareto — all from disk |
| Related work | Gap table + compact positioning (App C) |
| Tracker flips | A6/L10 → DONE (draft); WP9c DONE |
| Still BLOCKED (ops) | A3, H7, I1–I5, I11, K7, WP0 DICC |
| Claims | **59** — `verify_claims` green (no number invention) |
| Champion | **unchanged** `80a90f7cc210276300eaa90173a5a385` |
| Open writing | PI venue polish only; DICC cell insert when user opens session |

**Decision: DONE (local-complete camera-ready draft)** — not “journal submission final” until PI venue format + optional DICC.  
**Next:** PI polish **or** DICC when user opens dedicated session.

---

## 2026-07-22 — WP9b manuscript spine + residual PARTIAL tracker flips

**Mode:** documentation / packaging only · **no train** · GPU idle · champion frozen  
**Primary artifact:** `docs/execution_plan/WP9b_MANUSCRIPT_SPINE.md`

### Delivered
| Item | Detail |
|------|--------|
| Title policy | T1 multi-obj + Option A CUDA; full LLM-XAI title rejected (J10) |
| Abstract | Five-part Prof structure drafted from locked numbers |
| RQ answers | K4/K5 **DONE**; K6 local DONE; K7 BLOCKED DICC; K8 RUN_DOCUMENTED |
| Core tables | B14, classical, ablation, WP6b, XAI, ToN — all from disk |
| ToV addendum | Protocol-era threats (val/test, dual bars, local≠portable, Option A) |
| Tracker flips | A1/A2/A4–A6, C1, I6/I9/I10, K4/K5, L1/L10/L12 → terminal statuses |
| Still BLOCKED (ops) | A3, H7, I1–I5, I11, K7, WP0 DICC |
| Claims | **59** — `verify_claims` green (no number invention) |
| Champion | **unchanged** `80a90f7cc210276300eaa90173a5a385` |

**Decision: DONE (spine)** — camera-ready PDF + arch/class-dist figures remain writing pass.  
**Next:** PDF drafting from spine **or** DICC when user opens dedicated session.

---

## 2026-07-22 — WP6b local multi-session latency/energy ranges (Option A)

**Mode:** systems only · no train · champion frozen · wall **~147 s**  
**Script:** `scripts/run_wp6b_local_ranges.py`  
**Tag:** `wp6b_local_ranges/` + mirror `systems_i8_h3/`  
**Platform:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  
**Sessions:** **5** · warm-up **50** · timed iters **100** · energy iters **400** · CUDA trials/session **20**  
**Champion md5:** **unchanged** `80a90f7cc210276300eaa90173a5a385`

| Metric | Session-mean range | Mean ± std | CV% | 95% CI |
|--------|-------------------|------------|-----|--------|
| PT µs/sample @bs=256 | **24.15–25.68** | **24.90 ± 0.55** | 2.22 | [24.21, 25.59] |
| PT µs/sample @bs=128 | **29.61–31.32** | **30.59** | — | — |
| PT µs/sample @bs=1 | **1806–2396** | **1977** | high session drift | — |
| Energy mJ/flow @bs=128 | **0.920–0.943** | **0.933 ± 0.010** | 1.05 | [0.920, 0.945] |
| Energy thrput flows/s | **42085–42709** | **42322** | 0.55 | — |
| CUDA block3 FP16 µs | **503.2–508.5** | **505.6 ± 2.2** | 0.44 | — |
| CUDA derived pipeline µs | **565.2–570.3** | **567.4 ± 1.9** | 0.34 | — |
| Peak alloc VRAM (global max) | **322.2 MiB** | — | — | H3 |
| Batch256 peak alloc mean | **103.2 MiB** | — | — | I8 |

**I8 batch sensitivity:** bs∈{1,8,32,64,128,256,512,1024} multi-session µs/sample ranges written.  
**I7 warm-up:** 50 discarded sync forwards before each timed block.  
**Option A:** CUDA figures are per-block / derived pipeline sum — **not** full V3 parity.  
**Historical single-shot energy 0.786** mJ/flow remains labeled **HISTORICAL**; WP6b range is primary for multi-session claims.  
**Decision: DONE** — local ranges complete. DICC multi-day still BLOCKED.  
**Claims:** rebuild → **59** claims; `verify_claims` all green.

**Next:** WP9b manuscript spine when tracker largely green → DICC only when user opens session.

---

## 2026-07-22 — B14 sealed multi-seed BoT TEST (user lock path A)

**Mode:** full train+test · seeds 42–46 · GPU · wall **~4768 s (~79 min)**  
**User lock:** CAD-CBA-v1 · init path **A** · champion frozen  
**Init:** `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth`  
**HPs:** `config/hpo_best.yaml` · epochs≤10 patience=3  
**Script:** `scripts/run_sealed_test_b14.py` (gate-locked)  
**Tag:** `sealed_test/` · Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`

| Seed | val macro-F1 | test macro-F1 | test min-cls | test Theft |
|------|--------------|---------------|--------------|------------|
| 42 | 0.9791 | **0.9787** | 0.9333 | 1.0000 |
| 43 | 0.9587 | **0.9798** | 0.9369 | 1.0000 |
| 44 | 0.9797 | **0.9798** | 0.9375 | 1.0000 |
| 45 | 0.9787 | **0.9722** | 0.9014 | 1.0000 |
| 46 | 0.9483 | **0.9796** | 0.9369 | 1.0000 |

| Aggregate | Value |
|-----------|-------|
| **Test macro-F1 mean±std** | **0.9780 ± 0.0033** (n=5) |
| Test min-cls mean | **0.9292** |
| Test Theft mean | **1.0000** |
| Val macro-F1 mean±std | 0.9689 ± 0.0145 (matches HPO confirm) |

**Decision: DONE** — sealed multi-seed test after explicit freeze lock.  
**Do not mix** test 0.9780 with val-only WP1b 0.9714 without labeling.  
**Comparators:** protocol RF val 0.9778 · LGBM val 0.9818 · HPO val seed42 0.9791.  
**Claims:** rebuilt → 46 claims; `bot_sealed_test_*` LOCKED_TEST; `verify_claims` all green.  
**Note:** seed46 val weaker (0.9483) but test strong (0.9796) — report both; do not hide.

**Next:** WP6b local multi-session ranges → WP9b manuscript when tracker green → DICC when user opens session.

---

## 2026-07-22 — WP9a claims packaging + freeze card (no train / no sealed test)

**Mode:** packaging + tracker hygiene only. GPU idle. Champion md5 unchanged.

### Deliverables
| Artifact | Path |
|----------|------|
| Builder | `scripts/build_claims_package.py` |
| Claims JSON | `benchmarks/results/claims_package/protocol_claims.json` (42 claims, 11 minority rows) |
| Table (local) | `benchmarks/results/claims_package/table.md` |
| Registry (committed) | `docs/execution_plan/CLAIMS_REGISTRY.md` |
| Freeze gate | `docs/execution_plan/FINAL_CONFIG_FREEZE_CARD.md` **AWAITING USER LOCK** |
| Verifier | `scripts/verify_claims.py` extended with protocol claims → **all green** |
| Prose | `docs/paper_text_blocks.md` §11 Protocol-era numbers |

### Headlines locked (from disk; not invented)
WP1b **0.9714±0.0109** · HPO **0.9791** · LGBM **0.9818** · ensemble KD **0.9401** · XAI rank **0.9636** / faith **0.5109** / feature-mention **0.333** · energy **0.786** mJ/flow · composite G6 **0.9056** · ToN test **0.8110** vs RF **0.9393** · J10 DROP full XAI claim.

### Explicitly not done this session
- B14 sealed multi-seed BoT **test** (needs user lock text on freeze card)
- WP6b multi-session ranges (after lock)
- WP9b manuscript spine
- DICC

**Next science:** user pastes lock → B14 sealed test (init path A recommended) → WP6b → re-run claims builder.

---

## 2026-07-22 — WP7 XAI + F9 energy + WP8 ToN + WP6a re-export (science + systems)

### WP7 XAI suite (BoT val, champion frozen)
Scripts: `scripts/run_xai_suite.py`  
Protocol: `botiot_v1` / stage_b_ft / seed 42 / **val only** (test sealed)  
Checkpoint: production champion md5 **`80a90f7cc210276300eaa90173a5a385`** (unchanged)  
Tag: `xai/` · Summary md5 `4e1d869af8cc994db31637603e4a5f5a` · wall ~19 s

| Metric | Value | Tracker |
|--------|-------|---------|
| Occlusion top-3 | min, stddev, max | J2/J8 |
| Faithfulness top-3 mass | **0.5109** | J2 |
| Rank consistency (Spearman) | **0.9636** | J3 |
| Structured usefulness mean | **1.0** (n=8 auto rubric) | J5/J9 |
| LLM strict feature-mention | **0.333** (n=6 TinyLlama) | J6/J7 |
| Dispatch p99 overhead | **16.60 µs** | J4 |
| Generation mean | **~7400 ms** | J4 |

**Decision: RUN_DOCUMENTED** · **J10 = DROP_FULL_EXPLAINABLE_CLAIM_KEEP_STRUCTURED**  
Keep dispatch micro-result + structured evidence templates; do **not** title/abstract as full LLM-explainable IDS. shap/lime not installed (documented).

### F9 energy / systems table
Script: `scripts/run_energy_table_f9.py` (no retrain)  
Tag: `energy_table/` · RTX batch128 **~0.786 mJ/flow**; consolidates A100/cuML/latency/params  
**Decision: RUN_DOCUMENTED** — per-ablation mJ not re-measured (disclosed).

### WP8 ToN final method (CAD-CBA-v1 mapped)
Script: `scripts/run_toniot_final_method.py`  
Data: `data/processed_toniot/` **13 features**, 10 classes · seed 42  
Recipe: V3 dims + ensemble KD (α=0.6 T=10) + focal γ≈1.92 + dropouts/wd/cosine; ToN-scale kd_lr=1e-3 ft_lr=1e-4  
Wall ~295 s · BoT champion **unchanged**

| Model | val macro-F1 | test macro-F1 |
|-------|--------------|---------------|
| CAD-CBA-v1 (KD selected) | **0.8080** | **0.8110** |
| RF same-split | 0.9400 (val) | **0.9393** |
| Ensemble teacher val | 0.9618 | — |
| Historical clean CNN (26-feat) | — | 0.9526 (≠ this protocol) |

Pilot low-lr transfer (BoT FT lr) archived as `summary_pilot_lowlr.json` (test ~0.74).  
FT did not beat KD → selected KD ckpt.  
**Decision: RUN_DOCUMENTED** — honest RF gap; multi-dataset recipe evidence.

### WP6a re-export + numerical fidelity
`validate_weights.py` + `numerical_fidelity.py` · all blocks max|Δ|=0 · CUDA self-check all PASS  
`wp6_reexport/summary.json` · champion md5 unchanged.

**Next science:** sealed multi-seed **test** (B14) after explicit user lock; WP6b ranges; WP9 claims; DICC when user opens session.

---

## 2026-07-22 — Bounded C* playlist + E6 neural teacher + B2–B4 plateau (science)

### C* bounded (GPU)
Scripts: `scripts/run_bounded_cstar.py`, `model/method_variants.py`, losses ASL/SupCon in `scripts/protocol/losses.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seed **42** / epochs≤8 / patience=3 / **val only** (test sealed)  
CTRL/C7/C8: distill init + hpo_best-ish; C4/C5: scratch; C10: post-hoc MC-dropout on HPO confirm  
Tag: `cstar_bounded/` · Summary md5 `498241338cb114ae4010809302386191`  
Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.

| Rank | Row | Config | val macro-F1 | Min-cls | Theft | Decision |
|------|-----|--------|--------------|---------|-------|----------|
| — | **CTRL** | V3 focal distill+HPO | **0.9787** | 0.9333 | 1.0000 | CONTROL |
| 1 | C4 | multi-scale CNN–BiLSTM | 0.9167 | 0.7848 | 0.9091 | **RUN_DOCUMENTED** Δ−0.062 |
| 2 | C5 | gated CNN–BiLSTM | 0.9132 | 0.7442 | 1.0000 | **RUN_DOCUMENTED** Δ−0.065 |
| 3 | C8 | asymmetric multi-class loss | 0.8012 | 0.2857 | 0.2857 | **RUN_DOCUMENTED** Δ−0.178 |
| 4 | C7 | SupCon + focal (λ=0.1) | 0.7732 | 0.0000 | 0.0000 | **RUN_DOCUMENTED** Δ−0.206 |
| — | C10 | MC-dropout + entropy selective | det 0.9791 / mc 0.9767 | — | — | **RUN_DOCUMENTED** no high-cov lift; keep argmax |

**None incorporated into CAD-CBA-v1.** Package stays V3 + focal + ensemble KD + hpo_best + shuffle + argmax.

### B2–B4 arch HPO
Doc: `docs/execution_plan/B2B4_ARCH_HPO_PLATEAU_REJECT.md`  
**Decision: RUN_DOCUMENTED** — multi-seed plateau vs WP1b + KD transfer freeze + C4/C5 negative probes. No full arch Optuna.

### E6 neural teacher KD
Script: `scripts/run_neural_teacher_kd.py`  
Stage: **stage_a_kd** · teacher G11 cnn_bilstm (val 0.9494) · student V3 · α=0.6 T=10 γ=2 · epochs≤10  
Student best val macro-F1 **0.8513** ≪ WP4b ensemble **0.9401**  
**Decision: RUN_DOCUMENTED** — keep ensemble teacher INCORPORATED.  
Summary: `benchmarks/results/teachers_kd_neural/summary.json` wall ~1436 s.

### WP5c systems rebench (same day)
`scripts/run_pareto_multiobj.py` → `benchmarks/results/pareto_h8/`  
Rebench package ckpts + classical RF/XGB/LGBM CPU systems; composite G6 **0.9056**; mixed front includes LGBM/XGB.  
Complements analysis-only `pareto/` (already committed).

**Next science:** sealed multi-seed **test** (B14) after final lock; WP7 XAI or J10 drop; WP8 ToN; WP6 re-export; DICC when user opens session.

---

## 2026-07-22 — WP5c Pareto F1–latency–memory (H8, analysis)

Scripts: `scripts/run_pareto_wp5c.py` (no retrain; consolidates existing systems metrics)  
Sources: `ablation_ladder/summary.json` (A1–A7) + `baselines_neural/summary.json` (G6–G12) + classical handoff refs  
Outputs: `benchmarks/results/pareto/summary.json` md5 `893645611534c7ed681e2846d3c80246` · `table.md` · plots `pareto_f1_latency.png` / `pareto_f1_params.png`  
Latency protocol: CUDA forward val batch=256 (same harness as WP5a/b). Memory proxy: n_params + checkpoint_bytes.  
Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`. Test sealed.

| Front / score | Points | Notes |
|---------------|--------|-------|
| Best val macro-F1 | **A7** 0.9699 @ 26.02 µs/sample · 530181 params | Full CAD-CBA-v1 package |
| F1–latency Pareto | **A7, A3, G6** | Detection vs speed trade-off |
| F1–params / 3-obj | A7, A3, G6, G10, G8, A1 (+G11/G7 on 2-obj) | Slimmer models on front |
| Composite #1 (0.5 nF1 + 0.25(1−nLat) + 0.25(1−nParams)) | **G6 MLP** 0.762 · F1 0.9285 @ **4.33** µs | Efficiency-weighted; not method lock |
| Classical ref (val only) | LGBM **0.9818** · RF 0.9778 | No CUDA batch256 latency in this harness |

**Decision: DONE (H8)** — multi-obj tables written; CAD-CBA-v1 remains detection package; composite does not replace A7. Historical cuML ~2MB vs ~444MB noted as secondary VRAM evidence.

**Next science:** bounded C* (SupCon / multi-scale / gated / asymmetric / uncertainty) / B2–B4 arch HPO / E6 neural teacher. DICC only when user opens session.

---

## 2026-07-22 — G2 SVM full + G5 LGBM fix + D6 stratified batch + G13 (science)

### Classical G2/G5 (CPU)
Scripts: `scripts/run_classical_baselines.py` (LinearSVC hard labels; LGBM multiclass+balanced fix)  
Protocol: `botiot_v1` / **stage_b_ft** / seed **42** / full train (n=2,641,335) / **val only** (test sealed)  
Handoff table: `benchmarks/results/baselines_classical/summary_handoff.json` md5 `b93899d2d019f72d8deb97bef2de3b9e`  
Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.

| Model | val macro-F1 | Min-cls | Theft | train_sec | Decision |
|-------|--------------|---------|-------|-----------|----------|
| **LGBM (G5 fix)** | **0.9818** | 0.9231 | 0.9231 | 1131 | **DONE (val)** — class_weight=balanced + multiclass; tops protocol classical |
| RF (prior) | 0.9778 | 0.9231 | 0.9231 | — | DONE (val) |
| XGB (prior) | 0.9762 | 0.9231 | 0.9231 | — | DONE (val) |
| LR (prior) | 0.5231 | 0.0000 | 0.0000 | — | DONE (val) |
| **SVM LinearSVC (G2 full)** | **0.4268** | 0.0000 | 0.0000 | 44 | **RUN_DOCUMENTED** weak linear under imbalance; pilot ERROR fixed |

**G2 notes:** Pilot failed (CalibratedClassifierCV cv=3 with non-stratified subsample). Full train uses **LinearSVC dual=False**, hard labels, `class_weight=None` (protocol-fair vs RF/XGB defaults). Weak is honest — not a competitive classical bar.

**G5 notes:** Prior full run **0.5512** (Theft=0) was broken multi-class defaults. Fix: `objective=multiclass`, `class_weight=balanced`, `min_child_samples=5`, max_depth=8. **0.9818** now slightly above protocol RF. Report both legacy and fixed; official G5 = fixed.

**G13:** `docs/execution_plan/G13_LIGHTWEIGHT_IDS_NOTE.md` — **RUN_DOCUMENTED N/A** (no external public method re-implemented under protocol; use G6–G12 + classical suite).

### D6 stratified batch FT compare (GPU)
Scripts: `scripts/run_stratified_batch_compare.py`, `train_protocol_ft.py --train-sampler {shuffle,stratified}`  
Init: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` · HPs: `config/hpo_best.yaml` · loss focal · seed 42 · epochs≤8 patience=3  
Tag: `stratified_batch/` · Summary md5 `294669c7a7b2e3d318b54186a0ce917c` · Wall ~1479 s (~25 min)  
Thermal: soft 85 / hard 90; peak ~85°C; no hard trip.

| Sampler | val macro-F1 | Min-cls | Theft | elapsed |
|---------|--------------|---------|-------|---------|
| **shuffle** (control) | **0.9791** | 0.9351 | 1.0000 | 668 s |
| stratified (WeightedRandomSampler inv-freq) | 0.9209 | 0.7500 | 0.7500 | 763 s |

**Δ stratified − shuffle = −0.0582**  
**Decision: RUN_DOCUMENTED** — class-balanced stratified batches **hurt** macro and min-cls vs shuffle under CAD-CBA-v1 HPs; **keep shuffle default**. Champion unchanged.

**Also:** B9 class-weight close from existing `focal_cb` 0.9121 ≪ plain focal 0.9780 → RUN_DOCUMENTED keep no CB weights on neural focal.

**Next science:** (completed later same day → WP5c) then C* / B2–B4 / E6.

---

## 2026-07-22 — WP5b protocol-fair neural baselines G6–G12 (science)

Scripts: `scripts/run_neural_baselines.py`, `model/neural_baselines.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seed **42** / epochs≤8 / patience=3 / **CE** / scratch / **val only** (test sealed)  
Shared HPs: lr=1e-3 Adam batch=512 (G15 equal budget; no per-baseline Optuna)  
Tag: `baselines_neural/` (champion + multirun + ablation trees **not** clobbered)  
Summary: `benchmarks/results/baselines_neural/summary.json` md5 `dc85077cb129c3209d6f6148c18e925b`  
Wall ~4816 s (~80 min). Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
Thermal: soft 85 / hard 90; peak ~72°C; no hard trip.

| Rank | Row | Config | val macro-F1 | Min-cls | Theft | params | µs/sample |
|------|-----|--------|--------------|---------|-------|--------|-----------|
| 1 | **G11** | cnn_bilstm CE | **0.9493** | 0.8571 | 1.0000 | 463877 | 20.12 |
| 2 | G6 | mlp CE | 0.9285 | 0.7077 | 1.0000 | 400901 | 4.33 |
| 3 | G10 | cnn_lstm CE | 0.8159 | 0.5000 | 0.5000 | 212485 | 16.25 |
| 4 | G8 | lstm CE | 0.8099 | 0.3556 | 0.8000 | 153605 | 10.39 |
| 5 | G9 | bilstm CE | 0.8058 | 0.5000 | 0.5000 | 372229 | 16.01 |
| 6 | G7 | cnn1d CE | 0.6221 | 0.0000 | 0.0000 | 34821 | 6.15 |
| 7 | G12 | transformer CE | 0.5808 | 0.0000 | 0.0000 | 105221 | 10.72 |

**Comparators**
- WP5a A3 cnn_bilstm CE: **0.9493** — G11 exact match (architecture consistency)
- WP5a A1/A2: **0.6221 / 0.8058** — G7/G9 match
- WP5a A7 full package: **0.9699** (diff loss/init/HPO — not same row)
- WP1b multirun mean: **0.9714 ± 0.0109**
- Protocol RF val: **0.9778**
- Historical non-protocol MLP: ~0.962 — **do not mix**

**Interpretation**
- Under equal CE scratch budget, **CNN–BiLSTM (G11)** is the strongest pure neural architecture baseline.
- Protocol-fair **MLP (G6)** is competitive (0.9285) but below G11; historical 0.962 was a different pipeline.
- Lightweight **transformer (G12)** is weak under this budget (0.5808) — honest negative for free transformer claim.
- Pure **1D-CNN (G7)** insufficient alone (matches ablation A1).
- CAD-CBA-v1 package (A7 / multirun) remains above all pure CE baselines; RF still higher on protocol-fair detection.

**Decision:** G6–G12 **RUN_DOCUMENTED** (arch table); G11 also confirms G11 DONE; G15 **DONE** (equal-budget note). Champion unchanged.

**Also this session:** `scripts/finalize_neural_baselines_docs.py`; thermal `logs/thermal_guard_neural_baselines.sh`.

**Next science:** G2 SVM full + G5 LGBM fix **or** D6 stratified batch **or** WP5c Pareto / bounded C*. DICC only when user opens session.

---

## 2026-07-21 — WP5a ablation ladder A1–A7 (science)

Scripts: `scripts/run_ablation_ladder.py`, `model/ablation_variants.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seed **42** / epochs≤8 / patience=3 / **val only** (test sealed)  
Default HPs: lr=1e-3 Adam batch=512 γ=2 (A7 uses `config/hpo_best.yaml` AdamW/cosine)  
A6/A7 init: `model/teachers_kd/kd_ensemble_a0.6_T10.0_g2.0_seed42.pth`  
Tag: `ablation_ladder/` (champion + multirun trees **not** clobbered)  
Summary: `benchmarks/results/ablation_ladder/summary.json` md5 `988b826adcea79ef51f4b8144055825e`  
Wall ~5397 s (~90 min). Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
Thermal: soft 85 / hard 90; no hard trip (peak mid-70s).

| Rank | Row | Config | val macro-F1 | Min-cls | Theft | params | µs/sample |
|------|-----|--------|--------------|---------|-------|--------|-----------|
| 1 | **A7** | attn+focal+ens KD+HPO | **0.9699** | 0.8974 | 1.0000 | 530181 | 26.02 |
| 2 | A3 | cnn_bilstm CE scratch | 0.9493 | 0.8571 | 1.0000 | 463877 | 19.96 |
| 3 | A6 | attn+focal+ens KD | 0.9346 | 0.8462 | 1.0000 | 530181 | 26.39 |
| 4 | A5 | attn+focal scratch | 0.8684 | 0.7059 | 0.7059 | 530181 | 26.69 |
| 5 | A2 | bilstm_only CE | 0.8058 | 0.5000 | 0.5000 | 372229 | 15.26 |
| 6 | A4 | attn+CE scratch | 0.7378 | 0.0000 | 0.0000 | 530181 | 24.54 |
| 7 | A1 | cnn_only CE | 0.6221 | 0.0000 | 0.0000 | 34821 | 5.03 |

**Interpretation**
- Full CAD-CBA-v1 path (**A7**) wins the incremental ladder under seed42 / 8-ep budget.
- Plain **CNN–BiLSTM (A3)** is already strong (0.9493); **attention+CE alone (A4) underperforms A3** — do not claim free attention gain.
- Focal (A5), ensemble KD (A6), and HPO HPs (A7) each add ladder lift vs prior step.
- Single-seed ladder ≠ multi-seed means (WP1b 0.9714±0.0109; package 0.9639±0.0185).
- F9 systems: params + latency + CUDA mem logged; energy table still open.

**Decision:** F1–F7 **RUN_DOCUMENTED** (A7 tops table; package composition supported). Champion unchanged.

**Also this session:** `--skip-existing` resume on ladder script; `scripts/finalize_ablation_ladder_docs.py`; thermal `logs/thermal_guard_ablation.sh`.

**Next science:** WP5b neural baselines (protocol-fair) **or** D6 stratified batch. DICC only when user opens session.

---

## 2026-07-21 — Full-playlist / context hygiene (docs only)

User lock reaffirmed: **complete every tracker playlist item** (not only the critical path).  
Updated `PROF_FEEDBACK_TRACKER.md` from **existing disk/docs evidence** (no new training):

- Closed / upgraded where evidence already existed: B10 (historical α/T sweeps + recipe), B11 (seq len locked), C13, D8, D10, F3/F4/F8, G6 historical, G11, G14, H1/H5, J1, K1, L2–L4/L7/L9, etc.
- Open rows explicitly tagged **Playlist required** (SupCon, stratified batch, arch HPO B2–B4, neural baselines, XAI, ToN, Pareto, sealed test, …).
- Continuity + HANDOFF policy text aligned.

**No invented numbers. No science re-runs this micro-pass.**

---

## 2026-07-21 — Multi-seed HPO confirm (original distill + hpo_best) (science)

Scripts: `scripts/run_hpo_multiseed_confirm.py`, `scripts/train_protocol_ft.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seeds 42–46 / epochs≤10 patience=3 / **val only** (test sealed)  
Init: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth` (same as WP3 Optuna study)  
HPs: `config/hpo_best.yaml` (WP3 winner train recipe)  
Tag: `multirun_hpo_confirm/` (does **not** clobber WP1b `multirun/` or package `multirun_ensemble_hpo/`)  
Summary: `benchmarks/results/multirun_hpo_confirm/summary.json` md5 `ef6c92a592474c321a4c1300e19a8065`  
Wall ~2981 s (~50 min). Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
Thermal: soft 85 / hard 90; peak ~81°C; no hard trip.

| Seed | Best val macro-F1 | Min-cls | Theft | Elapsed |
|------|-------------------|---------|-------|---------|
| 42 | **0.9791** | 0.9351 | 1.0000 | ~8.1 min |
| 43 | 0.9587 | 0.9091 | 0.9091 | ~14.1 min |
| 44 | **0.9797** | 0.9367 | 1.0000 | ~8.1 min |
| 45 | 0.9787 | 0.9333 | 1.0000 | ~5.7 min |
| 46 | 0.9483 | 0.8333 | 0.8333 | ~12.0 min |

**Mean 0.9689 ± 0.0145** (n=5) · min 0.9483 · max 0.9797  
min-cls mean **0.9095** · Theft mean **0.9485**

**Comparators**
- WP3 HPO full-train seed42: **0.9791** — seed42 **reproduces** (0.979055 ≈ 0.979064)
- WP1b multirun (old distill + default HPs): **0.9714 ± 0.0109**
- Package ensemble KD + HPO: **0.9639 ± 0.0185**

**Interpretation**
- Fair multi-seed confirm of Optuna train HPs on the **same init** as the study.
- Seed42 is a clean repro of the WP3 winner; seed44 slightly higher (0.9797).
- Aggregate mean is **slightly below** WP1b default-HP multirun and **higher variance** (seed46 0.9483 and seed43 0.9587 drag).
- HPO HPs remain **INCORPORATED** as CAD-CBA-v1 train defaults (credible seed42 lift vs default 0.9780); multi-seed aggregate is **RUN_DOCUMENTED** evidence of seed sensitivity, not a mean-win claim over WP1b.
- Do not promote HPO multirun mean as “beats baseline multirun.”

**Decision: RUN_DOCUMENTED** (multi-seed aggregate) · train HPs stay **INCORPORATED** from WP3.

**Next science:** WP5a ablation ladder **or** neural baselines / Pareto. DICC only when user opens session.

---

## 2026-07-21 — Package FT multirun: ensemble KD init + HPO HPs (science)

Scripts: `scripts/train_protocol_ft.py` (HPO-aware AdamW/cosine/dropout), `scripts/run_package_ft_multirun.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seeds 42–46 / epochs≤10 patience=3 / **val only** (test sealed)  
Init: `model/teachers_kd/kd_ensemble_a0.6_T10.0_g2.0_seed42.pth`  
HPs: `config/hpo_best.yaml` (lr≈5.89e-5, batch=1024, γ≈1.92, drop≈0.148, att≈0.214, wd≈1.92e-4, cosine, AdamW)  
Summary: `benchmarks/results/multirun_ensemble_hpo/summary.json` md5 `1fa206e34c50e799d531f5eee70629e8`  
Wall ~84 min. Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
WP1b `multirun/` tree **not clobbered**.

| Seed | Best val macro-F1 | Min-cls | Theft | Elapsed |
|------|-------------------|---------|-------|---------|
| 42 | 0.9741 | 0.9333 | 1.0000 | ~14 min |
| 43 | 0.9328 | 0.8000 | 0.8000 | ~29 min |
| 44 | 0.9699 | 0.8947 | 1.0000 | ~13 min |
| 45 | **0.9803** | **0.9474** | **1.0000** | ~12 min |
| 46 | 0.9623 | 0.9091 | 0.9091 | ~14 min |

**Mean 0.9639 ± 0.0185** (n=5) · min 0.9328 · max 0.9803  

**Comparators**
- WP1b multirun (old distill + default HPs): **0.9714 ± 0.0109**
- WP3 HPO full-train seed42: **0.9791**
- Package seed45 max **0.9803** exceeds HPO seed42 point estimate

**Interpretation**
- Full CAD-CBA-v1 train path (ensemble KD → FT with Optuna HPs) is **run and documented**.
- Aggregate mean is **slightly below** WP1b and **higher variance** (seed 43 drags).
- Honest finding: HPO HPs optimized on old distill init do not transfer into a better multi-seed mean on ensemble KD init.
- Early epochs often collapse Theft then recover by ep3–4 (documented dynamics, not a bug; zero-train KD eval still 0.9401).
- **Decision: RUN_DOCUMENTED** for package multirun aggregate (not a mean win over WP1b). Component decisions unchanged (ensemble teacher, HPO HPs, focal, argmax).

**Also this session (tooling, not yet run to completion)**
- `model/ablation_variants.py` + `scripts/run_ablation_ladder.py` (WP5a ready)
- `scripts/run_hpo_multiseed_confirm.py` (multi-seed HPO on original distill init)
- Thermal guard `logs/thermal_guard.sh` (soft 85°C / hard pause 90°C)

**Next science:** multi-seed HPO confirm (n≥5, original init) **or** WP5 ablations (A1–A7) **or** neural baselines. DICC only when user opens session.

---

## 2026-07-21 — WP3 Optuna HPO under protocol (science)

Script: `scripts/hpo_optuna_botiot.py`  
Protocol: `botiot_v1` / **stage_b_ft** / seed 42 / full val / **test sealed**  
Init: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth`  
Arch: fixed `cnn_bilstm_v3_attention` (CAD-CBA-v1; no arch search this WP)  
Study: `botiot_stage_b_ft_hpo_v1` · SQLite `benchmarks/results/hpo/study.db`  
Summary: `benchmarks/results/hpo/summary.json` md5 `5ba39a920706100b13975e89c3b20924`  
Winner config: `config/hpo_best.yaml` md5 `598d87c9ea5d0f26847ce7b860a0eb68`  
Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
Wall ~69 min (Stage A ~38.6 min + refine ~29.7 min).

### Stage A (explore)
- n_trials=20 · epochs≤4 · patience=2 · **max_train=400_000** stratified (val full)  
- COMPLETE **11** / PRUNED **9**  
- Best Stage A: trial **11** val macro-F1 **0.9787** (min-cls 0.9351, Theft 1.0)

### Stage B (full-train refine top-3)

| Rank | Source trial | Full-train val macro-F1 | Min-cls | Theft | Bal-acc | Decision |
|------|--------------|-------------------------|---------|-------|---------|----------|
| **1 (winner)** | **8** | **0.9791** | **0.9351** | **1.0000** | **0.9863** | **INCORPORATE** |
| 2 | 11 | 0.9721 | 0.9014 | 1.0000 | 0.9645 | RUN_DOCUMENTED (Stage-A best collapsed) |
| 3 | 13 | 0.8656 | 0.5000 | 0.5000 | 0.8203 | RUN_DOCUMENTED (unstable) |

Baseline ref (multirun seed42 default HPs): **0.9780** · Δwinner **+0.0010**

### Winner train HPs (CAD-CBA-v1)
| HP | Value |
|----|-------|
| lr | 5.893e-5 |
| batch_size | 1024 |
| focal_gamma | 1.917 |
| dropout_rate | 0.148 |
| attention_dropout | 0.214 |
| weight_decay | 1.916e-4 |
| scheduler | cosine |

Ckpt: `model/hpo/refine_rank2_trial008_seed42.pth` md5 `f9360aec2c003815140823cfe9b2a386`

**Interpretation**
- Controlled Optuna search beats default multirun seed42 HPs slightly but **credibly** under full protocol val.  
- Cosine + lower lr + larger batch cluster on Stage A; Stage B refine is required (trial 11 did not transfer).  
- Arch dims not searched (KD init / CAD-CBA-v1 freeze) — WP2c only if plateaus.  
- Test still sealed; multi-seed confirm + FT from ensemble KD remain open.

Tracker: B1/B6–B8/L6/WP3 updated. CAD-CBA-v1 train HPs → `hpo_best.yaml`.

**Next science:** FT multirun from ensemble KD + HPO HPs **or** WP5 ablations **or** multi-seed HPO confirm.

---

## 2026-07-21 — WP4b teacher/KD under protocol (science)

Scripts: `scripts/train_protocol_kd.py`, `scripts/run_teacher_kd_compare.py`  
Protocol: `botiot_v1` / **stage_a_kd** / seed 42 / full train / **val only** (test sealed)  
Recipe: α=0.6, T=10.0, focal γ=2.0, epochs≤10, patience=4, batch=512, lr=1e-3  
Summary: `benchmarks/results/teachers_kd/summary.json` md5 `63ab4bd3f40e24adc6788fa1ca255bd8`  
Wall ~6879 s (~1.9 h). Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.

| Rank | Teacher | Student val macro-F1 | Min-cls | Theft | Teacher val | Decision |
|------|---------|----------------------|---------|-------|-------------|----------|
| 1 | **ensemble** (RF+XGB+LGBM mean) | **0.9401** | 0.8434 | 0.9231 | 0.9803 | **INCORPORATE** |
| 2 | rf | 0.9346 | 0.8000 | **1.0000** | 0.9750 | RUN_DOCUMENTED fallback |
| 3 | none (hard-label focal) | 0.9326 | 0.8409 | 0.9231 | — | RUN_DOCUMENTED control |
| 4 | xgb | 0.9270 | 0.8434 | 0.8571 | **0.9918** | RUN_DOCUMENTED |
| 5 | lgbm | 0.8829 | 0.7059 | 0.7059 | 0.5928 | RUN_DOCUMENTED (weak) |

**Interpretation**
- Best **student** is ensemble soft labels — not solo XGB despite XGB’s highest teacher hard-label F1.
- RF KD still strong and simpler; Theft F1=1.0 on best RF student ckpt.
- Hard-label `none` nearly matches RF KD (Δmacro ≈ +0.002 for RF) — KD lift modest under this budget.
- LGBM alone is a poor teacher on stage_a_kd (mirrors classical LGBM weakness).
- Numbers are **stage_a from-scratch KD**, not stage_b FT (do not mix with multirun mean 0.9714).

Ckpts: `model/teachers_kd/kd_{none,rf,xgb,lgbm,ensemble}_a0.6_T10.0_g2.0_seed42.pth`  
Tracker: E1–E5/E7/C9/WP4b updated. CAD-CBA-v1 KD teacher → **ensemble**.

**Next science:** WP3 Optuna HPO **or** WP5 ablations/neural baselines **or** stage_b FT from ensemble KD init.

---

## 2026-07-21 — WP2d val thresholds (science)

Script: `scripts/run_val_thresholds.py` (+ hardened `scripts/protocol/thresholds.py`)  
Checkpoint: `model/imbalance_loss/ft_focal_seed42.pth` md5 `170eaccc584ba12cce2a34ca52ebfbf2`  
Protocol: `botiot_v1` / `stage_b_ft` / seed 42 / **val only** (test sealed)  
Result: `benchmarks/results/imbalance_loss/thresholds_focal_seed42.json` md5 `2a6bc98d967883efc53d326535cf9d5b`

| Variant | Val macro-F1 | Min cls F1 | Theft F1 | Normal F1 |
|---------|--------------|------------|----------|-----------|
| argmax (baseline) | **0.9780** | 0.9315 | 1.0000 | 0.9315 |
| fixed t=0.5 | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| search macro_f1 | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| search min_per_class_f1 | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| joint macro→min | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| search class_f1 Theft | 0.9780 | 0.9315 | 1.0000 | 0.9315 |
| search class_f1 Normal | 0.9780 | 0.9315 | 1.0000 | 0.9315 |

**Decision: RUN_DOCUMENTED** — Δmacro=0, Δmin=0 vs argmax. Keep default **argmax** decode for CAD-CBA-v1.  
Interpretation: focal FT probabilities are already decisive on val; per-class thresholds add no selection signal for this checkpoint.  
Champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`.  
Tracker: B12 / C11 / D7 → RUN_DOCUMENTED. WP2d DONE.

**Next science:** WP4b teachers/KD under protocol **or** WP3 Optuna **or** WP5 ablations/neural baselines.

---

## 2026-07-21 — Handoff closure (docs only)

- **No training / no new science runs** (user request).
- Aligned tracker G3–G5, D*, C6, L5 with completed runs.
- Wrote / refreshed: `SESSION_CONTINUITY.md`, `HANDOFF.md`, `RESULTS_DISK_MANIFEST.md`, `METHOD_PACKAGE_DECISION.md`, `15_WORK_PACKAGES.md`.
- Jobs: train/multirun/imbalance **idle** (completed 2026-07-19).
- DICC: still **ABSENT**.
- Next chat: verify disk → continue tracker from continuity §5.

---

## 2026-07-19 — Foundation + first experiment wave

### Protocol foundation
- `scripts/protocol/*` live (`botiot_v1`, metrics, losses, thresholds, result_schema)
- Champion sealed eval val macro-F1 **0.9780** (`stage_b_ft`, seed 42)
- Production champion md5 **unchanged** `80a90f7cc210276300eaa90173a5a385`

### Fine-tune multirun (WP1b) — COMPLETE
Init checkpoint: `model/best_model_botiot_distill_a0.6_T10.0_focal2.pth`  
Scripts: `train_protocol_ft.py`, `run_baseline_multirun.py`  
Summary: `benchmarks/results/multirun/summary.json`

| Seed | Best val macro-F1 | Min per-class F1 | Elapsed (approx) |
|------|-------------------|------------------|------------------|
| 42 | 0.9780 | 0.9315 | 28 min |
| 43 | 0.9578 | 0.9091 | 25 min |
| 44 | **0.9840** | **0.9589** | 36 min |
| 45 | 0.9746 | 0.9143 | 16 min |
| 46 | 0.9624 | 0.9091 | 28 min |

**Mean 0.9714 ± 0.0109** (n=5) · min 0.9578 · max 0.9840  
Test sealed. Champion not overwritten.  
Ckpts: `model/multirun/ft_seed{42..46}.pth`

### Classical baselines (protocol `stage_b_ft`, val only) — DOCUMENTED
Script: `run_classical_baselines.py`  
**Authoritative table:** `benchmarks/results/baselines_classical/summary_handoff.json`  
(Note: `summary.json` may only hold the last single-model run — do not use alone.)

| Model | Train | Val macro-F1 | Min cls F1 | Theft F1 | Decision |
|-------|-------|--------------|------------|----------|----------|
| LR | full | 0.5231 | 0.0 | 0.0 | weak linear baseline |
| RF | full | **0.9778** | 0.9231 | 0.9231 | strong; protocol-fair |
| XGB | full | **0.9762** | 0.9231 | 0.9231 | strong |
| LGBM | full | 0.5512 | 0.0 | 0.0 | RUN_DOCUMENTED weak |
| SVM pilot 150k | subsample | FAILED | — | — | &lt;3 samples/class |

**Note:** Published README RF **0.9864** uses `rf_baseline_processed` — **different** from protocol-fair RF **0.9778**.

Pilot 100k earlier: LR 0.463 / RF 0.686 — RUN_DOCUMENTED pilot only.

### Imbalance loss compare — COMPLETE
Script: `run_imbalance_loss_compare.py` (5 ep, seed 42)  
Summary: `benchmarks/results/imbalance_loss/summary.json`

| Loss | Best val macro-F1 | Min cls F1 | Bal acc | Decision |
|------|-------------------|------------|---------|----------|
| ce | 0.9755 | 0.9189 | 0.975 | RUN_DOCUMENTED |
| **focal** | **0.9780** | **0.9315** | 0.975 | **INCORPORATE (default)** |
| focal_cb | 0.9121 | 0.8000 | 0.984 | RUN_DOCUMENTED (hurts macro) |
| logit_adj | 0.9225 | 0.8000 | 0.986 | RUN_DOCUMENTED (hurts macro) |

Ckpts: `model/imbalance_loss/ft_{ce,focal,focal_cb,logit_adj}_seed42.pth`

### Method package
Signed: **CAD-CBA-v1** in `METHOD_PACKAGE_DECISION.md` (keep V3 arch + **focal** + KD path; thresholds/HPO next).

### Explicitly NOT started this arc
- Optuna HPO  
- Ablation ladder / neural baselines suite  
- Teacher/KD under protocol (WP4b)  
- DICC multi-day  
- XAI quality suite  
- ToN final method  
- Manuscript  

*(Val thresholds: completed later same calendar day — see WP2d section above.)*

---

## Desirability
Multirun + protocol-fair RF/XGB put neural FT in the **same val band** as strong trees under one protocol — desirable baseline before HPO/method push. Focal remains best loss in the four-way compare.
