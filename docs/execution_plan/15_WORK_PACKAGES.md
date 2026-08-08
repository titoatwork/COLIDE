# 15 — Work Packages (One Package ≈ One Focused Session)

**Rule:** Do not stack DICC + full HPO + manuscript in one chat.  
**Order:** Respect phase dependencies.  
**Playlist law:** every WP below must reach DONE / RUN_DOCUMENTED / BLOCKED(ops). No silent skips; “optional” experiments still get a bounded run.  
**Live status:** update this board every session; authoritative narrative in `SESSION_CONTINUITY.md`.  
**Git:** land WPs on **`master`** (final). Open a branch only for a **true alternative option**; keep branch count low — `docs/BRANCHING_POLICY.md`.

---

## Package board

| ID | Name | Phase | Depends | Owner | Status |
|----|------|-------|---------|-------|--------|
| WP0 | UM DICC Day1+Day2+compare+scp | 0 | — | User + agent | **PARTIAL** — S1+S2+Day2 SUCCESS on cluster (6 runs); **laptop rsync + formal compare still open**. Flags: `docs/DICC_RESULTS_AND_FLAGS.md` |
| WP0b | Extract DICC JSON tables + fork decision | 0 | WP0 | Agent+user | **PARTIAL** — means extracted from cluster JSON; re-verify after rsync; **fork: B3 CUDA loses to PT on servers** |
| WP1a | Unified protocol + freeze card | 1 | — (prep ok) | Agent | **DONE 2026-07-19** |
| WP1b | Multi-run baseline driver (5 seeds) | 1 | WP1a | Agent | **DONE 2026-07-19** mean 0.9714±0.0109 |
| WP2a | Sign method package (MOD table) | 2 | — | User+agent | **DONE** CAD-CBA-v1 |
| WP2b | Implement losses + thresholds modules | 2 | WP1a | Agent | **DONE** |
| WP2b-run | Imbalance loss 4-way compare | 4 | WP2b | Agent | **DONE** focal INCORPORATE |
| WP2c | Architecture delta if any (attention/multi-scale) | 2 | WP2a | Agent | **DONE 2026-07-22** B2–B4 plateau reject + C4/C5 bounded probes RUN_DOCUMENTED |
| WP2d | Val threshold search on focal best | 2/4 | WP2b-run | Agent | **DONE 2026-07-21** RUN_DOCUMENTED keep argmax |
| WP3 | Optuna HPO study + sealed test | 3 | WP1–2 | Agent | **DONE 2026-07-21** HPs 0.9791; **B14 sealed test DONE 2026-07-22** test **0.9780±0.0033** path A |
| WP3b | Package FT multirun ensemble KD + HPO HPs | 3/4 | WP3, WP4b | Agent | **DONE 2026-07-21** mean 0.9639±0.0185 RUN_DOCUMENTED (not mean-win vs WP1b) |
| WP3c | Multi-seed HPO confirm (orig distill + hpo_best) | 3 | WP3 | Agent | **DONE 2026-07-21** mean 0.9689±0.0145 RUN_DOCUMENTED (seed42 0.9791 repro; not mean-win vs WP1b) |
| WP4a | Imbalance strategy sweep (beyond loss) | 4 | WP1–3 | Agent | **DONE 2026-07-22** D6 stratified RUN_DOCUMENTED; C7/C8/D9 SupCon/ASL RUN_DOCUMENTED |
| WP4b | Teacher sweep (RF/XGB/LGBM/ensemble) | 4 | WP1–3 | Agent | **DONE 2026-07-21** ensemble student 0.9401 INCORPORATE |
| WP5a | Full ablation ladder | 5 | Final recipe | Agent | **DONE 2026-07-21** A1–A7 seed42; A7 0.9699 tops ladder; F1–F7 RUN_DOCUMENTED |
| WP5b | Fair baseline suite | 5 | WP1a | Agent | **DONE 2026-07-22** neural G6–G12 + classical G1–G5 (SVM 0.4268; LGBM 0.9818) + G13 N/A |
| WP5c | Pareto figure + composite score | 5 | WP5a/b + systems | Agent | **DONE 2026-07-22** `pareto/` + `pareto_h8/` systems rebench; G6 composite; classical refs |
| WP6a | Re-export + fidelity | 6 | Final ckpt | Agent | **DONE 2026-07-22** re-export + bit-identical fidelity PASS |
| WP6b | Local re-bench ranges Option A | 6 | WP6a | Agent | **DONE 2026-07-22** multi-session n=5; energy 0.920–0.943; PT@256 24.15–25.68 µs; CUDA pipe 565–570 µs |
| WP6c | Re-DICC if model changed | 6 | WP6a | User/ops | **BLOCKED (N/A)** — champion md5 unchanged; re-open only if champion replaced + BACKUP |
| WP7 | XAI suite **or** drop explainable claims | 7 | Final detector | Agent | **DONE 2026-07-22** suite RUN_DOCUMENTED; J10 drop full claim keep structured+dispatch |
| WP8 | ToN final-method eval | 8 | Final recipe | Agent | **DONE 2026-07-22** val 0.8080 test 0.8110 RF 0.9393 RUN_DOCUMENTED |
| WP9a | Numbers-match + claim JSON packaging | 9 | All | Agent | **DONE 2026-07-22** + post-B14/WP6b/Table1b rebuild (**64** claims, sealed LOCKED_TEST, verify green) |
| WP9b | Manuscript spine + figures | 9 | WP9a | User+agent | **DONE 2026-07-22** spine `WP9b_MANUSCRIPT_SPINE.md` (title/abstract/RQs/tables/ToV) |
| WP9c | Camera-ready draft PDF + remaining figures | 9 | WP9b | User+agent | **DONE 2026-07-22** draft + **PI venue polish** (continuous abstract, Table 1b/5b, systems CI/CV, `scripts/build_manuscript_pdf.py`); journal class/BibTeX = PI; DICC cells open |

---

## What to run **right now** (next chat after verify)

| Priority | Action |
|----------|--------|
| 0 | Verify disk + `verify_claims` green (expect **64** claims) |
| 1 | **DICC** only when user opens dedicated session — else do not start multi-GPU work |
| 2 | Optional: PI fills authors / venue class file / BibTeX — **no invent numbers** |
| 3 | Keep `verify_claims` green after any prose/number edits; rebuild PDF via `scripts/build_manuscript_pdf.py` |
| Never | Invent multi-day / multi-GPU numbers; full CUDA vs full V3 parity; clobber champion without BACKUP |

---

## WP0 detail (highest priority when unblocked)

**Ops method (locked):** `docs/DICC_OPS_METHOD.md`  
- Browser → **DICC OnDemand → VNC Desktop**  
- Terminal → **`screen -S colide`**  
- Prefer **batch** `bash dicc_scripts/run_campaign.sh` (Day1 then Day2)  
- **Superseded:** campus-stable runner / third-party operator as default; long interactive `srun`/`salloc` over VPN  

**Inputs:** tarball or git tree on DICC, champion md5 `80a90f7…`  
**Outputs:** `benchmarks/results/dicc/**/SUCCESS`, compare outcome, laptop copy  
**Exit:** Phase 0 checklist in `04_PHASE0_DICC.md` complete  

---

## WP1a / WP1b exit (achieved)

- Protocol: `scripts/protocol/*` + freeze card  
- Multirun: 5 seeds, summary JSON, ckpts under `model/multirun/`  
- Test sealed; production champion md5 unchanged  

---

## Tracking

Update Status column in this file when a WP completes. Link commit hash in HANDOFF.  
Numbers inventory: `RESULTS_DISK_MANIFEST.md`.
