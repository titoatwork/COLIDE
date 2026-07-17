# COLIDE — Final execution plan (locked)

**Date:** 2026-07-17  
**Authority:** `docs/DESIGN_PLAN.md` (Option A) + this document.  
**PI:** Prof. Dr. Por Lip Yee · Support: Cheran (manuscript; cluster ops only if Prof agrees).  
**Operator constraint:** Remote SSH to DICC is unstable (India→UM). Prefer campus-stable runner or Cheran on his own account. **No credential sharing.**

---

## 1. Strategy freeze (do not reopen)

| Rule | Detail |
|------|--------|
| **Option A** | Valid **per-block** Custom CUDA vs PyTorch only |
| **Primary head-to-head** | **Block 3 (BiLSTM)** same GPU, same session |
| **Forbidden claim** | Full-pipeline Custom CUDA vs **full** PyTorch V3 speedup (attn/LN/GAP gap) |
| **Accuracy** | Systems paper; RF still higher (0.9864 vs 0.9790); do not claim SOTA / beating RF |
| **Champion** | `model/best_model_botiot_twostage.pth` md5 `80a90f7cc210276300eaa90173a5a385` — **no training, no clobber** |
| **Official cluster** | **UM DICC** only (not Rostam) |
| **June 2026 551/592 µs** | Legacy single-shot only; not multi-day official |
| **Invented numbers** | Never |

**Venue framing:** FGCS-leaning **systems / measurement** paper (CUDA inference + multi-platform protocol + LLM dispatch). IoT journal only if application/deployment is written carefully; accuracy is not the headline.

---

## 2. Current state (2026-07-17)

| Item | Status |
|------|--------|
| Design plan Option A | **Approved** |
| Local accuracy / laptop latency ranges / LLM / fidelity | **Frozen, ready** |
| Laptop tarball | `~/colide-master-for-dicc.tar.gz` (~357 MB, master `@ e6f55f1`) |
| User DICC home tree | Unpacked earlier; champion md5 verified |
| Partitions (from `sacct` jobs 363046/363047) | **`gpu-v100s`**, **`gpu-a100`**; GRES **`gpu:1`** |
| Day 1 / Day 2 new campaign | **Not completed** (SSH freezes; conda/`set -u`; then access strategy shifted) |
| Prof email | Sent: bottleneck + ask guidance re Cheran helping on cluster if appropriate |
| **Hard gate before any Prof “final” numbers email** | **Codebase-wide numbers match** + `verify_claims.py` green (see P2) |
| Pre-manuscript clock | **~1–1.5 weeks typical** after DICC runner unblocked (best ~4–7 days incl. match pass) |

---

## 3. Phased plan

### Phase P0 — Unblock DICC (now)

**Owner:** User + Prof decision; optional Cheran.

1. Await Prof reply on whether Cheran may run campaign scripts (he was assigned mainly for manuscript writing).  
2. If **yes** and Cheran has DICC: he runs on **his** account; returns `benchmarks/results/dicc/`.  
3. If **no**: user retries on better network / campus / `tmux`, same commands.  
4. Do **not** share passwords or keys.

**Cheran / operator run card (copy-paste):**

```bash
# Tree: git clone https://github.com/titoatwork/COLIDE.git  OR  tar xzf colide-master-for-dicc.tar.gz
cd ~/colide   # or actual path

python3 -m venv .venv-cluster
source .venv-cluster/bin/activate
pip install -U pip
pip install 'numpy>=2.0.0' 'scipy>=1.13.0' 'pyyaml>=6.0' 'scikit-learn>=1.5.0'
pip install --upgrade 'torch>=2.5.0,<2.7' --index-url https://download.pytorch.org/whl/cu121

export COLIDE_V100_PARTITION=gpu-v100s
export COLIDE_A100_PARTITION=gpu-a100
export COLIDE_SBATCH_GRES=gpu:1

bash dicc_scripts/run_campaign.sh              # Day 1
# after SUCCESS dirs exist:
bash dicc_scripts/run_campaign.sh --day 2      # Day 2 (same calendar day OK via _d2)

# Deliverable: entire benchmarks/results/dicc/
```

**Notes:** Prefer `.venv-cluster` (conda activate under script `set -u` breaks on DICC). Prefer **partitions** over fixed nodelists (`gpu05` / `gpu07` may change). Use `tmux` on DICC.

**Exit P0:** At least one GPU class has Day1 SUCCESS; ideally V100 **and** A100.

---

### Phase P1 — Multi-day complete + artifacts home (pre-writing critical path)

**Owner:** DICC runner + user (scp).

1. Day 2 SUCCESS for same GPU class(es) as Day 1.  
2. `scripts/compare_dicc_sessions.py` → **accept** (or document reject + partial use).  
3. Copy to laptop:

```bash
# results tree → /home/titoisalive/colide/benchmarks/results/dicc/
```

**Exit P1:** Local `benchmarks/results/dicc/**/SUCCESS` + compare outcome recorded.  
**Clock:** ~2–5 days wall time (queue-dominated) once runner is active.

---

### Phase P2 — Extract + **codebase-wide numbers match** + Prof report  
**(HARD GATE — do not email Prof until this phase exits)**

**Owner:** User + agent (laptop).  
**Playbook:** `docs/PROF_POR_3DAY.md` §4–§5 + this section.  
**User rule (2026-07-17):** Must perform a **sight / consistency match of numbers across the entire public codebase** before sending anything that looks like a final status pack.

#### P2a — Extract from artifacts only

1. Inventory SUCCESS paths; compare accept/reject.  
2. Fill `docs/PROF_POR_3DAY.md` §4 from JSON only (Block 3 CUDA vs PT; absolute full V3 PT; no invalid full CUDA/full V3 ratio).  
3. List local frozen metrics (F1, RF, LLM, laptop ranges) from source JSON.  
4. Label June 551/592 only if used as **legacy single-shot**.

#### P2b — Codebase-wide numbers match (required)

Reconcile **every** user-facing number with a single source of truth:

| Surface | Must match |
|---------|------------|
| `README.md` | Accuracy, latency ranges, DICC (when added), md5 |
| `docs/PROF_POR_3DAY.md` §4–§5 | Same cells as JSON / email |
| `docs/paper_text_blocks.md` / abstract drafts | No stale point ratios |
| `HANDOFF.md` | No contradictory frozen numbers |
| `scripts/verify_claims.py` manifest | Claims true; regressions fail closed |
| Email / slide text to Prof | **Identical** to locked §4 table |

Checks include: F1 **0.9790**, RF **0.9864**, gap **0.74%**, ToN-IoT **0.9526**, checkpoint md5, laptop **ranges** (not lucky points), new DICC µs, Option A language (no full-pipeline CUDA vs full V3).

#### P2c — Automated gate

```bash
cd /path/to/colide
PYTHONPATH=. python scripts/verify_claims.py   # must be green
```

#### P2d — Send Prof report

Only after P2a–P2c: email the numbered status (local + DICC + caveats).  
Stretch improvements (P5) and full manuscript (P4) stay **after** this send.

**Exit P2:** (1) locked number table, (2) public surfaces consistent, (3) `verify_claims.py` green, (4) Prof emailed.  
**Clock after P1:** **~1–2 days** (extract half-day + match 1–2 days if many stale strings).  
**Total P0→P2 after unblock:** **~5–7 days typical** (best ~4; buffer ~1–1.5 weeks).

---

### Phase P3 — Pre-manuscript closeout (residual hygiene)

**Owner:** User + agent.  
*Most claim work is now inside P2b; P3 is residual only.*

1. Threats-to-validity one-pager if not already in the Prof mail.  
2. Any remaining internal docs / comments.  
3. Confirm no training / champion touch.  

**Exit P3:** Pre-manuscript phase complete; safe to start P4 manuscript.  
**Clock:** ~0.5–1 day if P2 was thorough.  
**Total P0→P3:** **~1–1.5 weeks typical**.

---

### Phase P4 — Manuscript spine (after pre-manuscript)

**Owner:** User (+ Cheran on writing if that is his role).  
**Not started until P2 complete (report sent + numbers matched).**

Contribution spine:

1. Custom CUDA kernels for CNN-BiLSTM IoT IDS inference (Block 3 focus).  
2. Measurement methodology: multi-session local ranges + UM multi-day DICC + same-GPU PyTorch.  
3. On-device LLM explainability path with measured dispatch overhead.  
4. Honest limits: RF accuracy edge; no full V3 CUDA parity yet.

Minimum figures/tables: per-block latency (laptop + DICC); Block 3 CUDA vs PT; multi-day stability; accuracy vs RF; LLM dispatch.

**Clock:** ~1.5–3 months calendar for a solid FGCS-leaning draft (writing + PI feedback), separate from P0–P3.

---

### Phase P5 — Stretch (optional; not required for v1 submit path)

| Item | When |
|------|------|
| Nsight / short bottleneck note V100 vs A100 | If time after P3 (~few days) |
| Full TRT/ORT/compile matrix on DICC | Only if reviewers demand |
| Option B CUDA = full V3 | Large; only if full-pipeline claim is mandatory |
| Retrain to close RF gap | Out of scope for current freeze |

---

## 4. Success criteria

### Pre-manuscript “done”

- [ ] Day1 + Day2 SUCCESS on DICC for available GPUs  
- [ ] Compare accept (or explicit partial + reason)  
- [ ] Results on laptop under `benchmarks/results/dicc/`  
- [ ] **Codebase-wide numbers match** (README, docs, claims, email draft)  
- [ ] `verify_claims.py` green  
- [ ] Prof numbered update sent (**only after** match gate)  
- [ ] Option A respected in all public text  

### Good enough to *write* for FGCS (systems) / careful IoTJ

- Pre-manuscript done **and** framing is systems/measurement, not SOTA accuracy.  
- Tier-1 upgrades only (DICC multi-day + hygiene). Tier-2 (figures, related work, short profiling note) during writing.  
- Acceptance is never guaranteed; evidence base will be **defensible**.

---

## 5. Contingency

| Situation | Action |
|-----------|--------|
| Prof declines Cheran for cluster | User runs with `tmux` / better link; same run card |
| Only Day 1 finishes | Provisional single-day DICC + local pack; label clearly; **still run numbers-match gate** before send |
| Queue blocked entire window | Local pack + June legacy labeled **legacy, not multi-day** + “campaign in flight”; **still match local numbers across codebase** before send |
| Compare rejects | Report both days; no “stable multi-day” claim until fixed or explained |
| One partition missing | Document; publish the GPU that completed |

---

## 6. Roles

| Who | Does |
|-----|------|
| **User (Ibteshamul)** | Science ownership, Prof communication, final pack, manuscript direction |
| **Cheran** | Manuscript support; cluster runs **only if Prof approves** and he has DICC |
| **Agent** | Step coaching, ingest/pack from local artifacts, claim edits; **no DICC login**; **no invented numbers** |
| **Prof. Por** | Scope/approval, guidance |

---

## 7. Next concrete action

1. Wait for Prof’s reply on the access plan.  
2. If approved → send Cheran the **run card** in §3 P0 + tarball/repo link.  
3. When `benchmarks/results/dicc/` is on the laptop → chat: **“Prof Por pack — DICC results landed.”**  
4. Agent/user run **P2a → P2b numbers match → P2c verify_claims → P2d send** (no skip).  
5. Only then P3 residual closeout / P4 manuscript / P5 stretch.

---

## 8. Key file pointers

| Path | Role |
|------|------|
| `docs/DESIGN_PLAN.md` | Full solid/weak/invalid + WP list |
| `docs/PROF_POR_3DAY.md` | Prof pack tables + draft text |
| `docs/FINAL_PLAN.md` | **This document** |
| `dicc_scripts/README.md` / `run_campaign.sh` | Cluster entry |
| `HANDOFF.md` | Session lifecycle + status |
| `scripts/compare_dicc_sessions.py` | Cross-day gate |
| `scripts/verify_claims.py` | Claim regression |

---

## 9. Change log

| Date | Note |
|------|------|
| 2026-07-17 | Final plan from guided DICC session: partitions locked, Cheran path, pre-manuscript vs manuscript clocks, Option A freeze. |
| 2026-07-17 | **Hard gate:** codebase-wide numbers match + `verify_claims.py` green **before** any final Prof numbers email; P2 restructured (P2a–P2d); clocks +1–2 days for match pass. |
