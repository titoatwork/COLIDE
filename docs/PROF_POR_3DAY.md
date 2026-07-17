# COLIDE — Prof. Por update (≤3 days) — command plan

**Deadline:** give Prof. Dr. Por Lip Yee a **quick status update with numbers within ≤3 days**  
**Priority freeze:** **DICC multi-day numbers first**; manuscript spine / deep claim rewrites / stretch WPs **after** the update.  
**Strategy still locked:** Option A (valid **per-block** CUDA vs PyTorch only; **no** full-pipeline CUDA vs full V3 speedup).  
**Agent DICC access:** none (cannot SSH). **Guided mode is required:** next session is
interactive coaching — you run every command on DICC/laptop; agent gives exact steps,
interprets paste-backs, and only then packages the Prof brief from real artifacts.

**Clock (example if start = Day 0 = today):**

| Calendar | Focus | Owner |
|----------|--------|--------|
| **Day 0 (today)** | Sync tree to UM DICC + **Day 1 campaign** | **You** |
| **Day 1** | **Day 2 campaign** + compare + scp results home | **You** |
| **Day 2** | Fill DICC table in this brief + send to Prof | **You + agent** (agent if results are on laptop) |
| **After Day 2** | WP5 claim hygiene, WP6 manuscript, stretch | Later |

`run_campaign.sh --day 2` uses a `_d2` label so Day 2 **can** be the same calendar day if the queue is fast — prefer a real second day if possible, but **deadline wins**.

---

## 1. What “all the numbers” means for this update

Prof gets a **status pack**, not a finished paper. Include:

### A. Ready now (local, verified) — DO NOT retrain

| Metric | Number | Source |
|--------|--------|--------|
| Production model | Two-stage CNN-BiLSTM | `best_model_botiot_twostage.pth` |
| Checkpoint md5 | `80a90f7cc210276300eaa90173a5a385` | laptop |
| BoT-IoT test macro-F1 | **0.9790** | `twostage_botiot.json` |
| CPU RF (same splits) | **0.9864** | `rf_baseline_processed.json` |
| Gap to RF | **0.74%** | 0.9864 − 0.9790 |
| ToN-IoT clean macro-F1 | **0.9526** | README / toniot clean results |
| LLM dispatch p99 | **16.60 µs** | `llm_explainability.json` |
| Streaming throughput (RTX 3050) | **25,899 flows/s** (batch=128) | `streaming_throughput.json` |
| Custom CUDA FP16 “pipeline” range (laptop) | **594–675 µs** | multi-session; **sum-of-blocks methodology** |
| Framework speedup **ranges** (laptop, scoped) | Eager **3.04–3.78×**, torch.compile **2.25–2.99×**, TensorRT **3.60–4.99×**, ORT-GPU **5.72–7.83×** | README; WSL2 ranges |
| Block 3 FP16 progression (laptop) | **7.55–9.50×** vs naive; vs cuDNN-style block **1.30–1.47×** | ranges |
| Numerical fidelity | Export path bit-identical n=10; 6 CUDA self-checks PASS | `numerical_fidelity.json` |
| Tarball for cluster | `~/colide-master-for-dicc.tar.gz` (~356 MB) | already packed |

### B. Must come from UM DICC in the next 1–2 days

| Metric | Why Prof needs it | How |
|--------|-------------------|-----|
| V100 + A100 CUDA Block 1–4 means (n=100) | Cross-hardware systems story | Day1 + Day2 `SUCCESS` |
| V100 + A100 PyTorch Block 3 + full V3 absolute latency | Same-GPU baseline | `pytorch_gpu_stats.json` in each run dir |
| **Block 3** CUDA vs PyTorch ratio on each GPU | Only valid speedup language | Compute after compare accept |
| Day1 vs Day2 stability | Multi-day credibility | `compare_dicc_sessions.py` |
| Environment fingerprints | Reproducibility | `manifest.json`, `environment.txt`, `kernel_SHA256SUMS` |

### C. Say carefully / do **not** put as headline for Prof

| Claim | Rule |
|-------|------|
| Full-pipeline Custom CUDA vs full V3 PyTorch | **Do not claim** (architecture parity gap) |
| Rostam Day 1 means | Tooling trial only — **not** UM official |
| “Beats all frameworks” | ORT CPU **not** robust; use ranges + hedge |
| Laptop ratios as universal truth | Label **WSL2 RTX 3050**; DICC is bare-metal check |

---

## 2. Your exact operator sequence (do this first)

### Step 0 — Laptop (2 min)

```bash
# Confirm pack exists
ls -lh /home/titoisalive/colide-master-for-dicc.tar.gz
# If missing/stale, rebuild:
# cd /home/titoisalive && tar czf colide-master-for-dicc.tar.gz \
#   --exclude='colide/.venv' --exclude='colide/.venv-cluster' \
#   --exclude='colide/**/__pycache__' -C /home/titoisalive colide
```

### Step 1 — WP1 Sync (on DICC login)

```bash
# Park old diverged clone if still named colide
cd /home/user/ibteshamulhaque
# mv colide colide-old-diverged-20260714   # if needed

# From laptop:
scp /home/titoisalive/colide-master-for-dicc.tar.gz \
  ibteshamulhaque@login01.dicc.um.edu.my:/home/user/ibteshamulhaque/

# On DICC:
cd /home/user/ibteshamulhaque
tar xzf colide-master-for-dicc.tar.gz
cd colide
ls dicc_scripts/run_campaign.sh
md5sum model/best_model_botiot_twostage.pth
# expect: 80a90f7cc210276300eaa90173a5a385
```

Do **not** rely on `git fetch` on DICC (known freeze India→GitHub).

### Step 2 — WP2 Day 1

```bash
cd /home/user/ibteshamulhaque/colide
bash dicc_scripts/run_campaign.sh
# if partitions/GRES wrong, use env overrides from dicc_scripts/README.md
```

Wait for jobs. Check:

```text
benchmarks/results/dicc/<campaign>/<gpu>/<date>_job<id>/SUCCESS
```

for **both** V100 and A100 (or note which partitions actually exist).

### Step 3 — WP3 Day 2 + compare

Same tree, **same binaries** (do not recompile between days unless forced — if forced, document it).

```bash
bash dicc_scripts/run_campaign.sh --day 2
# then compare (adjust paths/GPU tags to what SUCCESS dirs show):
PYTHONPATH=. python scripts/compare_dicc_sessions.py --help
# run the compare for each GPU class that finished Day1+Day2
```

### Step 4 — Bring artifacts home

```bash
# On laptop (example):
scp -r ibteshamulhaque@login01.dicc.um.edu.my:/home/user/ibteshamulhaque/colide/benchmarks/results/dicc \
  /home/titoisalive/colide/benchmarks/results/
```

Then open an agent chat: **“Prof Por pack — DICC results landed.”**  
Agent fills §4 table below and polishes the email/slide text.

**HARD GATE before send (user-confirmed 2026-07-17):** do **not** email §5 (or any
“final” numbers pack) until a **codebase-wide numbers match** is done — README, docs,
`paper_text_blocks`, HANDOFF, §4 table, and draft email must agree with the same JSON
sources — and `PYTHONPATH=. python scripts/verify_claims.py` is **green**. See
`docs/FINAL_PLAN.md` phase **P2a–P2d**.

**Operator guide detail:** `dicc_scripts/README.md`.

---

## 3. If DICC slips (contingency for ≤3-day deadline)

| Situation | What to send Prof |
|-----------|-------------------|
| Day1+Day2 both accepted | Full pack: local + UM multi-day + Block 3 ratios |
| Only Day 1 done | Local pack + **provisional** Day1 UM numbers, clearly labeled **single-day / not final** |
| Queue blocked entire window | Local pack + **legacy June 2026 DICC single-shot** (551 / 592 µs pipeline totals) labeled **legacy, not multi-day** + honest “campaign in flight” |
| Rostam-only numbers | **Do not** substitute as UM results; at most “tooling trial on other site” |

**Never invent** DICC numbers. Prefer an honest partial table over a pretty false one.

---

## 4. DICC fill-in table (blank — you/agent complete after runs)

### Day 1 / Day 2 paths

| GPU | Day1 SUCCESS dir | Day2 SUCCESS dir | Compare result |
|-----|------------------|------------------|----------------|
| V100 | | | accept / reject |
| A100 | | | accept / reject |

### Core latencies (µs) — paste means from JSON

| Metric | V100 Day1 | V100 Day2 | A100 Day1 | A100 Day2 |
|--------|-----------|-----------|-----------|-----------|
| CUDA Block 3 FP16 | | | | |
| PyTorch Block 3 (matched) | | | | |
| PyTorch full V3 (absolute only) | | | | |
| CUDA B1 / B2 / B4 (optional) | | | | |

### Valid ratios only

| Ratio | V100 | A100 | Note |
|-------|------|------|------|
| CUDA B3 FP16 / PyTorch B3 | | | Prefer multi-day stable mean |
| Full CUDA pipeline / full V3 PT | **n/a** | **n/a** | Invalid under Option A |

---

## 5. Draft text for Prof (local half ready; DICC half TBD)

Copy/adapt:

> **COLIDE — short status for Prof. Por (deadline window)**  
>  
> **Goal:** FGCS-class systems paper: custom CUDA inference for CNN-BiLSTM IoT IDS + on-device LLM explainability; contribution is systems/measurement, not SOTA accuracy.  
>  
> **Accuracy (final local, frozen):** BoT-IoT two-stage model **macro-F1 = 0.9790** (checkpoint md5 `80a90f7c…`). Apples-to-apples CPU RF on same splits **0.9864** → gap **0.74%**. ToN-IoT clean **0.9526**. We do **not** claim beating RF.  
>  
> **Latency (laptop RTX 3050 / WSL2):** Custom CUDA FP16 block-sum pipeline **594–675 µs** (multi-session range). Framework comparisons reported as **ranges** because of real session-to-session drift. Block 3 FP16 optimization ladder **7.55–9.50×** vs naive.  
>  
> **Important scientific caveat (we are handling correctly):** full-model Custom CUDA vs full PyTorch V3 speedup is **not** claimed yet — V3 has attention/LayerNorm/GAP that CUDA does not implement. Valid head-to-head is **per-block**, especially **Block 3 (BiLSTM)**.  
>  
> **LLM:** async on-device TinyLlama; dispatch overhead **16.60 µs p99** (5k trials).  
>  
> **Throughput:** **25,899 flows/s** streaming (batch=128, RTX 3050).  
>  
> **UM DICC (in progress this week):** multi-day V100/A100 campaign with same-hardware PyTorch baselines — tables below after Day1+Day2+compare.  
>  
> **Next after this update:** claim hygiene + manuscript spine; optional CUDA parity only if needed.

*(Replace the DICC sentence with the filled §4 table when ready.)*

---

## 6. What we deliberately postpone until AFTER Prof update

| Item | Why wait |
|------|----------|
| WP5 deep README/abstract claim rewrite | Needs DICC numbers + quiet time |
| WP6 full manuscript spine | Not needed for “quick update” |
| WP7 stretch (true fused pipeline, Option B parity, more KD, TRT on DICC) | Non-blocking for Prof status |
| Any training / touching champion `.pth` | Forbidden; numbers already frozen |
| Treating Rostam as official | Wrong site for UM acknowledgment |

---

## 7. Agent commitments + session close pattern

**Every agent session** must end per `HANDOFF.md` **Session lifecycle**:
verify → update HANDOFF → commit → push → paste-ready next-session prompt.

**DICC is not “user alone.”** Default next chat is **Guided UM DICC** (HANDOFF §D):
agent coaches WP1→WP3 live; user pastes outputs. After `benchmarks/results/dicc/` is
on the laptop, same chat may start the Prof pack if time allows, or a follow-up
“Prof Por numbers pack” chat.

Agent will **not** invent cluster numbers and will **not** start manuscript-deep work
that delays DICC + Prof update.

---

## 8. Single checklist (print this)

- [ ] scp tarball to DICC (or Cheran runs on his account after Prof OK)  
- [ ] unpack; `run_campaign.sh` present; md5 `80a90f7c…`  
- [ ] Day 1 jobs → both GPUs `SUCCESS` (or document missing partition)  
- [ ] Day 2 jobs → `SUCCESS`  
- [ ] compare accept  
- [ ] scp `benchmarks/results/dicc/` home  
- [ ] fill §4 from JSON only  
- [ ] **codebase-wide numbers match** (README / docs / claims / email draft)  
- [ ] `verify_claims.py` green  
- [ ] **only then** send §5 text to Prof  
- [ ] after send: residual hygiene / manuscript (not blockers for the send)

**You have the operational wheel on DICC; I have the scientific/claim wheel once results land.**
