# Git branching policy (COLIDE) — locked for all agents

**Status:** LOCKED 2026-07-22  
**Canonical final branch:** **`master`** (always the integration / paper / handoff line)  
**Remote:** `origin` · GitHub `titoatwork/COLIDE`

---

## 1. Intent

Long-lived single-branch work on `master` has been **acceptable** for COLIDE (research mono-line, sequential WPs, champion freeze). Future sessions may open a **new branch only when the work is a true alternative option**, not for every small edit.

**Goals:**
1. **`master` is always final** — ship, handoff, claims, manuscript tip, and what the next chat resumes from.  
2. **Branches only for real options** — when work could diverge, be abandoned, or need isolation.  
3. **Branch count stays low** — strict; do not proliferate short-lived or vanity branches.

---

## 2. When you **must** create a new branch

Create a branch **before** starting work if **any** of these is true:

| Trigger | Example |
|---------|---------|
| **True alternative option** | Option B full-CUDA parity path vs Option A on master; experimental arch that may be discarded |
| **Risky / reversible isolation** | Large refactor, champion replace trial, invasive DICC harness rewrite |
| **Parallel lines** | One agent on manuscript typesetting while another on experimental code (rare) |
| **User explicitly asks** for a branch / PR / worktree |

**Rule of thumb:** *If this work might be thrown away or should not land on the paper tip until proven, it starts on a branch.*

---

## 3. When you **must not** create a new branch

Stay on **`master`** (or merge immediately and delete) for:

| Stay on master | Example |
|----------------|---------|
| Continuity / handoff / tracker / claims rebuild | HANDOFF, SESSION_CONTINUITY, verify_claims |
| Small prose / doc hygiene | Fix stale Cheran text, typo, progress log |
| Incremental WP that is already the locked path | CAD-CBA-v1 package work already on master |
| “Just because” or “cleaner history” with no option fork | No |
| One commit of polish after DICC insert | Prefer master unless user wants a PR |

**Anti-pattern:** opening `feat/foo`, `fix/bar`, `wip/agent-…` for every chat. That **must not** happen.

---

## 4. Strict branch budget

| Limit | Rule |
|-------|------|
| **Default open feature branches** | **0–2** beyond `master` |
| **Hard soft-cap** | Prefer **≤3** remote non-`master` branches total; if more exist, clean up before opening another |
| **Naming** | Short, purpose-first: `exp/option-b-cuda`, `exp/dicc-harness-v2`, `fix/venue-ieee` — never `tmp`, `test`, agent id spam |
| **Lifetime** | Branches are **short-lived**. Merge or delete within the same arc when possible |
| **After merge** | Delete local + remote branch (`git push origin --delete <branch>`) |
| **Stale remote** | If a remote branch is abandoned (e.g. old `final-polish`), document or delete after user OK — do not pile new ones beside it |

**If the tree already has extra remotes:** do **not** add another until one is merged/deleted or user allows an exception.

---

## 5. `master` is final

| Rule | Detail |
|------|--------|
| Handoff tip | Next session always resumes from **`origin/master`** unless user names another branch |
| Claims / manuscript / freeze card | Live on **`master`** only |
| Champion path docs | md5 and paths refer to tree on **`master`** |
| PR / branch work | Land on **`master`** via merge (prefer FF or clean merge); no long-lived “final” second branch |
| Force-push | **Never** force-push `master` unless user explicitly orders it |
| Protected assumption | Treat `master` as the paper of record |

---

## 6. Workflow when a branch **is** justified

```text
1. git checkout master && git pull
2. git checkout -b exp/<short-purpose>
3. Work; commit on the branch
4. Verify (claims / tests as relevant); do not invent DICC numbers
5. User review if risky (champion, DICC, destructive)
6. Merge to master (PR or local merge)
7. git checkout master && git pull
8. Delete branch local + remote
9. Update HANDOFF if handoff tip changed
```

**Do not** leave merged branches hanging.  
**Do not** keep two “final” branches.

---

## 7. Agent checklist (every session that touches git)

- [ ] Am I on **`master`** unless this is a true option fork?  
- [ ] If I want a new branch: is there a **clear option** that could be discarded?  
- [ ] Count remote non-master branches — if already high, **cleanup first** or stay on master  
- [ ] After merge: **delete** the feature branch  
- [ ] Push **`master`** for handoff continuity  

---

## 8. Relation to other rules

| Policy | Doc |
|--------|-----|
| Session lifecycle / handoff | `HANDOFF.md` |
| Continuity pack | `docs/execution_plan/SESSION_CONTINUITY.md` |
| Safety / claims | `docs/execution_plan/16_SAFETY_AND_RULES.md` |
| DICC ops | `docs/DICC_OPS_METHOD.md` |
| Champion no-clobber | Safety §1 |

Branching **does not** relax champion freeze, Option A, or no-invent-numbers.

---

## 9. Change log

| Date | Note |
|------|------|
| 2026-07-22 | Policy locked: master = final; branch only for true options; keep branch count low. |

*End.*
