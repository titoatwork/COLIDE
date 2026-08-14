# Knowledge-distillation objectives in COLIDE

This note separates the **historical BoT-IoT KD objective** (champion recipe) from
**canonical temperature-scaled KD** (Hinton-style). Do not treat `T=10` in the
historical path as equivalent to standard KD temperature ten.

---

## 1. `legacy_teacher_smoothed_kl` (historical / champion path)

**Where:** `scripts/train_protocol_kd.py`, `scripts/train_distill.py`,
ensemble/KD sweeps that produced Stage-1 and the CAD-CBA-v1 package.

**What it does:**

1. Fit teacher(s) on train rows; take `predict_proba` on those **same** train rows
   (not out-of-fold). Soft labels are therefore optimistic relative to true
   inductive KD.
2. Soften teacher probabilities offline:
   \[
   p'_i \propto (p_i)^{1/T}
   \]
   (temperature applied to probabilities, not logits).
3. Student hard loss: focal (historical `LegacyFocalLoss` / `FocalLoss`).
4. Distillation term:
   \[
   \mathrm{KD} = \mathrm{KL}\big(\mathrm{log\_softmax}(z_s) \,\|\, p'\big)
   \]
   with **no** temperature on student logits and **no** conventional \(T^2\) factor.
5. Mix: \(\alpha \cdot \mathrm{KD} + (1-\alpha)\cdot\mathrm{hard}\)
   (CLI flag still named `--alpha`; semantic = KD mix weight).

**Name in metadata / prose:** `legacy_teacher_smoothed_kl`.

**Implications:**

- Historical results are evidence for **this** objective, not for canonical KD.
- Do not re-interpret published Stage-1 / package numbers as standard \(T,T^2\) KD.
- Do not launch a new BoT-IoT or ToN-IoT KD sweep solely to “fix” the formula under
  the minimum-scope plan; the champion training formula must stay frozen.

---

## 2. Canonical temperature KD (\(T / T^2\)) — future utility only

Standard soft-target KD (Hinton et al.):

1. Teacher and student logits both divided by the same temperature \(T\).
2. Soft distributions: \(\mathrm{softmax}(z_t / T)\), \(\mathrm{softmax}(z_s / T)\).
3. KL (or CE) between student soft and teacher soft, typically scaled by \(T^2\)
   so gradient magnitude stays comparable when \(T\) changes.
4. Mix with hard-label loss via an unambiguous weight
   (`kd_weight` / `hard_label_weight`), not a reused focal “alpha”.

A corrected utility may be added for **future** experiments; it must not rewrite
historical JSON or retrain the sealed champion by default.

| Item | `legacy_teacher_smoothed_kl` | Canonical \(T/T^2\) |
|------|------------------------------|---------------------|
| Teacher soft | Prob-space \(p^{1/T}\) offline | \(\mathrm{softmax}(z_t/T)\) |
| Student soft | \(\mathrm{log\_softmax}(z_s)\) (T=1) | \(\mathrm{log\_softmax}(z_s/T)\) |
| \(T^2\) factor | No | Yes (usual convention) |
| Champion recipe | **Yes (frozen)** | No |

---

## 3. Result metadata recommendation

New KD result envelopes should record:

```json
{
  "kd_objective": "legacy_teacher_smoothed_kl",
  "kd_weight": 0.6,
  "temperature": 10.0,
  "temperature_applied_to": "teacher_probs_only",
  "t_squared_scaling": false,
  "teacher_probs_on": "teacher_training_rows"
}
```

---

## 4. Related code

- Protocol KD trainer: `scripts/train_protocol_kd.py`
- Losses: `scripts/protocol/losses.py` (`LegacyFocalLoss` vs `StandardFocalLoss`)
- Freeze card: `docs/execution_plan/FINAL_CONFIG_FREEZE_CARD.md`
