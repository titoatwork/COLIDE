# COLIDE Remediation and Limited-Scope Improvement Checklist

## Remediation status (2026-08-14)

Offline remediation for claim hygiene, ToN-IoT leakage-safe correction, CUDA Block-3 **source** fixes, protocol utilities, tests, and licensing is largely **DONE**. Hardware-bound CUDA validation remains open.

| Bucket | Status |
|--------|--------|
| Sealed BoT champion principal **0.9780±0.0033** (md5 `80a90f7cc210276300eaa90173a5a385`) | **DONE** |
| Issue register, stale-claim guard, README / claim-map / limitations hygiene | **DONE** |
| ToN “clean” quarantine + `train_toniot_clean` fail-fast | **DONE** |
| Corrected ToN pipeline (`toniot_leakage_safe`) RF **0.9626** / CNN **0.8075** | **DONE** |
| CUDA B3 double-buffer + reverse align in source (`fused_block3*.cu`) + `CUDA_WEIGHT_MAPPING.md` | **DONE** (code) |
| Focal / KD disclosure (no formula retrain); HPO CLI>hpo>defaults; champion path config | **DONE** |
| `requirements*.txt`, Dockerfile arch comments, Academic LICENSE, tests, result_schema, shebangs | **DONE** |
| Streaming → bulk, energy exploratory, LLM dispatch-only, incomplete pipeline **FORBIDDEN** (docs) | **DONE** |
| DICC rebench of fixed B3 kernels | **AWAITING_HARDWARE** |
| `compute-sanitizer` racecheck/synccheck/initcheck on optimized kernels | **AWAITING_HARDWARE** |
| Full manuscript rewrite / figure regeneration from corrected artifacts | **OPEN** |
| True streaming pacer implementation | **DROP** (reframe only) |
| Energy remeasure (controlled) | **DROP** (exploratory unless later needed) |
| Official ToN temporal/host split | **NOT AVAILABLE** (limitation recorded) |

See also: `docs/REMEDIATION_STATUS.md`, `docs/ISSUE_REGISTER.md`.

---

## Scope

This checklist applies to the audited `master` snapshot at commit `2608c71` from August 12, 2026. It intentionally avoids architectural redesigns, new datasets, large hyperparameter sweeps, new deployment platforms, and other major research extensions.

Priority labels:

- **P0 — Blocking:** must be completed before submitting, publishing, or presenting the affected result.
- **P1 — Required cleanup:** needed for a defensible repository and manuscript.
- **P2 — Small improvement:** useful only after all P0 and relevant P1 work is complete.
- **DROP:** remove or narrow a claim rather than spending time rebuilding the experiment.

---

# 1. Recommended minimum-change strategy

These decisions minimize additional work while resolving the important issues.

- [x] **Keep the existing sealed BoT-IoT champion.** Do not retrain or redesign the principal model unless a correction directly changes the model or its sealed evaluation.
- [x] **Make `0.9780 ± 0.0033` the principal BoT-IoT result.** Move the historical `0.9790` result into a clearly labelled development-results section. The current claim map already treats the sealed result as authoritative and forbids pure-F1 SOTA wording.
- [x] **Withdraw the existing ToN-IoT “clean” results and run one corrected, simple experiment.**
- [x] For the corrected ToN-IoT run, use:
  - [x] the corrected feature set;
  - [x] one frozen train/validation/test split;
  - [x] a random-forest baseline;
  - [x] a hard-label CNN training run;
  - [x] no knowledge distillation;
  - [x] no ordinary SMOTE on encoded categorical features.
- [x] **Fix CUDA Block 3 rather than developing new kernels.**
- [ ] Rerun only the benchmarks affected by the Block-3 corrections.
- [x] Do not implement custom attention, LayerNorm, residual, pooling, or a new full-V3 CUDA pipeline.
- [x] **Rename the existing streaming result as bulk batched throughput.** Do not build a full streaming simulator unless retaining a true streaming claim is essential.
- [x] **Treat the current energy results as exploratory.** Either move them to an appendix or perform one small corrected GPU-board-energy rerun.
- [x] **Retain the LLM result only as asynchronous alert-construction and queue-dispatch overhead.**
- [x] Do not add a new LLM, RAG pipeline, fine-tuning study, human evaluation, or production service.

---

# 2. Create one remediation authority

## 2.1 Central issue register

- [x] **P0:** Create `docs/ISSUE_REGISTER.md`.
- [ ] Give every issue a stable identifier, such as:
  - [x] `DATA-TON-001` — target-derived `label` included in ToN-IoT features;
  - [x] `CUDA-B3-001` — optimized Block-3 hidden-state race;
  - [x] `CUDA-B3-002` — reverse-sequence output alignment;
  - [x] `CUDA-B3-003` — CUDA/PyTorch output-contract mismatch;
  - [x] `CLAIM-PIPE-001` — incomplete CUDA pipeline compared with full V3;
  - [x] `LOSS-FOCAL-001` — class-weighted focal-loss formulation;
  - [x] `KD-001` — noncanonical temperature handling;
  - [x] `BENCH-STREAM-001` — offered-rate variable does not pace arrivals;
  - [x] `ENERGY-001` — nonintegrated and incomparable power measurements;
  - [x] `LLM-001` — dispatch latency presented too broadly.
- [ ] Record for every issue:
  - [x] severity;
  - [x] affected files;
  - [x] affected results;
  - [x] affected manuscript claims;
  - [x] remediation decision;
  - [x] completion evidence;
  - [x] status;
  - [x] date closed;
  - [x] commit that closed it.
- [x] Link the issue register from the README and the claim map.
- [x] Do not silently delete invalid historical results; preserve them with explicit invalidity metadata.

## 2.2 Repository-wide stale-claim search

- [x] **P0:** Search the entire repository for the following values and inspect every occurrence:
  - [x] `0.9526`;
  - [x] `0.9851`;
  - [x] `15.4%`;
  - [x] `0.9790`;
  - [x] `25,899`;
  - [x] `16.60`;
  - [x] `1.089`;
  - [x] old Block-3 latency and speedup values.
- [x] Search for the following phrases:
  - [x] `matching CUDA`;
  - [x] `full pipeline`;
  - [x] `end-to-end custom CUDA`;
  - [x] `state of the art`;
  - [x] `SOTA`;
  - [x] `real-time`;
  - [x] `production-ready`;
  - [x] `LLM-based explainability`;
  - [x] `verified research gap`;
  - [x] `streaming latency`;
  - [x] `CPU energy`.
- [ ] Inspect:
  - [x] README;
  - [x] claim maps;
  - [x] manuscript text;
  - [x] JSON result files;
  - [x] plotting scripts;
  - [x] generated figures;
  - [x] notebook outputs;
  - [x] comments and docstrings;
  - [x] archived reports.
- [x] Add a small repository script that fails when a forbidden stale claim appears outside an explicitly marked archive file.

---

# 3. Immediately quarantine invalid or misleading results

## 3.1 ToN-IoT result quarantine

The current “clean” ToN-IoT loader retains `label` as a numeric input while predicting `type`. It also fits categorical encoders before splitting and applies ordinary SMOTE to integer-encoded categorical fields. These results cannot remain active evidence.

- [x] **P0:** Add the following metadata to every affected result:
  - [x] `"valid": false`;
  - [x] `"invalid_reason": "Target-derived binary label included in multiclass feature matrix"`;
  - [x] `"superseded_by": null` until the corrected run exists;
  - [x] `"use_in_manuscript": false`.
- [x] **P0:** Remove the existing ToN-IoT `0.9526` CNN result from headline tables.
- [x] **P0:** Remove the existing ToN-IoT `0.9851` RF result from headline tables.
- [x] **P0:** Remove the claimed `+15.4%` improvement.
- [x] **P0:** Remove or invalidate derived figures, rankings, and paragraphs that use those numbers.
- [x] Add a short tombstone note explaining why the result was withdrawn.
- [x] Remove the invalid historical values embedded in `scripts/run_toniot_final_method.py`, so future summary files cannot reintroduce them. That script currently carries historical comparison values and an in-sample teacher path.
- [x] Preserve the invalid files under an `archive/invalidated/` or similarly explicit directory.
- [x] Never label the archived experiment “clean.”

## 3.2 CUDA benchmark quarantine

- [x] **P0:** Mark all existing optimized Block-3 numerical results as `pre_fix`.
- [x] **P0:** Mark all Block-3 speed comparisons as provisional until corrected real-weight parity passes.
- [ ] Add:
  - [x] `"kernel_correctness_status": "unverified_pre_fix"`;
  - [x] `"semantic_parity_status": "failed_or_unverified"`;
  - [x] `"benchmark_usable_for_claims": false`.
- [x] Keep old timing files only as historical artifacts.
- [x] Do not use old Block-3 latency values in the final main comparison after the source changes.
- [x] Mark old full custom-pipeline aggregates as:
  - [x] `legacy`;
  - [x] `non_parity`;
  - [x] `not_full_v3`;
  - [x] `not_usable_for_model_level_speedup`.

## 3.3 Streaming, energy, and LLM quarantine

- [x] **P0:** Rename existing “streaming throughput” artifacts to “bulk batched throughput.”
- [x] **P0:** Remove claims about controlled offered load, saturation, queueing delay, and sustainable stream rate from the existing experiment.
- [x] Label current energy numbers as:
  - [x] exploratory;
  - [x] board-power estimates where applicable;
  - [x] not directly comparable across CPU/GPU/model paths.
- [x] Label `16.60 µs` as alert construction and queue insertion overhead only.
- [x] Place dropped-alert and completed-generation counts immediately beside any LLM dispatch result.
- [x] Remove title-level wording that implies validated end-to-end LLM explainability.

---

# 4. Correct the ToN-IoT experiment

## 4.1 Define a leakage-safe schema

- [x] **P0:** Replace automatic “all numeric columns” feature selection with an explicit feature allowlist.
- [x] Create a central target-derived column blacklist containing at least:
  - [x] `label`;
  - [x] `type`;
  - [x] `attack`;
  - [x] `category`;
  - [x] normalized spelling and capitalization variants.
- [x] Normalize column names before checking:
  - [x] trim whitespace;
  - [x] lowercase;
  - [x] normalize punctuation and underscores.
- [x] Add a fatal assertion that the intersection of the feature list and target blacklist is empty.
- [x] Add a fatal assertion that `label` cannot appear in `X`.
- [x] Add a fatal assertion that the target column cannot appear in `X`.
- [x] Print the final ordered feature list at startup.
- [x] Save that ordered list in every result JSON.
- [x] Save a SHA-256 hash of the ordered feature list.
- [x] Save the target name and class mapping.
- [x] Add a unit test that deliberately inserts `label` into the candidate features and verifies that execution stops.
- [x] Add a test for capitalization variants such as `Label`, `TYPE`, and `Category`.
- [x] Check for unnamed index columns or CSV row-number columns and exclude them explicitly.
- [x] Check whether any remaining feature is a direct encoding of the target.
- [x] Record the number of rows removed for missing or invalid targets.

## 4.2 Freeze the split before preprocessing

- [x] **P0:** Create train, validation, and test partitions before fitting encoders, imputers, scalers, samplers, or models.
- [x] Prefer the official dataset split when the source files provide one.
- [x] If an official split is not available in the selected file:
  - [x] choose one fixed stratified split;
  - [x] choose the seed before seeing test performance;
  - [x] persist the exact row indices;
  - [x] do not regenerate the split during later runs.
- [x] Save:
  - [x] split seed;
  - [x] row counts;
  - [x] class counts;
  - [x] row-index hashes;
  - [x] source-file hashes.
- [x] Add a disjointness assertion for train, validation, and test row identifiers.
- [x] Check exact duplicate rows across partitions.
- [x] Report any cross-partition duplicates.
- [x] If duplicates are plentiful and safely removable:
  - [x] remove exact duplicates before splitting;
  - [x] record the removal count.
- [x] Do not introduce a complex host-aware or temporal splitting project unless the necessary grouping column already exists and can be used trivially.
- [x] State the limitation if the final split is random rather than temporal or host-independent.

## 4.3 Fit preprocessing on training data only

- [x] **P0:** Fit categorical encoders on the training partition only.
- [x] Define deterministic handling for unseen validation/test categories.
- [x] Fit imputation values on training data only.
- [x] Fit scaling parameters on training data only.
- [x] Apply the frozen transforms to validation and test.
- [x] Save:
  - [x] categorical vocabulary;
  - [x] unknown-category code;
  - [x] imputation values;
  - [x] scaler parameters;
  - [x] transformed feature order.
- [x] Record the number of rows used to fit every preprocessing component.
- [x] Add an assertion that no preprocessing component was fit on validation or test rows.
- [x] Ensure RF and CNN receive exactly the same base features and split.
- [x] Use a shared preprocessing module rather than duplicating logic between RF and CNN scripts.
- [x] Do not fit encoders on the combined dataframe.
- [x] Do not inspect test statistics when deciding preprocessing behavior.

## 4.4 Remove unsafe synthetic oversampling

- [x] **P0:** Remove ordinary SMOTE from integer-encoded categorical inputs.
- [x] For the minimum-scope corrected run, use no synthetic oversampling.
- [x] Use one simple imbalance treatment:
  - [x] class-weighted loss for the CNN; or
  - [x] a training-only weighted sampler.
- [x] Use `class_weight="balanced"` or a documented equivalent for RF.
- [x] Do not combine aggressive class weighting, oversampling, and focal loss without a new validation study.
- [x] Calculate all class weights from the training partition only.
- [x] Save the exact class weights in the result artifact.
- [x] Do not add a new SMOTENC experiment unless the no-SMOTE corrected run is unusable.

## 4.5 Keep the corrected ToN-IoT training simple

- [x] **P0:** Do not use knowledge distillation in the minimum corrected ToN-IoT run.
- [x] Train:
  - [x] one RF baseline;
  - [x] one hard-label CNN configuration.
- [x] Freeze the model architecture and main hyperparameters before the final test evaluation.
- [x] Use validation macro-F1 for epoch/model selection.
- [x] Save the validation-selected checkpoint.
- [x] Evaluate the test partition only after configuration selection is complete.
- [x] Do not tune after observing the corrected test result.
- [x] Predetermine and record the seed.
- [x] Use one seed if computational constraints are severe and explicitly report that limitation.
- [ ] **P1:** Run three fixed seeds only when the existing runtime makes this inexpensive.
- [x] Do not select the best of several test seeds.
- [x] Save all seed-level results rather than only the highest result.

## 4.6 Report corrected ToN-IoT results completely

- [x] Report:
  - [x] macro-F1;
  - [x] weighted-F1;
  - [x] accuracy;
  - [x] per-class precision;
  - [x] per-class recall;
  - [x] per-class F1;
  - [x] confusion matrix;
  - [x] class support.
- [x] Save prediction arrays or a compact prediction file.
- [x] Save predicted probabilities when available.
- [x] Report the RF and CNN on the exact same test split.
- [x] Label the result “corrected leakage-safe random split” or the precise applicable protocol.
- [x] Do not call it an official benchmark result unless it follows an official split.
- [x] Do not compare the corrected result against invalid historical numbers as though both were valid.
- [x] If corrected performance is substantially lower, report it without further test-driven tuning.
- [ ] Update every ToN-IoT figure directly from the corrected JSON artifact.

## 4.7 ToN-IoT completion gate

- [x] `label` is absent from the feature matrix.
- [x] The complete feature list is stored and hashed.
- [x] Splits are fixed and disjoint.
- [x] All preprocessing is fit on training data only.
- [x] No ordinary SMOTE is applied to encoded categorical features.
- [x] Test evaluation occurs only after validation selection.
- [x] RF and CNN use identical data.
- [x] Invalid historical results cannot appear in active tables.
- [x] The corrected result artifact has `"valid": true`.
- [x] The new artifact records the commit, command, data hashes, split hashes, seed, and checkpoint hash.

---

# 5. Correct CUDA Block 3

The optimized FP32 and FP16 Block-3 kernels use a shared hidden-state buffer that is read across hidden units and then overwritten before a barrier. The corrected naive kernel already demonstrates the safer double-buffer pattern.

## 5.1 Remove the hidden-state race

Apply the following to both:

- `inference/kernels/fused_block3.cu`
- `inference/kernels/fused_block3_fp16.cu`

Checklist:

- [x] **P0:** Allocate separate previous-state and next-state hidden buffers.
- [x] Ensure every recurrent gate calculation reads only from the previous-state buffer.
- [x] Ensure every thread writes its new hidden value only to the next-state buffer.
- [x] Synchronize all threads after completing next-state writes.
- [x] Swap buffer roles only after that synchronization.
- [x] Never read and overwrite the same hidden-state shared-memory array within one timestep.
- [x] Apply the same design to:
  - [x] first LSTM layer;
  - [x] second LSTM layer;
  - [x] forward direction;
  - [x] reverse direction;
  - [x] FP32 implementation;
  - [x] FP16 implementation.
- [x] Recalculate shared-memory requirements after doubling hidden-state storage.
- [x] Add a launch-time check that requested shared memory is supported.
- [x] Check for CUDA errors after every kernel launch.
- [x] Synchronize before interpreting validation output.
- [x] Remove comments that imply correctness before sanitizer and parity evidence exists.
- [x] Document the exact synchronization invariant in the source.

## 5.2 Correct bidirectional sequence alignment

The custom reverse direction processes position `sequence_length - 1 - t` but stores the output under recurrence index `t`. PyTorch aligns bidirectional outputs to the original sequence position. The full V3 model consumes the complete aligned sequence before attention and mean pooling.

- [x] **P0:** Store reverse-direction output at the original input position.
- [x] Conceptually use:
  - [x] `pos = sequence_length - 1 - t`;
  - [x] `reverse_output[pos] = reverse_hidden_at_pos`.
- [x] Confirm that forward and reverse channels at output position `k` both correspond to input position `k`.
- [x] Apply alignment after the first bidirectional layer.
- [x] Ensure the second LSTM layer receives aligned first-layer forward and reverse channels.
- [x] Apply alignment to the second-layer output.
- [ ] Update the CPU reference implementation accordingly.
- [x] Update comments and layout diagrams.
- [ ] Add a small deterministic sequence test whose expected reverse alignment can be inspected manually.
- [ ] Do not use the existing custom CPU reference as proof until it has been corrected independently.

## 5.3 Define one exact Block-3 contract

The PyTorch wrapper currently uses `output[:, -1, :]`, while the CUDA path extracts recurrent final states. These have the same shape but different bidirectional semantics.

- [ ] **P0:** Define the canonical Block-3 contract as the complete aligned output sequence:
  - [ ] input shape;
  - [ ] output shape `[batch, sequence_length, 128]`;
  - [ ] channel order;
  - [ ] sequence order;
  - [ ] dtype;
  - [ ] memory layout.
- [ ] Use the complete sequence because that is what V3 sends to attention.
- [ ] Update the PyTorch benchmark wrapper to return the complete sequence.
- [ ] Update the CUDA benchmark to return the complete sequence.
- [ ] Update the CPU reference to return the complete sequence.
- [ ] Remove the claim that `output[:, -1, :]` matches CUDA final-state extraction.
- [ ] If a final-state-only mode must remain:
  - [ ] label it `legacy_last_state`;
  - [ ] define whether it corresponds to `h_n` or `output[:, -1, :]`;
  - [ ] never label it the V3 Block-3 contract.
- [ ] Do not compare implementations merely because their output dimensions match.

## 5.4 Audit weight and state mapping

- [x] Verify PyTorch’s LSTM gate order against the CUDA gate order.
- [x] Verify matrix orientation and transposition.
- [x] Verify input-to-hidden weight mapping.
- [x] Verify hidden-to-hidden weight mapping.
- [x] Verify that `bias_ih` and `bias_hh` are combined correctly.
- [x] Verify first-layer forward weights.
- [x] Verify first-layer reverse weights.
- [x] Verify second-layer forward weights.
- [x] Verify second-layer reverse weights.
- [x] Ensure forward and reverse directions do not accidentally share weights.
- [x] Verify zero initialization of hidden and cell states.
- [x] Verify the sequence length and layer dimensions.
- [x] Verify activation functions:
  - [x] sigmoid;
  - [x] tanh;
  - [x] cell update;
  - [x] hidden update.
- [x] Verify FP16 conversion and storage boundaries.
- [x] Keep any FP32 accumulation behavior explicit.
- [x] Record the mapping in a short `CUDA_WEIGHT_MAPPING.md`.

## 5.5 Run CUDA sanitizers

- [ ] **P0:** Run `compute-sanitizer --tool racecheck`.
- [ ] **P0:** Run `compute-sanitizer --tool synccheck`.
- [ ] **P0:** Run `compute-sanitizer --tool initcheck`.
- [ ] **P0:** Run the normal memory checker.
- [ ] Test FP32 and FP16 separately.
- [ ] Test every supported direction/layer configuration.
- [ ] Test batch one.
- [ ] Test at least one batch larger than one when the kernel supports it.
- [ ] Test the production sequence and hidden sizes.
- [ ] Save the complete sanitizer logs.
- [ ] Record:
  - [ ] source commit;
  - [ ] compiler version;
  - [ ] compiler flags;
  - [ ] GPU;
  - [ ] executable SHA-256.
- [ ] Accept no race, synchronization, uninitialized-memory, or invalid-access warning.
- [ ] Do not classify intermittent sanitizer failures as harmless.
- [ ] Do not retry until a run happens to pass.

## 5.6 Test deterministic execution

- [ ] Run the same fixed input and weights repeatedly.
- [ ] Compare outputs bitwise for FP32 when feasible.
- [ ] At minimum, verify numerically identical outputs across repeated launches.
- [ ] Repeat for FP16.
- [ ] Run with other GPU work absent.
- [ ] Run enough repetitions to expose scheduling-dependent behavior.
- [ ] Treat any output variation on identical inputs as a correctness failure.
- [ ] Remove the current benchmark behavior that tolerates intermittent validation failure through retries. The existing statistical runner contains retry logic that could conceal nondeterminism.

## 5.7 Block-3 completion gate

- [x] Both optimized kernels use double buffering.
- [x] Reverse outputs are aligned to original sequence positions.
- [ ] CUDA, CPU reference, and PyTorch implement one documented contract.
- [ ] All sanitizers pass without retry.
- [ ] Identical runs are deterministic.
- [ ] Real production-weight parity passes.
- [ ] Old pre-fix result files are not used in active claims.
- [ ] A clean corrected benchmark has replaced old Block-3 headline values.

---

# 6. Build direct real-weight numerical parity

The current validation chain separates PyTorch/export checks from synthetic CUDA/self-reference checks. It does not directly prove that the production champion produces the same Block-3 output through PyTorch and custom CUDA.

## 6.1 Correct the purpose and naming of current validators

- [ ] Rename the current PyTorch/export comparison as `export_reference_integrity`.
- [ ] Remove conclusions that infer CUDA correctness without executing CUDA.
- [ ] State explicitly what each validation script tests.
- [ ] State explicitly what it does not test.
- [ ] Deprecate references to “V3 using V2 last-timestep for CUDA.”
- [ ] Do not allow synthetic self-consistency to substitute for production-weight parity.

## 6.2 Export complete model state

The model helper that iterates over `named_parameters()` does not include BatchNorm running means and variances, although those buffers affect inference. The full model architecture also uses complete sequence output through attention and normalization.

- [ ] **P0:** Export from `state_dict()`, not only `named_parameters()`.
- [ ] Include:
  - [ ] learned parameters;
  - [ ] BatchNorm running means;
  - [ ] BatchNorm running variances;
  - [ ] BatchNorm counters where relevant;
  - [ ] LayerNorm parameters;
  - [ ] attention parameters;
  - [ ] all forward and reverse LSTM weights;
  - [ ] all biases.
- [ ] Create a manifest mapping every exported tensor to its source state-dict key.
- [ ] Record:
  - [ ] shape;
  - [ ] dtype;
  - [ ] byte order;
  - [ ] offset or filename;
  - [ ] SHA-256.
- [ ] Fail if any expected state-dict key is missing.
- [ ] Fail if an unexpected shape is encountered.
- [ ] Verify the champion checkpoint hash before exporting.

## 6.3 Directly compare every custom block

- [ ] Use the frozen production champion.
- [ ] Use canonical preprocessed inputs.
- [ ] Include:
  - [ ] several real test samples;
  - [ ] fixed synthetic samples;
  - [ ] zero input;
  - [ ] small-magnitude random input;
  - [ ] boundary-valued scaled input.
- [ ] Compare PyTorch and CUDA Block 1.
- [ ] Compare PyTorch and CUDA Block 2.
- [ ] Compare the complete PyTorch and CUDA Block-3 sequence.
- [ ] Compare PyTorch and CUDA Block 4 under an exactly matched Block-4 input.
- [ ] Record:
  - [ ] maximum absolute error;
  - [ ] mean absolute error;
  - [ ] maximum relative error where stable;
  - [ ] cosine similarity;
  - [ ] shape equality;
  - [ ] NaN/Inf counts.
- [ ] Predeclare FP32 and FP16 acceptance tolerances before looking at final results.
- [ ] Do not relax tolerances after seeing a failure without a documented technical reason.

## 6.4 Add a hybrid downstream check

Because custom Block 3 does not implement V3’s attention and normalization suffix, validate it by feeding its corrected sequence output into the existing PyTorch suffix.

- [ ] Run the normal PyTorch model and save:
  - [ ] Block-3 output;
  - [ ] post-attention output;
  - [ ] final logits;
  - [ ] predicted class.
- [ ] Substitute the corrected CUDA Block-3 sequence into the PyTorch:
  - [ ] attention;
  - [ ] residual;
  - [ ] LayerNorm;
  - [ ] mean pooling;
  - [ ] classifier head.
- [ ] Compare final logits against normal PyTorch inference.
- [ ] Report class-agreement rate.
- [ ] Inspect samples with the smallest classification margin.
- [ ] Require no unexplained class disagreement for FP32.
- [ ] Report any FP16 disagreement rather than hiding it.

## 6.5 Machine-readable parity gate

- [ ] Write one parity JSON containing:
  - [ ] checkpoint SHA-256;
  - [ ] source commit;
  - [ ] executable SHA-256;
  - [ ] GPU details;
  - [ ] CUDA version;
  - [ ] compiler flags;
  - [ ] input hashes;
  - [ ] per-block error statistics;
  - [ ] downstream-logit errors;
  - [ ] pass/fail;
  - [ ] tolerance version.
- [ ] Make benchmark scripts refuse to run or refuse to emit claim-eligible output when parity has not passed.
- [ ] Add `"parity_artifact"` to every corrected benchmark JSON.
- [ ] Add `"parity_passed": true/false`.
- [ ] Make the result collector exclude parity-failed results automatically.

---

# 7. Rerun only the necessary CUDA benchmarks

## 7.1 Clean build provenance

The current closure record notes dirty-tree allowances in parts of the DICC campaign. Corrected results should come from clean source and fresh binaries.

- [ ] **P0:** Start from a clean Git tree.
- [ ] Record the exact commit.
- [ ] Compile every corrected binary fresh.
- [ ] Do not reuse an unverified pre-existing executable.
- [ ] Record:
  - [ ] source-file SHA-256;
  - [ ] executable SHA-256;
  - [ ] `nvcc` version;
  - [ ] complete compiler command;
  - [ ] CUDA architecture;
  - [ ] driver version;
  - [ ] GPU name and UUID.
- [ ] Include `source_dirty: false` in claim-eligible results.
- [ ] Exclude dirty-tree results from final headline tables.
- [ ] Preserve dirty-tree historical measurements only as supplementary evidence.

## 7.2 Minimal rerun scope

- [ ] Rerun corrected Block 3 on each server GPU currently used in the paper.
- [ ] One clean corrected session per GPU is acceptable under the limited-scope plan.
- [ ] Do not claim corrected-kernel multi-session stability if only one corrected session exists.
- [ ] Retain old multi-session results only as historical timing context, not corrected correctness evidence.
- [ ] Rerun B1, B2, and B4 only if:
  - [ ] their parity harness changes;
  - [ ] their source changes;
  - [ ] their old binary provenance is unsuitable.
- [ ] Otherwise, retain their latency measurements only after direct real-weight parity passes.
- [ ] Do not rerun a full HPO or training campaign.

## 7.3 Use defensible latency statistics

The current claim map acknowledges unequal statistical units between CUDA and PyTorch measurements. Exact p-values and effect sizes should not be emphasized under that design.

- [ ] Use the same benchmark boundary on both implementations.
- [ ] State whether data is already resident on the GPU.
- [ ] State whether allocations are excluded.
- [ ] State whether host/device transfers are excluded.
- [ ] State whether synchronization is included.
- [ ] Use the same outer-trial structure where practical.
- [ ] Report:
  - [ ] median;
  - [ ] interquartile range;
  - [ ] p95;
  - [ ] number of outer trials;
  - [ ] number of inner iterations.
- [ ] Report the mean only as a supplementary statistic.
- [ ] Drop or de-emphasize Welch-test p-values when observations come from different statistical units.
- [ ] Drop or de-emphasize huge Cohen’s `d` values from mismatched harnesses.
- [ ] Do not call theoretical occupancy actual GPU utilization.
- [ ] State that latency results are hardware-, shape-, batch-, and software-specific.
- [ ] Accept and report the result if corrected Block 3 remains slower than PyTorch.
- [ ] Stop optimizing Block 3 after the correctness repair and one fair rerun.

---

# 8. Correct the custom-pipeline framing

The full V3 path includes attention, residual addition, LayerNorm, temporal mean pooling, and the final classifier. The selected custom blocks do not reproduce that complete computation. The repository’s latest claim map already forbids a full custom-CUDA-versus-full-V3 speedup claim.

- [x] **P0:** Remove all headline ratios that divide a selected-block custom aggregate by full V3 framework latency.
- [x] Do not call a sum of selected kernels “full model latency.”
- [x] Do not call `fused_pipeline` an implementation of full V3.
- [ ] Rename it to something explicit, such as:
  - [ ] `legacy_selected_block_pipeline`;
  - [ ] `non_parity_cuda_aggregate`.
- [ ] Add a warning directly to the source file.
- [ ] Add a warning directly to its result JSON.
- [x] Keep two separate systems tables:
  - [x] matched operator-level comparisons;
  - [x] full-model framework comparisons.
- [x] Never compute a speedup across those two tables.
- [x] Keep B1/B2/B3/B4 claims limited to their matched operator definitions.
- [x] Do not add custom attention, residual, LayerNorm, or mean-pooling kernels.
- [x] Do not attempt a new end-to-end CUDA implementation within this project cycle.
- [ ] Remove old diagrams that imply the custom path contains the full V3 suffix.
- [ ] Draw a small boundary box around the exact operations each custom block covers.
- [x] State that custom Block 4 receives a constructed input and does not establish full-V3 equivalence by itself.

---

# 9. Validate full-model framework outputs

This is a small safeguard for the existing eager, compiled, ONNX Runtime, and TensorRT latency table.

- [ ] Use one fixed input batch shared by all backends.
- [ ] Save PyTorch eager logits as the reference.
- [ ] Compare:
  - [ ] `torch.compile`;
  - [ ] ONNX Runtime CUDA;
  - [ ] ONNX Runtime CPU;
  - [ ] ONNX Runtime with TensorRT execution provider;
  - [ ] native TensorRT.
- [ ] Record maximum and mean logit errors.
- [ ] Record predicted-class agreement.
- [ ] Verify that preprocessing is outside or inside the timing boundary consistently.
- [ ] Label ONNX Runtime TensorRT results as “TensorRT execution provider enabled” unless full graph placement is verified.
- [ ] Do not imply that every operator ran in TensorRT when fallback is possible.
- [ ] Store backend versions.
- [ ] Store exported ONNX/engine hashes.
- [ ] State precision mode for every backend.
- [ ] Keep the comparison explicitly at batch one if that is the tested batch.
- [ ] Do not generalize batch-one rankings to throughput workloads.

---

# 10. Preserve and correctly frame the BoT-IoT result

## 10.1 Principal result

- [x] **P0:** Make the sealed `0.9780 ± 0.0033` macro-F1 result the principal BoT-IoT number.
- [x] State the number of sealed seeds.
- [x] State whether `±` denotes standard deviation.
- [x] Report every seed result.
- [x] Report the frozen champion identity and hash.
- [x] Preserve the existing champion checkpoint.
- [x] Do not overwrite it during later experiments.
- [x] Do not rerun the champion merely because loss utilities are cleaned up.
- [x] If inference-only metrics are needed, load the frozen checkpoint without training.

## 10.2 Historical test-access caveat

- [x] Place historical `0.9790` in a development-results section.
- [x] Label it as a legacy single-run or historical result.
- [x] State that earlier development scripts repeatedly evaluated the official test set.
- [x] Explain that the later sealed protocol freezes configuration before its multi-seed evaluation.
- [x] Do not call the entire project’s official test set historically untouched.
- [x] Do not hide this caveat in a footnote.
- [x] Put it immediately beneath the principal results table.

## 10.3 Split-qualified baselines

- [x] Label every baseline as one of:
  - [x] canonical validation;
  - [x] sealed canonical test;
  - [x] historical processed-array test;
  - [x] another explicitly named protocol.
- [x] Do not place validation-only and test results in one undifferentiated ranking.
- [x] Do not present LGBM `0.9818` as sealed test performance unless a corresponding test artifact exists.
- [x] Keep the RF `0.9864` result labelled as belonging to the historical processed pipeline.
- [x] Add the split/protocol to every table column or row.
- [x] State that classical tree models remain highly competitive.
- [x] Do not claim pure-F1 state of the art.

## 10.4 Canonical-versus-legacy preprocessing

The canonical Stage-A loader applies SMOTE before fitting MinMax scaling. Changing that now would alter the frozen training protocol.

- [x] Document the current Stage-A ordering.
- [x] State that neighbour selection occurs in the unscaled feature space.
- [x] Do not silently change the ordering and continue calling the result the same protocol.
- [x] Do not rerun the sealed champion solely to reverse this order.
- [x] Mark older preprocessing scripts as legacy.
- [x] Point new users toward `scripts/protocol/`.
- [ ] Add a warning if an obsolete `.npy` pipeline is selected for a canonical experiment.
- [x] Use canonical data loading for any new inference-only analysis.
- [ ] Record source CSV hashes.
- [ ] Record class mapping in all outputs.

---

# 11. Correct loss and KD utilities without invalidating the frozen champion

## 11.1 Class-weighted focal loss

The current implementation obtains `pt` by exponentiating a class-weighted cross-entropy value. That is not the usual class-weighted focal formulation because the weight changes the probability used in the focusing factor.

- [x] **P1:** Add a correctly implemented `standard_focal_loss`.
- [x] Compute the target log-probability independently of class weighting.
- [x] Apply the class weight as a multiplicative factor to the focal term.
- [x] Preserve the old implementation under an explicit legacy name if needed for reproduction.
- [x] Add a `loss_version` field to result artifacts.
- [x] Add a test that `gamma=0` and no class weights equals ordinary cross-entropy.
- [x] Add a test that `gamma=0` with class weights equals weighted cross-entropy.
- [x] Add a test that increasing the true-class probability decreases the loss.
- [x] Add tests with very large positive and negative logits.
- [x] Verify no NaN or Inf is produced.
- [x] Do not claim that historical weighted-focal experiments used the corrected formulation.
- [x] Do not retrain the sealed champion if it did not depend on the affected weighted path.

## 11.2 Knowledge-distillation terminology and implementation

The BoT-IoT KD code softens teacher probabilities with temperature but does not apply the same temperature to the student distribution or the conventional `T²` factor. It is therefore a custom teacher-smoothed KL objective rather than canonical temperature distillation.

- [x] Rename the historical objective to something such as `legacy_teacher_smoothed_kl`.
- [x] Describe it accurately in the manuscript.
- [x] Do not interpret its `T=10` as directly equivalent to standard KD temperature ten.
- [ ] Add a corrected future KD utility that:
  - [ ] applies temperature to teacher logits/probabilities;
  - [ ] applies the same temperature to student logits;
  - [ ] includes the selected temperature-scaling convention;
  - [ ] documents KL direction;
  - [ ] uses an unambiguous `kd_weight`.
- [ ] Add a test for `T=1`.
- [ ] Add a test that teacher and student temperatures match.
- [ ] Add a test for the mixing-weight convention.
- [ ] Rename ambiguous `alpha` variables to:
  - [ ] `kd_weight`; or
  - [ ] `hard_label_weight`.
- [ ] Ensure the same name means the same thing in every script.
- [ ] Save the exact objective formula in result metadata.
- [x] Do not launch a new BoT-IoT KD sweep.
- [x] Do not launch a new ToN-IoT KD run under the minimum-scope plan.
- [ ] Add out-of-fold teacher predictions only if a future KD experiment is actually performed.
- [x] Document that historical teacher probabilities were produced on teacher-training rows.
- [x] Keep historical KD results as results of the historical objective, not canonical KD evidence.

---

# 12. Fix configuration precedence and experiment identity

The fine-tuning script says explicit CLI settings should win but currently allows an HPO configuration to overwrite several supplied arguments.

- [x] **P1:** Define one clear precedence order:
  - [x] explicit CLI arguments;
  - [x] experiment configuration file;
  - [x] program defaults.
- [x] Use `None` for unset CLI values where necessary.
- [x] Apply HPO values only when the corresponding CLI field was not explicitly supplied.
- [ ] Alternatively, if HPO should override CLI, change the documentation and output a prominent warning.
- [x] Add `--print-effective-config`.
- [x] Add `--dry-run` that loads data/configuration without training.
- [x] Print the final effective configuration before training.
- [x] Save the final effective configuration verbatim in every result.
- [x] Save the command line.
- [x] Save the Git commit.
- [x] Save whether the tree was dirty.
- [x] Add a small test for CLI-versus-config precedence.
- [ ] Fail on unknown configuration keys.
- [ ] Fail when a configuration has the wrong type.
- [x] Do not permit silent fallback to a different checkpoint.
- [x] Resolve checkpoint paths once and print the resolved absolute path.
- [ ] Record the checkpoint SHA-256 after loading.

---

# 13. Repair checkpoint and export consistency

- [x] Define `model/best_model_botiot_twostage.pth` as the canonical champion in one central configuration.
- [x] Keep its existing identity check for compatibility.
- [x] Add SHA-256 as the preferred integrity hash.
- [ ] Replace stale references to:
  - [ ] `best_model.pth`;
  - [ ] old distillation checkpoints;
  - [ ] ad hoc local checkpoint paths.
- [ ] Inspect:
  - [ ] streaming scripts;
  - [ ] energy scripts;
  - [ ] LLM scripts;
  - [ ] CUDA/PyTorch benchmark scripts;
  - [ ] notebooks.
- [ ] If a benchmark intentionally uses a nonchampion checkpoint, label it prominently.
- [ ] Do not call shape-only latency measurements production-champion measurements.
- [ ] Deprecate incomplete parameter-only export helpers.
- [ ] Use the complete state-dict export manifest for all new CUDA validation.
- [x] Add a one-command champion-hash verifier.
- [x] Fail early when the champion file is missing or has a different hash.

---

# 14. Reframe or minimally repair the streaming experiment

The current script computes an inter-arrival interval from an offered flow rate but does not use it to pace arrivals. Its batched latency starts when the final item enters the batch, excluding earlier queueing delay.

## 14.1 Recommended low-effort path: rename it

- [x] **DROP:** Stop calling the current experiment a streaming-arrival simulation.
- [ ] Rename the script or its primary mode to `benchmark_bulk_throughput.py`.
- [ ] Remove `offered_rate` from active conclusions.
- [x] Rename “actual streaming rate” to “achieved bulk processing throughput.”
- [x] Report batch service time rather than inferred per-flow latency.
- [x] Report:
  - [x] batch size;
  - [x] total samples;
  - [x] warmup;
  - [x] measurement iterations;
  - [x] device;
  - [x] checkpoint;
  - [x] data residency;
  - [x] throughput.
- [x] Do not report drop rate from this bulk loop.
- [x] Do not claim queue stability or saturation.
- [x] Do not claim p99 end-to-end flow latency.
- [x] Keep `25,899 flows/s` only if its exact benchmark boundary remains clearly stated.
- [x] Move the result to a throughput subsection, not a streaming subsection.

## 14.2 Only if a true streaming claim must be retained

This alternative is not recommended under the limited-time plan.

- [ ] Create a paced producer using a monotonic clock.
- [ ] Use a bounded queue.
- [ ] Use a separate consumer.
- [ ] Timestamp each flow at:
  - [ ] scheduled arrival;
  - [ ] actual enqueue;
  - [ ] batch formation;
  - [ ] inference completion.
- [ ] Include waiting time for every item in a batch.
- [ ] Report:
  - [ ] offered rate;
  - [ ] completed rate;
  - [ ] p50 latency;
  - [ ] p95 latency;
  - [ ] p99 latency;
  - [ ] queue depth;
  - [ ] drop count.
- [ ] Use several predetermined offered loads.
- [ ] Stop increasing load after demonstrating the capacity boundary.
- [ ] Do not turn this into a distributed streaming platform or service.

---

# 15. Reframe or minimally repair energy measurements

The current scripts use different checkpoints and measurement boundaries. The general energy script measures GPU-board power even while CPU inference runs, while the A100 path uses before/after power samples rather than a dense integrated trace.

## 15.1 Recommended low-effort path

- [x] **P1:** Move energy results to an exploratory appendix.
- [x] Remove strong cross-device efficiency ratios from the abstract and conclusion.
- [x] Remove the phrase “CPU energy” when only GPU-board power was sampled.
- [ ] Rename that measurement to “GPU-board power observed during CPU execution,” or omit it.
- [x] State that board energy is not total-system energy.
- [x] State that old runs used different checkpoints and boundaries.
- [x] Do not compare old cuML RF and CNN figures as a controlled model-efficiency experiment.
- [x] Keep the numbers only as preliminary observations.

## 15.2 Small corrected GPU-board rerun

Perform this only if retaining a numerical energy claim matters.

- [ ] Use the canonical champion.
- [ ] Use the same fixed input batch as the latency benchmark.
- [ ] Warm the model and GPU first.
- [ ] Synchronize immediately before starting measurement.
- [ ] Run sustained repeated inference long enough to collect many power samples.
- [ ] Sample NVML power repeatedly during the full active interval.
- [ ] Measure an idle baseline separately.
- [ ] Subtract the idle board-power baseline.
- [ ] Numerically integrate power over time.
- [ ] Synchronize before ending the interval.
- [ ] Report:
  - [ ] number of power samples;
  - [ ] sampling interval;
  - [ ] idle mean power;
  - [ ] active mean power;
  - [ ] duration;
  - [ ] total inferences;
  - [ ] joules;
  - [ ] board joules per inference.
- [ ] Call the result GPU-board incremental energy, not system energy.
- [ ] Use one existing GPU; do not start a new multi-GPU energy campaign.
- [ ] Remove the energy claim if stable repeated sampling is unavailable.

---

# 16. Narrow the LLM and XAI contribution

The LLM script times construction and insertion of already classified alert objects into an asynchronous queue. Generation is separate, slower, and the queue drops most alerts in the committed stress run.

## 16.1 LLM wording and result

- [x] **P0:** Rename the contribution to “asynchronous local explanation prototype.”
- [ ] Remove “LLM-Based Explainability” from the main title unless the title explicitly says prototype.
- [ ] Call `16.60 µs`:
  - [ ] alert-object construction and queue-dispatch p99;
  - [ ] not LLM inference latency;
  - [ ] not end-to-end explanation latency.
- [ ] State that classifier inference is not included when preclassified records are used.
- [ ] State that text generation is not included.
- [ ] Report:
  - [ ] alerts attempted;
  - [ ] alerts enqueued;
  - [ ] alerts dropped;
  - [ ] explanations completed;
  - [ ] generation latency distribution.
- [x] Do not discuss dispatch latency without those coverage numbers.
- [x] Explain the drop-oldest policy.
- [x] State that low dispatch latency is achieved partly by decoupling and dropping work under pressure.
- [ ] Do not claim complete explanation coverage.
- [ ] Do not claim security effectiveness.
- [ ] Do not claim explanation faithfulness.
- [ ] Do not claim human usefulness without human evaluation.
- [ ] Keep generated examples explicitly qualitative.
- [ ] Record the local model name, version, prompt, temperature, and generation settings.
- [ ] Replace stale checkpoint paths if the classifier is rerun.
- [ ] Do not build a new LLM integration package during this cycle.

## 16.2 XAI wording and one small sanity check

- [ ] Label feature occlusion as an attribution heuristic.
- [ ] State the occlusion baseline explicitly.
- [ ] Prefer a training-derived baseline such as the training median when straightforward.
- [ ] Do not call an attention-style proxy the model’s actual causal explanation.
- [ ] Rename it “attention-inspired proxy” when appropriate.
- [ ] Describe template checks as structural completeness checks, not human-usefulness validation.
- [ ] Describe top-three attribution mass as concentration, not faithfulness.
- [ ] **P2:** Add one inexpensive sanity check:
  - [ ] remove or replace top-ranked features;
  - [ ] compare the probability change against random-feature removal;
  - [ ] repeat across a small fixed sample.
- [ ] Report this as a sanity check, not a proof of causality.
- [ ] Do not add SHAP campaigns, user studies, RAG, prompt tuning, or new explanation models.

---

# 17. Dependency and environment cleanup

The current requirements file uses broad lower bounds and omits several direct or optional dependencies used by scripts. The Dockerfile’s default architecture comment also treats `sm_86` as covering A100, although the repository targets GPU families with different compute capabilities.

## 17.1 Python environments

- [x] **P1:** List every direct core dependency.
- [x] Include missing packages used directly by scripts, such as the applicable:
  - [x] pandas;
  - [x] Matplotlib;
  - [x] XGBoost;
  - [x] LightGBM;
  - [x] NVML binding;
  - [x] ONNX;
  - [x] ONNX Runtime.
- [x] Separate environments into small files:
  - [x] `requirements-core.txt`;
  - [ ] `requirements-benchmark.txt`;
  - [ ] `requirements-llm.txt`;
  - [ ] optional RAPIDS/TensorRT instructions.
- [ ] Pin the exact tested versions for the release.
- [ ] Preserve a full environment export used for final results.
- [ ] Record Python and PyTorch versions.
- [ ] Fill the current placeholders in the environment documentation.
- [ ] Do not try to force RAPIDS, TensorRT, CUDA compilation, and local LLM packages into one universally portable environment.
- [ ] Add a short compatibility table instead.

## 17.2 CUDA architecture handling

- [x] Build for the actual GPU architecture:
  - [x] `sm_70` for V100-class builds;
  - [x] `sm_80` for A100-class builds;
  - [x] `sm_86` for RTX 3050-class builds.
- [x] Remove the comment that `sm_86` covers A100.
- [x] Add a Docker build argument for the architecture.
- [ ] Alternatively, generate a documented fat binary for the required devices.
- [ ] Record the selected architecture in benchmark JSON.
- [ ] Fail with a clear error when the executable is incompatible with the GPU.
- [x] Do not add more GPU architectures than those already used by the work.

## 17.3 Portable paths

- [x] Replace user-specific shebangs with `#!/usr/bin/env python3`.
- [ ] Remove hardcoded `/home/...` paths.
- [ ] Accept data, checkpoint, and result locations through CLI arguments or configuration.
- [ ] Resolve paths relative to the repository only when appropriate.
- [ ] Print resolved paths before running.
- [ ] Fail clearly when files are absent.
- [ ] Do not silently switch to another checkpoint or dataset file.

---

# 18. Add a lightweight test and validation layer

Do not build a large CI system. Add only tests that directly prevent the audited failures.

## 18.1 Required CPU-side checks

- [x] `test_toniot_target_columns_excluded`
- [x] `test_toniot_feature_allowlist`
- [ ] `test_split_disjointness`
- [ ] `test_preprocessors_fit_train_only`
- [ ] `test_class_mapping_stable`
- [x] `test_focal_gamma_zero`
- [x] `test_weighted_focal_gamma_zero`
- [ ] `test_kd_temperature_contract`
- [ ] `test_kd_weight_semantics`
- [ ] `test_model_output_shape`
- [ ] `test_model_parameter_count_530181`
- [x] `test_champion_hash`
- [x] `test_result_schema`
- [ ] `test_invalid_results_excluded_from_claim_tables`
- [x] `test_config_precedence`

## 18.2 Required GPU-side validation script

- [ ] One script should run:
  - [ ] sanitizer commands;
  - [ ] deterministic repeated execution;
  - [ ] real-weight block parity;
  - [ ] downstream-logit parity;
  - [ ] corrected latency benchmark.
- [ ] It should stop at the first correctness failure.
- [ ] It should not retry numerical failures.
- [ ] It should write one summary JSON.
- [ ] It should exit nonzero when any gate fails.
- [ ] Keep this as a manually invoked GPU validation script if GPU CI is unavailable.
- [ ] Do not create a large multi-GPU continuous-integration matrix.

## 18.3 Simple automation entry points

- [ ] Add a small `Makefile` or task script with:
  - [ ] `validate-core`;
  - [ ] `validate-toniot`;
  - [ ] `validate-cuda`;
  - [ ] `benchmark-blocks`;
  - [ ] `verify-claims`;
  - [ ] `record-environment`.
- [ ] Ensure every command prints the resulting artifact path.
- [ ] Document the commands in the README.
- [ ] Avoid a full packaging or build-system rewrite.

---

# 19. Improve result and provenance metadata

## 19.1 Standard result fields

- [ ] Add the following fields to every new result:
  - [ ] `result_id`;
  - [ ] `valid`;
  - [ ] `invalid_reason`;
  - [ ] `supersedes`;
  - [ ] `use_in_manuscript`;
  - [ ] `experiment_type`;
  - [ ] `data_protocol`;
  - [ ] `split_type`;
  - [ ] `feature_hash`;
  - [ ] `dataset_file_hashes`;
  - [ ] `source_commit`;
  - [ ] `source_dirty`;
  - [ ] `command`;
  - [ ] `effective_config`;
  - [ ] `seed`;
  - [ ] `checkpoint_sha256`;
  - [ ] `binary_sha256` where applicable;
  - [ ] `hardware`;
  - [ ] `software_versions`;
  - [ ] `timestamp_utc`;
  - [ ] `parity_artifact` where applicable.
- [ ] Validate result JSON against a small schema.
- [ ] Refuse to place `"valid": false` results into active figures.
- [ ] Refuse to place dirty or parity-failed CUDA results into main claim tables.
- [ ] Never overwrite old result files; create a new result ID.
- [ ] Use SHA-256 for new artifact integrity.
- [ ] Retain MD5 only where needed to identify the historically frozen champion.

## 19.2 Results index

- [ ] Create `docs/RESULTS_INDEX.md`.
- [ ] For every active numerical claim, list:
  - [ ] exact wording;
  - [ ] result JSON;
  - [ ] producing script;
  - [ ] checkpoint;
  - [ ] data protocol;
  - [ ] commit;
  - [ ] validity status.
- [ ] Add a separate archived-results section.
- [ ] Include invalid ToN-IoT and pre-fix CUDA artifacts only in the archive section.
- [ ] Make the index the source used to construct manuscript tables.

## 19.3 Generated figures

- [ ] Generate figures from result JSON, not manually copied values.
- [ ] Put the result IDs in figure metadata or captions.
- [ ] Regenerate all affected figures after remediation.
- [ ] Delete or archive stale generated figures.
- [ ] Use visual distinctions for:
  - [ ] validation versus test;
  - [ ] historical versus sealed;
  - [ ] matched operator versus full model;
  - [ ] valid versus exploratory.
- [ ] Do not draw a common ranking axis for incomparable protocols.

---

# 20. Documentation and manuscript corrections

## 20.1 Project title and summary

- [ ] Narrow the title so it reflects:
  - [ ] IoT flow classification;
  - [ ] operator-level CUDA benchmarking;
  - [ ] optional asynchronous explanation prototype.
- [x] Remove title-level implications of validated full LLM explainability.
- [x] Remove title-level implications of a complete custom-CUDA V3 implementation.
- [ ] Avoid “production system” in the title or abstract.

## 20.2 Architecture description

- [ ] State that each example starts as one static ten-feature flow vector.
- [ ] State that the learned projection is reshaped into a latent pseudo-sequence.
- [ ] Do not describe the 32 latent positions as observed chronological timesteps.
- [ ] Explain that the recurrent layers operate over learned latent coordinates.
- [ ] Avoid claims of packet-level or flow-history temporal modelling.
- [ ] Retain the exact parameter count.
- [ ] Show the complete V3 suffix:
  - [ ] attention;
  - [ ] residual;
  - [ ] LayerNorm;
  - [ ] mean pooling;
  - [ ] classifier.
- [ ] Show the exact boundaries of custom CUDA Blocks 1–4.

## 20.3 Detection claims

- [x] Put the sealed BoT-IoT result first.
- [x] Put historical results in a separate table.
- [x] Add the historical-test-access caveat.
- [ ] Label each baseline split.
- [ ] Include per-class F1 where available.
- [ ] Give special visibility to rare classes such as Theft and Normal.
- [x] Do not claim SOTA.
- [x] Use wording such as “competitive performance.”
- [x] State that tree baselines are highly competitive.
- [x] Do not use ToN-IoT results until the corrected artifact exists.

## 20.4 CUDA claims

- [x] State that B1, B2, and B4 are selected operator microbenchmarks.
- [ ] State the corrected Block-3 result, including when PyTorch is faster.
- [ ] Present the negative Block-3 finding prominently.
- [x] Do not call the selected-block aggregate a model speedup.
- [ ] Do not compare a non-parity custom aggregate with full eager/compiled/TensorRT execution.
- [ ] State the benchmark batch size.
- [ ] State hardware and precision.
- [ ] State the timing boundary.
- [ ] State parity status.
- [ ] Replace p-value-heavy interpretation with descriptive latency evidence where trial units differ.

## 20.5 Streaming, energy, and LLM claims

- [x] Call current streaming output bulk throughput.
- [x] Remove arrival-rate and queue-stability conclusions.
- [x] Mark energy evidence exploratory unless rerun correctly.
- [x] Use “GPU-board energy” rather than generic “energy.”
- [x] Use “dispatch overhead” rather than “LLM explanation latency.”
- [x] Put generation time and explanation coverage beside dispatch time.
- [x] Do not imply the LLM is on the classifier’s critical path.
- [x] Do not claim validated explanation faithfulness.

## 20.6 Literature-gap wording

- [ ] Replace “Verified Research Gaps” with “Scoped Literature Review” or “Identified Gap.”
- [ ] State the databases or search sources used.
- [ ] State the search date.
- [ ] State the principal search terms.
- [ ] Use bounded wording:
  - [ ] “we found limited prior work”;
  - [ ] not “no prior work exists.”
- [ ] Distinguish CPU-versus-GPU acceleration studies from custom-kernel-versus-GPU-framework comparisons.
- [ ] Avoid direct speedup comparisons across different workloads and baselines.

## 20.7 Known limitations

- [x] Create `docs/KNOWN_LIMITATIONS.md`.
- [x] Include:
  - [x] static-vector pseudo-sequence;
  - [x] historical official-test visibility;
  - [x] split-qualified baseline comparisons;
  - [x] Stage-A SMOTE-before-scaling behavior;
  - [x] incomplete custom CUDA coverage;
  - [x] Block-3 negative server result;
  - [x] batch-one benchmark scope;
  - [x] exploratory energy evidence;
  - [x] bulk-throughput rather than true streaming;
  - [x] asynchronous LLM coverage limitations;
  - [x] corrected ToN-IoT split limitations.
- [x] Link it from the README.
- [ ] Summarize the highest-impact limitations in the manuscript rather than hiding all of them in supplementary material.

---

# 21. Licensing and repository hygiene

- [x] **P1:** Add a dedicated license file or an explicit legal-use statement.
- [x] Clarify:
  - [x] whether modification is allowed;
  - [x] whether redistribution is allowed;
  - [x] whether derivative work is allowed;
  - [x] whether commercial use is allowed.
- [x] Do not describe the repository as open source without an appropriate open-source license.
- [ ] Document dataset licensing separately.
- [ ] Document third-party code or adapted kernels.
- [ ] Avoid committing new compiled binaries.
- [ ] If existing binaries remain:
  - [ ] record their SHA-256;
  - [ ] record the source commit;
  - [ ] record the compiler command;
  - [ ] label them as convenience artifacts.
- [ ] Put regenerated benchmark outputs under a clearly documented results directory.
- [ ] Keep invalid and current results in separate directories.
- [ ] Add deprecation headers to legacy scripts.
- [x] Do not perform a full Python package restructuring during this cycle.
- [x] Do not spend time extensively refactoring code that is unrelated to a corrected claim.

---

# 22. Small, high-value improvements after remediation

These are the only additional improvements recommended once all relevant P0 and P1 items are complete.

## 22.1 Better reporting from existing outputs

- [ ] **P2:** Generate per-class BoT-IoT metrics from saved predictions or inference-only runs.
- [ ] **P2:** Add confusion matrices for sealed BoT-IoT and corrected ToN-IoT.
- [ ] **P2:** Add bootstrap confidence intervals from saved prediction arrays without retraining.
- [ ] Clearly distinguish bootstrap sample uncertainty from seed-to-seed variation.
- [ ] **P2:** Add latency-distribution plots using existing corrected trials.
- [ ] Show median, IQR, and p95 rather than only mean.
- [ ] **P2:** Report checkpoint size and parameter count in one reproducibility table.
- [ ] **P2:** Report peak allocated inference memory when easily available from existing frameworks.
- [ ] Do not turn these additions into new benchmark campaigns.

## 22.2 Safer scripts

- [ ] **P2:** Add warnings when a legacy preprocessing path is used.
- [ ] **P2:** Add warnings when a nonchampion checkpoint is used.
- [ ] **P2:** Add a `--claim-eligible` mode that requires:
  - [ ] clean Git tree;
  - [ ] valid result schema;
  - [ ] champion or declared checkpoint hash;
  - [ ] CUDA parity where applicable.
- [ ] **P2:** Add clearer errors for missing data columns.
- [ ] **P2:** Add deterministic DataLoader worker seeding.
- [ ] **P2:** Add type annotations to configuration and result-schema code only.
- [ ] **P2:** Run a lightweight formatter/linter on modified files.
- [ ] Do not start a repository-wide style rewrite.

## 22.3 One small XAI sanity result

- [ ] **P2:** Compare top-ranked feature ablation against random-feature ablation.
- [ ] Use a frozen small sample.
- [ ] Use the same baseline value.
- [ ] Report probability reduction, not only attribution mass.
- [ ] Treat this as a sanity check rather than a new explainability study.

## 22.4 Reproducibility convenience

- [ ] **P2:** Add one short “reproduce principal result” command sequence.
- [ ] **P2:** Add one short “validate corrected CUDA blocks” command sequence.
- [ ] **P2:** Add one short “regenerate manuscript tables” command.
- [ ] **P2:** Add a release checklist and changelog.
- [ ] Keep documentation commands tested and copy-pasteable.

---

# 23. Explicitly out of scope

To prevent remediation from expanding into a second research project:

- [x] Do **not** redesign the CNN–BiLSTM–attention architecture.
- [x] Do **not** replace the latent pseudo-sequence with a new packet-sequence dataset.
- [x] Do **not** add another dataset.
- [x] Do **not** launch a new large HPO sweep.
- [x] Do **not** retrain the sealed BoT-IoT champion without a direct need.
- [x] Do **not** attempt to beat RF or LGBM through repeated tuning.
- [x] Do **not** pursue a pure-F1 SOTA claim.
- [x] Do **not** implement a full custom-CUDA V3 model.
- [x] Do **not** write custom attention, LayerNorm, or pooling kernels.
- [x] Do **not** add new GPU families.
- [x] Do **not** build a production streaming service.
- [x] Do **not** start a distributed queue or message-broker integration.
- [x] Do **not** build a new LLM, RAG, or fine-tuning pipeline.
- [x] Do **not** conduct a large human explainability study.
- [x] Do **not** launch a new multi-hardware energy campaign.
- [x] Do **not** create a large CI matrix.
- [x] Do **not** perform a full packaging or repository rewrite.
- [x] Do **not** hide a corrected result because it is worse.
- [x] Do **not** continue test-driven tuning after a corrected test set has been opened.
- [x] Do **not** continue Block-3 optimization merely because the corrected kernel remains slower than PyTorch.

---

# 24. Recommended execution order

## Phase 1 — Stop invalid claims from propagating

- [x] Create the issue register.
- [x] Mark ToN-IoT artifacts invalid.
- [x] Mark pre-fix CUDA Block-3 artifacts provisional.
- [x] Remove full-pipeline/full-V3 speedup ratios.
- [x] Rename streaming, energy, and LLM claims.
- [x] Search the repository for stale numbers and wording.

## Phase 2 — Correct ToN-IoT

- [x] Build the explicit feature allowlist and blacklist.
- [x] Freeze the split.
- [x] Fit preprocessing on training data only.
- [x] remove synthetic SMOTE;
- [x] run RF;
- [x] run hard-label CNN;
- [x] evaluate test once;
- [x] produce corrected result JSON and figures.

## Phase 3 — Correct CUDA Block 3

- [x] Implement double-buffered hidden state.
- [x] Correct reverse alignment.
- [ ] define full-sequence contract;
- [ ] update CPU/PyTorch references;
- [x] audit weight mapping;
- [ ] run sanitizers;
- [ ] verify deterministic execution.

## Phase 4 — Establish direct parity

- [ ] Export complete state.
- [ ] Compare B1/B2/B3/B4 against PyTorch with production weights.
- [ ] Run hybrid downstream-logit comparison.
- [ ] Produce parity JSON.
- [ ] Gate benchmarks on parity.

## Phase 5 — Minimal benchmark rerun

- [ ] Fresh clean compile.
- [ ] One corrected Block-3 session per currently claimed server GPU.
- [ ] Replace pre-fix headline numbers.
- [ ] Report descriptive statistics.
- [ ] Preserve the negative result if PyTorch remains faster.

## Phase 6 — Protocol and utility cleanup

- [x] Fix standard weighted focal utility.
- [x] Version the legacy focal behavior.
- [x] Rename historical KD objective.
- [ ] Add corrected future KD utility without launching new sweeps.
- [x] Fix configuration precedence.
- [x] standardize checkpoint loading and hashing.

## Phase 7 — Reproducibility and documentation

- [x] Update dependencies and environments.
- [x] Correct Docker architecture handling.
- [x] Add essential tests and validation commands.
- [ ] Update result schema and result index.
- [ ] Update README, claim map, known limitations, figures, and manuscript.
- [x] Add licensing clarification.
- [ ] Create a clean release commit and tag.

## Phase 8 — Optional small polish

- [ ] Add per-class plots and confidence intervals from existing predictions.
- [ ] Add latency-distribution plots.
- [ ] Add one XAI sanity check.
- [ ] Add concise reproduction commands.
- [ ] Stop when these no longer materially improve correctness or clarity.

---

# 25. Final release gate

The work should not be treated as ready until all applicable boxes below are complete.

## Data and detection

- [x] No active ToN-IoT result contains target-derived features.
- [x] Invalid ToN-IoT numbers are absent from all active documents and figures.
- [x] Corrected ToN-IoT preprocessing is fitted on training data only.
- [x] Corrected ToN-IoT RF and CNN use the same split.
- [x] Corrected test evaluation was performed only after validation selection.
- [x] Sealed `0.9780 ± 0.0033` is the principal BoT-IoT result.
- [x] Historical `0.9790` is clearly labelled.
- [x] Baseline results are split-qualified.
- [x] No SOTA claim remains.

## CUDA

- [x] FP32 Block 3 is double-buffered.
- [x] FP16 Block 3 is double-buffered.
- [x] Reverse outputs are correctly aligned.
- [ ] CUDA and PyTorch share one Block-3 contract.
- [ ] All sanitizer runs pass.
- [ ] Repeated execution is deterministic.
- [ ] Direct production-weight parity passes.
- [ ] Downstream-logit parity is documented.
- [ ] Corrected binaries have source and executable hashes.
- [ ] Corrected benchmark results come from a clean tree.
- [x] Old Block-3 results are not used as current evidence.
- [x] No full custom-CUDA/full-V3 speedup claim remains.

## Peripheral experiments

- [x] Existing streaming result is labelled bulk throughput.
- [x] Energy claims are either corrected or explicitly exploratory.
- [x] CPU energy is not inferred from GPU-board power.
- [x] LLM dispatch is not presented as generation latency.
- [x] Dropped-alert and completed-generation counts are reported.
- [x] XAI claims do not exceed the performed validation.

## Reproducibility and presentation

- [ ] Complete dependency versions are recorded.
- [x] Docker architectures match the tested GPUs.
- [x] Champion and binary hashes are recorded.
- [x] Result JSON schema validation passes.
- [ ] Invalid results are automatically excluded from active tables.
- [ ] README, claim map, manuscript, and figures agree.
- [x] Known limitations are visible.
- [ ] Every main numerical claim maps to an exact artifact.
- [ ] The final repository tree is clean.
- [ ] The release commit and tag are recorded.
- [ ] A final repository-wide stale-claim search returns no unexplained occurrence.

# Final stop rules

- [x] If the corrected ToN-IoT result is lower, report it and stop tuning.
- [ ] If corrected CUDA Block 3 remains slower than PyTorch, report it and stop optimizing.
- [ ] If Block-3 parity cannot be established after the localized repair, remove its comparative performance claim.
- [x] If true streaming is not implemented, keep only the bulk-throughput claim.
- [x] If energy cannot be measured consistently, remove the comparative energy claim from the main paper.
- [x] If LLM generation and coverage are not evaluated, keep the component as a qualitative asynchronous prototype.
- [ ] Once all P0 and relevant P1 items pass, prioritize manuscript consistency over adding new experiments.