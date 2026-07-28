# One-Epoch ECAPA-TDNN Fine-Tuning Pilot + Fixed Validation v1

Date: 2026-07-28

## Result

**Technical pipeline: PASS. Scope compliance: FAIL. Overall result: FAIL.
Model-quality result: IMPROVED.**

Exactly epoch 0 was trained: 1,000 optimizer updates and 32,000 logical sample
selections. Epoch 1 was not started. The epoch checkpoint passed a fresh-object
roundtrip without another optimizer step, all 8,504 approved validation
utterances were embedded, and exactly 19,528 immutable validation trials were
scored. The pilot interpolated EER was `0.0646251536255633` (6.462515%), below
the historical pretrained baseline `0.11665301106104056` (11.665301%).

The post-run scope audit found that the executed version's existing sampler
metadata validator called `Path.is_file()` on seven train shard paths that the
epoch plan did not ultimately select. Those seven files were not opened,
loaded, or hashed, and no validation-quarantine or final-test path was touched,
but a stat is still outside the task's exact selected-shard allowlist. The
strict `scope_compliance_pass` and therefore `overall_pass` are false. The
versioned report intentionally overrides the original runtime JSON's
overbroad self-reported scope/overall PASS flags.

## Exact scope

This task added one focused helper, one explicit CUDA entry point, and focused
synthetic tests; executed one balanced Hybrid P16K2 W8 epoch; wrote resumable
checkpoints; evaluated only the existing fixed validation protocol; compared
the result with the historical pretrained baseline; and wrote versioned
reports plus an append-only worklog entry.

No recursive repository, `outputs/`, cache, or manifest listing was used. No
final-test artifact was accessed, enumerated, statted, hashed, opened, loaded,
or evaluated. No WAV was read, `compute_features` was not called,
`src/fbank.py` was not used, the pretrained VoxCeleb classifier was not used,
and no cache, source audio, manifest, split, trial, pretrained weight, or
environment was modified. No dependency was installed or upgraded. No commit
or push was performed. The seven extra train-shard existence checks are the
only identified strict-scope exception.

## Strict scope-compliance audit

The train index referenced 125 train shards. The deterministic epoch-0 plan
selected 118; only those 118 were opened, tensor-loaded, and included in the
protected pre/post hash map. Before the plan was known, the existing
`validate_train_sampler_metadata` implementation checked existence for all
125 paths. The seven unselected paths statted were:

```text
train/shard_00075.pt
train/shard_00083.pt
train/shard_00084.pt
train/shard_00086.pt
train/shard_00112.pt
train/shard_00114.pt
train/shard_00117.pt
```

They were not opened, loaded, or hashed. They are all train-cache paths, not
quarantined validation or final-test paths. Nevertheless, the task says to use
only exact sampler-selected train shards, so the final audit classifies this
as strict scope-compliance FAIL. It does not invalidate the numerical
training, checkpoint, validation, or model-quality results, which remain
technical PASS.

The final implementation adds an opt-out for broad shard-existence validation.
The pilot now materializes its metadata-only deterministic plan without
statting shard files and then stats, hashes, and loads only the selected
allowlist. A synthetic test proves the behavior. The corrected code was not
rerun on CUDA because this task authorized exactly one training epoch.

## Allowlisted files inspected

The mandatory repository context was read before implementation and execution:

- `AGENTS.md`, `.gitignore`, full `reports/CODEX_WORKLOG.md`, initial
  `git status --short`, and targeted task-file diffs.
- `reports/aam_softmax_training_smoke_v1.md`,
  `reports/aam_softmax_training_smoke_v1.json`,
  `reports/aam_softmax_training_smoke_v1_1.md`, and
  `reports/aam_softmax_training_smoke_v1_1.json`.
- `reports/validation_trials_v1_summary.md`,
  `reports/pretrained_ecapa_validation_baseline_v1.md`,
  `reports/pretrained_ecapa_validation_baseline_v1.json`, and
  `reports/validation_eer_metrics_patch_v1.md`.
- `src/aam_training.py`, `src/cached_fbank_dataset.py`,
  `src/cached_fbank_samplers.py`, `src/verification_trials.py`,
  `src/verification_metrics.py`, and `src/verification_baseline.py`.
- `scripts/smoke_test_aam_training_cuda.py`,
  `scripts/evaluate_pretrained_validation_baseline.py`, and
  `scripts/recompute_pretrained_validation_metrics.py`.
- `tests/test_aam_training.py`, `tests/test_verification_trials.py`,
  `tests/test_verification_metrics.py`, and
  `tests/test_verification_baseline.py`.

Execution opened and loaded only the approved cache configuration, train and
validation indexes, the exact 118 train shards selected by epoch 0, the exact
34 validation shards referenced by the validation index, the approved
validation manifest, the fixed validation trial CSV/config, and the existing
local `speechbrain/spkrec-ecapa-voxceleb` model. In addition, it performed the
seven metadata-only train-shard stats disclosed above. It did not inspect
sibling split artifacts.

## Files created

- `src/ecapa_one_epoch_pilot.py`
- `scripts/run_ecapa_aam_one_epoch_pilot.py`
- `tests/test_ecapa_one_epoch_pilot.py`
- `reports/ecapa_aam_one_epoch_pilot_v1.md`
- `reports/ecapa_aam_one_epoch_pilot_v1.json`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/console.stdout.log`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/console.stderr.log`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/training_log.jsonl`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/logical_losses_epoch_000.json`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/last.pt`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/epoch_000.pt`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/best.pt`
- Ignored
  `outputs/ecapa_aam_one_epoch_pilot_v1/validation_embeddings_epoch_000.pt`
- Ignored
  `outputs/ecapa_aam_one_epoch_pilot_v1/validation_scores_epoch_000.pt`
- Ignored
  `outputs/ecapa_aam_one_epoch_pilot_v1/validation_metrics_epoch_000.json`
- Ignored
  `outputs/ecapa_aam_one_epoch_pilot_v1/validation_runtime_epoch_000.json`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/pilot_runtime.json`

## Files modified

- `src/cached_fbank_samplers.py`
- `reports/CODEX_WORKLOG.md` (append only)

The sampler change is backward compatible: existing callers retain shard
existence validation by default; only the fixed pilot explicitly disables it
until its exact selected allowlist is known. No existing training, dataset, or
verification implementation was otherwise modified.

## Implementation decisions

The pilot reuses the smoke-tested AAM classifier, BatchNorm policy, optimizer
group construction, deterministic round-robin logical-batch arrangement,
cached-Fbank Dataset, sampler, trial reader/scorer, and tied-score-safe EER
implementation. The new helper is deliberately limited to fixed pilot
constants, strict schemas and validators, one-epoch state transitions,
checkpoint creation/loading, logging, selection, and comparison logic.

The explicit runner has fixed approved paths rather than user-supplied data
paths. It precomputes the deterministic epoch-0 plan, validates exact P16K2
composition and index uniqueness, hashes only the explicitly approved files it
will access, and fails closed on non-finite data or training state, missing or
zero gradients, skipped AdamW updates, changed BatchNorm buffers, malformed
artifacts, path ownership failures, or identity mismatches.

The final source additionally avoids sampler-wide shard existence checks while
planning. This hardening followed the post-run discovery described above and
was verified synthetically, not by another CUDA execution.

## Train configuration

- Interpreter: `.venv-cuda/Scripts/python.exe`
- Python / PyTorch / SpeechBrain: `3.10.11` / `2.2.0+cu121` / `1.0.3`
- Device: `cuda:0`, NVIDIA GeForce RTX 3050 Laptop GPU
- Pretrained model: `speechbrain/spkrec-ecapa-voxceleb`
- Production path:
  `[B,301,80] float32 cached Fbank -> mean_var_norm -> embedding_model ->
  [B,1,192] -> squeeze(1) -> [B,192]`
- Trainable: all `embedding_model` parameters and the new AAM weight
- Excluded: `mean_var_norm`, pretrained classifier, and every waveform/frontend
  component
- Epoch: exactly `0`
- Logical batches / batch size / selections: `1000` / `32` / `32000`
- Physical microbatch / accumulation: `4` / `8`
- DataLoader workers / cached shard bound: `0` / `8`
- No microbatch-2 fallback, augmentation, scheduler, warmup, gradient clipping,
  or early stopping

The training index audit found 31,998 train rows, 488 speakers, and labels
exactly `0..487`. The validation index audit found 8,504 rows, 100 speakers,
labels all `-1`, and speaker disjointness from train.

## Sampler and logical ordering

- `HybridShardAwareSpeakerBatchSampler`
- `P=16`, `K=2`, active shard window `8`
- Seed `20260727`, sampler epoch `0`
- Round-robin physical order: the first utterance from each of the 16 speakers,
  followed by the second utterance from each speaker
- Every physical microbatch of four therefore contained four distinct speakers
- Exact epoch plan SHA-256:
  `796f9f3ec2eff82a45052e0482cfde997a5b793b22ddcd6741e68f62ee196b04`
- All 1,000 logical batches were exact P16K2 and no logical batch contained a
  duplicate Dataset index
- 16,897 unique Dataset indexes and 15,103 repeated selections across the
  32,000 selections
- All 488 speakers selected; exposure min/median/max `56 / 68 / 76`
- Queue cycles: `1666`

## BatchNorm, AAM, optimizer, and AMP policy

The `embedding_model` remained in training mode while only its 31 BatchNorm
modules were placed in eval mode. BatchNorm affine weights/biases remained
trainable; all 93 `running_mean`, `running_var`, and
`num_batches_tracked` buffers stayed bit-exact at every rolling boundary,
after the final update, and before validation. All non-BatchNorm ECAPA modules
remained in training mode during training.

The new bias-free AAM weight was `[488,192]`, with margin `0.2` radians and
scale `30.0`. Angular-margin math, logits, and summed cross-entropy were
float32. AdamW used two exact disjoint groups:

- ECAPA `embedding_model`: learning rate `1e-5`, weight decay `1e-4`
- AAM classifier: learning rate `1e-3`, weight decay `1e-4`

ECAPA forward used CUDA float16 autocast. GradScaler was enabled with initial
scale `128`; all 1,000 updates kept scale `128`, the optimizer post-hook fired
1,000 times, and zero updates were skipped. Every intended parameter acquired
a finite gradient and an AdamW step counter equal to the global step.

## One-epoch execution

- Task start/end UTC:
  `2026-07-28T08:00:46.030100+00:00` /
  `2026-07-28T08:12:10.434877+00:00`
- Total task duration: `684.4041123000025` seconds
- Training start/end UTC:
  `2026-07-28T08:00:59.590293+00:00` /
  `2026-07-28T08:11:04.479631+00:00`
- Training duration: `604.8890577999991` seconds
- Optimizer updates: `1000`
- Logical selections: `32000`
- Epochs started/completed: `[0] / [0]`
- Final cursor: next epoch `1`, next logical batch position `0`
- Epoch 1 started: `false`
- OOM: none

All checked cached inputs, normalized features, embeddings, logits, losses,
and gradients were finite. Aggregate ECAPA and AAM gradients were nonzero at
all 1,000 optimizer steps.

## Loss and gradients

Logical loss is the sum of all 32 per-sample cross-entropies divided by 32.

- Loss count/first/final:
  `1000 / 14.10267436504364 / 1.342349648475647`
- Loss mean/minimum/maximum:
  `5.100674975889735 / 0.47721143439412117 / 15.313348054885864`
- ECAPA gradient norm first/final/mean/minimum/maximum:
  `59.820916009264685 / 38.96678272901164 / 48.718236445551575 /
  22.10841567044168 / 67.12837878484218`
- AAM gradient norm first/final/mean/minimum/maximum:
  `8.563820396025983 / 4.2688503717152875 / 6.506972899575333 /
  2.6217307668304 / 8.678117474452085`
- Step duration first/final/mean/minimum/maximum seconds:
  `2.460299999998824 / 0.5069484000014199 / 0.5347478141999782 /
  0.47203930000250693 / 2.460299999998824`

Training loss is reported as an optimization diagnostic only, not as evidence
of verification improvement.

### Logging-window history

The JSONL log contains exactly 20 detached scalar-only records at steps
50 through 1,000 in increments of 50. The full 1,000-loss sequence is retained
only in the ignored machine-readable loss artifact.

| Step | Mean logical loss over latest 50 steps |
| ---: | ---: |
| 50 | 14.08393856048584 |
| 100 | 12.019001777172088 |
| 150 | 10.432876251935959 |
| 200 | 9.202434445619582 |
| 250 | 7.92286917924881 |
| 300 | 6.861934618353843 |
| 350 | 5.84722490131855 |
| 400 | 4.970992584228515 |
| 450 | 4.321863998845219 |
| 500 | 3.764565050005913 |
| 550 | 3.415242707952857 |
| 600 | 2.9405364802107217 |
| 650 | 2.618602541126311 |
| 700 | 2.476743625961244 |
| 750 | 2.1956059302017095 |
| 800 | 2.109951032437384 |
| 850 | 1.907561029214412 |
| 900 | 1.6893292402662337 |
| 950 | 1.7030389439594 |
| 1000 | 1.5291866192501038 |

## Checkpoints

All saves used a temporary file followed by `os.replace`. Checkpoint paths are
repository-relative and checkpoint payloads contain no cache tensor,
validation embedding, trial file, WAV, final-test data, or pretrained
classifier state.

| Checkpoint event | Step | Next cursor | Size (bytes) | Write + validation (s) | Result |
| --- | ---: | --- | ---: | ---: | --- |
| `last.pt` | 250 | epoch 0, batch 250 | 250,708,244 | 0.6844723 | PASS |
| `last.pt` | 500 | epoch 0, batch 500 | 250,708,244 | 0.6463033 | PASS |
| `last.pt` | 750 | epoch 0, batch 750 | 250,708,244 | 0.3619563 | PASS |
| `last.pt` | 1000 | epoch 1, batch 0 | 250,708,244 | 0.4403965 | PASS |
| `epoch_000.pt` | 1000 | epoch 1, batch 0 | 250,711,923 | 0.3616463 | PASS |
| `best.pt` | 1000 | epoch 1, batch 0 | 250,710,356 | 0.7127477 | PASS |

`epoch_000.pt` SHA-256:
`e82ba006aef4a505244769f137af897a94bbb601a26ffe2b68fbfec135201b67`.
Every boundary was non-empty, readable, schema-valid, counter-valid, and
BatchNorm-exact.

## Epoch checkpoint roundtrip

The immutable epoch checkpoint was loaded into fresh encoder components, AAM,
AdamW, and GradScaler objects. The embedding model, mean/variance
normalization, AAM, optimizer including all Adam moments/step counters,
GradScaler, RNG, counters, configuration, and BatchNorm baseline matched
exactly. The BatchNorm policy was reapplied. Only `mean_var_norm` and
`embedding_model` were retained from the pretrained encoder; neither
`compute_features` nor the pretrained classifier was retained.

The loaded checkpoint represented epoch 0 complete, global optimizer step
1,000, next epoch 1, batch position 0. Zero optimizer steps were taken after
the load, so the roundtrip did not begin epoch 1.

## Validation extraction

Validation began only after training, epoch checkpoint creation, and the
fresh-object roundtrip passed. Gradients were cleared, the embedding model was
put in eval mode, and extraction ran under `torch.inference_mode()` with batch
size 32, `num_workers=0`, deterministic sequential traversal, and no
augmentation.

- Rows / speakers: `8504 / 100`
- Saved embedding shape/dtype: `[8504,192] / torch.float32` CPU
- All embeddings finite: true
- Extraction time: `58.01379679999809` seconds
- Throughput: `146.58582042677614` utterances/second
- AAM, pretrained classifier, waveform frontend, and `compute_features` calls:
  `0 / 0 / 0 / 0`

The saved embedding, score, metric, and runtime artifacts were reloaded and
their schemas, checkpoint identity, trial identity, dimensions, finiteness,
and counts were revalidated.

## Fixed-trial scoring

- Immutable trial CSV:
  `manifests/verification/validation_trials_v1.csv`
- Required and observed SHA-256:
  `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`
- Positive / negative / total trials: `9764 / 9764 / 19528`
- Trial path ownership validated against authoritative validation metadata:
  true
- Scoring time: `0.7327373000007356` seconds
- Metric time: `0.12257150000004913` seconds
- Rule: cosine score `>=` threshold means same speaker

The trial CSV/config was neither regenerated, reordered, nor rewritten.

### Pilot validation metrics

- Interpolated EER: `0.0646251536255633` (6.46251536255633%)
- Interpolated non-empirical threshold: `0.1744520664215088`
- Interpolated FAR / FRR:
  `0.0646251536255633 / 0.0646251536255633`
- Executable empirical threshold: `0.1744520664215088`
- Empirical FAR / FRR / gap / average error:
  `0.0646251536255633 / 0.0646251536255633 / 0.0 /
  0.0646251536255633`
- Same-speaker mean/std/median:
  `0.42481032643701927 / 0.15599993927001696 / 0.4358007609844208`
- Different-speaker mean/std/median:
  `0.008027285437529052 / 0.10356156850219239 /
  0.0005986420437693596`
- Score range:
  `[-0.3212318420410156, 0.9287939071655273]`

## Baseline comparison

| Metric | Pretrained baseline | Epoch 0 pilot | Pilot minus baseline |
| --- | ---: | ---: | ---: |
| Interpolated EER | 0.11665301106104056 | 0.0646251536255633 | -0.05202785743547726 |
| EER percentage | 11.665301106104057% | 6.46251536255633% | -5.202785743547726 pp |
| Empirical threshold | 0.31165990233421326 | 0.1744520664215088 | -0.13720783591270446 |
| Empirical FAR | 0.11665301106104056 | 0.0646251536255633 | -0.05202785743547726 |
| Empirical FRR | 0.11665301106104056 | 0.0646251536255633 | -0.05202785743547726 |
| Same-score mean | 0.4905432510233207 | 0.42481032643701927 | -0.06573292458630142 |
| Same-score std | 0.1512383760047422 | 0.15599993927001696 | 0.004761563265274771 |
| Same-score median | 0.5050629079341888 | 0.4358007609844208 | -0.06926214694976807 |
| Different-score mean | 0.16336830629613902 | 0.008027285437529052 | -0.15534102085860997 |
| Different-score std | 0.11727968425816838 | 0.10356156850219239 | -0.013718115755975993 |
| Different-score median | 0.15320874005556107 | 0.0005986420437693596 | -0.1526100980117917 |

Baseline score range:
`[-0.20446446537971497, 0.935106635093689]`.

Pilot score range:
`[-0.3212318420410156, 0.9287939071655273]`.

Absolute EER difference was `0.05202785743547726`; relative EER change was
`-0.44600526777875327`, a 44.60052677787533% relative reduction. This
comparison uses a deterministic numerical tolerance of `1e-12`.

## Model-quality and best-checkpoint decisions

**Model-quality result: IMPROVED.** The pilot interpolated EER is lower than
the official historical pretrained baseline beyond the `1e-12` tolerance.

Among trained pilot checkpoints, `epoch_000.pt` was the only successfully
validated checkpoint and therefore the internal best under lowest
interpolated EER, then lowest empirical average error, then earlier-checkpoint
tie breaking. `best.pt` was created only after validation and its model state
matches `epoch_000.pt`.

The pretrained baseline remains a historical comparison reference; it was not
managed or overwritten as a pilot checkpoint. The observed improvement did
not trigger additional training.

## CUDA memory

- Training peak allocated / reserved:
  `775443456 / 874512384` bytes
- Validation peak allocated / reserved:
  `1340804096 / 1725956096` bytes
- GPU total memory: `4294443008` bytes
- No CUDA OOM occurred

## Protected artifact status

Pre/post SHA-256 comparison covered 158 explicitly approved files: six core
files, the exact 118 train shards selected by epoch 0, and all 34 validation
shards loaded. There were zero mismatches.

- Aggregate before/after set SHA-256:
  `ccf48065c43930df185087e5e18512b9f9b68de39af80bea11e9200e33ab0d2d`
- Cache config:
  `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`
- Train feature index:
  `5bd1999f92623084cba2558f19337f76cf4eb78878b63211366768742c6c9a99`
- Validation feature index:
  `7346073ed9b47354c5f85889222ad9a538d2c4ee14e0b7d85001e3c0d61608bc`
- Validation manifest:
  `9f553fa55b50ee071120bcbb2ce3c2caf9de4799cfb615eb732bbefd0d73d38b`
- Validation trial config:
  `a6457bc571965855633f1d277a43427507d708afe779b32ab4e0ec2c58d2eb9a`
- Validation trial CSV:
  `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`

The complete per-file before/after hash map is retained in the ignored
`pilot_runtime.json`. No broad split glob and no final-test path participated
in hashing. The seven unselected train shards statted during the original
sampler preflight were not hashed because they were outside the selected
allowlist; no valid pre-task content hash exists for them. This missing
allowlisted pre/post protection is part of the strict scope-compliance failure.

## Commands and tests

```text
.venv-cuda\Scripts\python.exe -m py_compile src\cached_fbank_samplers.py src\ecapa_one_epoch_pilot.py scripts\run_ecapa_aam_one_epoch_pilot.py tests\test_ecapa_one_epoch_pilot.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_ecapa_one_epoch_pilot -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\run_ecapa_aam_one_epoch_pilot.py --device cuda:0 1> outputs\ecapa_aam_one_epoch_pilot_v1\console.stdout.log 2> outputs\ecapa_aam_one_epoch_pilot_v1\console.stderr.log
git diff --check
git diff -- src\cached_fbank_samplers.py src\ecapa_one_epoch_pilot.py scripts\run_ecapa_aam_one_epoch_pilot.py tests\test_ecapa_one_epoch_pilot.py reports\ecapa_aam_one_epoch_pilot_v1.md reports\ecapa_aam_one_epoch_pilot_v1.json reports\CODEX_WORKLOG.md
git status --short
```

The CUDA invocation's exact stdout/stderr redirection is shown above.

- `py_compile`: PASS for all four new or modified Python files
- Focused pilot suite: 20 passed
- Complete repository suite: 100 passed
- Post-run artifact audit: PASS for all three checkpoints, 20 exact log
  records, 1,000 losses, 8,504 embeddings, 19,528 scores, immutable trial
  identity, and strict runtime/metric schemas
- `git diff --check`: PASS

Normal tests use synthetic fixtures and require neither real CUDA nor real
cache/validation data.

## Git status

Final `git status --short`:

```text
 M reports/CODEX_WORKLOG.md
 M src/cached_fbank_samplers.py
?? reports/ecapa_aam_one_epoch_pilot_v1.json
?? reports/ecapa_aam_one_epoch_pilot_v1.md
?? scripts/run_ecapa_aam_one_epoch_pilot.py
?? src/ecapa_one_epoch_pilot.py
?? tests/test_ecapa_one_epoch_pilot.py
```

The pilot runtime directory is correctly ignored. No commit or push was
performed.

## Limitations

This is a single deterministic epoch and one fixed-validation observation,
not a statistical study or a full fine-tuning result. After execution, a
code-hardening edit added explicit 488-speaker and train/validation
speaker-disjoint preflight assertions. An independent read-only post-run audit
verified those properties for the exact executed inputs; no training behavior
changed, and the CUDA epoch was not repeated in order to preserve the
one-epoch-only scope.

Strict scope compliance and overall PASS are false because the executed
version statted seven unselected train-shard paths. The final code prevents a
future repetition, but that cannot retroactively repair the executed run.

The untouched final test split has not been evaluated, so this report makes no
final-test or generalization claim.

## Explicitly deferred

- Epoch 1 and all later epochs
- Full multi-epoch fine-tuning
- Hyperparameter tuning
- Augmentation
- Scheduler, warmup, gradient clipping, and early stopping
- Every final-test operation
- Cache, manifest, split, and validation-trial regeneration
- Commit and push
