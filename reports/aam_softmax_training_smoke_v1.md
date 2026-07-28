# AAM-Softmax training pipeline and CUDA smoke test v1

## Final result

**FAIL (strict scope compliance).**

The implemented training pipeline, main microbatch-2 CUDA run, atomic checkpoint
roundtrip, resumed optimizer step, focused tests, complete test suite, and
microbatch-4 capacity probe all passed. However, a mandatory-preflight recursive
file listing enumerated final-test cache filenames and file sizes. It did not
open a final-test index, manifest, shard tensor, or data content, and no training
code accessed final-test inputs, but the task prohibited any final-test artifact
access. Therefore this report does not label the overall task PASS.

## Exact scope

- Implemented only AAM-Softmax training helpers, logical-to-physical microbatch
  handling, BatchNorm policy enforcement, optimizer construction, checkpoint v1,
  the explicit CUDA smoke entry point, and focused deterministic tests.
- Used the stable `.venv-cuda` environment and the pretrained
  `speechbrain/spkrec-ecapa-voxceleb` encoder.
- Performed two optimizer steps for the main run and one optimizer step in a
  separate fresh-process capacity probe.
- Did not run an epoch, augmentation, validation extraction/scoring,
  thresholding, EER, checkpoint selection, or full fine-tuning.
- Did not commit or push.

## Files inspected

- `AGENTS.md`
- `reports/CODEX_WORKLOG.md`
- `.gitignore`
- `src/cached_fbank_dataset.py`
- `src/cached_fbank_samplers.py`
- `src/speechbrain_frontend.py`
- `scripts/smoke_test_cached_ecapa.py`
- `scripts/precompute_speechbrain_fbank.py`
- Existing Dataset, sampler, frontend, script, test, report, output, and git
  conventions
- `outputs/fbank_cache_v1/fbank_cache_config_v1.json`
- `outputs/fbank_cache_v1/train_feature_index_v1.csv`
- `manifests/portable/train_manifest_v1.csv`
- Git status and diff

The recursive structure listing also enumerated final-test cache filenames and
sizes. No final-test file content was opened or loaded.

## Files created

- `src/aam_training.py`
- `scripts/smoke_test_aam_training_cuda.py`
- `tests/test_aam_training.py`
- `reports/aam_softmax_training_smoke_v1_main.json`
- `reports/aam_softmax_training_smoke_v1_microbatch4.json`
- `reports/aam_softmax_training_smoke_v1.md`
- `outputs/aam_softmax_training_smoke_v1/checkpoint_v1.pt` (ignored)

## Files modified

- `src/cached_fbank_dataset.py`
  - Added aligned `dataset_index` values to samples and collated batches.
  - Added an optional dedicated DataLoader generator for deterministic resume
    without consuming the model RNG stream.
- `reports/CODEX_WORKLOG.md` (append only)

All unrelated pre-existing dirty and untracked work was preserved.

## Train cache and sampler

- Dataset: cached train Fbank only, `validate_finite=False`, LRU shard limit 8,
  `num_workers=0`.
- Every smoke-test tensor was explicitly checked for finiteness.
- Sampler: `HybridShardAwareSpeakerBatchSampler`.
- Seed/epoch: `20260727` / `0`.
- P/K/window: 16 / 2 / 8.
- Logical batch: 32, with 16 distinct speakers and exactly two samples per
  speaker.
- Both main logical batches contained 32 unique Dataset indexes.
- Round-robin order was label-deterministic: first utterance for all 16 speakers,
  then the second utterance for all 16.
- Every microbatch-2 physical batch contained two speakers; every probe
  microbatch-4 physical batch contained four speakers.
- Feature, label, Dataset index, speaker ID, relative path, split, and provenance
  metadata stayed aligned.

## Encoder path and forbidden calls

The only runtime encoder path was:

```text
cached Fbank [B, 301, 80]
-> encoder.mods.mean_var_norm [B, 301, 80]
-> encoder.mods.embedding_model [B, 1, 192]
-> squeeze(1) [B, 192]
```

The SpeechBrain checkpoint was loaded on CPU. The runtime retained and moved only
`mean_var_norm` and `embedding_model` to CUDA; `compute_features` and the
pretrained classifier were not retained. Static checks found no
`mods.compute_features`, `mods.classifier`, `encode_batch`, `src.fbank`,
`torchaudio`, or WAV call in the new training helper/script. Runtime results
record zero compute-feature and pretrained-classifier calls.

## AAM-Softmax

- Weight: `[488, 192]`, no bias, deterministic Xavier initialization under seed
  `20260727`.
- Margin: `m=0.2` radians.
- Scale: `s=30.0`.
- Embeddings and weights are converted to float32 and L2-normalized.
- Cosines are clamped to `[-1 + 1e-7, 1 - 1e-7]`.
- `phi = cos(theta) cos(m) - sin(theta) sin(m)`.
- Monotonic ArcFace handling uses `phi` when
  `cos(theta) > cos(pi - m)` and otherwise uses
  `cos(theta) - sin(pi - m) * m`.
- Only the target logit is replaced; all logits are scaled by `s`.
- Cross-entropy is outside the classifier, uses `reduction="sum"`, and each
  microbatch sum is divided by the complete logical size 32.
- Tests cover shapes, finite forward/backward, nonzero embedding and AAM
  gradients, target-only replacement, `m=0`, cosine-boundary backward, invalid
  labels, non-finite inputs, and deterministic initialization.

## BatchNorm and trainability policy

- Every `embedding_model` parameter was trainable.
- `embedding_model.train()` was applied, followed by eval mode for only its 31
  BatchNorm modules.
- BatchNorm affine weight/bias remained trainable.
- Non-BatchNorm modules remained in training mode.
- `mean_var_norm` was preserved, placed in eval mode, excluded from the
  optimizer, and called with full-length float32 tensors.
- Ninety-three BatchNorm running-statistics buffers were compared after each
  optimizer step and remained exactly equal.
- The policy was reapplied after construction and checkpoint loading.

## AMP and optimizer

- ECAPA forward used CUDA float16 autocast.
- AAM angular math, logits, and loss used float32.
- `torch.cuda.amp.GradScaler` remained enabled with an explicit initial scale
  of 128.
- Two explicit AdamW groups:
  - `embedding_model`: learning rate `1e-5`
  - AAM classifier: learning rate `1e-3`
  - shared weight decay: `1e-4`
- Parameter groups were complete, disjoint, and excluded `mean_var_norm` and the
  pretrained classifier.
- No scheduler and no gradient clipping were added.

Three fail-closed pre-step diagnostic attempts preceded the successful main run.
The first found non-finite ECAPA gradients. The AAM clamp was corrected from the
closed interval to an open float32-safe interval and a boundary-gradient test was
added. A further diagnostic identified NaN/Inf in
`blocks.0.norm.norm.weight` after unscale with GradScaler's default initial scale
65,536. The final run retained float16 autocast and enabled GradScaler but used
the recorded initial scale 128. No failed attempt reached `optimizer.step`, saved
a checkpoint, or retried in full float32.

## Main microbatch-2 result

**Technical result: PASS.**

- Physical microbatch: 2.
- Accumulation: 16.
- Logical batches consumed: 2.
- Optimizer steps: 2.
- No CUDA OOM.
- Main peak allocated VRAM: 661,258,752 bytes.
- Main peak reserved VRAM: 861,929,472 bytes.

### Step 1

- Logical loss: `14.102674305438995`.
- Duration: `1.9493115000004764` seconds.
- ECAPA gradient norm: `59.820380658764286`.
- AAM gradient norm: `8.563820396025983`.
- ECAPA non-BatchNorm maximum / aggregate absolute delta:
  `1.0013580322265625e-05` / `202.6730268294923`.
- AAM maximum / aggregate absolute delta:
  `0.0010000169277191162` / `93.6861801147461`.
- Peak allocated / reserved:
  `497169920` / `683671552` bytes.
- GradScaler scale before / after: `128.0` / `128.0`.
- BatchNorm running buffers exactly unchanged: true (93 compared).

### Step 2 after fresh-object resume

- Logical loss: `14.168766677379608`.
- Duration: `1.1059552999995503` seconds.
- ECAPA gradient norm: `60.42722928136366`.
- AAM gradient norm: `8.381696159213307`.
- ECAPA non-BatchNorm maximum / aggregate absolute delta:
  `1.0013580322265625e-05` / `131.15723157610046`.
- AAM maximum / aggregate absolute delta:
  `0.0010013654828071594` / `67.10301208496094`.
- Peak allocated / reserved:
  `661258752` / `861929472` bytes.
- GradScaler scale before / after: `128.0` / `128.0`.
- BatchNorm running buffers exactly unchanged: true (93 compared).
- Global optimizer step after resume: 2.

## Checkpoint roundtrip

- Path: `outputs/aam_softmax_training_smoke_v1/checkpoint_v1.pt`.
- Size: 250,699,903 bytes.
- Atomic temporary-file plus `os.replace`: PASS.
- Schema validation: PASS.
- Whole saved/loaded payload exact: PASS.
- Fresh `embedding_model`, `mean_var_norm`, AAM, AdamW, and GradScaler objects
  were constructed before loading.
- Every embedding state tensor: exact.
- Every mean-normalization state tensor: exact.
- Every AAM state tensor: exact.
- Optimizer state: exact.
- GradScaler state: exact.
- Configuration and counters: exact.
- Python, NumPy, torch CPU, and all available CUDA RNG states were stored and
  restored.
- The checkpoint contains repository-relative paths only and does not contain
  cache shard tensors, validation artifacts, trials, WAV data, final-test data,
  or pretrained classifier state.

The saved next position was 1. A fresh epoch-0 hybrid sampler regenerated and
skipped the consumed position. Dataset indexes, speaker IDs, and relative paths
for the resumed batch exactly matched both the checkpoint identity and the
second batch expected before step 1.

## Microbatch-4 capacity probe

**PASS (separate fresh process).**

- Physical microbatch / accumulation: 4 / 8.
- One logical batch and one complete optimizer step.
- All eight physical microbatches had four distinct speakers.
- Logical loss: `14.10267436504364`.
- Duration: `1.4691242999997485` seconds.
- ECAPA / AAM gradient norm:
  `59.82091489954806` / `8.563820396025983`.
- ECAPA maximum / aggregate absolute delta:
  `1.0013580322265625e-05` / `202.67372208437882`.
- AAM maximum / aggregate absolute delta:
  `0.0010000169277191162` / `93.68618774414062`.
- Peak allocated / reserved:
  `603883520` / `723517440` bytes.
- BatchNorm running buffers exactly unchanged: true.
- No checkpoint or resume was performed for the probe.

## Tests and commands

```text
.venv-cuda\Scripts\python.exe -m py_compile src\cached_fbank_dataset.py src\aam_training.py scripts\smoke_test_aam_training_cuda.py tests\test_aam_training.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_aam_training -v
.venv-cuda\Scripts\python.exe -m unittest tests.test_cached_fbank_dataset tests.test_cached_fbank_samplers tests.test_aam_training -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\smoke_test_aam_training_cuda.py --mode main --cache-dir outputs\fbank_cache_v1 --device cuda:0
.venv-cuda\Scripts\python.exe scripts\smoke_test_aam_training_cuda.py --mode probe --cache-dir outputs\fbank_cache_v1 --device cuda:0
git diff --check
git diff
git status --short
```

- Final focused new suite: 15 passed.
- Final complete repository suite: 74 passed.
- Compilation: PASS.
- The main command was launched four times: three fail-closed pre-step
  diagnostics described above and one complete PASS.
- The probe command was launched once after the main PASS.

## Protected artifacts

Train identity hashes were recorded before optimizer work and matched the final
hashes:

- `manifests/portable/train_manifest_v1.csv`:
  `1b8837d97901f10218cf2fe193e29a56e4695138bbd2037cbd9818e437a3b3fe`
- `outputs/fbank_cache_v1/fbank_cache_config_v1.json`:
  `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`
- `outputs/fbank_cache_v1/train_feature_index_v1.csv`:
  `5bd1999f92623084cba2558f19337f76cf4eb78878b63211366768742c6c9a99`

No source audio, manifest, split, cache tensor, environment, or pretrained weight
was modified. No WAV file was read. No validation or final-test content was
opened by the implementation or CUDA commands. The preflight metadata
enumeration described in the final-result section is the strict-scope failure.

## Limitations

- This is a two-step smoke test and one-step capacity probe, not an epoch or a
  model-quality evaluation.
- No validation metric, checkpoint selection, threshold, EER, or final-test
  result was produced.
- The explicit GradScaler initial scale 128 was required for finite accumulated
  gradients on this RTX 3050 Laptop GPU configuration.

## Explicitly deferred

Full fine-tuning, epochs, augmentation, validation embedding extraction,
validation scoring, threshold selection, EER, checkpoint selection, scheduler
work, final-test evaluation, cache/manifest/split regeneration, distributed
training, commit, and push.
