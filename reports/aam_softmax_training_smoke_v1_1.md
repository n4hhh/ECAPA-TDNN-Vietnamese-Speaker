# AAM-Softmax training smoke test scope-compliance rerun v1.1

## Result

**PASS.**

This is a narrow scope-compliance remediation and clean rerun of v1. The
historical v1 files remain unchanged: v1's technical pipeline result was PASS
and its strict scope-compliance result was FAIL. v1.1 does not rewrite, erase,
or relabel that historical failure.

For v1.1, all technical checks passed, the train-only guards passed, no recursive
repository/cache/output listing was used, no real validation artifact was
opened, stat-ed, hashed, or loaded, and no final-test artifact or metadata was
enumerated, stat-ed, hashed, opened, or loaded. The explicitly required
`git status --short` reported existing worktree pathnames only; it did not inspect
the contents or filesystem metadata of those artifacts.

## Scope

- Reviewed the existing v1 implementation without redesigning it.
- Added only train-only fail-closed request and metadata guards plus focused
  regression tests.
- Repeated the two-step microbatch-2 checkpoint/resume smoke in one fresh
  process.
- Repeated the microbatch-4 capacity probe in a separate fresh process after the
  main PASS.
- Did not run an epoch, full fine-tuning, augmentation, validation inference,
  verification scoring, threshold selection, EER, checkpoint selection,
  scheduler work, waveform access, or final-test evaluation.
- Did not commit or push.

## Allowlisted files inspected

The following files were opened directly by explicit path:

- `AGENTS.md`
- `reports/CODEX_WORKLOG.md`
- `reports/aam_softmax_training_smoke_v1.md`
- `reports/aam_softmax_training_smoke_v1.json`
- `src/aam_training.py`
- `src/cached_fbank_dataset.py`
- `src/cached_fbank_samplers.py`
- `scripts/smoke_test_aam_training_cuda.py`
- `tests/test_aam_training.py`
- `.gitignore`

Only targeted git status/diff commands were used. No repository tree, output
tree, cache tree, or unrelated directory was recursively listed.

## Historical v1 preservation

- `reports/aam_softmax_training_smoke_v1.md`: not modified.
- `reports/aam_softmax_training_smoke_v1.json`: not modified.
- `reports/aam_softmax_training_smoke_v1_main.json`: not overwritten.
- `reports/aam_softmax_training_smoke_v1_microbatch4.json`: not overwritten.
- `outputs/aam_softmax_training_smoke_v1/checkpoint_v1.pt`: not overwritten.

All v1.1 runtime and checkpoint paths were directly checked for absence before
the run.

## Dataset diff review

The existing `src/cached_fbank_dataset.py` diff is retained.

### Aligned Dataset index

Adding `dataset_index` to each Dataset sample and collated batch is necessary.
It preserves exact Dataset identity while the logical batch is reordered and
allows the resumed second batch to be compared against the originally expected
sampler identity.

- Public output schema: one additive `dataset_index` key.
- Existing keys and values: unchanged.
- Feature/label shapes and dtypes: unchanged.
- Row ordering and index mapping: unchanged.
- Cache schema and feature loading: unchanged.
- Sampler behavior: unchanged.
- Existing callers: compatible because all prior keys remain.
- Validation/test Dataset mechanics: the same additive key would be present if
  those splits were constructed; no such real Dataset was constructed in v1.1.

### Dedicated DataLoader generator

The optional generator argument is necessary for deterministic checkpoint
resume. DataLoader iterator construction uses the dedicated generator instead
of advancing the restored global torch RNG stream.

- Dataset rows are not mutated.
- Batch-sampler semantics and index order are not changed.
- Existing callers that omit the optional argument retain prior behavior.

A wrapper/manual-collation alternative would add more task-specific machinery;
the retained additive changes are the smaller implementation.

## Train-only fail-closed guards

Before Dataset construction, v1.1 requires:

- split exactly `train`;
- index filename exactly `train_feature_index_v1.csv`;
- cache root exactly `outputs/fbank_cache_v1`;
- index path exactly
  `outputs/fbank_cache_v1/train_feature_index_v1.csv`.

After index metadata construction but before tensor iteration, it requires:

- a non-empty tuple of train rows;
- every `final_split` exactly `train`;
- every label an integer in `0..487`, never `-1`;
- every shard path exactly `train/shard_NNNNN.pt`;
- every resolved shard path contained under
  `outputs/fbank_cache_v1/train/`.

The guard rejects non-train/ambiguous split values, validation/test index
filenames, ambiguous or outside cache roots, non-train rows, labels `-1` or
outside `0..487`, and non-train shard paths. The logical-batch validator also
rechecks all 32 `final_split` values and labels.

All 31,998 real train metadata rows passed. The existing sampler then
existence-checked row-referenced train shard paths directly; it did not list a
directory. Only the exact sampled shards below were loaded as tensors.

## Train-only artifacts accessed

No portable train manifest or speaker-label JSON was required or accessed.

The following train inputs were opened:

- `outputs/fbank_cache_v1/fbank_cache_config_v1.json`
- `outputs/fbank_cache_v1/train_feature_index_v1.csv`
- The exact 13 train shards selected by the two main batches:
  - `outputs/fbank_cache_v1/train/shard_00002.pt`
  - `outputs/fbank_cache_v1/train/shard_00007.pt`
  - `outputs/fbank_cache_v1/train/shard_00038.pt`
  - `outputs/fbank_cache_v1/train/shard_00039.pt`
  - `outputs/fbank_cache_v1/train/shard_00047.pt`
  - `outputs/fbank_cache_v1/train/shard_00050.pt`
  - `outputs/fbank_cache_v1/train/shard_00065.pt`
  - `outputs/fbank_cache_v1/train/shard_00078.pt`
  - `outputs/fbank_cache_v1/train/shard_00080.pt`
  - `outputs/fbank_cache_v1/train/shard_00085.pt`
  - `outputs/fbank_cache_v1/train/shard_00095.pt`
  - `outputs/fbank_cache_v1/train/shard_00110.pt`
  - `outputs/fbank_cache_v1/train/shard_00119.pt`

The probe loaded this seven-shard subset:

- `train/shard_00002.pt`
- `train/shard_00007.pt`
- `train/shard_00038.pt`
- `train/shard_00047.pt`
- `train/shard_00065.pt`
- `train/shard_00085.pt`
- `train/shard_00110.pt`

No glob spanning splits was used.

## Technical policy retained

- Cached Fbank `[301,80]`, no transpose or recomputation.
- Hybrid sampler P16K2, active window 8, seed 20260727, epoch 0.
- Logical batch 32 with deterministic round-robin microbatch ordering.
- AAM `[488,192]`, no bias, `m=0.2`, `s=30`, float32 angular math.
- Full `embedding_model` trainable.
- Only 31 BatchNorm modules in eval mode; affine parameters trainable.
- Ninety-three BatchNorm running buffers compared bit-exactly after each step.
- `mean_var_norm` used with full-length tensors but not optimized.
- Float16 autocast for ECAPA; enabled GradScaler with initial scale 128.
- AdamW: ECAPA `1e-5`, AAM `1e-3`, weight decay `1e-4`.
- `compute_features` and pretrained classifier were not retained or called.
- No scheduler and no gradient clipping.

## Main microbatch-2 result

**PASS.**

- Two logical P16K2 batches.
- Each batch: 32 unique Dataset indexes, 16 speakers, two samples/speaker,
  train rows only, labels in `0..487`, finite CPU float32 `[32,301,80]`.
- Each of 16 physical batches per step: two distinct speakers and input
  `[2,301,80]`.
- Normalized / raw embedding / squeezed embedding / logits:
  `[2,301,80]` / `[2,1,192]` / `[2,192]` / `[2,488]`.
- Two successful optimizer steps; no OOM.
- Main peak allocated/reserved:
  `661258752` / `861929472` bytes.

### Step 1

- Loss: `14.102674305438995`.
- Duration: `2.0199535000010655` seconds.
- ECAPA/AAM gradient norms:
  `59.820380658764286` / `8.563820396025983`.
- ECAPA maximum/aggregate absolute delta:
  `1.0013580322265625e-05` / `202.6730268294923`.
- AAM maximum/aggregate absolute delta:
  `0.0010000169277191162` / `93.6861801147461`.
- Peak allocated/reserved:
  `497169920` / `683671552` bytes.
- BatchNorm running buffers unchanged: true.

### Step 2 after resume

- Loss: `14.168766677379608`.
- Duration: `1.1580651999993279` seconds.
- ECAPA/AAM gradient norms:
  `60.42722928136366` / `8.381696159213307`.
- ECAPA maximum/aggregate absolute delta:
  `1.0013580322265625e-05` / `131.15723157610046`.
- AAM maximum/aggregate absolute delta:
  `0.0010013654828071594` / `67.10301208496094`.
- Peak allocated/reserved:
  `661258752` / `861929472` bytes.
- BatchNorm running buffers unchanged: true.
- Global optimizer step: 2.

## Checkpoint roundtrip and resume

**PASS.**

- New path:
  `outputs/aam_softmax_training_smoke_v1_1/checkpoint_v1.pt`.
- Size: 250,701,631 bytes.
- Atomic temporary save plus `os.replace`: PASS.
- Whole payload exact after save/load: true.
- Fresh SpeechBrain encoder, AAM, AdamW, and GradScaler objects constructed.
- Embedding, mean-normalization, AAM, optimizer, and GradScaler states exact.
- Configuration/counters exact; BatchNorm policy reapplied; RNG restored.
- Regenerated sampler position 1 matched the saved identity and the second batch
  expected before step 1 exactly.

## Microbatch-4 capacity probe

**PASS in a separate fresh process.**

- Eight microbatches and one complete optimizer step.
- Four distinct speakers in every physical microbatch.
- Loss: `14.10267436504364`.
- Duration: `1.1910272000004625` seconds.
- ECAPA/AAM gradient norms:
  `59.82091489954806` / `8.563820396025983`.
- ECAPA maximum/aggregate delta:
  `1.0013580322265625e-05` / `202.67372208437882`.
- AAM maximum/aggregate delta:
  `0.0010000169277191162` / `93.68618774414062`.
- Peak allocated/reserved:
  `603883520` / `723517440` bytes.
- BatchNorm running buffers unchanged: true.

## Protected train hashes

The cache config, train index, and exact 13 main-selected train shards were
hashed before CUDA work and again afterward. All 15 hashes matched; mismatches:
zero. Full values are recorded in
`reports/aam_softmax_training_smoke_v1_1.json`.

- Cache config:
  `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`
- Train index:
  `5bd1999f92623084cba2558f19337f76cf4eb78878b63211366768742c6c9a99`

No protected input was modified.

## Validation and final-test quarantine

- Real validation Dataset constructed: no.
- Validation cache/manifest/trials/embeddings/scores/threshold/EER file opened,
  stat-ed, or hashed: no.
- Final-test Dataset constructed: no.
- Final-test cache/manifest/output/file/directory enumerated, stat-ed, hashed,
  resolved, opened, or loaded: no.
- Broad or recursive repository/cache/output filesystem listing: no.
- Complete unittest discovery was the explicitly required test-suite operation;
  tests used temporary synthetic inputs and did not access real evaluation data.

## Tests and commands

```text
.venv-cuda\Scripts\python.exe -m py_compile src\aam_training.py src\cached_fbank_dataset.py src\cached_fbank_samplers.py scripts\smoke_test_aam_training_cuda.py tests\test_aam_training.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_aam_training -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\smoke_test_aam_training_cuda.py --mode main --cache-dir outputs\fbank_cache_v1 --device cuda:0 --checkpoint outputs\aam_softmax_training_smoke_v1_1\checkpoint_v1.pt --result-json reports\aam_softmax_training_smoke_v1_1_main_runtime.json
.venv-cuda\Scripts\python.exe scripts\smoke_test_aam_training_cuda.py --mode probe --cache-dir outputs\fbank_cache_v1 --device cuda:0 --result-json reports\aam_softmax_training_smoke_v1_1_probe_runtime.json --main-result reports\aam_softmax_training_smoke_v1_1_main_runtime.json
git diff --check
git diff -- src\aam_training.py src\cached_fbank_dataset.py src\cached_fbank_samplers.py scripts\smoke_test_aam_training_cuda.py tests\test_aam_training.py reports\CODEX_WORKLOG.md
git status --short
```

- `py_compile`: PASS.
- Focused AAM/remediation tests: 21 passed.
- Complete repository suite: 80 passed.
- All new guard tests use a temporary synthetic cache.

## Files created

- `reports/aam_softmax_training_smoke_v1_1.md`
- `reports/aam_softmax_training_smoke_v1_1.json`
- `reports/aam_softmax_training_smoke_v1_1_main_runtime.json`
- `reports/aam_softmax_training_smoke_v1_1_probe_runtime.json`
- Ignored `outputs/aam_softmax_training_smoke_v1_1/checkpoint_v1.pt`

## Files modified

- `src/aam_training.py`
- `scripts/smoke_test_aam_training_cuda.py`
- `tests/test_aam_training.py`
- `reports/CODEX_WORKLOG.md` (append only)

The earlier task's small `src/cached_fbank_dataset.py` diff was reviewed and
retained without further modification in v1.1.

## Git and limitations

- Pre-existing dirty/untracked sampler and validation work was preserved.
- The v1.1 checkpoint is ignored by the existing `outputs/` rule.
- No commit or push occurred.
- This remains a two-step smoke test plus a one-step capacity probe, not an
  epoch or a model-quality evaluation.

## Deferred

Full fine-tuning, epochs, augmentation, validation extraction/scoring,
thresholding, EER, checkpoint selection, scheduler/early stopping, waveform
work, final-test evaluation, cache/manifest/split regeneration, distributed
training, commit, and push.
