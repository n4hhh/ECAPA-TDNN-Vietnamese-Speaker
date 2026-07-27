# Codex project worklog

## Project baseline

- Dataset root: external `VieSpeaker` dataset supplied at runtime. Source audio is immutable.
- Approved speaker split v1: train 31,998 utterances / 488 speakers; validation 8,504 / 100; test 9,198 / 100.
- Portable manifests: `manifests/portable/train_manifest_v1.csv`, `validation_manifest_v1.csv`, and `test_manifest_v1.csv`.
- Label mapping: `manifests/portable/speaker_to_label_v1.json`; train labels are `0..487`, validation/test labels are `-1`.
- Production model: `speechbrain/spkrec-ecapa-voxceleb`.
- Production path: waveform `[B, 48000]` -> `compute_features` `[B, 301, 80]` -> `mean_var_norm` `[B, 301, 80]` -> `embedding_model` -> embedding `[B, 1, 192]`.
- Fbank cache v1: `outputs/fbank_cache_v1/`; 49,700 raw pre-normalization float32 features shaped `[301, 80]` in 195 shards of at most 256 utterances. Features are not transposed.
- Important repository rules: preserve audio/manifests/cache/model weights; use `.venv-cuda` for CUDA work; do not use the VoxCeleb classifier; do not continue stages, train, commit, or push without explicit instruction.
- Completed outputs: candidate split under `splits/`; portable manifests under `manifests/portable/`; cache producer and indexes under `scripts/` and `outputs/fbank_cache_v1/`; cache summary at `reports/fbank_cache_v1_summary.md`.

## 2026-07-27 11:39:34 +07:00 - Cached Fbank Dataset and ECAPA inference smoke test

### Goal

Implement a production-ready lazy Dataset/DataLoader for Fbank cache v1 and verify a small cached batch through SpeechBrain `mean_var_norm` and `embedding_model` without training.

### Repository state inspected

- Root `AGENTS.md`, repository status, source/test/script/report conventions.
- `scripts/precompute_speechbrain_fbank.py` and `reports/fbank_cache_v1_summary.md`.
- All cache config/index schemas and one real shard structure.
- Existing `src/speechbrain_frontend.py` and cache tests.

### Files created

- `src/cached_fbank_dataset.py`
- `scripts/smoke_test_cached_ecapa.py`
- `tests/test_cached_fbank_dataset.py`
- `reports/CODEX_WORKLOG.md`
- `reports/cached_fbank_dataset_v1_summary.md`

### Files modified

- None.

### Important implementation decisions

- Index/config files are authoritative; all rows are validated at Dataset construction.
- Shards load lazily on CPU into a deterministic bounded LRU cache (default two shards).
- Shard cache state is cleared when pickled for Windows DataLoader compatibility.
- Collation stacks fixed-size features without padding, augmentation, waveform loading, or stochastic transforms.
- The required smoke test uses `num_workers=0`, direct pretrained normalization/embedding modules, `eval()`, and `torch.inference_mode()`.
- All encoder parameters are cloned before inference and compared afterward to prove no update occurred.

### Commands executed

```text
.venv-cuda\Scripts\python.exe -m py_compile src/cached_fbank_dataset.py scripts/smoke_test_cached_ecapa.py tests/test_cached_fbank_dataset.py
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts/smoke_test_cached_ecapa.py --cache-dir outputs/fbank_cache_v1 --batch-size 4 --device cuda:0 --max-cached-shards 2
```

The first test run found one test-fixture expectation error; the fixture was corrected and the full suite reran successfully.

### Tests and observed shapes

- Unit/repository tests: 19 passed.
- Production index lengths: train 31,998; validation 8,504; test 9,198.
- Input cached Fbank: `[4, 301, 80]`, float32, `cuda:0`.
- Labels: `[4]`, int64.
- Normalized features: `[4, 301, 80]`, float32, finite.
- Embeddings: `[4, 1, 192]`, float32, finite.
- Train label range: `0..487`; validation/test labels: all `-1`.
- Parameters frozen and unchanged: true.

### Result

PASS

### Known limitations

- The required production smoke test covers deterministic sequential loading with `num_workers=0`; multi-worker throughput was not benchmarked.
- The Dataset is intentionally fixed to cache schema v1 and `[301, 80]` float32 features.

### Explicitly deferred

AAM-Softmax, training/fine-tuning, optimizers, backward propagation, checkpoints, verification trials, scoring, thresholds, EER, test evaluation, augmentation, and cache/manifest regeneration.

### Suggested next step (suggestion only)

After review and explicit approval, define the training-stage sampling and frozen/unfrozen encoder policy before implementing any training code.

## 2026-07-27 12:28:15 +07:00 - Train Cache Access Analysis and DataLoader Benchmark v1

### Goal

Measure train shard/speaker composition, diagnostic batch diversity, cache access cost, finite-check overhead, LRU behavior, and Windows DataLoader worker behavior without implementing a production sampler or running training.

### Files inspected

- `AGENTS.md`
- `reports/CODEX_WORKLOG.md`
- `src/cached_fbank_dataset.py`
- `scripts/smoke_test_cached_ecapa.py`
- `tests/test_cached_fbank_dataset.py`
- `reports/cached_fbank_dataset_v1_summary.md`
- `reports/fbank_cache_v1_summary.md`
- `scripts/precompute_speechbrain_fbank.py`
- `outputs/fbank_cache_v1/fbank_cache_config_v1.json`
- `outputs/fbank_cache_v1/train_feature_index_v1.csv`
- Real train shard directory/file metadata
- Repository file listing and git status

### Files created

- `scripts/benchmark_cached_fbank_access.py`
- `tests/test_cached_fbank_benchmark.py`
- `reports/train_cache_access_analysis_v1.md`
- `reports/train_cache_benchmark_v1.json`
- `reports/train_cache_shard_stats_v1.csv`
- `reports/train_cache_speaker_stats_v1.csv`

### Files modified

- `src/cached_fbank_dataset.py`
- `tests/test_cached_fbank_dataset.py`
- `reports/CODEX_WORKLOG.md` (append only)

### Commands run

```text
.venv-cuda\Scripts\python.exe -m py_compile src\cached_fbank_dataset.py scripts\benchmark_cached_fbank_access.py tests\test_cached_fbank_dataset.py tests\test_cached_fbank_benchmark.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_cached_fbank_dataset tests.test_cached_fbank_benchmark -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\benchmark_cached_fbank_access.py --cache-dir outputs\fbank_cache_v1 --batch-size 32 --warmup-batches 2 --measured-samples 512 --seed 20260727 --report-dir reports
```

The initial benchmark command was given a one-second shell timeout and was terminated before measurement; it was rerun normally. After clarifying total versus measured shard-load fields, the complete benchmark suite was rerun and the final artifacts contain the second run.

### Exact train counts detected

- Rows: 31,998
- Speakers: 488
- Labels: contiguous `0..487`
- Shards: 125
- All within-shard feature indexes valid and unique: true

### Shard-distribution findings

- Unique speakers/shard min/mean/Q1/median/Q3/max: 1 / 4.888 / 2 / 4 / 6 / 22
- Dominant share min/mean/Q1/median/Q3/max: 12.89% / 59.95% / 39.06% / 56.25% / 78.52% / 100%
- Dominance >=25%: 119/125; >=50%: 72/125; >=75%: 38/125
- Single-speaker shards: 22/125

### Speaker-distribution findings

- Utterances/speaker min/mean/Q1/median/Q3/max: 5 / 65.570 / 10 / 29 / 75.25 / 1,595
- Fewer than 10/20/50 utterances: 119 / 204 / 316 speakers
- More than 100/500 utterances: 84 / 6 speakers
- Index runs: 488; transitions: 487; same-speaker neighbor pairs: 98.478%

### Batch-diversity findings

At batch size 32:

- Sequential unique speakers min/mean/median/max: 1 / 1.470 / 1 / 6; >=50% one speaker in 95.40% of batches.
- Sample-random: 21 / 27.905 / 28 / 32; >=50% in 0%; >=25% in 0.10%.
- Diagnostic shard-local: 1 / 4.322 / 4 / 17; >=50% in 62.36%; >=25% in 96.50%.

Batch-size-16 results and full represented-speaker distributions are recorded in `reports/train_cache_benchmark_v1.json`.

### Benchmark configurations and exact results

All used batch size 32, two warm-up batches, 512 measured samples, and seed 20260727.

| Access | LRU | Workers | Finite | First batch s | Elapsed s | Batch/s | Sample/s | Measured loads |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Sequential | 2 | 0 | true | 0.0150 | 0.1142 | 140.15 | 4,484.73 | 2 |
| Sample-random | 2 | 0 | true | 0.2512 | 3.0295 | 5.28 | 169.00 | 502 |
| Sample-random | 2 | 0 | false | 0.1686 | 2.5359 | 6.31 | 201.90 | 502 |
| Sample-random | 8 | 0 | false | 0.1658 | 2.4332 | 6.58 | 210.42 | 484 |
| Sample-random | 8 | 1 | false | 2.2334 | 2.4616 | 6.50 | 208.00 | unavailable |
| Sample-random | 8 | 2 | false | 3.8157 | 1.7043 | 9.39 | 300.41 | unavailable |
| Shard-local | 2 | 0 | false | 0.0062 | 0.0251 | 636.86 | 20,379.57 | 2 |

All shapes and metadata were valid; all runs completed successfully. Both Windows worker counts worked without code changes.

### Tests run

- Focused Dataset/benchmark helper tests: 15 passed.
- Complete unit-test suite: 26 passed.
- `py_compile`: passed for every new or modified Python file.

### Overall result

**PASS**

### Known benchmark limitations

OS file caching and run order affect disk timing; 512 samples are not a full epoch; worker startup and caching differ in long-running training; worker shard loads were not aggregated; memory metrics were unavailable without installing a dependency; throughput does not determine training quality.

### Recommended next step (suggestion only)

In a later explicitly approved task, evaluate a hybrid speaker-balanced and shard-aware strategy against standard shuffle. The evidence shows standard random ordering provides high diversity but nearly one shard load/sample, while pure shard-local ordering is fast but speaker-concentrated. No sampler implementation was performed.

### Explicitly deferred

Production sampler implementation, training, loss/optimizer/backpropagation, checkpoints, verification, scoring, thresholds, EER, final test evaluation, audio work, cache/manifests/split regeneration, commit, and push.
