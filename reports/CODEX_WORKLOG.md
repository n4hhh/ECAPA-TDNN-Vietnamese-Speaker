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

## 2026-07-27 21:26:45 +07:00 - Training Batch Sampler Prototypes and Benchmark v1

### Goal

Implement and compare deterministic production-usable sample-random, global speaker-balanced P × K, and hybrid speaker-balanced/shard-aware training batch samplers without model inference or training.

### Files inspected

- `AGENTS.md`
- `reports/CODEX_WORKLOG.md`
- `src/cached_fbank_dataset.py`
- `scripts/benchmark_cached_fbank_access.py`
- `tests/test_cached_fbank_dataset.py`
- `tests/test_cached_fbank_benchmark.py`
- `reports/train_cache_access_analysis_v1.md`
- `reports/train_cache_benchmark_v1.json`
- `reports/train_cache_shard_stats_v1.csv`
- `reports/train_cache_speaker_stats_v1.csv`
- `reports/cached_fbank_dataset_v1_summary.md`
- `reports/fbank_cache_v1_summary.md`
- `outputs/fbank_cache_v1/fbank_cache_config_v1.json`
- `outputs/fbank_cache_v1/train_feature_index_v1.csv`
- Real train shard file names/existence under `outputs/fbank_cache_v1/train/`
- Repository file listing, git status, diff, and diff check

### Files created

- `src/cached_fbank_samplers.py`
- `scripts/benchmark_cached_fbank_samplers.py`
- `tests/test_cached_fbank_samplers.py`
- `tests/test_cached_fbank_sampler_benchmark.py`
- `reports/train_sampler_analysis_v1.md`
- `reports/train_sampler_benchmark_v1.json`
- `reports/train_sampler_batch_metrics_v1.csv`

### Files modified

- `src/cached_fbank_dataset.py`
- `reports/CODEX_WORKLOG.md` (append only)

### Sampler designs and epoch definition

- Deterministic sample-random baseline: batch size 32, complete single-pass shuffle, one 30-sample partial batch, `drop_last` configurable.
- Global speaker-balanced: P=16, K=2, 32 samples/batch, shuffled speaker cycles and shuffled per-speaker queues.
- Hybrid shard-aware: P=16, K=2, deterministic shuffled shard order, exposure-aware speaker selection, active windows 8, 16, and 32, active-first sample choice, exact structure with deterministic expansion.
- Seed: 20260727.
- P × K epoch: 1,000 full batches, `ceil(31,998 / 32)`, configurable explicitly.
- Balanced epochs repeat low-resource speakers and omit many high-resource-speaker utterances; no model-quality improvement is claimed.

### Train index invariants

- Rows: 31,998, all `final_split=train`.
- Speakers: 488.
- Labels: exactly contiguous `0..487`; no `-1`.
- Speaker ID -> label and label -> speaker ID are both one-to-one.
- Referenced shards: 125; all exist.
- Every row belongs to exactly one metadata-derived sampler group.

### Exact correctness and metadata metrics

- All candidates: same seed/epoch identical, different epoch different, Dataset rows unchanged, zero duplicate indexes inside any batch.
- All four P × K candidates: 1,000/1,000 exact full batches, 16 speakers/batch, 2 samples/speaker, zero structural violations.
- Sample-random: 100% utterance coverage, 0 repeats; speaker sample exposure min/median/max 5/29/1,595, CV 2.0216.
- Global P16K2: all speakers selected; exposure 64/66/66, CV 0.01249; unique coverage 16,555 (51.7376%); repeats 15,445 (48.2656%); queue cycles 1,740.
- Hybrid window 8: all speakers selected; exposure 56/68/76, CV 0.09243; unique coverage 16,903 (52.8252%); repeats 15,097 (47.1781%); queue cycles 1,661.
- Hybrid window 16: exposure 62/66/68, CV 0.02822; coverage 16,671 (52.1001%); repeats 15,329 (47.9031%); queue cycles 1,709.
- Hybrid window 32: exposure 64/66/68, CV 0.01366; coverage 16,591 (51.8501%); repeats 15,409 (48.1531%); queue cycles 1,728.
- Median distinct shards/batch: sample-random 28; global 16; hybrid W8/W16/W32 6/9/12.
- Metadata LRU-8 estimated loads/sample: sample-random 0.93537; global 0.68503; W8 0.20072; W16 0.33150; W32 0.49313.
- Metadata LRU-2 loads/sample: 0.98353 / 0.91741 / 0.60047 / 0.75200 / 0.84481 respectively.
- Metadata LRU-16 loads/sample: 0.86968 / 0.44394 / 0.19034 / 0.27606 / 0.35766 respectively.

### Real-cache benchmark methodology

- Real train cache, LRU 8, batch size 32, `validate_finite=False`, seed 20260727.
- Workers 0 and 2; three repeats per candidate/worker configuration; 30 successful runs.
- Two warm-up batches plus 32 measured batches per repeat.
- Candidate execution order deterministically shuffled and rotated per repeat.
- First-batch, end-to-end (including iterator/worker startup), and post-warm-up steady-state timings recorded separately.
- Every returned Fbank/label/metadata batch validated.
- OS cache effects remain; no cold-disk claim and no filesystem-cache clearing.
- Worker prefetch may benefit steady-state timing.

### Exact benchmark medians [min, max]

Worker 0:

- Sample-random: first 0.2592 [0.2085, 0.2724] s; end-to-end 7.5022 [6.2593, 7.6113] s; steady 150.46 [148.19, 180.16] samples/s; actual loads/sample 0.94531 [0.93750, 0.94922].
- Global P16K2: first 0.3283 [0.2674, 0.3404] s; end-to-end 5.2144 [4.7567, 5.6629] s; steady 220.45 [204.44, 244.06] samples/s; actual loads/sample 0.67578 [0.66602, 0.69531].
- Hybrid W8: first 0.8191 [0.6571, 0.8450] s; end-to-end 2.3860 [1.8957, 2.5576] s; steady 708.37 [634.16, 885.04] samples/s; actual loads/sample 0.19238 [0.18652, 0.19727].
- Hybrid W16: first 1.3319 [1.2582, 1.3617] s; end-to-end 3.6754 [3.5829, 4.1245] s; steady 476.67 [390.55, 487.48] samples/s; actual loads/sample 0.33594 [0.29688, 0.35645].
- Hybrid W32: first 2.3738 [2.1893, 2.8473] s; end-to-end 6.0035 [5.4744, 8.3948] s; steady 306.65 [216.99, 335.06] samples/s; actual loads/sample 0.48242 [0.48242, 0.48438].

Worker 2:

- Sample-random: first 4.8535 [4.8204, 5.2086] s; end-to-end 13.5642 [12.9605, 14.3623] s; steady 117.15 [111.90, 126.35] samples/s.
- Global P16K2: first 5.5486 [5.1322, 5.5938] s; end-to-end 10.4263 [9.7384, 10.5300] s; steady 214.48 [207.22, 224.17] samples/s.
- Hybrid W8: first 4.7785 [4.7581, 5.8598] s; end-to-end 5.9851 [5.7848, 6.8700] s; steady 1,018.40 [836.64, 1,063.43] samples/s.
- Hybrid W16: first 6.6110 [6.3827, 6.7443] s; end-to-end 8.7944 [8.3453, 8.8069] s; steady 497.43 [473.23, 524.17] samples/s.
- Hybrid W32: first 7.4473 [7.0753, 7.9690] s; end-to-end 10.4711 [9.7812, 10.8023] s; steady 382.19 [308.26, 414.88] samples/s.

Worker-local actual shard loads are intentionally unavailable for worker 2.

### Commands run

```text
.venv-cuda\Scripts\python.exe -m unittest tests.test_cached_fbank_samplers -v
.venv-cuda\Scripts\python.exe -m py_compile src\cached_fbank_dataset.py src\cached_fbank_samplers.py scripts\benchmark_cached_fbank_samplers.py tests\test_cached_fbank_samplers.py tests\test_cached_fbank_sampler_benchmark.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_cached_fbank_samplers tests.test_cached_fbank_sampler_benchmark -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\benchmark_cached_fbank_samplers.py --cache-dir outputs\fbank_cache_v1 --metadata-only --output reports\train_sampler_benchmark_v1.json
.venv-cuda\Scripts\python.exe scripts\benchmark_cached_fbank_samplers.py --cache-dir outputs\fbank_cache_v1 --output reports\train_sampler_benchmark_v1.json --repeats 3 --measured-batches 32 --warmup-batches 2 --workers 0 2
git diff --check
git diff
git status --short
```

The first focused run found two test-call configuration mistakes and both were corrected. The first full benchmark launch had a one-second shell timeout and terminated before measurement; the standard command was rerun with an adequate timeout.

### Tests

- Focused sampler/benchmark helper suite: 11 passed.
- Complete unit suite: 37 passed.
- `py_compile`: passed for every new or modified Python file.
- Real-cache benchmark: 30/30 candidate/worker/repeat runs passed with valid batches.

### Result

PASS

### Limitations

- OS cache state and execution order affect timing; these are not cold-disk measurements.
- Worker shard-load totals were not instrumented.
- The balanced epoch necessarily trades utterance coverage for speaker exposure balance.
- Hybrid W8 exposure dispersion is higher than global P16K2, although all speakers are selected.
- Throughput and sampler metadata do not establish downstream verification accuracy.

### Recommended sampler configuration (suggestion only)

Hybrid P=16, K=2, active shard window 8 is the leading candidate. It preserved exact batches and all-speaker exposure while producing the lowest LRU-8 shard-load rate and highest measured steady throughput. Monitor its higher speaker-exposure CV relative to global balance. This is not approval to train and not a model-quality claim.

### Explicitly deferred

AAM-Softmax, classification loss, ECAPA forward/fine-tuning, backward propagation, optimizer/scheduler/mixed precision, checkpoints, verification trials, cosine scoring, threshold selection, EER, final test evaluation, preprocessing, augmentation, cache/manifest/split regeneration, distributed training, commit, and push.

## 2026-07-27 22:09:22 +07:00 - Sampler determinism and benchmark-reporting review patch

### Goal and reviewed issues

Patch only the reviewed cross-process hybrid determinism, fail-closed benchmark reporting, custom artifact safety, and file-classification issues within the current sampler task.

### Exact fixes

- Replaced the hybrid sampler's unordered active-shard set with an ordered list preserving `shard_order`.
- Built eligibility in deterministic shard/index order and sorted usable speaker IDs before seeded random tie-break assignment.
- Added `out_of_range_indexes` and `malformed_batch_sizes` correctness metrics.
- Added `validate_report_payload()` and `write_validated_outputs()`. PASS is assigned only after all candidate and real-cache invariants succeed.
- Validation now rejects duplicates, out-of-range indexes, malformed sizes, P × K violations, nondeterminism, unchanged different epochs, Dataset mutation, incomplete speaker exposure, unsuccessful benchmark runs, and invalid real-cache batches.
- Replaced JSON-only path resolution with atomic JSON/Markdown/CSV path resolution. Custom runs require explicit JSON output, derive companion paths, and cannot target standard artifacts.
- Allowed short custom verification workloads while preserving the existing standard-run defaults and paths.
- Added an explicit run classification to generated Markdown.

### Cross-process determinism method

`tests/run_hybrid_sampler_fixture.py` is a Windows-safe entry point using the synthetic cache. The unit test launches it with `sys.executable`, `PYTHONHASHSEED=1`, and `PYTHONHASHSEED=999`, compares stable JSON batches for epoch 0, then proves epoch 1 differs and remains exact P × K.

### Tests and commands

```text
.venv-cuda\Scripts\python.exe -m py_compile src\cached_fbank_samplers.py scripts\benchmark_cached_fbank_samplers.py tests\test_cached_fbank_samplers.py tests\test_cached_fbank_sampler_benchmark.py tests\run_hybrid_sampler_fixture.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_cached_fbank_samplers -v
.venv-cuda\Scripts\python.exe -m unittest tests.test_cached_fbank_sampler_benchmark -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\benchmark_cached_fbank_samplers.py --cache-dir outputs\fbank_cache_v1 --candidates hybrid_w8 --workers 0 --repeats 1 --measured-batches 4 --warmup-batches 1 --output reports\train_sampler_determinism_patch_w8_verification_v1.json --run-label "post-fix hybrid W8 short verification"
git diff --check
git diff
git status --short
```

- Focused sampler tests: 9 passed, including the cross-process/hash-seed test.
- Focused benchmark/report tests: 6 passed.
- Complete unit suite: 41 passed.
- `py_compile`: passed.

### Short W8 real-cache verification

- P=16, K=2, batch size 32; 1,000/1,000 metadata batches exact.
- Duplicate indexes: 0; out-of-range indexes: 0; malformed sizes: 0; P × K violations: 0.
- Same seed/epoch reproducible: true; different epoch changed: true.
- All 488 speakers selected; Dataset unmodified.
- DataLoader Fbank `[32, 301, 80]` float32 CPU, labels `[32]` int64, metadata aligned.
- One worker-0 repeat, one warm-up batch, four measured batches.
- First batch: 0.86127 s; end-to-end: 1.20594 s; steady throughput: 443.27 samples/s.
- Actual measured shard loads/sample: 0.1953125.
- No WAV access, model inference, or training.
- Result: PASS.

### W8 metadata comparison

- Exact 16 × 2, zero structural violations, zero duplicates, and all-speaker exposure: unchanged.
- Unique coverage: 16,903 -> 16,897 (-6).
- Repeated selections: 15,097 -> 15,103 (+6).
- Exposure min/median/max: unchanged at 56/68/76.
- Exposure CV: 0.0924310554 -> 0.0926576360 (+0.00022658).
- Median distinct shards/batch: unchanged at 6.
- Metadata LRU-8 loads/sample: 0.20071875 -> 0.19956250 (-0.00115625, slightly better).

The exact sequence changed because removing unordered hash iteration changes seeded tie-break assignment. Correctness and median locality did not regress.

### Output safety and standard report preservation

The custom JSON produced custom-stem Markdown and CSV files. Standard full-run artifacts were not targeted. Their SHA-256 hashes were identical before and after:

- JSON `144EEAE76D001C9461AC425EFDF6118D09C2E79795A7245F84E763BB7F1F6370`
- Markdown `E089AC50D7A1EBFDD821E89CD065C8E910956059A5A9DF23D2BE00B07D99A613`
- CSV `920A14652169A85EB1F3210F2355077C6904528CBF040B08B837633FC51A94FA`

### File-classification correction

The sampler module, benchmark script, sampler tests, and standard reports already existed earlier in this current task and are classified as modified or preserved by this patch, not newly created by the patch. New patch files are the subprocess helper, patch report, and custom W8 verification artifacts. Historical worklog text was not rewritten.

### Result

PASS

### Deferred

AAM-Softmax, ECAPA forward/fine-tuning, training, optimization, checkpoints, verification, thresholds, EER, final evaluation, augmentation, preprocessing, cache/manifest/split changes, commit, and push.

## 2026-07-27 22:29:04 +07:00 - Deterministic Validation Trials and Pretrained ECAPA Baseline v1

### Goal and inspected files

Create fixed validation-only verification trials and measure the pretrained ECAPA baseline. Read `AGENTS.md`, this complete worklog, the cached Dataset/samplers, SpeechBrain frontend, cache producer/smoke test, five required prior reports, real validation config/index, approved validation manifest, tests, ignore rules, git status, and diffs.

### Validation invariants and trials

- Validation: 8,504 unique rows, 100 speakers, all labels `-1`, split `validation`, 34 shards.
- Manifest/index path sets, speakers, and filename groups agree; every position is valid/unique and every speaker has at least two utterances.
- Seed 20260727. Positives enumerate unordered within-speaker pairs and sample without replacement up to 100; low-resource speakers use all pairs.
- Negatives select two minimum-participation distinct speakers with seeded ties, select utterances, and reject duplicate canonical path pairs.
- Positive/negative: 9,764 / 9,764. Positive/speaker min/mean/median/max: 45/97.64/100/100; 10 below cap.
- Negative participation min/mean/median/max: 195/195.28/195/196; CV 0.002299256895.
- Unique negative speaker pairs: 4,271; per speaker-pair min/max: 1/8.
- Trial SHA-256: `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`.

### Preflight, extraction, and baseline

- Fbank `[4,301,80]`; normalized `[4,301,80]`; raw embedding `[4,1,192]`; scoring `[4,192]`.
- Lengths `[4]`, float32, all 1.0; all tensors float32, finite, and `cuda:0`.
- Inference mode; no transpose, `compute_features`, classifier-head call, gradient, update, or parameter change.
- Extracted 8,504 embeddings shaped `[8504,192]`, float32 CPU, batch 64/workers 0.
- Elapsed 45.944630 seconds; 185.092360 utterances/s.
- Peak allocated/reserved VRAM: 2,567,797,248 / 3,368,026,112 bytes.
- Embedding and score save/load maximum differences: 0.0 / 0.0.
- 19,528 finite cosine scores, range `[-0.2044644654, 0.9351066351]`.
- Same-speaker mean/std/median: 0.4905432510 / 0.1512383760 / 0.5050629079.
- Different-speaker mean/std/median: 0.1633683063 / 0.1172796843 / 0.1532087401.
- EER 0.11665301106104053 (11.665301106104053%), threshold 0.31165990233421326.
- FAR/FRR: 0.11665301106104053 / 0.11665301106104053. Accept same when score >= threshold.

### Files created and modified

Created verification trial/metric/baseline modules, two scripts, three focused test files, fixed trial CSV/config, two Markdown reports, JSON baseline report, and ignored embedding/score/config outputs. Modified this worklog append-only. Existing dirty sampler/Dataset work was preserved.

### Commands and tests

```text
.venv-cuda\Scripts\python.exe -m py_compile src\verification_trials.py src\verification_metrics.py src\verification_baseline.py scripts\generate_validation_trials_v1.py scripts\evaluate_pretrained_validation_baseline.py tests\test_verification_trials.py tests\test_verification_metrics.py tests\test_verification_baseline.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_verification_trials tests.test_verification_metrics tests.test_verification_baseline -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\generate_validation_trials_v1.py
.venv-cuda\Scripts\python.exe scripts\evaluate_pretrained_validation_baseline.py --batch-size 64 --device cuda:0
```

Focused tests: 13 passed. Complete suite: 54 passed. The first baseline launch had a one-second shell timeout and ended before model work; the identical command then completed successfully.

### Input preservation, result, limitations, and deferred

Before/after SHA-256 values remained identical: validation manifest `9f553f...d38b`, validation index `734607...08bc`, cache config `c829b3...29e9`. **PASS.**

The result is one fixed-trial pretrained validation baseline; it is not evidence of fine-tuning quality. Suggested next step only: review and approve the protocol/baseline before any separately authorized training task.

Deferred: AAM-Softmax, training/fine-tuning, backward/optimizer/scheduler, checkpoints, augmentation, cache or split changes, every final-test operation, commit, and push.

## 2026-07-27 - Validation Trial Integrity and EER Metrics Patch v1

### Scope and reviewed correctness issues

Patched only trial path-to-speaker integrity, EER performance, interpolated-versus-empirical threshold reporting, tests, and saved-score metric recalculation. The prior trial validator trusted speaker IDs declared in the trial, and the prior EER implementation rescanned all trials for every unique score. The previous threshold report also did not distinguish a linearly interpolated ROC crossing from an executable empirical threshold.

### Path ownership and safe paths

Added `validate_trials_against_metadata`, which rejects missing/duplicate authoritative metadata paths, declared owners that differ from real validation owners, targets contradicting real owners, and incomplete expected-speaker representation. Errors include trial IDs and offending paths when applicable. Relative paths now reject colons and Windows drive-relative forms in addition to absolute paths, backslashes, and parent traversal.

Production validation passed for all 19,528 trials and all 100 speakers against the approved 8,504-row validation manifest/index metadata.

### Optimized EER and threshold definitions

The metric implementation pairs scores/targets, sorts once descending, groups identical scores, and advances cumulative accepted-positive/negative counts. Complexity is `O(N log N)` sorting plus an `O(N)` scan. Ties are processed as groups and cannot depend on input order.

Interpolated EER is the linear FAR/FRR crossing between adjacent empirical operating points. Its threshold is explicitly marked `interpolated_non_empirical`; directly applying it need not reproduce the interpolated errors.

The executable empirical point minimizes `abs(FAR-FRR)`, then average error, then prefers the higher threshold. FAR/FRR are independently recomputed using `score >= threshold` and asserted equal to the selected point.

### Exact production comparison

- Previous interpolated EER: `0.11665301106104053`
- Optimized interpolated EER: `0.11665301106104056`
- Absolute difference: `2.7755575615628914e-17`
- Previous/new interpolated threshold: `0.31165990233421326`
- Interpolated FAR/FRR: `0.11665301106104056 / 0.11665301106104056`
- Empirical threshold: `0.31165990233421326`
- Actual empirical FAR/FRR: `0.11665301106104056 / 0.11665301106104056`
- Empirical gap/average error: `0.0 / 0.11665301106104056`

The negligible EER difference is floating-point arithmetic ordering: FRR is now computed directly from integer rejected-positive counts rather than as `1.0 - accepted_fraction`. It is not a changed threshold, tie grouping, or substantive metric correction. This production set happens to have an exact empirical FAR/FRR equality at the selected stored-score threshold.

### Metrics-only command, performance, and preservation

```text
.venv-cuda\Scripts\python.exe scripts\recompute_pretrained_validation_metrics.py
```

Loaded existing embeddings `[8504,192]` float32 CPU, 19,528 stored scores/targets/IDs, fixed trial CSV, and validation manifest/index metadata. Metric runtime was `0.05654350000259001` seconds for 19,528 trials and 19,523 unique scores.

Protected hashes before and after:

- Trial CSV: `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`
- Embeddings: `7c9f6845e55517a6981858bc4327cad09b9c037c2b288f1b26ff28120eaec828`
- Scores: `b67cd388793ef3eb1f722f38dffa20cef7f789de70c07d6f43d4ea2c64b02cc2`
- Validation manifest: `9f553fa55b50ee071120bcbb2ce3c2caf9de4799cfb615eb732bbefd0d73d38b`
- Validation cache index: `7346073ed9b47354c5f85889222ad9a538d2c4ee14e0b7d85001e3c0d61608bc`
- Cache config: `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`

No embedding extraction, cosine rescoring, SpeechBrain/ECAPA import or inference, CUDA operation, WAV access, final-test access, or training occurred.

### Files and tests

Modified verification trial/metric/baseline helpers, future full evaluator reporting, three focused test files, baseline JSON/Markdown, and this append-only worklog. Created the metrics-only recalculation script and `reports/validation_eer_metrics_patch_v1.md`. Trial CSV/config, tensor artifacts, validation trial summary, portable manifests, cache, sampler, environments, and model files were not modified.

```text
.venv-cuda\Scripts\python.exe -m py_compile src\verification_trials.py src\verification_metrics.py src\verification_baseline.py scripts\evaluate_pretrained_validation_baseline.py scripts\recompute_pretrained_validation_metrics.py tests\test_verification_trials.py tests\test_verification_metrics.py tests\test_verification_baseline.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_verification_trials -v
.venv-cuda\Scripts\python.exe -m unittest tests.test_verification_metrics -v
.venv-cuda\Scripts\python.exe -m unittest tests.test_verification_baseline -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Compilation passed. Focused trial/metric/baseline-report suites passed 6/7/5 tests. Complete suite passed 59 tests.

### Result and deferred

**PASS.** Deferred: trial regeneration, ECAPA inference/extraction, training/fine-tuning, AAM-Softmax, optimization, checkpoints, augmentation, preprocessing, cache/manifest/split changes, every final-test operation, commit, and push.

## 2026-07-28 13:44:49 +07:00 - AAM-Softmax Training Pipeline and CUDA Smoke Test v1

### Goal and exact scope

Implement only the project-owned AAM-Softmax training helpers, deterministic
P16K2 logical-to-microbatch path, prescribed ECAPA/BatchNorm trainability policy,
two-group AdamW optimizer, checkpoint v1, two-step CUDA smoke with fresh-object
resume, separate microbatch-4 capacity probe, focused tests, and reports. No
epoch, validation/EER/final evaluation, augmentation, scheduler, commit, or push.

### Repository state and files inspected

- Read root `AGENTS.md` and this complete worklog.
- Inspected git status/diff, `.gitignore`, repository structure, cached Dataset
  and DataLoader, hybrid sampler, SpeechBrain frontend/loading helper, cached
  inference/cache scripts, tests, reports, and output conventions.
- Preserved substantial pre-existing dirty and untracked sampler and validation
  work.
- A recursive structure listing unintentionally enumerated final-test cache
  filenames and sizes. It did not open a final-test index, manifest, shard
  tensor, or data content. This violates the task's strict prohibition on any
  final-test artifact access and determines the overall FAIL below.

### Implementation decisions

- Added `AAMSoftmax` with deterministic seed-20260727 Xavier weight
  initialization, `[488,192]` weight, no bias, `m=0.2`, `s=30`, float32
  normalization/angular math, open-interval cosine clamp, standard monotonic
  ArcFace fallback, and target-only margin replacement.
- Added deterministic label-major speaker round-robin reordering that preserves
  every aligned batch field and produces distinct-speaker microbatches at sizes
  2 and 4.
- Added exact loss scaling as each microbatch CE sum divided by logical size 32.
- Added exact optimizer coverage/disjointness validation for ECAPA at `1e-5`
  and AAM at `1e-3`, with AdamW weight decay `1e-4`.
- Applied `embedding_model.train()` then eval mode only to 31 BatchNorm modules;
  all ECAPA parameters including BatchNorm affine parameters remained trainable.
  Ninety-three running-statistics buffers were snapshotted and compared exactly
  after every step. `mean_var_norm` was preserved and excluded from optimization.
- Loaded the SpeechBrain checkpoint on CPU, retained only `mean_var_norm` and
  `embedding_model`, and moved only those modules to CUDA. Neither
  `compute_features` nor the pretrained classifier was retained or called.
- Used float16 CUDA autocast for ECAPA and float32 AAM/logits/loss. GradScaler
  remained enabled with explicit initial scale 128.
- Checkpoint v1 uses an atomic temporary save plus `os.replace`, repository-
  relative identities/hashes, complete model/AAM/optimizer/scaler/counter/config
  state, and Python/NumPy/CPU/CUDA RNG states.

### Files created

- `src/aam_training.py`
- `scripts/smoke_test_aam_training_cuda.py`
- `tests/test_aam_training.py`
- `reports/aam_softmax_training_smoke_v1.md`
- `reports/aam_softmax_training_smoke_v1.json`
- `reports/aam_softmax_training_smoke_v1_main.json`
- `reports/aam_softmax_training_smoke_v1_microbatch4.json`
- Ignored `outputs/aam_softmax_training_smoke_v1/checkpoint_v1.pt`

### Files modified

- `src/cached_fbank_dataset.py` (aligned Dataset indexes and optional dedicated
  DataLoader generator)
- `reports/CODEX_WORKLOG.md` (append only)

### Exact commands

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

The main command ran four times. Three fail-closed diagnostic attempts stopped
before `optimizer.step`: the initial run detected non-finite gradients; the AAM
clamp was corrected and a boundary test added; a diagnostic run then identified
NaN/Inf in `blocks.0.norm.norm.weight` with GradScaler's default initial scale
65,536. The final successful run kept float16 autocast and enabled GradScaler
with initial scale 128. No failed attempt saved a checkpoint, updated
parameters, fell back to full float32, or continued after failure.

### Tests

- Final focused new suite: 15 passed.
- Dataset/sampler/training combined suite: 35 passed before the boundary test
  addition.
- Final complete repository suite: 74 passed.
- `py_compile`: passed for every new or modified Python file.

### Main microbatch-2 result

Technical pipeline result: **PASS**.

- Two cached-train-only P16K2 logical batches, 16 microbatches per step, and two
  optimizer steps completed without OOM.
- Step 1 loss/duration: `14.102674305438995` /
  `1.9493115000004764` seconds.
- Step 1 ECAPA/AAM gradient norms:
  `59.820380658764286` / `8.563820396025983`.
- Step 1 ECAPA maximum/aggregate delta:
  `1.0013580322265625e-05` / `202.6730268294923`.
- Step 1 AAM maximum/aggregate delta:
  `0.0010000169277191162` / `93.6861801147461`.
- Step 1 peak allocated/reserved:
  `497169920` / `683671552` bytes.
- Step 2 loss/duration: `14.168766677379608` /
  `1.1059552999995503` seconds.
- Step 2 ECAPA/AAM gradient norms:
  `60.42722928136366` / `8.381696159213307`.
- Step 2 ECAPA maximum/aggregate delta:
  `1.0013580322265625e-05` / `131.15723157610046`.
- Step 2 AAM maximum/aggregate delta:
  `0.0010013654828071594` / `67.10301208496094`.
- Step 2 peak allocated/reserved:
  `661258752` / `861929472` bytes.
- BatchNorm buffers were exactly unchanged after both steps.

### Checkpoint and resume

- Atomic checkpoint size: 250,699,903 bytes.
- Whole file payload and every embedding, mean-normalization, AAM, optimizer,
  GradScaler, counter, and configuration state round-tripped exactly.
- Fresh SpeechBrain, AAM, AdamW, and GradScaler objects were reconstructed.
- RNG states were restored and the BatchNorm policy reapplied.
- The regenerated epoch-0 sampler position 1 identity exactly matched both the
  pre-step expected second batch and checkpoint identity.
- A complete resumed optimizer step succeeded; global optimizer step became 2.

### Microbatch-4 capacity probe

**PASS** in a separate fresh process after the main PASS.

- Eight microbatches, one complete backward/GradScaler/first AdamW step.
- Loss/duration: `14.10267436504364` / `1.4691242999997485` seconds.
- Peak allocated/reserved: `603883520` / `723517440` bytes.
- ECAPA/AAM gradient norms:
  `59.82091489954806` / `8.563820396025983`.
- Both parameter groups changed and all BatchNorm buffers remained exact.

### Protected artifacts and hashes

The hashes recorded before optimizer work matched the final hashes:

- Train manifest:
  `1b8837d97901f10218cf2fe193e29a56e4695138bbd2037cbd9818e437a3b3fe`
- Cache config:
  `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`
- Train cache index:
  `5bd1999f92623084cba2558f19337f76cf4eb78878b63211366768742c6c9a99`

No WAV was read. No source audio, manifest, split, cache, environment, or
pretrained weight was modified. The implementation and CUDA commands opened no
validation or final-test content. The preflight metadata enumeration remains the
strict-scope failure.

### Result

**FAIL (strict scope compliance).** All technical pipeline stages passed, but
the final PASS rule required no final-test artifact access and the preflight
listing enumerated final-test cache metadata.

### Limitations and explicitly deferred

This is a two-step smoke test and one-step capacity probe, not a model-quality
result. Deferred: full fine-tuning, epochs, augmentation, validation extraction
and scoring, threshold selection, EER, checkpoint selection, scheduler work,
all final-test evaluation, cache/manifest/split regeneration, distributed
training, commit, and push.

## 2026-07-28 - AAM-Softmax training smoke scope-compliance rerun v1.1

### Scope and remediation

Performed only the requested v1.1 remediation and rerun. The prior v1 report
remains unchanged: its technical result was PASS and its strict-scope result was
FAIL because preflight metadata enumeration exposed quarantined pathnames.

This rerun used direct train allowlisted paths only:

- `outputs/fbank_cache_v1/fbank_cache_config_v1.json`
- `outputs/fbank_cache_v1/train_feature_index_v1.csv`
- Exact train shard files selected by the deterministic sampler
- `speechbrain/spkrec-ecapa-voxceleb` through the existing local model cache
- Fresh v1.1 report, runtime, and checkpoint destinations

No recursive repository, cache, or output listing was used. The mandatory
`git status --short` inspection can report existing worktree pathnames, but no
validation artifact was opened, stat-ed, hashed, or loaded. No final-test
artifact was enumerated, opened, stat-ed, hashed, or loaded.

### Dataset diff review

Reviewed the existing `src/cached_fbank_dataset.py` diff before the rerun.
Keeping `dataset_index` in the additive sample/collate schema is necessary to
preserve exact row identity through round-robin sampling and restart checks.
Keeping the optional dedicated DataLoader `generator` is necessary so iterator
construction does not consume restored global model RNG. Existing keys,
row/index mapping, tensor shapes and dtypes, and sampler behavior remain
compatible. A separate wrapper or manual collate path would add duplicate
machinery without reducing scope.

### Train-only fail-closed guards

- The smoke entry point accepts only the literal split `train`.
- The cache root must resolve exactly to `outputs/fbank_cache_v1`.
- The index must be exactly `train_feature_index_v1.csv`.
- Request-path validation runs before Dataset construction.
- Loaded metadata must contain only `final_split == "train"`.
- Labels must be integers in the inclusive range 0 through 487.
- Every referenced feature path must resolve to an exact
  `train/shard_NNNNN.pt` path within the approved train shard directory.
- Logical batches are checked again for train-only split labels.
- Runtime artifact identity hashes only the cache configuration, train index,
  and exact selected train shards.

### Files created

- `reports/aam_softmax_training_smoke_v1_1.md`
- `reports/aam_softmax_training_smoke_v1_1.json`
- `reports/aam_softmax_training_smoke_v1_1_main_runtime.json`
- `reports/aam_softmax_training_smoke_v1_1_probe_runtime.json`
- Ignored `outputs/aam_softmax_training_smoke_v1_1/checkpoint_v1.pt`

### Files modified

- `src/aam_training.py`
- `scripts/smoke_test_aam_training_cuda.py`
- `tests/test_aam_training.py`
- `reports/CODEX_WORKLOG.md` (append only)

The previously reviewed additive changes in `src/cached_fbank_dataset.py` were
retained without further v1.1 modification.

### Exact commands

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

### Tests

- `py_compile`: passed for all five targeted Python files.
- Focused AAM training suite: 21 passed.
- Complete repository suite: 80 passed.
- Train-only negative guard cases used synthetic temporary cache metadata and
  did not access real validation or final-test inputs.

### Main microbatch-2 result

**PASS.** Two cached-train-only P16K2 logical batches completed two optimizer
steps using microbatch 2 and accumulation 16. Both batches had 32 unique
utterances, valid shapes and labels, and only the train split. No waveform
frontend or pretrained VoxCeleb classifier call occurred.

- Step 1 loss/duration:
  `14.102674305438995` / `2.0199535000010655` seconds
- Step 1 ECAPA/AAM gradient norms:
  `59.820380658764286` / `8.563820396025983`
- Step 1 ECAPA maximum/aggregate delta:
  `1.0013580322265625e-05` / `202.6730268294923`
- Step 1 AAM maximum/aggregate delta:
  `0.0010000169277191162` / `93.6861801147461`
- Step 1 peak allocated/reserved:
  `497169920` / `683671552` bytes
- Step 2 loss/duration:
  `14.168766677379608` / `1.1580651999993279` seconds
- Step 2 ECAPA/AAM gradient norms:
  `60.42722928136366` / `8.381696159213307`
- Step 2 ECAPA maximum/aggregate delta:
  `1.0013580322265625e-05` / `131.15723157610046`
- Step 2 AAM maximum/aggregate delta:
  `0.0010013654828071594` / `67.10301208496094`
- Step 2 peak allocated/reserved:
  `661258752` / `861929472` bytes
- All 93 tracked BatchNorm buffers remained exactly unchanged.

### Checkpoint and resume

- Atomic checkpoint size: 250,701,631 bytes.
- Checkpoint schema, whole payload, and every retained model, normalization,
  AAM, optimizer, GradScaler, counter, RNG, sampler, and configuration state
  round-tripped exactly.
- Fresh objects were reconstructed, RNG state was restored, and the expected
  original second logical batch identity matched exactly.
- A complete resumed optimizer step succeeded and global optimizer step became
  2.

### Microbatch-4 capacity probe

**PASS** in a separate fresh process after the main PASS.

- Eight microbatches completed one backward/GradScaler/AdamW step.
- Loss/duration:
  `14.10267436504364` / `1.1910272000004625` seconds
- Peak allocated/reserved:
  `603883520` / `723517440` bytes
- ECAPA/AAM gradient norms:
  `59.82091489954806` / `8.563820396025983`
- ECAPA maximum/aggregate delta:
  `1.0013580322265625e-05` / `202.67372208437882`
- AAM maximum/aggregate delta:
  `0.0010000169277191162` / `93.68618774414062`
- Both parameter groups changed and all BatchNorm buffers remained exact.

### Train artifacts and protected hashes

The deterministic main run loaded 13 exact train shards:

```text
train/shard_00002.pt
train/shard_00007.pt
train/shard_00038.pt
train/shard_00039.pt
train/shard_00047.pt
train/shard_00050.pt
train/shard_00065.pt
train/shard_00078.pt
train/shard_00080.pt
train/shard_00085.pt
train/shard_00095.pt
train/shard_00110.pt
train/shard_00119.pt
```

The probe used the seven-shard subset `00002`, `00007`, `00038`, `00047`,
`00065`, `00085`, and `00110`. Pre/post SHA-256 comparison covered the cache
configuration, exact train index, and the 13 exact main-run shards: 15 files
compared, zero mismatches.

- Cache configuration:
  `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`
- Train cache index:
  `5bd1999f92623084cba2558f19337f76cf4eb78878b63211366768742c6c9a99`

No source audio, manifest, split, cache, environment, or pretrained weight was
modified.

### Result

**PASS.** Technical pass: true. Scope-compliance pass: true. Overall pass:
true. The strict quarantine requirements were satisfied by direct train-only
path validation and exact selected-shard access.

### Limitations and explicitly deferred

This remains a two-step smoke test and one-step capacity probe, not a
model-quality result. Deferred: full fine-tuning, epochs, augmentation,
validation extraction and scoring, threshold selection, EER, checkpoint
selection, scheduler work, all final-test evaluation, cache/manifest/split
regeneration, distributed training, commit, and push.

## 2026-07-28 - One-Epoch ECAPA-TDNN Fine-Tuning Pilot + Fixed Validation v1

### Goal and exact scope

Implement and execute only one deterministic balanced ECAPA-TDNN + AAM epoch,
perform a full fresh-object checkpoint roundtrip without another optimizer
step, score the existing immutable validation protocol, compare epoch 0 with
the historical pretrained validation baseline, and stop. The task did not
authorize epoch 1, full fine-tuning, hyperparameter work, final-test access,
commit, or push.

**Outcome: technical PASS; scope-compliance PASS; overall PASS; model-quality
result IMPROVED.** Exactly 1,000 optimizer updates and 32,000 logical
selections completed in epoch 0. Epoch 1 was not started.

### Allowlisted files inspected

Before editing or execution, the following mandatory context was read:

- `AGENTS.md`, `.gitignore`, full `reports/CODEX_WORKLOG.md`, initial
  `git status --short`, and targeted task-file diffs
- `reports/aam_softmax_training_smoke_v1.md` and `.json`
- `reports/aam_softmax_training_smoke_v1_1.md` and `.json`
- `reports/validation_trials_v1_summary.md`
- `reports/pretrained_ecapa_validation_baseline_v1.md` and `.json`
- `reports/validation_eer_metrics_patch_v1.md`
- `src/aam_training.py`, `src/cached_fbank_dataset.py`,
  `src/cached_fbank_samplers.py`, `src/verification_trials.py`,
  `src/verification_metrics.py`, and `src/verification_baseline.py`
- `scripts/smoke_test_aam_training_cuda.py`,
  `scripts/evaluate_pretrained_validation_baseline.py`, and
  `scripts/recompute_pretrained_validation_metrics.py`
- `tests/test_aam_training.py`, `tests/test_verification_trials.py`,
  `tests/test_verification_metrics.py`, and
  `tests/test_verification_baseline.py`

No recursive repository, `outputs/`, cache, or manifest listing was used. No
final-test artifact was accessed, enumerated, statted, hashed, opened, loaded,
or evaluated.

### Implementation decisions

- Added a narrow fixed-constant pilot helper and one explicit CUDA runner,
  reusing the smoke-tested AAM, BatchNorm, optimizer grouping, logical
  round-robin, cached Dataset/sampler, verification scorer, and EER logic.
- Enforced epoch 0 only, exact step/cursor/checkpoint triggers, strict
  scalar-only logging, strict checkpoint/runtime/metric schemas, safe atomic
  writes, fixed allowlisted paths, immutable trial identity, and final-test
  path rejection.
- Used the production cached-feature path:
  `[B,301,80] -> mean_var_norm -> embedding_model -> [B,1,192] ->
  squeeze(1)`.
- Validated every trainable gradient and every AdamW parameter step counter on
  every optimizer update; used an optimizer post-hook to detect skipped
  GradScaler updates.
- Revalidated saved validation tensors and JSON artifacts after writing.
- After the CUDA run, added explicit 488-speaker and train/validation
  speaker-disjoint preflight assertions. A separate read-only audit verified
  those properties for the executed inputs; the one-epoch CUDA run was not
  repeated.

### Files created

- `src/ecapa_one_epoch_pilot.py`
- `scripts/run_ecapa_aam_one_epoch_pilot.py`
- `tests/test_ecapa_one_epoch_pilot.py`
- `reports/ecapa_aam_one_epoch_pilot_v1.md`
- `reports/ecapa_aam_one_epoch_pilot_v1.json`
- Ignored `outputs/ecapa_aam_one_epoch_pilot_v1/` runtime directory containing
  console logs, JSONL/logical-loss logs, `last.pt`, `epoch_000.pt`, `best.pt`,
  validation embedding/score/metric/runtime artifacts, and
  `pilot_runtime.json`

### Files modified

- `reports/CODEX_WORKLOG.md` (this append-only entry)

No existing source implementation, source audio, manifest, split, cache
tensor, trial protocol, pretrained weight, or environment was modified.

### Exact commands

```text
.venv-cuda\Scripts\python.exe -m py_compile src\ecapa_one_epoch_pilot.py scripts\run_ecapa_aam_one_epoch_pilot.py tests\test_ecapa_one_epoch_pilot.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_ecapa_one_epoch_pilot -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\run_ecapa_aam_one_epoch_pilot.py --device cuda:0
git diff --check
git diff -- src\ecapa_one_epoch_pilot.py scripts\run_ecapa_aam_one_epoch_pilot.py tests\test_ecapa_one_epoch_pilot.py reports\ecapa_aam_one_epoch_pilot_v1.md reports\ecapa_aam_one_epoch_pilot_v1.json reports\CODEX_WORKLOG.md
git status --short
```

The CUDA command's stdout and stderr were redirected to ignored files in the
pilot output directory.

### Training configuration

- Interpreter: `.venv-cuda/Scripts/python.exe`
- Python / PyTorch / SpeechBrain:
  `3.10.11 / 2.2.0+cu121 / 1.0.3`
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, `cuda:0`
- Model: `speechbrain/spkrec-ecapa-voxceleb`
- Sampler: `HybridShardAwareSpeakerBatchSampler`, P16K2, W8, seed `20260727`,
  epoch `0`
- Exact epoch: 1,000 logical batches x 32 = 32,000 selections
- Physical microbatch / accumulation: `4 / 8`; no microbatch-2 fallback
- AAM: 488 classes, 192 dimensions, no bias, margin `0.2`, scale `30`,
  float32 math
- AdamW groups:
  ECAPA `lr=1e-5`, AAM `lr=1e-3`, both `weight_decay=1e-4`
- AMP: ECAPA float16 autocast, AAM/loss float32, GradScaler initial scale 128
- BatchNorm: embedding model train mode; only its 31 BatchNorm modules eval;
  affine parameters trainable; all 93 running-stat buffers bit-exact
- No augmentation, scheduler, warmup, gradient clipping, or early stopping

The sampler plan SHA-256 was
`796f9f3ec2eff82a45052e0482cfde997a5b793b22ddcd6741e68f62ee196b04`.
All 1,000 batches were exact P16K2, no logical batch duplicated a Dataset
index, and all 488 speakers were selected.

### One-epoch result

- Task UTC start/end:
  `2026-07-28T08:00:46.030100+00:00` /
  `2026-07-28T08:12:10.434877+00:00`
- Task duration: `684.4041123000025` seconds
- Training UTC start/end:
  `2026-07-28T08:00:59.590293+00:00` /
  `2026-07-28T08:11:04.479631+00:00`
- Training duration: `604.8890577999991` seconds
- Optimizer updates / selections: `1000 / 32000`
- Epochs started/completed: `[0] / [0]`
- Final cursor: next epoch `1`, batch position `0`
- Epoch 1 started: false
- OOM / skipped optimizer updates: `false / 0`
- GradScaler stayed at `128` for all 1,000 updates

Loss count/first/final/mean/minimum/maximum was:

```text
1000
14.10267436504364
1.342349648475647
5.100674975889735
0.47721143439412117
15.313348054885864
```

ECAPA gradient norm first/final/mean/minimum/maximum was
`59.820916009264685 / 38.96678272901164 / 48.718236445551575 /
22.10841567044168 / 67.12837878484218`.

AAM gradient norm first/final/mean/minimum/maximum was
`8.563820396025983 / 4.2688503717152875 / 6.506972899575333 /
2.6217307668304 / 8.678117474452085`.

The JSONL log contains exactly 20 scalar-only records at steps 50 through
1,000, and the ignored logical-loss artifact contains exactly 1,000 values.
The full 20-window history is in the versioned report.

### Checkpoint results

Atomic rolling `last.pt` writes succeeded at:

- Step 250 -> next cursor `(epoch 0, batch 250)`
- Step 500 -> next cursor `(epoch 0, batch 500)`
- Step 750 -> next cursor `(epoch 0, batch 750)`
- Step 1000 -> next cursor `(epoch 1, batch 0)`

Each was non-empty, readable, schema/counter-valid, and BatchNorm-exact.
`epoch_000.pt` was 250,711,923 bytes with SHA-256
`e82ba006aef4a505244769f137af897a94bbb601a26ffe2b68fbfec135201b67`.

The fresh-object epoch checkpoint roundtrip matched the embedding model,
mean/variance normalization, AAM, AdamW including nonempty moments and all
step counters, GradScaler, RNG, schema/configuration, counters, and BatchNorm
buffers exactly. The loaded cursor was global step 1,000, next epoch 1,
position 0. No post-load optimizer step was taken.

After successful validation, `best.pt` was created atomically as the best among
trained pilot checkpoints; its model state matches `epoch_000.pt`. The
pretrained baseline remains a historical comparison reference only.

### Validation result

Validation ran only after the checkpoint roundtrip passed, in eval and
`torch.inference_mode()` with sequential cached-Fbank traversal, batch size 32,
`num_workers=0`, and no AAM or augmentation.

- Embeddings: 8,504 rows, 100 speakers, `[8504,192]` float32 CPU, all finite
- Extraction duration / throughput:
  `58.01379679999809` seconds / `146.58582042677614` utterances/second
- Fixed trials: `9764` positive + `9764` negative = `19528`
- Scoring / metric durations:
  `0.7327373000007356 / 0.12257150000004913` seconds
- Trial path ownership validated: true
- AAM / pretrained classifier / waveform frontend / `compute_features` calls:
  `0 / 0 / 0 / 0`
- Pilot interpolated EER:
  `0.0646251536255633` (6.46251536255633%)
- Interpolated non-empirical threshold:
  `0.1744520664215088`
- Executable empirical threshold:
  `0.1744520664215088`
- Empirical FAR / FRR / gap / average:
  `0.0646251536255633 / 0.0646251536255633 / 0.0 /
  0.0646251536255633`
- Same-speaker mean/std/median:
  `0.42481032643701927 / 0.15599993927001696 / 0.4358007609844208`
- Different-speaker mean/std/median:
  `0.008027285437529052 / 0.10356156850219239 /
  0.0005986420437693596`
- Score range:
  `[-0.3212318420410156, 0.9287939071655273]`

### Baseline comparison and model quality

The official historical pretrained interpolated EER was
`0.11665301106104056` (11.665301106104057%), with empirical threshold/FAR/FRR
`0.31165990233421326 / 0.11665301106104056 /
0.11665301106104056`.

Pilot minus baseline interpolated EER was `-0.05202785743547726`, or
`-5.202785743547726` percentage points. Relative EER change was
`-0.44600526777875327`, a 44.60052677787533% reduction. With numerical
tolerance `1e-12`, the model-quality classification is **IMPROVED**.

No automatic continuation was triggered by this observation.

### GPU memory

- Training peak allocated/reserved:
  `775443456 / 874512384` bytes
- Validation peak allocated/reserved:
  `1340804096 / 1725956096` bytes
- GPU total:
  `4294443008` bytes

### Protected hashes

Pre/post SHA-256 covered 158 explicitly approved accessed files: six core
inputs, the exact 118 train shards used by epoch 0, and the exact 34 validation
shards loaded. The before and after mappings were identical with zero
mismatches. Aggregate set SHA-256:
`ccf48065c43930df185087e5e18512b9f9b68de39af80bea11e9200e33ab0d2d`.

- Cache configuration:
  `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`
- Train index:
  `5bd1999f92623084cba2558f19337f76cf4eb78878b63211366768742c6c9a99`
- Validation index:
  `7346073ed9b47354c5f85889222ad9a538d2c4ee14e0b7d85001e3c0d61608bc`
- Validation manifest:
  `9f553fa55b50ee071120bcbb2ce3c2caf9de4799cfb615eb732bbefd0d73d38b`
- Validation trial config:
  `a6457bc571965855633f1d277a43427507d708afe779b32ab4e0ec2c58d2eb9a`
- Validation trial CSV:
  `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`

The immutable trial CSV hash matched the required value exactly. The complete
per-file hash mapping is in ignored `pilot_runtime.json`. No final-test file
was included.

### Tests and repository audit

- `py_compile`: PASS for all three new Python files
- Focused pilot suite: 19 passed
- Complete repository suite: 99 passed
- Post-run strict artifact audit: PASS for `last.pt`, `epoch_000.pt`,
  `best.pt`, 20 log records, 1,000 logical losses, 8,504 validation
  embeddings, 19,528 aligned scores, and runtime/metric schemas
- `git diff --check`: PASS
- No commit or push performed

### Limitations and deferred work

This was one deterministic epoch and one fixed-validation observation, not a
statistical study or full fine-tuning result. The final test split remains
untouched, so no final-test or generalization claim is made.

Explicitly deferred: epoch 1 and every later epoch, full fine-tuning,
hyperparameter tuning, augmentation, scheduler, warmup, gradient clipping,
early stopping, every final-test operation, cache/manifest/split/trial
regeneration, commit, and push.

### Post-run strict scope audit and final correction

This append-only correction supersedes the provisional scope/overall PASS
statements earlier in this dated entry.

The saved numerical artifacts, checkpoints, validation results, fixed-trial
identity, and 158-file protected pre/post hash map all validate cleanly.
However, the executed version constructed
`HybridShardAwareSpeakerBatchSampler` before the exact 118-shard epoch plan was
known. The existing sampler metadata validator called `Path.is_file()` for all
125 train shard paths referenced by the train index. The seven unselected
train-shard paths statted were:

```text
train/shard_00075.pt
train/shard_00083.pt
train/shard_00084.pt
train/shard_00086.pt
train/shard_00112.pt
train/shard_00114.pt
train/shard_00117.pt
```

Those seven train shards were not opened, tensor-loaded, or hashed. No
quarantined validation or final-test path was touched. Nevertheless, metadata
stat is outside the task's exact selected-shard allowlist, so the final
classification is:

- Technical pipeline: **PASS**
- Scope compliance: **FAIL**
- Overall result: **FAIL**
- Model-quality result: **IMPROVED**

The original ignored `pilot_runtime.json` self-reported scope/overall PASS
before this issue was discovered. The versioned JSON/Markdown reports
explicitly override those two runtime flags and are authoritative for the
post-run audit.

The final implementation was hardened without rerunning CUDA:

- `src/cached_fbank_samplers.py` now keeps shard-existence validation enabled
  by default but accepts an explicit `validate_shard_existence=False` for
  metadata-only planning.
- The fixed pilot uses that option, materializes its deterministic plan, and
  then stats, hashes, and loads only the selected shard allowlist.
- A synthetic focused test proves that planning can succeed without statting
  missing/unselected real shard paths.

Final Python/test commands and results:

```text
.venv-cuda\Scripts\python.exe -m py_compile src\cached_fbank_samplers.py src\ecapa_one_epoch_pilot.py scripts\run_ecapa_aam_one_epoch_pilot.py tests\test_ecapa_one_epoch_pilot.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_ecapa_one_epoch_pilot -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\run_ecapa_aam_one_epoch_pilot.py --device cuda:0 1> outputs\ecapa_aam_one_epoch_pilot_v1\console.stdout.log 2> outputs\ecapa_aam_one_epoch_pilot_v1\console.stderr.log
git diff --check
git diff -- src\cached_fbank_samplers.py src\ecapa_one_epoch_pilot.py scripts\run_ecapa_aam_one_epoch_pilot.py tests\test_ecapa_one_epoch_pilot.py reports\ecapa_aam_one_epoch_pilot_v1.md reports\ecapa_aam_one_epoch_pilot_v1.json reports\CODEX_WORKLOG.md
git status --short
```

- `py_compile`: PASS for four new or modified Python files
- Focused pilot suite: 20 passed
- Complete repository suite: 100 passed
- CUDA training commands rerun after hardening: zero
- Epoch 1 started: false
- Commit or push: none

The newly modified existing source file is
`src/cached_fbank_samplers.py`. The exact 118 selected train shards and 34
loaded validation shards remain unchanged under the original protected
pre/post map. The seven metadata-only stat paths cannot be given a valid
pre-task content hash retroactively and were not hashed after discovery.

## 2026-07-28 17:55:00 +07:00 - Resumed Multi-Epoch ECAPA-AAM Fine-Tuning v1

### Goal and scope

Resume only from the completed epoch-0 pilot checkpoint, train epochs 1
through 4 at most with per-step cosine decay and fixed validation, select the
best checkpoint, and stop under the configured early-stopping/max-epoch rule.
Epoch 0 was not retrained and no final-test artifact was accessed.

### Implementation

- Added `src/ecapa_multiepoch.py`,
  `scripts/run_ecapa_aam_multiepoch.py`, and
  `tests/test_ecapa_multiepoch.py`.
- Added exact rolling resume state, scheduler state/migration, early stopping,
  atomic epoch/last/best checkpoints, fixed-validation scoring, BatchNorm
  buffer guards, explicit protected-path hashing, and final-test path rejection.
- Preserved all existing dirty/untracked pilot work.

### CUDA result

- Start checkpoint SHA-256:
  `e82ba006aef4a505244769f137af897a94bbb601a26ffe2b68fbfec135201b67`
- Completed resumed epochs: `1, 2, 3, 4`, each exactly 1,000 optimizer updates.
- Validation EERs:
  epoch 1 `0.06022122081114297`,
  epoch 2 `0.057968045882834905`,
  epoch 3 `0.05735354362965998`,
  epoch 4 `0.057455960671855794`.
- Stop reason: `max_epoch`; best epoch: `3`.
- Best checkpoint SHA-256:
  `7c63f4e2a100ce4eede4bb2429064387f80302c36b534baf46a4ae5b0e9cdb4f`.
- Scheduler completed exactly 4,000 resumed steps and ended at factor `0.1`
  with ECAPA/AAM LRs `1e-6 / 1e-4`.
- Zero skipped/invalid updates, zero OOMs, and all 93 BatchNorm running buffers
  remained bit-exact.
- Peak CUDA allocated/reserved:
  `1595687424 / 1845493760` bytes.

### Validation, recovery, and tests

- Every epoch evaluated all 8,504 validation utterances and exactly 19,528
  fixed trials; trial SHA-256 remained
  `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`.
- All 167 explicitly protected inputs matched before/after hashes.
- Two path-allowlist issues failed closed before training and after the
  completed run during runtime serialization. Both reused pilot helpers were
  replaced. The final resume from completed `last.pt` took zero optimizer
  steps and wrote the runtime artifact without repeating training/validation.
- `py_compile`: PASS; focused tests: 12 passed; complete discovery: 112 passed;
  checkpoint/hash audit: PASS; `git diff --check`: PASS.
- Reports:
  `reports/ecapa_aam_multiepoch_v1.md` and
  `reports/ecapa_aam_multiepoch_v1.json`.
- Technical result: **PASS**. Model-quality result: **IMPROVED**.
- No commit or push.

### Deferred

Final-test access/evaluation, epochs after 4, further training, hyperparameter
changes, augmentation, cache/manifest/split/trial regeneration, commit, and
push.

## 2026-07-28 22:43:29 +07:00 - VieSpeaker2.0 Dataset Understanding and Integration Audit v2

### Goal and exact scope

Audit only the supplied Stage-A output at
`E:\VieSpeaker2.0\augmented_dataset`, inventory pilot-bound integration
assumptions, propose a future `full_manifest_v2` schema and config-driven
migration plan, refresh repository governance, and stop. VieSpeaker2.0
completely replaces the old `E:\VieSpeaker` pilot dataset for v2; the two were
not merged. All v1 data-dependent artifacts remain immutable historical pilot
evidence only.

The full prior worklog was read before dataset access. It confirms that the v1
pipeline was a pilot on the old dataset and that its latest completed stage was
resumed multi-epoch ECAPA-AAM fine-tuning through epoch 4.

### Repository context inspected

- Complete root `AGENTS.md`, `.gitignore`, and this complete worklog.
- Initial clean `git status`, targeted diffs, recent commit history, root
  structure, and top-level manifest/split directory conventions.
- Targeted source, script, test, report, cache-name, manifest-name, trial-name,
  checkpoint-name, split-name, count, shape, label, hash, and path searches.
- Relevant manifest/audit, split, portable-manifest, Fbank-cache, cached
  Dataset/sampler, AAM, one-epoch/multi-epoch, SpeechBrain frontend,
  verification-trial, metric, scoring, and synthetic-test implementations.
- No quarantined v1 final-test manifest, index, shard, score, embedding, or
  evaluation artifact was opened, loaded, hashed, statted, or evaluated.

### Dataset audit method

- Deterministic complete filesystem traversal with speaker identity derived
  only from each WAV's direct parent.
- Portable Unicode-normalized forward-slash path validation, directory-depth
  and placement classification, numeric-folder checks, empty/nested/root-level
  detection, extension inventory, and Windows case-collision analysis.
- RIFF/WAVE chunk and declared-boundary inspection without decoding waveform
  sample arrays.
- Exact distributions and per-speaker/provenance summaries.
- Deterministic case-aware provenance parser and syntactic
  provenance/shard/row candidate-source parser.
- Duplicate stages: cheap metadata group; SHA-256 of bounded first/last bytes;
  complete-file SHA-256 only for matching partial-digest candidates. Because
  every file was 96,078 bytes, the bounded stage covered every byte of every
  WAV (12,091,128,066 bytes); complete SHA-256 was recomputed for eight
  duplicate candidates (768,624 bytes).
- Atomic outputs, schema/read-back validation, full count/byte reconciliation,
  and matching pre/post path-size-mtime snapshots.

### Exact findings

- Files/WAV/non-WAV: `125,847 / 125,847 / 0`.
- Total bytes: `12,091,128,066` (`11.2607405204326 GiB`).
- Top-level speaker folders/numeric folders/valid candidate speakers:
  `1,675 / 1,675 / 1,675`.
- Readable/unreadable/zero-byte/zero-frame WAVs:
  `125,847 / 0 / 0 / 0`.
- Total duration: `377,541` seconds (`104.8725` hours).
- All files were exactly depth 2. Root-level files, nested WAVs/directories,
  empty directories/speaker directories, nonnumeric folders, unsafe paths,
  normalized duplicates, case collisions, and non-WAV entries were all zero.
- All 125,847 WAVs were 16 kHz, mono, 16-bit PCM, 48,000 frames, 3.0 seconds,
  and 96,078 bytes. No non-dominant format or duration outlier was observed.
- Utterances per speaker count/min/mean/population-std/p01/p05/p25/median/p75/
  p95/p99/max:
  `1675 / 1 / 75.13253731343283 / 184.64828782053348 / 1 / 1 / 4 /
  17 / 78 / 319.29999999999995 / 576.8399999999997 / 3037`.
- Speakers with fewer than 2/3/5/10/20/50 utterances:
  `128 / 246 / 427 / 670 / 875 / 1125`; more than 100/500:
  `350 / 29`. The 128 one-utterance speakers cannot form a positive pair from
  supplied data alone; none was removed.

### Duplicate, provenance, and source-group findings

- Four exact byte-duplicate groups containing eight files, all within the same
  speaker; zero cross-speaker exact duplicate groups. Potential repeated bytes:
  `384,312`. No file was deleted or excluded.
- Provenance:
  train `86,197` files / `1,024` speakers / `258,591` seconds;
  train_small `16,128 / 430 / 48,384`;
  part `17,858 / 536 / 53,574`;
  test `5,664 / 69 / 16,992`;
  other/unparseable `0`.
- Exactly 384 speakers span more than one provenance class. Provenance was not
  treated as a final split.
- All 125,847 filenames matched the exact supported
  `aug_<provenance>-<shard>-of-<total>_<row>.wav` grammar and yielded 125,847
  singleton candidate coordinates. The syntax is high-confidence, but the
  underlying-source/augmentation semantics are unproven. The grouping is not
  strong enough to constrain a future split without authoritative lineage
  metadata.

### Integration hard-code findings and migration

`reports/integration_hardcode_inventory_v2.csv` contains 41 targeted findings:
3 must change before full manifest v2; 4 before split v2; 4 before cache v2;
1 before trials v2; 3 before baseline v2; 7 before training v2; 6 are safe
reusable production logic; and 13 are historical v1 artifacts that must remain
untouched.

Unsafe pilot assumptions include the old dataset root; provenance-derived
trusted/quarantine pools; 31,998/8,504/9,198 rows; 488 classes and labels
0..487; v1 paths and hashes; fixed `[301,80]`/48,000-sample identities; 19,528
trials; P16K2/window-8/1,000-batch epochs; pilot thresholds; and pilot
checkpoint/resume identity. The staged plan derives dataset/artifact versions,
counts, labels, feature T, cache, trials, sampler, class count, checkpoints,
baseline, and validation protocol from future approved v2 manifests/config.

Reusable logic includes the native `[B,T,80]` SpeechBrain frontend/encoder
orientation, deterministic sampler mechanics, configurable AAM/optimizer and
checkpoint guards, portable trial ownership checks, O(N log N) EER metrics,
and cosine scoring.

### Governance changes

Root `AGENTS.md` now makes VieSpeaker2.0 exclusive for v2, forbids merging or
reusing pilot identities, requires direct-parent speaker IDs and
provenance-neutral future splitting, makes WAVs/dataset root immutable,
forbids preprocessing/augmentation until approval, reserves final test,
requires separate v2 paths and manifest/config-derived values, and preserves
the cached `[B,T,80] -> mean_var_norm -> embedding_model -> [B,1,192] ->
[B,192]` model contract without feature transposition or the VoxCeleb
classifier. Unapproved exact v2 counts/configuration were not added as rules.

### Files created

- `src/dataset_audit_v2.py`
- `scripts/audit_viespeaker2_dataset.py`
- `tests/test_dataset_audit_v2.py`
- `reports/dataset_understanding_v2.md`
- `reports/dataset_understanding_v2.json`
- `reports/dataset_speaker_summary_v2.csv`
- `reports/dataset_provenance_summary_v2.csv`
- `reports/integration_hardcode_inventory_v2.csv`
- Ignored `outputs/dataset_understanding_v2/` runtime inventory,
  exact-distribution table, duplicate-group table, and audit logs

The runtime per-file CSV is explicitly named
`dataset_audit_inventory_nonproduction_v2.csv` and contains no split or label.

### Files modified

- `AGENTS.md`
- `reports/CODEX_WORKLOG.md` (this append-only entry)

No production full manifest, split, portable manifest, mapping, Fbank cache,
trial, baseline, checkpoint, environment, pretrained weight, or source audio
was modified.

### Exact commands

```text
Get-Content -Raw -LiteralPath 'C:\Users\thanh\.codex\attachments\0cadeded-81f7-4cf6-84a6-173f4d6ed5f3\pasted-text.txt'
Get-Content -Raw -LiteralPath 'AGENTS.md'
Get-Content -Raw -LiteralPath '.gitignore'
Get-Content -Raw -LiteralPath 'reports\CODEX_WORKLOG.md'
(Get-Content -LiteralPath 'reports\CODEX_WORKLOG.md') | Select-Object -Skip 0 -First 250
(Get-Content -LiteralPath 'reports\CODEX_WORKLOG.md') | Select-Object -Skip 250 -First 250
(Get-Content -LiteralPath 'reports\CODEX_WORKLOG.md') | Select-Object -Skip 500 -First 250
(Get-Content -LiteralPath 'reports\CODEX_WORKLOG.md') | Select-Object -Skip 750 -First 250
(Get-Content -LiteralPath 'reports\CODEX_WORKLOG.md') | Select-Object -Skip 1000 -First 250
(Get-Content -LiteralPath 'reports\CODEX_WORKLOG.md') | Select-Object -Skip 1250 -First 251
(Get-Content -LiteralPath 'reports\CODEX_WORKLOG.md') | Select-Object -Skip 400 -First 300
(Get-Content -LiteralPath 'reports\CODEX_WORKLOG.md') | Select-Object -Skip 700 -First 300
git status --short --branch
Get-ChildItem -Force
rg --files src scripts tests reports | Sort-Object
git log -8 --oneline --decorate
git diff -- AGENTS.md .gitignore src scripts tests reports
Get-ChildItem -LiteralPath 'manifests' -Force
Get-ChildItem -LiteralPath 'splits' -Force
rg -n -i --glob 'src/**' --glob 'scripts/**' --glob 'tests/**' --glob 'reports/*.md' --glob 'reports/*.json' --glob 'README.md' 'VieSpeaker|fbank_cache_v1|validation_trials_v1|pretrained.*baseline|portable.*v1|speaker_to_label_v1|checkpoint|final[_-]?test' .
rg -n --glob 'src/**' --glob 'scripts/**' --glob 'tests/**' --glob 'reports/*.md' --glob 'reports/*.json' --glob 'README.md' '488|31,?998|8,?504|9,?198|49,?700|1000|1_000|301|48000|48_000|P16K2|active.{0,20}window|expected.*hash|manifest.*hash|trial.*hash|num_classes|class_count|final_split|train_manifest_v1|test_manifest_v1' .
rg -n --glob 'src/*.py' --glob 'scripts/*.py' --glob 'tests/*.py' 'E:\\VieSpeaker|VieSpeaker|31_?998|8_?504|9_?198|49_?700|488|range\(488\)|0\.\.487' .
rg -n --glob 'src/*.py' --glob 'scripts/*.py' --glob 'tests/*.py' 'fbank_cache_v1|portable_manifest|train_manifest_v1|validation_manifest_v1|test_manifest_v1|speaker_to_label_v1|validation_trials_v1|pretrained.*baseline|START_CHECKPOINT|best\.pt|checkpoint_v1' .
rg -n --glob 'src/*.py' --glob 'scripts/*.py' --glob 'tests/*.py' '301|48000|48_000|FRAMES|SAMPLES|feature_shape|EXPECTED_ROWS|EXPECTED_.*HASH|SHA256|sha256|1000|1_000|P16K2|ACTIVE_SHARD|active_shard|window_size|num_classes|class_count' .
Get-Content -Raw -LiteralPath 'src\build_manifest.py'
Get-Content -Raw -LiteralPath 'src\inspect_dataset.py'
Get-Content -Raw -LiteralPath 'src\analyze_manifest.py'
Get-Content -Raw -LiteralPath 'scripts\create_speaker_split.py'
Get-Content -Raw -LiteralPath 'scripts\create_portable_manifests.py'
Get-Content -Raw -LiteralPath 'scripts\precompute_speechbrain_fbank.py'
Get-Content -Raw -LiteralPath 'tests\test_speaker_split.py'
Get-Content -Raw -LiteralPath 'tests\test_portable_manifests.py'
Get-Content -Raw -LiteralPath 'tests\test_fbank_cache.py'
.venv-cuda\Scripts\python.exe -m py_compile src\dataset_audit_v2.py scripts\audit_viespeaker2_dataset.py tests\test_dataset_audit_v2.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_dataset_audit_v2 -v
$auditDir = 'E:\SpeakerVerification\outputs\dataset_understanding_v2'; New-Item -ItemType Directory -Force -Path $auditDir | Out-Null; $process = Start-Process -FilePath 'E:\SpeakerVerification\.venv-cuda\Scripts\python.exe' -ArgumentList @('scripts\audit_viespeaker2_dataset.py','--dataset-root','E:\VieSpeaker2.0\augmented_dataset') -WorkingDirectory 'E:\SpeakerVerification' -RedirectStandardOutput "$auditDir\audit.stdout.log" -RedirectStandardError "$auditDir\audit.stderr.log" -WindowStyle Hidden -PassThru; $process.Id
$source = (Resolve-Path -LiteralPath 'outputs\dataset_understanding_v2\dataset_inventory_v2.csv').Path; $target = [System.IO.Path]::GetFullPath('E:\SpeakerVerification\outputs\dataset_understanding_v2\dataset_audit_inventory_nonproduction_v2.csv'); $allowed = [System.IO.Path]::GetFullPath('E:\SpeakerVerification\outputs\dataset_understanding_v2') + [System.IO.Path]::DirectorySeparatorChar; if (-not $source.StartsWith($allowed, [System.StringComparison]::OrdinalIgnoreCase) -or -not $target.StartsWith($allowed, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'resolved paths are outside the intended audit output directory' }; Move-Item -LiteralPath $source -Destination $target
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
Import-Csv -LiteralPath 'reports\integration_hardcode_inventory_v2.csv'
Get-Content -Raw -LiteralPath 'reports\dataset_understanding_v2.json' | ConvertFrom-Json
git diff --check
git diff -- AGENTS.md src\dataset_audit_v2.py scripts\audit_viespeaker2_dataset.py tests\test_dataset_audit_v2.py reports\dataset_understanding_v2.md reports\dataset_understanding_v2.json reports\dataset_speaker_summary_v2.csv reports\dataset_provenance_summary_v2.csv reports\integration_hardcode_inventory_v2.csv reports\CODEX_WORKLOG.md
git status --short
git check-ignore -v outputs\dataset_understanding_v2\dataset_audit_inventory_nonproduction_v2.csv outputs\dataset_understanding_v2\dataset_metadata_distributions_v2.csv outputs\dataset_understanding_v2\dataset_duplicate_groups_v2.csv
Select-String -LiteralPath reports\dataset_provenance_summary_v2.csv,reports\dataset_speaker_summary_v2.csv,reports\integration_hardcode_inventory_v2.csv,reports\dataset_understanding_v2.json -Pattern '[A-Za-z]:[\\/]'
```

### Tests, preservation, and result

- New Python `py_compile`: PASS.
- Focused synthetic audit suite: 6 passed.
- Complete repository unittest discovery: 118 passed.
- The initial focused run had one fixture-expectation failure for
  `aug_train.wav`; the fixture was corrected to the genuinely malformed
  `aug_train-.wav`, after which the focused suite passed twice. Audit
  production logic was not changed for that correction.
- Persisted CSV/JSON schema parsing and inventory read-back: PASS.
- `git diff --check`: PASS.
- Pre/post files/bytes/directories:
  `125847/12091128066/1676` and identical.
- Pre/post path-size-mtime identity SHA-256:
  `40462a26fa9fc2aee83cc8a55b02a88087e75bbe97e4af1a79925b00d081a303`
  and identical; zero snapshot errors.
- Dataset-root immediate child set matched. No dataset file was changed and no
  artifact was written inside the dataset root.
- Result: **PASS** for the audit task. This is not split approval.

### Limitations and explicitly deferred

Header inspection does not measure SNR, reverberation, clipping, perceptual
quality, VAD quality, or normalization. Exact-byte hashing does not detect
acoustically equal re-encodings. Candidate source coordinates lack
authoritative lineage semantics. Folder IDs were not checked against an
external speaker registry.

Explicitly deferred: approved `full_manifest_v2`, every split/label decision,
portable manifests v2, Fbank/cache v2, SpeechBrain/ECAPA loading, embeddings,
trials v2, baseline/scoring/EER/thresholds, sampler approval, AAM/training,
checkpoint creation/resume, all final-test access, commit, and push. No commit
or push occurred.

## 2026-07-28 — VieSpeaker2.0 Production Full Manifest v2

### Scope and approved inputs

Created only the deterministic, unsplit VieSpeaker2.0 production full manifest
v2 and its immutable identity metadata. The approved inputs were the read-only
VieSpeaker2.0 dataset root, the nonproduction v2 audit inventory, and the
approved Dataset Understanding v2 Markdown/JSON reports. The old pilot dataset
was not merged or consulted. No split generation followed this task.

Approved input identities:

- Audit inventory SHA-256:
  `6f33cb9510b00f3503af41246349b921d000779b25de9715b9ea68aed1df6cd5`
- Audit Markdown SHA-256:
  `79bd00b9f922b5b3a2d492cf910a0f7d7e6e4c6b1403b4b21c7e1aab15c4493a`
- Audit JSON SHA-256:
  `305a641ac91f4e7fc41c389bf3818b810f59fea5ec18a7722046407582bc9af2`
- Dataset snapshot identity SHA-256:
  `40462a26fa9fc2aee83cc8a55b02a88087e75bbe97e4af1a79925b00d081a303`

### Schema, artifacts, and policy

The exact manifest columns are:

```text
audio_path,speaker_id,filename,provenance,provenance_parse_status,sample_rate_hz,channel_count,sample_width_bytes,bits_per_sample,wav_encoding,frame_count,duration_seconds,file_size_bytes,wav_status,duplicate_group,candidate_source_group,source_group_parse_status,manifest_version
```

Artifacts:

- `manifests/v2/full_manifest_v2.csv`
- `manifests/v2/full_manifest_v2_identity.json`
- `reports/full_manifest_v2_summary.md`
- `reports/full_manifest_v2_summary.json`

Rows are ordered by numeric direct-parent `speaker_id`, then ordinal
dataset-relative `audio_path`. Paths use `/` and contain no absolute prefix,
drive, backslash, or parent traversal. Filenames do not define speaker
identity. Provenance and candidate source groups are descriptive/syntactic
only and do not create a split or authoritative lineage.

All 128 singleton speakers were preserved. All four audited groups
`dup_000001` through `dup_000004` and all eight duplicate files were preserved;
no representative was chosen and no file was removed. All groups remain
within-speaker.

### Exact counts and identities

- Rows / unique paths / readable WAVs: `125847 / 125847 / 125847`
- Speakers: `1675`
- Bytes / duration seconds: `12091128066 / 377541`
- Provenance `train / train_small / part / test`:
  `86197 / 16128 / 17858 / 5664`
- Duplicate groups / duplicate files / cross-speaker groups: `4 / 8 / 0`
- Singleton speakers: `128`
- Manifest SHA-256:
  `26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f`
- Identity-file SHA-256:
  `f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e`

### Files created

- `src/full_manifest_v2.py`
- `scripts/create_full_manifest_v2.py`
- `tests/test_full_manifest_v2.py`
- `manifests/v2/full_manifest_v2.csv`
- `manifests/v2/full_manifest_v2_identity.json`
- `reports/full_manifest_v2_summary.md`
- `reports/full_manifest_v2_summary.json`
- Ignored reproducibility pair under
  `outputs/full_manifest_v2_reproduction_v2/`

### Files modified

- `reports/CODEX_WORKLOG.md` (this append-only entry)

Pre-existing dirty and untracked work was preserved.

### Exact commands

```text
.venv-cuda\Scripts\python.exe -m py_compile src\full_manifest_v2.py scripts\create_full_manifest_v2.py tests\test_full_manifest_v2.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_full_manifest_v2 -v
.venv-cuda\Scripts\python.exe scripts\create_full_manifest_v2.py --dataset-root "E:\VieSpeaker2.0\augmented_dataset" --audit-inventory outputs\dataset_understanding_v2\dataset_audit_inventory_nonproduction_v2.csv --output-dir manifests\v2
.venv-cuda\Scripts\python.exe scripts\create_full_manifest_v2.py --dataset-root "E:\VieSpeaker2.0\augmented_dataset" --audit-inventory outputs\dataset_understanding_v2\dataset_audit_inventory_nonproduction_v2.csv --output-dir outputs\full_manifest_v2_reproduction_v2 --reference-dir manifests\v2
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
git diff --check
```

### Tests, reproducibility, preservation, and result

- Python compilation: PASS.
- Focused synthetic suite: 12 passed.
- Complete repository unittest discovery: 130 passed.
- `git diff --check`: PASS.
- Finalized CSV and JSON schema/value/hash read-back: PASS.
- The independent verification build was byte-identical for both manifest and
  identity JSON.
- CSV uses UTF-8 without BOM, LF records, fixed columns, and six-decimal
  duration; JSON uses UTF-8 without BOM, LF, sorted keys, two-space indentation,
  and no timestamp.
- Both builds reported identical pre/post dataset snapshot identities and
  `dataset_preservation_match: true`; no artifact was written under the dataset
  root and no source WAV was changed.
- Result: **PASS** for the production full manifest v2 only. This does not
  approve a speaker split.

Deferred: every train/validation/test split and label decision, portable split
manifests v2, Fbank/cache v2, SpeechBrain/ECAPA loading, embeddings, trials,
baseline/scoring/EER/thresholds, sampler benchmarking, training, checkpoint
work, and all final-test access. No commit or push occurred.

## 2026-07-29 — VieSpeaker2.0 Final Speaker-Disjoint Split Package v2

### Exact scope and approved inputs

Created only the approved final speaker-disjoint v2 split, portable manifests,
train label mapping, policy/identity bindings, and versioned reports. The
authoritative inputs passed the mandatory fail-closed gate before any split
artifact was written:

- `manifests/v2/full_manifest_v2.csv`
  SHA-256 `26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f`
- `manifests/v2/full_manifest_v2_identity.json`
  SHA-256 `f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e`

The existing dirty and untracked Dataset Understanding v2 and full-manifest v2
work was preserved. Historical v1 outputs were not modified, and quarantined
v1 final-test artifacts were not recursively enumerated or opened.

### Deterministic policy and algorithm

- Split seed: `20260729`.
- Evaluation eligibility: at least 20 utterances.
- Train eligibility: at least 2 utterances.
- Singleton policy: excluded from portable train/validation/test manifests,
  retained in the authoritative full manifest.
- Evaluation buckets and capacities:
  `20-49=250`, `50-99=199`, `100-199=202`, `200-499=120`,
  `500+=29`.
- Exact largest-remainder selection quotas in configured bucket order:
  `63 / 50 / 50 / 30 / 7`.
- Within each bucket, selection uses ascending
  `SHA256(UTF8("<seed>:<speaker_id>"))`, then numeric speaker ID.
- Exact dynamic programming assigns the selected 200 speakers by priority:
  exact `100/100` counts; per-bucket count difference at most one; minimum
  absolute utterance difference; minimum absolute duplicate-file difference;
  configured-bucket/SHA-256 stable tie break.
- Filename provenance and candidate source groups do not influence assignment.

Validation/test bucket counts are:

```text
20-49:    32 / 31
50-99:    25 / 25
100-199:  25 / 25
200-499:  15 / 15
500+:      3 / 4
```

Validation and test each contain exactly 15,355 utterances, so the total
utterance difference is zero. Their duplicate-file counts are `0 / 4`.

### Exact split results

| Split | Speakers | Rows | Duration seconds | Duplicate groups | Duplicate files |
|---|---:|---:|---:|---:|---:|
| train | 1,347 | 95,009 | 285,027 | 2 | 4 |
| validation | 100 | 15,355 | 46,065 | 0 | 0 |
| test | 100 | 15,355 | 46,065 | 2 | 4 |
| excluded | 128 | 128 | 384 | 0 | 0 |

All 1,675 speakers and all 125,847 rows reconcile exactly. Speaker
intersections are empty. All 128 excluded speakers have one utterance; every
train speaker has at least two; every validation/test speaker has at least
twenty.

Train labels are assigned in numeric speaker-ID order and are exactly
contiguous `0..1346`. Validation and test labels are all `-1`. Portable paths
are dataset-root-relative and use `/`.

All four exact duplicate groups and all eight duplicate files remain in the
owning speaker's single final split. Nothing was deduplicated and no preferred
representative was selected. Later validation-trial generation must reject a
positive pair whose two files share the same non-empty `duplicate_group`.

### Created artifacts and SHA-256

- `splits/v2/speaker_split_v2.csv`:
  `cbbcdcd4d3561ff2470a6cd713187cbb612939e643e5bc8cf4ed25504539f87e`
- `splits/v2/speaker_split_v2_identity.json`:
  `87d2a542ae1716f5d143e27835bf0478413e672cfa131462c0410808498c3c67`
- `splits/v2/split_policy_v2.json`:
  `3e414836b4fe307841810cffa56c3f0040d0c662d265d276d73a7d00a16a5d10`
- `manifests/portable_v2/train_manifest_v2.csv`:
  `f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b`
- `manifests/portable_v2/validation_manifest_v2.csv`:
  `9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc`
- `manifests/portable_v2/test_manifest_v2.csv`:
  `14c782fa36d9c23040dd7a7fce26d91b9a57a230bdeebd19658dbcb2ee8364eb`
- `manifests/portable_v2/speaker_to_label_v2.json`:
  `9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8`
- `manifests/portable_v2/portable_manifests_v2_identity.json`:
  `29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5`
- `reports/speaker_split_v2_summary.md`
- `reports/speaker_split_v2_summary.json`
- `src/speaker_split_v2.py`
- `scripts/create_speaker_split_package_v2.py`
- `tests/test_speaker_split_v2.py`
- Ignored independent reproduction package under
  `outputs/speaker_split_v2_reproduction/`

### Files modified

- `AGENTS.md`, after package validation only, with approved v2 paths, speaker
  counts, singleton/label semantics, identity binding, and final-test
  quarantine.
- `reports/CODEX_WORKLOG.md` by this single append-only entry.

### Exact commands

```text
.venv-cuda\Scripts\python.exe -m py_compile src\speaker_split_v2.py scripts\create_speaker_split_package_v2.py tests\test_speaker_split_v2.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_speaker_split_v2 -v
.venv-cuda\Scripts\python.exe scripts\create_speaker_split_package_v2.py
$verificationRoot = 'outputs\speaker_split_v2_reproduction'; New-Item -ItemType Directory -Path $verificationRoot | Out-Null; .venv-cuda\Scripts\python.exe scripts\create_speaker_split_package_v2.py --output-root $verificationRoot --reference-root .
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
git diff --check
```

### Tests, reproducibility, and preservation

- `py_compile`: PASS.
- Focused synthetic split suite: 13 passed.
- Complete repository unittest discovery: 143 passed.
- Cross-process runs with `PYTHONHASHSEED=1` and `999`: byte-identical
  assignment output.
- Transactional publication rollback test: PASS.
- Final CSV/JSON read-back, schemas, hashes, paths, labels, row/speaker totals,
  duplicate ownership, and disjointness: PASS.
- Independent second build: all eight required artifacts byte-identical.
- Deterministic formatting: UTF-8 without BOM, LF, fixed CSV columns,
  six-decimal durations, sorted JSON keys, two-space indentation, and no
  identity timestamps.
- Full manifest and its identity matched before/after hashes.
- The builder has no dataset-root argument and did not open source WAV content;
  no source WAV was changed.
- `git diff --check`: PASS.

### Final-test quarantine, result, and deferred work

The final-test manifest was created and metadata-validated, then became
immutable/quarantined after the independent hash comparison. No test WAV
content was opened. No test Fbank, trials, model load, embeddings, scores, EER,
threshold, or accuracy were produced.

Result: **PASS** for the final speaker-disjoint split package v2 only.

Deferred: Fbank/cache v2, validation and final-test trials, SpeechBrain/ECAPA
loading, embeddings, baseline/scoring/EER/thresholds, sampler benchmarks,
training, checkpoints, every final-test content/evaluation action, commit, and
push. No commit or push occurred.

## 2026-07-29 — VieSpeaker2.0 Train/Validation Fbank Cache and Cached ECAPA Preflight v2

### Exact scope and approved identities

Performed only the approved v2 train/validation raw SpeechBrain Fbank cache,
full cache validation, versioned lazy Dataset/DataLoader integration, and one
cached train plus one cached validation ECAPA CUDA preflight. No trial, score,
metric, sampler, AAM, optimizer, training, checkpoint, or final-test stage
followed.

The fail-closed gate matched all nine supplied identities before source access
and again after extraction:

- Full manifest:
  `26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f`
- Full-manifest identity:
  `f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e`
- Speaker split / identity:
  `cbbcdcd4d3561ff2470a6cd713187cbb612939e643e5bc8cf4ed25504539f87e` /
  `87d2a542ae1716f5d143e27835bf0478413e672cfa131462c0410808498c3c67`
- Split policy:
  `3e414836b4fe307841810cffa56c3f0040d0c662d265d276d73a7d00a16a5d10`
- Portable-package identity:
  `29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5`
- Train / validation manifests:
  `f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b` /
  `9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc`
- Train label mapping:
  `9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8`

### Feature contract and cache results

The real train-only preflight measured raw `compute_features` output as
`[T,80] = [301,80]`, float32, finite, pre-normalization, and non-transposed.
The required CUDA batch-size-64 preflight produced `[64,301,80]` from
`[64,48000]`; no batch-size fallback was required.

The single production process used `cuda:0`, extraction batch size 64, and
shard size 256. It wrote 95,009 train rows in 372 shards and 15,355 validation
rows in 60 shards: 110,364 utterances and 10,638,641,568 shard bytes total.
Extraction took 206.745 seconds. Every shard was read back and every tensor,
label, path, speaker, split, manifest row index, shape, dtype, and finite-value
contract passed. Train labels cover exactly `0..1346`; validation labels are
all `-1`; global paths are unique; no unexpected shard or split directory
exists.

Config and index plans generated twice were byte-identical. Two fixed samples
per split were re-extracted in their original production batch contexts and
matched the cache exactly with maximum absolute difference `0.0`.

### Dataset, DataLoader, and cached ECAPA

`CachedFbankDataset` now detects v1 or v2 dynamically. V2 validation binds the
completion identity, config hash, index hashes, dynamic `[T,80]`, authoritative
row/class/label metadata, train/validation allowlist, deterministic
manifest-row/shard alignment, lazy CPU loading, bounded LRU behavior, and
portable paths. V1 config/index/shard/sample/collate behavior remains
compatible.

Real-cache DataLoader checks passed for each split: two sequential batches
with `num_workers=0` and one two-sample batch with `num_workers=2`, preserving
tensor, label, path, metadata, `dataset_index`, and `manifest_row_index`
alignment.

One train and one validation batch of four followed:

```text
cached Fbank [4,301,80]
-> mean_var_norm [4,301,80]
-> embedding_model [4,1,192]
-> squeeze(1) [4,192]
```

All tensors were CUDA float32 and finite. Train labels were valid and
validation labels were `-1`. Evaluation/inference mode, frozen parameters,
absent gradients, unchanged parameters, and no optimizer were verified.
Forward hooks recorded zero `compute_features` calls and zero pretrained
classifier calls.

### Cache identities

- Config:
  `ec71959ec64361038991e760e772d11bf1779e1b505a892dff364cf45aaeb018`
- Cache identity:
  `1a2d6af777311f687e887575bfaf20915ed0409fd2e05b2f1ac232d43cd0b8c8`
- Train index:
  `e20c320fc5842502a26684023bb307a7b2afa27a14a3cf1130fdffe31e85d4b9`
- Validation index:
  `1d3a95e95aaa5b13e6614c077fbbdf10f7c208c79970c2a85bdd461193b9f585`
- Runtime result:
  `71fa3164c0f2adf2eb435defa9559b1379bcb566942119264657deed4fbc15b9`
- Cached ECAPA result:
  `4d634d87a1c8fa614effb8f520ffa6265975336bebec50d53db67bd7bdd55f02`

### Files created

- `scripts/precompute_speechbrain_fbank_v2.py`
- `scripts/smoke_test_cached_ecapa_v2.py`
- `tests/test_fbank_cache_v2.py`
- `reports/fbank_cache_v2_summary.md`
- `reports/fbank_cache_v2_summary.json`
- Ignored `outputs/fbank_cache_v2/` config, completion identity, indexes,
  train/validation shards, runtime results, cached ECAPA result, and logs

### Files modified

- `src/cached_fbank_dataset.py`
- `AGENTS.md`
- `reports/CODEX_WORKLOG.md` by this one append-only entry

All pre-existing dirty and untracked work was preserved.

### Exact commands and validation

```text
.venv-cuda\Scripts\python.exe -m py_compile scripts\precompute_speechbrain_fbank_v2.py src\cached_fbank_dataset.py scripts\smoke_test_cached_ecapa_v2.py tests\test_fbank_cache_v2.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_fbank_cache_v2 -v
.venv-cuda\Scripts\python.exe scripts\precompute_speechbrain_fbank_v2.py --dataset-root "<runtime supplied approved v2 root>" --device cuda:0 --batch-size 64 --shard-size 256 --cache-dir outputs\fbank_cache_v2
.venv-cuda\Scripts\python.exe scripts\smoke_test_cached_ecapa_v2.py --cache-dir outputs\fbank_cache_v2 --device cuda:0 --batch-size 4
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
git diff --check
```

Long extraction stdout/stderr was redirected to ignored files inside the v2
cache. Python compilation passed, 13 focused v2 cache tests passed, all 156
repository tests passed, and `git diff --check` passed. A pre-final full-suite
run exposed three v2-only metadata keys in legacy v1 samples; they were
restricted to v2, after which the full suite passed.

### Preservation, quarantine, result, and deferred work

Approved train/validation source path-size-mtime state matched before/after for
all 110,364 rows. No source WAV, environment, or pretrained weight was
modified. No absolute dataset root was persisted.

The final-test manifest was not opened, statted, hashed, parsed, or loaded. No
final-test or excluded WAV was accessed, and no `test` cache directory, index,
shard, feature, embedding, or trial exists.

Result: **PASS** for the v2 train/validation Fbank cache and cached ECAPA
preflight only.

Deferred: trials, scores, EER, thresholds, accuracy, sampler benchmarks, AAM,
optimizer/training work, checkpoints, all final-test content/evaluation
actions, commit, and push. No commit or push occurred.

## 2026-07-29 18:48:52 +07:00 - VieSpeaker2.0 Training Readiness v2

### Goal and exact scope

Perform only the approved VieSpeaker2.0 training-readiness stage: validate and
freeze the requested hybrid sampler, generate fixed validation verification
trials, and evaluate the untouched pretrained ECAPA validation baseline. No
comparative sampler benchmark, AAM-Softmax, optimizer, training, checkpoint,
augmentation, final-test operation, commit, or push was performed.

### Approved input bindings

All required hashes were validated before use and remained unchanged:

- Portable package identity:
  `29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5`
- Train manifest:
  `f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b`
- Validation manifest:
  `9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc`
- Train label mapping:
  `9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8`
- Cache config:
  `ec71959ec64361038991e760e772d11bf1779e1b505a892dff364cf45aaeb018`
- Cache identity:
  `1a2d6af777311f687e887575bfaf20915ed0409fd2e05b2f1ac232d43cd0b8c8`
- Train cache index:
  `e20c320fc5842502a26684023bb307a7b2afa27a14a3cf1130fdffe31e85d4b9`
- Validation cache index:
  `1d3a95e95aaa5b13e6614c077fbbdf10f7c208c79970c2a85bdd461193b9f585`

Train manifest/cache alignment passed for 95,009 rows, 1,347 speakers, labels
0..1346, 372 shards, four rows in two nonempty duplicate groups, and zero
metadata mismatches. Validation alignment passed for 15,355 rows, 100
speakers, labels all `-1`, 60 shards, and zero metadata mismatches.

### Approved sampler and validation

`configs/v2/training_sampler_v2.json` approves only
`HybridShardAwareSpeakerBatchSampler` with P=16, K=2, batch size 32, active
shard window 8, seed 20260729, 2,969 batches and 95,008 logical selections per
epoch, workers 0, Dataset LRU size 8, and `validate_finite=False` for the
already fully validated immutable cache.

Full metadata-only plans for epochs 0 and 1 passed exact P x K, train-only
selection, all-speaker coverage, duplicate-group separation, metadata
immutability, same-epoch reproduction, and different-epoch change checks.
Epoch 0/1 plan hashes are
`010e80f042ae91f1c890045d12833005359833466ba6d5a4b5fc70e5c8f04a68`
and
`c5ff9c7fca4c88541c47738706b1852293e6727da9bb420350e75379179bba0f`.
The combined sampler-plan identity is
`b11a97fd45f11b8f71fa980ebd26cb335bac0f3bcdfb781e8f87ced535606e4f`
and reproduced under `PYTHONHASHSEED=1` and `987654`.

Epoch 0/1 LRU(8) simulated hit rates were
`0.8150997810710677 / 0.8136683226675648`; duplicate-group safeguards
rejected `1 / 2` candidates and produced zero conflicts. Speaker batch
exposure mean was `35.26651818856718` for both epochs, with population CV
`0.2550267930232451 / 0.23782636723207598`. The first 16 real cache batches
only (512 samples) passed exact shape `[32,301,80]`, float32, P x K, train-only,
duplicate-group, LRU-bound, and metadata-preservation checks. No comparative
benchmark was run.

Sampler config SHA-256:
`e59d3794371095acddcee17218ff99fef4393cb15a7f7176c2d7680120436899`.

### Fixed validation trial protocol

The deterministic trial package contains exactly 10,000 positive and 10,000
negative trials over all 100 approved validation speakers. Positives use 100
stable-SHA-ranked canonical unordered path pairs per speaker and reject a pair
sharing the same nonempty duplicate group. Negatives use 200 round-robin
rounds: two complete 99-round cycles plus the first two rounds of cycle three,
giving exactly 200 participations per speaker. Utterances use least-used
selection with stable SHA-256 tie breaks. Trial IDs and canonical unordered
path pairs are unique, and every path/speaker ownership check passed.

The CSV, config, and identity reproduced byte-for-byte both in-process and in
an independent process with `PYTHONHASHSEED=987654`. They contain no timestamp
or absolute local path.

- Trial CSV:
  `11bec5ff0a0a4ca4930e2664bdc391388a9a677795afaefe5de3fee2d0d39e3d`
- Trial config:
  `9e725ce006ae522f0f0274739b75e9aee7ebb302db331fead73c79cdc326822e`
- Trial identity:
  `09b55236ad3f80537e1517a5efa7ad1d7a7efc3454f6b56d9ba62af83cc6bf73`

### Untouched pretrained ECAPA validation baseline

The baseline ran sequentially on CUDA 0 with batch size 64 and workers 0 using
only:

```text
cached Fbank [B,301,80]
-> mean_var_norm [B,301,80]
-> embedding_model [B,1,192]
-> squeeze(1) [B,192]
```

It stored 15,355 ordered finite float32 CPU embeddings shaped `[15355,192]`
and exactly 20,000 finite cosine scores. Hooks recorded zero
`compute_features` calls and zero pretrained classifier calls. Parameters were
frozen and unchanged, gradients were absent, and no transpose, waveform
frontend, optimizer, AAM-Softmax, or training step was used. Tensor save/load
maximum absolute differences were zero.

Grouped, tie-aware O(N log N) metrics produced interpolated EER `0.1257`
(`12.57%`) at threshold `0.28827327489852905`. This crossing is also an actual
empirical threshold: FAR and FRR are both `0.1257`. With acceptance semantics
`score >= threshold`, TP/TN/FP/FN are `8743/8743/1257/1257`; accuracy,
precision, recall, and F1 are all `0.8743`.

Same-speaker score min/mean/std/median/max:
`-0.2114696353673935 / 0.47219380933633076 /
0.15749554871675106 / 0.49064262211322784 / 0.9752877950668335`.
Different-speaker:
`-0.2100071758031845 / 0.1534788316947641 /
0.11294603164726892 / 0.14373768866062164 / 0.602975070476532`.

Embedding extraction took `72.23180550000052` seconds at
`212.57948480880614` utterances/second; scoring took
`0.9620833999997558` seconds and metrics `0.0705762000006871` seconds.
Peak allocated/reserved VRAM was
`2567059456 / 3166699520` bytes on the NVIDIA GeForce RTX 3050 Laptop GPU.

Runtime baseline identity:
`5000ddabe804f4cfd7c905c3ed51bc8fd1d0273822ca2091019da923f387974c`.
Embedding and score artifact hashes:
`0851324f9f7b2187448f8fd5354bcca2aef82489c29f38bd84ab303c476fbfca`
and
`3945764d96cd41c28c2aa4c628826480696619433a5a28db485827df7161b95e`.

### Files created

- `configs/v2/training_sampler_v2.json`
- `manifests/verification_v2/validation_trials_v2.csv`
- `manifests/verification_v2/validation_trials_config_v2.json`
- `manifests/verification_v2/validation_trials_identity_v2.json`
- `src/training_readiness_v2.py`
- `src/verification_v2.py`
- `scripts/prepare_training_readiness_v2.py`
- `scripts/generate_validation_trials_v2.py`
- `scripts/evaluate_pretrained_validation_baseline_v2.py`
- `tests/test_training_readiness_v2.py`
- `tests/test_verification_v2.py`
- `reports/training_sampler_v2.json`
- `reports/training_sampler_v2.md`
- `reports/validation_trials_v2.json`
- `reports/validation_trials_v2.md`
- `reports/pretrained_ecapa_validation_baseline_v2.json`
- `reports/pretrained_ecapa_validation_baseline_v2.md`
- Ignored `outputs/pretrained_ecapa_validation_baseline_v2/` embeddings,
  scores, identity, runtime/configuration, and redirected logs
- Ignored `outputs/validation_trials_v2_reproduction/` independent
  reproduction artifacts

### Files modified

- `src/cached_fbank_samplers.py` for duplicate-safe P x K selection and
  rejection accounting while preserving v1 behavior
- `AGENTS.md` with the approved sampler/trial/baseline governance
- `reports/CODEX_WORKLOG.md` by this one append-only entry

All pre-existing dirty and untracked work, including the prior v2 cache task's
`src/cached_fbank_dataset.py` change, was preserved.

### Commands and tests

```text
.venv-cuda\Scripts\python.exe -m py_compile src\cached_fbank_samplers.py src\training_readiness_v2.py src\verification_v2.py scripts\prepare_training_readiness_v2.py scripts\generate_validation_trials_v2.py scripts\evaluate_pretrained_validation_baseline_v2.py tests\test_training_readiness_v2.py tests\test_verification_v2.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_cached_fbank_samplers tests.test_verification_trials tests.test_verification_metrics tests.test_verification_baseline tests.test_training_readiness_v2 tests.test_verification_v2 -v
.venv-cuda\Scripts\python.exe scripts\prepare_training_readiness_v2.py
PYTHONHASHSEED=1/987654 .venv-cuda\Scripts\python.exe scripts\prepare_training_readiness_v2.py --plan-hash-only
.venv-cuda\Scripts\python.exe scripts\generate_validation_trials_v2.py
PYTHONHASHSEED=987654 .venv-cuda\Scripts\python.exe scripts\generate_validation_trials_v2.py --output-dir outputs\validation_trials_v2_reproduction
.venv-cuda\Scripts\python.exe scripts\evaluate_pretrained_validation_baseline_v2.py
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

Nine new focused tests, 36 focused/regression tests, and all 165 repository
tests passed. Cross-process sampler and trial determinism under different
Python hash seeds passed.

### Preservation, quarantine, result, and deferred work

Approved manifests, cache configuration/identity/indexes, cache shards,
source WAVs, environments, pretrained weights, checkpoints, historical
reports, and prior artifacts were not modified. No absolute dataset root was
persisted.

The final-test manifest was not opened, statted, hashed, parsed, or loaded. No
final-test audio was opened, no final-test feature was extracted, no
final-test trial was generated, and no final-test evaluation occurred.

Result: **PASS** for the requested sampler approval, fixed validation trial
package, and untouched pretrained ECAPA validation baseline only.

Deferred: AAM-Softmax, optimizer creation, all training and checkpoints,
augmentation, comparative sampler benchmarking, final-test access/trials/
evaluation, commit, and push. No commit or push occurred.

## 2026-07-29 19:55:47 +07:00 - VieSpeaker2.0 ECAPA-AAM CUDA preflight, production epoch 0, and fixed validation

Performed only the approved v2 CUDA preflight, one production epoch zero,
fresh-object checkpoint roundtrip, and fixed validation task. No epoch-one
training, scheduling, augmentation, gradient clipping, early stopping,
threshold tuning, final-test work, commit, or push occurred.

### Approved inputs and implementation

Validated the approved portable package, train/validation manifests, train
label mapping, cache config/identity/indexes, sampler config and exact plans,
fixed validation trial package, and pretrained validation baseline before and
after the run. Their SHA-256 identities remained:

- Portable package: `29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5`
- Train/validation manifests: `f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b` / `9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc`
- Train label mapping: `9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8`
- Cache config/identity: `ec71959ec64361038991e760e772d11bf1779e1b505a892dff364cf45aaeb018` / `1a2d6af777311f687e887575bfaf20915ed0409fd2e05b2f1ac232d43cd0b8c8`
- Train/validation cache indexes: `e20c320fc5842502a26684023bb307a7b2afa27a14a3cf1130fdffe31e85d4b9` / `1d3a95e95aaa5b13e6614c077fbbdf10f7c208c79970c2a85bdd461193b9f585`
- Sampler config: `e59d3794371095acddcee17218ff99fef4393cb15a7f7176c2d7680120436899`
- Trial CSV/config/identity: `11bec5ff0a0a4ca4930e2664bdc391388a9a677795afaefe5de3fee2d0d39e3d` / `9e725ce006ae522f0f0274739b75e9aee7ebb302db331fead73c79cdc326822e` / `09b55236ad3f80537e1517a5efa7ad1d7a7efc3454f6b56d9ba62af83cc6bf73`
- Baseline identity: `5000ddabe804f4cfd7c905c3ed51bc8fd1d0273822ca2091019da923f387974c`
- Combined/epoch-0/epoch-1 sampler plans: `b11a97fd45f11b8f71fa980ebd26cb335bac0f3bcdfb781e8f87ced535606e4f` / `010e80f042ae91f1c890045d12833005359833466ba6d5a4b5fc70e5c8f04a68` / `c5ff9c7fca4c88541c47738706b1852293e6727da9bb420350e75379179bba0f`

Created `src/ecapa_one_epoch_v2.py`,
`scripts/run_ecapa_aam_one_epoch_v2.py`,
`tests/test_ecapa_one_epoch_v2.py`, and the tracked
`reports/ecapa_aam_one_epoch_v2.json` and `.md` reports. Modified
`src/verification_v2.py` to report TPR, specificity, and TNR, and updated
`AGENTS.md` only after the complete task passed.

The run used the pretrained `speechbrain/spkrec-ecapa-voxceleb` cached-feature
path, 1,347 train classes, AAM margin `0.2` and scale `30`, P=16/K=2 logical
batches of 32, physical microbatches of 4 with 8-way accumulation, AdamW
learning rates `1e-5` for ECAPA and `1e-3` for AAM, weight decay `1e-4`,
FP16 ECAPA with FP32 AAM/loss, and GradScaler initial scale 128. All
BatchNorm modules remained in evaluation mode with affine parameters
trainable. No pretrained VoxCeleb classifier was used.

### CUDA preflight and epoch zero

The isolated train-only CUDA preflight used fresh disposable objects for
exactly two optimizer updates. Losses were `15.970597863197327` and
`15.634921431541443`. ECAPA/AAM maximum parameter deltas were
`2.002716064453125e-05` / `0.002001367509365082`, with aggregate deltas
`287.1260554654291` / `402.4821472167969`. BatchNorm buffers were exact,
all optimizer state was finite, no scaler update was skipped, validation was
not run, and no production checkpoint was written. Peak allocated/reserved
CUDA memory was `778252800 / 880803840` bytes. These objects were discarded
before fresh production construction.

Production completed exactly 2,969 optimizer updates and 95,008 logical
selections, then stopped before epoch one. The 2,969 logical losses had
first/final/mean/min/max values
`15.970597863197327 / 0.5739374789409339 / 3.0659153226259708 /
0.07522520795464516 / 16.667139291763306`. Training took
`2103.319411900001` seconds; peak allocated/reserved CUDA memory was
`776417792 / 891289600` bytes. All gradients, parameters, and optimizer
states were finite, all updates advanced, BatchNorm buffers remained exact,
and the log contained the required 100-step cadence plus step 2,969.

### Checkpoints and fresh-object roundtrip

Atomic checkpoints were written with no temporary residue:

- `last.pt`: `68665317ae5d5279593c87ae985be60198e3a410bf768ae901db90bf0d7fbcc4`
- `epoch_000.pt`: `62ec327788fb349c08a96eebb5259d4e2879747cf1a00278949bfa5bb068da10`
- `best.pt`: `19cfd3482172ce482d21b65915fd347aeeac4ee1e51bc61e59484f46a31a38db`

Independent readback confirmed exact upstream and sampler bindings, AAM
shape `[1347, 192]`, AdamW step 2,969 for every state entry, cursor
epoch 1/batch 0/global step 2,969, and `epoch_1_started=false`. Fresh CPU
SpeechBrain, AAM, AdamW, and GradScaler objects exactly restored embedding,
normalizer, AAM, optimizer, scaler, DataLoader generator, RNG, BatchNorm
policy/buffers, and cursor state. A fresh sampler regenerated the approved
epoch-one plan hash without taking an optimizer step or starting epoch one.

### Fixed validation and baseline comparison

Sequential, non-augmented validation produced exactly 15,355 CPU float32
embeddings of shape `[15355, 192]`, then scored exactly 20,000 fixed trials
(10,000 positive and 10,000 negative). It made zero AAM,
`compute_features`, or pretrained-classifier calls. Embedding extraction
took `79.5039850000012` seconds at `193.13497304568781`
utterances/second; scoring and metrics took `1.0520947000004526` and
`0.0977578000001813` seconds. Peak validation allocated/reserved CUDA
memory was `2577747968 / 3196059648` bytes.

Interpolated EER was `0.064` (`6.4%`) at threshold
`0.16190975904464722`; this was also the empirical threshold, with
FAR=FRR=`0.064`. TP/TN/FP/FN were `9360/9360/640/640`; accuracy,
precision, recall/TPR, specificity/TNR, and F1 were all `0.936`.
Same-speaker score min/mean/std/median/max was
`-0.15145419538021088 / 0.428650140974205 /
0.15662893391829685 / 0.4465496391057968 / 0.9826944470405579`.
Different-speaker was
`-0.3604694604873657 / 0.014766331464692485 /
0.09456113088666378 / 0.01176312891766429 / 0.5310559272766113`.
Direct independent recomputation from the stored scores reproduced the
confusion matrix and FAR/FRR exactly.

Against baseline EER `0.1257`, the signed and absolute changes were
`-0.061700000000000005` and `0.0617` (percentage-point change `-6.17`,
relative change `-0.4908512330946699`), classified `IMPROVED`.

### Artifact identities, tests, and preservation

Output hashes beyond the checkpoints:

- CUDA preflight: `91ff16a0534184337605e61db21b9bc8bc54982db8b832d37668b60d3e9e4b74`
- Training log/losses: `d5b265af5d5abd0920d0af27a4ad43ab4965f3af88a7032ea6883c19af57a162` / `e3bda643bc488cc7dbfd3c90613f015dfa5daa9d9507423851a07131d6d79a89`
- Validation embeddings/scores/metrics: `2702d195f901a7c3c456f6f7590e8a65d3dd56aef339ce8717808e0816fd5b80` / `4447e0e24924f3c73efee2db5507772807f6ee41af730e870ea71006d3978716` / `841ae9be2c6149a5c247c1cbf382064930037399a319d1a40b96e0c78a0cacc4`
- Runtime: `f3e4f678dc0dee858441d969be892bb3dc23efc1b07fe7a293bcd56859e28758`
- Tracked JSON/Markdown reports: `ffea2e9b706404a75de40ce5fb68579d2890b351e38243a79d74c425f261afb3` / `77219ac68ebc1359025ab090a73d9f9f167cd9140447d68628c77484fce8ed2b`

Python compilation passed. The new module's 12 tests passed; 51 focused and
regression tests passed; the complete CUDA-environment repository suite
passed all 177 tests. `git diff --check` passed. JSON runtime/report and
checkpoint identities are portable and contain no absolute local paths.

Source WAVs, approved manifests, cache config/identity/indexes/shards,
environments, pretrained weights, historical reports, and pre-existing
checkpoints were not modified. The stable `.venv-cuda` environment was not
changed. The final-test manifest was not opened, statted, hashed, parsed, or
loaded; no final-test audio, cache, embeddings, trials, threshold
application, or evaluation occurred.

Result: **PASS** for the requested v2 CUDA preflight, production epoch zero,
checkpoint roundtrip, and fixed validation only.

Deferred: epoch-one training, scheduler/warmup, augmentation, gradient
clipping, early stopping, threshold tuning, final-test access/trials/
evaluation, commit, and push. No commit or push occurred.

## 2026-07-30 - VieSpeaker2.0 AMP overflow recovery and final multi-epoch completion v2

The preceding multi-epoch attempt failed closed during epoch 2 before batch
2,532 after detecting one non-finite ECAPA gradient. Its runtime, failure
record, reports, epoch-one checkpoint, and latest valid atomic checkpoint were
preserved as historical evidence. Recovery bound the exact `last.pt` SHA-256
`3b8d613012ee8340a30ad264729c83ea97501b793e6c97e1d0ed78555e5c96a0`
at epoch 2/batch position 2,031/global step 7,969, retained a byte-identical
copy, and removed only five uncommitted scalar-log suffix rows for resumed
steps 5,100–5,500.

The approved execution-only remediation allowed one retry of the exact same
materialized logical batch at half GradScaler scale when forward tensors,
loss, parameters, and optimizer state were finite and non-finiteness appeared
only in gradients. The historical batch 2,532 replayed finite and was recorded
as `historical_overflow_not_reproduced`. Three other isolated overflows
occurred at epoch 2/batch 2,585/global step 8,523, epoch 3/batch 1,095/global
step 10,002, and epoch 4/batch 594/global step 12,470. Each affected only
`blocks.0.norm.norm.weight`, changed scale 2,048 to 1,024, retried the same
batch identity once, advanced zero counters on the failed attempt, and then
completed exactly one optimizer update. Counts were three attempted, three
recovered, and zero failed; no logical batch was skipped.

Epochs 2–4 each completed exactly 2,969 updates and immutable validation ran
once per epoch. Validation EERs for epochs 0–4 were `0.064`, `0.0602`,
`0.0587`, `0.0584`, and `0.0584`. Epoch 4 tied epoch 3, so the earlier epoch 3
remained selected. Training stopped at epoch 4 with `max_epoch`, final global
step 14,845, resumed updates 11,876, and exact final cosine multiplier 0.1.
The selected `best.pt` is byte-identical to `epoch_003.pt`, SHA-256
`ba9d989c1b6a922f3f392cd297fb771bba05319d3fad99df740ac889d2836d6f`,
with EER `0.0584` and locked empirical threshold `0.16545939445495605`.

Independent fresh-object audits restored final `last.pt` and `best.pt`
without optimizer work, reproduced counters, RNG/generator state, BatchNorm
buffers, selected metrics, FAR/FRR, and confusion. The focused 31 tests,
combined 82 tests, and complete 208-test suite passed; Python compilation and
`git diff --check` passed. Approved inputs, source audio, cache, environments,
pretrained weights, and historical evidence remained unchanged. Final-test
content was not opened, statted, hashed, parsed, loaded, or evaluated. No
commit or push occurred.

Result: **PASS** for AMP overflow remediation, exact atomic recovery, resumed
epochs 2–4, fixed validation, final checkpoint selection, and threshold lock.

Deferred: final-test access, threshold application to final test, final-test
trials/cache/embeddings/evaluation, further training or tuning, commit, and
push.

## 2026-07-30 - VieSpeaker2.0 one-time authoritative final evaluation v2

Quarantine was lifted only for this explicitly approved task and only for the
approved test manifest plus its 15,355 referenced WAVs. The metadata-only
protocol was locked before test audio or model access. The fixed trial package
contains 10,000 positive and 10,000 negative trials and reproduced
byte-identically under a different `PYTHONHASHSEED`; its identity SHA-256 is
`b06ba77783a6ad3442f3b7f77ebc1c367f8772b9707ff30191f963476dddb1c6`.
The evaluation-lock identity SHA-256 is
`7bff4b3b5f5b78fc2314a8031f70566c0e527f7e2e3a2721e2e01479031b049d`.

The completely separate final-test Fbank cache contains exactly 15,355
manifest-ordered `[301,80]` float32 raw non-transposed features in 60 shards.
Its completion identity SHA-256 is
`bc2a88fa1560c356c12eca4050da64c4924cb58cbf8a65d18c8c6b87d7d0dfde`.
The path-size-mtime snapshot of all approved source WAVs matched before and
after extraction, and the existing train/validation cache remained unchanged.

The checkpoint and threshold were fixed in advance: only
`outputs/ecapa_aam_multiepoch_v2/best.pt`, byte-identical to `epoch_003.pt`
with SHA-256
`ba9d989c1b6a922f3f392cd297fb771bba05319d3fad99df740ac889d2836d6f`,
was evaluated, and the operational threshold remained the selected epoch-3
validation threshold `0.16545939445495605`. One inference process produced
exactly 15,355 `[192]` embeddings and 20,000 aligned cosine scores. At the
locked threshold, TP/TN/FP/FN were `9598/9426/574/402`, FAR was `0.0574`,
FRR was `0.0402`, and accuracy was `0.9512`.

The descriptive test EER was `0.0453` at descriptive threshold
`0.17728488147258759`. It is non-operational, did not replace the locked
validation threshold, and caused no test-side training, tuning, calibration,
checkpoint selection, threshold adjustment, rerun, or rescoring. The
finalized state prevents another normal evaluation; the dedicated
dependency-light metrics-only command reproduced the saved-score metrics
exactly without SpeechBrain, CUDA, WAV, Fbank, embedding extraction, or
rescoring access.

Python compilation passed; 31 focused tests, 91 combined tests, and the
complete 239-test suite passed. `git diff --check` passed. Source WAVs,
approved manifests, train/validation cache, environments, pretrained weights,
checkpoints, and historical reports remained unchanged. No excluded or
unrelated audio was accessed. No commit or push occurred.

Result: **PASS** for the one-time authoritative VieSpeaker2.0 final
evaluation, locked-threshold metrics, descriptive non-operational test
diagnostic, immutable finalization, and metrics-only reproducibility.

Deferred: any further training, tuning, checkpoint selection, threshold
adjustment, final-test trial regeneration, feature or embedding extraction,
normal inference, rescoring, commit, and push.
