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
