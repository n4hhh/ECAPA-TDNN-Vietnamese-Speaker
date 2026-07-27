# Train Cache Access Analysis and DataLoader Benchmark v1

## 1. Executive result

**PASS.** Metadata confirms 31,998 train rows, 488 speakers, contiguous labels `0..487`, and 125 shards. The existing index and shards are strongly speaker-grouped. Fully sample-random ordering provides much better batch diversity, but with the current small LRU it nearly reloads a shard for every sample. Pure shard-local ordering is extremely fast but has unacceptable diagnostic speaker concentration. A later hybrid speaker-balanced/shard-aware design is suggested for further evaluation; no sampler was implemented.

## 2. Environment

- Date: 2026-07-27
- OS: Windows 10 build 26200
- Python: 3.10.11
- torch: 2.2.0+cu121
- Environment: `.venv-cuda`
- Seed: 20260727
- Cache path in artifacts: `outputs/fbank_cache_v1`

## 3. Repository and cache state inspected

The repository rules, complete worklog, current git status, cached Dataset, smoke test, Dataset tests, both prior cache reports, cache producer, real cache config, and real train index were inspected. Only index metadata was used for composition and diversity analysis. Feature tensors were read only by the benchmark. Source WAVs, manifests, split files, and cache files were not modified.

## 4. Train shard composition

All feature indexes were valid, unique, and contiguous within their shard.

| Metric | Min | Mean | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Unique speakers/shard | 1 | 4.888 | 2 | 4 | 6 | 22 |
| Dominant-speaker share | 12.89% | 59.95% | 39.06% | 56.25% | 78.52% | 100% |

- At least 25% dominant: 119/125 shards (95.2%)
- At least 50% dominant: 72/125 (57.6%)
- At least 75% dominant: 38/125 (30.4%)
- Single-speaker shards: 22/125 (17.6%)

Measured fact: shard composition is highly concentrated. Interpretation: shard-only batches are unlikely to provide broad speaker diversity.

## 5. Train speaker utterance distribution

| Metric | Min | Mean | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Utterances/speaker | 5 | 65.570 | 10 | 29 | 75.25 | 1,595 |

- Fewer than 10: 119 speakers
- Fewer than 20: 204
- Fewer than 50: 316
- More than 100: 84
- More than 500: 6

The exact per-speaker utterance and shard counts are in `reports/train_cache_speaker_stats_v1.csv`. Measured fact: the distribution is strongly imbalanced. Interpretation: unconstrained sample shuffle reflects that imbalance; this alone does not prove a balanced sampler will improve model quality.

## 6. Existing index-order analysis

- Same-speaker runs: 488, exactly one per speaker
- Run length: min 5, mean 65.570, Q1 10, median 29, Q3 75.25, max 1,595
- Speaker transitions: 487 across 31,997 neighbor pairs
- Neighbor rows with the same speaker: 98.478%

Measured fact: the index is ordered in contiguous speaker runs. Interpretation: sequential training order would be unsuitable for speaker diversity.

## 7. Batch-diversity analysis

The table reports complete metadata-only batches and unique speakers/batch as `min / mean / median / max`. The 25% and 50% columns are the percentages of batches where one speaker reaches those shares.

| Order | Batch | Complete | Unique speakers/batch | Duplicate IDs | >=25% one speaker | >=50% one speaker |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | 16 | 1,999 | 1 / 1.229 / 1 / 4 | 100.00% | 100.00% | 98.60% |
| Sequential | 32 | 999 | 1 / 1.470 / 1 / 6 | 100.00% | 100.00% | 95.40% |
| Sample-random | 16 | 1,999 | 10 / 14.863 / 15 / 16 | 68.83% | 1.70% | 0.00% |
| Sample-random | 32 | 999 | 21 / 27.905 / 28 / 32 | 99.00% | 0.10% | 0.00% |
| Shard-local | 16 | 1,999 | 1 / 3.665 / 3 / 13 | 100.00% | 97.90% | 65.43% |
| Shard-local | 32 | 999 | 1 / 4.322 / 4 / 17 | 100.00% | 96.50% | 62.36% |

For batch size 32, the maximum samples from one speaker had mean/median/max `28.683/32/32` sequentially, `2.740/3/8` sample-random, and `19.387/18/32` shard-local. Full quartiles and represented-speaker count distributions are in the JSON. These simulations diagnose ordering behavior; they do not establish final training quality.

## 8. Benchmark methodology

Each run used the real train cache, batch size 32, two warm-up batches, 512 measured samples (16 complete batches), seed 20260727, and CPU DataLoader access only. First-batch latency was measured separately before warm-up. Every returned shape, dtype, device, label shape, and metadata length was checked. No SpeechBrain, ECAPA, CUDA inference, WAV access, or training was used.

## 9. Benchmark results

All seven configurations completed and validated successfully.

| Access | LRU | Workers | Finite | First batch (s) | Measured (s) | Batch/s | Sample/s | Measured loads | Loads/sample |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Sequential | 2 | 0 | true | 0.0150 | 0.1142 | 140.15 | 4,484.73 | 2 | 0.003906 |
| Sample-random | 2 | 0 | true | 0.2512 | 3.0295 | 5.28 | 169.00 | 502 | 0.980469 |
| Sample-random | 2 | 0 | false | 0.1686 | 2.5359 | 6.31 | 201.90 | 502 | 0.980469 |
| Sample-random | 8 | 0 | false | 0.1658 | 2.4332 | 6.58 | 210.42 | 484 | 0.945312 |
| Sample-random | 8 | 1 | false | 2.2334 | 2.4616 | 6.50 | 208.00 | N/A | N/A |
| Sample-random | 8 | 2 | false | 3.8157 | 1.7043 | 9.39 | 300.41 | N/A | N/A |
| Shard-local | 2 | 0 | false | 0.0062 | 0.0251 | 636.86 | 20,379.57 | 2 | 0.003906 |

For `num_workers=0`, total Dataset load counters (including first batch and warm-up) were respectively 3, 598, 598, 576, and 3; final LRU occupancy was respectively 2, 2, 2, 8, and 2.

## 10. Windows num_workers findings

Both `num_workers=1` and `num_workers=2` ran correctly under Windows with valid batches. No compatibility fix was needed. One worker produced 208.00 samples/s, effectively equal to the comparable zero-worker 210.42 samples/s, while two workers produced 300.41 samples/s (+42.8%). Startup cost was substantial: first-batch latency rose from 0.166 seconds to 2.233 and 3.816 seconds. Worker shard-load counters are unavailable because they were not aggregated across processes.

## 11. Finite-validation overhead comparison

At sample-random/LRU 2/workers 0, disabling the repeated scan increased observed throughput from 169.00 to 201.90 samples/s (+19.5%) with identical 502 measured shard loads. This is an observed short-run comparison, not an isolated microbenchmark; OS caching and run order may contribute. The default remains strict `True`. `False` is documented only for an already validated immutable cache.

## 12. LRU-size comparison

At sample-random/workers 0/finite false, increasing LRU 2 to 8 reduced measured shard loads from 502 to 484 (-3.6%) and increased throughput from 201.90 to 210.42 samples/s (+4.2%). This modest gain does not materially solve random-access reload behavior and costs storage for six more approximately 24.7 MB shards (roughly 148 MB of feature-shard file payload).

## 13. Sequential versus random versus diagnostic shard-local

Sequential and shard-local access were far faster because each needed only two measured shard loads. Sequential order is unusable diagnostically because the median 32-sample batch contained one speaker. Shard-local reached 20,379.57 samples/s but the median 32-sample batch contained only four speakers and 62.36% of batches were at least half one speaker. Sample-random provided a median of 28 speakers but paid almost one shard load per sample with a small LRU.

## 14. Evidence-based recommendation

**Suggestion only:** evaluate a hybrid speaker-balanced and shard-aware sampling design in a later explicitly approved task. Pure sequential and pure shard-local order concentrate speakers too severely. Standard sample shuffle provides good diversity but causes 0.98 measured shard loads/sample with LRU 2, and LRU 8 only reduces this to 0.95. The strong utterance imbalance (median 29, max 1,595) also motivates measuring a speaker-balanced component. A later design should retain cross-speaker batches while grouping enough requests by a bounded working set of shards; it must be benchmarked against standard shuffle and validated for training behavior. No production sampler was implemented here.

## 15. Limitations

- OS file caching affects repeated reads and configuration order.
- This is a 512-sample short diagnostic, not a full epoch.
- Multi-worker startup and per-worker caches differ from long-running training.
- Worker shard loads were not aggregated.
- Resident/peak memory was unavailable without adding a dependency; no estimate is reported.
- File sizes are not identical to resident tensor/storage overhead.
- Throughput and metadata diversity alone do not determine model quality.

## 16. Files created and modified

Created: `scripts/benchmark_cached_fbank_access.py`, `tests/test_cached_fbank_benchmark.py`, `reports/train_cache_access_analysis_v1.md`, `reports/train_cache_benchmark_v1.json`, `reports/train_cache_shard_stats_v1.csv`, and `reports/train_cache_speaker_stats_v1.csv`.

Modified: `src/cached_fbank_dataset.py`, `tests/test_cached_fbank_dataset.py`, and append-only `reports/CODEX_WORKLOG.md`.

## 17. Tests and commands

```text
.venv-cuda\Scripts\python.exe -m py_compile src\cached_fbank_dataset.py scripts\benchmark_cached_fbank_access.py tests\test_cached_fbank_dataset.py tests\test_cached_fbank_benchmark.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_cached_fbank_dataset tests.test_cached_fbank_benchmark -v
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv-cuda\Scripts\python.exe scripts\benchmark_cached_fbank_access.py --cache-dir outputs\fbank_cache_v1 --batch-size 32 --warmup-batches 2 --measured-samples 512 --seed 20260727 --report-dir reports
```

Focused tests: 15 passed. Complete suite: 26 passed. Benchmark: 7/7 configurations passed.

## 18. Explicitly deferred stages

Production sampling, training, AAM-Softmax, losses, backpropagation, optimizer work, checkpoints, verification, scoring, thresholds, EER, test evaluation, preprocessing, augmentation, cache/manifests/split regeneration, and all later pipeline stages remain deferred.

## 19. Overall result

**PASS**
