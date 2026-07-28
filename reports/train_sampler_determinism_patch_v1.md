# Training Sampler Determinism and Reporting Patch v1

## Result

**PASS.** Hybrid sampling is deterministic across separate Python processes with different hash seeds, benchmark reporting now fails closed, and custom runs cannot overwrite any standard full-run artifact.

## Scope

This patch changes sampler determinism, benchmark correctness gating, and artifact-path safety only. It does not access source WAV files or run model inference or training.

## Determinism fix

The hybrid sampler now preserves the shuffled active-shard order in a list, builds eligible speakers in deterministic shard/index order, sorts usable speaker IDs, and only then assigns seeded random tie-break values. The exposure policy, exact P × K structure, active-window expansion, and active-first sample preference are unchanged.

## Cross-process test

`tests/run_hybrid_sampler_fixture.py` runs the synthetic cached-Fbank fixture through a proper Python entry point. The unit test invokes it using `sys.executable` with `PYTHONHASHSEED=1` and `PYTHONHASHSEED=999`. Same seed/epoch batches are byte-equivalent after stable JSON serialization; epoch 1 differs from epoch 0 and remains structurally valid.

## Fail-closed reporting

Before any JSON, Markdown, or CSV output is written, the benchmark now rejects:

- duplicate or out-of-range indexes;
- P × K violations or malformed batch sizes;
- same-epoch nondeterminism or failure to change across epochs;
- Dataset mutation or missing all-speaker exposure;
- unsuccessful real-cache runs or invalid returned batches.

`overall_result: PASS` is added only after this validation succeeds. Tests prove a false metric raises and leaves all candidate output paths absent.

## Output safety

The standard full run retains:

- `reports/train_sampler_benchmark_v1.json`
- `reports/train_sampler_analysis_v1.md`
- `reports/train_sampler_batch_metrics_v1.csv`

A custom run requires an explicit JSON output. Its Markdown and batch CSV are derived from that custom stem. A custom run targeting a standard path is rejected.

The short verification used:

- `reports/train_sampler_determinism_patch_w8_verification_v1.json`
- `reports/train_sampler_determinism_patch_w8_verification_v1.md`
- `reports/train_sampler_determinism_patch_w8_verification_v1_batches.csv`

## W8 regression comparison

| Metric | Previous full report | Patched W8 |
|---|---:|---:|
| Exact structure | 16 × 2 | 16 × 2 |
| P × K violations | 0 | 0 |
| Duplicate indexes inside batches | 0 | 0 |
| All 488 speakers selected | yes | yes |
| Unique utterances | 16,903 | 16,897 |
| Repeated selections | 15,097 | 15,103 |
| Exposure min/median/max | 56 / 68 / 76 | 56 / 68 / 76 |
| Exposure CV | 0.0924310554 | 0.0926576360 |
| Median distinct shards/batch | 6 | 6 |
| Metadata LRU-8 loads/sample | 0.20071875 | 0.19956250 |

The exact sequence changed because previously unordered hash iteration affected seeded tie-break assignment. Coverage changed by six selections, exposure dispersion changed by approximately 0.00023, and the LRU-8 estimate improved by approximately 0.00116 loads/sample. Exact batch correctness and median locality did not regress.

## Short real-cache verification

- Candidate: hybrid W8
- P=16, K=2, 1,000-batch metadata epoch
- Real-access workload: one repeat, one warm-up batch, four measured batches, workers=0
- Returned Fbank: `[32, 301, 80]`, float32, CPU
- Labels: `[32]`, int64; metadata aligned
- Same epoch reproducible: yes
- Different epoch changed: yes
- Actual measured shard loads/sample: 0.1953125
- Steady-state throughput: 443.27 samples/s
- Result: PASS

This was a post-fix verification, not a replacement benchmark.

## Standard report preservation

The standard JSON, Markdown, and batch CSV SHA-256 hashes were identical before and after the short verification:

- JSON: `144EEAE76D001C9461AC425EFDF6118D09C2E79795A7245F84E763BB7F1F6370`
- Markdown: `E089AC50D7A1EBFDD821E89CD065C8E910956059A5A9DF23D2BE00B07D99A613`
- CSV: `920A14652169A85EB1F3210F2355077C6904528CBF040B08B837633FC51A94FA`

## Tests

- Focused sampler tests: 9 passed.
- Focused benchmark/report tests: 6 passed.
- Complete suite: 41 passed.
- Python compilation: passed.
- Cross-process determinism test: passed on Windows.

## Deferred

Training, ECAPA inference/fine-tuning, AAM-Softmax, optimization, verification, thresholds, EER, final evaluation, audio work, cache/manifest/split changes, commit, and push remain deferred.
