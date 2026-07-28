# Validation Trial Integrity and EER Metrics Patch v1

## Result

**PASS.** Trial paths now validate against authoritative validation speaker ownership, EER uses a grouped `O(N log N)` implementation, and interpolated results are explicitly separated from executable empirical operating points. Production metrics were recalculated only from existing saved CPU artifacts.

## Trial integrity

- Every trial path is required to exist exactly once in validation metadata.
- Declared left/right speaker IDs must equal each path's real owner.
- Targets are checked against real owners, and all 100 validation speakers must be represented.
- Errors include the trial ID and offending path when applicable.
- Unsafe paths now reject absolute paths, backslashes, parent traversal, colons, and Windows drive-relative forms such as `C:audio/file.wav`.
- Production result: 19,528 trials passed authoritative ownership validation.

## Optimized metric implementation

Scores are paired with targets, sorted once in descending order, grouped by identical score, and processed with cumulative accepted-positive/negative counts. Ties cannot depend on input order. Complexity is `O(N log N)` for sorting plus `O(N)` for the grouped scan. Tests compare every empirical point against a brute-force reference on deterministic randomized fixtures.

## Interpolated result

- Previous EER: `0.11665301106104053`
- Optimized EER: `0.11665301106104056`
- Absolute difference: `2.7755575615628914e-17`
- Previous threshold: `0.31165990233421326`
- Optimized interpolated threshold: `0.31165990233421326`
- Interpolated FAR / FRR: `0.11665301106104056 / 0.11665301106104056`

The negligible EER difference comes from computing FRR directly as integer rejected positives divided by total positives instead of subtracting an accepted fraction from `1.0`. It is floating-point arithmetic ordering, not a changed operating point or tie-group correction.

An interpolated threshold generally describes a non-empirical ROC crossing and is not claimed to reproduce interpolated errors when applied directly.

## Executable empirical operating point

Policy: minimize `abs(FAR-FRR)`, then average error, then prefer the higher threshold.

- Threshold: `0.31165990233421326`
- Actual FAR: `0.11665301106104056`
- Actual FRR: `0.11665301106104056`
- Gap: `0.0`
- Average error: `0.11665301106104056`
- Semantics: accept same speaker when `score >= threshold`

For this production score set, the selected stored-score threshold happens to be an exact empirical FAR/FRR equality. FAR and FRR were independently recomputed across all saved scores.

## Metrics-only verification

- Embeddings: `[8504, 192]`, float32 CPU
- Scores/trials: 19,528; positive/negative: 9,764/9,764
- Unique scores: 19,523
- Optimized metric runtime: `0.05654350000259001` seconds
- No embedding extraction, cosine rescoring, SpeechBrain/ECAPA import or inference, CUDA work, WAV access, final-test access, or training occurred.

## Protected SHA-256 hashes

- Trial CSV: `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`
- Embeddings: `7c9f6845e55517a6981858bc4327cad09b9c037c2b288f1b26ff28120eaec828`
- Scores: `b67cd388793ef3eb1f722f38dffa20cef7f789de70c07d6f43d4ea2c64b02cc2`
- Validation manifest: `9f553fa55b50ee071120bcbb2ce3c2caf9de4799cfb615eb732bbefd0d73d38b`
- Validation cache index: `7346073ed9b47354c5f85889222ad9a538d2c4ee14e0b7d85001e3c0d61608bc`
- Cache config: `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`

All remained byte-for-byte unchanged.

## Deferred

Trial regeneration, embedding extraction, model inference, training/fine-tuning, AAM-Softmax, optimization, checkpoints, augmentation, preprocessing, cache/split changes, every final-test operation, commit, and push.
