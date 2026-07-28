# Pretrained ECAPA validation baseline v1

## Executive result

**PASS.** The saved-score, optimized interpolated validation EER is **11.665301106104%**. No embedding extraction or model inference was run for this metrics patch.

## Preserved baseline and inputs

- Validation: 8504 utterances / 100 speakers / labels all `-1`
- Embeddings: `[8504, 192]`, float32 CPU
- Trials: 9764 positive / 9764 negative / 19528 scores
- Score range: `[-0.20446446537971497, 0.935106635093689]`
- Protected SHA-256: `{"manifests/portable/validation_manifest_v1.csv": "9f553fa55b50ee071120bcbb2ce3c2caf9de4799cfb615eb732bbefd0d73d38b", "manifests/verification/validation_trials_v1.csv": "3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f", "outputs/fbank_cache_v1/fbank_cache_config_v1.json": "c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9", "outputs/fbank_cache_v1/validation_feature_index_v1.csv": "7346073ed9b47354c5f85889222ad9a538d2c4ee14e0b7d85001e3c0d61608bc", "outputs/validation_pretrained_ecapa_baseline_v1/validation_embeddings_v1.pt": "7c9f6845e55517a6981858bc4327cad09b9c037c2b288f1b26ff28120eaec828", "outputs/validation_pretrained_ecapa_baseline_v1/validation_trial_scores_v1.pt": "b67cd388793ef3eb1f722f38dffa20cef7f789de70c07d6f43d4ea2c64b02cc2"}`
- Trial paths were validated against authoritative validation path ownership.

## Interpolated result

- Interpolated EER fraction: 0.11665301106104056
- Interpolated EER percentage: 11.665301106104057
- Interpolated threshold: 0.31165990233421326
- Interpolated FAR / FRR: 0.11665301106104056 / 0.11665301106104056
- Kind: `linearly_interpolated_roc_crossing`; threshold kind: `interpolated_non_empirical`

The interpolated threshold describes a linearly interpolated ROC crossing. With discrete scores it may not be an empirical operating point, so directly applying it can produce errors different from the interpolated FAR/FRR.

## Executable empirical operating point

- Selection: minimize `abs(FAR-FRR)`, then average error, then prefer the higher threshold
- Empirical threshold: 0.31165990233421326
- Actual FAR / FRR: 0.11665301106104056 / 0.11665301106104056
- FAR/FRR gap: 0.0
- Average error: 0.11665301106104056
- Semantics: `accept same speaker when score >= threshold`
- FAR/FRR were independently recomputed from every saved score using the stated threshold rule.

## Regression and performance

- Previous / optimized interpolated EER: 0.11665301106104053 / 0.11665301106104056
- Absolute EER difference: 2.7755575615628914e-17
- Previous / optimized interpolated threshold: 0.31165990233421326 / 0.31165990233421326
- Metric runtime: 0.056543500 seconds
- Unique scores: 19523
- Complexity: `O(N log N)` sorting plus `O(N)` tied-group scan

## Original extraction context

- Original extraction elapsed: 45.94463000000178 seconds; throughput: 185.09236008647085 utterances/s
- Original peak allocated/reserved VRAM: 2567797248 / 3368026112 bytes
- Embedding and score save/load differences: 0.0 / 0.0
- No SpeechBrain/ECAPA import, inference, scoring, parameter access, CUDA use, WAV access, or final-test access occurred in this patch.

## Overall result

**PASS**
