# Repository Rules

- Inspect the repository before editing and keep each task limited to its explicitly approved stage.
- Do not automatically continue to another pipeline stage.
- Do not commit or push unless explicitly instructed.
- Preserve source audio, `manifests/full_manifest.csv`, environments, pretrained weights, caches, checkpoints, and historical reports.
- Do not commit datasets, checkpoints, environments, caches, or manifests containing absolute local paths.
- The stable CUDA environment is `.venv-cuda`; do not modify it or the existing CPU environment.

## VieSpeaker2.0 governance

- The final v2 dataset root is `E:\VieSpeaker2.0\augmented_dataset`.
- VieSpeaker2.0 completely replaces `E:\VieSpeaker` for v2; the two datasets must never be merged.
- The old dataset and all v1 artifacts are immutable historical pilot evidence only.
- Do not reuse v1 splits, labels, label mappings, portable manifests, Fbank cache, trials, baselines, thresholds, checkpoints, or other data-dependent identities for v2.
- V2 artifacts must use separate versioned paths. Dataset-dependent counts and identities must come from an explicitly approved v2 manifest or configuration.
- Derive speaker identity from each WAV file's direct parent folder. Filename provenance is not an authoritative final split.
- The final v2 speaker split is approved only through the versioned split and identity artifacts below; do not reinterpret or regenerate it without explicit approval.
- Source WAVs are immutable. Do not write into the dataset root or run new preprocessing, resampling, channel conversion, VAD, normalization, cropping, padding, or augmentation before explicit approval.
- Do not hard-code the final v2 speaker/label count, feature length, sampler configuration, batches per epoch, split ratios, or source-group interpretation before explicit approval.
- Final splits must be speaker-disjoint, and every utterance from one speaker must remain in one final split.
- Validation and final test must not receive stochastic training augmentation. Validation will select checkpoints and thresholds; final test remains untouched until the explicitly approved final-evaluation task.
- The approved v2 split artifacts are `splits/v2/speaker_split_v2.csv`, `splits/v2/speaker_split_v2_identity.json`, and `splits/v2/split_policy_v2.json`; the approved portable package and binding identity are under `manifests/portable_v2/`.
- V2 speaker assignments are train 1,347, validation 100, final test 100, and excluded 128. The excluded speakers are exactly the singleton speakers and remain represented in `manifests/v2/full_manifest_v2.csv`.
- `manifests/portable_v2/speaker_to_label_v2.json` assigns train speakers contiguous labels `0..1346` in numeric speaker-ID order; every validation and final-test row has label `-1`.
- `manifests/portable_v2/test_manifest_v2.csv` is immutable. Its one-time authorized final evaluation is complete; do not reopen final-test audio or regenerate any final-test artifact.
- Every downstream v2 task must validate and bind the approved full-manifest and split hashes recorded in the v2 identity artifacts; do not substitute hand-maintained row counts.
- The approved raw train/validation Fbank cache is `outputs/fbank_cache_v2/`, with `fbank_cache_config_v2.json` and `fbank_cache_identity_v2.json` as its configuration and completion identity.
- The measured v2 cached feature shape is `[301, 80]`, float32, raw pre-normalization, and non-transposed.
- The immutable completed final-test cache is `outputs/fbank_cache_final_test_v2/`; its completion identity is `outputs/fbank_cache_final_test_v2/fbank_cache_identity_final_test_v2.json`, SHA-256 `bc2a88fa1560c356c12eca4050da64c4924cb58cbf8a65d18c8c6b87d7d0dfde`.
- Every downstream v2 cache consumer must validate and bind the v2 cache config and identity hashes rather than relying on path or shape alone.
- The approved v2 training sampler is defined only by `configs/v2/training_sampler_v2.json`: `HybridShardAwareSpeakerBatchSampler`, P=16, K=2, batch size 32, active shard window 8, seed 20260729, 2,969 batches and 95,008 logical selections per epoch, DataLoader workers 0, and Dataset LRU size 8 with `validate_finite=False` for the already validated immutable cache.
- Training must validate the sampler config hash and sampler-plan identity, call `set_epoch(epoch)`, preserve exact P x K batches, and keep rows sharing a nonempty duplicate group out of the same speaker batch. Do not reinterpret, benchmark alternatives, or regenerate this approved configuration without explicit approval.
- The fixed validation verification package is `manifests/verification_v2/validation_trials_v2.csv`, `validation_trials_config_v2.json`, and `validation_trials_identity_v2.json`. It contains exactly 10,000 positive and 10,000 negative trials over all 100 validation speakers and is immutable downstream input.
- The approved untouched pretrained validation reference is recorded in `reports/pretrained_ecapa_validation_baseline_v2.json`; its executable empirical threshold is validation-only evidence and must not be applied to final test before the explicitly approved final-evaluation task.
- The approved completed v2 epoch-zero run is `outputs/ecapa_aam_one_epoch_v2/`: `epoch_000.pt` is the immutable epoch-completion checkpoint, `last.pt` is the rolling/final-step checkpoint, and `best.pt` is the validation-selected trained checkpoint.
- The immutable v2 epoch-zero training origin is `outputs/ecapa_aam_one_epoch_v2/best.pt`, SHA-256 `19cfd3482172ce482d21b65915fd347aeeac4ee1e51bc61e59484f46a31a38db`; it produced validation EER `0.064` at validation-only threshold `0.16190975904464722`.
- Final v2 multi-epoch training is complete under `outputs/ecapa_aam_multiepoch_v2/` after resumed epochs 1–4, using at most one exact-same-batch AMP gradient-overflow retry at half GradScaler scale. Three overflow events recovered, zero failed, and training stopped at epoch 4 with reason `max_epoch`.
- The selected final v2 checkpoint is `outputs/ecapa_aam_multiepoch_v2/best.pt`, byte-identical to `epoch_003.pt`, SHA-256 `ba9d989c1b6a922f3f392cd297fb771bba05319d3fad99df740ac889d2836d6f`. Its validation EER is `0.0584` and its locked validation empirical threshold is `0.16545939445495605`.
- The one-time final v2 evaluation is complete. The immutable trial package is `manifests/verification_v2/final_test_trials_v2.csv`, `final_test_trials_config_v2.json`, and `final_test_trials_identity_v2.json`; the identity SHA-256 is `b06ba77783a6ad3442f3b7f77ebc1c367f8772b9707ff30191f963476dddb1c6`.
- The immutable evaluation lock is `configs/v2/final_evaluation_v2.json` with identity `configs/v2/final_evaluation_v2_identity.json`, SHA-256 `7bff4b3b5f5b78fc2314a8031f70566c0e527f7e2e3a2721e2e01479031b049d`.
- The evaluated checkpoint remains `outputs/ecapa_aam_multiepoch_v2/best.pt`, SHA-256 `ba9d989c1b6a922f3f392cd297fb771bba05319d3fad99df740ac889d2836d6f`. The operational threshold remains `0.16545939445495605`, sourced only from selected epoch-3 validation.
- Primary final-test metrics and the descriptive non-operational test EER are recorded in `reports/final_evaluation_v2.json` and `reports/final_evaluation_v2.md`. Test-derived thresholds are non-operational.
- Final-test cache, trials, embeddings, scores, and reports are immutable. Do not rerun normal model inference or rescoring; only metrics-only recomputation from the saved scores is permitted.
- Do not use final-test results for further training, tuning, checkpoint selection, or threshold adjustment.

## Model invariants

- Use the pretrained `speechbrain/spkrec-ecapa-voxceleb` model; do not train ECAPA-TDNN from scratch.
- The production cached-feature path remains:

```text
cached Fbank [B, T, 80]
-> mean_var_norm
-> embedding_model
-> embedding [B, 1, 192]
-> squeeze
-> embedding [B, 192]
```

- Do not transpose SpeechBrain features or replace the pretrained SpeechBrain frontend with `src/fbank.py`.
- Do not use the pretrained VoxCeleb classifier for the Vietnamese task.
