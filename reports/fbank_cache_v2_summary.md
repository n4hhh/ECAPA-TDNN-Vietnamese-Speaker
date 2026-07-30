# VieSpeaker2.0 train/validation Fbank cache and cached ECAPA preflight v2

## Scope and result

Built and fully validated only the approved VieSpeaker2.0 train and validation
raw SpeechBrain Fbank cache, generalized the lazy cached Dataset without
changing v1 behavior, exercised Windows DataLoaders, and ran one cached train
batch and one cached validation batch through the frozen ECAPA normalization
and embedding modules.

Result: **PASS**.

No trial, score, EER, threshold, accuracy, sampler benchmark, AAM object,
optimizer step, training, checkpoint, or final-test artifact was created.

## Approved inputs

All identities matched before extraction and remained unchanged afterward:

- Full manifest:
  `26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f`
- Full-manifest identity:
  `f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e`
- Speaker split:
  `cbbcdcd4d3561ff2470a6cd713187cbb612939e643e5bc8cf4ed25504539f87e`
- Speaker-split identity:
  `87d2a542ae1716f5d143e27835bf0478413e672cfa131462c0410808498c3c67`
- Split policy:
  `3e414836b4fe307841810cffa56c3f0040d0c662d265d276d73a7d00a16a5d10`
- Portable-package identity:
  `29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5`
- Train manifest:
  `f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b`
- Validation manifest:
  `9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc`
- Train label mapping:
  `9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8`

The authoritative manifests yielded 95,009 train rows, 15,355 validation
rows, 1,347 train classes with labels `0..1346`, and validation labels all
`-1`.

## Feature preflight and environment

- Model: `speechbrain/spkrec-ecapa-voxceleb`
- Python: `3.10.11`
- PyTorch / torchaudio: `2.2.0+cu121 / 2.2.0+cu121`
- SpeechBrain: `1.0.3`
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Device: `cuda:0`
- Four fixed real train rows measured raw `compute_features` output as
  `[4,301,80]`, float32 and finite.
- The measured per-utterance feature shape is `[301,80]`.
- The required batch-size-64 preflight produced `[64,301,80]` from
  `[64,48000]` waveforms.
- Batch-size fallback was not used.
- Features are raw pre-`mean_var_norm` output and were never transposed.

PCM decoding used the established SpeechBrain-compatible torchaudio path.
There was no resampling, cropping, padding, augmentation, gain/loudness
normalization, or source rewrite.

## Cache configuration and layout

- Cache: `outputs/fbank_cache_v2/`
- Config: `outputs/fbank_cache_v2/fbank_cache_config_v2.json`
- Completion identity: `outputs/fbank_cache_v2/fbank_cache_identity_v2.json`
- Train index: `outputs/fbank_cache_v2/train_feature_index_v2.csv`
- Validation index:
  `outputs/fbank_cache_v2/validation_feature_index_v2.csv`
- Feature dtype and shape: float32 `[301,80]`
- Shard size: 256
- Extraction batch size: 64
- Included splits: exactly `["train","validation"]`
- Deterministic order: portable-manifest row order
- Deterministic assignment: `manifest_row_index divmod shard_size`
- Shard paths are cache-relative and use `/`.
- Audio paths are dataset-root-relative and use `/`.
- No absolute dataset root is persisted.

Per-shard writes used same-directory temporary files followed by atomic
replacement. Resume requires byte-identical configuration and index plans and
validates every existing shard before skipping it; incompatible or corrupt
partial caches fail closed.

## Train and validation cache

| Split | Rows | Shards | Shard bytes | Labels |
|---|---:|---:|---:|---|
| train | 95,009 | 372 | 9,158,497,768 | exact `0..1346` coverage |
| validation | 15,355 | 60 | 1,480,143,800 | all `-1` |

Total: 110,364 utterances, 432 shards, and 10,638,641,568 shard bytes.
Production extraction took 206.745 seconds. Peak CUDA allocated/reserved memory
was 217,820,672 / 253,755,392 bytes.

Every shard was read back. Every feature tensor was CPU float32, finite,
contiguous, and `[301,80]`; every index row matched its exact tensor position,
path, speaker, label, split, and manifest row identity. All global cached paths
were unique. There were no unexpected shards or split directories.

## Cached Dataset and DataLoader

`CachedFbankDataset` now detects v1 or v2 config dynamically. V2 uses its
measured `[T,80]`, authoritative row/class metadata, completion identity,
config hash, index hashes, strict train/validation allowlist, deterministic
row/shard alignment, lazy CPU loading, and bounded LRU shard cache. The v1
config, index, shard, sample, and collate behavior remains compatible.

Real-cache checks passed for both splits:

- Dataset lengths matched the authoritative indexes.
- Two sequential batches with `num_workers=0` preserved tensors, labels,
  portable paths, metadata, `dataset_index`, and `manifest_row_index`.
- One two-sample batch with `num_workers=2` passed on Windows.
- Train and validation batch shapes were `[4,301,80]` for the sequential
  checks and `[2,301,80]` for the worker checks.

## Cached ECAPA CUDA preflight

For one train and one validation batch:

```text
cached Fbank [4,301,80]
-> mean_var_norm [4,301,80]
-> embedding_model [4,1,192]
-> squeeze(1) [4,192]
```

All tensors were CUDA float32 and finite. Train labels were valid and
validation labels were all `-1`. The model was in evaluation and inference
mode; parameters were frozen and byte-equal before/after; parameter gradients
were absent; no optimizer existed. Forward hooks recorded zero calls to
`compute_features` and zero calls to the pretrained VoxCeleb classifier.

## Identity and reproducibility

- Config SHA-256:
  `ec71959ec64361038991e760e772d11bf1779e1b505a892dff364cf45aaeb018`
- Cache identity SHA-256:
  `1a2d6af777311f687e887575bfaf20915ed0409fd2e05b2f1ac232d43cd0b8c8`
- Train index SHA-256:
  `e20c320fc5842502a26684023bb307a7b2afa27a14a3cf1130fdffe31e85d4b9`
- Validation index SHA-256:
  `1d3a95e95aaa5b13e6614c077fbbdf10f7c208c79970c2a85bdd461193b9f585`
- Runtime result SHA-256:
  `71fa3164c0f2adf2eb435defa9559b1379bcb566942119264657deed4fbc15b9`
- Cached ECAPA result SHA-256:
  `4d634d87a1c8fa614effb8f520ffa6265975336bebec50d53db67bd7bdd55f02`

Config and index planning were each generated twice and matched byte-for-byte.
Two fixed train features and two fixed validation features were re-extracted
in their original batch contexts. Maximum absolute difference was exactly
`0.0` for both splits.

## Tests and preservation

- Python compilation: PASS.
- Focused v2 cache suite: 13 passed.
- Complete repository unittest discovery: 156 passed.
- Existing v1 Fbank cache and Dataset tests: PASS.
- `git diff --check`: PASS.
- Approved train/validation source path-size-mtime state before/after:
  identical for all 110,364 rows.
- Approved input hashes before/after: identical.
- Source WAV rewrites: zero.
- Environments and pretrained weights were not modified.

An early full-suite check identified that three v2-only metadata keys had been
added to legacy v1 samples. The implementation was corrected to expose those
keys only for v2, after which the complete 156-test suite passed.

## Final-test quarantine and deferred work

The final-test manifest was not opened, statted, hashed, parsed, or loaded.
No final-test or excluded WAV was accessed. No `test` cache directory, index,
shard, feature, embedding, or trial exists. The occurrence of `test` in
`filename_group` remains descriptive provenance inside approved train or
validation rows and does not represent final-test access.

Deferred: trial generation, scoring, EER, threshold or accuracy calculation,
sampler benchmarking, AAM construction, optimizer work, training, checkpoints,
all final-test content/evaluation actions, commit, and push.
