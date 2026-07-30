# VieSpeaker2.0 Final Speaker-Disjoint Split Package v2

## Goal and exact scope

Created and validated only the approved speaker-disjoint split, portable
manifests, train label mapping, and immutable identity bindings. No audio
content, feature, model, trial, score, or training stage was accessed.

## Approved input identities

- Full manifest: `manifests/v2/full_manifest_v2.csv`
- Full manifest SHA-256: `26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f`
- Full-manifest identity: `manifests/v2/full_manifest_v2_identity.json`
- Full-manifest identity SHA-256: `f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e`

## Deterministic selection algorithm

- Seed: `20260729`
- Identifier: `sha256_bucket_largest_remainder_exact_balance_v1`
- Eligible speakers have at least 20 utterances.
- Largest-remainder proportional allocation uses configured bucket order
  to resolve equal remainders.
- Within each bucket speakers are ordered by
  `SHA256(UTF8('<seed>:<speaker_id>'))`, then numeric speaker ID.
- The selected 200 speakers are assigned by exact dynamic programming:
  exact 100/100 counts, per-bucket difference at most one, minimum total
  utterance difference, then minimum duplicate-file difference, then
  configured-bucket/SHA-256 stable tie breaking.
- Provenance and candidate source groups do not influence assignment.

## Bucket allocation

| Bucket | Eligible | Selected | Validation | Test |
|---|---:|---:|---:|---:|
| 20-49 | 250 | 63 | 32 | 31 |
| 50-99 | 199 | 50 | 25 | 25 |
| 100-199 | 202 | 50 | 25 | 25 |
| 200-499 | 120 | 30 | 15 | 15 |
| 500+ | 29 | 7 | 3 | 4 |

## Exact counts

| Split | Speakers | Rows | Duration seconds | Duplicate groups | Duplicate files |
|---|---:|---:|---:|---:|---:|
| train | 1347 | 95009 | 285027.000000 | 2 | 4 |
| validation | 100 | 15355 | 46065.000000 | 0 | 0 |
| test | 100 | 15355 | 46065.000000 | 2 | 4 |
| excluded | 128 | 128 | 384.000000 | 0 | 0 |

All 125,847 full-manifest rows reconcile across train, validation,
test, and excluded. All 128 excluded speakers are singletons retained
in the authoritative full manifest and omitted only from portable
train/validation/test manifests.

## Evaluation balance

- Validation/test selected utterances: `15355 / 15355`
- Absolute utterance difference: `0`
- Validation/test duplicate files: `0 / 4`
- Absolute duplicate-file difference: `4`

## Label mapping and portable manifests

- Train classes: `1347`.
- Train labels: contiguous `0..1346` in numeric speaker-ID order.
- Validation/test labels: `-1`.
- Paths are dataset-root-relative, use `/`, and contain no absolute
  prefix, drive, backslash, or parent traversal.
- Portable schema preserves descriptive provenance as `filename_group`
  and carries `duplicate_group`; neither controls assignment.

## Duplicate and final-test policy

All four exact duplicate groups and all eight files remain present in
their owning speaker's single final split. No representative was
selected. Later validation-trial generation must reject a positive
pair when both files share the same non-empty `duplicate_group`.

`test_manifest_v2.csv` is immutable and quarantined after this
validation. This task did not open test WAV content, extract test
features, create trials, load test data into a model, or compute
embeddings, scores, EER, thresholds, or accuracy.

## Artifact hashes

- `splits/v2/speaker_split_v2.csv`: `cbbcdcd4d3561ff2470a6cd713187cbb612939e643e5bc8cf4ed25504539f87e`
- `splits/v2/speaker_split_v2_identity.json`: `87d2a542ae1716f5d143e27835bf0478413e672cfa131462c0410808498c3c67`
- `splits/v2/split_policy_v2.json`: `3e414836b4fe307841810cffa56c3f0040d0c662d265d276d73a7d00a16a5d10`
- `manifests/portable_v2/train_manifest_v2.csv`: `f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b`
- `manifests/portable_v2/validation_manifest_v2.csv`: `9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc`
- `manifests/portable_v2/test_manifest_v2.csv`: `14c782fa36d9c23040dd7a7fce26d91b9a57a230bdeebd19658dbcb2ee8364eb`
- `manifests/portable_v2/speaker_to_label_v2.json`: `9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8`
- `manifests/portable_v2/portable_manifests_v2_identity.json`: `29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5`
- Split identity file: `87d2a542ae1716f5d143e27835bf0478413e672cfa131462c0410808498c3c67`
- Portable identity file: `29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5`

## Reproducibility and formatting

- UTF-8 without BOM, LF newlines, fixed CSV columns, fixed six-decimal
  duration serialization, sorted JSON keys, two-space JSON indentation,
  numeric speaker ordering, and no identity timestamps.
- A second build under the ignored
  `outputs/speaker_split_v2_reproduction/` path matched all eight required
  production artifacts byte-for-byte.

## Validation, preservation, and result

- Final CSV/JSON schema, hash, row, speaker, label, path, duplicate,
  disjointness, and reconciliation read-back: PASS.
- The builder does not accept or access a dataset root. Source WAV
  content was not opened, and no source WAV could be changed.
- Full manifest pre/post hash: identical.
- Result: **PASS** for the split package only.

## Files created and modified

Created:

- `src/speaker_split_v2.py`
- `scripts/create_speaker_split_package_v2.py`
- `tests/test_speaker_split_v2.py`
- `splits/v2/speaker_split_v2.csv`
- `splits/v2/speaker_split_v2_identity.json`
- `splits/v2/split_policy_v2.json`
- `manifests/portable_v2/train_manifest_v2.csv`
- `manifests/portable_v2/validation_manifest_v2.csv`
- `manifests/portable_v2/test_manifest_v2.csv`
- `manifests/portable_v2/speaker_to_label_v2.json`
- `manifests/portable_v2/portable_manifests_v2_identity.json`
- `reports/speaker_split_v2_summary.md`
- `reports/speaker_split_v2_summary.json`
- Ignored independent reproduction package under
  `outputs/speaker_split_v2_reproduction/`

Modified:

- `AGENTS.md`
- `reports/CODEX_WORKLOG.md` by one append-only entry

All pre-existing dirty and untracked work was preserved.

## Exact commands and tests

```text
.venv-cuda\Scripts\python.exe -m py_compile src\speaker_split_v2.py scripts\create_speaker_split_package_v2.py tests\test_speaker_split_v2.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_speaker_split_v2 -v
.venv-cuda\Scripts\python.exe scripts\create_speaker_split_package_v2.py
$verificationRoot = 'outputs\speaker_split_v2_reproduction'; New-Item -ItemType Directory -Path $verificationRoot | Out-Null; .venv-cuda\Scripts\python.exe scripts\create_speaker_split_package_v2.py --output-root $verificationRoot --reference-root .
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
git diff --check
```

- Python compilation: PASS.
- Focused synthetic suite: 13 passed.
- Complete repository unittest discovery: 143 passed.
- Transactional finalized-artifact read-back: PASS.
- Independent reproduction: PASS for all eight required artifacts.
- `git diff --check`: PASS.

## Explicit exclusions

No Fbank/cache, trial generation, SpeechBrain/ECAPA load, embedding,
score, EER, threshold, accuracy, sampler benchmark, training,
checkpoint, final-test content access/evaluation, commit, or push.
