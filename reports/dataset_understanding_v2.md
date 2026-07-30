# VieSpeaker2.0 Dataset Understanding and Integration Audit v2

## 1. Executive summary

**Result: PASS for the audit task.** The complete read-only traversal reconciled 125,847 files, all WAV, in 1,675 numeric direct-parent speaker folders. Every WAV header was structurally readable and every file had the same observed physical format: 16 kHz, mono, 16-bit PCM, 48,000 frames, 3.0 seconds, and 96,078 bytes. The dataset contains four exact byte-duplicate pairs, all within the same speaker, and no cross-speaker exact duplicate.

PASS means that traversal, reconciliation, output validation, duplicate checking, and preservation checks completed. It does **not** approve a production full manifest, a split, labels, feature extraction, trials, a baseline, or training.

## 2. Exact dataset root audited

The runtime root was `E:\VieSpeaker2.0\augmented_dataset`, the final v2 dataset root governed by `AGENTS.md`. Persisted inventory rows contain only dataset-root-relative forward-slash paths.

VieSpeaker2.0 replaces the old `VieSpeaker` dataset for v2. No old and new dataset content was merged.

## 3. Scope and explicit exclusions

This task audited the supplied output of Stage A and performed a pilot-to-v2 integration/governance review. It did not resample, convert channels, run VAD, normalize, crop, pad, augment, decode complete waveforms, extract Fbank, load SpeechBrain or ECAPA, compute embeddings, construct splits or labels, generate trials, score verification pairs, calculate EER, select thresholds, train, resume a checkpoint, or access a v1 final-test artifact.

The per-file CSV is `dataset_audit_inventory_nonproduction_v2.csv`. It is an audit inventory, not `full_manifest_v2`, and carries no `final_split` or label.

## 4. Scan completeness

- Deterministic traversal completed with zero traversal errors.
- Detail rows reconciled with all filesystem file and byte totals.
- Inventory rows reconciled with WAV count.
- Speaker summary rows reconciled with valid candidate speakers.
- Duplicate hashing completed with zero hash errors.
- All persisted schemas were validated and read back.
- Pre/post path-size-mtime snapshot identity matched exactly.
- Result: PASS.

## 5. Exact totals

| Measure | Exact value |
|---|---:|
| All files | 125,847 |
| WAV files | 125,847 |
| Non-WAV files | 0 |
| Bytes | 12,091,128,066 |
| GiB | 11.2607405204326 |
| Top-level speaker folders | 1,675 |
| Numeric top-level folders | 1,675 |
| Valid candidate speakers | 1,675 |
| Readable WAV headers | 125,847 |
| Unreadable WAVs | 0 |
| Total duration | 377,541 seconds |
| Total duration | 104.8725 hours |

A “candidate speaker” here means a numeric direct-parent directory with at least one correctly placed WAV. The audit does not independently prove real-world identity correctness.

## 6. Directory-layout findings

All 125,847 files have relative depth 2 and match `speaker_id/file.wav`. There were no root-level files, nested WAVs, nested directories within speaker directories, empty directories, empty speaker directories, non-numeric top-level folders, non-WAV files, unsupported extensions, symbolic links, duplicate normalized relative paths, Windows case-collision groups, or detected unsafe/non-portable WAV paths.

Speaker identity was derived only from the WAV direct parent. Filenames were not used for identity.

## 7. Speaker distribution

Utterances per candidate speaker:

| Statistic | Value |
|---|---:|
| Count | 1,675 |
| Minimum | 1 |
| Mean | 75.13253731343283 |
| Population standard deviation | 184.64828782053348 |
| p01 | 1 |
| p05 | 1 |
| p25 | 4 |
| Median | 17 |
| p75 | 78 |
| p95 | 319.29999999999995 |
| p99 | 576.8399999999997 |
| Maximum | 3,037 |

Readable utterance counts are identical because no WAV was unreadable. Speaker duration is exactly three times utterance count: minimum 3 seconds, mean 225.3976119402985, median 51, p95 957.9, p99 1,730.52, maximum 9,111, and population standard deviation 553.9448634616004 seconds.

Threshold observations, not exclusion rules:

| Condition | Speakers |
|---|---:|
| Fewer than 2 utterances | 128 |
| Fewer than 3 | 246 |
| Fewer than 5 | 427 |
| Fewer than 10 | 670 |
| Fewer than 20 | 875 |
| Fewer than 50 | 1,125 |
| More than 100 | 350 |
| More than 500 | 29 |

The 128 one-utterance speakers cannot form a positive verification pair using their supplied data alone. No speaker was removed. Exact per-speaker counts, durations, provenance composition, duplicate membership, and candidate source-group counts are in `reports/dataset_speaker_summary_v2.csv`.

## 8. WAV format distributions

The metadata is homogeneous:

| Property | Value | Files |
|---|---:|---:|
| Sample rate | 16,000 Hz | 125,847 |
| Channels | 1 | 125,847 |
| Sample width | 2 bytes / 16 bits | 125,847 |
| Encoding | PCM | 125,847 |
| Frame count | 48,000 | 125,847 |
| File size | 96,078 bytes | 125,847 |

No readable file differed from the dominant metadata combination. The exact distributions, including frame count, duration, and file size, are in the ignored runtime table `outputs/dataset_understanding_v2/dataset_metadata_distributions_v2.csv`.

These properties are consistent with the expected Stage-A output, but the audit did not alter audio to enforce them and does not hard-code them as future v2 production configuration.

## 9. Duration distribution

Every readable file is exactly 3.0 seconds. Count/minimum/mean/median/maximum are 125,847/3.0/3.0/3.0/3.0; population standard deviation is 0.0 and every requested percentile from p01 through p99 is 3.0. The duration histogram has all 125,847 files in `[3,4)` seconds and zero in every other configured bucket. No statistical, extremely short, or extremely long duration candidates were detected.

## 10. Invalid, corrupt, zero-byte, and zero-frame files

- Structurally unreadable or invalid WAVs: 0.
- Corrupt/truncated header errors: 0.
- Zero-byte WAVs: 0.
- Zero-frame WAVs: 0.
- Speakers containing invalid WAVs: 0.
- Readable non-dominant formats: 0.

The implementation inspected RIFF/WAVE chunks and declared data boundaries without decoding sample arrays.

## 11. Provenance parsing

All filenames matched one of four exact lowercase patterns:

| Provenance | Files | Speakers | Duration seconds | Pattern |
|---|---:|---:|---:|---|
| train | 86,197 | 1,024 | 258,591 | `aug_train-{n}-of-{n}_{n}.wav` |
| train_small | 16,128 | 430 | 48,384 | `aug_train_small-{n}-of-{n}_{n}.wav` |
| part | 17,858 | 536 | 53,574 | `aug_part-{n}-of-{n}_{n}.wav` |
| test | 5,664 | 69 | 16,992 | `aug_test-{n}-of-{n}_{n}.wav` |
| other | 0 | 0 | 0 | none |
| unparseable | 0 | 0 | 0 | none |

There were 384 speakers represented across more than one provenance class. This is not a split violation: these tokens are source provenance only and were not used to construct a final split.

## 12. Candidate source-group analysis

The exact filename grammar supports a deterministic syntactic coordinate:

```text
aug_<provenance>-<shard>-of-<declared_total>_<row>.wav
-> <provenance>/<shard>-of-<declared_total>/<row>
```

All 125,847 files parsed with `exact_pattern_supported`, producing 125,847 candidate coordinates. Every candidate group has size 1; no group spans speakers or provenance classes.

The evidence supports high confidence that the components are a provenance token plus shard/total/row-like numeric coordinates. It does **not** demonstrate what upstream object the coordinate identifies, whether the final number is an original row, or whether any omitted suffix would identify augmentation variants.

Explicit answers:

- Can augmented variants from one underlying source utterance be inferred? **No, not reliably from this dataset alone.**
- Confidence: high for syntactic parsing, low for the claimed underlying-source semantics.
- Is grouping strong enough to constrain future splitting? **No.** Every inferred group is a singleton, so it supplies no empirical variant linkage.
- Needed before approval: authoritative pre-augmentation utterance IDs, an original-to-augmented lineage table, augmentation recipe/index semantics, and evidence that identifiers remain stable across provenance shards.

No source grouping was used for a split.

## 13. Duplicate analysis

All files shared the same cheap metadata key, so the partial stage inspected every file. Because each file is smaller than the first-plus-last 128 KiB window, this stage read all 12,091,128,066 bytes. Full-file SHA-256 was then recomputed only for the eight files in matching partial-digest groups: 768,624 bytes.

Results:

- Exact byte-duplicate groups: 4.
- Files in exact duplicate groups: 8.
- Within-speaker groups: 4.
- Cross-speaker groups: 0.
- Potential repeated storage: 384,312 bytes.
- Duplicate normalized paths: 0.
- Duplicate filenames: 0.

The pairs are:

| Group | Speaker | Relative paths |
|---|---:|---|
| dup_000001 | 1387 | `1387/aug_train-00100-of-00204_318.wav`; `1387/aug_train_small-00061-of-00119_81.wav` |
| dup_000002 | 1387 | `1387/aug_train-00100-of-00204_595.wav`; `1387/aug_train_small-00060-of-00119_597.wav` |
| dup_000003 | 1396 | `1396/aug_train-00202-of-00204_370.wav`; `1396/aug_train_small-00118-of-00119_507.wav` |
| dup_000004 | 1446 | `1446/aug_train-00105-of-00204_750.wav`; `1446/aug_train_small-00063-of-00119_369.wav` |

No file was deleted or excluded. Human approval is required for any future duplicate policy.

## 14. Proposed `full_manifest_v2` schema

This is a proposal only; no production full manifest was generated.

| Category | Proposed field | Availability/status in this task |
|---|---|---|
| Stable source identity | `audio_path` | Known; dataset-root-relative forward-slash path |
| Stable source identity | `speaker_id` | Derived from direct parent |
| Stable source identity | `filename` | Known |
| Stable source identity | `manifest_version` | Proposed later; do not populate as an approved production identity now |
| Physical metadata | `sample_rate_hz` | Derived during audit |
| Physical metadata | `channel_count` | Derived during audit |
| Physical metadata | `sample_width_bytes` | Derived during audit |
| Physical metadata | `bits_per_sample` | Derived during audit |
| Physical metadata | `wav_encoding` | Derived during audit |
| Physical metadata | `frame_count` | Derived during audit |
| Physical metadata | `duration_seconds` | Derived during audit |
| Physical metadata | `file_size_bytes` | Known during audit |
| Filename provenance | `provenance` | Derived; never an authoritative split |
| Filename provenance | `provenance_parse_status` | Derived |
| Candidate source group | `candidate_source_group` | Derived syntactically; semantics unapproved |
| Candidate source group | `source_group_confidence` | Derived parse status; not an approved grouping decision |
| Audit findings | `wav_status` | Derived |
| Audit findings | `duplicate_group` | Derived exact-byte group |
| Audit findings | `audit_status` | Derived finding classification |
| Audit findings | `exclusion_reason` | Recommendation only; no approved exclusion |
| Future split/label | `final_split` | **Forbidden to populate in this task** |
| Future split/label | `label` | **Forbidden to populate in this task** |

The approved schema should also bind a config identity and explicitly define nullability, numeric precision, encoding vocabulary, and whether audit recommendations survive into the production manifest.

## 15. Pilot hard-code inventory summary

The targeted audit records 41 findings in `reports/integration_hardcode_inventory_v2.csv`:

| Classification | Findings |
|---|---:|
| Must change before full manifest v2 | 3 |
| Must change before approved split v2 | 4 |
| Must change before Fbank cache v2 | 4 |
| Must change before verification trials v2 | 1 |
| Must change before baseline v2 | 3 |
| Must change before training v2 | 7 |
| Safe reusable production logic | 6 |
| Historical v1 artifact that must remain untouched | 13 |

Unsafe pilot assumptions include the old dataset root, provenance-based trusted/quarantine pools, 31,998/8,504/9,198 split counts, 488 classes and labels 0..487, v1 manifest/cache/trial paths and hashes, fixed `[301,80]` cached shape, fixed 48,000-sample input, 19,528 validation trials, P16K2/window-8/1,000-batch epochs, pilot baseline thresholds, and pilot checkpoint identities/resume state.

## 16. Reusable dataset-independent production components

The following logic can be reused after replacing pilot identities/configuration:

- `SpeechBrainECAPAFrontend` model selection and native `[B,T,80]` feature orientation.
- `mean_var_norm -> embedding_model -> [B,1,192] -> squeeze` encoder path.
- Deterministic sampler mechanics and metadata validation, with approved v2 P/K/window/epoch values.
- AAM-Softmax math, optimizer coverage, BatchNorm policy, and atomic checkpoint mechanics, with a v2-derived class count and identity.
- Portable path and trial ownership validation.
- O(N log N) EER operating-point logic and explicit empirical threshold semantics.
- Cosine scoring and saved-artifact order/shape validation.

The pretrained VoxCeleb classifier remains forbidden.

## 17. Required migration stages

1. **Human audit approval and schema decision:** decide duplicate handling, one-utterance speaker policy, source-group interpretation, and final `full_manifest_v2` schema.
2. **Full manifest v2:** obtain the dataset root from explicit config; write portable paths; derive speaker/file counts; bind manifest version/hash; keep split and label unset.
3. **Approved split v2:** use a versioned split-policy config; derive train/validation/test counts from the resulting speaker-disjoint assignments; build a new speaker-to-label mapping only for approved training speakers.
4. **Fbank cache v2:** derive cache path/schema/version from config; derive `[T,80]` from the approved waveform/frontend contract; bind approved manifest hashes and split row counts.
5. **Verification trials v2:** derive the validation manifest path, speaker/row counts, pair policy, and trial identity/hash from approved validation data.
6. **Baseline v2:** bind the approved cache/trials/protocol identity; select no threshold from v1.
7. **Training v2:** derive class count, P/K/window, epoch length, checkpoint data identity, validation protocol identity, and scheduler policy from approved v2 configs; start without pilot checkpoint state.

No migration stage beyond the audit/governance work was implemented here.

## 18. `AGENTS.md` changes

The root rules now identify VieSpeaker2.0 as the exclusive v2 dataset, forbid merging/reuse of pilot data-dependent artifacts, require direct-parent speaker identity and provenance-neutral splitting, keep source WAVs immutable, prohibit new preprocessing/augmentation and dataset-root writes, reserve final-test access, require separate v2 paths and config/manifest-derived counts, and preserve the dynamic `[B,T,80]` SpeechBrain-to-192-dimensional production path.

No unapproved exact v2 counts, label count, split ratio, feature length, sampler configuration, epoch length, or source-group semantics were added as governance facts.

## 19. Files created and modified

Created:

- `src/dataset_audit_v2.py`
- `scripts/audit_viespeaker2_dataset.py`
- `tests/test_dataset_audit_v2.py`
- `reports/dataset_understanding_v2.json`
- `reports/dataset_understanding_v2.md`
- `reports/dataset_speaker_summary_v2.csv`
- `reports/dataset_provenance_summary_v2.csv`
- `reports/integration_hardcode_inventory_v2.csv`
- Ignored runtime artifacts under `outputs/dataset_understanding_v2/`

Modified:

- `AGENTS.md`
- `reports/CODEX_WORKLOG.md` append-only

No production full manifest, split, label mapping, cache, trial, baseline, or checkpoint was created or modified.

## 20. Exact commands executed

The substantive commands, including required validation, were:

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

Targeted repository inspection did not open, load, hash, stat, or evaluate quarantined v1 final-test manifests, indexes, shards, scores, embeddings, or evaluation outputs.

## 21. Test results

- `py_compile` for all new Python files: PASS.
- Focused synthetic audit suite: 6 tests passed.
- Complete repository unittest discovery: 118 tests passed.
- The first focused run had one test-fixture expectation failure for
  `aug_train.wav`; the fixture was corrected to use the genuinely malformed
  `aug_train-.wav`, then the focused suite passed twice. Production audit logic
  did not change for that fixture correction.
- Audit output schema parsing/read-back: PASS.
- `git diff --check`: PASS.
- CUDA/model/Fbank/training/baseline/verification commands run: 0.

## 22. Dataset preservation evidence

Before and after the audit:

- Files: 125,847 / 125,847.
- Bytes: 12,091,128,066 / 12,091,128,066.
- Directories including root: 1,676 / 1,676.
- Path-size-mtime identity SHA-256: `40462a26fa9fc2aee83cc8a55b02a88087e75bbe97e4af1a79925b00d081a303` / identical.
- Snapshot errors: 0 / 0.
- Dataset-root immediate child-name set: identical.

No file was created inside the dataset root. No WAV was modified, renamed, moved, deleted, resampled, decoded into a waveform tensor, or augmented.

## 23. Limitations

- Header/container inspection cannot measure SNR, reverberation, clipping, VAD quality, loudness normalization, or perceptual corruption.
- Complete waveform samples were not decoded. The duplicate audit read bytes, not audio tensors.
- Exact-byte hashing does not identify acoustically equivalent files with different encodings or headers.
- Filename coordinates have no authoritative semantic documentation and every candidate group is a singleton.
- Candidate folder identity was not cross-checked against external speaker identity records.
- The preservation snapshot binds path, size, and mtime; it is not a retained per-file content-hash manifest.

## 24. Decisions still requiring human approval

- Whether the audit evidence is accepted as the basis for a production full manifest.
- Final `full_manifest_v2` schema and artifact identity rules.
- Treatment of four within-speaker duplicate pairs.
- Eligibility policy for 128 speakers with only one utterance and other low-resource speakers.
- Whether authoritative augmentation-lineage metadata can be supplied.
- Final split policy, counts/ratios, seed, stratification, and speaker eligibility.
- V2 label mapping, feature length/cache schema, trial protocol, sampler configuration, epoch length, baseline identity, and checkpoint identity.

Split v2 must not begin until the audit is reviewed and the relevant decisions are explicitly approved.

## 25. Final result

**PASS** for VieSpeaker2.0 Dataset Understanding and Integration Audit v2, with four within-speaker exact duplicate groups and 128 one-utterance speakers recorded for human review. The dataset has not been approved for splitting by this task.

All later pipeline stages are explicitly deferred.
