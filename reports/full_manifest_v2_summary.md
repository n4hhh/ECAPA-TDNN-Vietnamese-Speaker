# VieSpeaker2.0 Production Full Manifest v2

## Goal and scope

This task created and validated the production, unsplit VieSpeaker2.0 full
manifest and its immutable identity metadata. It did not create a
train/validation/test split, labels, eligibility decisions, portable split
manifests, features, trials, model outputs, or checkpoints.

Result: **PASS** for `full_manifest_v2` only. This result is not approval of a
speaker split.

## Inputs used

- Runtime dataset root: `E:\VieSpeaker2.0\augmented_dataset`, read-only.
- Audited per-file inventory:
  `outputs/dataset_understanding_v2/dataset_audit_inventory_nonproduction_v2.csv`.
- Approved audit report: `reports/dataset_understanding_v2.md`.
- Approved audit JSON: `reports/dataset_understanding_v2.json`.
- Dataset identity: `VieSpeaker2.0_augmented_dataset`.

The old pilot dataset was not merged or consulted. The expensive full
duplicate hashing was not repeated; the builder preserved and reconciled the
audited duplicate identities.

## Approved audit identity

| Input | SHA-256 |
|---|---|
| Audit inventory | `6f33cb9510b00f3503af41246349b921d000779b25de9715b9ea68aed1df6cd5` |
| Audit Markdown | `79bd00b9f922b5b3a2d492cf910a0f7d7e6e4c6b1403b4b21c7e1aab15c4493a` |
| Audit JSON | `305a641ac91f4e7fc41c389bf3818b810f59fea5ec18a7722046407582bc9af2` |
| Dataset path/size/mtime snapshot | `40462a26fa9fc2aee83cc8a55b02a88087e75bbe97e4af1a79925b00d081a303` |

The builder fails closed if the inventory schema, audit counts, audit hashes,
dataset snapshot, or per-file path/metadata reconciliation differs from the
approved inputs.

## Manifest schema

The exact CSV column order is:

```text
audio_path,speaker_id,filename,provenance,provenance_parse_status,sample_rate_hz,channel_count,sample_width_bytes,bits_per_sample,wav_encoding,frame_count,duration_seconds,file_size_bytes,wav_status,duplicate_group,candidate_source_group,source_group_parse_status,manifest_version
```

`speaker_id` is the numeric direct-parent directory name only; filenames never
define identity. `provenance` is descriptive and creates no split.
`manifest_version` is `v2`. There are no `final_split`, `label`,
`training_eligible`, or `verification_eligible` columns.

## Portability and ordering

Every `audio_path` is relative to the supplied dataset root, uses `/`, has no
drive prefix, absolute prefix, backslash, empty component, `.` component, or
`..` traversal, and resolves inside the dataset root.

Rows are sorted first by numeric `speaker_id` ascending and then by
`audio_path` using ordinal string ordering.

## Exact reconciliation

| Measure | Validated value |
|---|---:|
| Manifest rows | 125,847 |
| Unique audio paths | 125,847 |
| Unique speaker IDs | 1,675 |
| Readable WAV rows | 125,847 |
| Total bytes | 12,091,128,066 |
| Total duration | 377,541 seconds |
| `train` provenance | 86,197 |
| `train_small` provenance | 16,128 |
| `part` provenance | 17,858 |
| `test` provenance | 5,664 |
| Singleton speakers | 128 |
| Exact duplicate groups | 4 |
| Files in duplicate groups | 8 |
| Cross-speaker duplicate groups | 0 |

All 125,847 audited rows are represented exactly once. All referenced WAVs
exist and reconcile to the audited size and WAV metadata. Every row is
readable 16 kHz, mono, 16-bit PCM with 48,000 frames and a fixed
`3.000000`-second serialization.

## Singleton and duplicate preservation

All 128 singleton speakers remain in the manifest. No low-resource speaker
was excluded.

The audited identifiers `dup_000001` through `dup_000004` are preserved on all
eight duplicate rows. No file was removed, no representative was selected,
and each group remains within exactly one speaker.

## Source-group semantic limitation

`candidate_source_group` and `source_group_parse_status` preserve the audit's
filename-derived syntactic information. They do not establish augmentation
lineage, source recording identity, or a split decision.

## Manifest identity

- Production CSV:
  `manifests/v2/full_manifest_v2.csv`
- Identity JSON:
  `manifests/v2/full_manifest_v2_identity.json`
- Manifest SHA-256:
  `26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f`
- Identity-file SHA-256:
  `f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e`

The identity JSON contains the dataset and manifest versions, schema version,
relative artifact paths and path semantics, sort order, exact counts,
provenance counts, manifest hash, source-audit paths and hashes, dataset
snapshot identity, and creation-tool identity. It contains no dataset-root
absolute path or timestamp. The finalized CSV and JSON were read back and
validated after atomic publication.

## Reproducibility

A second independent build was published under the ignored verification path
`outputs/full_manifest_v2_reproduction_v2/` and compared with the production
pair. Both the CSV and identity JSON were byte-identical.

- CSV: UTF-8 without BOM, LF records, fixed column order, and fixed
  six-decimal duration.
- JSON: UTF-8 without BOM, LF newlines, two-space indentation, lexicographically
  sorted keys, and a final newline.
- Identity timestamps: absent.
- Production publication: CSV and JSON staged, validated, and installed as one
  directory rename. Existing nonidentical output is never overwritten.

## Files created and modified

Created:

- `src/full_manifest_v2.py`
- `scripts/create_full_manifest_v2.py`
- `tests/test_full_manifest_v2.py`
- `manifests/v2/full_manifest_v2.csv`
- `manifests/v2/full_manifest_v2_identity.json`
- `reports/full_manifest_v2_summary.md`
- `reports/full_manifest_v2_summary.json`
- Ignored verification pair under
  `outputs/full_manifest_v2_reproduction_v2/`

Modified:

- `reports/CODEX_WORKLOG.md` by one append-only entry.

All pre-existing dirty and untracked work was preserved.

## Exact commands

```text
.venv-cuda\Scripts\python.exe -m py_compile src\full_manifest_v2.py scripts\create_full_manifest_v2.py tests\test_full_manifest_v2.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_full_manifest_v2 -v
.venv-cuda\Scripts\python.exe scripts\create_full_manifest_v2.py --dataset-root "E:\VieSpeaker2.0\augmented_dataset" --audit-inventory outputs\dataset_understanding_v2\dataset_audit_inventory_nonproduction_v2.csv --output-dir manifests\v2
.venv-cuda\Scripts\python.exe scripts\create_full_manifest_v2.py --dataset-root "E:\VieSpeaker2.0\augmented_dataset" --audit-inventory outputs\dataset_understanding_v2\dataset_audit_inventory_nonproduction_v2.csv --output-dir outputs\full_manifest_v2_reproduction_v2 --reference-dir manifests\v2
.venv-cuda\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
git diff --check
```

## Tests and preservation

- Python compilation: PASS.
- Focused `tests.test_full_manifest_v2`: 12 passed.
- Complete repository discovery: 130 passed.
- `git diff --check`: PASS.
- Production finalized-artifact read-back: PASS.
- Independent byte-for-byte reproduction: PASS for CSV and JSON.
- Both production and reproduction runs reported identical pre/post dataset
  snapshot identities and `dataset_preservation_match: true`.
- No artifact was written beneath the dataset root and no source audio was
  changed.

## Explicit exclusions

No speaker split, final-test access, label mapping, portable split manifest,
deduplication decision, Fbank/cache, SpeechBrain/ECAPA load, embedding,
verification trial, score, EER, threshold, sampler benchmark, training,
checkpoint, commit, or push was performed.

