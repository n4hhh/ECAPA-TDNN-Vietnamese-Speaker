# Candidate speaker split v1

> Approved for internal training and evaluation. The original `test` and `part` groups remain quarantined.

# Approval status: APPROVED FOR INTERNAL TRAINING
- This split is approved for internal model development and evaluation.
- The original `test` and `part` provenance groups remain quarantined.

## Input

- Manifest: `manifests\full_manifest.csv`
- Detected columns: `audio_path`, `speaker_id`, `filename_group`
- Rows: 62435
- Unique speakers: 1463

## Trusted pool

- Utterances: 50308
- Speakers: 959
- train_only: 583
- train_small_only: 53
- train_and_train_small: 323

## Eligibility

- Fewer than 5: 271
- 5-9: 119
- At least 10: 569

## Final split

- train: 488 speakers, 31998 utterances
- validation: 100 speakers, 8504 utterances
- test: 100 speakers, 9198 utterances
- excluded: 271 speakers, 608 utterances

## Speaker bucket distribution

| Split | 10-19 | 20-49 | 50+ |
|---|---:|---:|---:|
| train | 85 | 112 | 172 |
| validation | 23 | 30 | 47 |
| test | 23 | 30 | 47 |

## Quarantine audit

| Group | Utterances | Speakers | Trusted overlap | Quarantine-only | In both test and part |
|---|---:|---:|---:|---:|---:|
| test | 2908 | 67 | 0 | 67 | 0 |
| part | 9219 | 437 | 0 | 437 | 0 |

## Validation

- Train/validation overlap: 0
- Train/test overlap: 0
- Validation/test overlap: 0
- Quarantine records assigned to final splits: 0
- All manifest totals reconciled: yes
- Deterministic for the recorded seed: yes
