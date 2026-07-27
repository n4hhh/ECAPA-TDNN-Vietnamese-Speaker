# Portable manifest v1 summary

## Input

- Detected columns: `audio_path`, `speaker_id`, `filename_group`
- Dataset root: supplied at runtime (not persisted for portability)
- Rows: 62435
- Speakers: 1463

## Output manifests

- Train: 31998 rows, 488 speakers
- Validation: 8504 rows, 100 speakers
- Test: 9198 rows, 100 speakers
- Path schema: dataset-root-relative WAV path
- Path separator: `/`
- Path existence: PASS

## Label mapping

- Mapped speakers: 488
- Minimum label: 0
- Maximum label: 487
- Unique and contiguous: PASS

## Validation

- Train/validation intersection: 0
- Train/test intersection: 0
- Validation/test intersection: 0
- Excluded rows included: 0
- Quarantine rows included: 0
- Duplicate paths: 0
- Absolute paths: 0
- Unresolved paths: 0
- Approved totals reconciled: PASS
- Deterministic rerun (all CSV and JSON decision outputs byte-for-byte): PASS
- Tests (`unittest discover`, 7 tests): PASS

## Overall result

PASS
