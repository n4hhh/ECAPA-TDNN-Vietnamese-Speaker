# VieSpeaker2.0 resumed multi-epoch ECAPA-AAM

Result: **FAILED CLOSED**. This run does not satisfy the requested PASS
conditions and does not establish a final governance-locked checkpoint or
threshold.

## Outcome

Epoch 0 was not retrained. Epoch 1 completed exactly 2,969 updates, was
validated on the immutable 20,000-trial protocol, and improved EER from
`0.064` to `0.0602` at empirical threshold `0.16285736858844757`.

Epoch 2 did not complete and was not validated. Before the optimizer step for
batch 2,532 (attempted global step 8,470), the fail-closed guard detected an
elementwise non-finite gradient in ECAPA parameter
`blocks.0.norm.norm.weight`. No optimizer update was applied for that batch.
No altered-hyperparameter retry was attempted.

| epoch | completed updates | global step | validation EER | threshold | status |
|---:|---:|---:|---:|---:|---|
| 0 | 2,969 | 2,969 | 0.0640 | 0.161909759045 | approved start; not retrained |
| 1 | 2,969 | 5,938 | 0.0602 | 0.162857368588 | completed and validated |
| 2 | incomplete | last atomic 7,969 | — | — | failed before batch-2,532 optimizer step |

## Checkpoints

- Last valid atomic recovery checkpoint:
  `outputs/ecapa_aam_multiepoch_v2/last.pt`,
  SHA-256
  `3b8d613012ee8340a30ad264729c83ea97501b793e6c97e1d0ed78555e5c96a0`,
  phase `training`, epoch 2 batch 2,031, global step 7,969.
- Provisional best completed checkpoint:
  `outputs/ecapa_aam_multiepoch_v2/best.pt`, SHA-256
  `a7820ec84eec75acaca595219a81603d7d9056382024d004617613150271711b`.
  It is byte-identical to `epoch_001.pt`.

The epoch-1 checkpoint and threshold are validation evidence only. Because the
configured training run failed before an approved early-stopping or max-epoch
stop, they were not added to governance and are not locked for final
evaluation.

## Validation

Epoch-1 validation extracted exactly 15,355 float32 CPU embeddings of shape
`[15355,192]` and scored 10,000 positive plus 10,000 negative fixed trials.
FAR and FRR were both `0.0602`; TP/TN/FP/FN were
`9398/9398/602/602`. Accuracy, precision, recall/TPR, specificity/TNR, and F1
were all `0.9398`.

Extraction took `82.61652979999963` seconds at
`185.8586899882119` utterances/second. Peak validation allocated/reserved CUDA
memory was `2835630080 / 3642753024` bytes.

## Audit

All approved starting, manifest, mapping, cache, index, sampler, and fixed
trial identities remained unchanged. Source audio, cache shards, environments,
pretrained weights, and the original epoch-zero checkpoint were not modified.

The final-test manifest was not opened, statted, hashed, parsed, or loaded.
No final-test/excluded audio, feature, trial, score, threshold application, or
evaluation occurred.

All 23 new focused tests, 74 focused/regression tests, and all 200 repository
tests passed. No commit or push occurred. PASS-only `AGENTS.md` governance and
worklog updates were intentionally not made.
