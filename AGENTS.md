# Repository Rules

- Inspect the repository before editing and keep every task limited to the requested scope.
- Do not automatically continue to the next pipeline stage.
- Do not commit or push unless explicitly requested.
- Preserve source audio and `manifests/full_manifest.csv`.
- Do not commit datasets, checkpoints, environments, caches, or manifests containing absolute local paths.
- The stable CUDA environment is `.venv-cuda`; do not modify the existing CPU environment.
- Use the pretrained `speechbrain/spkrec-ecapa-voxceleb` model. Do not train ECAPA-TDNN from scratch.

The production embedding path is:

```text
waveform [B, 48000]
-> compute_features [B, 301, 80]
-> mean_var_norm [B, 301, 80]
-> embedding_model
-> embedding [B, 1, 192]
```

- Do not transpose SpeechBrain features.
- Do not replace the pretrained SpeechBrain frontend with `src/fbank.py`.
- Current filename groups are provenance labels, not authoritative final splits.
- `train_small` may be used for smoke tests.
- Combine `train` and `train_small` by speaker ID for the trusted pool.
- Keep `test` and `part` quarantined until explicitly approved.
- Final train, validation, and test splits must be speaker-disjoint.
- Keep all utterances from one speaker in one final split.
- Validation and test data must not receive stochastic training augmentation.
- Do not use the pretrained VoxCeleb classifier for the Vietnamese task.
- Use validation data to select checkpoints and thresholds; keep the final test split untouched until final evaluation.
- Treat generated dataset splits as candidates until explicitly approved.