# Repository Rules

- Inspect the repository, this policy, and `reports/CODEX_WORKLOG.md` before editing; keep each task limited to its explicitly approved stage. Preserve pre-existing user changes.
- Do not automatically continue to another pipeline stage or preemptively generate downstream artifacts. Complete and review each stage before proceeding under the next explicit instruction.
- Do not commit or push unless explicitly instructed.
- Do not commit datasets, WAVs, environments, pretrained weights, caches, checkpoints, large runtime artifacts, or machine-local path-bound outputs.
- The stable CUDA environment is `.venv-cuda`; do not modify it or the existing CPU environment.

## Production dataset authority

- Main repository: `E:\SpeakerVerification`. Preprocessing belongs to the independent `E:\SpeakerDataPipeline` repository.
- The only approved downstream dataset is `E:\adaptive_augmented_3s`. Do not train directly on the original `E:\adaptive_augmented`, use VieSpeaker/VieSpeaker2.0 as production inputs, or merge these datasets.
- Approved contract: 610 speakers and 60,803 WAV files; each WAV is 16,000 Hz, mono, exactly 48,000 samples and 3.0 seconds. SpeakerDataPipeline preprocessing validation passed with zero failures. These are supplied contract values to verify in the authorized manifest stage, not a new audit performed by this policy update.
- Layout is `dataset_root/speaker_id/*.wav`. Derive authoritative speaker identity from each WAV's direct parent directory.
- Filename suffixes such as `_noise` and `_rev` must not define splits, classification labels, provenance assumptions, or experimental groups.
- Normalization used WebRTC VAD-assisted deterministic contiguous-window cropping for long recordings, preserved exactly 3-second recordings, and symmetrically zero padded short recordings. No additional noise/reverb augmentation was added during normalization.
- Never modify source audio or write into `E:\adaptive_augmented_3s`. Write generated artifacts elsewhere. Do not perform new preprocessing, resampling, channel conversion, VAD, normalization, cropping, padding, or augmentation without an explicitly approved task; approval to generate artifacts does not authorize source changes.

## Historical evidence and stale artifacts

- All data-dependent artifacts derived from `E:\adaptive_augmented`, VieSpeaker, or VieSpeaker2.0 are stale for this production phase unless explicitly regenerated and verified against `E:\adaptive_augmented_3s`.
- This includes full/split/portable manifests, speaker assignments, labels and mappings, Fbank caches and identities, sampler plans/configurations/identities, verification trials, baselines, EERs, thresholds, checkpoint dataset identities, old final-test artifacts, and dataset-specific counts or hashes.
- Regenerate in separate versioned paths during the authorized stage; never relabel an old artifact as belonging to the new dataset or overwrite immutable historical evidence. Do not copy VieSpeaker2.0 data-dependent values into production.
- Preserve old datasets, `manifests/full_manifest.csv`, pretrained weights, caches, checkpoints, and historical reports. Prior v1/v2 identities and run results remain documented in `reports/CODEX_WORKLOG.md` and their versioned reports/configurations; they are historical evidence, not current production authority. Preserve worklog history; future authorized worklog updates should be append-only.
- VieSpeaker2.0's one-time final evaluation is complete. Its final-test manifest, cache, trials, evaluation lock, embeddings, scores, and reports remain immutable. Do not reopen its final-test audio, regenerate artifacts, rerun inference, or rescore. Only an explicitly requested metrics-only recomputation from saved scores is permitted; results must not guide further training, tuning, checkpoint selection, or threshold adjustment.

## Speaker split and evaluation isolation

- Intended production split: all 610 speakers assigned to train 488, validation 61, and final test 61. The split unit is the speaker; every utterance from one speaker stays in exactly one split, with zero speaker overlap. Never randomly split individual utterances.
- Generate and verify deterministic speaker assignments only in the authorized split stage. The approved counts do not constitute an existing split package, seed, label mapping, or approval to reuse earlier assignments or exclusion rules.
- Validation supports checkpoint selection, hyperparameter decisions, threshold selection, model comparison, and verification tuning. Validation and final test must not receive stochastic training augmentation.
- Lock final test after its authorized creation and metadata validation. Do not open or evaluate it early or generate downstream final-test artifacts during train/validation work. Use task-specific paths and avoid recursive inspection of quarantined final-test artifacts.
- Final test must never support hyperparameter tuning, checkpoint/threshold selection, model tuning, or exploratory comparison. Its one-time evaluation requires an explicit final-evaluation task after the training/validation pipeline is finalized, with checkpoint and operational threshold locked from validation.
- Preserve final-test artifacts after evaluation. Any descriptive test-derived EER threshold is non-operational and must not replace the validation-selected threshold.

## Artifact identity and reproducibility

- Every data-dependent artifact must bind to the normalized dataset and its approved upstream identities: dataset fingerprint, full manifest, speaker split, train label mapping, Fbank cache, sampler plan, validation trials, baseline, and checkpoint training identity as applicable.
- Downstream consumers must validate the relevant manifest/split hashes and cache configuration/completion identity hashes before use. Path, tensor shape, or hand-maintained counts alone are insufficient; fail on identity mismatches.
- Derive runtime counts, labels, feature length, sampler settings, epoch size, seeds, and trial configuration from newly approved artifacts/configurations. Verify the supplied dataset/split contract, but do not inherit old constants, hashes, thresholds, or checkpoint/resume identities.
- Prefer deterministic generation with stable ordering, explicit seeds, and deterministic hashing; do not depend on filesystem traversal order, unordered containers, or implicit RNG state.
- Keep committed portable manifests dataset-root-relative. Any exceptional nonportable path requirement needs a strong documented reason; machine-local absolute-path artifacts must remain ignored.
- Preserve exact approved sampler behavior when reused: validate configuration and plan identity, call `set_epoch(epoch)`, and keep exact P x K batches. Where authoritative duplicate groups exist, keep duplicate-group rows out of the same speaker batch. Do not infer those groups from filename suffixes or disable integrity checks based on validation of an old cache.

## Model invariants

- Use the pretrained `speechbrain/spkrec-ecapa-voxceleb` model with project-owned AAM-Softmax for training; do not train ECAPA-TDNN from scratch. Do not redesign the architecture unless a real incompatibility is identified.
- Waveform path: waveform -> SpeechBrain `compute_features` (80-d Log-Mel Fbank) -> `mean_var_norm` -> `embedding_model` -> 192-dimensional speaker embedding.
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

- Cache raw pre-normalization, non-transposed Fbank features. Establish the new dataset's feature length and cache compatibility in the authorized Fbank compatibility stage.
- Reuse suitable existing mechanisms: `CachedFbankDataset`, `HybridShardAwareSpeakerBatchSampler`, AAM-Softmax, AMP, gradient accumulation, frozen BatchNorm running statistics, checkpoint/resume, cosine learning-rate scheduling, fixed validation trials, cosine scoring, EER, and best-checkpoint selection. Reuse of mechanisms does not authorize reuse of their dataset-dependent identities or settings.

## Production stage discipline

- Typical sequence: normalized dataset -> full manifest -> speaker-disjoint split -> portable split manifests -> train label mapping -> Fbank compatibility check -> Fbank cache -> sampler configuration -> fixed validation trials -> untouched pretrained ECAPA validation baseline -> ECAPA + AAM fine-tuning -> validation/checkpoint selection -> one-time final-test evaluation.
- The sequence is guidance, not authorization. Implement only the requested stage or explicitly requested stage package.
- Before real fine-tuning, complete the dedicated repository cleanup/readiness stage and the production README gate below.

## Code readability and focused validation

- Prefer short focused functions, clear names, simple control flow, explicit data flow, small modules, minimal abstraction, straightforward CLI entrypoints, useful errors, and comments explaining why.
- Avoid oversized files, deep helper layers, generic frameworks for simple tasks, duplicated logic, dead branches, speculative extensibility, excessive configuration machinery, unnecessary classes, trivial wrappers, and comments narrating obvious code.
- Existing complexity does not justify new complexity. Simplify a touched component where reasonable within scope while preserving behavior and data integrity; do not perform unrelated large refactors.
- Use focused checks for speaker-disjointness, dataset path identity, label mapping correctness, cache/data compatibility, checkpoint identity, and final-test isolation.
- Keep production validation/integrity logic that protects real data and evaluation correctness. Do not classify it as unnecessary merely because it validates something.
- Do not build a large testing framework or many synthetic test scripts unless necessary. Development-only tests, temporary scripts, synthetic benchmarks, exploratory utilities, and obsolete harnesses are candidates for safe removal in the dedicated cleanup stage.

## Production minimalism and readiness gate

- Target a compact repository where every retained source file has a clear role: environment installation, artifact/cache preparation, fixed validation trials, pretrained baseline, ECAPA+AAM fine-tuning, checkpoint validation/resume, or final evaluation.
- Cleanup must audit obsolete/debug/experimental/migration/one-off/redundant scripts, dead modules, development-only tests, synthetic benchmarks, duplicate implementations, unused CLI entrypoints, stale dataset references/constants, and documentation drift.
- Also audit accidental runtime artifacts, absolute local paths, datasets, caches, checkpoints, environments, and generated outputs. Preserve protected data/evidence; checking ignore coverage does not authorize deleting them.
- Remove legacy files only in an explicitly requested cleanup task, after verifying they are absent from the retained production dependency graph and documented workflow. Age alone is not grounds for deletion.
- Before training, ensure `.gitignore` prevents accidental commits of datasets/WAVs, virtual environments, caches, checkpoints, large runtime artifacts, local-machine outputs, and other non-source production artifacts.

## README gate before real fine-tuning

- Make `README.md` practical and production-ready in its authorized documentation stage, sufficient for another person to clone, install, and run the supported workflow.
- Document the overview, repository structure, supported environment and Python/PyTorch/torchaudio/SpeechBrain versions, installation, virtual environment setup, and dependencies.
- Document dataset structure/audio contract; actual commands for artifact preparation, required Fbank cache, validation trials, pretrained baseline, exact fine-tuning invocation and important arguments, checkpoint/output locations, resume if supported, validation, and final-test evaluation.
- Include final-test isolation, reproducibility details, and important seeds/identities where appropriate. Commands must match retained production entrypoints; do not document obsolete or unused scripts.

## Environment authority

- Stable stack: Python 3.10.11; PyTorch and torchaudio 2.2.0+cu121; SpeechBrain 1.0.3; RTX 3050 Laptop GPU with 4 GB VRAM.
- Existing CUDA environment: `E:\SpeakerVerification\.venv-cuda` (repository-relative `.venv-cuda`, as verified on disk and used in the worklog). The production brief's `E:\SpeakerVerification.venv-cuda` omits the directory separator; do not create or migrate to that path.
- Preserve the CUDA and existing CPU environments. Prefer the stable stack; introduce dependencies or environment changes only for a demonstrated incompatibility within an explicitly authorized task.

## Commit discipline

- Bring each meaningful stage to a commit-ready state before moving on: functional correctness, data integrity, reproducibility, readable implementation, minimal production surface, correct artifact identity, a clean scoped diff, and relevant documentation.
- Commit and push only when explicitly requested. Keep commits logically scoped; use meaningful stage-specific titles such as `feat(data): build normalized dataset manifest` or `feat(split): add deterministic speaker-disjoint dataset split`, with a concise body describing additions, changes, removals, and validation. Avoid vague titles such as "update", "fix stuff", "final", or "done".

## Task 0 boundary

- This task changes only `AGENTS.md`. Read the existing policy and worklog first.
- Do not modify production code, scripts, tests, README, manifests, artifacts, configs, training code, or the worklog; do not implement manifest/split generation, regenerate artifacts, delete legacy files, commit, or push.
- Next recommended stage only: production full manifest + deterministic speaker-disjoint 488 / 61 / 61 split package for `E:\adaptive_augmented_3s`. Do not start it without an explicit request.
