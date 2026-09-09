# Adaptive Augmented 3-Second Speaker Verification

`adaptive_augmented_3s_v1` is the repository's single production experiment. It fine-tunes the pretrained `speechbrain/spkrec-ecapa-voxceleb` ECAPA-TDNN encoder for speaker verification using cached SpeechBrain Fbank features and a project-owned AAM-Softmax training head.

The experiment is complete and closed. The final test set was used once for final reporting only.

## Architecture

Training uses full ECAPA fine-tuning with 192-dimensional embeddings and AAM-Softmax. BatchNorm affine parameters are trainable while running statistics remain frozen. The inference path is:

```text
raw cached Fbank [B, 301, 80] -> mean_var_norm -> ECAPA embedding_model -> [B, 192] -> cosine scoring
```

The pretrained VoxCeleb classifier and the AAM head are never used for verification scoring.

## Environment and dataset

Use Windows PowerShell, Python 3.10.11, PyTorch/torchaudio 2.2.0+cu121, SpeechBrain 1.0.3, and CUDA. The validated hardware is an RTX 3050 Laptop GPU with 4 GB VRAM.

```powershell
$Python = ".\.venv-cuda\Scripts\python.exe"
$DatasetRoot = "E:\adaptive_augmented_3s"
```

The immutable dataset has 60,803 WAV files from 610 speakers. Every file is mono, 16 kHz, and exactly 3.0 seconds. The speaker-disjoint split is 48,640 train utterances from 488 speakers, 6,076 validation utterances from 61 speakers, and 6,087 final-test utterances from 61 speakers.

## Production workflow

These are the active, repository-specific commands. Existing approved artifacts validate their identities and refuse incompatible replacement.

```powershell
& $Python scripts\create_adaptive_augmented_3s_package.py --dataset-root $DatasetRoot
& $Python scripts\build_adaptive_augmented_3s_fbank_cache.py --dataset-root $DatasetRoot --cache-root outputs\fbank_cache_adaptive_augmented_3s_v1 --device cuda:0 --batch-size 64 --shard-size 512
& $Python scripts\generate_adaptive_augmented_3s_validation_trials.py
& $Python scripts\evaluate_adaptive_augmented_3s_pretrained_validation.py --batch-size 32
& $Python scripts\run_adaptive_augmented_3s_training.py --run --device cuda:0
```

Training uses logical batches of 32, microbatches of 4, accumulation of 8, AdamW, ECAPA/AAM learning rates of `1e-5`/`1e-3`, weight decay `1e-4`, cosine decay, a 10-epoch maximum, and early-stopping patience of 2. The immutable training contract is [adaptive_augmented_3s_training_v1.json](configs/adaptive_augmented_3s_training_v1.json).

`last.pt` is the rolling resume checkpoint. Resume only the same approved run with:

```powershell
& $Python scripts\run_adaptive_augmented_3s_training.py --run --resume outputs\ecapa_aam_adaptive_augmented_3s_v1\last.pt --device cuda:0
```

Outputs are under `outputs/ecapa_aam_adaptive_augmented_3s_v1/`: `best.pt` is validation-selected and `last.pt` is resumable. The final-test package uses separate frozen trials, an isolated cache, and [adaptive_augmented_3s_final_evaluation_v1.json](configs/adaptive_augmented_3s_final_evaluation_v1.json). Its evaluator is closed and must not be rerun.

## Validation and final-test protocol

Validation uses 20,000 frozen trials for checkpoint selection. The final-test set was not used for training, hyperparameter tuning, checkpoint selection, or threshold selection. Its empirical EER threshold is descriptive evaluation information only; it is not a deployment threshold.

| Evaluation | EER |
|---|---:|
| Pretrained ECAPA — validation | 12.16% |
| Fine-tuned best — validation | 5.27% |
| Fine-tuned best — final test | 6.93% |

The two validation values share the same protocol. The final-test value comes from the separate unseen final-test protocol and is not a direct cross-set improvement comparison. The best checkpoint is epoch 6; the validation-to-final-test generalization gap is 0.0166. The compact result record is [adaptive_augmented_3s_v1_final_results.json](reports/adaptive_augmented_3s_v1_final_results.json).

## Repository structure

| Path | Purpose |
|---|---|
| `configs/` | Immutable training and final-evaluation bindings |
| `manifests/` and `splits/` | Portable dataset, split, label, and frozen-trial identities |
| `src/` | Production package, cache, sampler, training, verification, and evaluation code |
| `scripts/` | Active production entry points |
| `reports/` | Final observed results and pretrained validation baseline |
| `outputs/` | Ignored local caches and checkpoints |

## Using a new dataset

The current scripts and immutable identities are specific to `adaptive_augmented_3s_v1`; there is no generic `prepare_dataset.py` command. For a new dataset, recreate and bind a new dataset identity, manifests, speaker-disjoint split, speaker-to-label map, Fbank-cache identity, frozen validation trials, training config, output namespace, and final-test package.

The normal sequence is:

```text
new audio -> normalize and verify contract -> manifests and speaker IDs -> speaker-disjoint split
-> Fbank cache -> frozen validation trials -> disposable one-batch dry-run -> fine-tuning
-> validation-only best selection -> frozen final-test package -> one final evaluation
```

### Required checks

- Valid audio paths, direct-parent speaker ownership, and labels.
- Expected sample rate, channels, and duration.
- Speaker-disjoint train, validation, and final-test splits.
- Valid Fbank shape, dtype, count, and identity.
- Valid frozen verification trials.
- One disposable train-batch dry-run.
- No final-test access before validation freezes model selection.

### Development checks that do not need to be repeated

Do not repeat cache or sampler benchmarks, SpeechBrain tensor-shape investigations, gradient-accumulation or AMP audits, checkpoint/resume audits, migration tests, historical comparisons, or retired smoke-test suites unless a concrete new incompatibility requires one.
