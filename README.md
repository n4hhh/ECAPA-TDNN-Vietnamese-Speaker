# Vietnamese Speaker Verification

This repository fine-tunes the pretrained
[`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
encoder for the `adaptive_augmented_3s_v1` speaker-verification package.
It uses cached SpeechBrain Fbank features, a project-owned AAM-Softmax head,
and fixed validation trials. The pretrained VoxCeleb classifier is never used
for Vietnamese training or scoring.

## Environment

The supported stack is Windows PowerShell, Python 3.10.11, PyTorch and
torchaudio 2.2.0+cu121, SpeechBrain 1.0.3, and CUDA. The validated hardware
is an RTX 3050 Laptop GPU with 4 GB VRAM.

```powershell
py -3.10 -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install --upgrade pip
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements-cuda.txt
$Python = ".\.venv-cuda\Scripts\python.exe"
```

The `.venv-cuda` directory, pretrained model files, caches, and checkpoints
are local runtime artifacts and are ignored by Git.

## Dataset contract

The external, read-only dataset root is:

```powershell
$DatasetRoot = "E:\adaptive_augmented_3s"
```

It contains direct speaker folders:

```text
adaptive_augmented_3s/
  <speaker_id>/
    *.wav
```

The approved package contains 60,803 mono 16 kHz WAV files, each exactly
three seconds long, across 610 speakers. The speaker-disjoint split has
48,640 train utterances from 488 speakers, 6,076 validation utterances from
61 speakers, and a locked final test split of 61 speakers. Source audio is
immutable.

## Production workflow

All commands are stage-specific. Existing approved artifacts are immutable;
the preparation scripts validate compatible existing output and refuse to
replace conflicting output.

### 1. Dataset package and split

```powershell
& $Python scripts\create_adaptive_augmented_3s_package.py `
  --dataset-root $DatasetRoot
```

This creates the portable manifests, speaker-disjoint split, label map, and
identities under `manifests/` and `splits/`.

### 2. Raw Fbank cache

The production trainer consumes the approved cache at
`outputs/fbank_cache_adaptive_augmented_3s_v1`.

```powershell
& $Python scripts\build_adaptive_augmented_3s_fbank_cache.py `
  --dataset-root $DatasetRoot `
  --cache-root outputs\fbank_cache_adaptive_augmented_3s_v1 `
  --device cuda:0 `
  --batch-size 64 `
  --shard-size 512
```

Cached features are float32 `[301, 80]` raw Fbank tensors before
`mean_var_norm`; they are never transposed.

### 3. Frozen validation trials

```powershell
& $Python scripts\generate_adaptive_augmented_3s_validation_trials.py
```

The fixed package contains 10,000 genuine and 10,000 impostor validation
trials. It is validation-only input for model selection and threshold work.

### 4. Untouched pretrained validation baseline

```powershell
& $Python scripts\evaluate_adaptive_augmented_3s_pretrained_validation.py `
  --batch-size 32
```

This evaluates only the pretrained encoder on the frozen validation trials;
it does not train AAM-Softmax or modify ECAPA weights.

### 5. Production fine-tuning

```powershell
& $Python scripts\run_adaptive_augmented_3s_training.py `
  --run `
  --device cuda:0
```

The default runtime directory is:

```text
outputs/ecapa_aam_adaptive_augmented_3s_v1/
```

The trainer validates cache, trial, model, and configuration identities before
training. It fine-tunes all ECAPA parameters, keeps BatchNorm affine
parameters trainable while freezing their running statistics, and trains an
AAM head with 488 classes, 192-dimensional embeddings, margin 0.2, and scale
30.

Each optimizer update is one logical 32-utterance batch: 16 speakers × 2
utterances, round-robin reordered into eight microbatches of four. ECAPA runs
under FP16 autocast; AAM and cross-entropy run in FP32. AdamW uses learning
rates `1e-5` for ECAPA and `1e-3` for AAM, with weight decay `1e-4`.

Cosine learning-rate decay advances once per successful optimizer update. The
schedule permits 1,520 updates per epoch, at most 10 epochs, and 15,200
updates total.

### Resume

`last.pt` is the rolling checkpoint. Resume the same output directory with:

```powershell
& $Python scripts\run_adaptive_augmented_3s_training.py `
  --run `
  --resume outputs\ecapa_aam_adaptive_augmented_3s_v1\last.pt `
  --device cuda:0
```

Resume restores ECAPA, AAM, optimizer, cosine scheduler, GradScaler, sampler
epoch/cursor, BatchNorm reference state, early-stopping state, and RNG state.
It rejects a checkpoint whose configuration, cache/trial identities, or
pretrained model binding differs.

### Validation and checkpoint selection

Validation runs once after each completed epoch using cosine similarity on all
20,000 frozen validation trials. Minimum validation EER selects `best.pt`.
Early stopping uses patience 2 and requires at least `0.0001` EER improvement.

- `last.pt`: latest resumable training state.
- `best.pt`: checkpoint with the lowest selected validation EER.

## Final-test isolation

The final test is locked throughout model development. Do not open final-test
audio, manifests, cached features, trials, scores, or embeddings during
training, validation, checkpoint selection, or threshold selection.

There is no adaptive final-test evaluation CLI in the current production
workflow. Its command and protocol remain a later finalization requirement
after the training checkpoint and operational threshold are locked from
validation.

## Repository map

| Path | Role |
|---|---|
| `configs/adaptive_augmented_3s_training_v1.json` | Immutable training configuration and identities |
| `src/adaptive_augmented_3s_package.py` | Dataset package and speaker-disjoint split generation |
| `scripts/build_adaptive_augmented_3s_fbank_cache.py` | Train/validation Fbank cache builder |
| `src/adaptive_augmented_3s_training_input.py` | Cache-backed train and validation inputs |
| `src/cached_fbank_samplers.py` | Deterministic P × K sampler |
| `scripts/generate_adaptive_augmented_3s_validation_trials.py` | Frozen validation trial generator |
| `scripts/evaluate_adaptive_augmented_3s_pretrained_validation.py` | Untouched pretrained validation baseline |
| `scripts/run_adaptive_augmented_3s_training.py` | Production ECAPA/AAM trainer |

Tracked source and identities remain in the repository. Datasets, pretrained
weights, cache shards, checkpoints, embeddings, scores, environments, and
runtime outputs remain ignored.
