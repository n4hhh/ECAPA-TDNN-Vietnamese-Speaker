# ECAPA-TDNN Vietnamese Speaker Verification with VieSpeaker2.0

This repository implements a complete Vietnamese speaker-verification
pipeline using a pretrained SpeechBrain ECAPA-TDNN encoder, cached raw Fbank
features, AAM-Softmax fine-tuning, fixed validation trials, and a one-time
final evaluation.

The task is **text-independent speaker verification**, not speaker
identification: given two utterances, the system estimates whether they were
spoken by the same person.

## Project overview

The encoder starts from
`speechbrain/spkrec-ecapa-voxceleb`. Its pretrained VoxCeleb classifier is
never used. During fine-tuning, AAM-Softmax classification over the Vietnamese
training speakers provides the learning objective. During verification, the
system extracts 192-dimensional embeddings and compares pairs with cosine
similarity.

Validation data is used to select checkpoints and the operational threshold.
The final test was evaluated exactly once with the threshold already locked on
validation.

```text
VieSpeaker2.0 audit
-> production manifest
-> speaker-disjoint split
-> raw SpeechBrain Fbank cache
-> speaker-balanced shard-aware sampling
-> ECAPA-TDNN + AAM-Softmax fine-tuning
-> fixed validation verification
-> best checkpoint selection
-> one-time final evaluation
```

## Dataset

VieSpeaker2.0 is supplied externally and is not committed to Git. It fully
replaces the historical VieSpeaker v1 pilot dataset for this v2 pipeline; the
two datasets must not be merged.

The complete read-only audit found:

| Property | Value |
|---|---:|
| WAV files | 125,847 |
| Speakers | 1,675 |
| Total duration | 104.8725 hours |
| Sample rate | 16 kHz |
| Channels | Mono |
| Encoding | 16-bit PCM |
| Frames per file | 48,000 |
| Duration per file | 3.0 seconds |

Speaker identity is derived only from each WAV file's direct parent directory.
Filename tokens such as `train`, `train_small`, `part`, and `test` describe
source provenance; they do not define the final split.

The audit found four exact duplicate groups containing eight files. Every
duplicate group is within one speaker. Source audio is immutable: the pipeline
does not rewrite, resample, convert, normalize, crop, pad, apply VAD to, or
augment these finalized WAVs.

### Final speaker-disjoint split

| Split | Speakers | Utterances |
|---|---:|---:|
| Train | 1,347 | 95,009 |
| Validation | 100 | 15,355 |
| Final test | 100 | 15,355 |
| Excluded singleton speakers | 128 | 128 |

The excluded speakers are exactly the speakers with one utterance. They remain
represented in the production full manifest but cannot supply a positive
verification pair. Training labels are contiguous from `0` through `1346`;
validation and final-test labels are `-1`. No speaker occurs in more than one
split.

The repository tracks portable manifests and their identities, not the
dataset files.

## Dataset placement

Supply the external root as a PowerShell variable:

```powershell
$DatasetRoot = "E:\VieSpeaker2.0\augmented_dataset"
```

The directory layout must be:

```text
augmented_dataset/
  <speaker_id>/
    *.wav
```

Do not merge this directory with the historical VieSpeaker dataset.

## Feature and model contract

SpeechBrain produces 80-dimensional Fbank frames. The production cache stores
raw, pre-normalization, finite float32 features in native, non-transposed
orientation:

```text
cached Fbank [B, 301, 80]
-> mean_var_norm
-> embedding_model
-> embedding [B, 1, 192]
-> squeeze
-> embedding [B, 192]
```

Caching avoids repeating waveform frontend extraction during training and
verification. `src/fbank.py` is not the production SpeechBrain feature path.
Neither Vietnamese training nor verification uses the pretrained VoxCeleb
classifier.

## Training configuration

| Setting | Approved value |
|---|---|
| Encoder | `speechbrain/spkrec-ecapa-voxceleb` |
| Embedding dimension | 192 |
| Training speakers/classes | 1,347 |
| Loss | AAM-Softmax |
| AAM margin | 0.2 |
| AAM scale | 30 |
| Sampler | `HybridShardAwareSpeakerBatchSampler` |
| P x K | 16 x 2 |
| Logical batch | 32 |
| Physical microbatch | 4 |
| Gradient accumulation | 8 microbatches |
| Updates per epoch | 2,969 |
| Logical selections per epoch | 95,008 |
| Optimizer | AdamW |
| ECAPA base learning rate | `1e-5` |
| AAM base learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Precision | FP16 ECAPA; FP32 AAM logits and loss |
| Scheduler | Cosine multiplier from 1.0 to 0.1 over resumed epochs |
| BatchNorm | Running statistics frozen; affine parameters trainable |
| Maximum completed epoch | 4 |
| Best selected epoch | 3 |

The deterministic AMP recovery policy permits at most one retry of the exact
same materialized logical batch at half the GradScaler scale. A failed attempt
does not advance the optimizer, scheduler, cursor, or global step. Three
isolated overflows were recovered, and no logical batch was skipped.

## Results

All validation rows use the same fixed 10,000-positive/10,000-negative trial
protocol.

| Stage | EER | Notes |
|---|---:|---|
| Untouched pretrained validation baseline | 12.57% | Fixed validation trials |
| Fine-tuned epoch 0 validation | 6.40% | First completed v2 epoch |
| Fine-tuned epoch 1 validation | 6.02% | Improved |
| Fine-tuned epoch 2 validation | 5.87% | Improved |
| Fine-tuned epoch 3 validation | 5.84% | Selected best checkpoint |
| Fine-tuned epoch 4 validation | 5.84% | Tie; earlier epoch retained |
| Final-test descriptive EER | 4.53% | Non-operational diagnostic |

### Primary final-test result at the locked validation threshold

The sole operational threshold is `0.16545939445495605`, selected on epoch-3
validation before final-test access. The decision rule accepts a pair as the
same speaker when `cosine_score >= threshold`.

| Metric | Final-test result |
|---|---:|
| TP | 9,598 |
| TN | 9,426 |
| FP | 574 |
| FN | 402 |
| FAR | 5.74% |
| FRR | 4.02% |
| Accuracy | 95.12% |
| Precision | 94.3571% |
| Recall / TPR | 95.98% |
| Specificity / TNR | 94.26% |
| F1 | 95.1616% |

The final-test trial set is artificially balanced 50/50. Accuracy, precision,
and F1 therefore describe this protocol and do not represent real-world
same-speaker/different-speaker prevalence.

The test-derived EER threshold `0.17728488147258759` is a **descriptive,
non-operational diagnostic**. It did not replace the locked validation
threshold, influence checkpoint selection, or authorize further tuning. The
lower descriptive test EER is not presented as evidence of tuning success or
performance beyond this dataset.

## Selected checkpoint

Checkpoints are intentionally excluded from Git because of their size. The
approved final model must be placed at:

- Local runtime path: `outputs/ecapa_aam_multiepoch_v2/best.pt`
- Selected epoch: 3
- Byte-identical epoch checkpoint: `epoch_003.pt`
- SHA-256:
  `ba9d989c1b6a922f3f392cd297fb771bba05319d3fad99df740ac889d2836d6f`

```text
Checkpoint download: <ADD_EXTERNAL_CHECKPOINT_URL>
```

Replace the placeholder only after the checkpoint is uploaded separately to a
real Google Drive, OneDrive, or GitHub Release location. No download URL is
currently asserted by this repository.

Verify a local copy in PowerShell:

```powershell
Get-FileHash `
  outputs\ecapa_aam_multiepoch_v2\best.pt `
  -Algorithm SHA256
```

A checkpoint with a different hash is not the approved final model.

## Repository structure

| Path | Purpose |
|---|---|
| `configs/v2/` | Locked sampler and final-evaluation configurations |
| `manifests/v2/` | Production full manifest and identity |
| `manifests/portable_v2/` | Portable split manifests and label mapping |
| `manifests/verification_v2/` | Fixed validation and final-test trial packages |
| `splits/v2/` | Speaker assignments, split policy, and identity |
| `src/` | Reusable dataset, cache, sampler, training, verification, and metric logic |
| `scripts/` | Audited stage entry points |
| `tests/` | Synthetic and regression tests |
| `reports/` | Tracked summaries and final results |
| `outputs/` | Ignored caches, checkpoints, embeddings, scores, logs, and runtime state |

## Environment and installation

The completed workflow used Windows PowerShell, Python 3.10.11, PyTorch
2.2.0+cu121, torchaudio 2.2.0+cu121, SpeechBrain 1.0.3, and a CUDA-capable
NVIDIA GPU. The approved run used an NVIDIA GeForce RTX 3050 Laptop GPU with
4 GB VRAM. These report the validated environment; they are not a general
compatibility claim.

The tracked CUDA dependency file supports this setup:

```powershell
py -3.10 -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install -r .\requirements-cuda.txt
$Python = ".\.venv-cuda\Scripts\python.exe"
```

`requirements.txt` records the broader dependency set, while
`requirements-cuda.txt` pins the CUDA 12.1 environment used by the completed
run. Do not commit or share `.venv-cuda`.

## Reproduction commands

The commands below document the actual stage interfaces in pipeline order.
They are intended for a reviewed reconstruction in fresh output locations,
with sufficient external storage and the required upstream artifacts. They
are **not** an instruction to rerun stages in this finalized repository.
Several scripts deliberately fail closed when immutable artifacts already
exist or identities differ.

Set these variables first:

```powershell
$Python = ".\.venv-cuda\Scripts\python.exe"
$DatasetRoot = "E:\VieSpeaker2.0\augmented_dataset"
```

### 1. Dataset audit

```powershell
& $Python scripts\audit_viespeaker2_dataset.py `
  --dataset-root $DatasetRoot
```

### 2. Production full manifest

This requires the audit inventory generated under the ignored audit runtime
directory.

```powershell
& $Python scripts\create_full_manifest_v2.py `
  --dataset-root $DatasetRoot `
  --audit-inventory outputs\dataset_understanding_v2\dataset_audit_inventory_nonproduction_v2.csv
```

### 3. Speaker-disjoint split package

This requires the approved full manifest and its identity.

```powershell
& $Python scripts\create_speaker_split_package_v2.py
```

### 4. Train/validation Fbank cache

This requires the approved portable train/validation manifests and local
SpeechBrain pretrained files. The cache needs substantial ignored storage.

```powershell
& $Python scripts\precompute_speechbrain_fbank_v2.py `
  --dataset-root $DatasetRoot `
  --device cuda:0 `
  --batch-size 64 `
  --shard-size 256 `
  --cache-dir outputs\fbank_cache_v2
```

An interrupted compatible build may add `--resume`; finalized shards are
validated before being skipped.

### 5. Training readiness and fixed validation trials

These stages require the approved train/validation cache and portable
manifests.

```powershell
& $Python scripts\prepare_training_readiness_v2.py
& $Python scripts\generate_validation_trials_v2.py
```

### 6. Untouched pretrained validation baseline

```powershell
& $Python scripts\evaluate_pretrained_validation_baseline_v2.py `
  --device cuda:0 `
  --batch-size 64
```

### 7. Epoch-zero training

This requires the approved cache, sampler, validation trials, and local
SpeechBrain pretrained model files. The script refuses to overwrite existing
v2 epoch artifacts.

```powershell
& $Python scripts\run_ecapa_aam_one_epoch_v2.py --device cuda:0
```

### 8. Historical multi-epoch recovery and completion

The current script is bound to the approved historical recovery state,
including the exact epoch-zero checkpoint, provisional epoch-one artifacts,
failure records, and recovery `last.pt`. It is not a generic resume command
and is not runnable from an arbitrary checkpoint or from the finalized model.
The audited invocation was:

```powershell
& $Python scripts\run_ecapa_aam_multiepoch_v2.py `
  --device cuda:0 `
  --resume outputs\ecapa_aam_multiepoch_v2\last.pt `
  --amp-overflow-retries 1 `
  --amp-overflow-scale-factor 0.5
```

### 9. Historical one-time final evaluation

The following commands document the completed protocol only. **Do not run
them in this repository now.** The lock, final-test cache, embeddings, scores,
and runtime state are finalized and immutable; the normal evaluator explicitly
refuses a second inference or scoring run.

```powershell
# Historical protocol-lock command — do not rerun after finalization.
& $Python scripts\prepare_final_evaluation_v2.py --seed 20260729

# Historical final-test cache command — do not re-extract the finalized cache.
& $Python scripts\precompute_final_test_fbank_v2.py `
  --dataset-root $DatasetRoot `
  --device cuda:0 `
  --batch-size 64 `
  --shard-size 256 `
  --cache-dir outputs\fbank_cache_final_test_v2

# Historical authoritative inference command — never rerun after finalization.
& $Python scripts\evaluate_final_test_v2.py `
  --device cuda:0 `
  --batch-size 64
```

### 10. Allowed post-finalization metrics-only verification

This is the only permitted final-test recalculation. It requires the ignored,
immutable saved score artifact and does not load SpeechBrain, CUDA, WAVs,
Fbank shards, or embeddings and does not rescore trials.

```powershell
& $Python scripts\recompute_final_test_metrics_v2.py
```

### 11. Unit tests

```powershell
& $Python -m unittest discover -s tests -p "test_*.py" -v
```

## Artifacts and reproducibility

- Tracked: manifests, configs, split and trial identities, reports, scripts,
  and tests.
- Not tracked: dataset audio, Fbank caches, checkpoints, embeddings, scores,
  Python environments, logs, and runtime state.
- SHA-256 identities bind the production manifest, speaker split, portable
  manifests, caches, trial protocols, checkpoints, and final evaluation.
- The split is deterministic and speaker-disjoint.
- Validation trials are fixed.
- Final-test trials and the evaluation configuration were locked before
  final-test audio or model access.
- The final checkpoint and sole operational threshold were selected using
  validation before final-test evaluation.
- Final-test inference is finalized and must not be rerun.
- Metrics-only recomputation from the immutable saved scores is permitted.

## Limitations

- Results are specific to VieSpeaker2.0 and the fixed validation/final-test
  protocols; no broader generalization claim is made.
- Every supplied utterance is three seconds long, so the evaluation does not
  represent wider real-world duration variation.
- Balanced verification trials do not represent deployment prevalence.
- The evaluation does not establish robustness to arbitrary microphones,
  channels, codecs, background conditions, spoofing, or unseen domains beyond
  the evaluated data.
- This project is not an anti-spoofing system.
- The descriptive final-test EER threshold is not operational.
- No claim of state-of-the-art or production-ready performance is made.

## Reports and further reading

- [Dataset understanding](reports/dataset_understanding_v2.md)
- [Production full manifest](reports/full_manifest_v2_summary.md)
- [Speaker-disjoint split](reports/speaker_split_v2_summary.md)
- [Train/validation Fbank cache](reports/fbank_cache_v2_summary.md)
- [Training sampler](reports/training_sampler_v2.md)
- [Fixed validation trials](reports/validation_trials_v2.md)
- [Untouched pretrained validation baseline](reports/pretrained_ecapa_validation_baseline_v2.md)
- [Epoch-zero training](reports/ecapa_aam_one_epoch_v2.md)
- [Final multi-epoch training](reports/ecapa_aam_multiepoch_v2_final.md)
- [AMP recovery](reports/ecapa_aam_multiepoch_v2_recovery.md)
- [Final-test Fbank cache](reports/final_test_fbank_cache_v2.md)
- [One-time final evaluation](reports/final_evaluation_v2.md)

## Project status

| Stage | Status |
|---|---|
| Dataset integration | Complete |
| Speaker-disjoint split | Complete |
| Train/validation cache | Complete |
| Training | Complete |
| Validation checkpoint selection | Complete |
| One-time final evaluation | Complete and finalized |
| Final model checkpoint | External runtime artifact; not in Git; download link pending |
| Deployment integration | Not claimed |

Further training, checkpoint selection, or threshold tuning based on the
final-test results is not permitted by the project protocol.
