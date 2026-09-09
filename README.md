# Adaptive Augmented 3-Second Speaker Verification

This repository is the production pipeline for `adaptive_augmented_3s_v1`. It fine-tunes `speechbrain/spkrec-ecapa-voxceleb` with a project-owned AAM-Softmax head and fixed speaker-verification trials.

Use Python 3.10.11 with PyTorch/torchaudio 2.2.0+cu121 and SpeechBrain 1.0.3:

```powershell
$Python = ".\.venv-cuda\Scripts\python.exe"
$DatasetRoot = "E:\adaptive_augmented_3s"
```

The immutable source dataset has 60,803 mono 16 kHz, three-second WAV files for 610 speakers. The speaker-disjoint package has 488 train, 61 validation, and 61 final-test speakers.

```powershell
& $Python scripts\create_adaptive_augmented_3s_package.py --dataset-root $DatasetRoot
& $Python scripts\build_adaptive_augmented_3s_fbank_cache.py --dataset-root $DatasetRoot --cache-root outputs\fbank_cache_adaptive_augmented_3s_v1 --device cuda:0 --batch-size 64 --shard-size 512
& $Python scripts\generate_adaptive_augmented_3s_validation_trials.py
& $Python scripts\evaluate_adaptive_augmented_3s_pretrained_validation.py --batch-size 32
& $Python scripts\run_adaptive_augmented_3s_training.py --run --device cuda:0
```

Training uses `configs/adaptive_augmented_3s_training_v1.json` and writes `best.pt` and `last.pt` under `outputs/ecapa_aam_adaptive_augmented_3s_v1/`.

Epoch 6 was selected from validation at EER 0.0527. The closed final evaluation reported EER 0.0693; immutable identities and results are in `reports/adaptive_augmented_3s_v1_final_results.json`. Final-test thresholds are descriptive only and must not be used as deployment thresholds.
