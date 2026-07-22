# Vietnamese Speaker Verification - Dataset Inspection and Fbank Smoke Tests

This repository contains read-only dataset inspection and Stage B acoustic
feature smoke tests. Fbank features are computed lazily in memory and discarded;
the project does not preprocess or augment audio, define a model, assign final
splits, or train anything.

The inspector recursively inventories `E:\VieSpeaker`, checks that each regular
file can be opened, reads metadata from a bounded sample of audio headers, and
inspects Parquet metadata/rows in bounded batches. It never writes to or deletes
anything in the dataset directory, and it never prints binary audio payloads.
Links and Windows reparse points are not followed, so traversal stays inside the
requested root.

## Setup (Windows PowerShell)

From the repository root:

```powershell
Set-Location 'E:\SpeakerVerification'
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

`pyarrow` is used only when Parquet files are present. `soundfile` reads audio
container/header metadata without loading complete waveforms into memory.
PyTorch and torchaudio provide the in-memory Fbank implementation. Install
matching PyTorch and torchaudio builds for your CPU or CUDA environment.

## Run

```powershell
python .\src\inspect_dataset.py --root 'E:\VieSpeaker'
```

The dataset root defaults to `E:\VieSpeaker`, so this is equivalent:

```powershell
python .\src\inspect_dataset.py
```

By default, the report prints metadata for 10 representative normal audio files
and five rows total across readable Parquet files, if those file types exist.
The sample sizes can be changed without changing the dataset:

```powershell
python .\src\inspect_dataset.py --root 'E:\VieSpeaker' --audio-samples 10 --parquet-rows 5
```

Sample-rate, channel-count, duration, and format distributions are explicitly
reported as observations from the inspected audio sample, not assumptions about
every file. Speaker folders and split names inferred from paths are labeled as
heuristics. Shard-like labels embedded in filenames are reported separately so
names such as `train_small` are not silently merged into `train`; a Parquet
`speaker_id` or split-like column is streamed directly when available.

Without a manifest, an unlisted/absent file cannot be inferred. The missing-file
checks cover broken links, files that disappear during inspection, and missing
path references in sampled Parquet rows. Audio container validity is checked for
the reported sample; basic open/readability is checked for every regular file.

## Build the WAV metadata manifest and filename-group reports

This remains an inspection/data-understanding step. It does not assign final
train, validation, or test splits and does not preprocess or decode waveforms.

```powershell
python .\src\build_manifest.py --dataset-root 'E:\VieSpeaker'
```

Progress is printed during header scanning and candidate hashing. The builder
uses a temporary SQLite index to keep memory bounded. SHA-256 is computed only
for files in repeated-size groups; if most WAV files share one size, most files
will necessarily be hash candidates.

Generated files:

- `manifests\full_manifest.csv`
- `reports\dataset_split_report.txt`
- `reports\speaker_distribution.csv`
- `reports\speaker_group_overlap.csv`

`audio_path` is absolute. Invalid numeric parent folders produce an empty
`speaker_id`; filenames outside the strict recognized shard pattern receive the
group `unknown`. Outputs are written under the project root and never under the
dataset root.

## Analyze filename-group relationships (without assigning splits)

```powershell
python .\src\analyze_manifest.py
```

This streams the existing manifest and compares a deterministic bounded PCM
sample from speakers shared by `train` and `train_small`. It writes:

- `reports\low_utterance_speakers.csv`
- `reports\group_relationship_report.txt`
- `reports\recommended_split_report.txt`

The analysis is advisory. It does not change the manifest or WAV files, does not
declare `part` to be training data, and does not assign final splits.

## Stage B: acoustic feature smoke tests

The reusable `FbankExtractor` defaults to mono 16 kHz audio, 80 Mel bins, a
25 ms frame length, and a 10 ms frame shift. It returns contiguous float32
log-Mel tensors and does not save them. The smoke-test dataset reads only rows
whose `filename_group` is exactly `train_small`; `test` and `part` remain
quarantined.

Run the synthetic, real-WAV, ten-file, and four-item DataLoader tests:

```powershell
python .\src\test_fbank.py
```

Run the bounded end-to-end benchmark over at most 1,000 `train_small` files:

```powershell
python .\src\benchmark_fbank.py --max-samples 1000
```

The benchmark times lazy WAV decoding, Fbank extraction, and DataLoader
collation, then linearly estimates the full-manifest runtime. It never processes
more than 1,000 files and never writes features or changes dataset files.

## Frozen pretrained SpeechBrain ECAPA integration smoke test

The production-pretraining path loads
`speechbrain/spkrec-ecapa-voxceleb` with the checkpoint's own
`compute_features`, sentence `mean_var_norm`, and pretrained
`embedding_model`. SpeechBrain features remain in their native `[B, T, 80]`
layout. The independently verified custom `FbankExtractor` remains available
for comparison and debugging; it is not substituted into the pretrained path.

This environment uses SpeechBrain 1.0.3 with Hugging Face Hub 0.36.0 alongside
the existing PyTorch and torchaudio 2.2.0 builds. These versions are pinned
because SpeechBrain 1.1.0's forward path requires a newer `torch.amp` API than
PyTorch 2.2.0 supplies. SpeechBrain 1.0.3 passes the legacy `use_auth_token`
download argument, which Hugging Face Hub 0.36.0 still supports and Hub 1.x
removed.

Run the one-file and batch-of-four test on CPU:

```powershell
python .\src\test_pretrained_ecapa.py --device cpu
```

If CUDA is available in the installed PyTorch build, select it explicitly:

```powershell
python .\src\test_pretrained_ecapa.py --device cuda:0
```

The first run caches checkpoint files under
`pretrained_models\spkrec-ecapa-voxceleb`. The test selects exactly four
`train_small` files, keeps all pretrained parameters frozen, and performs no
classification, optimizer creation, backpropagation, weight update, split
assignment, augmentation, or feature persistence.
