# SpeechBrain Fbank cache v1 summary

## Environment

- Python: 3.10.11
- torch: 2.2.0+cu121
- torchaudio: 2.2.0+cu121
- SpeechBrain: 1.0.3
- CUDA available: True
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU

## Input

- Train: 31998
- Validation: 8504
- Test: 9198
- Total: 49700
- Dataset root used: `E:\VieSpeaker`

## Preflight

- Waveform shape: `[1, 48000]`
- Fbank batch shape: `[1, 301, 80]`
- Saved feature shape: `[301, 80]`
- Save/load maximum difference: 0.0
- Embedding maximum difference: 0.0
- Embedding allclose (atol=1e-5, rtol=1e-4): True

## Cache

- Shards: train=125, validation=34, test=36
- Cached rows: train=31998, validation=8504, test=9198
- Shard size: 256
- Total cache size: 4,797,464,952 bytes
- Completed resume build/validation duration: 117.278 seconds
- Initial interrupted run wall time: 10.423 seconds
- Observed end-to-end wall time: 127.701 seconds
- End-to-end throughput: 389.190 utterances/second
- Peak allocated VRAM: 157784576 bytes
- Peak reserved VRAM: 381681664 bytes

## Validation

- Missing rows: 0
- Duplicate rows: 0
- Invalid shard indexes: 0
- Incorrect shapes: 0
- NaN or Inf tensors: 0
- Label ranges: PASS
- Sampled embedding equivalence: PASS
- Actual-cache embedding samples: 4
- Actual-cache maximum difference: 0.0
- Actual-cache allclose: True
- Tests: PASS (11 tests)

## Overall result

PASS
