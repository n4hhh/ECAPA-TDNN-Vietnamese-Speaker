# VieSpeaker2.0 AMP overflow recovery

Result: **PASS**.

The exact atomic checkpoint `outputs/ecapa_aam_multiepoch_v2/recovery_start_last.pt` with
SHA-256 `3b8d613012ee8340a30ad264729c83ea97501b793e6c97e1d0ed78555e5c96a0` was preserved and resumed
from epoch 2, batch position 2,031.

## Historical replay

The historical failed batch result was `not_reproduced`. Its replay
batch identity was `3a7e984fc826e9e5e0a0ad4b07af744bd8ac370e7e325f2b3984b02cee189bfd`.

## Overflow events

- Epoch 2, batch 2585, global step 8523: scale 2048.0 -> 1024.0; affected `blocks.0.norm.norm.weight`; result `RECOVERED`.
- Epoch 3, batch 1095, global step 10002: scale 2048.0 -> 1024.0; affected `blocks.0.norm.norm.weight`; result `RECOVERED`.
- Epoch 4, batch 594, global step 12470: scale 2048.0 -> 1024.0; affected `blocks.0.norm.norm.weight`; result `RECOVERED`.

Attempts: 3; recovered:
3; unrecovered:
0. Failed attempts advanced no training
counters and no logical batch was skipped.

The historical failed-run reports, runtime, failure record, and epoch-one
checkpoint remained byte-identical. Final-test content remained quarantined.
