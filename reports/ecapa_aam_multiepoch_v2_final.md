# VieSpeaker2.0 resumed multi-epoch ECAPA-AAM

Result: **PASS**. Epoch zero was not retrained. Training stopped after epoch
4 with reason `max_epoch`.

## Epoch history

| epoch | updates | global step | validation EER | empirical threshold | selected best | bad epochs |
|---:|---:|---:|---:|---:|:---:|---:|
| 0 | 2969 | 2969 | 0.0640000000 | 0.161909759045 | yes | 0 |
| 1 | 2969 | 5938 | 0.0602000000 | 0.162857368588 | yes | 0 |
| 2 | 2969 | 8907 | 0.0587000000 | 0.164791092277 | yes | 0 |
| 3 | 2969 | 11876 | 0.0584000000 | 0.165459394455 | yes | 0 |
| 4 | 2969 | 14845 | 0.0584000000 | 0.166259393096 | no | 1 |

## Selection

- Best epoch: 3
- Best checkpoint: `outputs/ecapa_aam_multiepoch_v2/best.pt`
- Best checkpoint SHA-256: `ba9d989c1b6a922f3f392cd297fb771bba05319d3fad99df740ac889d2836d6f`
- Validation EER: 0.0584
- Locked validation empirical threshold: 0.16545939445495605

## Scheduler

`0.1 + 0.9 * 0.5 * (1 + cos(pi * r / 11875))` was applied before each successful
resumed optimizer update, with no warmup. Stored completed updates:
11876.

## Preservation

All approved input identities remained unchanged. No source audio or cache
artifact was modified. The final-test manifest and all final-test/excluded
content remained quarantined and were not opened, statted, hashed, parsed, or
loaded. No commit or push occurred.
