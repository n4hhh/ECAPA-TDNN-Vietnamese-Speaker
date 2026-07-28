# Resumed Multi-Epoch ECAPA-AAM Fine-Tuning v1

Date: 2026-07-28

## Result

**Technical result: PASS. Model-quality result: IMPROVED.**

The approved epoch-0 checkpoint was migrated without retraining epoch 0.
Epochs 1 through 4 each completed exactly 1,000 optimizer updates and one
fixed validation. Training stopped at the approved maximum epoch. The best
fixed-validation checkpoint is epoch 3 with interpolated EER
`0.05735354362965998` (5.735354%), compared with epoch 0 at
`0.0646251536255633` (6.462515%) and the pretrained baseline at
`0.11665301106104056` (11.665301%).

No final-test artifact was accessed, no epoch beyond 4 started, and no commit
or push was performed.

## Start checkpoint

- Path: `outputs/ecapa_aam_one_epoch_pilot_v1/epoch_000.pt`
- SHA-256:
  `e82ba006aef4a505244769f137af897a94bbb601a26ffe2b68fbfec135201b67`
- Schema validated with the existing pilot validator: yes
- Epoch / global optimizer step: `0 / 1000`
- Next cursor: epoch `1`, logical batch `0`
- AdamW parameter steps: all exactly `1000`
- Optimizer LRs: ECAPA `1e-5`, AAM `1e-3`
- Scheduler state present: no, as expected for the sole allowed migration
- Epoch-0 validation EER / empirical threshold:
  `0.0646251536255633 / 0.1744520664215088`
- Epoch 0 retrained: no
- Start-checkpoint hash after the run: unchanged

## Epoch metrics

`Best` means the row passed the strict `new_eer < best_eer - 0.0001` rule and
became the best checkpoint at that validation. The pretrained baseline is a
historical comparison, not a managed training checkpoint.

| Model | Train loss | Validation EER | EER % | Empirical threshold | FAR | FRR | End ECAPA LR | End AAM LR | Best | Patience |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| Pretrained baseline | n/a | 0.116653011061041 | 11.665301 | 0.311659902334213 | 0.116653011061041 | 0.116653011061041 | n/a | n/a | no | n/a |
| Epoch 0 | 5.1006749759 | 0.064625153625563 | 6.462515 | 0.174452066421509 | 0.064625153625563 | 0.064625153625563 | 1e-5 | 1e-3 | initial | 0 |
| Epoch 1 | 1.0174463168 | 0.060221220811143 | 6.022122 | 0.171141743659973 | 0.060221220811143 | 0.060221220811143 | 8.681980515e-6 | 8.681980515e-4 | yes | 0 |
| Epoch 2 | 0.6285350316 | 0.057968045882835 | 5.796805 | 0.169464990496635 | 0.057968045882835 | 0.057968045882835 | 5.5e-6 | 5.5e-4 | yes | 0 |
| Epoch 3 | 0.4801272847 | 0.057353543629660 | 5.735354 | 0.170602425932884 | 0.057353543629660 | 0.057353543629660 | 2.318019485e-6 | 2.318019485e-4 | yes | 0 |
| Epoch 4 | 0.4216509168 | 0.057455960671856 | 5.745596 | 0.170072004199028 | 0.057455960671856 | 0.057455960671856 | 1e-6 | 1e-4 | no | 1 |

Every resumed epoch used Hybrid P16K2 W8, seed `20260727`,
`sampler.set_epoch(current_epoch)`, 1,000 logical batches of 32, physical
microbatch 4, and accumulation 8. All 4,000 resumed losses and both gradient
groups were finite and nonzero. Invalid or skipped optimizer updates: zero.
CUDA OOMs: zero.

## Score distributions

| Epoch | Positive mean / std / median | Negative mean / std / median | Score range |
| ---: | --- | --- | --- |
| 0 | 0.4248103264 / 0.1559999393 / 0.4358007610 | 0.0080272854 / 0.1035615685 / 0.0005986420 | [-0.3212318420, 0.9287939072] |
| 1 | 0.4288242109 / 0.1555503413 / 0.4395031333 | 0.0097320016 / 0.0992191445 / 0.0046791709 | [-0.3404123485, 0.9353240132] |
| 2 | 0.4304648676 / 0.1550564167 / 0.4411588460 | 0.0090676599 / 0.0979141775 / 0.0048456579 | [-0.3542872667, 0.9337859154] |
| 3 | 0.4322530661 / 0.1544587435 / 0.4426439404 | 0.0098346642 / 0.0975810228 / 0.0056887395 | [-0.3530446589, 0.9325098991] |
| 4 | 0.4326574131 / 0.1543406511 / 0.4428882152 | 0.0095855423 / 0.0975127181 / 0.0056061540 | [-0.3550921381, 0.9316360354] |

## Scheduler

The scheduler used the required per-successful-update factor:

```text
0.1 + 0.9 * 0.5 * (1 + cos(pi * completed_resumed_steps / 4000))
```

Progress was clamped to `[0,1]`, stepped only after successful AdamW updates,
and applied identically to both optimizer groups.

- Resumed step 0 factor / LRs: `1.0 / (1e-5, 1e-3)`
- Resumed step 1,000 factor: `0.8681980515339464`
- Resumed step 2,000 factor: `0.55`
- Resumed step 3,000 factor: `0.23180194846605365`
- Resumed step 4,000 factor / LRs: `0.1 / (1e-6, 1e-4)`
- Saved scheduler count in `last.pt`: `4000`
- Epoch-0 migration explicitly recorded in every new checkpoint: yes
- Scheduler checkpoint restore test: passed

## Early stopping

- Initial best: epoch `0`, EER `0.0646251536255633`
- Patience / min delta: `2 / 0.0001`
- Epochs 1, 2, and 3 each improved by more than the strict min delta.
- Epoch 4 did not improve epoch 3 by the required delta; patience became 1.
- Early stopping did not trigger before the maximum epoch.
- Stop reason: `max_epoch`
- No epoch after a stop condition started.

## Best checkpoint

- Selected epoch: `3`
- Selected fixed-validation EER: `0.05735354362965998`
- Path: `outputs/ecapa_aam_multiepoch_v1/best.pt`
- SHA-256:
  `7c63f4e2a100ce4eede4bb2429064387f80302c36b534baf46a4ae5b0e9cdb4f`
- `best.pt` state exactly matches `epoch_003.pt`: yes
- Best improvement from epoch 0: `0.007271609995903323` absolute EER,
  11.251981% relative
- Best improvement from pretrained: `0.059299467431380586` absolute EER,
  50.834065% relative

## Checkpoints and resume

All saves used a temporary file followed by `os.replace`, reloaded successfully,
and passed the multi-epoch schema/counter validator.

| Checkpoint | SHA-256 | Epoch | Global / resumed step | Next cursor | Scheduler step | Best epoch | Patience | Stop |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `epoch_001.pt` | `3ac4388ab34638064650fd2703f7ced19625ec4a48a9c7373048042e329440ea` | 1 | 2000 / 1000 | 2 / 0 | 1000 | 1 | 0 | none |
| `epoch_002.pt` | `f153c557e0bff0b997c6758255f39a642390d75dfd53c492f50227d968302e28` | 2 | 3000 / 2000 | 3 / 0 | 2000 | 2 | 0 | none |
| `epoch_003.pt` | `cc56b02ba2750a148098e40701df83ec65e17c51f5ec7cb19cb84c4d9c0efe30` | 3 | 4000 / 3000 | 4 / 0 | 3000 | 3 | 0 | none |
| `epoch_004.pt` | `5e58713b81ba74dd42b1296c85018e37cc98ff03a53ec5d25a1646553261541e` | 4 | 5000 / 4000 | 5 / 0 | 4000 | 3 | 1 | max_epoch |
| `last.pt` | `1bc676ea8d10717c61ca0c122d40005a9224981c3b113b4d04ed329644491471` | 4 | 5000 / 4000 | 5 / 0 | 4000 | 3 | 1 | max_epoch |
| `best.pt` | `7c63f4e2a100ce4eede4bb2429064387f80302c36b534baf46a4ae5b0e9cdb4f` | 3 | 4000 / 3000 | 4 / 0 | 3000 | 3 | 0 | none |

`last.pt` was atomically rolled every 250 resumed optimizer steps and at each
validated epoch boundary. It exactly matches the final epoch-4 state.

Two fail-closed reporting-path issues occurred and are preserved in ignored
diagnostic JSON files:

1. The first launch stopped before model construction because a reused pilot
   path-hash helper rejected the approved pilot checkpoint path.
2. The corrected full CUDA process completed all training, validation,
   checkpoint, protected-hash, and stop-line work, then the same old allowlist
   surfaced in an aggregate digest helper during runtime JSON serialization.

Both old helpers were replaced with multi-epoch-specific containment,
quarantine, and hash functions. A final resume from completed `last.pt`
performed zero optimizer steps, printed the same max-epoch stop decision, and
successfully wrote `runtime.json`; its stderr was empty. No training or
validation was repeated.

## Duration and CUDA memory

- Primary CUDA process wall span from console timestamps:
  `2741.07` seconds
- Postflight-only successful resume process span: `19.05` seconds
- Combined process spans: `2760.12` seconds (46 minutes, 0.12 seconds)
- Sum of measured epoch training durations: `2435.5558691` seconds
- Training peak allocated / reserved:
  `774599680 / 893386752` bytes
  (`738.716 / 852.000` MiB)
- Overall peak allocated / reserved during validation:
  `1595687424 / 1845493760` bytes
  (`1521.766 / 1760.000` MiB)
- OOM: none

## BatchNorm and model policy

The run retained cached float32 Fbank `[301,80] -> mean_var_norm ->
embedding_model -> [B,1,192] -> [B,192]`. ECAPA used float16 autocast; AAM and
loss math were float32; GradScaler remained enabled. The full embedding model
and BatchNorm affine parameters were trainable, while all 31 BatchNorm modules
were placed in eval mode during training. All 93 running-statistic buffers
remained bit-exact through all updates, checkpoints, and validations.
`mean_var_norm` was not optimized. The pretrained classifier,
`compute_features`, WAV access, augmentation, gradient clipping, and AAM during
validation were not used.

## Fixed validation and protected inputs

Each completed epoch embedded all 8,504 validation utterances and scored
exactly 19,528 fixed trials (9,764 positive and 9,764 negative). Path ownership
validation passed every time. The O(N log N) grouped/sorted EER implementation
was used.

- Trial CSV SHA-256:
  `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`
- Protected files compared: `167`
- Before/after set SHA-256:
  `9b21810b2c1c1abfb699d8c13835c1cac393e385f95660bfb3e9d5c16e062d04`
- Mismatches: zero
- Final-test manifest/cache/path/metadata/score/report access: none
- Final-test path rejection tests: passed

## Logging and tests

- Progress lines: exactly `80` (20 per resumed epoch at 50-step cadence)
- Epoch result lines: exactly `4`
- Primary stop lines: exactly `1`
- Final postflight resume stop lines: exactly `1`
- `py_compile`: passed
- Focused multi-epoch tests: `12 / 12` passed
- Complete unittest discovery: `112 / 112` passed
- `git diff --check`: passed
- Checkpoint schema/hash/state audit: passed
- Commit/push: none

## Files created

- `src/ecapa_multiepoch.py`
- `scripts/run_ecapa_aam_multiepoch.py`
- `tests/test_ecapa_multiepoch.py`
- `reports/ecapa_aam_multiepoch_v1.md`
- `reports/ecapa_aam_multiepoch_v1.json`
- Ignored `outputs/ecapa_aam_multiepoch_v1/` runtime artifacts and checkpoints

## Deferred

Final-test access/evaluation, any epoch after 4, further training,
hyperparameter changes, augmentation, cache/manifest/split/trial regeneration,
commit, and push.
