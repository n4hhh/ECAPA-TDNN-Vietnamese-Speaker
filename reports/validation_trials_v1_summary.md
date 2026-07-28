# Validation verification trials v1

## Result

**PASS.** The approved validation manifest and cache index agree exactly: 8,504 unique paths, 100 speakers, labels all `-1`, split all `validation`, and 34 existing referenced shards. Speaker IDs and filename groups agree; within-shard positions are valid and unique.

## Protocol

- Seed: `20260727`
- Schema: `trial_id,target,left_relative_audio_path,right_relative_audio_path,left_speaker_id,right_speaker_id`
- Positive policy: enumerate each speaker's unique unordered utterance pairs, seeded shuffle, sample without replacement, cap at 100, and use all pairs when fewer exist.
- Negative policy: repeatedly select two distinct minimum-participation speakers using seeded stable ties, select one utterance from each, canonicalize the unordered path pair, and reject duplicates until counts balance.
- Trial generation is metadata-only; no Fbank tensor or WAV is loaded.

## Results

- Positive / negative trials: 9,764 / 9,764
- Positive trials/speaker min/mean/Q1/median/Q3/max: 45 / 97.64 / 100 / 100 / 100 / 100
- Speakers below the cap: 10
- Negative participation/speaker min/mean/Q1/median/Q3/max: 195 / 195.28 / 195 / 195 / 196 / 196
- Negative-participation coefficient of variation: 0.002299256895
- Unique negative speaker pairs: 4,271
- Trials per represented speaker pair: 1 to 8
- All 100 speakers occur in positive and negative trials.
- Self-pairs: 0; duplicate unordered path pairs: 0; target contradictions: 0
- Trial SHA-256: `3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f`

## Input hashes

- Validation manifest: `9f553fa55b50ee071120bcbb2ce3c2caf9de4799cfb615eb732bbefd0d73d38b`
- Validation cache index: `7346073ed9b47354c5f85889222ad9a538d2c4ee14e0b7d85001e3c0d61608bc`
- Cache config: `c829b31d0795ff8460d1bcc6a8aac9a57e6c419d788d9888db0e34289bb029e9`

## Files and tests

Created `manifests/verification/validation_trials_v1.csv` and `validation_trials_config_v1.json`. Trial/metric/baseline focused tests: 13 passed; complete suite: 54 passed. Same seed/input reproduced identical CSV bytes and hash; a different seed produced different valid trials.

## Overall result

**PASS**
