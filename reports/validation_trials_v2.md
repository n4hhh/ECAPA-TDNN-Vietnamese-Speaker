# VieSpeaker2.0 fixed validation trials

Result: **PASS**

The fixed validation protocol contains 10,000 positive and 10,000 negative trials over all 100 approved validation speakers.

- Positive balance: exactly 100 pairs per speaker
- Negative balance: exactly 200 round-robin participations per speaker
- Negative schedule: two complete 99-round cycles plus the first 2 rounds of cycle three
- Pair identity: canonical unordered relative-path pairs with no duplicates
- Positive duplicate protection: same nonempty duplicate group is rejected
- Utterance choice: least-used selection with stable SHA-256 tie breaks

- Trial CSV SHA-256: `11bec5ff0a0a4ca4930e2664bdc391388a9a677795afaefe5de3fee2d0d39e3d`
- Trial config SHA-256: `9e725ce006ae522f0f0274739b75e9aee7ebb302db331fead73c79cdc326822e`
- Speakers: 100
- Positive duplicate-group candidate rejections: 0

The artifacts contain no timestamps or absolute local paths.
