# VieSpeaker2.0 training sampler readiness

Result: **PASS**

The approved sampler is `HybridShardAwareSpeakerBatchSampler` with P=16, K=2, batch size 32, active shard window 8, seed 20260729, and 2969 batches per epoch.

Metadata-only plans for epochs 0 and 1 passed exact P x K, train-only selection, speaker coverage, duplicate-group separation, LRU simulation, same-epoch reproduction, and different-epoch change checks.

- Sampler plan identity: `b11a97fd45f11b8f71fa980ebd26cb335bac0f3bcdfb781e8f87ced535606e4f`
- Epoch 0 plan: `010e80f042ae91f1c890045d12833005359833466ba6d5a4b5fc70e5c8f04a68`
- Epoch 1 plan: `c5ff9c7fca4c88541c47738706b1852293e6727da9bb420350e75379179bba0f`
- Epoch 0 LRU(8) hit rate: 0.815100
- Epoch 1 LRU(8) hit rate: 0.813668
- Real-cache smoke: 16 batches / 512 samples in 0.669 seconds

No comparative sampler benchmark was run. This task approves only the fixed requested configuration; it does not start training.
