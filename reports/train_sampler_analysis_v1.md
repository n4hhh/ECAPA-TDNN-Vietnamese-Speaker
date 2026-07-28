# Training Batch Sampler Prototypes and Benchmark v1

## 1. Executive result

**PASS.** All sampler invariants, exact P × K structures, determinism checks, metadata analysis, and real-cache benchmark configurations passed.

## 2. Environment

- OS: Windows-10-10.0.26200-SP0
- Python: 3.10.11
- torch: 2.2.0+cu121
- Environment: `.venv-cuda`
- Seed: 20260727

## 3. Inputs and invariants

- Train rows: 31998
- Speakers: 488
- Labels: `0..487`
- Referenced shards: 125
- Every row is train-only, non-negative-labelled, assigned once, and references an existing shard.

## 4. Speaker-label bijection validation

The real train index passed both directions of the speaker ID/label bijection. No mapping was regenerated.

## 5. Candidate sampler designs

- Sample-random: deterministic full single-pass shuffle.
- Global P=16, K=2: balanced shuffled speaker queue plus per-speaker utterance queues.
- Hybrid P=16, K=2: exposure-aware speaker choice within deterministic active shard windows 8, 16, and 32.

## 6. Epoch definition

P × K candidates use 1,000 full batches: `ceil(31,998 / 32)`. This balances speaker slots, so low-resource speakers repeat and high-resource speakers need not expose every utterance. Sample-random remains a 31,998-sample single pass with one partial batch.

## 7. Batch-structure metrics

| Candidate | Batches | Unique speakers median | Shards/batch median | P×K violations |
|---|---:|---:|---:|---:|
| sample_random | 1000 | 28.0 | 28.0 | 0 |
| global_p16_k2 | 1000 | 16.0 | 16.0 | 0 |
| hybrid_w8 | 1000 | 16.0 | 6.0 | 0 |
| hybrid_w16 | 1000 | 16.0 | 9.0 | 0 |
| hybrid_w32 | 1000 | 16.0 | 12.0 | 0 |

## 8. Speaker-exposure metrics

- sample_random: selected all=True; samples/speaker min/median/max 5/29/1595; CV=2.0216.
- global_p16_k2: selected all=True; samples/speaker min/median/max 64/66/66; CV=0.0125.
- hybrid_w8: selected all=True; samples/speaker min/median/max 56/68/76; CV=0.0924.
- hybrid_w16: selected all=True; samples/speaker min/median/max 62/66/68; CV=0.0282.
- hybrid_w32: selected all=True; samples/speaker min/median/max 64/66/68; CV=0.0137.

## 9. Utterance coverage and repetition

- sample_random: coverage=100.00%; omitted=0; repeated selections=0 (0.00%); queue cycles=0.
- global_p16_k2: coverage=51.74%; omitted=15443; repeated selections=15445 (48.27%); queue cycles=1740.
- hybrid_w8: coverage=52.83%; omitted=15095; repeated selections=15097 (47.18%); queue cycles=1661.
- hybrid_w16: coverage=52.10%; omitted=15327; repeated selections=15329 (47.90%); queue cycles=1709.
- hybrid_w32: coverage=51.85%; omitted=15407; repeated selections=15409 (48.15%); queue cycles=1728.

## 10. Shard-locality metrics

Exact min/mean/quartile/median/max distributions and transitions are in the JSON. The batch CSV contains every diagnostic batch.

## 11. Metadata-derived LRU simulations

- sample_random: LRU 2: 0.9835 loads/sample, LRU 8: 0.9354 loads/sample, LRU 16: 0.8697 loads/sample.
- global_p16_k2: LRU 2: 0.9174 loads/sample, LRU 8: 0.6850 loads/sample, LRU 16: 0.4439 loads/sample.
- hybrid_w8: LRU 2: 0.6005 loads/sample, LRU 8: 0.2007 loads/sample, LRU 16: 0.1903 loads/sample.
- hybrid_w16: LRU 2: 0.7520 loads/sample, LRU 8: 0.3315 loads/sample, LRU 16: 0.2761 loads/sample.
- hybrid_w32: LRU 2: 0.8448 loads/sample, LRU 8: 0.4931 loads/sample, LRU 16: 0.3577 loads/sample.

## 12. Real-cache benchmark methodology

Real immutable train cache, LRU 8, finite re-scan disabled, workers 0 and 2, 3 repeats, 32 measured batches after warm-up. Candidate order was deterministically rotated. End-to-end includes iterator creation and worker startup. OS file caching affects results; this is not a cold-disk measurement.

## 13. Benchmark results

| Candidate | Workers | First batch s median [min,max] | End-to-end s median [min,max] | Steady samples/s median [min,max] |
|---|---:|---:|---:|---:|
| sample_random | 0 | 0.2592 [0.2085,0.2724] | 7.5022 [6.2593,7.6113] | 150.46 [148.19,180.16] |
| global_p16_k2 | 0 | 0.3283 [0.2674,0.3404] | 5.2144 [4.7567,5.6629] | 220.45 [204.44,244.06] |
| hybrid_w8 | 0 | 0.8191 [0.6571,0.8450] | 2.3860 [1.8957,2.5576] | 708.37 [634.16,885.04] |
| hybrid_w16 | 0 | 1.3319 [1.2582,1.3617] | 3.6754 [3.5829,4.1245] | 476.67 [390.55,487.48] |
| hybrid_w32 | 0 | 2.3738 [2.1893,2.8473] | 6.0035 [5.4744,8.3948] | 306.65 [216.99,335.06] |
| sample_random | 2 | 4.8535 [4.8204,5.2086] | 13.5642 [12.9605,14.3623] | 117.15 [111.90,126.35] |
| global_p16_k2 | 2 | 5.5486 [5.1322,5.5938] | 10.4263 [9.7384,10.5300] | 214.48 [207.22,224.17] |
| hybrid_w8 | 2 | 4.7785 [4.7581,5.8598] | 5.9851 [5.7848,6.8700] | 1018.40 [836.64,1063.43] |
| hybrid_w16 | 2 | 6.6110 [6.3827,6.7443] | 8.7944 [8.3453,8.8069] | 497.43 [473.23,524.17] |
| hybrid_w32 | 2 | 7.4473 [7.0753,7.9690] | 10.4711 [9.7812,10.8023] | 382.19 [308.26,414.88] |

Medians are primary; JSON also records min/max and every raw repeat.

## 14. num_workers=0 versus num_workers=2

Worker-zero runs include actual Dataset shard loads. Worker-two load totals are marked unavailable because main-process counters do not observe worker-local caches.

## 15. First-batch versus end-to-end versus steady-state timing

These are reported separately. Worker prefetch can benefit steady-state throughput, so steady-state alone is not used to recommend a sampler.

## 16. Trade-offs

Global balance equalizes speaker exposure but sacrifices locality. Hybrid candidates retain exact P × K batches while trading window breadth against shard reuse. Metadata quality and I/O throughput do not establish model quality.

## 17. Evidence-based recommendation

**Suggestion only:** use hybrid P=16, K=2, window=8 as the leading training-sampler candidate. It retained exact P × K batches and all 488 speakers while producing the best locality and real-access throughput among the balanced candidates. Its speaker exposure CV is higher than global balance, so keep monitoring exposure. This does not claim improved model or verification quality.

## 18. Known limitations

- OS cache state and worker prefetch affect timings.
- No cold-cache clearing or worker shard-load instrumentation was attempted.
- No model, SpeechBrain inference, waveform access, or training was performed.

## 19. Files created and modified

Created sampler module, benchmark script, sampler tests, JSON/CSV/Markdown artifacts. Modified the cached Dataset module only to add the batch-sampler DataLoader helper; the worklog was appended.

## 20. Tests and commands

Commands and exact test counts are recorded in the appended worklog.

## 21. Explicitly deferred work

AAM-Softmax, ECAPA fine-tuning/forward passes, optimization, verification, thresholds, EER, test evaluation, augmentation, and all cache/manifest/split changes.

## 22. Overall result

**PASS**
