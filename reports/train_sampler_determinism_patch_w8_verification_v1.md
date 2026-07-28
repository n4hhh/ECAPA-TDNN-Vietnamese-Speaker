# Training Batch Sampler Prototypes and Benchmark v1

## 1. Executive result

**PASS.** All sampler invariants, exact P × K structures, determinism checks, metadata analysis, and real-cache benchmark configurations passed.

## 2. Environment

- OS: Windows-10-10.0.26200-SP0
- Python: 3.10.11
- torch: 2.2.0+cu121
- Environment: `.venv-cuda`
- Seed: 20260727
- Run classification: post-fix hybrid W8 short verification

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
| hybrid_w8 | 1000 | 16.0 | 6.0 | 0 |

## 8. Speaker-exposure metrics

- hybrid_w8: selected all=True; samples/speaker min/median/max 56/68/76; CV=0.0927.

## 9. Utterance coverage and repetition

- hybrid_w8: coverage=52.81%; omitted=15101; repeated selections=15103 (47.20%); queue cycles=1666.

## 10. Shard-locality metrics

Exact min/mean/quartile/median/max distributions and transitions are in the JSON. The batch CSV contains every diagnostic batch.

## 11. Metadata-derived LRU simulations

- hybrid_w8: LRU 2: 0.5942 loads/sample, LRU 8: 0.1996 loads/sample, LRU 16: 0.1891 loads/sample.

## 12. Real-cache benchmark methodology

Real immutable train cache, LRU 8, finite re-scan disabled, workers 0 and 2, 1 repeats, 4 measured batches after warm-up. Candidate order was deterministically rotated. End-to-end includes iterator creation and worker startup. OS file caching affects results; this is not a cold-disk measurement.

## 13. Benchmark results

| Candidate | Workers | First batch s median [min,max] | End-to-end s median [min,max] | Steady samples/s median [min,max] |
|---|---:|---:|---:|---:|
| hybrid_w8 | 0 | 0.8613 [0.8613,0.8613] | 1.2059 [1.2059,1.2059] | 443.27 [443.27,443.27] |

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
