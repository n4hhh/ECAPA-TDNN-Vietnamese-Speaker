# VieSpeaker2.0 One-Time Authoritative Final Evaluation

Result: **PASS**

## PRIMARY — LOCKED VALIDATION THRESHOLD

- Threshold: 0.16545939445495605
- Source: selected epoch-3 validation threshold
- Decision: cosine score >= threshold accepts same speaker
- TP / TN / FP / FN: 9598 / 9426 / 574 / 402
- FAR / FRR: 0.0574 / 0.0402
- FAR/FRR gap: 0.0172
- Average error: 0.048799999999999996
- Accuracy: 0.9512
- Precision: 0.9435705859221392
- Recall / TPR: 0.9598
- Specificity / TNR: 0.9426
- F1: 0.9516161015268689

Accuracy, precision, and F1 describe this balanced 50/50 trial protocol and do not represent real-world same/different-speaker prevalence.

## DESCRIPTIVE TEST DIAGNOSTIC — NON-OPERATIONAL

- Interpolated EER: 0.0453
- Interpolated non-empirical threshold: 0.17728488147258759
- Executable empirical threshold: 0.17728488147258759
- Interpolated FAR / FRR: 0.0453 / 0.0453
- Empirical FAR / FRR: 0.0453 / 0.0453
- Empirical FAR/FRR gap: 0.0
- Empirical average error: 0.0453

The test-derived thresholds are descriptive only and never replace the locked
validation threshold.

## Runtime

- Fbank extraction seconds: 31.948287599996547
- Embedding extraction seconds: 74.70890539999527
- Scoring seconds: 0.045247500005643815
- Locked metrics seconds: 0.0076336000056471676
- Descriptive EER seconds: 0.05011839998769574
- Total authoritative duration seconds: 150.58457500001532
- CUDA peak allocated bytes: 2561522176
- CUDA peak reserved bytes: 3166699520
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- OOM count: 0
- Model inference processes: 1

The finalized normal evaluator refuses another inference or scoring run.
Only metrics-only recomputation from the immutable saved score artifact is permitted.
