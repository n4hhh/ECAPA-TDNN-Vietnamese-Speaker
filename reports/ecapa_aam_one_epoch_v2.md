# VieSpeaker2.0 ECAPA-AAM epoch 0

Result: **PASS**. Exactly 2,969 optimizer updates completed; epoch 1 did not start.

## Training

- Cached Fbank `[B,301,80] -> mean_var_norm -> embedding_model -> [B,1,192] -> [B,192]`
- P=16, K=2, logical batch 32; physical microbatch 4; accumulation 8
- AdamW ECAPA/AAM learning rates `1e-5 / 1e-3`; no scheduler or gradient clipping
- Training duration: 2103.319412 seconds
- Loss first/final/mean: 15.970597863197327 / 0.5739374789409339 / 3.0659153226259708

## Validation

- Embeddings: `[15355,192]` float32 CPU; trials: 10,000 positive / 10,000 negative
- Interpolated EER: 0.064 (6.400000%)
- Empirical threshold: 0.16190975904464722
- Empirical FAR / FRR: 0.064 / 0.064
- TP / TN / FP / FN: 9360 / 9360 / 640 / 640

## Baseline comparison

- Pretrained EER: 0.1257
- Signed difference: -0.061700000000000005
- Relative change: -0.4908512330946699
- Classification: **IMPROVED**

## Checkpoint and stop

- Approved future-resume checkpoint: `outputs/ecapa_aam_one_epoch_v2/best.pt`
- Best checkpoint SHA-256: `19cfd3482172ce482d21b65915fd347aeeac4ee1e51bc61e59484f46a31a38db`
- Final cursor: epoch 1, batch position 0, global step 2,969
- Epoch 1 started: false
- Final test remained quarantined
