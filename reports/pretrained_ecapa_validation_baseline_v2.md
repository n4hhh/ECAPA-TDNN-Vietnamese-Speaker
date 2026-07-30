# VieSpeaker2.0 pretrained ECAPA validation baseline

Result: **PASS**

Untouched pretrained interpolated validation EER: **12.570000%**.

## Fixed evaluation

- Validation embeddings: [15355, 192] float32 CPU
- Trials: 10000 positive / 10000 negative
- Score: cosine similarity
- EER: grouped O(N log N) tie-aware operating points
- Threshold semantics: accept same speaker when score >= threshold

## Model invariants

- Path: cached Fbank [B,T,80] -> mean_var_norm -> embedding_model -> [B,1,192] -> squeeze -> [B,192]
- No feature transpose, compute_features call, pretrained classifier call, gradient, optimizer, AAM-Softmax, or parameter update
- Parameters unchanged: True

## EER and executable threshold

- Interpolated EER: 0.1257 (12.570000%)
- Interpolated threshold: 0.28827327489852905
- Interpolated FAR / FRR: 0.1257 / 0.1257
- Empirical threshold: 0.28827327489852905
- Empirical FAR / FRR: 0.1257 / 0.1257
- TP / TN / FP / FN: 8743 / 8743 / 1257 / 1257
- Accuracy / precision / recall / F1: 0.8743 / 0.8743 / 0.8743 / 0.8743

## Score distributions

- Same-speaker: `{"count": 10000, "maximum": 0.9752877950668335, "mean": 0.47219380933633076, "median": 0.49064262211322784, "minimum": -0.2114696353673935, "q1": 0.3816196694970131, "q3": 0.5825836062431335, "standard_deviation": 0.15749554871675106}`
- Different-speaker: `{"count": 10000, "maximum": 0.602975070476532, "mean": 0.1534788316947641, "median": 0.14373768866062164, "minimum": -0.2100071758031845, "q1": 0.07482428289949894, "q3": 0.22558770701289177, "standard_deviation": 0.11294603164726892}`

## CUDA runtime

- Device: cuda:0 (NVIDIA GeForce RTX 3050 Laptop GPU)
- Embedding extraction: 72.231806s (212.579485 utterances/s)
- Trial scoring: 0.962083s
- Metrics: 0.070576s
- Peak allocated/reserved VRAM: 2567059456 / 3166699520 bytes

Tensor artifacts and runtime identity/configuration remain under the ignored `outputs/pretrained_ecapa_validation_baseline_v2/` directory.
