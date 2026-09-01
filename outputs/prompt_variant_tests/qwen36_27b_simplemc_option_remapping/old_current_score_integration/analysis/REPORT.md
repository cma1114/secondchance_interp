# Existing-data old-score/current-score integration analysis

## Bottom line

Fresh second-presentation evidence is the strongest single predictor of the final candidate ranking, but first-presentation evidence adds held-out predictive information in both Game and Neutral. The increment is larger in Neutral. A simple old-by-current interaction has a small effect in the linear specification but does not robustly improve the flexible model. A W1 term remains most clearly in the Game-minus-Neutral contrast.

The causal matching-edge endpoint is more diagnostic. After controlling current score and both displayed positions, stronger old evidence predicts a more suppressive matching effect in Game relative to Neutral on both frozen splits. Current second-presentation evidence independently predicts Game's matching-edge effect. The separate Neutral old-score coefficient points toward support but is uncertain. Thus the semantic match is not merely carrying a condition-independent old score: its causal use depends on both historical and current candidate evidence.

This is a predictive decomposition of saved causal and natural outputs. It does not identify which vectors encode either score and does not replace a same-semantic/different-score transplant.

## Held-out predictive increments

All values are increases in confirmation-set R² from adding the named feature family to the nested discovery-fitted model.

| Endpoint | Condition | Add old score (linear) | Add old score (flexible) | Add old×current (linear) | Add old×current (flexible) | Add W1 after flexible scores |
|---|---|---:|---:|---:|---:|---:|
| Final centered logits | Game | +0.020 [+0.006, +0.033] | +0.020 [-0.011, +0.050] | +0.006 [+0.001, +0.011] | +0.006 [-0.008, +0.021] | +0.000 [-0.001, +0.001] |
| Final centered logits | Neutral | +0.041 [+0.019, +0.064] | +0.061 [+0.027, +0.099] | +0.013 [-0.003, +0.029] | +0.017 [-0.001, +0.036] | +0.003 [+0.000, +0.007] |
| Final centered logits | Game − Neutral | +0.032 [+0.009, +0.057] | +0.090 [+0.056, +0.125] | -0.000 [-0.044, +0.033] | -0.002 [-0.028, +0.024] | +0.007 [+0.000, +0.012] |
| Matching-edge lesion | Game | +0.016 [-0.001, +0.034] | +0.009 [-0.018, +0.037] | -0.001 [-0.006, +0.004] | -0.001 [-0.023, +0.019] | -0.000 [-0.001, +0.001] |
| Matching-edge lesion | Neutral | +0.011 [+0.000, +0.020] | +0.014 [-0.001, +0.029] | +0.002 [+0.000, +0.004] | -0.023 [-0.061, +0.005] | +0.008 [+0.001, +0.014] |
| Matching-edge lesion | Game − Neutral | +0.056 [+0.029, +0.082] | +0.089 [+0.057, +0.121] | -0.002 [-0.004, +0.000] | +0.002 [-0.016, +0.018] | +0.008 [+0.003, +0.013] |

## Linear causal coefficients

Scores are standardized using discovery data. Positive lesion coefficients mean that the intact match becomes more opposing as the predictor increases.

| Split | Condition | Old score | Current score | Old×current | W1 |
|---|---|---:|---:|---:|---:|
| Discovery | Game | +0.122 [+0.025, +0.239] | +0.238 [+0.138, +0.336] | +0.014 [-0.032, +0.082] | +0.129 [-0.039, +0.281] |
| Discovery | Neutral | -0.053 [-0.176, +0.083] | -0.010 [-0.133, +0.106] | +0.008 [-0.022, +0.047] | -0.122 [-0.288, +0.040] |
| Discovery | Game − Neutral | +0.175 [+0.082, +0.275] | +0.248 [+0.151, +0.345] | +0.005 [-0.032, +0.058] | +0.251 [+0.100, +0.400] |
| Confirmation | Game | +0.148 [+0.038, +0.260] | +0.181 [+0.082, +0.299] | -0.001 [-0.035, +0.069] | +0.046 [-0.129, +0.213] |
| Confirmation | Neutral | -0.087 [-0.213, +0.035] | +0.023 [-0.072, +0.123] | +0.040 [+0.003, +0.097] | -0.424 [-0.599, -0.246] |
| Confirmation | Game − Neutral | +0.235 [+0.127, +0.346] | +0.158 [+0.069, +0.249] | -0.041 [-0.075, +0.016] | +0.470 [+0.305, +0.629] |

## Interpretation

The results support a two-evidence account as a serious leading hypothesis: a repeated candidate is evaluated using both its first-presentation evidence and its fresh second-presentation evidence. They do not establish that the matching source value itself contains the old score; the old-score dependence could enter through the receiver query or another state correlated with first-pass evidence.

The W1 increment also survives flexible score terms in the held-out condition difference, but the gain is small. It could reflect a categorical winner variable or remaining nonlinear score structure. A causal rank manipulation is required to distinguish them.

## Validation

- Questions: 500.
- Frozen discovery/confirmation split: 251/249.
- Natural logits and causal lesions are taken from the exact canonical all-candidate run; no new model inference was used.
- Every model includes question-centered controls for the candidate's displayed position in both presentations.
