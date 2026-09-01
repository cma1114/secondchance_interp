# Positive-only W2 semantic ablation at the final decision position

This intervention removes only positive projection onto the exact semantic direction for W2, the answer selected by a fresh Baseline solution of the remapped second presentation. The primary analysis is restricted to conflict trials where W1 differs from W2.

## Discovery

| Condition | Natural W2 | Ablated W2 | W2 change | Natural W1 | Ablated W1 | W1 change | W2–W1 margin change | Accuracy change | Entropy change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Game (n=137) | 46.0% | 38.7% | -7.30 [-13.14, -1.46] pp | 21.9% | 29.2% | +7.30 [+2.19, +12.41] pp | -0.22 [-0.30, -0.14] | +0.00 [-5.11, +5.11] pp | -0.00 [-0.04, +0.03] bits |
| Neutral (n=137) | 46.0% | 34.3% | -11.68 [-18.25, -5.84] pp | 36.5% | 43.8% | +7.30 [+2.19, +13.14] pp | -0.18 [-0.26, -0.10] | +2.92 [-2.92, +8.76] pp | +0.02 [-0.00, +0.05] bits |

Game-minus-Neutral intervention difference in W2 selection: **+4.38 [-3.65, +12.41] pp**. In W1 selection: **+0.00 [-7.30, +7.30] pp**.
Because a semantic switch is any second answer other than W1, the switch-rate changes were **-7.30 [-12.41, -2.19] pp** in Game and **-7.30 [-13.14, -2.19] pp** in Neutral.

## Confirmation

| Condition | Natural W2 | Ablated W2 | W2 change | Natural W1 | Ablated W1 | W1 change | W2–W1 margin change | Accuracy change | Entropy change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Game (n=136) | 43.4% | 34.6% | -8.82 [-13.97, -3.68] pp | 19.9% | 23.5% | +3.68 [-0.74, +8.82] pp | -0.32 [-0.42, -0.22] | -2.21 [-6.62, +2.21] pp | -0.01 [-0.04, +0.03] bits |
| Neutral (n=136) | 38.2% | 28.7% | -9.56 [-16.18, -3.68] pp | 40.4% | 44.9% | +4.41 [-0.74, +10.29] pp | -0.19 [-0.28, -0.09] | -0.74 [-5.88, +4.41] pp | +0.02 [-0.01, +0.05] bits |

Game-minus-Neutral intervention difference in W2 selection: **+0.74 [-5.88, +7.35] pp**. In W1 selection: **-0.74 [-6.62, +5.15] pp**.
Because a semantic switch is any second answer other than W1, the switch-rate changes were **-3.68 [-8.82, +0.74] pp** in Game and **-4.41 [-10.29, +0.74] pp** in Neutral.

## Direction overlap

On the 273 W1 != W2 conflict trials, the mean W1–W2 semantic-direction cosine at L64 is -0.282 (median -0.316). Layerwise values are in `summary.json`. This quantifies how much W2 removal may mechanically favor W1 because the two within-question contrast directions overlap negatively.

## Validation

A fresh natural companion measured W2 projection in the same forward regime as the intervention, and its A–D logits were required question-by-question to match the specified trusted 500-question reference exactly. The runner preserved each question's historical physical batch-of-four cohort, SDPA implementation, prompt serialization, and model revision. Exact W2 directions were reconstructed from the already-saved four mapping residual arrays, avoiding new baseline collection.
