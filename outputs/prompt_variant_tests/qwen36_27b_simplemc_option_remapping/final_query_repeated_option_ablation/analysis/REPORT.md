# Final decision → repeated-option causal edge test

## Bottom line

The natural conflict-trial W1-avoidance difference is 19.8 percentage points (Neutral minus Game W1 choice).

The final query does directly use the repeated W1 option line, but the net read is **pro-W1, not suppressive**: blocking that line lowers W1 relative to the control option lines in both conditions. The dependence is much stronger in Neutral, so the lesion shrinks the Game--Neutral W1-avoidance gap by disrupting Neutral reinstatement more than Game.

The condition difference is consistent across splits in the continuous W1−W2 margin: discovery +0.17 [+0.14, +0.20] logits; held-out confirmation +0.19 [+0.15, +0.22] logits. On discrete W1 choice it is weak in discovery (+1.46 [-4.38, +7.30] points) but clear in confirmation (+8.33 [+3.19, +13.48] points).

At only the final decision query, the intervention blocks attention to one complete second-presentation option line across every ordinary-attention block 4--64. The W1 line is compared with the per-question mean of separately blocking each of the other three option lines.

## Held-out conflict trials

- Direct W1-line lesion effect in Game W1 choice: -0.74 [-2.21, +0.00] percentage points.
- Direct W1-line lesion effect in Neutral W1 choice: -8.82 [-13.97, -4.41] percentage points.
- Game W1-line effect minus mean other-line effect: -2.45 [-4.90, -0.49] percentage points.
- Neutral W1-line effect minus mean other-line effect: -10.78 [-15.44, -6.62] percentage points.
- Reduction in the Game-Neutral W1-avoidance gap: +8.33 [+3.19, +13.48] percentage points.
- Game W1−W2 margin contrast: -0.13 [-0.16, -0.11] logits.
- Neutral W1−W2 margin contrast: -0.32 [-0.36, -0.28] logits.
- Game-minus-Neutral margin contrast: +0.19 [+0.15, +0.22] logits.

## Held-out no-conflict trials

- Game W1-choice contrast: -8.26 [-13.86, -3.24] percentage points.
- Neutral W1-choice contrast: -15.34 [-21.83, -9.14] percentage points.

## Mechanistic interpretation

This rejects the simple last-hop story in which the final decision suppresses W1 by directly reading the repeated W1 line. The direct final read instead reinforces W1, especially in Neutral. Therefore the previously established Game-specific causal influence from the original W1 line into the repeated W1 line must affect Game through an earlier downstream relay or state update before the final query, not through a suppressive final-query attention edge.

## Validation

Same-batch natural choices matched the trusted run on `988/1000` condition-question outputs (98.8%).
The maximum trusted-logit discrepancy was `0.125` logits, consistent with the changed NVIDIA-driver numerical regime. All causal contrasts use the same-batch natural companion, not the trusted logits.
Maximum absolute intervention-induced A-D logit change: `1.375000`.

Canonical figure: `figures/qwen36_final_query_repeated_option_ablation.png`.
