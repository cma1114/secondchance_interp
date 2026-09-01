# First-answer-boundary lesion and final W1 semantic activation

## Bottom line

The answer is **no, not in a replicated Game-specific way**. On held-out W1≠W2 questions, the lesion increased Game's W1−W2 output margin while leaving Neutral's margin approximately unchanged, reproducing the earlier causal result. But the final W1 semantic projection decreased by similar, statistically uncertain amounts in both conditions. The Game-minus-Neutral projection interaction was small and crossed zero. Discovery also showed no reliable interaction and did not reproduce the confirmation projection direction.

Therefore the first-answer-boundary GLA writes affect how the later computation ranks W1, but the effect is **not mediated by simply adding or removing the one-dimensional W1 semantic activation measured at the final decision position**. This narrows the missing mechanism to other residual dimensions or to condition-dependent use of an intact candidate representation.

The lesion is identical to the prior experiment. Natural and lesioned projections are paired within the present same-host run; saved historical logits are retained as a cross-host numerical validation reference.

A positive projection effect means that removing the boundary write makes the final decision residual more aligned with W1; a negative effect means it removes W1-aligned activation.

## Final-readout effects

| Split | Subset | Game | Neutral | Game minus Neutral |
|---|---|---:|---:|---:|
| Discovery | conflict (n=137) | +0.020 [-1.412, +1.418] | +0.808 [-0.407, +2.008] | -0.788 [-1.962, +0.452] |
| Discovery | no conflict (n=114) | +2.566 [+0.848, +4.294] | +2.739 [+1.059, +4.449] | -0.174 [-1.532, +1.111] |
| Discovery | conflict w1 a (n=89) | -0.138 [-1.834, +1.586] | +1.191 [-0.230, +2.640] | -1.329 [-2.810, +0.259] |
| Discovery | conflict w1 bcd (n=48) | +0.314 [-2.296, +2.825] | +0.097 [-2.130, +2.397] | +0.217 [-1.741, +2.174] |
| Confirmation | conflict (n=136) | -1.017 [-2.604, +0.479] | -1.173 [-2.771, +0.348] | +0.156 [-0.864, +1.175] |
| Confirmation | no conflict (n=113) | +1.372 [-0.130, +2.905] | +1.062 [-0.313, +2.417] | +0.309 [-0.817, +1.497] |
| Confirmation | conflict w1 a (n=94) | -1.354 [-3.343, +0.626] | -1.151 [-3.163, +0.820] | -0.203 [-1.388, +1.005] |
| Confirmation | conflict w1 bcd (n=42) | -0.261 [-2.784, +2.255] | -1.221 [-3.524, +1.058] | +0.960 [-0.822, +2.863] |
| Pooled | conflict (n=273) | -0.496 [-1.563, +0.547] | -0.179 [-1.184, +0.809] | -0.317 [-1.106, +0.460] |
| Pooled | no conflict (n=227) | +1.971 [+0.798, +3.130] | +1.905 [+0.836, +3.012] | +0.067 [-0.792, +0.973] |
| Pooled | conflict w1 a (n=183) | -0.763 [-2.064, +0.540] | -0.012 [-1.292, +1.230] | -0.751 [-1.728, +0.194] |
| Pooled | conflict w1 bcd (n=90) | +0.046 [-1.763, +1.813] | -0.518 [-2.161, +1.139] | +0.564 [-0.835, +1.961] |

## Held-out conflict-trial dissociation

- Game W1−W2 output-margin effect: +0.222 [+0.087, +0.352] logits.
- Neutral W1−W2 output-margin effect: -0.016 [-0.138, +0.099] logits.
- Game final W1-projection effect: -1.017 [-2.604, +0.479] residual units.
- Neutral final W1-projection effect: -1.173 [-2.771, +0.348] residual units.
- Game-minus-Neutral projection interaction: +0.156 [-0.864, +1.175] residual units.
- Projection-change/output-margin correlation: Game r=0.176; Neutral r=0.066.

The lesion also increased absolute projection magnitude on conflict trials in both conditions; this reflected mixtures of stronger positive and negative projections rather than selective removal of W1. Those descriptive sign-resolved quantities are stored in `summary.json`.

## Cross-host numerical validation

The present run uses the same model revision, prompts, batch-of-four cohorts, SDPA path, and intervention. Exact low-order bfloat16 equality is not expected across retained hosts. The summary records maximum A-D logit deviations and discrete A-D argmax agreement against both historical natural and lesion outputs.

## Interpretation limits

The measured projection is the one-dimensional four-mapping W1 candidate direction. A lesion effect on this projection shows that first-boundary GLA writes help construct or regulate that representation. A null effect would instead imply that the earlier behavioral/logit effect operates through other dimensions or through how an intact W1 representation is used. Neither result by itself proves that the boundary stores a portable semantic memory; the prior donor-state transplants directly tested and rejected that stronger claim.

Canonical figure: `figures/qwen36_first_boundary_semantic_projection.png`.
