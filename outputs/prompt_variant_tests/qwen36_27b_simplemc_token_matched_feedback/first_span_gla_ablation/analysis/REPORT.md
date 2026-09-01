# First-presentation GLA-memory ablation

The intervention removes selected first-presentation writes from all 48 GLA layers while preserving each target's exact historical four-question cohort.

Discrete answers in this report resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. The report was regenerated after the remapped-answer tie audit; all continuous quantities are invariant to that correction.

## Natural behavior

Discovery discordant W1/W2 questions: **137**; natural targeting contrast: **+0.235 [+0.073, +0.396] logits**.
Confirmation discordant questions: **136**; natural targeting contrast: **+0.587 [+0.391, +0.770] logits**.

## Frozen confirmation

The most interpretable direct outcome is the change in the W1-minus-W2 margin within each condition. Positive values mean the lesion makes the model more likely to retain the semantic answer it reached on the first presentation rather than choose the answer it would have reached by freshly solving the remapped presentation.

| Source writes removed | Game W1-W2 margin | Game W1 choice | Neutral W1-W2 margin | Neutral W1 choice | Difference-in-differences reduction |
|---|---:|---:|---:|---:|---:|
| Question + options | -0.671 [-0.926, -0.424] | -8.1% | -1.158 [-1.444, -0.880] | -26.5% | +0.486 [+0.274, +0.708] |
| Options only | -0.255 [-0.494, -0.025] | +0.0% | -0.883 [-1.152, -0.627] | -20.6% | +0.628 [+0.446, +0.799] |
| First-answer boundary | +0.243 [+0.105, +0.377] | +9.6% | +0.009 [-0.109, +0.125] | +4.4% | +0.234 [+0.096, +0.372] |
| Question/options + boundary | -0.650 [-0.924, -0.381] | +0.7% | -1.501 [-1.829, -1.201] | -26.5% | +0.851 [+0.625, +1.070] |

All margin changes and choice-rate changes are lesion minus natural within the named condition. The difference-in-differences column is secondary because it can change either by weakening Game suppression or by weakening Neutral retention.

## Interpretation

The first-answer-boundary lesion is the clean evidence for the route that carries the prior semantic decision into Game. On frozen confirmation it raises Game's W1-minus-W2 margin and W1 selection, while leaving Neutral's W1-minus-W2 margin approximately unchanged. Thus, recurrent GLA writes made while processing the empty first assistant boundary preserve information that the later incorrect-feedback computation uses to disfavor the semantic answer reached on the first presentation.

The option-token lesion has a different role: it lowers the W1-minus-W2 margin in both conditions, especially Neutral. Those writes primarily support retaining/reconstructing the first answer, rather than implementing Game-specific suppression. The combined lesion therefore should not be described as a unitary suppression mechanism even though it has the largest difference-in-differences effect.

## Historical-run validation

Discovery natural A–D winners matched the saved run on 251/251 Game and 251/251 Neutral questions.
Confirmation matched on 249/249 Game and 249/249 Neutral questions.
