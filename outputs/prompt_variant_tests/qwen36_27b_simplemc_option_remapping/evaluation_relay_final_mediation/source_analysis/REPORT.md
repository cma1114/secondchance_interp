# Feedback-source localization

This experiment reciprocally swaps only the downstream ordinary-attention and GLA memory writes of each feedback token from `incorrect/lost` through the final period. The source token's own residual remains natural. Game and Neutral are reported separately.

The prespecified complete-suffix relay-mediation gate **passed**.

## Confirmation results

| Source write crossed over | Game transfer (95% CI) | Game switch | Neutral transfer (95% CI) | Neutral switch |
|---|---:|---:|---:|---:|
| incorrect / lost | 0.076 [0.054, 0.099] | 61.4% | 0.393 [0.357, 0.429] | 51.8% |
| first period | 0.303 [0.278, 0.325] | 53.8% | 0.381 [0.353, 0.411] | 49.8% |
| Choose | 0.247 [0.214, 0.280] | 53.8% | 0.236 [0.200, 0.269] | 47.4% |
| the | 0.102 [0.081, 0.124] | 55.0% | 0.169 [0.139, 0.201] | 47.8% |
| answer | 0.033 [0.017, 0.048] | 61.4% | 0.137 [0.116, 0.159] | 47.0% |
| again | 0.045 [0.034, 0.057] | 62.2% | 0.037 [0.025, 0.049] | 47.0% |
| final period | 0.050 [0.039, 0.062] | 61.0% | 0.068 [0.053, 0.085] | 45.0% |
| complete suffix | 0.925 [0.910, 0.939] | 47.0% | 0.941 [0.927, 0.955] | 60.6% |

Natural switch rates and every discovery estimate are retained in `summary.json`.

## Interpretation rule

A token is a causal downstream source only to the extent that crossing over its complete ordinary-attention and GLA memory writes moves the recipient toward the paired donor task. Individual token effects are not added because later contextualized tokens can redundantly carry information from the earlier keyword.

Canonical figure: [qwen36_feedback_source_localization.png](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_feedback_source_localization.png)
