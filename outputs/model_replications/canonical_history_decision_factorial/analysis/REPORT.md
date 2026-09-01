# Canonical non-remapped matching-history and first-decision-source factorial

## Question

On prompts with no option remapping, does the later answer depend on candidate-specific reads from the first option lines, on the complete state at the first assistant answer boundary, or on redundant use of both routes?

## Exact interventions

- **Matching:** every token of each second-presentation option line was prevented from attending to every token of the identical first-presentation option line at every ordinary-attention layer.
- **Cyclic wrong-line control:** the same four receiver lines were instead denied a cyclically wrong first option line.
- **First-decision source:** every later ordinary-attention query was prevented from reading the exact token at which the first assistant answer would have begun. In Qwen, that token's write into every recurrent GLA layer was also removed.
- **Joint:** matching-line and first-decision routes were removed together.

The first-decision intervention removes the token's complete outgoing state. It tests whether that position is a causal source, but does **not** by itself prove that any effect is carried specifically by a literal answer-letter coordinate.
The answer letter itself is omitted from the history in this paradigm. The tested source is the final token of the first assistant-generation prefix—the residual state whose logits would have generated that letter (Qwen's final blank-line token, Seed's final `assistant` wordpiece, or Gemma's following newline).

## Main finding

The candidate-matched first-option-line route, not the first answer-boundary state, carries the dominant old-winner policy effect on canonical non-remapped prompts. On frozen confirmation, matching-line blockade weakened the Game-versus-Neutral W1-logit difference in all six model/dataset cells. The cyclic wrong-line control was approximately null. In contrast, cutting every outgoing route from the first answer-boundary state was small: it did not reproducibly reduce the choice gap in any cell, and its W1-logit effect was only a small fraction of the natural gap.

This run alone cannot separate semantic content from displayed letter because those are aligned in a non-remapped prompt. Combined with the prior remapped result—where matching follows candidate meaning rather than the old literal letter—it shows that the semantic matching route is not an artifact created by remapping. The missing first-answer boundary is not the main recollection source.

## Frozen-confirmation results

The natural columns give the unedited Neutral-minus-Game old-W1 choice gap and Game-minus-Neutral centered W1 logit. Negative natural logits mean that Game suppresses W1 relative to Neutral. Intervention entries are reductions in those natural effects; positive values mean that the intervention weakened the policy difference.

| Model | Dataset | Natural choice pp | Natural W1 logit | Matching choice pp | Decision choice pp | Joint choice pp | Matching W1 logit | Decision W1 logit | Joint W1 logit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B | SimpleMC | +9.24 | -0.52 | +10.44 [+4.02, +16.87] | +2.81 [-3.21, +8.84] | +7.23 [+1.20, +13.25] | +0.55 [+0.46, +0.64] | +0.05 [-0.02, +0.12] | +0.77 [+0.63, +0.92] |
| Qwen3.6-27B | TriviaMC | +5.60 | -1.60 | +6.00 [+2.00, +10.00] | +1.60 [-2.00, +4.80] | +7.20 [+3.20, +11.60] | +1.27 [+1.10, +1.43] | -0.04 [-0.14, +0.05] | +1.76 [+1.55, +1.98] |
| Seed-OSS 36B | SimpleMC | +11.24 | -1.44 | +25.30 [+18.47, +32.53] | -0.40 [-4.42, +3.61] | +23.29 [+16.47, +30.52] | +1.95 [+1.77, +2.12] | +0.08 [+0.04, +0.12] | +1.94 [+1.76, +2.12] |
| Seed-OSS 36B | TriviaMC | +4.40 | -2.11 | +6.40 [+2.80, +10.40] | -2.80 [-5.61, +0.00] | +6.80 [+3.20, +10.80] | +1.92 [+1.70, +2.15] | +0.12 [+0.07, +0.17] | +1.89 [+1.66, +2.11] |
| Gemma 4 31B | SimpleMC | +0.80 | -0.94 | +0.40 [-4.02, +4.82] | -0.80 [-2.81, +1.20] | -0.80 [-5.22, +3.61] | +0.86 [+0.68, +1.05] | -0.03 [-0.08, +0.02] | +0.82 [+0.64, +1.00] |
| Gemma 4 31B | TriviaMC | +3.20 | -1.03 | +2.00 [-0.40, +4.40] | +1.20 [-0.40, +3.20] | +2.80 [+0.40, +5.20] | +0.91 [+0.72, +1.10] | +0.09 [+0.02, +0.16] | +0.90 [+0.71, +1.10] |

## Redundancy check

The joint cell matters because a null first-decision blockade alone could be hidden by a backup route. Direct joint-minus-matching contrasts show little additional effect in Seed or Gemma. Qwen is the exception at the continuous-logit endpoint: once the matching line route is already cut, also cutting the first boundary pushes W1 farther in the same direction. This is a nonlinear backup/interaction, not evidence that the boundary is the primary route: the boundary cut alone remains approximately null, while matching blockade alone removes the replicated choice effect and most of the logit effect.

| Model | Dataset | Joint − matching choice pp | Joint − matching W1 logit |
|---|---|---:|---:|
| Qwen3.6-27B | SimpleMC | -3.21 [-7.63, +1.20] | +0.22 [+0.12, +0.33] |
| Qwen3.6-27B | TriviaMC | +1.20 [-2.00, +4.40] | +0.50 [+0.38, +0.61] |
| Seed-OSS 36B | SimpleMC | -2.01 [-4.42, +0.40] | -0.01 [-0.03, +0.02] |
| Seed-OSS 36B | TriviaMC | +0.40 [-1.20, +2.00] | -0.04 [-0.07, -0.01] |
| Gemma 4 31B | SimpleMC | -1.20 [-4.82, +2.01] | -0.05 [-0.11, +0.02] |
| Gemma 4 31B | TriviaMC | +0.80 [+0.00, +2.00] | -0.01 [-0.08, +0.08] |

## Validity

All six runs used 500 questions, reproduced the frozen natural A–D logits exactly, produced finite outputs, and edited the complete architecture-specific ordinary-attention inventory. Qwen's first-decision cells additionally edited all 48 recurrent layers. Discovery/confirmation membership came from the previously frozen dataset split files.

Canonical figure: `/Users/christopherackerman/repos/secondchance_interp/figures/model_replications/canonical_history_decision_factorial.png`
