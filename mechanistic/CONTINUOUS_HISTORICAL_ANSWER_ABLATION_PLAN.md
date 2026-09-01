# Continuous historical-answer identity ablation

## Question

Does the model use a linearly represented identity of its first answer at the
historical first-answer generation endpoint when choosing its second answer?

## Intervention

On the frozen 249-question SimpleMC confirmation set, intervene at the exact
historical first-answer endpoint before every Mixer from one-based layers 33
through 64. At every layer, use that layer's learned JLens decoder to perform
one of two projections:

1. Remove the live matched-Baseline winner-versus-other-answers direction.
2. Remove the complete centered three-dimensional A-D identity subspace.

The projection is repeated before every selected Mixer so that later layers
cannot simply reconstruct the linearly decoded identity at this position.
Nothing is changed at other token positions.

Each intervention has a deterministic A-D-orthogonal perturbation at the same
position and layers, with its norm matched separately to the semantic
projection at every layer. Game and Neutral are tested separately.

## Predictions

If the same retrieved first-answer representation supports both conditions,
erasing it should make Game less able to avoid the first answer and Neutral
less able to repeat it. The primary signature is therefore decreased switching
in Game together with increased switching in Neutral. Secondary outcomes are
the matched-Baseline winner's probability and logit contrast, A-D entropy, and
the fraction of final answers changed by the intervention.

This experiment cannot remove first-answer information copied to other token
positions before layer 33, nor nonlinear identity information outside the
JLens A-D subspace.

