# Layerwise course of W1 suppression after option remapping

## Definitions

- **W1**: the semantic option selected by Baseline on the original first presentation.
- **W2**: the semantic option selected when Baseline solves the remapped second presentation afresh.
- **Conflict trial**: W1 and W2 are different semantic contents (n=273).
- **Centered evidence**: the decoded score for an option's currently displayed A-D token minus the mean decoded score across all four answer tokens for that same question, condition, and layer.
- **Game minus Neutral**: a paired within-question difference. A negative W1 value means Game represents W1 less strongly than Neutral does.

The plot uses both JLens and ordinary logit lens at the final decision position. Pointwise 95% confidence intervals bootstrap questions within original-W1 letter strata.

## Layerwise result

Game and Neutral remain very similar through the early and middle readouts.
Both lenses then show the same late, rank-specific transformation:

- the Game-minus-Neutral W1 contrast first becomes practically noticeable at
  readout 52;
- it grows across readouts 53--55;
- the largest single step occurs at readout 56; and
- it remains strongly negative through the final output readout.

For JLens, the paired W1 contrasts at readouts 52, 54, 56, and 60 are
-0.168, -0.425, -1.241, and -0.945 score units. The corresponding ordinary
logit-lens contrasts are -0.143, -0.472, -1.043, and -0.760 logit units.
Agreement between two independently calibrated readouts makes the late onset
robust to the choice of lens. This localizes where the semantic redistribution
becomes explicit at the final decision position; it does not by itself identify
the upstream component or state that causes it.

## Final readout check

At readout 64, JLens and logit lens coincide with the model's natural output readout. The paired Game-minus-Neutral centered-evidence differences are:

- W1: -0.476 [-0.553, -0.401] logits.
- W2: -0.065 [-0.143, +0.009] logits.
- Mean of the other two contents: +0.270 [+0.229, +0.313] logits.

Thus the final transformation is not merely a uniform decrease in all answer
evidence. Game selectively reduces W1, leaves W2 close to unchanged on average,
and redistributes evidence toward the other two semantic options. This is the
layerwise expression of the previously established behavioral result that Game
avoids the first-pass semantic winner under remapping.

The figure is `figures/qwen36_simplemc_remapped_w1_layerwise.png`.
