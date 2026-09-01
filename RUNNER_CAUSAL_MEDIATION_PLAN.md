# Qwen3.5-397B PopMC runner-residual causal test

## Question

Does the candidate-specific residual that appears around readout 49 causally
mediate the model's excess Second Chance switches to its baseline runner-up?

The natural behavioral gap is defined before intervention as the number of
Game answers equal to the baseline runner-up minus the corresponding number in
Neutral. In the paired self-hosted run this gap was 24 trials.

## Candidate signal

The candidate signal is exactly the previously reported, cross-fitted flexible
leave-runner-out residual at readout 49. The compression model is fit without
ever seeing runner-up observations. For each question, the signal is the Game
runner residual minus the Neutral runner residual. It is computed without
conditioning on the eventual Game or Neutral answer.

The residual's scalar contrast is the runner-up score minus the mean score of
the other three A-D candidates. Thus it tests a runner-specific exception to
broad compression; it is not a generic Game-minus-Neutral residual direction.

## Intervention

At the final semantic answer position immediately after block 48 (readout 49):

- **Necessity:** change the Game residual just enough to remove 0.5x or 1.0x of
  that question's signed candidate signal.
- **Sufficiency:** change the Neutral residual just enough to add 0.5x or 1.0x
  of the same signal.

The residual update follows the local minimum-norm gradient of the native
final-norm-plus-unembedding runner contrast. It is iteratively calibrated so
that its measured logit-lens contrast change equals the requested amount before
the remaining eleven blocks run normally. No final logits are edited.

## Controls

- Apply the same signed contrast change to baseline rank 3 instead of the
  runner-up, at 0.5x and 1.0x.
- Apply an equal-L2-norm random residual update locally orthogonal to all four
  A-D scores, at 0.5x and 1.0x.
- Apply the runner update at readout 40 rather than the predictive readout 49.
- Add the runner signal to the ordinary baseline capability prompt to measure
  generic answer disruption and accuracy cost.
- Re-run natural Baseline, Game, and Neutral on the same software stack as an
  exact replication check.

## Primary estimands

1. Change in Game-to-runner switches under removal.
2. Change in Neutral-to-runner switches under addition.
3. Closure of the natural net runner-switch gap.
4. Dose response from 0.5x to 1.0x.
5. Specificity versus rank-3, answer-orthogonal, and early-layer controls.

The necessity-mediated fraction is

`(natural runner gap - runner gap after Game removal) / natural runner gap`.

Sufficiency is reported separately rather than being folded into that fraction.
A positive, dose-dependent necessity effect plus the converse Neutral effect is
the criterion for proceeding to module-output localization around blocks 48-49.

## Budget and lifecycle

Run the mediation stage first under the standing $15 batch ceiling. Proceed to
module localization only if the primary intervention has a clear bidirectional
effect and sufficient budget remains. Stop, but do not destroy, the cached Vast
instance after retrieval.
