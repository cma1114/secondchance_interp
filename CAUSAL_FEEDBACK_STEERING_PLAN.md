# Causal feedback-direction steering plan

## Goal

Test whether the residual stream contains a direction that is not merely
correlated with the faithful Second Chance (`incorrect`) prompt, but causally
controls the characteristic change in final answer behavior.

The primary contrast remains the published paradigm:

- **Game:** `Your answer was incorrect. Choose a different answer.`
- **Neutral:** `Your response was lost in transmission. Choose again.`
- The previous assistant answer remains `[redacted]` in both cases.

No paraphrases or replacement behavioral paradigm are required for this first
causal test. Cross-wording generalization is a later test of whether a causal
direction represents abstract negative feedback rather than the faithful prompt
contrast.

## 1. Direction discovery from existing activations

For question \(q\) and residual readout \(l\), let
\(h^G_l(q)\) and \(h^N_l(q)\) denote the saved final-position residuals in the
Game and neutral conditions. Define the paired contrastive direction

\[
v_l = \frac{\mathbb{E}_{q\in\mathrm{train}}
    [h^G_l(q)-h^N_l(q)]}
   {\left\|\mathbb{E}_{q\in\mathrm{train}}
    [h^G_l(q)-h^N_l(q)]\right\|_2}.
\]

This is a one-dimensional linear probe: classification uses projection onto
\(v_l\), with the threshold halfway between the training Game and neutral
projection means. Questions—not condition rows—are assigned to five folds, so
the two versions of a question never cross the train/test boundary.

The direction used for causal intervention is trained on four folds. The fifth
fold is held out completely and supplies intervention questions. Sixty held-out
questions are selected with 15 examples for each original baseline-winner
letter, preventing a fixed A-D imbalance from driving the causal estimates.

For every layer, save:

- held-out balanced accuracy and ROC AUC;
- the Game-minus-neutral projection gap;
- direction stability across training folds;
- the unit feedback direction;
- a matched control direction obtained from a randomly signed mean of paired
  Game-minus-neutral residual differences and orthogonalized to the feedback
  direction.

Decodability is only a localization and quality-control result. It is not the
mechanistic claim.

## 2. Causal intervention

At residual readout \(l\), the intervention is applied after the corresponding
transformer block and before the next block:

\[
h'_l = h_l + \alpha\,\Delta_l v_l,
\]

where \(\Delta_l\) is the training-set difference between the mean Game and
neutral projections onto \(v_l\). Thus \(\alpha=1\) moves a residual by one
natural Game-neutral activation gap; the same unit has the same interpretation
in Game and neutral.

The primary scan uses readouts 24, 30, and 36:

- feedback-direction doses \(-1\) and \(+1\) in both Game and neutral;
- at readout 30, additional doses \(-0.5\), \(+0.5\), and \(+2\) in both
  conditions;
- at readout 30, matched orthogonal-control doses \(-1\) and \(+1\) in both
  conditions;
- unsteered Game and neutral runs.

This schedule explicitly tests:

- **Neutral + feedback:** sufficiency;
- **Game - feedback:** necessity;
- **Game + feedback:** amplification of switching/compression;
- **Neutral - feedback:** reverse-direction control;
- dose response, layer specificity, and matched-direction specificity.

All interventions operate only at the final prompt position. They do not alter
the prompt, show the model its redacted answer, append generated text, or change
the original game.

## 3. Outcomes

Every steered pass records exact final A-D logits, the canonical A-D argmax, and
the full-vocabulary top token. Analyses are paired by question and report:

- switch rate relative to the self-hosted baseline capability answer;
- original-winner probability and margin;
- runner-up probability relative to ranks 3-4;
- A-D entropy and A-D spread;
- accuracy, and accuracy among baseline-incorrect trials;
- invalid/non-A-D full-vocabulary top-token rate;
- paired effects relative to the unsteered run of the same condition.

Results are macro-averaged over original baseline-winner letter. Confidence
intervals resample questions within those four strata.

## 4. Evidence standard

A causal feedback-control result requires all of the following:

1. Adding the direction to neutral moves final behavior toward unsteered Game.
2. Removing it from Game moves final behavior toward unsteered neutral or
   baseline.
3. Adding it to Game amplifies at least one characteristic Game effect, such as
   switching or answer-distribution compression.
4. Effects are ordered by dose and are stronger near the identified layer range
   than at adjacent scan layers.
5. The matched orthogonal direction does not reproduce the effect.
6. Valid A-D output remains high; the result is not generic model damage.

Failure of these criteria means the binary direction is a correlate or prompt
identifier, not an established control variable.

## 5. Interpretation after causal validation

Only if the residual intervention succeeds will component interpretation begin:

1. Project each block's residual update onto the causal direction to identify
   candidate writers.
2. On a small question subset, record attention from the answer position to the
   system and final-user feedback tokens.
3. Ablate or steer the highest-contributing attention/MLP/MoE component and test
   the same exact final-logit outcomes.
4. Use wording variants only afterward to determine whether the causal feature
   generalizes from this faithful prompt contrast to abstract incorrectness.

## 6. Compute and spending guardrail

The initial run uses one 80 GB A100 or H100-class GPU and 60 held-out questions.
The model is loaded once, and the runner is resumable at the question/scenario
level. Before the full schedule, two held-out questions must pass a smoke test,
including numerical agreement between the unsteered logits and the previously
saved self-hosted logits.

The total Vast charge must remain below **$15**. Prefer a reliable offer whose
estimated full-run charge is substantially below that amount. Stop the instance
immediately after results and logs are downloaded and verified.
