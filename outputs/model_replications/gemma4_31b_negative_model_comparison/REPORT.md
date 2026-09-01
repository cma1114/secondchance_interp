# Gemma 4 31B negative-model comparison

## Executive result

Gemma 4 31B is not a clean example of a model that fails to understand or
implement the Game policy.  In the current clean prompt, it shows almost no
*choice-rate* Game advantage over Neutral after semantic remapping, but its
continuous answer scores contain the same qualitative policy signature seen in
the successful Qwen and Seed models: relative to Neutral, Game selectively
lowers the semantic first-presentation winner and raises the lower-ranked
candidates.  The feedback suffix causally transfers essentially all of that
continuous policy effect.  Matching semantic first-presentation history is
also causally used, and the way it is used differs by task.

The principal failure is therefore late and quantitative rather than an absence
of the mechanism: Gemma constructs a policy state, recollects semantic candidate
history, and combines the two, but the resulting score changes usually do not
cross its larger or more stable winner margins.  The internal Game/Neutral
difference is robust while the displayed choice difference is small.

This report distinguishes four evidence classes throughout: behavioral choices,
activation/decoding, causal intervention, and inference.  No activation result
alone is treated as evidence of causal use.

## Executed scope and controls

- Model: official `google/gemma-4-31B-it` revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16, Transformers 5.5.4.
- Datasets: all 500 canonical SimpleMC questions and all 500 frozen
  difficulty-filtered TriviaMC questions.
- Prompt: baseline-matched empty first assistant turn; Game and Neutral differ
  only in `incorrect` versus `lost`, followed identically by
  `Choose the answer again.`
- Architecture: every text layer L1--L60, comprising 50 sliding-attention and
  10 full-attention layers.
- Candidate identity: semantic first-presentation rank from Gemma's own
  aggregated A--D scores; complete derangements in remapped trials; displayed
  order resolves exact ties.
- Every causal run has finite-output checks, real identity controls, prompt-hash
  checks, positive edited-edge counts, and exact reproduction of its trusted
  one-row reference.  Gemma's low-precision logits vary with batch composition,
  so cross-row causal effects use same-composition contrasts and report raw
  mixed-batch drift separately.

The frozen design is in [PLAN.md](PLAN.md).

## 1. Behavior: the choice-rate effect fails, but the score policy survives

### SimpleMC

After complete semantic remapping, Game switches away from the semantic old
winner on 46.8% of questions and Neutral on 46.2%, a +0.6-point paired gap with
95% CI [-2.2,+3.4].  The frozen confirmation gap is +1.6 points
[-2.4,+5.6].  Thus the successful-model behavioral gate does not replicate.

Nevertheless, the all-question Game-minus-Neutral centered A--D score change by
old rank is `R1 -0.850, R2 +0.121, R3 +0.366, R4 +0.362` logits.  The effect
tracks semantic old-winner content rather than its former displayed letter.

### TriviaMC

After remapping, Game switches on 18.4% and Neutral on 17.6%, a +0.8-point gap
[-0.6,+2.2]; confirmation is +1.2 points [-0.8,+3.2].  Again there is no robust
choice-rate replication.  The continuous rank profile does replicate:
`R1 -0.650, R2 +0.176, R3 +0.283, R4 +0.190` logits.

**Behavioral conclusion.** Gemma has a stable semantic Game scoring policy on
both datasets, but that policy only rarely changes the argmax relative to
Neutral.  Describing Gemma simply as “not switching strategically” would miss
the main result.

Primary artifacts: SimpleMC [summary](simplemc/behavior/analysis/summary.json)
and [figure](../../../figures/model_replications/gemma4_31b_simplemc_clean_behavioral_gate.png);
TriviaMC [summary](triviamc/behavior/analysis/summary.json) and
[figure](../../../figures/model_replications/gemma4_31b_triviamc_clean_behavioral_gate.png).

## 2. Final-position dynamics: the policy-adjusted answer pattern exists before the logit lens can read it

The standard Gemma logit lens shows little stable answer-rank separation until
the late layers.  On questions that ultimately switch, R2 is already the
largest output-readable old-rank score when the late separation becomes clear
in both Game and Neutral; there is no visible sequence in which Game first
commits to R1 and then uniquely suppresses it.  The Game and Neutral switch
trajectories are strikingly similar, consistent with the small choice-rate gap.

That does not mean the final-position residual is uninformative earlier.
Discovery-fitted linear decoders evaluated only on held-out confirmation
questions recover question-specific final answer structure well before the
standard output basis does.  The policy-adjusted Game-minus-Neutral rank vector
is persistently above its frozen sign-shuffle null from L35 in SimpleMC and L37
in TriviaMC.  At L40 its held-out cosine with the exact final policy vector is
0.443 [0.377,0.508] on SimpleMC and 0.255 [0.183,0.326] on TriviaMC; by L44 it is
0.675 [0.627,0.723] and 0.607 [0.552,0.659].  The exact final vectors are
`[-0.890,+0.171,+0.302,+0.416]` and
`[-1.041,+0.317,+0.098,+0.626]`.

Shared and cross-condition prospective decoders perform close to matched-task
decoders, so Game and Neutral do not appear to encode the upcoming answer in
wholly different bases.  The task difference emerges in a largely shared
prospective-answer subspace.

**Activation/decoding conclusion.** Gemma forms a linearly accessible,
policy-adjusted prospective answer pattern in the middle-late network, then
rotates/amplifies it into directly output-readable A--D logits later.  This is
noncausal evidence about timing, not proof that the decoded coordinates are
used.

Artifacts: [trajectory summary](trajectory_analysis/summary.json),
[prospective-decoder summary](prospective_analysis/summary.json), and the
indexed figures in `figures/model_replications/`.

## 3. Matching-history blockade: semantic recollection is causally active

On remapped prompts, every token in each second-presentation option line was
prevented from attending to every token in its semantically matching
first-presentation option line at all L1--L60.  The control blocked the same
receiver lines and layer range but used a cyclic wrong source line.  This is an
ordinary-attention edge intervention; it is not a residual ablation.

On frozen SimpleMC confirmation questions, matching-minus-cyclic changes the
Game-minus-Neutral rank profile by
`R1 +1.360 [1.049,1.667], R2 -0.220 [-0.484,0.047],`
`R3 -0.470 [-0.699,-0.241], R4 -0.671 [-0.876,-0.470]` logits.
TriviaMC independently gives
`R1 +0.716 [0.438,0.994], R2 -0.352 [-0.614,-0.091],`
`R3 -0.214 [-0.406,-0.019], R4 -0.150 [-0.351,0.060]`.

The positive R1 interaction means that removing the true semantic-history route
specifically undoes Game's relative suppression of the old winner.  The
lower-rank signs show the complementary loss of Game's relative support for
alternatives.  This is a cross-dataset causal interaction in continuous scores.

The discrete W1-choice-gap changes are much smaller and uncertain: +0.8 points
[-4.0,+5.6] on SimpleMC confirmation and +2.8 [-0.4,+6.0] on TriviaMC.  That is
not a contradiction.  The intervention changes the candidate scores in the
predicted direction, but Gemma's argmax often remains on the same candidate.

Artifacts: SimpleMC [report](simplemc/matching_history/analysis/REPORT.md),
[summary](simplemc/matching_history/analysis/summary.json), and
[figure](../../../figures/model_replications/gemma4_31b_simplemc_matching_history.png);
TriviaMC [report](triviamc/matching_history/analysis/REPORT.md),
[summary](triviamc/matching_history/analysis/summary.json), and
[figure](../../../figures/model_replications/gemma4_31b_triviamc_matching_history.png).

## 4. Feedback-suffix crossover: the policy source is intact

The complete seven-token contextualized feedback suffix, spanning
`incorrect/lost` through the final period in `Choose the answer again.`, was
reciprocally installed through its downstream ordinary-attention K/V state at
every L1--L60.  The source tokens' own residual outputs remain recipient-natural;
this is a downstream K/V crossover, not a residual transplant.

On SimpleMC confirmation, the donor suffix transfers 0.9995 of the natural
continuous Game/Neutral score vector in both directions; discovery gives
0.9998.  On TriviaMC, transfer is 0.9554 on confirmation and 0.9750 on
discovery.  Every transfer interval is positive and every real same-task
identity control is exactly zero-error.

**Causal conclusion.** Gemma constructs a distinct policy state in the same
contextualized feedback suffix as the successful models, and that state is
sufficient to impose nearly the entire opposite-task A--D score transformation.
The small choice changes reflect argmax margins, not failure of policy transfer.

Artifacts: SimpleMC [report](simplemc/feedback_suffix/analysis/REPORT.md),
[summary](simplemc/feedback_suffix/analysis/summary.json), and
[figure](../../../figures/model_replications/gemma4_31b_simplemc_feedback_suffix.png);
TriviaMC [report](triviamc/feedback_suffix/analysis/REPORT.md),
[summary](triviamc/feedback_suffix/analysis/summary.json), and
[figure](../../../figures/model_replications/gemma4_31b_triviamc_feedback_suffix.png).

## 5. Direct policy × recollection factorial

This final stage crossed installed Game versus Neutral suffix state with intact,
matching-blocked, and cyclic-wrong history access in both recipient prompts.
The primary endpoint asks whether the installed policy changes the
matching-specific rank route independently of the recipient's original task
word.

It does.  On frozen SimpleMC confirmation questions, installed Game minus
installed Neutral changes the matching-minus-cyclic route by
`R1 +1.398 [1.087,1.713], R2 -0.195 [-0.468,0.079],`
`R3 -0.499 [-0.732,-0.274], R4 -0.705 [-0.909,-0.497]` logits after averaging
over recipient prompts.  The effect replicates on frozen TriviaMC confirmation:
`R1 +0.683 [0.392,0.973], R2 -0.348 [-0.614,-0.091],`
`R3 -0.212 [-0.400,-0.013], R4 -0.124 [-0.330,0.090]`.

Within numerical precision, each interaction vector is identical in native Game
and native Neutral recipient prompts.  Thus it is the installed seven-token
suffix state—not some unedited task residue elsewhere in the recipient
prompt—that determines how matching semantic history is transformed.  The
factorial's intact-access reciprocal cells reproduce the earlier suffix
crossover exactly (maximum error 0.0), every real same-policy identity is exact,
and every output is finite.

The displayed-choice interactions remain small and uncertain: +1.6 points
[-2.8,+6.0] on SimpleMC confirmation and +2.0 [-1.0,+5.0] on TriviaMC after
averaging recipients.  This supplies the cleanest statement of Gemma's negative
result: **policy × recollection is causally present in answer scores, but usually
does not alter which candidate wins.**

Artifacts: SimpleMC [report](simplemc/policy_recollection/analysis/REPORT.md),
[summary](simplemc/policy_recollection/analysis/summary.json), and
[figure](../../../figures/model_replications/gemma4_31b_simplemc_policy_recollection_factorial.png);
TriviaMC [report](triviamc/policy_recollection/analysis/REPORT.md),
[summary](triviamc/policy_recollection/analysis/summary.json), and
[figure](../../../figures/model_replications/gemma4_31b_triviamc_policy_recollection_factorial.png).

## Integrated interpretation

The negative comparison localizes the difference from Qwen3.6-27B and Seed-OSS
36B more sharply than a behavior-only test could:

1. **Not missing policy.** The feedback suffix carries an almost complete,
   transferable Game/Neutral score policy.
2. **Not missing recollection.** Matching semantic first-presentation history
   causally changes second-pass final scores.
3. **The two are combined.** Directly crossing suffix policy with history access
   proves that the installed policy changes how the matching recollection route
   treats R1 and the lower-ranked candidates, in either recipient prompt.
4. **Not mere noise.** The final Game-minus-Neutral vector has a stable semantic
   rank structure, is decodable before it becomes output-readable, transfers
   with the policy source, and is selectively undone by cutting the matching
   semantic-history route.
5. **The likely bottleneck is commitment.** Gemma's internal score adjustment
   is typically too small relative to its candidate margins to create a robust
   Game-minus-Neutral choice-rate difference.  This is an inference from the
   joint behavioral and causal evidence, not a separately lesioned “margin
   threshold” module.

The result matters because it separates *having the strategic computation* from
*expressing it as extra switches*.  Across models, choice-level behavioral
success is a thresholded readout of a richer policy-conditioned scoring process.
