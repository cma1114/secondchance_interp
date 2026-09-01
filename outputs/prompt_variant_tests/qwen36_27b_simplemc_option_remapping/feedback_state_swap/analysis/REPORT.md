# Remapped W1 feedback-state swap and sublayer localization

> **Partial supersession (2026-08-24).** Result 1 below was produced by a
> runner that applied unpadded feedback-token indices to left-padded batches.
> It is not valid evidence by itself. The corrected state-swap analysis in
> `source_contributions_and_swaps/analysis/REPORT.md` independently confirms
> the same null. Result 2 used final-position indices and is unaffected.

## Design

- All 500 frozen SimpleMC questions; primary semantic analysis on the 273 W1 != W2 conflict trials.
- Exact historical batch-of-four SDPA cohorts.
- At readouts 48--56, replace the complete post-block state of Game's `incorrect` token with Neutral's paired `lost` state, or vice versa, and let that one replacement propagate through the rest of the model.
- In the unmodified passes, decode the final-decision residual before the mixer, after the mixer, and after the MLP in blocks 52--56.
- Natural A--D logits reproduce the trusted remapped run exactly (maximum absolute error 0.0).

## Result 1: the late `incorrect` / `lost` token states are not the controller

The numerical estimates in this section are historical and superseded because
of the padding bug described above; cite the corrected rerun for this claim.

Replacing the complete state of one evaluation token with the paired state of
the other condition has essentially no effect at any tested readout. In the
theoretically important Neutral-into-Game direction, the change in W1 evidence
ranges only from -0.0044 to -0.0001 centered logits. W1 selection changes by at
most +0.73 percentage points. Game-into-Neutral is similarly null.

There are tiny effects on W2 (for example, +0.0104 logits when the readout-48
Neutral state is inserted into Game), but these are two orders of magnitude
smaller than the natural -0.476-logit Game-minus-Neutral W1 contrast. The late
state at the literal evaluation token is therefore neither necessary nor
sufficient for the semantic suppression expressed at the final decision
position. The condition information must already have been routed elsewhere or
be distributed across other positions/recurrent state by these layers.

## Result 2: four late sublayers create most of the explicit W1 divergence

The natural within-block decomposition is highly localized. Values below are
the incremental change in the paired Game-minus-Neutral centered W1 contrast:

| Sublayer | Incremental change |
|---|---:|
| Mixer 52 | -0.286 |
| Mixer 53 | -0.182 |
| MLP 54 | -0.128 |
| Mixer 56 | -0.966 |

All other mixers/MLPs in blocks 52--56 change the contrast by at most 0.094 in
absolute value. Mixer 56 is the dominant transformation, but its operation is
best described as **differential sharpening**, not a literal negative W1 write:

- immediately before Mixer 56, W1 evidence is +0.219 in Game and +0.600 in
  Neutral;
- immediately after it, W1 evidence is +0.448 in Game and +1.795 in Neutral;
- Mixer 56 therefore boosts W1 in both conditions, but by about +0.229 in Game
  versus +1.195 in Neutral, enlarging the condition contrast by -0.966.

MLP 56 partially reverses that difference (+0.304). Mixer 52 is different: it
reduces Game W1 evidence (+0.156 to +0.027) while increasing Neutral
(-0.007 to +0.150), so it contains both literal Game-side reduction and
Neutral-side reinforcement. Mixer 53 and MLP 54 principally extend the
condition-dependent divergence.

Thus the final-output statement that Game "suppresses W1" decomposes into a
sequence of operations, dominated by Neutral strongly reinstating/sharpening W1
at Mixer 56 while Game receives only modest reinforcement. It is not primarily
a component writing a large negative W1 logit in Game.

## Conflict versus no-conflict trials

The late localization is not peculiar to W1 != W2 trials. On the 227 W1 = W2
trials, Mixer 52 changes the Game-minus-Neutral W1 contrast by -0.180, Mixer 53
by -0.179, MLP 54 by -0.093, and Mixer 56 by -1.096; MLP 56 then partially
reverses the divergence by +0.266. The corresponding conflict-trial changes
are -0.286, -0.182, -0.128, -0.966, and +0.304. Thus the same sequence of late
transformations appears in both groups, with Mixer 56 dominant in each.

The important difference is the state entering this sequence. Immediately
before Mixer 52, Game-minus-Neutral W1 evidence is +0.163 on conflict trials but
-0.160 on no-conflict trials. By the output of MLP 56 it is strongly negative
in both groups (-1.043 and -1.378, respectively). The late computation is
therefore a general condition-dependent W1/recommitment transformation;
conflict is not what turns it on. Conflict trials remain essential for
interpreting the target as the previous semantic answer rather than merely the
answer favored by both presentations.

The W2 decomposition explains the apparently paradoxical pre-52 sign. On
conflict trials immediately before Mixer 52, Game foregrounds W1 over W2
(centered evidence +0.156 versus -0.122), whereas Neutral foregrounds W2 over
W1 (-0.007 versus +0.487). By the output of MLP 56 these relations have
reversed: Game favors W2 (+0.587 versus +0.242 for W1), while Neutral favors W1
(+1.285 versus +0.808 for W2). Thus Game's initially higher W1 evidence does not
mean that Game is preparing to repeat W1. It is consistent with Game first
representing the old answer as the item to be rejected and then moving away
from it, while Neutral moves from a fresh remapped solution toward
reinstatement of the historical answer. On no-conflict trials W1 and W2 are the
same semantic content, so these two influences cannot oppose one another and
Neutral supports the shared candidate more strongly from the outset. This is a
representational trajectory, not yet a causal demonstration of the proposed
two-stage interpretation.

The `incorrect`/`lost` token-state swaps remain null in both groups: their
largest W1-evidence effects are only about 0.0044 logits on conflict trials and
0.0035 logits on no-conflict trials, with W1-choice changes below one percentage
point.

## Artifacts

- `figures/qwen36_simplemc_remapped_feedback_state_swap.png`
- `figures/qwen36_simplemc_remapped_late_sublayers.png`
- `summary.json`
