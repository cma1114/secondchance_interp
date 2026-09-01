# Mixer 56 semantic-target transfer

## Bottom line

This experiment asks whether Mixer 56 contains a causal, question-specific
comparison between the answer implicit in the first presentation and the
answers in the repeated presentation.  It keeps the visible Game prompt after
the first presentation fixed and changes only which semantic answer won in a
donor first presentation.

The frozen primary intervention (heads 0, 2, 6, and 15: final query, output
gate, and repeated-option/choice-cue K/V state) produced an immediate Mixer 56
target-transfer effect of **-0.437 [-0.693, -0.194]**
and a final-logit target-transfer effect of
**-0.036 [-0.094, +0.019]** on the held-out confirmation
questions (n=104). Positive values mean evidence moved away from
the donor's first-pass semantic winner and toward the recipient's original
winner.

The sign is therefore decisive: **the donor state made Mixer 56 reinforce the
donor's first-pass semantic answer, not suppress it.** This reverse immediate
effect replicated in discovery
(-0.359 [-0.674, -0.072]). It was then
mostly cancelled downstream: the held-out final-logit interval includes zero,
and the behavioral choice changes below are small. The broad all-head patch
does not rescue suppression-target transfer.

Query alone produced a very small positive final effect in confirmation
(+0.018 [+0.002, +0.035]), but not in discovery
(+0.005 [-0.011, +0.021]), and did not yield a
meaningful behavioral transfer. It is not convincing evidence for a separate
query-carried suppression target.

The all-head positive control gave **-0.016 [-0.074, +0.038]**
at the final logits. The same-winner/different-order control gave
**+0.054 [-0.024, +0.137]**. Its available sample is
n=45.

Because the frozen changed-winner sample contains many original-A winners, the
prespecified content-bias sensitivity is important. On the held-out questions
whose recipient first-pass winner was not A (n=35), the primary
final-logit target-transfer effect was
**-0.059 [-0.132, +0.015]**. Letter-stratified effects for
all four recipient winners are preserved in `summary.json`.

The primary patch changed the selected output on
10/104 held-out questions. Donor-winner
selection changed by -1.923 [-6.731, +2.885] percentage
points and recipient-winner selection by
+0.962 [-2.885, +4.808] percentage points.

## What was patched

- One ordinary attention component only: Mixer 56.
- Query heads 0, 2, 6, and 15 at the final decision position.
- Their per-head output gates at that position.
- K/V state for the corresponding KV heads over all four repeated option lines
  and the final `Your choice (A, B, C, or D): ` cue.
- Factorial query-only, gate-only, query+gate, K/V-only, and joint conditions,
  plus the all-head and same-winner controls, are in `summary.json` and the
  figure.

This is not a Game-to-Neutral prefix replacement: both donor and recipient are
Game prompts, and the feedback plus complete second presentation are identical.

## Interpretation

This experiment rejects the proposed narrow story in which Mixer 56 computes
"this repeated option matches my old answer" and uses that match to inhibit the
old answer. Its K/V pathway instead carries a donor-specific **reinstatement or
reconstruction** signal: changing the implicit first-pass answer changes which
semantic answer Mixer 56 amplifies. That is a genuine content-specific causal
effect at the component output, but downstream computation nearly cancels it,
so Mixer 56 is not the behavioral revision mechanism by itself. The natural
Game-versus-Neutral difference at Mixer 56 is now best read as weaker
reinstatement in Game, not as direct suppression executed by Mixer 56.

## Validation

- Natural recipient logits were bit-exact against the frozen cross-order Game
  run that defined these donors: **True**; maximum absolute
  difference 0.
- All 500 natural winners matched (500/500).
- Donor and recipient token sequences and patch coordinates were audited over
  the identical repeated-option/cue suffix.
- Discovery and confirmation use the pre-existing frozen 251/249 split and
  independently frozen cross-order donor plan.

## Figure

![Mixer 56 semantic-target transfer](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_simplemc_mixer56_semantic_target_transfer.png)
