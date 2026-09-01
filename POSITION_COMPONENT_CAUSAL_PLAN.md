# Qwen3.6-27B position-resolved component mediation

## Question

Under the corrected empty-history prompt, which mixer and MLP outputs carry
the Game-specific transformation from the feedback prompt into the final A-D
decision?  The analysis is deliberately agnostic about whether the relevant
effect is compression, rank-selective redistribution, or switching.

## Fixed model, prompts, and split

- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Dataset: the frozen 500-question SimpleMC manifest.
- Prompt format: `baseline_matched_empty_history` as specified in
  `PROMPT_SPEC.md`.  Baseline, Game, and Neutral are token-identical through
  the first assistant boundary; Game and Neutral contain an empty historical
  assistant turn.
- Discovery: the already-frozen 251-question half.
- Confirmation: the already-frozen 249-question half.

## Positions and components

The exhaustive screen covers all 64 mixers and all 64 MLPs at each of three
semantic positions:

1. `feedback_end`: `.` ending `Your previous answer was incorrect. Please
   provide a different answer.` in Game and `You lost track of the conversation.
   Please provide your answer again.` in Neutral.
2. `second_user_end`: the final token `):` of the repeated answer cue.
3. `decision`: the final assistant-prefix token `\n\n`, whose residual state
   supplies the answer-token logits.

This gives 384 position-component targets.  Position labels are resolved from
the rendered prompt and their decoded tokens are written to
`position_audit.json`; hard-coded token indices are not used.

## Causal intervention

For each discovery question, collect every component output at all three
positions in natural Game and natural Neutral.  Then, one target at a time,
replace the Game output with the same question's Neutral output and allow all
downstream computation to run normally.  Discovery therefore tests necessity
for the Game-specific transformation without fitting a direction or projecting
onto a preselected semantic axis.

Qwen's BF16 logits depend measurably on physical batch size.  Every patched
forward therefore contains exactly 15 intervention slots plus one unpatched
control row.  Partial chunks are padded.  The saved causal result is the
patched-minus-control delta from that same physical batch, recentered on the
single-question natural logits.  Two independent fixed-size calibration runs
matched exactly for all saved A-D logit vectors.

## Outcomes

Every target is described using paired question-level effects on:

- A-D entropy and centered spread;
- Baseline-winner advantage and switching away from the Baseline winner;
- total centered A-D movement;
- movement parallel and orthogonal to the question's Baseline evidence vector;
- the four Baseline-rank-aligned causal writes;
- an ordered rank-redistribution slope across ranks 1 through 4.

The rank write is not a fixed-letter average.  For each question, A-D are first
reordered by that question's Baseline logit ranks, and only then averaged.

## Discovery and held-out confirmation

Candidates are eligible through any of three sign-constrained families:
literal flattening, switching/winner-advantage, or ordered rank redistribution.
The eight largest eligible effects overall are selected with no family quota.
Thus the data may select eight components from one family and none from another.

On the untouched 249 questions, every selected component is patched in both
directions (Neutral into Game and Game into Neutral), as are the selected
family groups and their union.  Interpretation emphasizes effect size,
question-bootstrap confidence intervals, agreement of outcome measures, and
reciprocity; discovery rankings alone are not treated as confirmation.

## Outputs

The main layerwise figures show, separately for mixers and MLPs at each prompt
position:

- causal rank-1, rank-2, rank-3, and rank-4 writes with 95% intervals;
- entropy/spread and switching/winner-advantage effects;
- total, Baseline-parallel, and Baseline-orthogonal movement.

The run is resumable.  Compact analyses, figures, plans, prompt/position audits,
and a results report are retrieved locally.  The Vast instance is stopped, not
destroyed, after verification.
