# Canonical remapped W1 receiver-path search

## Scientific question

The first-presentation option line containing W1 has a causal conventional-
attention K/V effect on the eventual semantic answer, but the final decision
query does not read that line directly. Which earlier downstream query first
reads this information, and does that read mediate Game's preferential
avoidance of W1?

## Frozen paradigm

- Qwen3.6-27B, pinned model revision and explicit raw Qwen ChatML.
- Current action-matched remapped SimpleMC prompts.
- Game: `Your answer was incorrect. Choose the answer again.`
- Neutral: `Your answer was lost. Choose the answer again.`
- Exact historical batch-of-four SDPA cohorts and trusted natural outputs.
- W1 is the semantic answer selected by the original Baseline.
- W2 is the answer selected by a fresh Baseline under the remapped second
  presentation.
- Report conflict (W1 != W2) and no-conflict (W1 == W2) trials separately.
- Preserve the existing frozen 251-question discovery and 249-question
  confirmation split. Candidate selection uses discovery only.

## Stage 1: receiver screen

In one unmodified natural forward per condition, recover the exact ordinary-
attention weights at all 16 conventional-attention blocks. For every query
position causally after the W1 option line, compute:

1. attention mass assigned to the complete W1 option line;
2. norm of the W1-line contribution to the gated, output-projected attention
   write;
3. the corresponding matched unselected-option-line quantities; and
4. norm of the W1-line write minus the matched-line write.

Store every token position. For cross-question summaries, use structurally
aligned receiver roles: exact feedback-token slots, exact historical- and
final-assistant-prefix slots, choice-cue slots, and broad variable-length
question/option regions. Within a multi-token role, aggregate by the per-
question maximum before averaging so long regions are not favored merely by
token count.

The terminal final-decision query remains visible in the observational screen
but is ineligible for causal candidate selection, because the immediately
preceding canonical experiment already established that its direct edge to the
first W1 option line is null. This search is specifically for an earlier relay.

Candidate selection is frozen from discovery data as the union of:

- the strongest individual block/receiver-role source-specific writes;
- the strongest Game-versus-Neutral changes in that write; and
- the strongest receiver roles after maximizing across ordinary-attention
  blocks.

The screen nominates candidates; it is not itself causal evidence.

## Stage 2: exact causal receiver-edge validation

For each frozen candidate, block only the specified query-to-W1-line attention
edge. Leave all other queries, sources, layers, residuals, GLA states, and
tokens untouched. Test:

- the nominated individual block/receiver edge;
- the same receiver role across all ordinary-attention blocks 4--48, to allow
  redundant reads across depth; and
- a token-count-matched unselected option line for every intervention.

Run all 500 questions in their original cohorts, analyze discovery and held-out
confirmation separately, and report within Game and Neutral:

- W1 selection and switching away from W1;
- W1-W2 logit margin on conflict trials;
- W1 centered A-D advantage;
- W2 selection; and
- A-D entropy.

The primary causal signature is an increase in Game W1 choice and W1-W2 margin
when the W1 read is blocked, larger than the matched-line effect, thereby
reducing the natural Game-specific W1-avoidance effect. A receiver is considered
localized only if its direction replicates on held-out confirmation. If no
individual edge is sufficient but the all-block receiver intervention is,
the result supports redundant depth-wise reads at that position.

## Next step only if a receiver validates

Capture the receiver's post-read residual difference between natural and
edge-blocked runs. Restore that difference in the edge-blocked run. Rescue of
the final W1 effect would establish mediation. Then separately prevent that
receiver state from entering later conventional-attention K/V, GLA recurrent
state, or its local MLP to identify the next carrier.

## Completion

Completed on all 500 canonical remapped questions. The discovery-only screen
froze 12 candidates; exact natural logits reproduced with zero error in causal
validation. The repeated second-presentation W1 option line across ordinary-
attention blocks 4--48 validated on the held-out split. The individually
tested repeated-W1 receiver edges were only blocks 52, 60, and 64, selected
automatically from the discovery screen by source-specific projected-write
magnitude/onset rather than raw attention mass. They were null; blocks 4--48
were not decomposed individually. See [the causal report](validation/analysis/REPORT.md).
