# Complete-residual JLens at both feedback periods

## Question

In the action-matched paradigm, what does the model's complete residual stream
represent at the two punctuation tokens in the feedback turn?

1. **Evaluation-closing period:** the period after `Your answer was
   incorrect.` or `Your answer was lost.`
2. **Action-closing period:** the period after the shared sentence `Choose the
   answer again.`

This comparison addresses why the earlier causal work emphasized the first
period even though earlier exploratory JLens plots used the final period.

## Method

The analysis uses all 500 remapped SimpleMC questions and the exact action-matched
prompts from the behavioral factorial. At each period, it captures the complete
post-block residual after every one of Qwen3.6-27B's 64 blocks. Each question's
state is transported through the published Qwen3.6-27B JLens, final-RMS-normalized,
and then averaged. The interactive explorer shows:

- Evaluation minus Matched Neutral;
- Evaluation alone;
- Matched Neutral alone;
- every residual readout from 1 through 64;
- unrestricted top and bottom vocabulary tokens, with non-English tokens given
  literal English glosses;
- fixed Baseline-rank A-D pseudotokens, correctly remapped into the second
  presentation's letter order.

[Open the interactive explorer](action_matched_period_jlens_explorer.html).

## Main result: the two periods carry different readable stages

The two positions do not show the same vocabulary content.

### Evaluation-closing period

The Evaluation-minus-Neutral contrast is mostly incoherent through the early
and middle layers. Around readout 42 it begins to surface error/mistake and
misleading/trap content. At readout 43 the contrast becomes unmistakably about
the evaluation itself: 8 of the top 10 ordinary tokens are glossed as
`error`, `wrong`, or `incorrect`. This remains strongly incorrectness-focused
through approximately readout 53. Later readouts shift toward being told,
feedback, admission/acceptance, and reflection, while the Neutral-pointing side
is dominated by repeat/reconstruction/reproduction concepts.

In plain language: at the first period, the readable distinction is primarily
**“the previous answer has been evaluated as wrong” versus “repeat/reconstruct
what was lost.”**

### Shared action-closing period

The final period shows a clearer transformation from evaluation into action.
Correction/revision and wrongness tokens become coherent around readouts 39–43.
Around readout 49, the highest-scoring tokens change to `exclude`, `reject`, and
related concepts. From readouts 54 through 64, exclusion/elimination dominates
the Evaluation-pointing top tokens. At readout 56, for example, the leading
English-glossed concepts are `exclude/eliminate`, `exclusion`, `exclude`, and
`excludes`; at readout 64 they remain `eliminate`/`exclude` variants.

In plain language: by the final shared period, the model's readable state has
advanced from **“the previous answer was wrong”** to **“exclude/eliminate a
candidate.”**

## Interpretation

This gives a much cleaner answer to the position question than the individual
GLA-write lenses did:

- The **evaluation-closing period** is causally important because its GLA update
  installs the condition-specific state; the complete residual there becomes
  explicitly readable as incorrectness/feedback in the early 40s.
- The **action-closing period** is more legible as the model's downstream
  revision policy. After the shared `Choose the answer again.` sentence, the
  condition contrast develops from correction into a stable
  exclusion/elimination representation.

Thus the final period was not selected because it was known to be the unique
causal source. It was selected in the old exploratory visualization simply as
the end of the entire feedback clause. The causal transplantation and source
deletion later localized the source to the earlier evaluation period. The new
paired JLens analysis shows that these facts are complementary: the earlier
period carries the evaluation, while the later period expresses the
instruction-conditioned action more explicitly.

This does not by itself identify which semantic answer is excluded. It is a
condition-average vocabulary readout. The remapping behavior and causal source
trace provide the answer-specific evidence; this analysis supplies the missing
readable cognitive-stage account.

## Numerical-reproducibility note

The prompt hashes match the trusted uninstrumented factorial exactly. Requesting
all intermediate states changes low-order bfloat16/SDPA numerics: the
instrumented A-D argmax agrees with the trusted uninstrumented run on 94.4% of
Evaluation questions and 96.6% of Matched-Neutral questions. A separate
no-hook `output_hidden_states=True` diagnostic reproduced the same changed
logits on its four-question cohort, while preloading versus not preloading the
JLens matrices produced bit-identical instrumented logits. The difference is
therefore inherent to requesting intermediate states, not to the JLens
transport or prompt construction. The vocabulary trajectories should be read
as a reproducible description of the instrumented model computation, with that
limitation kept explicit.

## Files

- Interactive explorer: [action_matched_period_jlens_explorer.html](action_matched_period_jlens_explorer.html)
- Compact displayed readouts: [top_tokens_with_baseline_ranks.json](../run/top_tokens_with_baseline_ranks.json)
- Prompt and token-position audit: [position_audit.json](../run/position_audit.json)
- Run metadata: [run_metadata.json](../run/run_metadata.json)
- English token glossary: [token_english_glosses.json](token_english_glosses.json)
