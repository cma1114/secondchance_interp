# Fixed-A final-query selected-option attention-edge ablation

**Status: complete. The prespecified direct-final-query prediction was not confirmed.**

## Question

Does Game's preferential avoidance of the semantic first answer require a
direct ordinary-attention read from the first selected-option line at the final
answer decision?

## Frozen design

Use the existing fixed-A X/Y crossover cohort: the first literal answer is `A`
in both histories, but `A` denotes semantic answer X in one history and Y in
the other. The second presentation is identical. Evaluation and Matched Neutral
differ only by `incorrect` versus `lost`.

At the final decision query only, set attention-mask entries to the first
presentation's selected `A` option line to negative infinity in every head of:

1. ordinary-attention block 44;
2. ordinary-attention blocks 36, 40, 44, and 48;
3. every ordinary-attention block from 4 through 48.

Repeat each lesion on an unselected option line chosen to have the nearest
token count to the selected line. Every earlier query, all GLA state, all MLPs,
and all unlisted attention edges remain unchanged. Preserve the established
batch-of-four SDPA regime. Discovery and confirmation remain separate.

## Primary behavioral endpoint

For each history, W1 is the semantic content selected on the first
presentation. The primary endpoint is the lesion-induced increase in Game's
W1-choice rate. The sharper condition-specific endpoint is the reduction of
the natural preferential-avoidance gap:

`Neutral W1 choice - Game W1 choice`.

Also report the W1-versus-counterfactual semantic logit margin, Neutral's W1
choice, entropy, and matched-source controls. A clean active-suppression result
requires the selected-line lesion to make Game choose W1 more and to reduce the
preferential-avoidance gap, with substantially smaller effects from the
unselected-line controls.

## Follow-up within the same requested batch

If the fixed-A result confirms the causal route, apply the confirmed edge set
to the full non-remapped action-matched SimpleMC cohort. There, the decisive
behavioral prediction is reduced Evaluation switching and increased Neutral
switching. Record the existing four-mapping W1 semantic projection at the final
decision so that loss of semantic access can be related directly to changed
switching behavior.

Because the fixed-A result did not confirm the route, the non-remapped
follow-up was not launched. See
[`analysis/REPORT.md`](analysis/REPORT.md) for the complete result.
