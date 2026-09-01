# Rank-opposed transformation analysis

## Target

The paired JLens figure shows an ordered Game-minus-Baseline change: the generated Baseline winner is suppressed most, followed by the Baseline runner-up and ranks 3--4, allowing negative suppression (boosting). The immediate target is therefore a rank-opposed four-option transformation, not entropy or switching alone.

## Observational test

For every question and layer:

1. Center the four JLens answer-letter scores within condition, matching the paired fixed-rank figure that motivated this analysis.
2. Fix option ranks from the self-hosted generated Baseline answer and final Baseline A--D logits. Do not substitute the historical OpenRouter Baseline answer stored in the frozen trial manifest.
3. Subtract the same-question Baseline vector from Game or Neutral.
4. Project the aligned change onto `[-1.5, -0.5, +0.5, +1.5]`.

Report the equal-answer-letter mean slope, the fraction of trials with a positive slope, the fraction whose four changes are strictly monotone, and the fraction of four-option change energy captured by the rank axis. Separately fit the condition change against the full same-layer Baseline JLens vector. This distinguishes a robust question-specific negative-feedback operation from an ordered aggregate that is only weakly coherent within trials.

## Causal reuse

Reanalyze the existing exhaustive Neutral-into-Game component sweep and reciprocal held-out confirmation patches. For each mixer and MLP patch, measure how much of the final Game-minus-Neutral rank-opposition gap it removes or induces. Use the actually generated Baseline answer for rank 1.

Discovery results identify hypotheses only. Components not already included in the held-out reciprocal split must be confirmed on the untouched 249-question split before being treated as causal mediators.

## Mechanistic signatures

- Generic negative feedback: a component's Game-like write is graded against the full incoming or Baseline evidence vector.
- Thresholded winner suppression: the write concentrates on rank 1 and depends nonlinearly on its incoming margin.
- Runner-up boosting: rank 2 receives a selective positive write.
- Generic noise: weak mean rank-opposition but increased variance or orthogonal energy.

The JLens change point near readout 48 motivates attention to block 48 (zero-indexed layer 47), but the causal sweep is analyzed across all layers.

## Current-leader test

For each held-out component intervention, identify the leading option immediately before the component in two independent ways:

- the JLens A--D answer-letter scores at the preceding post-block residual;
- direct unembedding of the exact pre-mixer or pre-MLP residual captured in the sublayer run.

Define the component's Game-like causal write as natural Game minus Neutral-into-Game patched output, or Game-into-Neutral patched output minus natural Neutral. Test whether this write lowers the current leader, whether the suppression grows with the incoming leader margin, and whether the result is reciprocal across patch directions. This is the direct test of thresholded or graded leading-candidate suppression; it is not inferred from final entropy or switching alone.
