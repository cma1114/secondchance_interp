# Causal role of the post-list answer cue

## Method

The exact trailing cue-space token remains present and its own residual is kept natural. At every ordinary-attention layer, later tokens either receive paired donor K/V or cannot read the source. At every GLA layer, later recurrent outputs receive the paired donor write or the source write is removed. The structural control applies the identical removal to the immediately preceding `):` token.

## Held-out results

**Game:**
- `cue_swapped`: centered-logit displacement 0.137 `[0.126,0.149]`; winner changed 9.2% [6.0,12.9].
- `cue_ablated`: centered-logit displacement 0.124 `[0.116,0.132]`; winner changed 10.4% [6.8,14.5].
- `colon_ablated`: centered-logit displacement 0.080 `[0.075,0.085]`; winner changed 4.8% [2.4,7.6].
- Swap transfer toward donor cue vector: 0.073 `[0.056,0.090]`.

**Neutral:**
- `cue_swapped`: centered-logit displacement 0.138 `[0.127,0.149]`; winner changed 6.8% [4.0,10.0].
- `cue_ablated`: centered-logit displacement 0.123 `[0.114,0.133]`; winner changed 8.0% [4.8,11.6].
- `colon_ablated`: centered-logit displacement 0.091 `[0.084,0.098]`; winner changed 4.8% [2.4,7.6].
- Swap transfer toward donor cue vector: 0.080 `[0.065,0.094]`.

## Behavioral switching

**Conflict:**
- Game: natural 82.4% [75.7,88.2], cue_swapped 75.0% [67.6,82.4], cue_ablated 81.6% [75.0,87.5], colon_ablated 78.7% [71.3,85.3].
- Neutral: natural 60.3% [52.2,68.4], cue_swapped 61.0% [52.9,69.1], cue_ablated 61.0% [52.9,69.1], colon_ablated 61.0% [52.9,69.1].

**No conflict:**
- Game: natural 38.9% [30.1,47.8], cue_swapped 35.4% [26.5,44.2], cue_ablated 38.9% [30.1,47.8], colon_ablated 38.1% [29.2,46.9].
- Neutral: natural 26.5% [18.6,34.5], cue_swapped 29.2% [21.2,38.1], cue_ablated 27.4% [19.5,35.4], colon_ablated 25.7% [17.7,33.6].

## Paired causal contrasts

- Game, cue ablation minus colon control: logit displacement +0.044 `[+0.035,+0.053]`; winner-change difference +5.6 points `[+1.6,+10.0]`.
- Neutral, cue ablation minus colon control: logit displacement +0.033 `[+0.023,+0.043]`; winner-change difference +3.2 points `[+0.0,+6.4]`.
- All, cue swap change in preferential Game switching: -7.2 points `[-11.2,-3.2]`.
- Conflict, cue swap change in preferential Game switching: -8.1 points `[-14.7,-2.2]`.
- No conflict, cue swap change in preferential Game switching: -6.2 points `[-11.5,-1.8]`.

## Bottom line

Complete downstream ablation of the cue is now a live ordinary-attention-plus-GLA intervention. It changes individual final rankings above the structural-colon control, so the cue is causally used. However, it does not materially reduce the main Game-minus-Neutral switching difference: the all-question change is +1.6 points `[-3.6,+6.8]` on discovery and -1.2 points `[-5.2,+3.2]` on confirmation. Thus the cue carries and contributes task-specific policy information, but the main behavioral difference does not require this cue route.

## Validation

- Same-batch natural maximum A-D logit error: `0`.
- Cue-state invariance under cue swap/ablation: `0`.
- Every ordinary-attention and GLA layer was covered.
- Discovery results are retained in `summary.json` for replication assessment.

## Artifacts

- Canonical figure: `figures/qwen36_cue_memory_causality.png`
- Compact statistics: `summary.json`
