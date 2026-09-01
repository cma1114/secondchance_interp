# Fixed-A selected-option K/V layer localization

## Status

Completed. Ordinary-attention block 44 is the only large, replicated
individual semantic reader. Blocks 4--48 jointly produce the complete
condition difference, with early blocks acting nonlinearly; blocks 52--64 are
dispensable.

- [Canonical report](analysis/REPORT.md)
- [Machine-readable summary](analysis/summary.json)
- [Presentation figure](../../../../figures/qwen36_fixed_a_kv_layer_localization.png)

## Question

Which ordinary-attention layers read the semantic content stored at the
first-presentation option line selected as literal `A`, and which layers account
for its substantially weaker reinstatement under `incorrect` than under `lost`?

## Frozen design

Use the same 64-question discovery and 73-question confirmation fixed-A X/Y
crossover cohort as the source-localization experiment. The visible recipient
prompt, all GLA state, and all K/V entries except the selected option line remain
fixed.

For the selected option line, transplant donor K/V in:

1. each of the 16 ordinary-attention blocks individually: 4, 8, ..., 64;
2. four non-overlapping bands: 4--16, 20--32, 36--48, and 52--64;
3. four leave-one-band-out combinations; and
4. all 16 ordinary-attention blocks as the positive control.

Identity reinsertion is the negative control. Individual blocks test precise
localization; four-block bands test joint sufficiency; leave-one-band-out cells
test necessity in the presence of the other twelve blocks. Discovery and
confirmation use the historical batch-of-four SDPA cohort and are analyzed
separately.

## Primary endpoint

As before, symmetrically transplant X into Y and Y into X. Negative semantic
margin transfer means movement toward the donor history's previous answer.
Report Game, Neutral, and Game-minus-Neutral. Also report donor-answer selection,
A--D entropy, answer changes, and natural/cached identity validation.

## Interpretation rule

A layer or band is considered localized only if its sign and approximate
magnitude replicate on confirmation. Individual-layer nulls do not defeat a
joint-band result; a band is strongest evidence when it is both sufficient and
its omission measurably reduces the all-layer effect.
