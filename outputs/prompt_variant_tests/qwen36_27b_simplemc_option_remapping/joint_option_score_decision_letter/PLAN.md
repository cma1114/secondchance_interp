# Joint option-score and first-decision-letter factorial

## Question

Are two decoded signals redundant routes for identifying the first-presentation winner?

1. The centered one-dimensional candidate-value score at all four first-presentation option-closing newlines.
2. Centered A-D letter identity at the empty first-decision position.

If they are redundant, removing both should recover W1 in Game more strongly than either lesion alone, especially on conflict trials where W1 differs from the fresh remapped winner W2.

## Frozen design

- Dataset and prompts: the canonical 500-question action-matched option-remapping paradigm.
- Frozen splits: 251-question discovery and 249-question confirmation.
- Conditions: Game and Neutral.
- Modes in one matched runner: natural, zero-delta identity K/V, score-only, decision-letter-only, and joint score-plus-letter.
- Score lesion: remove the validated centered affine score coordinate from all four first-presentation option-newline carriers at every ordinary-attention block (4, 8, ..., 64), using projected-minus-unprojected live K/V deltas.
- Letter lesion: remove the three-dimensional centered A-D JLens decoder subspace from the exact empty first-decision residual, convert that residual change through each ordinary-attention block's own norm and K/V projections, and add only the K/V delta at the exact first-decision position.
- Joint lesion: compose the two edits at their distinct source positions.

## Primary endpoint

On conflict trials, intervention-minus-identity change in W1 choice for Game minus the corresponding change in Neutral.

Continuous W1-minus-W2 margin, W1-centered evidence, probabilities, entropy, spread, no-conflict trials, and the factorial interaction are prespecified secondary outcomes.

## Validation requirements

- Natural and zero-delta identity paths must match exactly.
- Natural and score-only outputs must reproduce the prior validated same-host run exactly.
- First-decision choices must reproduce the frozen Baseline closely, with any older cross-host drift reported.
- Post-lesion option score and post-lesion centered A-D norm must be near zero.
- The score-only and letter-only doses in the joint mode must exactly match their corresponding single-lesion doses.
- All 500 questions must complete with finite outputs.

## Completion

The exact 11-forward benchmark passed. The full 500-question run completed in 85.1 minutes. Compact outputs were retrieved hash-identically after local disk cleanup, and the retained Vast host was stopped.

Result: the redundancy prediction failed. See [analysis/REPORT.md](analysis/REPORT.md), [analysis/summary.json](analysis/summary.json), and the canonical figure at `figures/qwen36_joint_option_score_decision_letter.png`.

