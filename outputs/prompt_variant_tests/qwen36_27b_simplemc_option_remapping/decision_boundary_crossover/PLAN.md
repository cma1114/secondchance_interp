# First-decision boundary crossover

## Question

When the model later treats one semantic option as its previous winner, does it read a persistent record written specifically at the empty first-answer position, or reconstruct the winner from the distributed first-presentation history?

## Frozen cohort

Use the 222 previously frozen six-permutation pairs in `cohort_plan.json` (115 discovery, 107 confirmation; original winner displayed at A/B/C/D on 77/44/64/37 questions). Each pair contains the same question and semantic options. History X selects semantic answer X; a distractor permutation keeps X at the same displayed position but makes a different semantic answer Y win. Feedback and the canonical second presentation are identical.

## Causal crossover

For every exact-regime eligible pair, save the complete 64-block layer-input trajectory at the empty first-answer boundary. Then evaluate:

1. **Identity:** recipient pre-boundary state plus its own boundary trajectory.
2. **Crossed boundary:** recipient pre-boundary state plus the other history's boundary trajectory.
3. **Full donor:** donor pre-boundary state plus donor boundary trajectory.

Replaying the trajectory makes the live model itself create the ordinary-attention K/V entry and GLA convolution/recurrent update for that boundary token. This avoids pretending that an accumulated GLA state has an independently addable “decision delta.” Both reciprocal chimeras X-history/Y-update and Y-history/X-update are run in Game and Neutral.

## Prespecified endpoints

- Manipulation gate: the crossed immediate boundary decision must match the donor on at least 80% of exact questions; the full-donor boundary and suffix paths must reproduce donor identity within `1e-4` A–D logits.
- Primary endpoint: reciprocal donor-winner-minus-recipient-winner semantic margin transfer, Neutral minus Game, separately in discovery and held-out confirmation.
- Secondary endpoints: Game and Neutral semantic transfer separately; donor semantic choice; donor semantic versus old-literal-letter evidence; fraction of the complete-history condition interaction carried by the boundary update.

Predictions:

- **Stored-decision account:** the crossed boundary update transfers the donor semantic winner and captures a substantial, replicating fraction of the full-history Game/Neutral interaction.
- **Reconstruction account:** the crossed boundary update successfully changes the immediate decision but has little downstream semantic effect, while the full-history donor control has a substantial effect.
- If the manipulation gate fails or split effects disagree, the result is unresolved rather than evidence for reconstruction.

## Exact execution and controls

- Established action-matched empty-history prompt, raw Qwen ChatML, reasoning off, batch-four SDPA.
- Exact first decisions are rechecked under the experimental batch; ineligible or token-unaligned frozen rows are recorded before intervention outcomes are examined.
- Nine model calls for each exact question: inclusive prefix capture, unsplit natural, pre-boundary cached prefix, three one-token boundary steps, and three suffix continuations.
- Identity split-path behavior is compared with the unsplit natural forward.
- Discovery and confirmation remain separate; no layer selection occurs.
- One complete representative question must be benchmarked before full launch. Runtime and cost are forecast from all nine calls, not only the crossed suffix.
- Combined batch spending cap: $15.

## Outputs

- Compact resumable `results.npz` and audit metadata for each split.
- `analysis/summary.json`, `analysis/effects.csv`, and `analysis/REPORT.md`.
- One canonical PNG in the top-level `figures/` directory.

## Result

Complete. The exact cohort retained 105/115 discovery and 93/107 confirmation
questions. Crossing the complete boundary trajectory changed the immediate
decision to the donor on 97.1% and 98.9% of questions, so the manipulation
worked. It did **not** transfer the donor semantic winner differentially between
Game and Neutral: the Neutral-minus-Game semantic-margin transfer was -0.019
[-0.045, +0.005] logits in discovery and -0.009 [-0.035, +0.016] in
confirmation. Complete donor history produced +0.428 [+0.252, +0.603] and
+0.466 [+0.268, +0.672] logits.

The crossed boundary did transport a small literal donor-letter trace, but not
the semantic answer bound to that letter. This supports reconstruction from
distributed first-presentation history rather than a portable semantic-winner
record written at the empty first-answer boundary. See `analysis/REPORT.md`.
