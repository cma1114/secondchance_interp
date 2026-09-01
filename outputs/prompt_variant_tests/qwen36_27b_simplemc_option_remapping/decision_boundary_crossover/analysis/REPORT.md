# First-decision boundary crossover

## Bottom line

The complete local boundary update does not carry the condition-specific semantic winner effect; the result supports reconstruction from distributed first-presentation history.

The manipulation gate passed. The crossed one-token boundary trajectory made the immediate boundary decision match the donor on +98.925 [+97.312, +100.000]% of held-out confirmation questions. The complete-history donor control matched on +98.925 [+97.312, +100.000]%.

## Primary semantic result

On held-out confirmation, the crossed boundary update's Neutral-minus-Game donor-semantic transfer was -0.009 [-0.035, +0.016] logits. The complete-history positive control was +0.466 [+0.268, +0.672] logits. The descriptive boundary fraction of that full condition interaction was -0.019424807015421066.

Discovery gave -0.019 [-0.045, +0.005] logits for the boundary update and +0.428 [+0.252, +0.603] for the complete history.

Held-out crossed effects by condition were +0.052 [+0.016, +0.089] in Game and +0.043 [+0.009, +0.078] in Neutral. Corresponding complete-history effects were +0.397 [+0.150, +0.651] and +0.863 [+0.579, +1.156].

## What the boundary update did carry

The crossed update left a small donor-**letter** trace at the final decision without transporting the donor's semantic answer. In confirmation, centered evidence at the donor's old literal letter increased by +0.087 [+0.065, +0.111] logits in Game and +0.077 [+0.054, +0.102] in Neutral. Evidence for the donor semantic answer at its current second-presentation letter changed by only +0.019 [-0.001, +0.040] and +0.013 [-0.007, +0.035]. Discovery showed the same separation: donor-old-letter effects +0.104 [+0.079, +0.130]/+0.100 [+0.072, +0.134], versus donor-semantic effects -0.011 [-0.031, +0.008]/-0.016 [-0.035, +0.002].

The donor-semantic descriptive endpoint is not perfectly letter-decoupled: the donor semantic answer differs from the donor's old literal letter on 76.9% of confirmation rows and 82.9% of discovery rows. The remaining 23.1%/17.1% retain letter overlap. The prespecified reciprocal Neutral-minus-Game semantic-margin contrast is differenced within each row and is unaffected by interpreting the auxiliary donor-semantic level as fully letter-pure.

Thus the first-decision boundary contains a portable record of **which output letter was about to be emitted**, but not a portable binding from that letter to the answer's semantic content. This agrees with the separate continuous A--D scrub: explicit answer-letter geometry exists there, but is not the mechanism that produces preferential semantic W1 avoidance.

## Validation

- Frozen/exact questions: discovery 105/115; confirmation 93/107.
- Full-donor suffix maximum A-D error: discovery 0; confirmation 0.
- Full-donor boundary maximum A-D error: discovery 0; confirmation 0.
- Identity split-path versus unsplit natural answer changes: discovery 20; confirmation 16.
- Identity boundary versus inclusive-prefix answer changes: discovery 12; confirmation 4.
- All analyzed outputs finite: discovery True; confirmation True.

The split cached execution is not numerically identical to a single unsplit forward in this recurrent architecture, and it changed some near-boundary natural choices. All causal estimates therefore use the same split identity path as their baseline. The complete-donor positive control reproduces that split donor path exactly; no causal contrast mixes split and unsplit logits.

## What was crossed

For each question, two natural first presentations selected different semantic answers X and Y. The second presentation and feedback were identical. The intervention retained X's accumulated state through the token immediately before the empty first-answer position, then replayed Y's complete 64-block boundary trajectory for that one token (and reciprocally Y-history/X-update). This makes the model itself write donor-driven ordinary-attention K/V and GLA updates into recipient history. The final block output was also clamped to the donor so the immediate A--D decision manipulation was exact; that final clamp does not alter the K/V or recurrent state already written inside the block. It is therefore a direct conflict between presentation history and the persistent state written at the decision boundary, not another one-dimensional decoder ablation.
