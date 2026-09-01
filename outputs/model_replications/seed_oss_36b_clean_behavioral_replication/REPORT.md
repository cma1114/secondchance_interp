# Seed-OSS 36B clean behavioral replication

## Result

`ByteDance-Seed/Seed-OSS-36B-Instruct` reproduces the clean Second Chance behavioral signature on both frozen datasets. In both SimpleMC and TriviaMC, `incorrect` causes more semantic switching than the otherwise identical token-matched control `lost`; the gap survives a complete four-option letter remapping; and the logit change targets the semantic content of the model's first-presentation winner rather than its former displayed letter.

This is a **behavioral** replication. It establishes that Seed shows the same mapping-invariant selective reranking phenotype that motivated the Qwen mechanistic program. It does not establish that Seed implements that behavior with the same internal circuit.

## Executed design

- Model revision: `497f1dca95ebdec98e41d517b9f060ee753c902f`, BF16, Transformers 4.57.6.
- Native Seed chat template with `thinking_budget=0`; the rendered prompt audit confirms the closed thinking block.
- Empty first assistant turn: no old answer letter is visible.
- Conditions per dataset: Baseline; same-order Game and Neutral; fully remapped Game and Neutral.
- Game and Neutral differ exactly at `incorrect` versus `lost` in `Your answer was ... Choose the answer again.`
- All 500 SimpleMC questions and all 500 difficulty-filtered TriviaMC questions; no accuracy or outcome filtering.
- Five complete forwards per four-question cohort: 625 physical batched forwards and 2,500 question-condition outputs per dataset.
- Existing frozen balanced derangements and frozen discovery/confirmation splits.

All 5,000 condition-question logit vectors were finite. Exact duplicate-Baseline benchmark errors were `0.0` on both datasets.

## SimpleMC

| Endpoint | Game | Neutral | Paired Game − Neutral |
|---|---:|---:|---:|
| Same-order semantic switching | 31.0% | 16.8% | +14.2 pp [10.4, 18.2] |
| Remapped semantic switching | 63.6% | 55.0% | +8.6 pp [5.4, 12.0] |

On the frozen 249-question confirmation split, the remapped semantic switching gap was +9.2 points [4.8, 13.7]. In the full remapped sample, Game-minus-Neutral centered evidence by first-presentation rank was W1/W2/W3/W4 = −0.585/+0.217/+0.298/+0.070 logits. Thus Seed specifically lowers its old semantic winner in Game while increasing the alternatives.

The remapping control separates semantics from letters. Neutral-minus-Game suppression was +0.585 logits [0.461, 0.713] for the old winner content at its new letter, versus −0.143 [−0.258, −0.026] for the old literal letter now attached to different content. The semantic-minus-letter contrast was +0.728 [0.545, 0.914] logits.

Answer-only compliance ranged from 92.6% to 100% across conditions, so the prespecified primary analysis used the conditional A-D logit argmax on all questions. The unrestricted-output complete-case sensitivity gives the same result: +15.4 points [11.4, 19.4] same-order (n=448) and +9.2 [5.8, 12.6] remapped (n=468).

[Detailed SimpleMC report](simplemc/analysis/REPORT.md) · [summary](simplemc/analysis/summary.json) · [figure](../../../figures/model_replications/seed_oss_36b_simplemc_clean_behavioral_gate.png) · [compact results](simplemc/run/results.json)

## TriviaMC

| Endpoint | Game | Neutral | Paired Game − Neutral |
|---|---:|---:|---:|
| Same-order semantic switching | 16.0% | 8.4% | +7.6 pp [5.2, 10.2] |
| Remapped semantic switching | 30.6% | 25.2% | +5.4 pp [2.4, 8.4] |

On the frozen 250-question confirmation split, the remapped semantic switching gap was +7.2 points [3.2, 11.6]. In the full remapped sample, Game-minus-Neutral centered evidence by first-presentation rank was W1/W2/W3/W4 = −1.068/+0.497/+0.421/+0.150 logits. The selective old-winner decrease is therefore even larger than on SimpleMC.

Neutral-minus-Game suppression was +1.068 logits [0.911, 1.220] for old-winner content at its new letter and +0.181 [0.048, 0.318] for the old literal letter. The semantic-minus-letter contrast was +0.887 [0.642, 1.127] logits.

Answer-only compliance ranged from 98.8% to 100%. The unrestricted-output complete-case sensitivity again agrees: +7.7 points [5.1, 10.3] same-order (n=494) and +5.0 [2.0, 8.0] remapped (n=498).

[Detailed TriviaMC report](triviamc/analysis/REPORT.md) · [summary](triviamc/analysis/summary.json) · [figure](../../../figures/model_replications/seed_oss_36b_triviamc_clean_behavioral_gate.png) · [compact results](triviamc/run/results.json)

## Interpretation and next decision

The two datasets agree on the core behavioral facts:

1. The effect is not a same-letter response habit: it persists when every option changes letters.
2. The model does not merely become generically more variable after `incorrect`: relative to `lost`, its largest rank-specific change is a decrease in the old winner's semantic content.
3. The phenotype is not peculiar to one question collection: both the switching gap and semantic targeting replicate on the prespecified TriviaMC generalization set.

Accordingly, Seed passes the behavioral gate for a mechanistic cross-architecture replication. Any such work should begin with the smallest decisive subset of the existing Qwen causal suite and should treat circuit identity as an open question.

[Frozen plan](PLAN.md)

## Mechanistic follow-up

The first cross-architecture causal test is now complete. Across every one of
Seed's 64 grouped-query attention layers, blocking all four true matching
1P-option-line→2P-option-line reads rather than four cyclic wrong-line reads
raises held-out Game W1 evidence by +1.197 logits and lowers W3/W4 by
-0.527/-0.625. Natural held-out old-W1 choice is 36.5% in Game versus 45.8% in
Neutral; after matching blockade it is 49.0% versus 47.8%. The
matching-minus-cyclic change in the task gap is +7.2 points `[+1.6,+12.9]`,
and discovery independently gives +8.8 `[+2.0,+15.5]`.

Seed therefore causally replicates the core Game recollection mechanism:
matching semantic history selectively disfavors the old winner and supports
weaker candidates under `incorrect`. Natural Neutral does not show a stable
matching-specific rank profile, so the stronger task-shared Neutral route
claim does not cross-model replicate in this test. The all-500 TriviaMC causal
replication now independently gives a +1.277 `[+0.942,+1.618]` W1 task
interaction and a +10.0 `[+5.2,+14.8]`-point matching-minus-cyclic task-gap
change. Policy-state transfer is also complete: crossing only the seven-token
feedback suffix's outgoing K/V at all 64 Seed layers transfers essentially
100% of the paired natural Game/Neutral A--D score difference on both datasets.
Thus semantic recollection and the contextualized policy source both cross
architecture and dataset; a direct Seed policy-by-history factorial remains
untested.

[Integrated mechanistic report](../seed_oss_36b_mechanistic_replication/REPORT.md) ·
[SimpleMC matching report](../seed_oss_36b_mechanistic_replication/simplemc/matching_history/analysis/REPORT.md) ·
[summary](../seed_oss_36b_mechanistic_replication/simplemc/matching_history/analysis/summary.json) ·
[figure](../../../figures/model_replications/seed_oss_36b_simplemc_matching_history.png) ·
[frozen mechanistic plan](../seed_oss_36b_mechanistic_replication/PLAN.md)
