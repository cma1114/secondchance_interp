# OLMo 2 32B SimpleMC behavioral gate

## Bottom line

**OLMo 2 32B does not reproduce Qwen3.6-27B's preferential Game switching in the clean paradigm, so the planned mechanistic replication is not warranted on this model.** The strongest mapping-invariant result points in the opposite direction: relative to `lost`, `incorrect` makes OLMo preserve its first-presentation semantic winner more strongly.

This is a completed 500-question behavioral result, not an activation or causal-intervention claim about OLMo's internal mechanism.

## Exact design and validity

- Model: `allenai/OLMo-2-0325-32B-Instruct`, revision `b96024342a77a69aa0dda815c3454a671f477463`.
- Dataset: all 500 canonical SimpleMC questions; the frozen 251/249 discovery/confirmation partition is retained for robustness checks.
- OLMo's native chat template, with an empty first assistant turn so no old answer letter is visible.
- Game: `Your answer was incorrect. Choose the answer again.`
- Neutral: `Your answer was lost. Choose the answer again.`
- Within each mapping, the rendered prompts differ only at `incorrect`/`lost`.
- The mapping control moves every second-presentation option content to a new A-D letter using the frozen balanced derangement.
- Unrestricted next-token choice is primary. Aggregated A-D logits provide continuous rank and entropy outcomes.

The run produced 2,500 condition-question outputs in 625 physical batched forwards. Every unrestricted output was A-D, all logits were finite, and the benchmark's duplicate Baseline reproduced with exactly 0.0 A-D logit error. One aggregate-logit tie occurred in remapped Neutral and does not affect the unrestricted primary outcome. W1 in the rank analysis is explicitly the unrestricted first-pass choice; the other candidates are then ordered by their Baseline A-D evidence.

## Behavioral result

With the same option order, Game switched away from its first-presentation answer on **18.6%** of questions and Neutral on **18.4%**. The paired difference was **+0.2 percentage points [−3.4, +4.0]**. Discovery was −2.8 points [−8.0,+2.4], while confirmation was +3.2 [−2.0,+8.8]: both include zero and point in opposite directions.

After every option moved letters, Game switched semantic answers on **65.6%** and Neutral on **72.0%**. The paired Game-minus-Neutral gap was **−6.4 points [−9.6, −3.2]**. This inverse direction appears in both frozen halves: discovery **−7.6 [−11.6,−3.6]** and confirmation **−5.2 [−10.0,0.0]**.

The effect follows semantic content, not the old displayed character. Avoidance of the old literal letter was 52.0% in Game and 50.4% in Neutral, a nonsignificant **+1.6-point [−2.2,+5.4]** difference. That cannot explain the −6.4-point semantic switch difference.

## What changes in the answer scores

After remapping, Game-minus-Neutral centered evidence by OLMo's actual first-pass choice rank was:

| Baseline rank | Game − Neutral centered logit |
|---|---:|
| W1 | **+0.206 [+0.166,+0.245]** |
| W2 | −0.004 [−0.041,+0.033] |
| W3 | **−0.068 [−0.105,−0.032]** |
| W4 | **−0.133 [−0.169,−0.098]** |

The same-order profile is qualitatively identical: W1/W2/W3/W4 = **+0.248/−0.028/−0.090/−0.130 logits**. This is the reverse of Qwen's old-winner suppression. In OLMo, `incorrect` gives W1 more relative support than `lost`, chiefly at the expense of lower-ranked old candidates.

The remapping control localizes the comparison behaviorally. Neutral-minus-Game centered evidence at W1's **new** letter was **−0.206 [−0.244,−0.165] logits**; at W1's former letter, now attached to different content, it was **−0.108 [−0.135,−0.079]**. The semantic-minus-literal contrast was **−0.098 [−0.150,−0.045]**. Negative values mean greater evidence in Game. The larger Game advantage tracks W1's semantic content.

## Entropy

Mean A-D entropy was **1.513 bits** at Baseline. It was **1.459 Game versus 1.620 Neutral** with the same order and **1.471 versus 1.615** after remapping. Thus Game was 0.160 bits less entropic than Neutral without remapping and 0.143 bits less entropic with remapping; it was also slightly less entropic than Baseline. OLMo does not reproduce Qwen's combination of higher Game entropy and semantic W1 suppression.

## Decision

The prespecified follow-up gate failed all three confirmation requirements: no positive remapped semantic switching gap, no Game suppression of semantic W1 relative to Neutral, and no semantic suppression exceeding literal-letter suppression. These high-quality data identify OLMo as a **behavioral non-replication with an inverse semantic policy effect**.

I recommend **not** spending the next batch on an OLMo replication of the Qwen mechanism. A comparative experiment asking why `incorrect` stabilizes OLMo's old winner would be a different, potentially interesting project, but it would not test whether the established recollection-and-suppression mechanism generalizes.

## Artifacts

- [Machine-readable summary](summary.json)
- [Frozen plan](../PLAN.md)
- [Compact run results](../run/results.json)
- [Canonical figure](../../../../figures/model_replications/olmo2_32b_simplemc_behavioral_gate.png)
