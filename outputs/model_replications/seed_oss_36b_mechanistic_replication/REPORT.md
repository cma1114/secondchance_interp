# Seed-OSS 36B mechanistic replication

## Result

`ByteDance-Seed/Seed-OSS-36B-Instruct` reproduces the core causal Second
Chance mechanism on both frozen clean-paradigm datasets despite replacing
Qwen's mixed ordinary-attention/GLA stack with 64 standard grouped-query
self-attention layers.

The replicated account has two experimentally distinct parts:

1. **Candidate recollection is causally used.** Preventing each repeated option
   line from reading its semantically matching first-presentation option line,
   while applying the same all-layer structural lesion to a cyclic wrong line
   as control, removes preferential Game avoidance of the old winner.
2. **The feedback suffix causally supplies the task policy.** Exchanging only
   the outgoing ordinary-attention K/V state emitted by the seven tokens from
   `incorrect/lost` through `Choose the answer again.` transfers essentially
   the complete natural Game/Neutral answer-score difference in both
   directions and on both datasets.

Together these results show that Seed's clean behavioral effect is not merely
a surface replication. The one-word feedback contrast creates a
contextualized policy state in the feedback suffix, and Seed uses semantically
matching first-presentation history differently under that policy.

## Matching-history replication

### SimpleMC

On the frozen 249-question confirmation split, matching blockade minus cyclic
control changed Game W1/W2/W3/W4 centered evidence by
`+1.197/-0.045/-0.527/-0.625` logits. The W1 Game-minus-Neutral interaction was
`+1.078` `[+0.702,+1.700]`. Natural old-W1 choice was 36.5% in Game versus
45.8% in Neutral; matching blockade changed this to 49.0% versus 47.8%. The
primary matching-minus-cyclic change in the task gap was +7.2
`[+1.6,+12.9]` percentage points, with discovery independently +8.8
`[+2.0,+15.5]`.

### TriviaMC

On the frozen 250-question confirmation split, the corresponding Game
W1/W2/W3/W4 changes were `+1.607/-0.327/-0.603/-0.677`. The W1 task
interaction was `+1.277` `[+0.942,+1.618]`. Natural old-W1 choice was 68.4%
in Game versus 75.6% in Neutral; matching blockade made both 76.8%, while the
cyclic control retained a 68.0% versus 78.0% gap. The primary
matching-minus-cyclic task-gap change was +10.0 `[+5.2,+14.8]` points.

The stable cross-dataset and cross-architecture claim is therefore that Game's
preferential old-W1 avoidance depends on matching semantic 1P-to-2P access.
Natural Neutral does not show a stable matching-specific W1 profile in Seed,
so the stronger Qwen claim that this exact route consistently supports W1 in
Neutral does not cross-model replicate.

## Complete feedback-suffix crossover

For each question, donor and recipient prompts were identical except for
`incorrect` versus `lost`. At every Seed layer, the intervention replaced the
seven feedback-suffix tokens' ordinary-attention keys and values as seen by
causally later queries. It did not replace the source tokens' residual outputs.
Real same-task transfer between distinct duplicated rows was bit-exact, and
all natural controls exactly reproduced the trusted behavioral artifacts.

On the frozen confirmation splits, donor-policy transfer was:

| Dataset | Into Game from Neutral | Into Neutral from Game |
|---|---:|---:|
| SimpleMC | 1.0002 `[1.0000,1.0007]` | 1.0002 `[1.0000,1.0007]` |
| TriviaMC | 0.9999 `[0.9996,1.0000]` | 0.9999 `[0.9996,1.0000]` |

A value of one means the swap-induced centered A--D logit vector reaches the
same-question natural donor-minus-recipient vector. The rank and choice
readouts reverse accordingly. On SimpleMC confirmation, Neutral suffix state
installed in Game raises W1 by +0.603 centered logits and raises old-W1 choice
from 36.5% to 45.8%; Game suffix state installed in Neutral lowers W1 by
-0.603 and lowers old-W1 choice from 45.8% to 36.1%. On TriviaMC, the
corresponding W1 changes are +1.064 and -1.064, and old-W1 choice changes
68.4%→75.6% and 75.6%→68.4%.

This is causal sufficiency of the **complete contextualized suffix's outgoing
K/V state** for essentially the entire measured task-conditioned answer-score
difference in Seed. It does not localize the policy to `incorrect/lost`, the
period, the following identical instruction, one layer, or one downstream
receiver token.

## Direct policy × recollection interaction

The factorial combined the suffix crossover and matching-history blockade in
the same forwards. It therefore asked whether the installed Game or Neutral
feedback state changes how Seed uses matching first-presentation candidate
history, rather than merely showing two separate main effects.

On frozen confirmation questions, installing the Game rather than Neutral
suffix increased the matching-history old-W1 route by +1.078 centered logits
on SimpleMC and +1.279 on TriviaMC. The same interaction appeared whether the
surrounding recipient prompt was Game or Neutral. All intact, matching-blocked,
and cyclic-control identity cells exactly reproduced their prior experiments.
Thus the feedback state causally controls the interpretation of recollected
candidate rank; it is not simply an independent output bias added after
recollection.

## Fresh 2P × recollected-history removal

Step 5 tested the reverse side of the proposed double dissociation. Seed-native
linear directions were fitted only on discovery questions from standalone
remapped candidate scores. At every layer L1--L64, the intervention removed the
component unique to that fresh score from every second-presentation option
semantic wordpiece and option-line newline, while immediately preserving the
decoded old-score coordinate. A deterministic random direction with exactly
the same L2 edit supplied the causal dose control. The fresh edit was also
crossed with the complete matching-history blockade.

The manipulation was strong and selective: it removed 97.26% of the targeted
fresh coordinate on SimpleMC and 98.31% on TriviaMC, while changing the
protected old-score coordinate by only 0.103% and 0.070%, respectively. Natural,
duplicate-natural, and no-op hook controls were all exactly identical, and all
stored logits were finite.

The original primary analysis was underpowered for its interpretation. On the
frozen confirmation **conflict** sets, the natural Game-minus-Neutral
old-W1-avoidance gap was only +4.8 points `[0.0,+10.3]` on SimpleMC (`n=126`)
and +5.5 `[-5.5,+16.4]` on TriviaMC (`n=55`). The fresh-minus-random choice
interactions of +4.0 `[-2.4,+10.3]` and -3.6 `[-16.4,+9.1]` therefore cannot
show that fresh removal preserved a clearly detected choice-level effect.

At score resolution, the fresh coordinate is causally contributory in Seed.
The natural complete-confirmation centered old-W1 Game-minus-Neutral gaps were
-0.604 logits on SimpleMC and -1.065 on TriviaMC. Fresh scrub minus same-dose
random reduced those gaps by +0.215 `[+0.143,+0.292]` on full SimpleMC and
+0.111 `[-0.034,+0.253]` on full TriviaMC. On TriviaMC conflicts, where the
natural score gap was -0.553, the reduction was +0.249 `[+0.010,+0.491]`.
Thus the decoded fresh 2P score accounts for a significant minority of Seed's
policy-conditioned old-winner suppression in at least one informative slice
of each dataset. Removing it does not eliminate the policy effect, and the
experiment still bounds only this validated linear coordinate rather than all
distributed fresh computation.

The matching-history result also splits by conflict status. On complete
confirmation sets, matching blockade changed the choice gap by -10.4 points
`[-16.9,-4.0]` on SimpleMC and -7.2 `[-12.4,-2.0]` on TriviaMC. On conflict
questions it changed the gap by only -0.8 `[-9.5,+7.9]` and exactly 0.0
`[-14.5,+14.5]`; the aggregate collapse was carried by non-conflicts. After
matching blockade and fresh removal were combined, significant conflict-set
score gaps remained (-0.259 SimpleMC and -0.362 TriviaMC). The two lesions do
not exhaust Seed's conflict pathway. Because the matching lesion edits only
1P-option-line to matching-2P-option-line attention, direct final-position
reads of 1P history are one plausible Seed-specific bypass, but this run does
not localize the remainder.

The corrected cross-model verdict is therefore split: **aggregate dependence
on matching recollection replicates in Seed, but the Qwen recollection-versus-
fresh-evidence dissociation does not.** Seed combines matching recollection,
a material fresh-score contribution, and an unassigned conflict-route
component.

## Final-decision layerwise trajectories

The complete follow-up measured the exact final prompt position after every
Seed block L1--L64 on all 500 questions in each dataset. Because no compatible
Seed Jacobian lens exists, the fixed readout is the standard Seed logit lens:
the exact final RMS norm and A--D unembedding rows. Separate held-out linear
decoders test whether the final answer pattern is already present before this
fixed readout can expose it.

The directly output-readable rank pattern appears abruptly around L40--L42 in
both datasets. At the exact final layer, Game-minus-Neutral centered rank
effects are `-1.530/+0.121/+0.450/+0.958` for R1--R4 on SimpleMC and
`-2.142/+0.766/+0.720/+0.656` on TriviaMC. Thus the replicated Seed policy is
an organized rank-specific transformation: Game selectively weakens the old
winner and supports lower-ranked alternatives relative to Neutral.

On held-out confirmation questions, the complete question-specific final
Game-minus-Neutral pattern becomes persistently linearly decodable at L36 on
SimpleMC and L39 on TriviaMC. At L40, decoder cosine is 0.575 versus 0.223 for
the standard logit lens on SimpleMC, and 0.446 versus 0.154 on TriviaMC.
Shared-condition decoders are essentially as strong as condition-specific
decoders after the late transition, while cross-condition decoders also
generalize well. The prospective answer therefore lives mainly in a shared
linear basis; the task changes the represented rank vector rather than putting
Game and Neutral answers into wholly different coordinate systems.

Seed does **not** exactly reproduce Qwen's narrow switch-panel timing. On
eventual Seed switch trials, R1 is often already output-readable when the late
pattern first separates, and R2 overtakes it later. Because these panels select
questions using the eventual output, they cannot establish an R1-then-suppress
causal computation. The stronger all-question result is that a structured,
question-specific policy adjustment is already accessible before it becomes
fully output-readable. The matching-history blockade above remains the causal
evidence that recollection is used to create preferential Game R1 avoidance.

## Validation and cost

- Matching history: 500 questions and 750 complete forwards per dataset; all
  four candidate routes and all Seed layers 1--64 in every intervention.
- Feedback suffix: 500 questions and 1,000 complete forwards per dataset; all
  seven source tokens and all Seed layers 1--64.
- Every stored logit is finite; trusted-natural and duplicated-row identity
  errors are exactly 0.0.
- All requested GPU runs are stopped. The matching-history plus suffix
  batch cost $1.195. The later all-layer trajectory batch increased the same
  authenticated managed-host rows by another $0.492, including benchmark,
  inference, activation retrieval, and stop latency. Each batch remained far
  below its separate $15 authorization; unrelated concurrent hosts are
  excluded from the accounting.
- The later policy-by-recollection plus fresh-removal batch cost $8.326 in
  authenticated managed-instance charges from its prestart baseline, including
  GPU, disk, and transfer charges; Step 5 itself added $4.657 after Step 4. This
  also remained below its $15 cap.

## Canonical artifacts

- [Frozen plan](PLAN.md)
- [SimpleMC matching-history report](simplemc/matching_history/analysis/REPORT.md) ·
  [summary](simplemc/matching_history/analysis/summary.json)
- [TriviaMC matching-history report](triviamc/matching_history/analysis/REPORT.md) ·
  [summary](triviamc/matching_history/analysis/summary.json)
- [SimpleMC feedback-suffix report](simplemc/feedback_suffix/analysis/REPORT.md) ·
  [summary](simplemc/feedback_suffix/analysis/summary.json)
- [TriviaMC feedback-suffix report](triviamc/feedback_suffix/analysis/REPORT.md) ·
  [summary](triviamc/feedback_suffix/analysis/summary.json)
- [Final-decision trajectory report](../seed_oss_36b_final_position_trajectories/analysis/REPORT.md) ·
  [held-out decoder report](../seed_oss_36b_final_position_trajectories/prospective_decoding/analysis/REPORT.md) ·
  [frozen design](../seed_oss_36b_final_position_trajectories/PLAN.md)
- [SimpleMC policy × recollection report](simplemc/policy_recollection_factorial/analysis/REPORT.md) ·
  [TriviaMC policy × recollection report](triviamc/policy_recollection_factorial/analysis/REPORT.md)
- [SimpleMC fresh-removal report](simplemc/fresh_removal/analysis/REPORT.md) ·
  [summary](simplemc/fresh_removal/analysis/summary.json) ·
  [TriviaMC fresh-removal report](triviamc/fresh_removal/analysis/REPORT.md) ·
  [summary](triviamc/fresh_removal/analysis/summary.json)
- Figures: [SimpleMC matching history](../../../figures/model_replications/seed_oss_36b_simplemc_matching_history.png),
  [TriviaMC matching history](../../../figures/model_replications/seed_oss_36b_triviamc_matching_history.png),
  [SimpleMC feedback suffix](../../../figures/model_replications/seed_oss_36b_simplemc_feedback_suffix.png),
  [TriviaMC feedback suffix](../../../figures/model_replications/seed_oss_36b_triviamc_feedback_suffix.png),
  [SimpleMC displayed-letter-controlled trajectories](../../../figures/model_replications/seed_oss_36b_simplemc_nonremapped_rank_trajectories_letter_controlled.png),
  [TriviaMC displayed-letter-controlled trajectories](../../../figures/model_replications/seed_oss_36b_triviamc_nonremapped_rank_trajectories_letter_controlled.png),
  [SimpleMC fresh removal](../../../figures/model_replications/seed_oss_36b_simplemc_fresh_history_double_dissociation.png),
  [TriviaMC fresh removal](../../../figures/model_replications/seed_oss_36b_triviamc_fresh_history_double_dissociation.png)
