# Seed-OSS 36B mechanistic replication

## Objective

Determine whether the clean Seed behavioral replication is implemented by the
same architecture-general causal logic established in Qwen3.6-27B:

1. matching first-presentation candidate history is causally used in the
   second presentation;
2. the `incorrect`/`lost` feedback contrast supplies a transferable policy;
3. that policy changes how retrieved candidate rank affects the final answer.

The program is gated. The first experiment below must be completed and
reported before policy crossover or a policy-by-history factorial is launched.

## Execution status

- Step 1 SimpleMC matching-history blockade: **complete**, 500/500 questions,
  all outputs finite, exact natural reproduction, all 64 attention layers
  exercised in every intervention. The frozen confirmation result causally
  replicates Game's use of matching candidate history and elimination of the
  preferential old-W1-avoidance gap. Natural Neutral does not show a stable
  matching-specific rank profile.
- Step 2 TriviaMC matching-history replication: **complete; gate passed** on
  both frozen confirmation endpoints. The W1 task interaction was +1.277
  [0.942, 1.618], and the matching-versus-cyclic change in the Game-minus-Neutral
  old-W1 choice gap was +10.0 [5.2, 14.8] percentage points.
- Step 3 complete feedback-suffix crossover on both datasets: **complete**,
  500/500 questions and 1,000 complete forwards per dataset. All outputs were
  finite; trusted-natural and distinct-row identity errors were exactly 0.0;
  every seven-token source span and all 64 layers were exercised. On the
  confirmation splits, transfer of the paired natural Game/Neutral centered
  A--D vector was 1.0002 in both directions on SimpleMC and 0.9999 in both
  directions on TriviaMC.
- Step 4 direct policy × recollection factorial: **complete** on all 500
  SimpleMC and 500 TriviaMC questions. Every exact identity and prior-result
  reproduction control passed. The installed Game-versus-Neutral suffix changed
  the matching-history W1 route by +1.078 on SimpleMC and +1.279 on TriviaMC
  confirmation, identically in native Game and Neutral recipient prompts.
- Step 5 Seed-specific fresh-2P × recollection removal: **complete** on all 500
  SimpleMC and 500 TriviaMC questions, with every layer L1--L64 edited. The
  intervention removed 97.26%/98.31% of the decoded fresh coordinate while
  disturbing the protected old-score coordinate by only 0.103%/0.070%; all
  identity controls were exact. The prespecified confirmation-conflict choice
  endpoint had weak natural gaps (+4.8 points SimpleMC and +5.5 TriviaMC, both
  intervals including zero), so its null scrub interactions cannot establish
  preservation of a clearly detected effect. At score resolution, fresh scrub
  minus same-dose random reduces the complete-confirmation SimpleMC old-W1 task
  gap by +0.215 `[+0.143,+0.292]` logits and the TriviaMC conflict gap by
  +0.249 `[+0.010,+0.491]`. Matching blockade removes the aggregate choice gap
  but not the conflict gap, and significant conflict score gaps survive the
  joint lesion. Corrected verdict: aggregate recollection dependence
  replicates, but the Qwen recollection-versus-fresh-evidence dissociation does
  not; Seed has a material fresh contribution and an unassigned conflict-route
  component.

[Canonical Step-1 report](simplemc/matching_history/analysis/REPORT.md) ·
[machine-readable summary](simplemc/matching_history/analysis/summary.json) ·
[figure](../../../figures/model_replications/seed_oss_36b_simplemc_matching_history.png) ·
[integrated completed report](REPORT.md)

## Step 1 — SimpleMC matching-history blockade

- **Model:** `ByteDance-Seed/Seed-OSS-36B-Instruct`, revision
  `497f1dca95ebdec98e41d517b9f060ee753c902f`, BF16, Transformers 4.57.6,
  native chat template, `thinking_budget=0`.
- **Questions:** all 500 frozen SimpleMC questions. The existing result-independent
  251/249 discovery/confirmation split is retained. No correctness or behavioral
  outcome filter is applied.
- **Prompt:** canonical empty-first-assistant clean paradigm. Only fully remapped
  Game and Neutral prompts are used because the causal question requires semantic
  candidate identity to be dissociated from displayed A-D letters.
- **Candidates:** all four candidates jointly. W1--W4 are frozen from Seed's own
  trusted first-presentation aggregated A-D logits.
- **Sources:** every token in each complete first-presentation option line.
- **Receivers:** every token in the semantically matching complete second-presentation
  option line.
- **Layers:** every Seed decoder layer 1--64. Seed has ordinary grouped-query
  causal self-attention at every layer; no Qwen layer cutoff is inherited.
- **Causal edit:** remove only the selected receiver-query to source-key attention
  edges. Source and receiver residual states are not replaced.
- **Conditions per task:** natural; all-four matching blockade; all-four cyclic
  wrong-line blockade (`W1<-W2`, `W2<-W3`, `W3<-W4`, `W4<-W1`). The cyclic
  control edits the same four receiver lines and all 64 layers but is an equal
  structural dose rather than an exact source-token-count match.
- **Complete work:** six complete forwards per four-question cohort, 125 cohorts,
  750 complete batched model forwards after load.
- **Primary outcomes:** matching-minus-cyclic candidate-centered A-D evidence by
  W1--W4; the Game-minus-Neutral rank interaction; and the matching-minus-cyclic
  change in the Game-minus-Neutral old-W1 choice gap. Discovery, confirmation,
  and all-question paired bootstrap intervals are reported.
- **Validity:** all logits finite; every source/receiver span nonempty; natural
  logits exactly reproduce the trusted Seed behavioral run; every intervention
  executes in all 64 attention layers; prompt pair differs only at
  `incorrect`/`lost`.

This is a whole-line causal test. It does not localize the route to semantic
wordpieces, test direct final-query reads, edit a recurrent state, measure
correctness, or establish policy transmission.

## Step 2 — TriviaMC matching-history replication

Repeat Step 1 without narrowing any source, receiver, candidate, or layer:
all 500 frozen difficulty-filtered TriviaMC questions, the existing 250/250
discovery/confirmation split, all four matching and cyclic-wrong lines, and all
64 Seed attention layers. The prespecified pass gate is evaluated on the frozen
confirmation split and requires both (a) a positive old-W1 Game-minus-Neutral
matching-versus-cyclic centered-logit interaction whose 95% paired-bootstrap CI
excludes zero, and (b) a positive matching-minus-cyclic change in the
Game-minus-Neutral old-W1 choice gap whose 95% paired-bootstrap CI excludes
zero. This demands replication in both continuous evidence and displayed
choice, not merely a same-signed point estimate.

## Conditional Step 3 — complete feedback-suffix crossover

If Step 2 passes, run the complete reciprocal Game/Neutral feedback-source
crossover separately on all 500 SimpleMC and all 500 TriviaMC questions.

- **Source:** the full contiguous feedback suffix from `incorrect` or `lost`
  through the final period immediately before the repeated instruction/question.
- **State and layers:** source-token ordinary-attention K/V at every Seed layer
  1--64, visible only to causally later queries. Source residual outputs are not
  replaced. Seed has no GLA, so no unrelated state is substituted for Qwen GLA.
- **Conditions:** natural, real duplicated-row same-task identity, and reciprocal
  Game-to-Neutral / Neutral-to-Game crossover for the same question.
- **Validity:** exact trusted-natural reproduction, bit-exact duplicated-row
  identity, aligned source spans, finite logits, and all 64 patchers firing.
- **Primary outcome:** fraction of the natural Game-minus-Neutral old-W1 choice
  gap and candidate-centered W1 evidence transferred by the opposite-task
  feedback suffix, on the existing frozen discovery and confirmation splits.
- **Scope:** this establishes causal sufficiency/transfer of the complete
  feedback suffix's outgoing K/V state; it does not localize policy to the
  evaluation word, period, or any single layer.

## Step 4 — direct policy × recollection factorial

The completed Step-1/2 blockades and Step-3 suffix crossover establish two
separate causal main effects. They do not by themselves establish that the
feedback policy changes how the matching-history route is used. Step 4 combines
both edits in the same model evaluations.

- **Datasets and splits:** all 500 frozen SimpleMC and all 500 frozen
  difficulty-filtered TriviaMC questions; the existing discovery/confirmation
  halves remain fixed and confirmation is primary.
- **Recipient prompt:** Game and Neutral, fully remapped, exact clean format.
- **Installed feedback state:** same-task identity or reciprocal opposite-task
  transfer of the complete seven-token suffix from `incorrect/lost` through
  `Choose the answer again.`. At every Seed layer L1--L64, causally later
  queries see donor suffix K/V while the suffix tokens' own residual outputs
  remain recipient-natural.
- **Matching-history access:** intact; all four matching complete 1P-to-2P
  option-line query/key edges blocked; or the equal-structure cyclic wrong-line
  blockade. Every Seed attention layer L1--L64 is included.
- **Complete cells:** both installed suffix states crossed with all three
  history-access levels inside both recipient prompts. Same-task identity cells
  reproduce the prior natural/matching/cyclic results; reciprocal-intact cells
  reproduce the prior suffix crossover.
- **Primary interaction:** the matching-minus-cyclic W1 effect under installed
  Game suffix minus the same matching-minus-cyclic effect under installed
  Neutral suffix, for centered W1 evidence and old-W1 displayed choice. Report
  W1--W4 rank vectors, both crossover directions, both frozen splits, and all
  questions.
- **Validity:** exact trusted-natural reproduction; bit-exact distinct-row
  same-task suffix identity under every access level; reproduction of prior
  matching/cyclic and reciprocal-intact cells; all logits finite; nonempty
  source/receiver spans; all 64 suffix patchers and edge lesions fire.

This is the direct causal policy-by-recollection interaction. It does not test
whether all fresh second-presentation reconstruction is unnecessary.

## Step 5 — Seed fresh-2P × recollected-history removal

Seed has no frozen 2P option-state fresh-score decoder. A Seed-specific
coordinate must therefore be constructed before intervention; Qwen directions
must not be transferred across architectures.

1. Collect every L1--L64 post-block residual at all second-presentation
   semantic wordpieces and option-boundary newlines for Game and Neutral on
   both datasets. Fit fresh-score and old-score decoders on discovery questions
   only and validate candidate-wise prediction on confirmation.
   Before fitting, obtain Seed's own **fresh-score target** from an answer-only
   bare remapped-question condition containing the second displayed question
   and option order but no first answer, feedback, or Second Chance history.
   This is a separate Seed forward and must not be substituted with Qwen's
   remapped-baseline logits or Seed's first-presentation logits. The old and
   fresh targets are candidate-centered and each is residualized on the other
   target plus both displayed positions using discovery questions only.
2. At every L1--L64 block, remove the component geometrically unique to fresh
   score from every 2P semantic wordpiece and newline while immediately
   restoring the decoded old-score coordinate. Seed has no GLA state, so only
   its ordinary residual stream is edited.
3. Cross the fresh scrub with the complete matching-history blockade. Include
   native natural, complete-path identity, no-op hook, same-dose deterministic
   random edit, matching-only, fresh-only, joint fresh+matching, and
   matching+random scenarios.
4. Require exact natural/no-op identity, finite outputs, removal of at least
   90% of the targeted fresh coordinate, negligible old-coordinate change at
   the edit, and identical random-edit L2 dose. The held-out Game-minus-Neutral
   old-W1-avoidance gap is primary; destination choice is secondary and no
   correctness endpoint is introduced.

A null bounds only the validated Seed decoded fresh subspace; it must not be
described as removing every possible distributed reconstruction process.
