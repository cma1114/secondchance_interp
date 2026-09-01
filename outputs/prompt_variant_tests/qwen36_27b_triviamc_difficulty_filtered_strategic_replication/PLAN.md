# Qwen3.6-27B TriviaMC strategic-switching replication

## Objective

Test whether the core SimpleMC mechanism reproduces on the existing frozen
500-question difficulty-filtered TriviaMC set with the same pinned
Qwen3.6-27B revision and canonical raw-ChatML, empty-history, answer-only
prompt. The target account is policy-conditioned semantic reinstatement:

1. Game preferentially leaves the semantic first-pass winner after remapping,
   and its final rank profile selectively compresses the old ranking.
2. Both Game and Neutral causally use matching first-presentation candidate
   history; Neutral supports all old candidates most strongly at W1, whereas
   Game suppresses W1/W2 and relatively supports weak old candidates.
3. The one-word `incorrect`/`lost` input contrast produces a transferable
   contextualized feedback policy.
4. That policy causally changes how the matching-history route uses retrieved
   first-pass rank.

This document freezes the complete program, but each step is gated. Step 1 is
complete and reported. Steps 2--4 remain unlaunched; no later GPU experiment
may start until the user reviews Step 1 and explicitly proceeds.

## Execution status

- Step 1: **complete**, 500/500 questions in Baseline, Game, and Neutral; all
  outputs finite; exact one-token `incorrect`/`lost` prompt-pair audit passed.
  Canonical [report](step1/analysis/REPORT.md),
  [summary](step1/analysis/summary.json), and
  [figure](../../../figures/qwen36_triviamc_strategic_replication_step1.png).
- Step 2: **complete**, 500/500 questions; all causal and natural outputs
  finite; exact natural reproduction; canonical
  [report](step2/analysis/REPORT.md),
  [summary](step2/analysis/summary.json), and
  [figure](../../../figures/qwen36_triviamc_matching_history_step2.png).
- Step 3: **complete**, 500/500 questions; all outputs finite; real
  duplicated-row transplant identity, corrected identity, and trusted-natural
  reproduction all exactly 0.0-error; canonical
  [report](step3/analysis/REPORT.md),
  [summary](step3/analysis/summary.json), and
  [figure](../../../figures/qwen36_triviamc_feedback_suffix_step3.png).
- Step 4: **complete**, 500/500 questions; all outputs finite; corrected
  trusted-natural reproduction exactly 0.0-error; canonical
  [report](step4/analysis/REPORT.md),
  [summary](step4/analysis/summary.json), and
  [figure](../../../figures/qwen36_triviamc_policy_rank_step4.png).

## Frozen common design

- Dataset: `outputs/reproduction/triviamc_qwen36_27b/stimulus_manifest.json`,
  the existing seed-42 manifest of all 500 rows from
  `TriviaMC_difficulty_filtered.jsonl`; answer-option order is preserved from
  the source.
- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, BF16.
- Serialization: raw Qwen ChatML, thinking disabled, explicit empty first
  assistant answer, unrestricted full-vocabulary next token with A-D as the
  required answer format.
- Feedback pair: `Your answer was incorrect. Choose the answer again.` versus
  `Your answer was lost. Choose the answer again.` The two prompts differ only
  at the single tokenizer token `incorrect`/`lost`; this must be audited on the
  executed tokenizer.
- Remapping: all four second-presentation answer contents move to different
  A-D letters. The nine derangements are balanced within the local
  first-presentation winner-letter strata using seed 20260828. Game and Neutral
  receive the same question-specific mapping.
- Split: a result-independent seed-20260828 shuffle of the 500 frozen question
  IDs, 250 discovery and 250 confirmation, frozen before any Game/Neutral
  result is collected.
- Choice rule: the first-presentation W1 and W1--W4 ordering are frozen from
  the aggregated bare-plus-space A-D logits, exactly as required by the
  existing remapping-plan utility and the original old-rank analysis. The
  unrestricted final top token is the primary second-presentation choice;
  aggregated A-D final choice is reported as a secondary robustness readout.
  Exact aggregated-logit ties are resolved in displayed A-D order before
  semantic remapping. The local first-presentation unrestricted/aggregated
  agreement is reported explicitly, so any dataset-specific discrepancy is
  visible rather than silently mixed.
- Statistics: question-paired, first-winner-letter-stratified bootstrap 95%
  intervals; all questions and both frozen halves reported. Correctness is not
  a scientific endpoint.

## Step 1 — natural remapped behavior and old-rank transformation

### Executed conditions

1. Local prompt-identical first-presentation Baseline, used to freeze W1--W4
   and the remapping plan.
2. Remapped `incorrect_again` (Game).
3. Remapped `lost_again` (Neutral).

This is exactly three complete natural forwards per question, physically
batched in fours: 125 Baseline forwards and 250 second-chance forwards, 375
complete model calls total. No remapped standalone second-presentation
baseline, accuracy analysis, activation collection, lesion, transplant, or
relay condition belongs to Step 1.

### Primary outcomes

- Semantic-content switch away from W1 in Game and Neutral.
- Avoidance of the old literal A-D letter as the remapping dissociation.
- First-to-final raw logit, question-centered logit, and per-question softmax
  probability profiles aligned by W1--W4.
- Final choice rates by old rank.
- Game-minus-Neutral rank-shaped final evidence and A-D entropy as a secondary
  uncertainty measure.

### Gate

Report Step 1 and stop. Later work is scientifically earned if Game shows a
replicated positive semantic-switch difference over Neutral and a rank-shaped
transformation that is inconsistent with equal candidate noise. A weak or
absent result is reported as such; it does not automatically launch Step 2.

## Step 2 — full-range all-candidate semantic-history blockade

Replicate only the completed SimpleMC experiment's critical cells in both
tasks: natural, all-four matching 1P-to-2P option-line blockade, and an
equal-structure cyclic wrong-line blockade across every ordinary-attention
layer. The cyclic control denies each 2P old-rank receiver the complete 1P line
for the next old rank (W1 receives W2, W2 receives W3, W3 receives W4, and W4
receives W1). Thus both lesions edit four complete source-to-receiver line
relations at the same 16 layers. Complete option lines have naturally unequal
token counts, so this is an equal structural dose—not an exact token-count
match—and the executed source/query counts are saved question by question.

### Frozen Step-2 scope and rationale

- Questions and splits: all 500 frozen TriviaMC questions, with the existing
  result-independent 250/250 discovery/confirmation split. This is the entire
  dataset rather than a selected cohort.
- Tasks: both `incorrect_again` (Game) and `lost_again` (Neutral), because the
  scientific target is whether the same recollection route is used under both
  policies and whether its rankwise effect differs by policy.
- Sources: all tokens in each complete first-presentation option line,
  including its displayed letter, punctuation, semantic wordpieces, and line
  boundary. Receivers: all tokens in the semantically corresponding complete
  second-presentation option line. This exactly replicates the completed
  SimpleMC whole-line causal test; it does not claim token-level localization.
- Candidates: all four semantic candidates are blocked jointly. The requested
  replication concerns the complete candidate-history route; individual-line
  lesions and the 32-mask receiver factorial are deliberately excluded.
- Layers and mechanism: every ordinary-attention block in Qwen3.6-27B,
  one-based layers 4, 8, ..., 64. No earlier or later ordinary-attention layer
  exists in this architecture. GLA state is not edited because this test asks
  whether direct ordinary-attention reads from 1P option lines into their 2P
  semantic matches are causal; it is not a claim that GLA is absent elsewhere.
- Conditions: exactly natural, joint matching blockade, and joint cyclic
  wrong-line blockade per task. There are six complete forwards per physical
  four-question cohort, 125 cohorts, and 750 complete model calls after load.
- Outcomes: old-rank-aligned, candidate-centered A-D logit effects for W1--W4;
  stable displayed-order aggregated-A-D W1-choice rates; and the change in the
  Game-minus-Neutral W1-choice difference. The matching-minus-cyclic contrast
  is primary; matching-minus-natural and cyclic-minus-natural are retained to
  expose both sides of that contrast. Discovery, confirmation, and all-question
  estimates receive paired W1-letter-stratified bootstrap intervals.
- Exclusions: no correctness endpoint, no fresh-2P baseline, no conflict-only
  subset, no activation decoding, no individual candidate cells, no semantic-
  token-only receiver cells, and no policy transplant. Those answer different
  questions and would cease to be the requested compact replication.

### Step-2 result and gate

On the untouched confirmation half, matching blockade minus cyclic control
raises Game old-W1 candidate-centered evidence by +0.698 logits and lowers
W3/W4 by -0.164/-0.442. The Game-minus-Neutral rank interaction is
+0.767/-0.137/-0.288/-0.341, with all four paired intervals excluding zero.
Natural aggregated-A-D old-W1 choice is 68.4% in Game versus 73.2% in Neutral;
under the matching blockade it is 74.4% versus 74.0%, eliminating the task
difference. The primary matching-minus-cyclic change in the task W1-choice gap
is +9.6 points `[+4.4,+14.8]`; discovery gives +8.0 `[+2.4,+13.6]`.

The policy-dependent Game result therefore reproduces, including the causal
loss of preferential Game W1 avoidance. Neutral's rankwise matching-specific
profile is not stable across the two halves, so the stronger SimpleMC claim
that Neutral reliably supports old candidates through this exact route does
not independently reproduce here. Stop and report before Step 3; no Step-3
condition is implicitly authorized by Step-2 completion.

## Step 3 — complete-feedback-suffix policy crossover

Replicate only natural, exact same-task identity, and reciprocal complete
feedback-suffix outgoing-state crossover. Cross downstream ordinary-attention
and GLA-memory writes from `incorrect/lost` through the final period while
preserving each source token's own local output. Omit individual-token,
grouped-span, layer-localization, and relay-mediation cells.

### Frozen Step-3 scope and rationale

- Questions and splits: all 500 frozen TriviaMC questions and the existing
  result-independent 250/250 discovery/confirmation split.
- Paired execution: each complete four-question cohort is evaluated as two
  four-row paired subbatches, each containing the same two questions in Game
  and Neutral. This permits exact same-question, opposite-task donors without
  changing the validated physical batch size.
- Source span: exactly seven contiguous tokenizer tokens—`incorrect/lost`, the
  first period, `Choose`, `the`, `answer`, `again`, and the final period. The
  executed tokenizer positions and decoded token strings are audited.
- State crossed: downstream ordinary-attention keys and values at all 16
  ordinary-attention blocks and recurrent GLA k/v/g/beta writes at all 48 GLA
  blocks. Each source token's own local residual/output remains recipient-
  natural; only what later tokens can receive from that source is crossed.
- Scenarios: natural; a real same-task identity pass using the complete patcher
  path between distinct duplicated rows with identical prompts; and reciprocal
  Game-to-Neutral plus Neutral-to-Game complete-suffix crossover. The patcher
  deliberately rejects a row as its own donor, so each two-question paired
  subbatch uses one duplicated-row identity forward per task. Every identity
  output must reproduce its paired same-batch natural logits exactly.
- Complete work: natural and reciprocal crossover are two forwards per paired
  subbatch; duplicated-row Game and Neutral identity are two more. With two
  paired subbatches per canonical cohort, this is eight forwards per cohort,
  125 cohorts, and 1,000 complete model calls after load.
- Primary outcome: projection of the crossover-induced centered A-D logit
  change onto the paired natural donor-minus-recipient task vector, reported
  separately for Game recipients and Neutral recipients on both frozen halves.
  Secondary outcomes are old-W1 choice and old-rank-aligned centered-logit
  changes. Statistics are paired and stratified by first-presentation W1
  letter. Correctness is not an endpoint.
- Exclusions: no individual feedback token, feedback-sentence/instruction
  grouping, layer localization, relay restoration, convolution intervention,
  matching-history blockade, or policy-by-route factorial. Those belong to
  other mechanistic questions or Step 4.

### Step-3 result and gate

The complete seven-token feedback suffix transfers the paired donor task
vector strongly and reciprocally on both frozen halves. Discovery transfer is
0.920 `[0.907,0.932]` into Game and 0.933 `[0.921,0.944]` into Neutral;
confirmation is 0.905 `[0.892,0.918]` and 0.920 `[0.905,0.934]`. The
prespecified gate therefore passes in all four task-by-split cells.

The rank-shape also reverses in the expected direction. On confirmation,
Neutral suffix state installed into Game raises W1 by +0.621 centered logits
and lowers W2/W3/W4 by -0.199/-0.187/-0.235; Game suffix state installed into
Neutral lowers W1 by -0.656 and raises W2/W3/W4 by
+0.210/+0.207/+0.239. Old-W1 choice rises by 5.6 points `[1.6,10.0]` in
Game and falls by 4.0 points `[-8.0,0.0]` in Neutral. Thus the contextualized
feedback suffix causally supplies essentially the same opposite rank-shaped
policy on TriviaMC as on SimpleMC, despite the weaker and less stable Neutral
matching-history result in Step 2.

This establishes policy-source transfer, not the interaction between policy
and retrieved candidate rank. Stop and report before Step 4; no Step-4
condition was launched.

## Step 4 — policy × retrieved-rank factorial

Replicate the existing reciprocal policy-state transplant crossed with the
all-four matching-history versus cyclic-wrong control. The primary endpoints
are the rankwise policy-by-route interaction and conflict-trial W1 choice.
Omit MLP-49 restoration and individual-layer refinement from the initial
cross-dataset test.

### Step-4 executed scope and result

The exact historical factorial was replicated without its MLP-49 add-on. For
each two-question paired subbatch, six complete forwards crossed natural versus
reciprocally transplanted evaluation-closing-period GLA updates with no route
blockade, the all-four matching blockade, and the all-four cyclic-wrong
blockade. The period transplant covered all 48 GLA layers with
`preserve_source_output=False`; it therefore tests the complete period-token
GLA update, including donor-conditioned local output, rather than isolated
recurrent memory. Matching and cyclic controls covered all 16 ordinary-
attention layers and every token of the four complete source and receiver
option lines. One standalone remapped-baseline forward per canonical cohort
froze the original conflict definition, W1 unequal to the unrestricted fresh
second-presentation winner. Total work was 1,625 complete model forwards.

The held-out policy-by-route interaction is strong and reciprocal. In natural
Game, matching blockade minus cyclic control raises W1 by +0.698 logits and
lowers W3/W4 by -0.166/-0.442, so the intact route selectively suppresses the
recollected winner and supports weaker candidates. Installing Neutral's period
update reduces that W1 lesion effect by 0.607 logits to +0.091. Natural Neutral
again has no stable confirmation route profile, but installing Game's period
update creates a Game-like effect: +0.822 at W1 and -0.206/-0.519 at W3/W4.
The corresponding Neutral policy-by-route interaction is
+0.890/-0.141/-0.330/-0.419 across W1--W4; all four confirmation intervals
exclude zero, and the rank shape replicates on discovery.

On the 74 confirmation conflict trials, Neutral policy installed in Game raises
old-W1 choice by +13.5 points `[+5.4,+21.6]`; Game policy installed in Neutral
lowers it by -12.2 `[-21.6,-2.7]`. Discovery independently gives +16.4 and
-19.7 points. Thus the feedback-period update causally changes how matching
candidate history is used, rather than merely adding a task bias after
recollection. The Game-conditioned route generalizes strongly; the stronger
SimpleMC claim of stable natural-Neutral support through this exact route does
not.

## Operational requirements

Every step independently follows the repository Vast protocol: authenticated
fleet audit, exact forward-count inspection, complete-path benchmark, forecast
before a long launch, atomic per-batch checkpoints, host-local detached
supervision, monitoring at intervals no longer than 15 minutes, retrieval and
validation before stopping the GPU, charge query, ledger update, and final
fleet audit. The standing cap is $15 per explicitly requested batch.
