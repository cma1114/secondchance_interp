# How Qwen3.6-27B solves Game and Neutral

## Scope

This is the integrated account for the current Qwen3.6-27B SimpleMC prompt,
especially the semantic-remapping paradigm. Most decisive mechanistic tests use
the action-matched contrast:

- **Game / Evaluation:** `Your answer was incorrect. Choose the answer again.`
- **Neutral:** `Your answer was lost. Choose the answer again.`

The standard Game wording says `Choose a different answer.` The behavioral
factorial shows that `incorrect` supplies most of the targeted semantic effect;
`different` adds relatively little W1 avoidance while increasing uncertainty.
This lets the mechanistic account focus on the one-word policy difference
without pretending that the full prompt is identical.

Throughout, **W1** is the semantic candidate preferred on the first
presentation and **W2** is the candidate a fresh solve of the remapped second
presentation would prefer. Conflict trials have W1 != W2.

The one-word effect also exists **without remapping**: `incorrect` produces
8.8 [+4.6, +12.8] percentage points more switching than `lost`, raises entropy
by 0.232 bits, and sends 71.0% of its switches to the original runner-up. That
run is behavioral. The position- and layer-resolved interventions below were
performed in the remapped diagnostic unless explicitly stated otherwise; an
equivalent non-remapped mechanistic path has not been collected.

## Position-by-position, layer-by-layer account

### 1. First-presentation first option line (literal A, including its newline)

- **Both conditions:** Game and Neutral are identical here; the feedback word
  has not appeared. Any difference in later use must therefore be imposed
  downstream.
- **What is locally readable:** the line obviously contains option A's text.
  Corrected cross-mapping decoding finds a modest mapping-invariant content
  component at the content end/newline around layers **32--43**. By the late
  layers the newline is dominated by displayed-letter/position structure, not
  a clean semantic code. A separate candidate-value ranker peaks at layer
  **53** (64.9% held-out winner prediction versus 51.9% from letter alone).
  But for the *first* option, A, the same-content/same-letter
  chosen-versus-unchosen score difference is exactly zero: its prefix is
  identical before B--D have appeared. Thus the A newline does **not** locally
  contain the demonstrated context-dependent final-winner signal seen at the
  later B--D newlines.
- **Causal semantic source:** in the fixed-A crossover, transplanting ordinary-
  attention K/V for the entire selected A line (about ten tokens: label,
  content, and newline) while leaving the visible recipient prompt and all
  other source positions unchanged moves the final decision toward the donor
  semantic answer. On held-out questions the donor-semantic margin transfer is
  0.558 logits in Game and 3.161 in Neutral; donor-answer selection rises 15.1
  and 43.7 percentage points. The first-decision boundary is near zero, and
  adding it to the line does not change the line's effect. The rest of the
  first question weakly opposes the condition difference.
- **Which persistent-state family carries it:** ordinary-attention K/V alone
  reproduces the complete-cache semantic transfer. Transplanting GLA
  convolutional state or recurrent matrices without K/V produces negligible
  donor-semantic transfer. GLA still processes these tokens and options-only
  GLA-write deletion perturbs later choices, but that does not identify GLA as
  the carrier of this line's semantic identity.
- **Who reads it:** natural-history lesions establish that repeated option
  lines read their semantically matching original lines. However, the original
  restricted fixed-A serial-mediation estimate was reversed by the exact
  full24 recalibration and did not generalize to B--D: the token-count-matched
  nonmatching control removed at least as much counterfactual donor transfer.
  Thus matching-specific mediation of selected-line transplantation is not
  established. Direct final-decision attention back to the original line is
  null.
- **Layers of the read:** the all-candidate matching-edge blockade is causal
  jointly over ordinary-attention layers **4--48**. Layers **52--64** alone
  change candidate evidence by only about 0.01 logits, and extending 4--48
  through layer 64 changes held-out effects by approximately 0.01 logits
  (maximum 0.0101 before rounding). In the
  different fixed-A donor-transplant analysis, layer **44** is the strongest
  individually sufficient source K/V transplant and layers **36--48** are
  jointly sufficient; layers **4--32** are weak alone but make the all-layer
  transplant larger when retained. These results concern different causal
  operations and do not identify one unique relay layer.
- **What is not established:** the semantic source is localized to the whole A
  line, not specifically its newline or a single residual direction. Removing
  the fitted one-dimensional candidate-value coordinate from all four newline
  K/V states has only a small, nonreplicating policy interaction. We therefore
  have not found a portable local feature at the first option that says “this
  will be the winner.”

#### What generalizes to the later first-presentation option lines (B--D)

- **Context-dependent value is readable at their newlines:** unlike A, the
  later option lines have earlier candidates available for comparison. At
  layer **53**, holding semantic content and displayed letter fixed, the probe
  score is higher when the option will win than when it will lose: B +4.31,
  C +9.73, and D +3.81 probe-score units; A is exactly 0. This is a decoding
  result, not evidence that the model causally uses that one fitted direction.
- **Every whole line is causally read by its semantic match in the second
  presentation:** the all-candidate intervention separately blocked A, B, C,
  and D from the second-presentation line containing the same answer content,
  with a cyclic nonmatching-source blockade as control. On held-out questions,
  Neutral's candidate-evidence changes from blocking the matching source were
  A -1.219, B -0.717, C -0.778, and D -0.605 logits. Thus B--D are causal
  semantic-history sources, not untested lines. The same result replicated in
  discovery (A -1.182, B -0.729, C -0.606, D -0.618).
- **Game uses these lines according to first-pass rank, not literal line:**
  averaging by literal source position partly cancels the effect (held-out A
  +0.127, B +0.184, C -0.061, D +0.079 logits), because the sign changes with
  the candidate's first-pass rank: the intact matching route demotes R1 and
  R2, is near neutral for R3, and supports R4. Literal-position averages are
  therefore not the right summary of Game's computation.
- **Layers:** these all-line matching-edge effects are carried jointly by
  ordinary-attention layers **4--48**; adding **52--64** changes them by
  approximately 0.01 logits (maximum 0.0101). The separate findings that layer 44 is the strongest
  single source layer and that 36--48 is sufficient come only from the fixed-A
  donor transplant and have not been demonstrated independently for B--D.
- **Direct B--D donor transplants now qualify the fixed-A result:** replacing
  the entire selected-line K/V history with same-letter/different-content
  donor history strongly transfers donor semantics in Neutral at every later
  position (held-out B +1.927, C +1.475, D +1.058 logits), but not in Game
  (B +0.290, C +0.009, D -0.128; every interval includes zero). Thus B--D
  lines are causal semantic-history sources, but `incorrect` prevents a
  substituted semantic state from simply controlling the final answer. The
  fixed-A positive Game donor transfer is position-specific.
- **B--D serial mediation is not matching-specific:** blocking the matching
  repeated line removes a modest part of Neutral transfer, but blocking the
  token-count-matched nonmatching line removes more at B, C, and D on both
  splits. Unlike fixed A, the counterfactual donor effect is not uniquely
  routed through the matching receiver. This does not contradict the natural-
  history lesion above: actual candidate-to-match relays remain causal for
  Game switching, while counterfactual donor substitution is largely rejected
  by Game. The larger unresolved question remains which distributed state
  carries the nonlinear graded ranking used downstream.

### 2. First-decision position

- **Both conditions:** the position contains a late A--D output-letter state.
  Replacing its complete boundary trajectory changes the immediate first
  decision to the donor letter on about 99% of held-out cases.
- **Information sent onward:** a small trace of the donor **letter** survives
  to the final decision, but the donor answer's semantic identity does not.
- **Layers:** continuously deleting the A--D letter subspace after every
  readout **48--63** changes Game and Neutral almost identically and explains
  0% of the held-out preferential W1-avoidance gap.
- **How known:** boundary crossover and continuous letter-scrub experiments.
  The complete first-presentation history is a positive semantic-transfer
  control; the one-position boundary is null for semantic transfer.

### 3. The word `incorrect` or `lost`

- **Game:** receives the literal evaluation `incorrect`.
- **Neutral:** receives the literal status `lost`.
- **Information at the position:** the one-word behavioral manipulation is
  sufficient to produce 15.2 percentage points more W1 avoidance and a -0.516
  centered-W1-logit change under Game. We have **not** isolated which recurrent
  update is written at the word itself rather than consolidated one token later
  at the period.
- **Information sent onward:** by readouts **48--56**, swapping the complete
  `incorrect`/`lost` token state between conditions changes W1 evidence by at
  most about 0.004 logits. The late literal-word state is therefore not the
  controller; condition information has already entered recurrent memory and
  other contextual positions.
- **How known:** behavioral feedback factorial and corrected late token-state
  swaps.

### 4. Period closing `incorrect.` or `lost.`

- **Game information at the position:** the complete residual becomes readable
  as error/wrong/incorrect around **42--53**.
- **Neutral information at the position:** the opposite vocabulary direction
  concerns repetition, reconstruction, and reproduction.
- **Causal state written here:** the output-preserved GLA update is distributed.
  **25--32** is the only isolated band sufficient on both frozen splits and
  carries 58.4% [42.5, 82.5]% of the corrected all-GLA route. No individual
  GLA is an indispensable bottleneck. The complete corrected transplant moves
  the W1--W2 margin by 0.097 logits Evaluation-to-Neutral and 0.091 logits in
  the reverse direction, versus a natural 0.469-logit task gap.
- **Information sent onward:** later queries to the same GLA memories carry
  this period's update through the rest of the prompt. At the final decision,
  its answer-aligned cumulative trace separates in the middle layers and its
  largest final-decision retrieval-norm differences occur at **49, 33, and
  47**. Its final direct trace favors W2 over W1. Corrected global deletion,
  with the source output preserved, removes 30.0% [9.5, 50.0%] of the
  conflict-trial choice gap and 0.097 logits [0.059, 0.135] of the margin gap.
  The persistent GLA route is causal but is not most of the policy effect.
- **Limit:** the period write has an answer-specific *later effect*, but we have
  not shown that W1 identity is locally present at this period. Later semantic
  matching may supply the identity.

### 5. Shared action clause and its closing period

- **Game information at the closing period:** correction/revision becomes
  readable around **39--43**; exclusion/rejection dominates from **49 onward**.
- **Neutral information:** the opposite direction continues to emphasize
  repetition/reconstruction.
- **Information arriving here:** the accumulated all-GLA recurrent matrices
  still contain most of the upstream condition effect. Swapping those matrices
  transfers 51--84% of the conflict-trial margin gap, depending on direction
  and whether the residual is also swapped. The precise carrier blocks are
  **not localized**.
- **Information newly sent from this period:** removing only this token's GLA
  write with its local residual output preserved has small effects. The
  corrected ordinary-attention route is real: blocking later reads changes
  Neutral conflict-trial W1-minus-W2 margin by -0.033 logits [-0.051,
  -0.014], replicating at -0.033 on each frozen split, and raises Game
  no-conflict W1 centered advantage by +0.048 [+0.031, +0.066], again with the
  same sign on both splits. No pooled W1-selection effect excludes zero and the
  joint lesion does not reproduce the main task gap.
- **How known:** two-period JLens, accumulated-state mediation, and
  action-period source lesion. The evaluation-period source trace remains
  large in norm here, but its direct W1 component is weak and does not predict
  its final causal effect.

### 6. Repeated question stem

- **Both conditions:** no special semantic-winner lookup has been localized to
  the question words themselves.
- **Information sent onward:** at final-decision Mixer 56, the aggregate region
  initially called “repeated question/option-boundary states” has a large
  condition difference, but token decomposition shows almost none comes from
  the question words. It comes mainly from the newlines after repeated options
  B and C, which summarize the preceding option sequence.
- **How known:** exact per-token Mixer-56 source decomposition.

### 7. Each repeated option line

- **Both conditions, movement:** a separate receiver-edge experiment keeps the
  original K/V natural and blocks only attention from a repeated option line to
  its matching original option line. Blocking that edge jointly over **4--48**
  is causal. Layers 4--48 were not tested one at a time. An automatic
  discovery screen based on source-specific projected-write magnitude/onset,
  not raw attention mass, selected only late repeated-W1 receiver edges at
  **52, 60, and 64** for individual tests; those were null. The causal layer or
  smaller band inside 4--48 is therefore unknown.
  A complete-range follow-up now shows that layers **52--64 alone** have only
  approximately 0.01-logit effects and no reliable W1-choice effect. Extending
  the all-candidate blockade from **4--48** through **64** changes every
  held-out R1--R4 effect by approximately 0.01 logits (maximum 0.0101). This localizes the necessary
  original-line -> repeated-line route to the joint **4--48** range, although
  it does not isolate a single layer inside that range. The original restricted
  fixed-A serial-mediation experiment appeared to assign part of selected-line
  donor transfer to the matching repeated receiver. That result was reversed by
  the exact full24 fixed-A recalibration and did not generalize to B--D: the
  token-count-matched nonmatching receiver removed at least as much donor
  transfer. Matching-specific mediation of the selected-line transplant is
  therefore not established. Natural matching attention rises from about 6%
  at **4** to
  12--15% by **12**, stays elevated through **44**, peaks around **48--52**, and
  drops sharply at **56**.
- **Neutral use of the retrieved history:** the intact match supports every
  candidate in first-pass rank order: R1 most, then R2, R3, and R4. Lesion
  effects are -1.279, -0.816, -0.681, and -0.533 centered logits; negative
  means removing the match harms that candidate.
- **Game use:** the intact match suppresses R1/W1 and R2, is approximately
  neutral for R3, and supports R4. The corresponding lesion effects are
  +0.343, +0.160, +0.011, and -0.160 logits.
- **Rank information at this position:** the condition difference changes
  nonlinearly with graded first-pass score and competitor gap. A linear model's
  apparent extra W1 term disappears under the flexible nonlinear audit; no
  additional categorical winner state is established here.
- **Selected-line transplant mediation:** under the matched full24 fixed-A
  pipeline, the held-out matching-specific estimate is **-0.398 logits**; the
  nonmatching receiver removes more donor transfer than the matching receiver.
  B--D likewise show no matching-specific mediation. The natural matching edge
  remains causal for ordinary history retrieval, but it has not been shown to
  be the unique route of the counterfactual selected-line donor effect.
- **Layer-specific activation:** through **48**, R1--R4 receive similar
  absolute matching attention. From **52 onward**, both conditions modestly
  prefer R1, but that preference is smaller in Game; the relative Game
  attenuation is already detectable at **48**. This is descriptive, not the
  causal layer localization. The complete-range causal test shows explicitly
  that the late 52--64 attention is not an additional necessary route.
- **How known:** matching-edge lesions against cyclic nonmatching controls,
  matched-pipeline fixed-A/B/C/D donor-receiver comparisons, joint all-candidate
  blockade, regression on first-pass evidence,
  and complete natural-attention trajectories. Joint blockade eliminates the
  held-out discrete W1-choice gap and removes 49% of the continuous margin gap.

### 8. Post-option choice cue and final assistant prefix

- **Causal status of the exact cue-space token:** later tokens do use its
  condition-specific summary. Reciprocal Game/Neutral cue-memory swaps move the
  final task-specific ranking 10.6--14.8% toward the donor across both frozen
  splits. Whether the final computation requires this token is open: the
  historical cue/colon lesions removed only GLA writes because their
  ordinary-attention mask edit was a no-op. The cue is causally used; its
  redundancy awaits the corrected complete ablation.
- **Information arriving here:** later pre-final queries read the complete
  contextualized repeated-option states. Blocking all such reads from repeated
  W1 lowers W1 choice by 12.5 points in Game and 30.9 in Neutral, so this route
  is net pro-W1 in both and much stronger in Neutral.
- **Information sent to the final decision:** at **Mixer 52**, important source
  differences include the second choice cue (-0.038 W1 units), final assistant
  prefix (-0.031), option-boundary states (-0.046), and repeated W1 (-0.022).
  At **Mixer 56**, the choice cue contributes -0.314, option-boundary states
  -0.407, and repeated W1 -0.226 to the Game-minus-Neutral W1 write. Most of the
  cue contribution is stored at its final space token, a contextual summary
  position.
- **How known:** comprehensive repeated-W1 relay lesions and exact additive
  per-token/per-head source decomposition.

### 9. Final decision position

- **Both conditions:** direct attention from this position to the original W1
  option line is causally null. Direct attention to the repeated W1 line is
  pro-W1 in both conditions, but stronger in Neutral: blocking it changes the
  W1--W2 margin by a W1-line-minus-other-lines **contrast** of -0.135 in Game
  and -0.320 in Neutral (raw W1-line lesion effects -0.098 and -0.259).
- **Information appearing by layer:** the complete Game-minus-Neutral residual
  shows revision vocabulary at **18--29**. The answer-aligned trace sourced
  specifically from the earlier evaluation period begins near **33**.
  Individual GLA contextual changes express incorrect/rejection at **42**,
  replacement at **43**, and retry/override at **47**. The first replicating
  negative W1--W2 GLA changes occur at **49 and 53**. The complete residual's
  W1 difference becomes practically large at **52--54** and takes its largest
  step at **Mixer 56**.
- **Game computation at 52--56:** Mixer 52 lowers W1 from +0.156 to +0.027.
  Mixer 56 then boosts it from +0.219 to +0.448.
- **Neutral computation at 52--56:** Mixer 52 raises W1 from -0.007 to +0.150.
  Mixer 56 boosts it from +0.600 to +1.795. Thus Mixer 56 creates the largest
  condition divergence through much stronger Neutral reinstatement, not a
  negative Game write. MLP 56 partially reverses the divergence.
- **Source positions at Mixer 56:** repeated option-boundary summaries, the
  second choice cue, and repeated W1 dominate. The repeated-W1 contribution is
  pro-W1 in both conditions (+0.449 Game, +0.675 Neutral). Literal feedback
  tokens are not important late sources.
- **How known:** complete-residual JLens/logit-lens trajectory, within-block
  mixer/MLP decomposition, exact source contributions, original-line final-edge
  null, and repeated-line final-edge lesion.

### What remains unlocalized

The experiments now localize graded old-rank transfer to matching-line
ordinary-attention writes at layers 32--48 and the strongest replicating
task-dependent rank reshaping to MLP 49 at the final 2P semantic token. A
completed serial factorial now shows that the evaluation-period GLA update
causally changes how the matching route uses rank and changes W1 choice in both
tasks. Restoring natural MLP-49 output at the four final semantic tokens leaves
94--97% of the final policy effect, so MLP 49 is a strong additive readout but
not the necessary local mediator.

A new all-layer causal removal test now narrows the role of fresh 2P evidence.
It removes more than 98% of the frozen decoded fresh-score component unique of
old score from all second-option semantic wordpieces and newlines, while
restoring the old-score coordinate and matching the intervention dose with
random orthogonal directions. On held-out canonical conflicts, the targeted
scrub changes the Game-minus-Neutral old-W1-avoidance gap by -5.1 points
[-13.2, +2.9] relative to random, whereas matching-history blockade changes it
by -24.3 [-34.6, -14.7]. At score resolution, however, fresh removal reduces
the Game-minus-Neutral old-W1 suppression gap by +0.055 logits [+0.025,+0.084]
on the complete confirmation split and +0.041 [+0.002,+0.080] on conflicts,
against natural gaps of -0.521 and -0.522. The decoded fresh coordinate is
therefore not purely task-shared: it makes a small but statistically reliable
policy-conditioned contribution (about 10% and 8% of the natural score gaps).
It is not necessary for the existence of the choice-level preferential-
switching effect, while matching recollection is much larger and necessary at
that endpoint. This bounds the tested decoded subspace; it does not eliminate
every distributed form of recomputation.

## How to read the evidence

The account uses four distinct kinds of claim:

- **Behavior:** prompt manipulation changes answers or output logits; this
  establishes what computation must be explained, not where it happens.
- **Activation observation:** attention, lens, or probe measurements correlate
  with the computation; these localize or characterize a signal but do not by
  themselves establish causal use.
- **Causal mechanism:** patching, transplanting, or blocking an internal state
  changes the relevant output with frozen discovery/confirmation controls.
- **Inference:** the simplest joint interpretation of separate results. These
  claims are explicitly weaker than a serial mediation experiment.

## Evidence map

| Claim | Evidence type | Quantitative anchor | Canonical evidence |
|---|---|---|---|
| Game follows the old answer's semantic content, not its old letter | Behavior | Game--Neutral content switching +18.8 points; old-letter avoidance -9.0 points; content suppression 0.633 logits versus old-letter suppression 0.139 | [Remapping behavior](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/analysis/REPORT.md#behavioral-results) |
| `incorrect` supplies most targeted revision | Behavior | With action fixed, W1 avoidance +15.2 points and W1 centered logit -0.516; 81% of the full behavioral contrast | [Feedback factorial](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/analysis/REPORT.md) |
| The minimal effect survives without remapping | Behavior | `incorrect` versus `lost`, with both saying `again`, raises switching by 8.8 points on unchanged-order SimpleMC | [Non-remapped report](outputs/prompt_variant_tests/qwen36_27b_simplemc_action_matched_nonremapped/analysis/REPORT.md) |
| The evaluation-period GLA update seeds revision and answer-targeted redistribution | Output-preserved causal transplant and exact source trace | Corrected reciprocal transplant transfers 0.097/0.091 logits of a 0.469-logit margin gap; corrected deletion removes 30.0% [9.5, 50.0%] of the behavioral gap and 0.097 logits [0.059, 0.135] of the margin gap. Blocks 25--32 carry 58.4% of the recurrent route | [Update transplant](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_update_transplant/output_preserved/analysis/REPORT.md), [source-trace report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_period_source_trace/output_preserved_analysis/REPORT.md), [source-trace figure](figures/qwen36_evaluation_period_source_trace.png) |
| Feedback policy is transmitted by a distributed L28--50 sequence, with reproducible maxima at L36 and L45 | Complete L1--64 causal sufficiency and necessity | On confirmation, L36 alone transfers 0.130/0.119 of donor policy into Game/Neutral and changes the bivalent rank score -0.087/+0.056; L45 transfers 0.070/0.068 and changes it -0.036/+0.032. Sufficiency, necessity, direction, and peak order replicate on discovery | [Individual-layer report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/policy_source_layer_decomposition/individual_all_layers_analysis/REPORT.md), [figure](figures/qwen36_feedback_policy_individual_layers.png) |
| Feedback sentence and following identical instruction carry overlapping policy state | Output-preserved grouped source crossover | On confirmation, the feedback sentence transfers 0.643/0.665 into Game/Neutral and the following instruction transfers 0.450/0.575; complete-minus-sum is -0.168/-0.298, so neither source is complete and their effects are strongly subadditive. Both tasks and discovery replicate | [Grouped-source report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/evaluation_relay_final_mediation/source_group_crossover/analysis/REPORT.md), [figure](figures/qwen36_feedback_source_groups.png) |
| The later action-closing period is a secondary causal source, not the main policy bottleneck | Output-preserved causal localization | Blocking later ordinary-attention reads changes Neutral conflict margin by -0.033 logits and Game no-conflict W1 advantage by +0.048, with split replication; its own GLA write and all pooled W1-selection effects remain small | [Action-period source-lesion report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/action_period_source_lesion/analysis/REPORT.md) |
| Original option lines carry semantic history | Causal mechanism | Fixed-A selected-line K/V transplant moves donor-semantic margin by 0.558 logits in Game and 3.161 in Neutral; boundary and non-option spans are near zero | [Source-localization figure](figures/qwen36_fixed_a_kv_source_localization.png) |
| Counterfactual selected-line semantics generalize asymmetrically beyond A | Causal mechanism | Held-out B/C/D donor transfer is +1.927/+1.475/+1.058 logits in Neutral but +0.290/+0.009/-0.128 in Game; fixed-A matching-specific mediation does not generalize | [Fixed-B/C/D report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_bcd_line_generalization/mediation_analysis/REPORT.md), [figure](figures/qwen36_fixed_bcd_selected_line_semantic_transfer.png) |
| Repeated option lines semantically read matching original lines | Causal mechanism | Blocking original-W1 -> repeated-W1 changes held-out W1 choice +10.3 points in Game and -29.4 in Neutral | [Receiver-path figure](figures/qwen36_remapped_receiver_edge_validation.png) |
| Matching-specific mediation of selected-line transplantation is not established | Matched-pipeline causal null | The exact full24 fixed-A held-out matching-specific estimate is -0.398 logits, and B--D also fail the matching-specific mediation gate. This does not negate the causal natural-history matching edge | [Fixed-A recalibration](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_a_full24_calibration/analysis/REPORT.md), [B/C/D generalization](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fixed_bcd_line_generalization/mediation_analysis/REPORT.md) |
| Neutral reinstates old candidates in rank order | Causal mechanism | Blocking each matched relay changes that candidate's held-out centered advantage by R1 -1.279, R2 -0.816, R3 -0.681, R4 -0.533 logits. Negative lesion effects mean the intact match had supported the candidate | [All-candidate figure, panel A](figures/qwen36_all_candidate_matched_relay.png) |
| Game redistributes away from the old leaders | Causal mechanism | The same controlled lesions change candidate advantage by R1 +0.343, R2 +0.160, R3 +0.011, R4 -0.160 logits. Thus the intact Game match suppresses R1/R2, is neutral for R3, and supports R4 | [All-candidate figure, panel A](figures/qwen36_all_candidate_matched_relay.png) |
| Game directly compresses its old semantic ranking | Post hoc within-Game natural-logit analysis plus held-out causal lesions | From first presentation to Game's final answer, mean A-D probabilities change from 58.6/24.0/12.0/5.3% by old rank to 31.5/25.5/22.9/20.1%; matching-history lesions establish that semantic history causally constructs part of this rank-dependent change | [Strategic-switching report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/strategic_switching_evidence/analysis/REPORT.md) |
| The decoded unique-fresh 2P subspace makes a small policy-conditioned logit contribution but is not necessary for the choice-level preferential-switching effect | All-layer residual removal with old-coordinate preservation and same-dose random controls | Fresh-vs-random changes the Game-minus-Neutral W1-avoidance choice gap by -5.1 points [-13.2,+2.9], but reliably reduces differential old-W1 suppression by +0.055 logits [+0.025,+0.084] on the full confirmation split and +0.041 [+0.002,+0.080] on conflicts. Matching recollection has a much larger causal effect | [Fresh × history report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/fresh_history_double_dissociation/analysis/REPORT.md), [figure](figures/qwen36_fresh_history_double_dissociation.png) |
| The two matching policies differ nonlinearly with first-pass rank | Statistical comparison of causal effects | Their held-out Game-minus-Neutral differences are R1 1.622, R2 0.976, R3 0.692, R4 0.374 logits. A linear model suggested a separate W1 increment, but that term includes zero on both splits after flexible control for candidate score and competitor gap | [All-candidate figure, panel B](figures/qwen36_all_candidate_matched_relay.png), [nonlinear audit](figures/qwen36_categorical_winner_nonlinearity_audit.png) |
| Game selectively attenuates the W1 matching read from L48 onward | Activation observation | R1-specific Game attenuation is -0.37, -0.53, -0.23, -0.18, and -0.18 attention percentage points at L48, L52, L56, L60, and L64; every paired interval excludes zero | [Direct attention figure](figures/qwen36_late_winner_attention_attenuation.png) |
| Repeated option lines read the whole first-pass candidate set, boundary, and policy sentence | Activation observation | Across L12--52, all 1P option lines receive 33--41% attention: 13--16% to the semantic match and 20--25% to the other three. At L36, Game reads feedback at 4.15% versus 1.07% in Neutral | [Exhaustive source map](figures/qwen36_second_presentation_attention_distribution.png), [report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_attention_distribution/analysis/REPORT.md) |
| Matching 1P lines write graded old rank into 2P semantic tokens | Exact additive source attribution | Matching-line/old-score correlation rises from 0.265/0.267 in Game/Neutral at L32 to 0.401/0.399 at L48. At L32 the exact R1--R4 writes are nearly identical: Game +0.131/-0.039/-0.224/-0.400; Neutral +0.131/-0.040/-0.224/-0.396 | [Score-source report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/score_source_attribution/REPORT.md), [figure](figures/qwen36_second_presentation_score_source_attribution.png) |
| The first strongly localized task-dependent old-rank write is at MLP 49 | Exact additive component attribution, not yet a lesion | Absolute R1--R4 writes are Game +1.177/+0.639/+0.410/-0.263 and Neutral +1.386/+0.685/+0.292/-0.432. The held-out Game-minus-Neutral bivalent shift, R4 minus mean(R1,R2), is +0.296 [+0.223,+0.371] | [Score-source report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/score_source_attribution/REPORT.md), [figure](figures/qwen36_second_presentation_score_source_attribution.png) |
| The evaluation-period update causally controls matching-route rank use | Non-output-preserved serial causal factorial | Swapping Neutral's complete period update into Game changes the held-out bivalent route effect +0.426 logits and raises conflict W1 choice +15.4 points; swapping Game's update into Neutral changes the route effect -0.727 and W1 choice -19.1 points. Because source output was not preserved, this tests the complete period update, not persistent memory alone | [Policy × rank report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/policy_rank_factorial/analysis/REPORT.md), [figure](figures/qwen36_policy_rank_factorial.png) |
| MLP 49 is not the necessary local mediator | Causal restoration null | Restoring natural MLP-49 output at all four final 2P semantic tokens removes its local policy projection, but leaves 94% of Game's and 97% of Neutral's final bivalent effect and 86%/88% of the Game/Neutral behavioral effects | [Policy × rank report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/policy_rank_factorial/analysis/REPORT.md), [figure panel C](figures/qwen36_policy_rank_factorial.png) |
| Direct reads of the other three 1P lines carry shared rank evidence, not the task-specific policy | Causal mechanism and causal null | Blocking all three nonmatches while preserving each semantic match changes the held-out within-task rank slope +0.093 in Game and +0.089 in Neutral, but changes the Game-minus-Neutral policy slope only +0.004 [-0.021, +0.029] | [Nonmatching-history report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/nonmatching_history_factorial/analysis/REPORT.md), [figure](figures/qwen36_nonmatching_history_factorial.png) |
| The W1 semantic match has opposite causal consequences | Causal mechanism | Blocking W1's match raises W1 evidence +0.362 logits in Game but lowers it -1.178 in Neutral | [All-candidate report: direct W1 effect](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/REPORT.md#direct-w1-matching-edge-effect-versus-natural) |
| The repeated W1 line as a whole remains pro-W1 | Causal mechanism | Blocking only the final W1-line read lowers W1--W2 margin by 0.135 logits in Game and 0.320 in Neutral; blocking all pre-final reads lowers it by 0.62 and 1.01 | [Final-query figure](figures/qwen36_final_query_repeated_option_ablation.png), [pre-final relay figure](figures/qwen36_repeated_w1_relay.png) |
| Semantic matching explains preferential switching | Causal mechanism | Joint relay blockade moves held-out conflict W1 choice from 17.6%/39.7% in Game/Neutral to 27.2%/27.2%; the gap reduction is 22.1 [11.8, 32.4] points and margin mediation is 49% | [All-candidate report: joint mediation](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/REPORT.md#joint-mediation-on-held-out-conflict-trials) |
| W1 demotion becomes explicit late at the final decision | Activation/logit trajectory | At L64, Game--Neutral is -0.476 logits for W1, -0.065 for W2, and +0.270 for the other two candidates | [Layerwise W1 figure](figures/qwen36_simplemc_remapped_w1_layerwise.png) |
| A policy-adjusted answer pattern is linearly present before it is output-readable | Held-out all-question activation decoding in the clean non-remapped prompt | The question-specific Game-minus-Neutral pattern becomes stably decodable at L33 on SimpleMC and L32 on TriviaMC. At L40 learned cosine is 0.311/0.520 while fixed JLens is 0.006/-0.035. Exact final rank effects are SimpleMC R1/R2/R3/R4 -0.532/+0.001/+0.181/+0.350 and TriviaMC -1.600/+0.392/+0.604/+0.604 | [Policy-adjusted report](outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/prospective_decoding/policy_analysis/REPORT.md), [SimpleMC figure](figures/prospective_decoding/qwen36_simplemc_policy_adjusted_prospective_decoding.png), [TriviaMC figure](figures/prospective_decoding/qwen36_triviamc_policy_adjusted_prospective_decoding.png) |
| Late sharpening is attenuated in Game | Causal mechanism under the broader canonical prompt | Removing Mixer 56 reduces winner advantage by 0.043 logits in Game versus 0.132 in Neutral; Neutral-minus-Game sharpening difference 0.089 [0.074, 0.105] | [Mixer-56 report](outputs/causal/qwen36_27b_mixer56_across_conditions/analysis/REPORT.md) |
| The empty first-answer boundary is not the semantic-memory route | Causal crossover and causal scrub | Boundary crossover makes the immediate decision match the donor on 98.9%, but leaves only +0.087/+0.077 centered-logit evidence for the donor's old literal letter at the final answer and changes final donor choice by only +0.5/-1.1 points in Game/Neutral; it does not transfer the semantic winner. Letter scrub explains 0% of the held-out choice gap | [Boundary crossover](figures/qwen36_first_decision_boundary_crossover.png), [letter scrub](figures/qwen36_first_decision_letter_scrub.png), [source localization](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/semantic_binding_fixed_a_kv_source_transplant/analysis/REPORT.md) |

## The two tasks, described separately

The cleanest controlled quantity here is the effect of blocking one candidate's
true semantic matching edge, relative to blocking an equally sized cyclic
nonmatching source. The outcome is that candidate's centered logit advantage.
A **negative lesion effect** means that the intact semantic match normally
supported the candidate; a **positive lesion effect** means that the intact
match normally pushed the candidate down.

| First-pass rank | Neutral lesion effect | What intact Neutral match does | Game lesion effect | What intact Game match does |
|---|---:|---|---:|---|
| R1 / W1 | -1.279 [-1.456, -1.107] | Strongly reinstates the old winner | +0.343 [+0.182, +0.502] | Suppresses the old winner |
| R2 | -0.816 [-0.935, -0.699] | Reinstates R2 | +0.160 [+0.032, +0.292] | Suppresses R2 more weakly |
| R3 | -0.681 [-0.774, -0.590] | Reinstates R3 | +0.011 [-0.092, +0.120] | Approximately neutral |
| R4 | -0.533 [-0.609, -0.459] | Reinstates R4 least strongly | -0.160 [-0.246, -0.069] | Supports the old weakest candidate |

### Neutral computation

Neutral reads each repeated option's semantic history and adds old evidence
back into that candidate. It does this for all four candidates, but in a graded
way: the previous winner receives the largest reinstatement, followed by R2,
R3, and R4. Downstream reads of the complete repeated W1 line are also strongly
pro-W1. Cognitively, Neutral combines a fresh solve with recovery of the old
candidate ranking, especially its winner.

### Game computation

Game also performs the semantic matches, but does not merely turn Neutral's
reinstatement off. The controlled causal effects change sign across old rank:
the matches push W1 and R2 down, have little net effect on R3, and support R4.
This is a nonlinear rank-reversing redistribution of retrieved history. Game
selectively reduces W1's matching attention from L48
onward. The complete repeated W1 state nevertheless remains net pro-W1 because
it also contains current candidate evidence; Game uses that downstream
reinstatement substantially less than Neutral.

A focused within-Game synthesis makes the operation concrete. From the first
decision to the final Game decision, the same semantic candidates' raw logits
change by R1 -2.114, R2 -1.181, R3 -0.410, and R4 +0.632. Their mean
per-question A-D probabilities change from 58.6/24.0/12.0/5.3% to
31.5/25.5/22.9/20.1%. This is aggressive old-rank compression, not equal noise
on four answers. The before/after profile is descriptive; the matching-edge
lesions supply the causal evidence that Game constructs part of it from
semantic history. See the [focused strategic-switching report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/strategic_switching_evidence/analysis/REPORT.md).

Leaving the old winner and selecting the next candidate are separable. On the
135 questions where the standalone second-presentation winner differs from
both the old winner and old runner-up, natural Game switches choose the fresh
winner 65.8% of the time and the old runner-up 22.5%; the preference replicates
on discovery and confirmation. Under the joint matching-history blockade the
same conditional difference shrinks from +43.2 to +6.6 points and its interval
spans zero, so the stored blockade does not show that strong fresh-winner
steering survives independently of history. Separately, crossing complete 2P
option-line state between paired orderings redirects discrete choices toward
the donor ordering beyond two generic-drift nulls on both frozen splits. Thus
current-presentation state causally structures the destination, but its
necessity remains untested. This destination analysis uses no correctness
labels.

### Comparison only after the two descriptions

The Game-minus-Neutral values summarize how far apart these two causal policies
are; they are not the primary cognitive description. Their decrease from R1
to R4 is what the earlier phrase “continuous gradient” referred to. A
linear regression initially suggested a +0.205-logit increase per unit of
first-pass candidate evidence and an additional W1 term. The nonlinear audit
shows that the separate W1 term is not robust once the candidate's gap to its
best competitor is modeled flexibly. The supported comparison is therefore a
graded but nonlinear dependence on prior rank, not a demonstrated categorical
winner state.

### 1. First pass: construct candidate semantics and preferences

**Evidence status: causal semantic localization plus correlational decoding.**

The model processes the original question and four options, producing a
preference ordering. The prompt contains no visible historical answer token,
so later behavior cannot be explained by copying a displayed answer.

The original option-closing states contain two kinds of recoverable
information:

- **Semantic identity is causal.** Transplanting conventional-attention K/V
  from a selected original option line transfers the corresponding semantic
  answer downstream.
- **Candidate value is decodable.** A held-out linear ranker at the option
  newline predicts the first-pass winner at 64.9%, above the 51.9% majority-A
  baseline. But removing that fitted one-dimensional direction does not
  reliably alter preferential Game revision, so it is a correlate rather than
  the identified causal winner code.

The all-candidate lesion shows that the later matching circuit is sensitive to
graded first-pass evidence in a strongly nonlinear way. It preferentially
affects the old winner behaviorally, but the present data do not require a
separate portable categorical `winner bit`.

### 2. Feedback: construct a revision policy

**Evidence status: behavioral sufficiency plus causal state transplant and
deletion.**

Behaviorally, the evaluation word is the principal policy signal. Holding the
action clause fixed at `Choose the answer again`, changing `lost` to
`incorrect` increases W1 avoidance by 15.2 percentage points and lowers W1's
centered logit by 0.516. This reproduces about 81% of the full standard-Game
W1-avoidance difference. The explicit `different` instruction adds only 3.4
points of W1 avoidance under `lost`.

This is not a perfectly surgical instruction: `incorrect` also raises A--D
entropy by 0.136 bits. Cognitively, Game both revises the old answer and becomes
less certain more broadly.

Mechanistically, the period ending the evaluation sentence writes a distributed
Gated DeltaNet state. In the corrected output-preserved interventions, reciprocal
transplant transfers 0.097/0.091 logits of the natural 0.469-logit W1--W2 margin
difference, and deletion removes 30.0% [9.5, 50.0%] of the behavioral gap plus
0.097 logits [0.059, 0.135] of the continuous margin gap. Blocks 25--32 carry
58.4% of the all-GLA recurrent route. No individual GLA is an indispensable
bottleneck. The much larger historical estimates also broadcast donor state
through the source token's own residual and are superseded for route-specific
attribution.

Calling this only a `policy` state is too weak. Exact deletion source tracing
shows that the period write's causal contribution at the final decision separates
W1 from W2 across the middle layers and has its largest retrieval-norm
differences at blocks 49, 33, and 47. Its cumulative Evaluation-minus-Neutral contribution
ultimately favors W2 over W1 by 0.0089 logits before downstream amplification.
Complete-state lenses separately show the instruction becoming readable at
18--29; GLA-specific contextual changes then become readable as
incorrectness/rejection at 42, replacement at 43, and retry/override at 47;
the first replicating behavior-aligned W1-minus-W2 movements occur at 49 and
53. Thus the evaluation-period update is the causal source of a staged,
answer-targeted transformation, not merely a generic instruction flag.

The remaining qualification matters: source tracing follows what later queries
do with the period write. It does not distinguish a locally bound `W1 is wrong`
record from a content-free revision operator that becomes answer-specific only
when later combined with semantic history.

The later period ending `Choose the answer again.` is a secondary causal source,
not the main bottleneck. Its own preserved GLA write is small. Blocking its
ordinary-attention reads changes Neutral conflict margin by -0.033 logits and
Game no-conflict W1 advantage by +0.048, but pooled W1-selection effects remain
uncertain and the joint lesion does not reproduce the task gap.

The complete feedback suffix nevertheless continues to transmit policy through
both architectures. Crossing all of its downstream writes transfers 92.5% of
the paired Neutral state into Game and 94.1% of the paired Game state into
Neutral on confirmation. The grouped source crossover shows how that effect is
distributed across sentences. On confirmation, `incorrect/lost .` alone
transfers 0.643 into Game and 0.665 into Neutral (69.5%/70.6% of the complete
suffix). The following, lexically identical `Choose the answer again .` alone
transfers 0.450/0.575 (48.7%/61.0% of complete). Its causal state therefore
depends on the preceding feedback even though its text is shared. The isolated
effects are not additive: their sum exceeds the complete suffix by 0.168 in
Game and 0.298 in Neutral, with paired confidence intervals excluding zero and
the same result on discovery. The two spans therefore carry overlapping or
redundant policy information; neither span by itself is the complete source.

Exhaustive L1--64 sufficiency and all-except-layer
necessity tests identify the same two individual maxima in both tasks and both
frozen splits: ordinary-attention **L36** first and GLA **L45** second. The
other practically important individual effects form a broad L28--50 cluster.
No individual layer transfers most of the policy; this is a distributed route,
not a single policy bottleneck.

The corrected evaluation-to-final relay mediation gives a lower-bound map of
the downstream path. Nominally, restoring every later post-feedback token's
outgoing ordinary-attention and recurrent-GLA writes removes 51.8% [49.4,
54.0%] of Game-recipient transfer and 41.4% [38.6, 44.4%] of Neutral-recipient
transfer on confirmation, with 53.8% and 43.6% discovery replication. Each
instruction, question, option-line, choice-cue/query, and assistant-prefix
region carries a replicated portion; none is the unique bottleneck. The
restoration-only control is bit-exact, but the downstream-only operation keeps
each restored token's source-crossed local output. Because it also omits the
short causal GLA q/k/v convolution, source-crossed prefix output immediately
before the readout can leak through that channel. The joint percentages are
therefore lower bounds, and their surviving fractions do not establish a
physiological bypass split among direct attention and recurrent memory.

### 3. Second pass: semantically match every repeated option

**Evidence status: causal attention-edge interventions, with a complementary
natural-attention observation.**

When the same content reappears under a new A--D letter, tokens on that repeated
option line read the matching original option line using ordinary attention
across blocks 4--48. This is semantic, not positional or letter copying.
Blocking the original-W1 to repeated-W1 read has large, opposite effects in
Game and Neutral, including when the semantic content moves to a different
letter.

The all-candidate factorial establishes what the matching operation computes.
Here, **first-pass candidate evidence** means that candidate's centered A--D
logit in the standalone first presentation: how strongly the model preferred
it relative to the other three before any feedback. It is not a discovered
activation direction. After first measuring the separate Neutral and Game
lesion effects above, we regress their difference on that pre-feedback logit:

- a linear model gives a +0.205-logit increase per unit of centered first-pass
  evidence, but that linearity assumption is inadequate;
- after flexible control for both score and the gap to the best competitor, the
  apparent extra W1 term is -0.161 [-0.434,+0.098] in discovery and +0.183
  [-0.115,+0.458] in confirmation, and it worsens held-out prediction slightly;
- W1's matching edge contributes +0.362 logits against W1 in Game when removed
  (so the intact edge was anti-W1), but its removal lowers W1 by 1.178 logits in
  Neutral (so the intact edge was pro-W1).

Natural attention shows the corresponding allocation policy. Beginning at
L48, Game selectively attenuates the matched-line read for R1/W1 relative to
R2--R4. The R1-specific Game attenuation is -0.37 percentage points at L48,
-0.53 at L52, -0.23 at L56, and -0.18 at L60; every held-out paired interval
excludes zero. The output endpoint adds no mechanistic localization.

The complete source distribution changes the interpretation of “matching
read.” A repeated option line does not read only its matching original line.
On held-out questions, the four first-presentation option lines jointly receive
**35.6%** of its attention at layer 12, **33--37%** at layers 28--44,
**40%** at layer 48, and **37%** at layer 52. The matching line accounts for
about **13--16%** over this interval; the other three original lines together
account for **20--25%**. The first-answer cue and boundary receive another
**6--9%** through most of layers 4--60. Therefore the repeated line has the
inputs needed to compare its semantic match against the whole first-pass
candidate set, rather than receiving an isolated content copy. The causal
factorial shows that these nonmatching reads do contribute ranking information:
removing all three shifts both Game and Neutral away from R1 and toward R3/R4.
However, it leaves the Game-minus-Neutral rank policy unchanged. The direct
reads of the other three lines are therefore a shared evidence-comparison
route, not the source of the condition-specific rank transformation.

The same map identifies a separate policy read. At layers 28, 36, 44, and 48,
Game assigns **3.43%, 4.15%, 3.98%, and 2.19%** of repeated-line attention to
the `incorrect` sentence; Neutral assigns **1.59%, 1.07%, 0.95%, and 0.57%**
to the `lost` sentence. This is direct activation evidence that the 2P option
computation reads policy and candidate history together. It does not establish
which nonmatching candidate or boundary feature contains winner rank. The new
factorial establishes causal necessity for the aggregate nonmatching set's
shared rank effect, but not for the Game-specific policy.

Exact source attribution now identifies what the matching-line read carries
inside the 2P semantic residual. In **Game**, the correlation between the
matching 1P line's exact attention write and old score is 0.265 at layer 32,
0.342 at 36, 0.370 at 44, and 0.401 at 48. In **Neutral**, the corresponding
values are 0.267, 0.336, 0.371, and 0.399. At layer 32, the exact normalized
writes for R1 through R4 are `[+0.131,-0.039,-0.224,-0.400]` in Game and
`[+0.131,-0.040,-0.224,-0.396]` in Neutral. Thus both tasks initially write
essentially the same graded first-pass ranking from each matching 1P line into
the corresponding 2P semantic tokens. This is stronger than the attention map:
it identifies the information carried by the read, not merely its weight.

The receiver-token and downstream-relay factorials now make this a serial
causal account rather than an isolated source attribution. Across all 32
subsets of the five physical 2P option-line token classes, the semantic
wordpieces are the dominant entry route in both tasks: blocking only their
matching-line reads nearly reproduces the complete matching-edge lesion, and
opening only them from the all-closed state recovers most of the route.
Newlines are a smaller secondary entry route; leading spaces, option letters,
and colons are individually small. Balanced wrong-line lesions at identical
receiver tokens are much smaller, ruling out lesion size as the explanation.

After entry, the same semantic positions are the strongest single downstream
relay. The original analyzer used Neutral's final answers instead of the
canonical remapped-baseline answers to define W2, so its 234-question subset
was Neutral-switching rather than W1!=W2 conflict. The corrected analyzer
validates prompt provenance and uses all 273 canonical conflicts. On
confirmation, restoring semantic outgoing ordinary-attention and recurrent-GLA
state while the matching source lesion remains active recovers **59.0%**
`[54.1%,64.4%]` of the lesioned R1--R4 history vector in Game and **61.9%**
`[55.4%,68.8%]` in Neutral. Newline, option-structure, cue/query, and
assistant-prefix singles recover 37.6%, 31.7%, 25.0%, and 16.5% in Game and
39.5%, 40.1%, 31.9%, and 19.0% in Neutral. Discovery independently preserves
the route ordering.

Restoring all four groups before the final assistant prefix recovers **94.1%**
`[91.3%,96.5%]` in Game and **94.1%** `[92.6%,95.6%]` in Neutral. The completed
convolution-safe control independently sharpens this estimate. The nominal
all-five cells recover only 36.8% and 48.4% because restored prefix tokens keep
their lesioned local outputs beside the readout. Leaving the final four prefix
tokens free while restoring the rest of the five-region tail raises recovery
to **97.7%** `[96.3%,99.1%]` in Game and **96.5%** `[95.7%,97.4%]` in Neutral
on confirmation; discovery independently gives 97.9% and 96.3%. All natural,
restoration-only, and shared-cell reproduction errors are exactly 0.0. The
previous collapse is therefore an intervention-boundary artifact from the
unintercepted multilayer GLA convolution, not antagonistic prefix physiology.
The five-region causal-tail inventory accounts for essentially the whole
measured candidate-history path. The ordinary-only and GLA-only nominal joint
values remain non-additive lower bounds because convolution was not separately
intercepted.

Fresh 2P score has no comparable single attention source. Its clearest
component writes are distributed MLP computations at layers **29--31**, with
positive fresh-score writes continuing through the 40s. This is additive
source evidence that the 2P semantic residual contains both retrieved old rank
and newly computed current evidence; it is not yet a causal decomposition of
their nonlinear combination.

The strongest localized, held-out-replicating task-dependent old-rank write is
made by **MLP 49** at the final semantic token. In Game its absolute R1--R4
writes are `[+1.177,+0.639,+0.410,-0.263]`; in Neutral they are
`[+1.386,+0.685,+0.292,-0.432]`. Both writes preserve old rank, but Neutral is
more top-heavy. Relative to Neutral, Game reduces R1/R2 reinstatement and
shifts support toward R3/R4; the held-out bivalent difference is +0.296
`[+0.223,+0.371]`. This locates a concrete task-dependent computation at layer
49. Because it is an exact component attribution rather than an intervention,
MLP 49's necessity for final switching remains to be tested.

No single tested attention layer is sufficient for the letter-general receiver
effect. In the sharper fixed-A source localization, layer 44 is the strongest
individual direct semantic reader, the 36--48 band is jointly sufficient, and
4--32 is required as nonlinear enabling context. Those causal results localize
the broader relay; the exact-write result above separately nominates MLP 49 as
the first concrete task-dependent writer.

### 4. Downstream: preserve candidate evidence while changing W1's status

**Evidence status: causal lesions for both the semantic-match component and
the complete repeated-option state; the additive decomposition below is an
interpretive model.**

The repeated W1 option line is not simply an anti-W1 representation. Blocking
later reads of the entire repeated W1 line lowers W1 in both conditions, and it
lowers W1 much more in Neutral. Likewise, blocking only the final query's read
of that line lowers the W1--W2 margin by 0.135 logits in Game and 0.320 in
Neutral. The repeated line therefore remains a net pro-W1 evidence source.

The two causal findings are compatible because they remove different things:

- blocking **original W1 -> repeated W1** removes the match-dependent
  modification, which is anti-W1 in Game and pro-W1 in Neutral;
- blocking **repeated W1 -> later computation** removes the repeated option's
  entire contextualized state, including its ordinary pro-W1 candidate
  evidence.

A useful cognitive decomposition is therefore:

`contextualized repeated W1 = current candidate evidence + retrieved history + policy-dependent winner modification`.

Neutral gives the retrieved-history/winner terms a large positive contribution.
Game weakens the read and adds a negative W1-specific modification, while the
underlying candidate evidence remains positive.

### 5. Final decision: express revision as answer redistribution

**Evidence status: activation/logit trajectory, supplemented by causal mixer
lesions from the broader canonical prompt.**

At the final decision position, W1-specific Game--Neutral separation becomes
large in the fixed output readout only late. It begins around readouts 52--54, takes its largest step near
56, and persists through the output. At readout 64, Game relative to Neutral is
-0.476 logits on W1, approximately null on W2, and +0.270 on the mean of the
other two candidates. This is a ranked redistribution away from W1, not merely
uniform noise.

Held-out prospective decoding in the clean non-remapped prompt shows that the
policy-adjusted pattern is present earlier in another linear basis. Across all
paired confirmation questions, the exact final question-specific
Game-minus-Neutral vector becomes stably decodable at L33 on SimpleMC and L32
on TriviaMC; at L40 learned cosine is 0.311/0.520 while fixed JLens remains
0.006/-0.035. The rank components already acquire the final strategic signs in
the low-to-mid 30s: R1 is reduced and lower old-ranked candidates are raised.
Thus late JLens onset marks output alignment and amplification, not the first
existence of policy-adjusted answer information. This is activation decoding,
not causal localization.

Game also exhibits broader answer-space contraction and weaker late
sharpening. Mixer 56 sharpens candidate differences in Baseline and Neutral but
does so much less in Game; it does not reverse sign into a special flattening
module. These late effects are compatible with the semantic-match mechanism,
but they have not been placed into one complete serial mediation model under
the minimal action-matched prompt.

## What the mechanism explains

The behavioral remapping result is semantic: Game suppresses the content that
won first, not merely its original A--D letter. Across all questions, Game
switches semantic content on 65.6% versus 46.8% for Neutral, while it actually
avoids the old literal letter less often than Neutral. This rules out a simple
letter-switch rule.

On held-out conflict trials, Game naturally chooses W1 on 17.6% and Neutral on
39.7%. Jointly blocking all four semantic matching relays moves both conditions
to 27.2%, so the in-sample point estimates coincide; the gap-reduction interval
is compatible with partial through over-elimination rather than proving an
exact population zero. Discovery
independently reduces its 15.3-point gap to 1.5 points. The continuous W1--W2
margin gap is only 49% mediated, so the matching relay explains the replicated
choice difference better than it explains every logit change.

In non-remapped SimpleMC, the same one-word `incorrect` versus `lost` contrast
raises switching by 8.8 points. The phenomenon therefore does not depend on
moving option letters; remapping is the diagnostic that reveals its semantic
content.

## What is established versus inferred

### Established causally

- The evaluation-closing GLA update carries a causal, distributed portion of
  the revision policy; the output-preserved persistent route explains about
  30% of the discrete task gap.
- Original option-line ordinary-attention K/V carries semantic history.
- Repeated option lines read their semantic matches in the original
  presentation.
- That matching computation depends nonlinearly on graded old evidence and has
  its largest effect on the old winner.
- The W1 match has opposite causal consequences under `incorrect` and `lost`.
- The combined matching relays explain essentially all of the discrete
  Game--Neutral W1-choice gap on held-out conflict trials.
- Reciprocal transplantation of the evaluation-period GLA update changes the
  matching route's rank dependence in opposite directions in Game and Neutral,
  and changes conflict-trial W1 choice by +15.4 and -19.1 points respectively.
  This historical transplant was non-output-preserved, so the causal object is
  the complete evaluation-period GLA update rather than persistent memory alone.
- The final direct read of repeated W1 is pro-W1, especially in Neutral; there
  is no simple final-token inhibitory edge.
- At the exact final answer position, paired Game/Neutral task state becomes
  causally effective around layer 48, reaches roughly one-third transfer over
  layers 52--60, and is 82--85% transferable by layer 63. Replacing all local
  sequence-mixer writes transfers essentially the complete donor task vector
  and switch behavior on both frozen splits; replacing all MLP writes does not.

### Strongly supported but not one serial causal proof

- Game's late W1-specific attention attenuation is part of the same revision
  policy. It is replicated observationally, while the broader matching-edge
  interventions are causal.
- The matching 1P line supplies graded old rank to the 2P semantic residual at
  layers 32--48 in almost exactly the same way in Game and Neutral. MLP 49 then
  makes the strongest replicating task-dependent old-rank write, but direct
  restoration shows that this local write is not necessary for the final
  policy effect.
- Broad answer-space compression and attenuated Mixer-56 sharpening help
  express the final distribution, but their exact share under the minimal
  one-word paradigm is not known.

## Important blanks

### 1. How is the nonlinear dependence on graded old rank computed?

Exact source attribution shows each matching 1P line carrying that candidate's
old score into its 2P semantic tokens at layers 32--48, and MLP 49 reshaping
that ranking differently across tasks. The earlier apparent categorical W1
remainder does not survive flexible nonlinear control for candidate score and
competitor gap. What remains missing is the causal computation that converts
the graded retrieved score into the sharply nonlinear rank policy expressed at
the output.

Simple accounts have failed:

- The first-decision boundary update transfers the impending literal letter
  almost perfectly at that boundary (98.9% donor match), but leaves only a
  small literal-letter echo at the final answer (+0.087/+0.077 centered logits
  in Game/Neutral) and no reliable final donor-semantic-choice change (+0.5/-1.1
  points). It does not transfer the letter's semantic binding. Consequential
  old-answer identity is reconstructed from distributed first-presentation
  option history, not read from the empty answer boundary.
- Continuously scrubbing the late A--D letter subspace at that boundary does
  not reduce preferential Game revision.
- A one-dimensional option-newline value probe predicts W1, but removing it at
  all four options does not reliably alter the condition difference.
- Combining the score-direction and decision-letter lesions is not synergistic.

The remaining plausible account is a multidimensional, distributed sequential
comparison trace across the first-presentation history. That is a hypothesis,
not a result.

### 2. Does the feedback GLA update contain W1 identity or only policy?

The transplant experiments keep the question and W1 fixed while swapping
`incorrect` and `lost`; they establish policy transfer, not semantic-target
transfer. The update may contain a bound `W1 is wrong` state, or it may be a
content-free revision gate that later combines with winner information at the
repeated option. We have not run the reciprocal semantic-winner crossover at
the evaluation-period GLA update.

### 3. Where is the policy-conditioned rank transformation redundantly maintained?

MLP 49 is not the necessary local bottleneck. The final-position program now
resolves part of the alternative route. At the exact final decision token,
ordinary-attention writes at layers 52 and 56--not the local MLP writes--supply
the replicated task separation in old-rank geometry. Exact source
reconstruction and reciprocal causal swaps localize that input to
contextualized 2P question/cue scaffold states. Swapping those states transfers
the old-rank geometry in opposite directions on both frozen splits. The
held-out Game conflict-switch reduction is 4.4 points, but that thresholded
behavioral effect is absent on discovery.

The final-position state crossover now establishes that this task state becomes
causally effective at the final token around layer 48, grows through layers
52--62, and undergoes its largest nontrivial consolidation at layer 63, where
82--85% of the paired donor task vector is already sufficient to reproduce
nearly all donor switching behavior. Layer 63 is a GLA layer; layer 64 is only
the exact finished-donor control. A global component crossover further shows
that sequence mixers, not local MLPs, write the effective final state.

The source-unit blank is now narrower: the two-token feedback sentence carries
about 70% of the complete suffix, while the contextualized following
instruction carries another overlapping 49--61%. The remaining question is
which interactions create their redundancy and which relay states feed the
late final-position mixers. MLP 49 may be one readable local
expression of this distributed state, but it is neither the sole carrier nor
the final receiver route.

### 4. What explains the remaining continuous-logit effect?

The semantic relays remove essentially all of the discrete choice gap but only
about half of the held-out W1--W2 margin gap. Output-preserved evaluation-period
GLA deletion removes 30% [9.5, 50.0%] of the behavioral gap. Possible residual routes include
other feedback-token computations, broad compression, nonlinear interactions
among recurrent and ordinary-attention states, and candidate evidence that is
not isolated by the matching-versus-cyclic lesions. Their shares are unknown.

### 5. How general is this circuit?

The account is no longer supported only on SimpleMC. On the frozen 500-question
difficulty-filtered TriviaMC set, remapped preferential semantic-W1 avoidance
reproduces behaviorally; the complete matching-history lesion causally removes
the distinctive Game avoidance; and the complete seven-token feedback suffix
reciprocally transfers 90.5--93.3% of the paired Game/Neutral task vector in
both tasks and both frozen halves. The transferred rank shape is also the
expected opposite one: Neutral suffix state raises W1 and lowers W2--W4 in
Game, while Game suffix state lowers W1 and raises W2--W4 in Neutral. However,
TriviaMC Neutral does not show a stable matching-history support profile across
the two halves.

The compact policy-by-retrieved-rank factorial now also reproduces. On the
held-out half, natural Game's matching-minus-cyclic lesion effect is
+0.698/-0.090/-0.166/-0.442 logits across W1--W4. Installing Neutral's
evaluation-period GLA update reduces the W1 effect by 0.607 logits; installing
Game's update into Neutral creates a strong Game-like
+0.822/-0.097/-0.206/-0.519 route profile. On conflict trials, the reciprocal
period transplant raises old-W1 choice by 13.5 points in Game and lowers it by
12.2 points in Neutral, with both held-out intervals excluding zero and both
directions independently repeating on discovery. Thus TriviaMC replicates the
causal claim that feedback policy changes how retrieved candidate rank is used,
not only the separate policy source and recollection route. It still does not
replicate stable natural-Neutral rank support through this exact route.

The token-resolved entry map and downstream relay inventory have not yet been
replicated on TriviaMC, and no core causal circuit has yet been replicated on
another model. See the [TriviaMC Step-4 report](outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step4/analysis/REPORT.md).

## Best current cognitive description

Neutral identifies each repeated option and, at layers 32--48, retrieves its
graded first-pass score from the semantically matching 1P line. Distributed
MLPs around layers 29--31 also build fresh 2P score. The evaluation-period GLA
update causally tells the downstream matching computation to retain the old
leaders: transplanting Game's update into Neutral reverses the route's rank
shift and sharply reduces W1 choices. MLP 49 displays a strongly top-heavy
version of the old ranking, but is not the sole carrier; later computation can
reconstruct the policy effect when that local output is restored.

Game performs essentially the same initial old-rank retrieval at layers
32--48 and the same broad fresh-score construction. Its evaluation-period GLA
update causally changes the matching route toward suppressing R1/R2 and
supporting R4; transplanting Neutral's update largely reverses that shift and
raises W1 choice. MLP 49 reflects the same less-top-heavy policy but is not its
necessary bottleneck. From layer 48 onward Game also
attends less to the old winner's matching line. The broader causal matching
lesions show that the intact Game route ultimately suppresses R1/R2, is near
neutral for R3, and supports R4 before the final answer.

The model therefore appears to implement **policy-conditioned semantic
reinstatement**, not literal answer copying, a generic instruction to be
random, or one final `not W1` vector.

At the final decision position, both tasks still contain old and fresh evidence.
The four 2P option lines supply a mostly shared old-ranking contribution around
layer 40. Layers 52 and 56 then read task-specific old-rank treatment from the
contextualized repeated-question and choice-cue scaffold. Neutral's scaffold
strongly preserves the old leaders; Game's is much weaker, so the final ranking
is freer to move away from W1 and relatively toward lower old ranks. Direct
rereading of all four raw 1P option lines across ordinary-attention layers
4--64 is not a replicated explanation of the final policy.

The exact final-position crossover adds the causal endpoint: task-specific
state is weak through layer 44, becomes meaningful at layer 48, is about
one-third established over layers 52--60, and becomes nearly behaviorally
sufficient after the layer-63 GLA update. Across the complete computation,
sequence-mixer writes reproduce the donor task state and behavior, whereas MLP
writes alone do not. The new candidate-history relay mediation fills much of
the upstream evidence path: semantic 2P tokens are the dominant history entry
and strongest single relay, with redundant secondary routes through option
boundaries, structure, and the cue scaffold.

The completed policy-binding crossover now localizes part of the missing
combination. Reciprocal same-question Game/Neutral crossover of all four 2P
semantic relays transfers **19.6%** `[16.6%,22.6%]` of the donor task vector
into Game and **24.3%** `[19.6%,29.2%]` into Neutral on confirmation conflicts.
Swapping only one candidate's semantic relay produces **7.5--12.2 percentage
points** more donor-policy transfer on that candidate than on untouched ranks
in Game and **10.3--14.4 points** more in Neutral; every held-out paired
interval is above zero. Policy is therefore already candidate-bound at the 2P
semantic wordpieces, rather than being attached only at the final decision.
Crossing the complete pre-prefix tail transfers **52.5%** `[46.9%,58.0%]` and
**60.5%** `[55.3%,65.3%]`, showing further policy accumulation through option
boundaries, structure, and cue/query state. The full feedback suffix remains a
stronger positive control at 93.0% in both directions. Discovery independently
replicates the joint ordering. The frozen manifest still lacks an independent
same-candidate high-versus-low old-evidence donor, so the exact nonlinear
old-evidence-by-policy interaction remains unresolved rather than inferred
from R1--R4 strata.

## Canonical evidence

- [Behavioral feedback factorial](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/analysis/REPORT.md)
- [Corrected output-preserved evaluation-period GLA update transplant](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_update_transplant/output_preserved/analysis/REPORT.md)
- [Corrected output-preserved evaluation-period causal source trace](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/feedback_factorial/evaluation_period_source_trace/output_preserved_analysis/REPORT.md)
- [Semantic receiver-path validation](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/receiver_path_search/validation/analysis/REPORT.md)
- [Final-query repeated-option ablation](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_query_repeated_option_ablation/analysis/REPORT.md)
- [Comprehensive repeated-W1 relay](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/repeated_w1_relay/analysis/REPORT.md)
- [All-candidate winner-binding factorial](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/all_candidate_matched_relay/analysis/REPORT.md)
- [Exact old/fresh score source attribution](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/score_source_attribution/REPORT.md)
- [Final-decision evidence and causal source report](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/final_position_program/REPORT.md)
- [Final-position state crossover](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/final_position_state_crossover/REPORT.md)
- [Corrected evaluation-to-final relay mediation](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/evaluation_relay_final_mediation/relay_corrected_analysis/REPORT.md)
- [Candidate-history entry factorial](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/candidate_history_pathway/stage_a/analysis/REPORT.md)
- [Candidate-history downstream relay mediation](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/candidate_history_pathway/stage_b/analysis/REPORT.md)
- [Candidate-history policy-binding crossover](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/candidate_history_pathway/stage_c/analysis/REPORT.md)
- [Canonical remapping analysis](outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/analysis/REPORT.md)
