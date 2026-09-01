# Second-presentation option-remapping test

## Definitions

- A **content switch** means that the model selects a different answer option
  from its Baseline choice, tracking the option's text after it moves to a new
  letter. This is the primary outcome.
- **Old-letter avoidance** means that the model avoids the literal A/B/C/D
  letter it emitted in Baseline. Since every option moved, old-letter avoidance
  and content switching are deliberately different outcomes in this test.
- **Positive suppression** means that the centered A–D logit is lower in Game
  than in Neutral (`Neutral - Game`).

The first question presentation is unchanged. In the second presentation, all
four option contents move to different letters using a frozen balanced
derangement. Each question uses the same remapping in Game and Neutral.

## Behavioral results

| Outcome | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Content switch | 65.6% | 46.8% | +18.8 [+14.6, +23.0] pp |
| Old-letter avoidance | 63.0% | 72.0% | -9.0 [-13.2, -4.8] pp |
| Content accuracy | 32.2% | 40.8% | -- |

For comparison, without remapping the established switch rates were
43.2% in Game and
29.0% in Neutral, a
14.2% gap. Remapping changed
that paired Game-minus-Neutral content-switch gap by +5.0 [-1.0, +11.0] pp.

The secondary aggregated-A–D decision gives a content-switch gap of
+18.0 [+13.6, +22.4] pp and an
old-letter-avoidance gap of
-9.6 [-13.8, -5.4] pp.

| Selected option's frozen Baseline rank | Game | Neutral | Game minus Neutral |
|---|---:|---:|---:|
| Rank 1 | 34.4% | 53.2% | -18.8 [-23.2, -14.6] pp |
| Rank 2 | 28.4% | 25.4% | +3.0 [-1.0, +7.0] pp |
| Rank 3 | 23.6% | 13.8% | +9.8 [+6.2, +13.4] pp |
| Rank 4 | 13.6% | 7.6% | +6.0 [+3.0, +9.0] pp |

## Does Game suppression follow content or the old letter?

| Target | Game-specific suppression (logits) |
|---|---:|
| Baseline-selected content at its new letter | +0.633 [+0.558, +0.709] |
| Baseline's old literal letter, now attached to different content | +0.139 [+0.092, +0.188] |
| Content suppression minus old-letter suppression | +0.494 [+0.409, +0.579] |

The full Game-minus-Neutral centered-logit profile, after aligning option
contents by their frozen Baseline ranks, is:

| Baseline rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---:|---:|---:|---:|
| -0.633 [-0.708, -0.558] | +0.079 [+0.009, +0.151] | +0.227 [+0.164, +0.292] | +0.327 [+0.274, +0.379] |

Negative rank values mean Game favors that original content less than Neutral;
positive values mean Game favors it more. This remapping distinguishes a
transformation that follows the previously preferred **content** from one that
merely follows the literal answer **letter**.

The continuous Baseline-content suppression was positive under all nine
derangements (range
0.411
to
0.819
logits). The behavioral Game-minus-Neutral content-switch difference was
positive under eight derangements and exactly zero under one, so the aggregate
result is not produced by a single favorable mapping.

## Causal semantic receiver path

A discovery-screened, held-out-confirmed edge intervention now localizes where
the model compares W1's repeated semantic content with its first presentation.
The receiver is the **second-presentation option line containing W1**. Blocking
ordinary-attention reads from that repeated line back to the original W1 option
line across blocks 4--48 increased Game W1 choice on held-out conflict trials
by 10.3 points [2.9, 17.6], while the token-count-matched unselected-line
control changed it by 2.9 points [0.0, 6.6]. The same selected-line intervention
decreased Neutral W1 choice by 29.4 points [21.3, 37.5]. The selected-minus-
control intervention therefore reduced and reversed the natural Game--Neutral
W1-avoidance gap by 38.2 points [27.9, 48.5]; discovery gave +32.1 points
[22.6, 41.6].

No individual tested block was sufficient. The effect required the same
receiver position across ordinary-attention blocks 4--48, consistent with
redundant semantic reads across depth. Held-out no-conflict trials showed the
same condition-dependent use (+10.6 points in Game and -24.8 points in Neutral,
selected line minus matched line). This establishes a semantic matching relay:
while processing the repeated options, the model reads the matching original
option content. Under `incorrect`, that match helps the model avoid W1; under
`lost`, it helps retain W1. The later operation that converts the retrieved
match into those opposite policies remains to be localized.

- [Full causal report](../receiver_path_search/validation/analysis/REPORT.md)
- [Canonical figure](../../../../figures/qwen36_remapped_receiver_edge_validation.png)
- [Machine-readable summary](../receiver_path_search/validation/analysis/summary.json)

## Does the final decision directly read the repeated W1 line?

Yes, but not in the originally hypothesized suppressive direction. At only the
final pre-answer query, we blocked ordinary attention to each complete
second-presentation option line across blocks 4--64. Blocking the W1 line
lowered W1 relative to the mean of separately blocking the other three lines
in both conditions: on held-out conflict trials, the W1--W2 margin changed by
-0.135 [-0.159, -0.111] logits in Game and -0.320 [-0.364, -0.279] in Neutral.
Thus the direct final read supplies **pro-W1 evidence**, especially in Neutral.

Because Neutral depends more strongly on this final W1 read, the lesion reduced
the Game--Neutral W1-avoidance gap by 8.3 [3.2, 13.7] percentage points on
held-out conflict trials. The continuous condition difference replicated
closely across the frozen splits: +0.169 [0.138, 0.202] logits in discovery and
+0.186 [0.155, 0.218] in confirmation. The discrete choice difference was
weaker in discovery (+2.4 [-3.4, 8.5] points) than confirmation.

This rejects a simple last-hop suppression edge. The earlier validated
original-W1-line → repeated-W1-line path affects Game switching, but its
suppressive consequence must be relayed into intermediate downstream states
before the final query. The final query's own direct read of the repeated W1
line is instead a residual reinstatement route that Game uses less than
Neutral.

- [Final-query repeated-option report](../final_query_repeated_option_ablation/analysis/REPORT.md)
- [Canonical figure](../../../../figures/qwen36_final_query_repeated_option_ablation.png)
- [Machine-readable summary](../final_query_repeated_option_ablation/analysis/summary.json)

## Can first-pass selectedness be varied without changing content or position?

Yes. A six-permutation Baseline screen kept W1 at the identical displayed
letter and option position while permuting only the other three candidates.
There are 222/500 questions where the same W1 content is selected in the
identity ordering but not selected in at least one alternative ordering: 115
discovery and 107 held-out confirmation questions. Coverage is 77 A, 44 B, 64
C, and 37 D questions.

The W1=A subset is especially clean (41 discovery, 36 confirmation). Every
token through the A option line is identical across the chosen and unchosen
presentations, so any selectedness signal must be constructed after the model
sees later competitors. This provides the missing design for separating option
semantics from “this option was my first decision.” It also explains why the
earlier fixed-A donor experiments could localize semantic content at the A
option line but could not identify a separate choice-binding signal: both donor
and recipient already chose A.

That frozen causal test is now complete. Under its exact four-row cached regime,
38/41 discovery and 32/36 confirmation pairs retained the required donor-A /
recipient-not-A selectedness contrast. Transplanting ordinary-attention K/V at
only the empty first-answer decision boundary did **not** transplant a usable
“A was selected” signal. The centered semantic-A effect was −0.033 in Game and
−0.043 in Neutral in discovery, then +0.015 and +0.019 in confirmation. The
Neutral-minus-Game interaction was −0.010 [−0.039, +0.017] and +0.005
[−0.029, +0.038], respectively. Pooling the prespecified splits descriptively,
the interaction was −0.004 [−0.026, +0.018] logits and the semantic-A choice
interaction was +1.43 [−2.86, +7.14] percentage points.

The complete-cache donor positive control and untouched donor rows both
reproduced with exactly zero A-D logit error, so this is an interpretable null:
the missing selectedness binding is not carried by conventional-attention K/V
at that single boundary token. The small target-versus-recipient-winner margin
movement was same-direction rather than Game/Neutral-opposed and did not
replicate as an interaction. The prespecified gate therefore failed and no
depth localization was launched.

- [Feasibility-screen report](../w1_fixed_permutation_screen/analysis/REPORT.md)
- [Machine-readable screen summary](../w1_fixed_permutation_screen/analysis/summary.json)
- [Frozen eligible pairs](../w1_fixed_permutation_screen/analysis/eligible_pairs.json)
- [Frozen six-permutation plan](../w1_fixed_permutation_screen/plan.json)
- [Boundary-K/V causal report](../w1_selectedness_boundary_kv/analysis/REPORT.md)
- [Boundary-K/V canonical figure](../../../../figures/qwen36_w1_selectedness_boundary_kv.png)
- [Boundary-K/V machine-readable summary](../w1_selectedness_boundary_kv/analysis/summary.json)

## Does the option-closing newline carry “this seems right” information?

Yes, linearly and out of sample. We collected only the exact option-closing
newline residuals previously used to construct content-aligned directions, trained one
shared ranker on 251 questions across all six mappings, removed static
displayed-letter means, and tested on 249 entirely held-out questions. The
ranker reached 64.9% [60.5, 69.1] top-1 accuracy at descriptive peak readout 53
and 62.0% [57.8, 66.3] at readout 64. The model's majority-letter (A) predictor
achieved 51.9% [46.7, 56.9]; the paired readout-53 gain was 13.0 percentage
points [5.3, 20.8]. Thus the local state contains candidate-specific value
information beyond absolute letter bias.

The stricter matched analysis held W1 content and displayed letter fixed while
distractor ordering changed whether W1 won. At readout 53, the identical W1
option's score was 4.36 units higher [3.06, 5.79] when it won than when it lost.
The difference was exactly zero for A (n=36), whose complete prefix through the
A newline is token-for-token identical, but positive for B (+4.31), C (+9.73),
and D (+3.81), where preceding competitor context can alter the local state.
The signal becomes visible in the thirties and is strongest in the late
forties/early fifties.

This supplies the representation-level fact missing from the boundary-K/V
null: the option-closing newline carries a context-dependent “candidate
value/selectedness” component. It remains a correlational decoder and does not
establish a clean mapping-invariant semantic code at that exact token; causal
use of the fitted direction has not yet been established.

- [Option-newline selected-answer probe report](../option_newline_choice_probe/analysis/REPORT.md)
- [Canonical option-newline probe figure](../../../../figures/qwen36_option_newline_choice_probe.png)
- [Machine-readable probe summary](../option_newline_choice_probe/analysis/summary.json)
- [Frozen probe plan](../option_newline_choice_probe/PLAN.md)

The causal follow-up is now complete, and it is an informative null. It changed
only this decoded coordinate at the first-presentation W1 newline over readouts
33--56, moving it from the chosen-presentation score to the matched unchosen-
presentation score. On 38 held-out conflict questions, the W1-minus-W2 effect
was -0.013 [-0.062, +0.036] logits in Game and -0.053 [-0.107, 0.000] in
Neutral. The prespecified interaction was therefore +0.040 [-0.029, +0.102]
logits. W1 choice rose by +5.3 [0.0, +13.2] points in Game and 0.0 [-7.9,
+7.9] in Neutral, but the +5.3 [-5.3, +18.4] point interaction was uncertain.

The discovery margin interaction was +0.057 [+0.006, +0.115], but the discrete
effect did not match confirmation, and the equal/opposite sign control did not
replicate (-0.080 logits in discovery versus -0.011 in confirmation). The exact
zero-dose sham had zero effect, natural logits reproduced exactly, and the
devaluation clamp reached its score target with small error. Thus the probe
really decodes a candidate-value correlate, but its single linear direction is
not established as the causal selectedness-binding channel. The binding may be
nonlinear, multidimensional, distributed across option states, or encoded in a
different basis.

- [Option-newline candidate-value causal report](../option_newline_value_causal/analysis/REPORT.md)
- [Option-newline candidate-value causal plan](../option_newline_value_causal/PLAN.md)
- [Machine-readable causal summary](../option_newline_value_causal/analysis/summary.json)
- [Canonical causal figure](../../../../figures/qwen36_option_newline_value_causal.png)

**Correction and resolved rerun:** the first all-four projection run is invalid
as a causal test. The probe score was defined after subtracting a
displayed-letter mean, but that runner projected the raw residual without the
centering and used absolute cached-K/V replacement. It therefore removed static
A/B/C/D structure and accumulated float16 cache reconstruction error. Its
outputs remain audit-only.

The corrected run removed only `((z - mu_letter) dot u)u`, added the resulting
projected-minus-unprojected K/V delta to the model's live K/V entries, covered
all 16 ordinary-attention blocks 4--64, and included the same hook path with an
exactly zero identity delta. Across all 1,000 condition-question identity
comparisons, the maximum A--D logit difference was exactly 0.0 and no answer
changed.

The corrected causal result is much smaller and not robustly Game-specific. On
136 held-out conflict questions, W1 choice changed by +1.5 [-2.2, +5.1] points
in Game and -0.7 [-3.0, +1.5] in Neutral; interaction +2.2 [-2.2, +6.6]. The
W1-minus-W2 margin changed by +0.020 [+0.003, +0.038] logits in Game and -0.002
[-0.022, +0.017] in Neutral; interaction +0.023 [-0.003, +0.049]. Discovery's
margin interaction was only +0.002 [-0.025, +0.028]. Thus the held-out logit
effect weakly points toward W1 reinstatement in Game, but it does not replicate
in magnitude and the discrete effect is uncertain. The one-dimensional decoded
coordinate participates in option scoring, but is not established as the
semantic selectedness-binding mechanism behind preferential Game revision.

- [All-four candidate-value projection report](../option_newline_all_four_projection/analysis/REPORT.md)
- [All-four projection summary](../option_newline_all_four_projection/analysis/summary.json)
- [All-four projection plan](../option_newline_all_four_projection/PLAN.md)
- [Canonical all-four projection figure](../../../../figures/qwen36_option_newline_all_four_projection.png)
- [Corrected centered all-four report](../option_newline_all_four_centered_projection/analysis/REPORT.md)
- [Corrected centered all-four summary](../option_newline_all_four_centered_projection/analysis/summary.json)
- [Corrected centered all-four plan](../option_newline_all_four_centered_projection/PLAN.md)
- [Corrected canonical figure](../../../../figures/qwen36_option_newline_all_four_centered_projection.png)

The matched redundancy factorial removed the centered candidate-value
coordinate at all four original option newlines, centered A--D identity at the
first-decision position, or both. Joint removal did not produce the predicted
larger recovery of W1. On 136 held-out conflict questions, letter-only removal
gave a +3.7 [+0.7, +7.4] point Game-minus-Neutral W1-choice effect, but the
discovery interaction was 0.0 [-4.4, +4.4], and the held-out continuous-margin
interaction was only +0.006 [-0.022, +0.035] logits. The joint lesion changed
Game W1 choice by 0.0 [-2.2, +2.2] points and was antagonistic rather than
synergistic. The one-dimensional score and first-decision letter identity are
therefore not established as redundant causal routes for preferential Game
revision.

- [Joint score/letter causal report](../joint_option_score_decision_letter/analysis/REPORT.md)
- [Joint score/letter summary](../joint_option_score_decision_letter/analysis/summary.json)
- [Joint score/letter plan](../joint_option_score_decision_letter/PLAN.md)
- [Joint score/letter canonical figure](../../../../figures/qwen36_joint_option_score_decision_letter.png)

The stronger live-residual version of the letter test is also complete. Rather
than changing only ordinary-attention K/V, it orthogonally removed the full
centered three-dimensional A--D JLens subspace from the exact first-decision
token after **every readout 48--63**. The projection was reapplied after each
block, so all subsequent attention, GLA, and MLP computation received the
scrubbed state. It removed a mean residual component norm of 2.01 while leaving
at most 3.33e-5 A--D coefficient norm after each edit. Natural and identity-hook
logits were exactly equal, and the natural path exactly reproduced the prior
validated same-host run.

This decisive test is negative. On 136 held-out conflict questions, natural W1
choice was 17.6% in Game and 40.4% in Neutral. Scrubbing raised W1 choice by the
same +0.7 points in both conditions, giving a Game-minus-Neutral interaction of
0.0 [-2.2, +2.2] points and explaining 0% of the 22.8-point avoidance gap at
the point estimate. The W1-minus-W2 interaction was +0.001 [-0.015, +0.017]
logits. Discovery's +4.4-point discrete interaction did not replicate in
confirmation or in its continuous margin (+0.010 [-0.007, +0.028] logits).
Restricting confirmation to questions whose current first decision matched W1
also gave exactly 0.0 points and +0.002 [-0.015, +0.019] logits.

The shared effect was small generic flattening: held-out conflict-trial
W1-minus-W2 rose by +0.019 logits in Game and +0.018 in Neutral, while A--D
spread fell by 0.035 and 0.040 logits and entropy rose slightly in both. Thus
the explicit late answer-letter state at the first-decision token participates
in generic candidate geometry, but it is not the condition-specific binding
route that makes Game avoid the semantic first winner. An earlier relay before
readout 48, or winner information in other dimensions or positions, remains
possible.

- [Continuous first-decision letter-scrub report](../first_decision_letter_scrub/analysis/REPORT.md)
- [Continuous letter-scrub summary](../first_decision_letter_scrub/analysis/summary.json)
- [Continuous letter-scrub plan](../first_decision_letter_scrub/PLAN.md)
- [Continuous letter-scrub canonical figure](../../../../figures/qwen36_first_decision_letter_scrub.png)

## Does the complete first-decision update bind the chosen letter to its semantics?

No. A reciprocal crossover now pits the distributed first-presentation history
against the complete local update at the empty first-answer boundary. Each
question supplied two histories that naturally chose different semantic
answers X and Y, followed by identical feedback and an identical second
presentation. The intervention retained X's accumulated pre-boundary state but
replayed Y's full 64-block boundary trajectory (and reciprocally Y-history /
X-update). It made the model itself write the crossed conventional-attention
K/V and GLA state at that token. A final-output clamp made the immediate A--D
decision manipulation exact without changing the cache writes already made.

The manipulation succeeded: the crossed immediate decision matched the donor
on 97.1% of exact discovery questions and 98.9% of confirmation questions.
Nevertheless, it did not transfer the donor semantic winner in the
condition-specific way required. Neutral-minus-Game donor-semantic transfer was
-0.019 [-0.045, +0.005] logits in discovery and -0.009 [-0.035, +0.016] in
confirmation. Complete donor history, by contrast, produced +0.428 [+0.252,
+0.603] and +0.466 [+0.268, +0.672] logits. In confirmation, complete-history
semantic transfer was +0.397 logits in Game versus +0.863 in Neutral.

The boundary update did carry a weak portable **literal-letter** trace. In
confirmation it raised centered evidence at the donor's old letter by +0.087
logits in Game and +0.077 in Neutral, while evidence for the donor semantic
answer at its current remapped letter changed by only +0.019 and +0.013, both
uncertain. Discovery showed the same letter/semantic separation.

Thus the boundary contains the impending output letter, but not a portable
`letter -> semantic answer` binding that controls later revision. The semantic
winner is instead reconstructed from, or remains distributed across, the
first-presentation history. This connects the earlier original-option-line K/V
localization to the late decision: later computation can retrieve a semantic
candidate from history, while which candidate won is not stored as one
transplantable boundary-token state.

The split cached identity path is numerically different from a single unsplit
forward and changed some near-tied natural choices. Every causal estimate uses
the matched split identity baseline, and the full-donor split path reproduced
the donor exactly (0.0 maximum A--D logit error), so the crossover contrasts do
not mix numerical regimes.

- [Boundary-crossover report](../decision_boundary_crossover/analysis/REPORT.md)
- [Boundary-crossover summary](../decision_boundary_crossover/analysis/summary.json)
- [Boundary-crossover plan](../decision_boundary_crossover/PLAN.md)
- [Boundary-crossover canonical figure](../../../../figures/qwen36_first_decision_boundary_crossover.png)

## Mechanism audit after the winner-letter and candidate-score nulls

The completed interventions rule out two simple storage accounts. The complete
first-decision boundary update carries the impending literal answer letter but
does not carry a portable letter-to-semantic binding, and neither continuous
late A--D residual scrubbing nor the all-block boundary-K/V letter lesion
reliably alters preferential semantic revision. Likewise, removing the fitted
centered one-dimensional candidate-value coordinate from **all four** original
option newlines at every ordinary-attention carrier block does not reproduce a
stable Game-specific effect. These mechanisms should not be proposed again as
leading explanations without new evidence.

The robust positive results instead separate three functions:

1. Conventional-attention K/V at the original option lines preserves semantic
   history. A selected-line transplant moves the corresponding semantic answer,
   and tokens on the matching repeated option line causally read that history
   across blocks 4--48.
2. The evaluation-closing GLA update carries a real but partial condition/policy
   state. In the corrected output-preserved transplant, Evaluation-to-Neutral
   transfers 0.097 logits and Neutral-to-Evaluation transfers 0.091 logits of a
   natural 0.469-logit W1--W2 margin difference. Corrected deletion removes
   5.5 points [1.5, 9.5], or 30.0% [9.5, 50.0%], of the natural behavioral gap
   and 0.097 logits [0.059, 0.135] of the margin gap. Blocks 25--32 carry about
   58% of the recurrent route, with no single indispensable GLA. The much
   larger historical estimates also changed the source token's residual and
   are superseded for route-specific attribution.
3. Late computation is dominated by differential reinstatement. The repeated
   W1 line and Mixer 56 support W1 in both conditions, substantially more in
   Neutral. Direct final and pre-final reads from the repeated W1 line are
   pro-W1, not an anti-W1 message sent by Game.

This leaves a specific unresolved fork: does the evaluation-period GLA update
already bind the semantic identity of W1, or is it a content-free revision
policy that changes how later repeated-option queries reconstruct candidate
history?

### New supporting analysis: continuous evidence rather than a binary tag

As a post-hoc diagnostic, the standalone first-pass W1 margin was compared with
the later Game-specific reduction in W1 centered advantage
(`Neutral - Game`). Across all 500 questions, Spearman rho was 0.469. The
association remained 0.406 for original W1=A and 0.531 for W1=B--D. Mean
Game-specific W1 suppression rose from 0.206 logits for first-pass margins
below 0.25 to 1.571 logits for margins above 1.0. This is not a frozen causal
test, but it supports a continuous preference/reinstatement account over a
single hidden winner bit.

### H1: the evaluation-period GLA update contains a bound semantic error target

This is plausible because the update is already strongly causal and occurs
after the full first presentation. It has not been tested: previous update
transplants changed `incorrect` versus `lost` while holding the question and W1
fixed, so they could not reveal whether semantic identity traveled with the
update.

The decisive test is a reciprocal semantic crossover. Use existing same-question
X/Y histories with different semantic first winners and an identical second
presentation. At the period after `incorrect.`, transplant all 48 GLA updates
from the X-winner history into the Y-winner history and vice versa. A donor-
targeted change in the final X-versus-Y margin would show that the update itself
contains the bound answer identity. Exact reinsertion, same-winner/different-
order, within-history Evaluation/Neutral policy transfer, and complete-history
donor controls make the null interpretable. No previous experiment performed
this semantic crossover at the evaluation-period update.

### H2: the evaluation update is a content-free gate and selectedness is reconstructed at the repeated option

On this account, original option-line K/V supplies semantic/evidential content,
while the repeated option's query is conditioned by the complete first history
and the evaluation GLA state. The conjunction is therefore receiver-side; it
need not be a score or winner tag stored at the source. This fits the source
letter/score nulls, the condition-opposed original-to-repeated edge lesion, and
the much stronger Neutral reinstatement from the repeated line.

There is an unusually clean test using the frozen W1=A permutation pairs. In
each pair, the complete prefix through the original A option line—and hence its
causal K/V—is exactly identical, but later distractor order makes A win in one
presentation and lose in the other. Measure the causal effect of blocking the
identical original-A-to-repeated-A edge in both members. A replicated
chosen-versus-unchosen difference cannot reside in the source A state; it must
come from the receiver query or other later context. The directional prediction
is stronger A reinstatement in Neutral when A previously won, and attenuation
or reversal under Evaluation. The prior receiver experiment included only
first-pass winners and therefore did not make this matched comparison.

A second-stage mediation test should combine the already validated
Evaluation-to-Neutral GLA-update transplant with presence versus blockade of
the original-W1-to-repeated-W1 edge. If the transplanted revision effect
collapses when that semantic-match edge is unavailable, the evaluation state
causally gates this match rather than independently suppressing the final
answer. This exact joint intervention has not been run.

### H3: first-pass choice is a distributed sequential comparison trace

Because A's option-line state is causally fixed before B--D are seen, its final
winner status cannot be written at A. The decoder's exact-zero chosen/unchosen
difference for A, positive differences for B--D, strong distractor-order
sensitivity, and first-answer-boundary nulls instead suggest a running
comparison: later option states record whether each new candidate displaced the
current incumbent. The model may reconstruct the winner from these comparator
traces rather than storing a discrete answer record.

The direct test again uses W1=A chosen/unchosen pairs. Transplant the complete
post-A first-presentation conventional-attention K/V suffix (the B--D option
region) from an A-chosen history into its matched A-unchosen history, leaving
the byte-identical A source untouched. If this makes Neutral reinstate semantic
A and Evaluation avoid it, the missing selectedness information is in the
later comparison history. Boundary-only transplantation is already null and
serves as a prior negative control; complete donor history is the positive
control. Only after a successful suffix prerequisite should individual later
option regions be localized.

### Priority

H1 is the first test because it interrogates the strongest known causal state
with a single missing semantic contrast. If H1 fails while its within-history
policy-transfer control succeeds, H2 becomes the leading account. The
identical-source W1=A edge comparison then tests H2 without any decoder. H3 is
the localization follow-up if that receiver-side selectedness effect is real.
This ordering prevents another broad search and makes each result determine the
next experiment.

## All-candidate matched relay: nonlinear rank dependence at semantic matches

The canonical 500-question follow-up resolves the main ambiguity left by the
fixed-A receiver test. Instead of testing only W1=A, it ranked every original
candidate W1--W4 by first-pass logits and separately blocked each original
option line from its semantically matching repeated line at ordinary-attention
blocks 4, 8, ..., 48. Each individual lesion was compared with a cyclic
nonmatching-source lesion of the same repeated option, and a joint intervention
blocked all four matching relays.

The Game-minus-Neutral matching-specific effect is graded and nonlinear in
first-pass candidate evidence. A linear question-centered regression initially
gave an apparent additional W1 term, but that model did not exhaust the graded
alternative. A frozen follow-up with flexible curves for both candidate score
and the gap to the best competing candidate leaves the R1 term uncertain on
both splits: -0.161 [-0.434,+0.098] in discovery and +0.183
[-0.115,+0.458] in confirmation. Adding R1 worsens held-out prediction by 0.7%,
and near-tie R1-minus-R2 contrasts also include zero. The supported result is a
nonlinear graded-rank effect, not a separate categorical winner state.

The direct W1 matching-edge lesion has opposite signs relative to natural:
+0.362 [+0.204, +0.521] logits in Game and -1.178 [-1.350, -1.013] in Neutral.
Thus Game is not merely failing to reinstate W1; the semantic W1 match itself
contributes against W1 under `incorrect` while supporting W1 under `lost`.

This causal pathway explains the discrete preferential-switching behavior. On
held-out conflict trials, tie-safe displayed-order argmax gives natural W1
choice of 17.6% in Game versus 39.7% in Neutral. Jointly blocking all four
semantic matching relays produced the same 27.2% point estimate in both tasks,
reducing the -22.1-point natural gap by +22.1 points [+11.8, +32.4]. The
confidence interval spans partial through over-elimination, so equality of the
two sample proportions is not evidence of exact population-level elimination.
Discovery independently moved
from 22.6% versus 38.0% (-15.3 points) to 31.4% versus 32.8% (-1.5 points).
The continuous result is less complete: the held-out W1--W2 margin gap shrank
by 0.314 [+0.130, +0.509] logits, from -0.646 to -0.332 (49%), while discovery
margin mediation was uncertain.

The resulting mechanistic account is policy-dependent semantic reinstatement.
First-presentation option lines feed their matching second-presentation lines;
Neutral reinstates candidates along those matches in proportion to prior
evidence and especially W1, whereas `incorrect` sharply changes that same
matching computation nonlinearly with prior rank. Natural
attention mass, context norm, and projected-write norm diverge most strongly
at ordinary-attention layer 36 and remain different through layer 48. These
layerwise decompositions are observational; the causal evidence is the
matching-versus-cyclic factorial and joint mediation.

The exhaustive natural source map now shows that “feed their matching lines”
is not an isolated one-to-one copy. Across layers 12--52, each repeated option
line devotes **33--41%** of its attention to the four original option lines:
about **13--16%** to its semantic match and **20--25%** to the other three.
The first-answer cue and boundary receive another **6--9%** through most of
layers 4--60. The repeated line therefore has simultaneous access to its
semantic match and the entire first-pass comparison context.

Policy is read into the same computation. At layers 28/36/44/48, Game assigns
**3.43/4.15/3.98/2.19%** of repeated-line attention to the `incorrect`
sentence, versus **1.59/1.07/0.95/0.57%** for Neutral's `lost` sentence. This
is observational evidence for joint availability of policy and candidate
history, not yet a causal localization of the winner-rank carrier.

The follow-up causal factorial separates the semantic match from those other
three direct reads. At every ordinary-attention layer 4--64, it blocked all
three nonmatching 1P option lines while preserving each 2P line's matching 1P
line. On held-out questions this shifted evidence away from R1 and toward
R3/R4 in both tasks: the within-task rank-slope change was +0.093
[+0.066, +0.120] in Game and +0.089 [+0.060, +0.118] in Neutral. The
nonmatching lines are therefore causally used as a **shared rank-evidence
source**.

They do not generate the task-specific policy. The Game-minus-Neutral rank
slope was +0.243 [+0.207, +0.279] naturally and +0.247 [+0.213, +0.280] with
all three nonmatching reads removed; the change was +0.004
[-0.021, +0.029]. The same null held separately for conflict and no-conflict
trials. By contrast, blocking the semantic matches collapsed the policy slope
to +0.025 [-0.006, +0.056]. The result narrows the unresolved carrier: the
task-specific mapping of prior rank into retention versus revision travels
through the matching line or another preserved input already bound to it, not
through direct 2P reads of the other three 1P option lines.

This establishes that the matching computation has causal effects organized by
first-pass evidence and rank. It does **not** establish that a rank label is
stored in the matching source or that a categorical semantic-to-rank binding
exists. The rank dependence could instead arise when an old candidate score is
combined with current second-presentation evidence and feedback-conditioned
receiver state.

- [Canonical all-candidate report](../all_candidate_matched_relay/analysis/REPORT.md)
- [Machine-readable summary](../all_candidate_matched_relay/analysis/summary.json)
- [Frozen design](../all_candidate_matched_relay/PLAN.md)
- [Canonical figure](../../../../figures/qwen36_all_candidate_matched_relay.png)
- [Nonlinear categorical-winner audit](../second_presentation_residual_workspace/categorical_winner_audit/REPORT.md)
- [Nonlinear audit figure](../../../../figures/qwen36_categorical_winner_nonlinearity_audit.png)
- [Exhaustive repeated-line source map](../../../../figures/qwen36_second_presentation_attention_distribution.png)
- [Exhaustive source-distribution report](../second_presentation_attention_distribution/analysis/REPORT.md)
- [Nonmatching-history causal factorial](../nonmatching_history_factorial/analysis/REPORT.md)
- [Nonmatching-history figure](../../../../figures/qwen36_nonmatching_history_factorial.png)

## Existing-data old-score/current-score integration

A frozen-split analysis of the saved natural logits and matching-edge lesions
now directly tests whether the second-pass computation combines evidence from
both presentations. Every model controls the candidate's displayed position
in both presentations. On held-out questions, adding first-pass candidate
evidence to a model already containing fresh remapped-presentation evidence
raises final-logit R-squared by +0.020 [+0.006, +0.033] in Game and +0.041
[+0.019, +0.064] in Neutral. Flexible score terms give +0.020
[-0.011, +0.050] and +0.061 [+0.027, +0.099].

The causal matching-edge endpoint shows the condition difference more clearly.
Positive lesion effects mean that the intact semantic match opposed the named
candidate. After controlling current evidence and both displayed positions,
the standardized old-score coefficient in Game minus Neutral is +0.175
[+0.082, +0.275] in discovery and +0.235 [+0.127, +0.346] in confirmation.
Current second-pass evidence independently contributes +0.248
[+0.151, +0.345] and +0.158 [+0.069, +0.249]. The simple old-by-current
interaction does not replicate robustly after flexible score control.

Thus a two-evidence account is now the leading economical hypothesis: the
matching computation is conditioned by both historical and current candidate
evidence. This remains predictive rather than representational. It does not
show that the old score resides in the matching source value, and the small
residual W1 increment could reflect either categorical winner status or
remaining nonlinear score structure.

- [Old/current score report](../old_current_score_integration/analysis/REPORT.md)
- [Machine-readable score analysis](../old_current_score_integration/analysis/summary.json)

## Causal location of old candidate value and a post-list comparison state

The predictive two-score result left two distinct possibilities: candidate-
local old evidence stored on a matching 1P line, and a global comparison state
formed after the complete 1P option list. A frozen crossover tested both while
holding target semantics, its literal D position, its complete line text, and
the entire 2P presentation fixed. Only the order of A--C changed, placing the
same D candidate in a high- versus low-first-pass-evidence history.

Replacing the complete D line's ordinary-attention keys and values at every
ordinary-attention layer (4, 8, ..., 64) transferred old candidate value. On
confirmation, the target's centered final logit moved +0.387 [+0.299, +0.488]
and +0.359 [+0.279, +0.447] in Game under low/high current evidence, and
+0.440 [+0.331, +0.585] and +0.446 [+0.333, +0.588] in Neutral. Replaying
only the D-closing newline's state through all 64 layers produced smaller but
replicated effects: +0.168/+0.190 in Game and +0.186/+0.224 in Neutral.

Because D is also the target's own line, that target effect alone would not
prove a global comparison state. The prespecified four-semantic-candidate
follow-up supplies the missing discrimination. Excluding D, the D-line K/V
transfer vector aligns with the exact complete-history transfer vector on
confirmation at cosine +0.472 [+0.298, +0.637] in Game and +0.515 [+0.407,
+0.633] in Neutral. The D-closing newline alone gives +0.510 [+0.354, +0.649]
and +0.516 [+0.341, +0.679]. All slopes and intervals are positive on both
frozen splits. The final option-closing position therefore contains a
causally portable summary of the broader completed comparison, not merely the
local D candidate's value.

This carrier is shared rather than policy-specific. Its effect is similar in
Game and Neutral, and high versus low fresh evidence does not reliably modulate
the local transfer. In contrast, the complete-history transfer is much larger
under high current evidence on confirmation: high-minus-low +0.511 [+0.191,
+0.836] in Game and +1.154 [+0.800, +1.521] in Neutral. Thus the model stores
usable old candidate value and a partial global comparison summary at the end
of the 1P list, but the nonlinear integration with current evidence and the
Game-versus-Neutral transformation also use information outside that isolated
carrier or later downstream computation.

- [Causal crossover report](../d_line_score_transfer/analysis/REPORT.md)
- [Four-candidate vector summary](../d_line_score_transfer/analysis/global_vector_summary.json)
- [Frozen design](../d_line_score_transfer/PLAN.md)
- [Canonical figure](../../../../figures/qwen36_d_line_score_transfer.png)

## Historical two-mapping fixed-A serial mediation

The prior source transplant and receiver-edge experiments left one step
inferred: whether the semantically matching second-presentation option actually
mediates the causal effect of changing the stored first selected-option line.
The fixed-A X/Y follow-up crossed the selected-option K/V transplant with an
all-ordinary-layer blockade from that original line to either (a) the repeated
line containing the donor semantic answer or (b) a token-count-matched
nonmatching repeated line.

Within that historical two-mapping cohort, the serial path replicated on its
frozen held-out split. In Game, open donor
transfer was +0.548 logits and fell to +0.260 when the matching receiver was
blocked: mediation +0.288 [+0.204, +0.370]. In Neutral, transfer fell from
+3.092 to +2.508: mediation +0.584 [+0.455, +0.737]. The nonmatching blockade
did not remove transfer. The matching-specific removed component was +0.450
[+0.290, +0.613] logits in Game and +0.516 [+0.295, +0.752] in Neutral; their
difference was not reliable. At the discrete level, matching blockade removed
10.2 percentage points of donor-answer selection transfer in each condition.

This result does **not** survive the later matched 24-ordering calibration.
Under the same cohort and pair-construction procedure used for B--D, fixed-A
Game donor transfer is null, and the nonmatching control blockade removes more
Neutral transfer than the matching blockade. The estimates above remain valid
for the historical two-mapping cohort but should not be used to claim a robust
matching-specific serial mechanism. The matched A--D result is reported below.

- [Human-readable mediation report](../fixed_a_donor_receiver_mediation/analysis/REPORT.md)
- [Machine-readable mediation summary](../fixed_a_donor_receiver_mediation/analysis/summary.json)
- [Frozen mediation design](../fixed_a_donor_receiver_mediation/PLAN.md)

## Complete ordinary-attention range: layers 52--64 add no necessary relay

The earlier all-candidate causal intervention ended at layer 48 even though
the corrected descriptive attention trajectory continued through layer 64.
The complete-range follow-up closes that mismatch by testing, on all 500
questions, both layers 52--64 alone and the full layers-4--64 matching-edge
blockade, with the same cyclic nonmatching controls.

The late-only effect is practically null. On held-out questions, individual
R1--R4 effects are about 0.005--0.015 logits in both Game and Neutral, with no
reliable policy interaction and no reliable W1-choice effect. The full 4--64
effects reproduce the prior 4--48 results: extending through layer 64 changes
each held-out candidate mean by approximately 0.01 logits (maximum 0.0101
before rounding).

The complete-range causal result therefore separates two observations. Natural
matching attention is visibly rank- and policy-dependent at layers 52--64, but
those late reads are not an additional necessary route for preferential
switching. The causal semantic relay is jointly carried somewhere in layers
4--48; the responsible layer or smaller cooperative band inside 4--48 remains
unresolved.

- [Complete-range report](../all_candidate_matched_relay_full_range/analysis/REPORT.md)
- [Machine-readable complete-range summary](../all_candidate_matched_relay_full_range/analysis/summary.json)
- [Frozen complete-range design](../all_candidate_matched_relay_full_range/PLAN.md)
- [Complete-range figure](../../../../figures/qwen36_all_candidate_matched_relay_full_range.png)

## Selected-line semantic transfer under one matched A--D pipeline

The apparent fixed-A exception was caused by comparing different cohort
constructions. We reran fixed A using exactly the complete 24-ordering screen,
pair-selection rule, 251/249 split, full selected-line source span, and
ordinary-attention layers 4, 8, ..., 64 used for fixed B--D. The repeated
presentation and all visible recipient tokens were held fixed.

Under that matched pipeline, all four literal positions show the same policy
separation. Held-out Neutral donor-semantic transfer was +2.444 [+2.005,
+2.929] logits at A, +1.927 [+1.502, +2.407] at B, +1.475 [+1.170, +1.803]
at C, and +1.058 [+0.798, +1.336] at D. Donor-answer choice rose by 31.5,
30.3, 20.8, and 17.0 percentage points.

Game did not show comparable transfer at any position. The new held-out A
effect was +0.064 [-0.265, +0.380] logits with a +3.8-point donor-choice
change; B--D were +0.290 [-0.067, +0.677], +0.009 [-0.267, +0.283], and
-0.128 [-0.431, +0.156]. The old two-mapping fixed-A Game estimate (+0.548
logits) remains a valid result for that historical cohort, but it does not
replicate under the position-matched design and should not support an
A-specific mechanism.

The repeated-line mediation is likewise consistent across positions. For
fixed A, blocking the semantically matching repeated line removed 0.367 logits
of held-out Neutral transfer, but the token-count-matched nonmatching blockade
removed 0.764; matching-specific mediation was therefore -0.398 [-0.767,
-0.061]. Fixed B--D showed the same negative-specificity pattern. Game had no
reliable open donor transfer to mediate. Counterfactual selected-line semantics
are thus carried strongly into Neutral's final answer, whereas `incorrect`
largely prevents that substituted history from controlling the answer; the
remaining Neutral effect is not uniquely routed through the matching repeated
line.

This qualifies, but does not overturn, the all-candidate natural-history
lesion. Blocking each *actual* first-presentation candidate's matching relay
still causally explains preferential Game switching. The new result says that
substituting a counterfactual semantic history at the selected line behaves
differently: Neutral follows it, whereas Game largely rejects it, and the
remaining Neutral transfer is distributed across repeated-option processing
rather than uniquely localized to the matching receiver.

- [Same-pipeline fixed-A calibration](../fixed_a_full24_calibration/analysis/REPORT.md)
- [Fixed-A calibration summary](../fixed_a_full24_calibration/analysis/summary.json)
- [Frozen fixed-A calibration design](../fixed_a_full24_calibration/PLAN.md)
- [Canonical A--D comparison figure](../../../../figures/qwen36_fixed_a_full24_calibration.png)
- [Fixed-B/C/D component report](../fixed_bcd_line_generalization/mediation_analysis/REPORT.md)

## Minimal-policy transport through the second-presentation residual stream

The reusable 500-question activation workspace now resolves where the single
`incorrect` versus `lost` prompt difference appears during the repeated
presentation. It retains all 64-layer residual trajectories at every 2P option
token and reconstructs the exact ordinary-attention write from each feedback
token. The primary comparison is the complete same-layer J/R-lens vocabulary
profile at source and receiver; semantic word families were not used to select
the result.

Direct feedback writes into all four 2P option-closing newlines follow a
replicated cascade. The literal evaluation word peaks at ordinary-attention
layer 32 (held-out mean Game-minus-Neutral write RMS 0.0143 across R1--R4), its
closing period peaks at layer 44 (0.0141), and the contextualized but textually
identical `Choose` token peaks at layer 60 (0.0300). The discovery/confirmation
write-vector cosines are 0.9981, 0.9995, and 0.9981. Thus L60 is the largest
single direct source write, not the onset of policy information.

The complete 2P residual becomes plainly evaluative in the R-lens at layers
24--28 (`wrongly`, `failure`, `unsuccessful`). Both J- and R-lenses converge on
retry semantics at layer 36 (`again`, `retry`, `failed`), then shift through
replacement semantics at layers 40--52 and remapping/permutation semantics at
56--60. The four first-pass ranks look similar in this policy-only contrast;
the analysis therefore locates generic policy availability in 2P but not the
rank-specific policy binding.

Two direct post-list routes are especially source preserving. At layer 36, the
choice-cue space attends to the literal evaluation word 4.823% in held-out Game
versus 0.229% in Neutral; the exact write decodes as incorrect/failure/mistake
versus continuity/restore. At layer 28, the final decision attends to the
evaluation-closing period 5.889% versus 1.280%, after which its residual reads
out incorrect/failure/unsuccessful. These are exact activation-path
decompositions, not yet causal feature interventions.

- [Human-readable policy-transport report](../second_presentation_residual_workspace/REPORT.md)
- [Canonical policy-transport figure](../../../../figures/qwen36_second_presentation_policy_transport.png)
- [Frozen workspace design](../second_presentation_residual_workspace/PLAN.md)

## Causal feedback source and downstream relay

The activation paths above are now tied to behavior by a reciprocal causal
crossover. For each same-question Game/Neutral pair, only the feedback suffix's
downstream ordinary-attention K/V and GLA key/value/decay/write-strength updates
were exchanged; the source token's own residual stayed natural. The complete
suffix (`incorrect/lost . Choose the answer again .`) transferred 0.925
[0.910, 0.939] of the held-out donor task vector into Game and 0.941
[0.927, 0.955] into Neutral. It also transferred the opposing rank behavior:
Game switching fell from 62.7% to 47.0%, whereas Neutral switching rose from
45.0% to 60.6%.

The literal feedback word is not the sole source. On confirmation, its transfer
was 0.076 into Game but 0.393 into Neutral; the first period transferred
0.303/0.381 and contextualized `Choose` transferred 0.247/0.236. The suffix is
therefore a contextualized causal unit with asymmetric contributions rather
than a one-token label.

The corrected exhaustive path-specific interception then restored the outgoing
ordinary-attention and GLA writes of every later prompt region from an exact
natural cache while leaving each region's source-crossed residual intact. The
second instruction, repeated stem, four option lines, choice cue/query, and
final assistant prefix each mediated a replicated portion of policy transfer.
Restoring all five jointly removed 0.479 [0.455, 0.501] of Game-recipient
transfer and 0.390 [0.363, 0.415] of Neutral-recipient transfer—51.8% and 41.4%
of their respective source-only effects. No single downstream token region is
the policy bottleneck. The corrected `cache_restored_no_source_swap` scenario
executed the actual restoration-only intervention and reproduced natural A-D
logits with exactly 0.0 error. Moreover, every natural/source/interception
array shared with the historical run is bit-for-bit identical, so the audit
identified a vacuous advertised control rather than an incorrect mediation
estimate.

The surviving 48--59% is mechanistically informative. It does not require
re-encoding into the outgoing writes of any intervening token. It can persist
in GLA memory written at the feedback source, be retrieved directly from that
source by later ordinary attention, and/or travel through the short causal GLA
q/k/v convolution that the relay restoration does not intercept. This
experiment does not by itself separate these three bypass routes.

- [Corrected complete causal report](../evaluation_relay_final_mediation/relay_corrected_analysis/REPORT.md)
- [Feedback-source figure](../../../../figures/qwen36_feedback_source_localization.png)
- [Relay-mediation figure](../../../../figures/qwen36_feedback_relay_mediation.png)
- [Corrected machine-readable relay summary](../evaluation_relay_final_mediation/relay_corrected_analysis/summary.json)
- [Frozen plan](../evaluation_relay_final_mediation/PLAN.md)
