# Evidence that Game switching is structured rather than pure noise

## Question

Does the `incorrect` condition merely make the answer unstable, or does Game
use the semantic first-presentation ranking to perform a structured revision?

The answer is: **Game strongly compresses the old ranking, and a held-out
causal intervention shows that it constructs part of this compression by
reading each candidate's semantically matching first-presentation option.**
Game also increases uncertainty, so the result supports structured revision
plus uncertainty. Here “structured” refers only to which semantic candidate
the model leaves and which candidate it selects next; correctness is not an
endpoint in this analysis.

## Behavioral target: semantic switching, not output-letter switching

In the remapping experiment, every answer's text moves to a different A--D
letter before the second decision. Game switches away from its earlier answer
content on 65.6%
of questions, versus
46.8%
in Neutral, a difference of
18.8
percentage points. But old-letter avoidance moves in the opposite direction:
Game is 9.0
points *less* likely than Neutral to avoid the literal letter used previously.
The extra switching therefore follows the earlier answer's **semantic content**,
not the character `A`, `B`, `C`, or `D` that previously named it.

That establishes what the behavioral effect targets. It does not by itself
distinguish a deliberate computation from confusion. The next two analyses do:
first describe the transformation entirely within Game, then intervene on the
specific semantic-history route that helps create it.

## Direct within-Game change from first decision to final decision

Candidates are aligned by semantic content after remapping and ranked by their
first-presentation aggregated A-D logits. These are raw, uncentered logits.

| First rank | Mean first logit | Mean Game final logit | Raw paired change | Change after removing the common four-logit shift |
|---|---:|---:|---:|---:|
| R1 | 24.934 | 22.820 | -2.114 [-2.262, -1.963] | -1.346 [-1.453, -1.239] |
| R2 | 23.705 | 22.524 | -1.181 [-1.318, -1.045] | -0.413 [-0.501, -0.323] |
| R3 | 22.751 | 22.341 | -0.410 [-0.543, -0.274] | +0.358 [+0.269, +0.451] |
| R4 | 21.673 | 22.305 | +0.632 [+0.496, +0.771] | +1.400 [+1.309, +1.494] |

The mean across the four raw logits falls by
0.769
logits. That common shift cannot affect the A-D softmax. Beyond it, R1 falls
another 1.346 logits, R2 falls
0.413, R3 rises 0.358, and R4 rises
1.400.

The first-presentation differences are not small merely because values such as
24.934 and 21.673 look close on their absolute scale. The relevant quantity is
their difference: the mean R1--R4 gap is 3.261 logits. Applied
to those mean logits, that is an unnormalized softmax-weight ratio of
`exp(3.261) = 26.1`. At the Game final
decision the mean gap has contracted to 0.515 logits, a ratio of
only 1.7. The per-question probabilities below avoid
the additional distortion that would come from softmaxing logits only after
averaging different questions.

The probability presentation is often easier to read. For each question, take
a softmax over its four aggregated A-D logits and then average the resulting
probabilities:

| First rank | Mean first-presentation probability | Mean Game final probability |
|---|---:|---:|
| R1 | 58.6% | 31.5% |
| R2 | 24.0% | 25.5% |
| R3 | 12.0% | 22.9% |
| R4 | 5.3% | 20.1% |

Thus Game changes the average old-rank profile from
**58.6/24.0/12.0/5.3%** to
**31.5/25.5/22.9/20.1%**. This is an aggressive
compression of the original ranking, with the previous winner losing the most
and the previous weakest candidate gaining the most.

## The second presentation reconstructs fresh candidate evidence

The option-list computation is not merely carrying old history forward. A
fresh-score target was defined from a standalone Baseline that saw only the
remapped second presentation. For each semantic candidate, its centered A--D
logit was residualized against its old 1P score and both displayed positions.
The resulting target therefore asks how strongly the current presentation
supports that candidate independently of old evidence and letter placement.

At each 2P option line, the analysis averaged the residual over only the
wordpieces spelling that option's text, excluding its displayed letter,
punctuation, spacing, and newline. On the 249 held-out questions, the complete
residual's correlation with unique fresh evidence rises across layers 29--31:

| Residual after layer | Shared correlation | Game | Neutral |
|---:|---:|---:|---:|
| 28 | 0.109 | 0.109 | 0.109 |
| 29 | 0.143 | 0.142 | 0.144 |
| 30 | 0.168 | 0.166 | 0.170 |
| 31 | 0.191 | 0.191 | 0.192 |
| 32 | 0.181 | 0.182 | 0.180 |

The exact additive MLP writes at those same option-text wordpieces are also
fresh-evidence aligned:

| MLP write | Game correlation | Neutral correlation |
|---:|---:|---:|
| 29 | 0.149 | 0.153 |
| 30 | 0.182 | 0.189 |
| 31 | 0.188 | 0.182 |

This is additive-write and held-out decoding evidence, not an individual-MLP
lesion. A separate fresh-evidence crossover supplies the causal link: holding
the target candidate, 1P history, feedback, and target 2P position fixed while
crossing the four complete 2P option-line states changes the target final logit
in the donor-predicted direction by
+0.426
in Game and
+0.558
in Neutral on confirmation.

The layerwise directions were fitted separately, so the rising correlations
are not a literal cumulative sum in one fixed residual coordinate. Their
near-identity in Game and Neutral also means this is primarily shared fresh
solving, not the distinctive Game policy. It matters to the strategic account
because it shows that the model genuinely recomputes current candidate quality
before combining it with the Game-specific treatment of old rank; the final
switch is not generated by undirected noise alone.

## Destination selection: the fresh winner, not the old runner-up

Leaving the old winner and selecting a destination are different operations.
To separate them, this analysis uses the 135 questions where the standalone
second-presentation winner differs from both the old first-presentation winner
and the old first-presentation runner-up. “Fresh winner” here is the model's
semantic answer to the standalone remapped second presentation; it includes
whatever content and displayed-order preferences determine that answer. No
correct-answer label enters the definition or analysis.

The fixed-denominator columns report choices over the same pre-intervention
question set. The final three columns condition descriptively on an actual
switch away from the old winner. Because an intervention can change which
questions switch, causal comparisons across intervention conditions should use
the fixed-denominator columns; the switch-conditional columns describe where
the resulting switches landed.

| Split | Task | State | Eligible questions | Switch rate | Fresh winner / all | Old runner-up / all | Fresh winner / switches | Old runner-up / switches | Fresh minus old runner-up / switches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Discovery | Game | Natural | 65 | 78.5% | 53.8% | 15.4% | 68.6% | 19.6% | +49.0 [+25.5, +70.6] |
| Discovery | Game | Matching-history blockade | 65 | 63.1% | 27.7% | 24.6% | 43.9% | 39.0% | +4.9 [-24.4, +31.7] |
| Discovery | Neutral | Natural | 65 | 56.9% | 35.4% | 13.8% | 62.2% | 24.3% | +37.8 [+10.8, +64.9] |
| Discovery | Neutral | Matching-history blockade | 65 | 60.0% | 24.6% | 26.2% | 41.0% | 43.6% | -2.6 [-30.8, +25.6] |
| Confirmation | Game | Natural | 70 | 85.7% | 54.3% | 21.4% | 63.3% | 25.0% | +38.3 [+16.7, +60.0] |
| Confirmation | Game | Matching-history blockade | 70 | 71.4% | 35.7% | 30.0% | 50.0% | 42.0% | +8.0 [-20.0, +34.0] |
| Confirmation | Neutral | Natural | 70 | 61.4% | 34.3% | 21.4% | 55.8% | 34.9% | +20.9 [-7.0, +48.8] |
| Confirmation | Neutral | Matching-history blockade | 70 | 72.9% | 28.6% | 34.3% | 39.2% | 47.1% | -7.8 [-33.3, +17.6] |
| Pooled | Game | Natural | 135 | 82.2% | 54.1% | 18.5% | 65.8% | 22.5% | +43.2 [+27.0, +58.6] |
| Pooled | Game | Matching-history blockade | 135 | 67.4% | 31.9% | 27.4% | 47.3% | 40.7% | +6.6 [-12.1, +25.3] |
| Pooled | Neutral | Natural | 135 | 59.3% | 34.8% | 17.8% | 58.8% | 30.0% | +28.7 [+8.8, +48.8] |
| Pooled | Neutral | Matching-history blockade | 135 | 66.7% | 26.7% | 30.4% | 40.0% | 45.6% | -5.6 [-24.4, +13.3] |

In natural Game, the destination preference is large and replicates: among
switches it favors the fresh winner over the old runner-up by 49.0 points on
discovery and 38.3 points on confirmation. Pooled natural Game selects the
fresh winner on 65.8% of switches and the old runner-up on 22.5%. Natural
Neutral also favors the fresh winner when pooled, although its held-out
difference is less precise.

Under the matching-history blockade, Game still switches on 67.4% of the 135
eligible questions, but its pooled switch destinations are 47.3% fresh winner
and 40.7% old runner-up: a +6.6-point difference with a confidence interval of
[-12.1, +25.3]. Neutral's point estimate reverses. The blockade therefore does
not support a claim that fresh-winner steering remains strong after semantic
history is removed. It also does not establish that fresh computation itself
was removed; the intervention was designed to cut matching history, not to
isolate the fresh representation.

### Choice-level redirection in the existing fresh-state crossover

The earlier fresh-state crossover supplies a separate causal test. For each
question it has two second-presentation orderings with the target candidate in
the same displayed position. The intervention installs the opposite ordering's
complete 2P option-line outgoing state. The table below restricts to questions
where the two unmodified orderings naturally select different semantic answers
and averages the two reciprocal directions within question.

“Donor adoption” means the crossed run selected the answer naturally selected
by the opposite ordering. The generic-change null asks how often that specified
answer would be reached if the swap merely caused an undirected change: one of
three alternatives, so the expected rate is one third of the observed
any-change rate. A stricter leave-one-out null preserves the empirical
donor-answer frequencies conditional on the recipient's natural answer. The
exact duplicate-natural execution path has 0.0 logit error and 100% choice
agreement.

| Split | Task | Discordant questions | Donor adoption | Any change | Donor adoption among changes | Excess over equal-alternative null | Excess over frequency-matched null |
|---|---|---:|---:|---:|---:|---:|---:|
| Discovery | Game | 126 | 22.2% [17.9, 26.6] | 51.2% | 43.4% [37.3, 49.3] | +5.2 [+2.1, +8.3] | +4.9 [+1.7, +8.1] |
| Discovery | Neutral | 122 | 28.3% [23.0, 33.6] | 65.2% | 43.4% [37.7, 49.3] | +6.6 [+2.6, +10.4] | +6.7 [+2.9, +10.7] |
| Confirmation | Game | 132 | 24.2% [19.7, 28.8] | 58.3% | 41.6% [35.9, 47.0] | +4.8 [+1.5, +8.0] | +5.2 [+1.9, +8.5] |
| Confirmation | Neutral | 124 | 27.0% [22.2, 31.9] | 61.3% | 44.1% [38.4, 49.7] | +6.6 [+3.1, +9.9] | +6.7 [+3.1, +10.3] |

The choice redirection is modest but replicating. On confirmation, the donor
choice is adopted in 24.2% of Game directions and 27.0% of Neutral directions.
Among directions whose answer changes, donor adoption is 41.6% in Game and
44.1% in Neutral, above the 33.3% undirected-change expectation. This extends
the established donor-aligned logit movement to discrete choices. It shows
that current-presentation state causally structures destinations; it remains a
movability result, not a removal test establishing necessity.

### Causal removal of the decoded fresh-2P component

A new 500-question removal test targets the frozen, discovery-fitted component
of each second-presentation option-line state that uniquely predicts fresh
current-presentation score after geometric removal of the old-score direction.
It scrubs semantic wordpieces and their closing newlines after every layer
L1--64, immediately restores the old-score coordinate at each edit, and
compares against deterministic random directions with the exact same
candidate-wise L2 dose. The intervention leaves 1.16% of the targeted fresh
coordinate while changing the old coordinate by only 0.0020 on average. Native
natural and complete-sequence identity logits reproduce with exactly 0.0
error.

On the 136 held-out W1 != W2 questions, the Game-minus-Neutral old-W1-avoidance
gap is 22.8 points [15.4, 30.9] under the dose-matched random edit and 17.6
points [10.3, 25.0] under the fresh scrub. The prespecified difference is -5.1
points [-13.2, +2.9]. Thus the decoded unique-fresh component is **not shown
necessary** for preferential Game switching. The same scrub does causally
lower old-W1 centered logit advantage in both tasks relative to random (Game
-0.080 [-0.156, -0.003]; Neutral -0.121 [-0.203, -0.040]), so it is not a
behaviorally inert edit; its effect is primarily shared rather than the source
of the Game-specific gap.

The predicted clean destination loss also does not appear. On the fixed 70-
question held-out destination set, fresh-W2 choice changes by -7.1 points
[-17.1, +2.9] in Game but +5.7 [-4.3, +15.7] in Neutral relative to the random
control. The Game decrease is directionally compatible with fresh steering but
is individually uncertain, while the opposite Neutral sign rules out a simple
task-shared necessity result.

When the matching-history route is already blocked, the fresh scrub has a
small sign-reversed effect: relative to matching-plus-random it raises the
Game-minus-Neutral avoidance interaction by 7.4 points [0.7, 14.0] and raises
Game fresh-W2 choice by 10.0 points [2.9, 18.6]. This establishes a nonlinear
interaction with the history-lesioned computation, not necessity for the
natural Game advantage. The narrow conclusion is therefore that the tested
decoded fresh subspace is causally active but does not supply the distinctive
preferential-switching effect. Other distributed forms of recomputation remain
outside this scrub's scope. See the [complete causal report](../../fresh_history_double_dissociation/analysis/REPORT.md),
[machine-readable summary](../../fresh_history_double_dissociation/analysis/summary.json),
and [canonical figure](../../../../../figures/qwen36_fresh_history_double_dissociation.png).

### Direct question-stem rereading is used, but it restrains switching

A complete 500-question 2x2 edge intervention blocks ordinary-attention reads
of the original question wording after the first decision, the repeated
question wording during later 2P processing, or both, across every ordinary-
attention layer L4, L8, ..., L64. It leaves both sets of option lines and all
GLA recurrent state untouched.

On held-out W1!=W2 conflicts, blocking the original question stem changes Game
old-W1 avoidance by only -0.7 points [-6.6, +5.2], but raises Neutral avoidance
by 14.0 points [+6.6, +22.1]. Discovery independently gives +2.2 and +15.3
points. Thus direct 1P-question rereading is causally used, but it is not
required for preferential Game switching; its clearest role is to stabilize
the prior winner under Neutral.

Blocking the repeated question stem reduces the held-out fresh-evidence
alignment at 2P semantic wordpieces in the middle/late computation (at L48,
-0.211 in Game and -0.244 in Neutral), yet fresh-W2 choice increases by 11.0
points [+4.4, +18.4] in Game and 8.8 [+1.5, +16.2] in Neutral. Direct access to
the repeated question wording therefore contributes to the decoded fresh
state but is not necessary to choose the freshly favored candidate.

Blocking both copies shrinks the held-out Game-minus-Neutral avoidance gap from
22.1 to 5.9 points because Neutral avoidance rises by 21.3 points; Game does
not lose its old-winner avoidance. The best-supported interpretation is that
question-stem rereading supplies a stabilizing/reconsideration signal rather
than the distinctive Game suppression policy. This intervention does not
erase question information already embedded in option states or GLA memory,
so distributed recomputation remains possible. See the [complete report](../../question_stem_access_factorial/analysis/REPORT.md),
[machine-readable summary](../../question_stem_access_factorial/analysis/summary.json),
and [canonical figure](../../../../../figures/qwen36_question_stem_access_factorial.png).

### Where the policy is present at 2P—and what that does not prove

A separate same-question Game/Neutral crossover shows that task policy is
already present candidate by candidate in the complete outgoing state of the
2P semantic wordpieces. Crossing all four candidates' semantic states transfers
19.6%
of the donor task-specific answer-score pattern into Game and
24.3%
into Neutral on held-out conflict questions. Crossing only one candidate's
semantic state preferentially changes that same candidate: every R1--R4
target-minus-off-target interval is positive.

This establishes **where** candidate-specific policy has become causally
effective: at the tokens spelling the 2P answers. It does not establish that
the policy transforms the newly computed fresh score. Stage C crossed each
token's complete state and could not independently exchange old evidence while
holding fresh evidence fixed. Together with the nearly identical L29--31 fresh
trajectories, the best-supported interpretation is narrower: the 2P semantic
state combines shared fresh evidence with a task-specific treatment of
retrieved old evidence. Neutral reinstates the old ranking more strongly;
Game compresses it. A distinctive Game transformation of fresh evidence itself
remains unestablished.

## Strongest evidence: the within-Game causal semantic-history test

The before/after profile alone could partly reflect a fresh solve, regression
to the mean, or generic flattening. The decisive test blocks, across ordinary-
attention layers 4--48, each repeated candidate's reads of its truly matching
first-presentation option line. An equally sized cyclic wrong-line block is
the control. Positive values below mean that removing the true match raises
the candidate, so the intact Game route had been suppressing it.

| First rank | Matching-specific lesion effect on that candidate's centered advantage |
|---|---:|
| R1 | +0.343 [+0.182, +0.502] |
| R2 | +0.160 [+0.032, +0.292] |
| R3 | +0.011 [-0.092, +0.120] |
| R4 | -0.160 [-0.246, -0.069] |

With the semantic-history route intact, Game therefore suppresses R1, suppresses
R2 more weakly, is approximately neutral for R3, and supports R4. Jointly
blocking all four matching routes raises held-out conflict-trial R1 choice by
**9.6 percentage points**
`[1.5, 17.6]`.

This is the strongest evidence against pure answer noise: cutting a specific
semantic-history pathway predictably reverses part of the rank transformation
inside Game. Noise added only at the final decision would not depend on whether
a repeated candidate can read its matching earlier option line.

The matched remapped-versus-non-remapped attention comparison sharpens what
that pathway means. In both prompt versions, the complete question and all four
answer texts are already present again in 2P, yet the 2P option lines devote
approximately 31--32% of their ordinary attention to the four 1P option lines.
Remapping barely changes that total. It instead spreads the fixed history read:
the semantic match receives 15.6% in non-remapped Game versus 12.3% when
remapped, and 15.7% versus 12.4% in Neutral. Even after remapping separates
semantic identity from displayed letter and position, the semantic match beats
the same-letter old line by 3.8 percentage points in both tasks.

Because no question or answer text is missing from 2P, this old-line access is
not required merely to recover the problem statement. Attention alone remains
observational, but the balanced matching-line lesions above establish causal
use of candidate-specific first-pass history. Taken together, the evidence is
best described as **recollection or reuse of prior candidate evaluation**—old
candidate evidence and rank—alongside separately constructed fresh 2P evidence.
It is not evidence that the model must look backward simply to solve the same
visible question again.

## What is and is not established

Established:

- The preferential behavioral switch follows the earlier answer's semantic
  content after remapping, not its old output letter.
- Fresh candidate evidence is reconstructed in the 2P option-text residuals
  and causally affects the final candidate logits.
- Game lowers the previous winner's raw final logit by 2.114 on average.
- The change is rank-structured rather than equal across candidates.
- Semantic matching to first-presentation candidates causally produces part of
  that structure.
- The broad 2P-to-1P option-line read is present with or without option
  remapping. Since 2P already repeats the complete problem, its combination
  with the causal matching-line result supports reuse of prior candidate
  evaluation rather than recovery of missing question or answer text.
- The resulting operation is well described as **old-rank compression**:
  suppress old leaders and relatively support old weak candidates.

Not established:

- That the model represents an explicit symbolic rule such as `not W1`.
- That all of the compression is carried by the matching route; fresh second-
  presentation evidence and broader uncertainty also contribute.
- That fresh recomputation as a whole is necessary for destination selection.
  The new removal test strongly deletes the frozen decoded unique-fresh
  coordinate but does not produce a clean task-shared destination loss; it does
  not exhaust every distributed form of recomputation.
- That the operation is entropy-free. Game increases A-D entropy and weakens
  late sharpening, so generalized uncertainty is a real secondary component.

## Evidence classes and provenance

The first-to-final tables are a post hoc paired analysis of existing natural
logits; they describe what Game does but do not by themselves identify its
cause. The remapping result is behavioral evidence about the target of the
switch. The matching-edge effects are prespecified held-out causal lesions
with balanced wrong-line controls. Input paths and SHA-256 hashes are recorded
in `summary.json`.
