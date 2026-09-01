# Qwen3.6-27B conditional source-route results

## Bottom line

We can now say where Mixer 56 distributes ordinary attention and which prompt
token writes survive in Mixer 63's recurrent state. We did **not** find a
single, clean feedback-to-answer source route that explains the coordinated
eight-component mechanism.

Mixer 56 attends broadly to structural and historical parts of the prompt. Its
candidate answer-changing routes come mainly from the repeated answer options
and final choice cue, but their held-out causal effects are small and do not
have consistent signs in both intervention directions.

Mixer 63 has a much sharper source localization: its largest condition-sensitive
writes come from the final assistant prefix, including the second empty-thinking
scaffold and final query token. Those writes are causally consequential, but
they are heterogeneous. Some push in the Game direction, some oppose it, and
several large Game-minus-Neutral differences arise because Neutral reinforces
the Baseline winner more strongly—not because Game adds a new suppressive
write. Thus Mixer 63 is a late condition-sensitive transformation site, not yet
the upstream mechanism that reads the incorrect-feedback instruction.

## Experiment

- Model: `Qwen/Qwen3.6-27B`, pinned revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Prompt: corrected `baseline_matched_empty_history` raw Qwen ChatML.
- Discovery: the frozen 251-question set.
- Confirmation: the untouched frozen 249-question set.
- Every prompt token was assigned to exactly one of 21 semantic spans.
- Mixer 56 (zero-based layer 55) used exact final-query attention-edge removal
  and renormalization, retaining Qwen's learned output gate and output
  projection.
- Mixer 63 (zero-based layer 62) used exact Gated DeltaNet recurrence replay
  with beta writes removed for one source span and value head.
- Discovery selected eight positive Game-minus-Neutral ordered-rank routes for
  each mixer. Confirmation removed each route while the other seven known
  components and every other route remained patched.
- Confirmation used both Game-into-Neutral insertion and Neutral-into-Game
  removal, with matched fixed-batch numerical controls and paired bootstrap
  95% confidence intervals.

The full method is frozen in `SOURCE_ROUTE_CAUSAL_PLAN.md`.

## Where Mixer 56 attends

Across its 24 heads, the largest mean fractions of final-query attention were:

| Source span | Game | Neutral |
|---|---:|---:|
| Historical assistant turn | 25.1% | 23.9% |
| Final assistant prefix | 16.3% | 15.7% |
| System prompt | 10.7% | 8.0% |
| Other ChatML structure | 7.8% | 9.5% |
| First choice cue | 7.4% | 6.8% |
| First question stem | 4.4% | 6.1% |
| Second choice cue | 3.9% | 4.2% |
| Repeated question stem | 3.7% | 3.6% |

Attention mass alone is not the causal result: values and the learned output
gate determine what the attended information writes. The strongest
Game-minus-Neutral answer-rank writes came from heads 0, 6, and 15 reading the
repeated options and second choice cue. Direct routes from the feedback subject,
feedback condition, and feedback action were much smaller at Mixer 56. The best
head for each had discovery differentials of only 0.0011, 0.0013, and 0.0032
ordered-rank units, respectively.

## Mixer 63's recurrent source writes

All eight discovery-selected Mixer 63 routes came from the final assistant
prefix. Their immediate ordered-rank writes reveal two different phenomena:

| Head | Game write | Neutral write | Game−Neutral |
|---:|---:|---:|---:|
| 2 | −0.0008 | −0.0638 | +0.0629 |
| 36 | +0.0421 | +0.0196 | +0.0225 |
| 46 | +0.0008 | −0.0204 | +0.0212 |
| 20 | −0.0399 | −0.0546 | +0.0147 |
| 42 | −0.0382 | −0.0520 | +0.0137 |
| 32 | +0.0133 | +0.0022 | +0.0112 |
| 6 | +0.0072 | −0.0033 | +0.0105 |
| 19 | +0.0245 | +0.0149 | +0.0096 |

Positive writes oppose the Baseline answer-rank pattern: they tend to lower
rank 1 and/or raise lower-ranked answers. Negative writes reinforce that
pattern. Heads 36, 32, and 19 therefore look like quantitatively stronger Game
compression. Heads 2 and 46 primarily reflect the absence in Game of a Neutral
winner-reinforcing write. Heads 20 and 42 reinforce the winner in both
conditions, but more strongly in Neutral.

## Held-out causal confirmation

As a calibration, the complete eight-component intervention still worked under
eager attention. It inserted 55.3% [46.4, 64.5] and removed 46.7% [39.6, 54.0]
of the continuous ordered-rank gap. It changed switching by +10.04 percentage
points [4.82, 15.26] in Neutral and −12.45 points [6.43, 18.47] in Game. The
natural eager-attention Game-minus-Neutral switch gap was 11.65 points.

### Mixer 56

No selected attention route made a reliable positive contribution to the
continuous Game-like redistribution in both directions.

- The largest positive insertion effect was head 0 reading repeated option C:
  1.91% [0.59, 3.29] of the natural rank gap.
- Head 15 reading repeated option B instead had a negative insertion effect of
  −1.25% [−2.24, −0.29], while its reciprocal estimate was +1.00%
  [0.08, 1.91].
- Several Game-into-Neutral route removals changed discrete switching by
  2.8–3.6 points despite tiny continuous effects, consistent with argmax
  threshold crossings. Their reciprocal estimates were null or negative.

This is evidence that repeated-option/cue routes can affect output selection in
the transplanted Game computation, but not that one stable attention path
drives Mixer 56's natural condition difference.

### Mixer 63

Mixer 63 routes had larger continuous effects but strongly disagreed across
directions.

- Head 46 contributed +4.05% [2.68, 5.40] on Game insertion but −2.47%
  [−3.62, −1.34] on removal.
- Head 20 contributed −13.99% [−17.01, −10.96] on insertion but +12.65%
  [10.03, 15.29] on removal.
- Head 42 similarly gave −4.86% [−6.17, −3.51] and +4.01%
  [2.73, 5.30].
- Heads 36 and 6 had modest positive insertion effects; their reciprocal
  intervals included zero.

The large sign reversals are not statistical noise. Game and Neutral carry
different absolute recurrent writes through the same head and prefix span. A
Game route and the corresponding Neutral route are therefore not interchangeable
instances of one fixed “compression route.” This is precisely why the
Game-minus-Neutral discovery score did not translate into reciprocal causal
mediation.

## Interpretation

The new information is useful but narrower than a full circuit:

1. Mixer 56 is genuinely integrating repeated answer-option and choice-cue
   information at the final decision, but no single selected attention route
   explains its condition-sensitive role.
2. Mixer 63's condition-sensitive computation is concentrated in recurrent
   writes made at the final assistant prefix/current query. It mixes genuine
   Game-opposing writes with reduced Neutral winner reinforcement.
3. The explicit feedback words are not carried into these two late mixers by a
   dominant direct source edge. The instruction must already have altered the
   residual state reaching the final assistant prefix, or arrive through
   distributed/redundant routes.
4. Consequently, these results do not justify presenting one source route as
   the primary mechanism. They refine the localization from “late mixers” to
   “Mixer 56 reads repeated answer content; Mixer 63 applies a
   condition-sensitive local recurrent transformation at the final prefix.”

The natural next experiment is to split the final assistant prefix token by
token and trace the condition-sensitive input state entering Mixer 63. That
would distinguish the structural assistant boundary, empty-thinking tokens,
and final query token, and would test where the condition information is
already present before Mixer 63 acts.

## Artifacts

- `outputs/causal/qwen36_27b_source_routes_corrected/discovery_analysis/source_route_screen.png`
- `outputs/causal/qwen36_27b_source_routes_corrected/discovery_analysis/source_route_screen.csv`
- `outputs/causal/qwen36_27b_source_routes_corrected/discovery_analysis/confirmation_plan.json`
- `outputs/causal/qwen36_27b_source_routes_corrected/confirmation_analysis/conditional_source_route_effects.png`
- `outputs/causal/qwen36_27b_source_routes_corrected/confirmation_analysis/conditional_source_route_effects.csv`
- `outputs/causal/qwen36_27b_source_routes_corrected/confirmation_analysis/conditional_source_route_summary.json`

## Compute and instance state

The preserved A100 instance ran for 44.4 minutes at $1.0389/hour, approximately
$0.77. It was stopped after the compact results were retrieved and was not
destroyed.
