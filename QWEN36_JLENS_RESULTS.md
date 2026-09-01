# Qwen3.6-27B Jacobian-lens results on SimpleMC

## Design

The pretrained Neuronpedia Jacobian lens
`Qwen3.6-27B_jacobian_lens_n1000.pt` was applied to the exact self-hosted
Qwen3.6-27B checkpoint used in the previous SimpleMC mechanistic experiments
(revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`). The artifact contains 63
learned 5,120 x 5,120 transports, fitted on 1,000 WikiText prompts. Following
the official convention, learned lens layers 0--62 read post-block residuals
1--63; readout 64 is the model's actual final residual and unembedding.

Preflight checks established that:

- the artifact width and layer count match the checkpoint;
- the exact chat-template prompts and seven semantic token-position anchors
  can be collected in both conditions;
- applying the model's final norm and answer-token unembedding to the stored
  final residual reproduces the cached answer-token logits within 0.0042.

Two analyses were run:

1. All 500 questions in Baseline, Second Chance, and Neutral were read out at
   the final prompt position across every layer.
2. A fixed, answer-letter x Game-switch-stratified sample of 128 questions was
   read out across all layers at seven semantic positions: end of the first
   question, end of `[redacted]`, condition keyword, action keyword, end of the
   feedback sentence, end of the repeated question, and final decision.

The analysis examined both unrestricted vocabulary readouts and preregistered
single-token families for switch/change/different/alternative/other,
repeat/same/again/keep/continue, incorrect-feedback words, and lost-transmission
words. A-D scores aggregate the valid bare-letter and leading-space token
variants. This experiment is observational; no steering or intervention was
performed.

## Main result: an interpretable alternative-selection representation

At the final decision position, the Second Chance residual contains much more
switch-versus-repeat JLens content than the matched Neutral residual. The
paired contrast becomes consistently positive around readout 40, rises sharply
over readouts 43--47, and peaks at readout 47:

- mean Game-minus-Neutral switch-versus-repeat contrast: **1.454**;
- 95% CI: **[1.386, 1.522]**;
- the signal remains positive through readout 54 before becoming less stable
  in the final motor/output layers.

This is not driven only by the literal words *different* and *again*. At the
peak, the largest unrestricted Game-minus-Neutral vocabulary readouts at the
decision position are:

| Token/concept | JLens contrast |
|---|---:|
| `instead` | 4.81 |
| `failure` | 4.50 |
| `remaining` | 4.50 |
| `failed` | 4.44 |
| `other` | 4.44 |
| `alternative` | 4.22 |
| `unsuccessful` | 4.12 |

The prespecified concept scores agree. At readout 47, Game-minus-Neutral
contrasts are +3.15 for *alternative*, +2.91 for *change*, +2.89 for *other*,
and +1.72 for *switch*. The literal *different* contrast is only +0.85. Wrong,
incorrect, and mistake representations are also strongly positive.

## Where it appears

The direct readout at the action phrase is extremely large, as expected from
the lexical contrast between *different answer* and *again*. A related
alternative-selection representation is readable again at the final decision
position, but it is weak and incoherent at the sampled end of the repeated
question around the key readouts. The layer-47 decision signal is therefore not
merely the local residual of the word *different*, but these sparse samples do
not establish that one continuously readable signal persists between the two
locations.

The representation is strongest in a band centered around readouts 45--50.
That precedes the late answer commitment and overlaps the point at which the
JLens A-D trajectories begin to separate by condition.

### Unrestricted vocabulary dynamics

An interactive full-vocabulary explorer was added so that the interpretation
does not depend on the prespecified switch and repeat word families. At every
one of the seven sampled prompt positions and all 64 readouts, it displays the
largest positive and negative JLens-estimated vocabulary logits for Game,
Neutral, or their paired contrast.

The unrestricted Game-minus-Neutral ranking shows a suggestive temporal and
positional transition:

- At the end of the feedback sentence, `exclude` first enters the top 20 at
  readout 41 and is among the leading tokens over readouts 42--48, alongside
  `restrict`, `restriction`, and later rejection/banning-related tokens.
- At the final decision position, retry/change-related tokens become prominent
  over readouts 41--44 (`Changed`, `Retry`, `instead`, `change`). Over readouts
  46--50, the leading tokens shift toward `instead`, `other`, `alternative`,
  and `another`.
- At the end of the repeated question, the unrestricted top tokens are weak
  and generally incoherent around the same readouts. Consequently, these seven
  samples do not demonstrate that one continuously JLens-readable signal is
  carried from feedback to decision. They show an exclusion-like representation
  during feedback processing and an alternative-selection representation at
  decision time, with the intervening implementation unresolved.

This exploratory result was discovered from the unrestricted vocabulary
ranking rather than by choosing those semantic categories in advance.

### Expanded instruction-token explorer

The canonical interactive explorer was subsequently expanded from seven to
twelve prompt anchors. It retains the original positions and adds the exact
instruction tokens proposed for causal follow-up:

- system `incorrect`, `new`, and the `answer` in `new answer`;
- the final period of the shared system instruction, which follows the complete
  condition-specific system prefix in Game;
- user `different` in addition to the already-collected user `incorrect`,
  action-ending `answer`, and feedback-ending period.

Game-only tokens are shown only in the Game view. They are explicitly marked
unavailable in Neutral and Game-minus-Neutral rather than being paired with an
unrelated Neutral token. Shared anchors retain the paired comparison.

An initial unrestricted-vocabulary inspection identifies several distinct
onsets that can define the starting layers for direction-level intervention:

- At system `incorrect`, incorrect-related tokens enter the top 20 at readout
  14 and peak around readouts 44--45.
- At system `new`, new-related variants are already readable by readout 3;
  this very early local lexical readout should not by itself be interpreted as
  an instruction mechanism.
- At system `answer`, answer-related tokens become prominent around readout 38.
- At the shared system-ending period, exclusion-related tokens do not become
  prominent until readouts 53--59; the paired Game-minus-Neutral exclusion
  contrast is concentrated at readout 59.
- At user `different`, alternative-related tokens first become prominent around
  readouts 40--42 and peak around readout 47.
- At the feedback-ending period, retry-related tokens appear around readouts
  41--43, followed by restriction/exclusion/rejection-related tokens mainly
  after readout 48 and peaking around readouts 55--59.

These are top-20 entry points in the unrestricted JLens vocabulary ranking,
not causal findings. Their purpose is to delimit interpretable layer windows
for subsequent removal and insertion of the corresponding JLens directions.

A subsequent causal transmission experiment found that the feedback-ending
period does carry some condition-sensitive information to the later
alternative-selection representation, but that transmission occurs primarily
over readouts 17--40, before `exclude` becomes plainly readable at readout 41.
All-layer replacement mediated about 13% of the later representational gap but
no net Game switching. See `QWEN36_JLENS_EXCLUSION_TRANSMISSION_RESULTS.md`.

A later exhaustive direction-level intervention tested the feedback-end JLens
exclusion coordinate separately at every causally actionable readout L41--63.
Removing it from Game did not reduce switching, while inserting it into Neutral
usually reduced rather than increased switching, despite a successful JLens
manipulation check. See `QWEN36_JLENS_EXCLUSION_LAYERWISE_RESULTS.md`.

A subsequent answer-representation analysis exposed all four A-D trajectories
and compared them with question-specific option-text readouts. The A-D code is
highly reliable and shows a large late Game-specific redistribution away from
the original winner. The exact option-text readout is only modestly above
chance and cannot serve as a reliable semantic-answer decoder. See
`QWEN36_JLENS_ANSWER_REPRESENTATIONS.md`.

## Relationship to behavior

The signal is not merely a constant condition label. At readout 47:

- the paired Game-minus-Neutral signal predicts whether Game switches with
  AUC **0.676**;
- after calculating AUC separately within each prior-answer letter and then
  averaging, macro AUC is **0.641**, bootstrap 95% CI **[0.572, 0.709]**;
- within-letter AUCs are A = 0.643, B = 0.507, C = 0.742, and D = 0.670.

The B estimate has little power because only 4 of 72 prior-B trials switch in
Game. Thus some of the unadjusted predictive relationship reflects the model's
large answer-letter switching asymmetry, but a moderate question-level
relationship remains after controlling for prior letter.

The switch contingency is 159 Game-only switches, 8 Neutral-only switches, 142
switches in both conditions, and 191 switches in neither.

## A-D answer representations

JLens identifies a late, condition-specific loss of the original answer:

| Readout | Baseline | Second Chance | Neutral |
|---:|---:|---:|---:|
| 44 | -0.374 | -0.640 | -0.311 |
| 47 | -0.507 | -0.715 | -0.389 |
| 52 | +0.591 | -0.194 | +0.217 |
| 56 | +1.233 | +0.647 | +1.390 |
| 64 | +1.399 | +0.389 | +1.616 |

Values are the original Baseline answer's JLens score minus its strongest
competitor. The Game-specific divergence is visible before the final output
layers and is much larger than the Baseline-Neutral difference.

JLens does **not** show a clean broad reduction in total A-D spread throughout
this layer range. Around readouts 40--56, its Game A-D spread is often larger
than Neutral's, even while the original answer is selectively disadvantaged.
This is not a contradiction with the native-logit compression result: JLens
reads future-verbalizable residual content, not the actual next-token
distribution at that layer. It does mean that the new result is more naturally
described as an alternative-selection strategy representation plus delayed or
reduced original-answer commitment, rather than as a JLens measurement of
generic compression.

## Interpretation and limitation

This is the clearest interpretable correlate found so far for the computation
that distinguishes Second Chance from neutral repetition. The model represents
the feedback as a strategy of selecting an alternative, carries that
representation to the final decision position, and the strength of the paired
condition contrast moderately predicts switching.

It is not yet a demonstrated causal mechanism. JLens directions are defined
using average causal sensitivities, but observing a readout on a particular
trial does not prove that the represented strategy drives the later answer
change. A targeted J-space ablation or Game/Neutral coordinate swap around
readouts 45--50 would be the direct causal test if we decide to proceed.

## Artifacts

- Canonical figure:
  `outputs/mechanistic/qwen36_27b_jlens/analysis/preserved_figures/jlens_condition_representations.png`
- Canonical interactive full-vocabulary explorer:
  `outputs/mechanistic/qwen36_27b_jlens/analysis/preserved_figures/jlens_unrestricted_token_explorer.html`
- Final-position and multi-position JLens scores:
  `outputs/mechanistic/qwen36_27b_jlens/jlens_scores.npz`
- Expanded twelve-anchor JLens scores:
  `outputs/mechanistic/qwen36_27b_jlens_expanded/jlens_scores.npz`
- Expanded unrestricted top-token readouts and token audit:
  `outputs/mechanistic/qwen36_27b_jlens_expanded/top_tokens.json` and
  `outputs/mechanistic/qwen36_27b_jlens_expanded/position_audit.json`
- Unrestricted top-token readouts:
  `outputs/mechanistic/qwen36_27b_jlens/top_tokens.json`
- Position audit:
  `outputs/mechanistic/qwen36_27b_jlens/position_audit.json`
- Summary:
  `outputs/mechanistic/qwen36_27b_jlens/analysis/jlens_summary.json`
- Final metrics:
  `outputs/mechanistic/qwen36_27b_jlens/analysis/jlens_final_metrics.csv`
- Position metrics:
  `outputs/mechanistic/qwen36_27b_jlens/analysis/jlens_position_metrics.csv`
- Strategy concept decomposition:
  `outputs/mechanistic/qwen36_27b_jlens/analysis/jlens_strategy_concepts.csv`
- Behavioral diagnostics:
  `outputs/mechanistic/qwen36_27b_jlens/analysis/jlens_strategy_switch_diagnostics.csv`
- Collection code: `mechanistic/jlens_collect.py`
- Analysis code: `mechanistic/analyze_jlens.py`

The stopped Vast instance retains the exact model/lens cache and a reusable
1.1 GB position-residual cache. It was stopped after result retrieval and was
not destroyed.
