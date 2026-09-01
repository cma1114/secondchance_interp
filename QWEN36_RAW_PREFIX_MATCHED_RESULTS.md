# Qwen3.6-27B SimpleMC: raw, prefix-matched rerun

## What was corrected

This rerun does not call Hugging Face `apply_chat_template`. It serializes the
system, user, and assistant roles explicitly. Baseline, Game, and Neutral use
the same generic system prompt and the exact same capabilities-test wording,
including placing the response-only instruction before the question. The
rendered and tokenized Baseline prompt is exactly the Game and Neutral prefix
immediately before `[redacted]` on all 500 questions.

Qwen3.6-27B requires an assistant no-thinking scaffold: without one, its
unrestricted next token was `<think>` on every one of 24 preflight questions.
The scaffold is therefore written explicitly and identically in all three
conditions rather than being silently inserted by the HF template.

There is one consequential limitation. After seeing a historical
`[redacted]`, the unrestricted model sometimes predicts `[` at the final
assistant turn. The analysis therefore treats the response as an explicitly
A-D-constrained one-token decision, aggregating the saved bare and
leading-space token variants for each letter. This is labeled in the run
metadata as `decision_mode: ad_constrained`; the unrestricted invalid outputs
were not silently recoded.

## Behavioral and A-D probability results

| Condition | Accuracy | Switch rate | Mean A-D entropy |
|---|---:|---:|---:|
| Baseline | 41.6% | -- | 1.359 bits |
| Game | 39.8% | 34.8% | 1.672 bits |
| Neutral | 43.2% | 19.8% | 1.421 bits |

- Game-minus-Neutral switching: **+15.0 points**, 95% paired bootstrap CI
  **[+10.6, +19.6]**.
- Of 174 Game switches, **128 (73.6%)** go to the Baseline runner-up.
- On changed, Baseline-wrong Game trials, **55/110 (50.0%)** move to the
  correct answer.
- Game-minus-Baseline entropy: **+0.314 bits**
  [0.281, 0.347].
- Neutral-minus-Baseline entropy: **+0.062 bits**
  [0.043, 0.082].
- Game-minus-Neutral entropy: **+0.251 bits**
  [0.221, 0.283].

The corrected formatting therefore retains the qualitative behavioral profile
but reduces the switch-rate difference from the earlier HF-template,
Baseline-matched run (+24.0 points) to +15.0 points.

## Unrestricted open-bracket audit

A fresh full-vocabulary forward pass was run on the exact saved prompt for all
500 questions in each condition. The table uses the literal ordering returned
by the model's top-k operation; probabilities are softmax probabilities over
the complete vocabulary, not renormalized A-D probabilities.

| Condition | `[` is top token | `[` is in top 4 | Mean `[` probability | Mean total A-D probability |
|---|---:|---:|---:|---:|
| Baseline | 0/500 (0.0%) | 0/500 (0.0%) | <0.00001% | 99.995% |
| Game | 135/500 (27.0%) | 410/500 (82.0%) | 20.10% | 79.47% |
| Neutral | 17/500 (3.4%) | 315/500 (63.0%) | 10.35% | 89.58% |

The Game-Neutral differences are strongly paired within question: `[` is the
top token only in Game on 124 questions, only in Neutral on 6, and in both on
11. It is in the returned top four only in Game on 127 questions, only in
Neutral on 32, and in both on 283. Its probability is higher in Game on
412/500 questions, with a mean paired difference of +9.75 probability points.

Because the forward pass uses bfloat16, some output logits are exactly tied.
If every token tied for a probability rank is included, rather than taking the
literal four-token top-k list, `[` is tied for first on 150 Game and 23 Neutral
trials and has rank at most four on 417 Game and 328 Neutral trials. It remains
absent from Baseline's top ten on every trial. In both second-chance conditions
it is in the top ten on all 500 trials. Thus the invalid-token issue is not a
rare decoding accident: the historical `[redacted]` creates a large bracket
continuation tendency, especially in Game, and A-D filtering removes a
condition-dependent share of the natural next-token distribution.

## Validation before `[redacted]`

- Initial and repeated question messages match Baseline on **500/500** trials.
- Rendered prefixes and token IDs match Baseline through the explicit
  no-thinking scaffold on **500/500** trials.
- On the fixed 128-question JLens sample, a fresh batch-size-one full-vocabulary
  forward pass at this exact shared prefix matches the saved Baseline top token
  on **128/128** trials. Game and Neutral are therefore identical to Baseline
  at the first answer boundary, not merely similar under an A-D readout.

An earlier 120/128 check was based on unembedding cached FP16 residuals and
occasionally flipped near-tied argmaxes. It has been replaced by the fresh
full-precision model-forward validation above.

## Layerwise result

The corrected JLens fixed-rank contrast gives the clearest account. Values
below are centered option-score changes ordered by the Baseline ranks
`[winner, runner-up, rank 3, rank 4]`.

| Readout | Game - Baseline | Game - Neutral |
|---:|---|---|
| 48 | [-0.091, +0.107, +0.043, -0.059] | [-0.053, -0.018, +0.036, +0.035] |
| 52 | [-0.458, +0.048, +0.183, +0.227] | [-0.219, +0.031, +0.093, +0.095] |
| 56 | [-0.889, -0.285, +0.332, +0.842] | [-1.092, +0.139, +0.441, +0.513] |
| 64 | [-0.958, -0.123, +0.349, +0.731] | [-0.754, +0.075, +0.275, +0.403] |

The substantive, rank-ordered redistribution starts around readout 52, not
before the first answer. By readout 56, Game strongly lowers the Baseline
winner relative to Neutral and raises ranks 2--4 in inverse Baseline order.
This is more specific than undifferentiated noise, although it is still an
observational readout rather than a causal circuit identification.

The independent centroid-probe view agrees qualitatively: from roughly
readout 52 onward, Game has much less original-winner advantage than Baseline
or Neutral. At readout 56, balanced probe accuracy for the condition's eventual
answer is 79.5% in Baseline, 58.7% in Game, and 74.4% in Neutral. JLens is the
cleaner readout here and reaches approximately 100% balanced accuracy at the
final layer.

## Instruction representation

JLens finds a large Game-minus-Neutral switch/alternative representation after
the feedback, peaking at final-decision readout 47 (1.723 [1.666, 1.780]). At
the action-keyword endpoint it peaks around readout 46, and at the feedback
endpoint around readout 53. The representation is only weakly predictive of
which Game trials actually switch: letter-controlled macro AUC 0.603
[0.538, 0.671]. It is therefore a condition/instruction representation, not by
itself an explanation of individual switching.

## Canonical artifacts

- Behavioral report and values:
  `outputs/mechanistic/qwen36_27b_simplemc_raw_chatml_matched/analysis/`
- JLens report, exact prefix audit, and static condition figure:
  `outputs/mechanistic/qwen36_27b_jlens_raw_chatml_matched/analysis/`
- Canonical static three-panel fixed-rank figure:
  `outputs/mechanistic/qwen36_27b_jlens_raw_chatml_matched/analysis/preserved_figures/jlens_fixed_rank_contrasts.png`
- Interactive fixed-rank Game-Baseline, Neutral-Baseline, and Game-Neutral
  panels:
  `outputs/mechanistic/qwen36_27b_jlens_raw_chatml_matched/analysis/rank_contrasts/jlens_fixed_rank_contrasts.html`
- Interactive unrestricted-token JLens explorer:
  `outputs/mechanistic/qwen36_27b_jlens_raw_chatml_matched/analysis/jlens_unrestricted_token_explorer.html`
- Interactive option-letter/content trajectories:
  `outputs/mechanistic/qwen36_27b_jlens_raw_chatml_matched_answer_content/analysis/jlens_answer_representation_explorer.html`
- Rebuilt native-lens and cross-fitted-probe figures:
  `outputs/mechanistic/qwen36_27b_simplemc_raw_chatml_matched/analysis/trajectories/`
- Full-vocabulary open-bracket audit summary and per-trial tables:
  `outputs/mechanistic/qwen36_27b_simplemc_raw_chatml_matched/analysis/bracket_audit/`

The Vast instance was stopped, not destroyed, after retrieval. GPU uptime for
this corrected batch was about 51 minutes at $1.039/hour, approximately $0.89
plus negligible network charges.
