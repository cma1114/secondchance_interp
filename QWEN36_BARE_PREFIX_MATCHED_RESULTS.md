# Qwen3.6-27B SimpleMC: exact bare-prefix rerun

Date: 2026-08-06

## Prompt invariant

This rerun manually serialized literal Qwen ChatML. It did not use the
Hugging Face chat template and did not insert `<think>` or `</think>`.

For every one of the 500 questions:

- Baseline, Game, and Neutral used the same system message.
- The first question message was identical in all three conditions.
- The rendered text and token IDs were identical through the first bare
  assistant-role header.
- Game and Neutral inserted `[redacted]` immediately after that shared header.
- The repeated question was identical to the Baseline question message.

The full audit, including an exact rendered example, is in
`outputs/mechanistic/qwen36_27b_simplemc_bare_chatml_matched/prompt_prefix_audit.json`.

With this genuinely bare prefix, the unrestricted next-token argmax was
`<think>` on all 128 freshly verified prompts. Therefore the experiment used
the established A--D-constrained decision rule: the response is the highest
scoring answer letter, while the unrestricted top token is separately logged.
No thinking token was added to the prompt.

## Behavioral results (500 questions per condition)

| Condition | Accuracy | Change from Baseline | Mean A--D entropy |
|---|---:|---:|---:|
| Baseline | 36.6% | -- | 1.448 bits |
| Game | 40.2% | 63.4% | 1.656 bits |
| Neutral | 43.2% | 45.8% | 1.378 bits |

- Lift: +17.6 percentage points, 95% CI [+13.6, +21.6]. The paired
  discordance was 103 Game-only changes versus 15 Neutral-only changes
  (one-sided exact p = 1.27e-17).
- Second choice: 232/317 Game changes (73.2%) went to the Baseline runner-up
  (one-sided exact p versus 1/3 = 1.49e-47).
- AccIncor: among changed Baseline-wrong trials, 114/221 (51.6%) moved to the
  correct answer (one-sided exact p versus 1/3 = 1.74e-8).
- Entropy preservation fails: Game minus Baseline is +0.208 bits, 95% CI
  [+0.178, +0.238]. Neutral minus Baseline is -0.071 bits.

The model therefore reproduces its established behavioral profile: it passes
Lift, Second Choice, and changed-trial AccIncor, but not the entropy-preserving
criterion.

## Layerwise results

The fixed-Baseline-rank JLens result is qualitatively robust to removing the
previously inserted thinking scaffold.

- Game-minus-Neutral starts visibly redistributing answer evidence in the late
  forties. Original-winner evidence is -0.063 JLens score units at L48.
- The complete inverse-rank ordering becomes stable at L52 and remains through
  L64: the original winner is affected most negatively, followed by the
  runner-up, rank 3, and rank 4.
- At L64 the Game-minus-Neutral vector is [-0.887, +0.079, +0.238, +0.570]
  for original ranks 1--4.
- The full late-layer Game-minus-Neutral trajectory correlates 0.973 with the
  earlier scaffolded run. Its final rank-1-minus-rank-4 contrast is larger in
  magnitude (-1.457 versus -1.157).
- The answer-letter readout remains much cleaner than the option-content
  readout. At L64, option-content Game-minus-Neutral is only
  [-0.153, +0.036, +0.077, +0.040].

The switch/alternative semantic contrast also replicates almost exactly. At
the final decision position it peaks at L47 at 1.688 [1.614, 1.762], versus
1.723 in the scaffolded run. At that layer, the strongest Game-pointing
concepts include `other` (+2.83), `alternative` (+2.76), and `change` (+1.94).
Those representations are already visible more weakly around L33.

However, the semantic contrast no longer predicts which individual questions
switch after controlling for the original answer letter: macro AUC 0.451,
95% CI [0.375, 0.524]. Thus it is a strong condition-level representation, not
a demonstrated trial-level switching mechanism.

The strange open-bracket token is still present near the natural final readout
in the unrestricted explorer. It is therefore not caused by the removed
thinking scaffold or by a mismatch before `[redacted]`. In this bare format the
model's unrestricted next token is `<think>`, so very-late unrestricted JLens
tokens should not be interpreted as clean answer-selection content.

## Canonical artifacts

- Behavioral report:
  `outputs/mechanistic/qwen36_27b_simplemc_bare_chatml_matched/analysis/BEHAVIORAL_REPORT.md`
- Three-panel fixed-rank figure:
  `outputs/mechanistic/qwen36_27b_jlens_bare_chatml_matched/analysis/preserved_figures/jlens_fixed_rank_contrasts.png`
- Four-panel JLens figure:
  `outputs/mechanistic/qwen36_27b_jlens_bare_chatml_matched/analysis/preserved_figures/jlens_condition_representations.png`
- English-glossed interactive vocabulary explorer:
  `outputs/mechanistic/qwen36_27b_jlens_bare_chatml_matched/analysis/jlens_unrestricted_token_explorer.html`
- Interactive answer-content explorer:
  `outputs/mechanistic/qwen36_27b_jlens_bare_chatml_matched_answer_content/analysis/jlens_answer_representation_explorer.html`
- Exact first-token verification:
  `outputs/mechanistic/qwen36_27b_jlens_bare_chatml_matched/analysis/first_answer_exact_verification.json`

Vast instance 46566562 was stopped, not destroyed. Its 2.18 GB positional
residual cache remains available for follow-up. The rerun used 54.2 minutes of
the A100 at $1.0389/hour (about $0.94), plus negligible glossary API cost.
