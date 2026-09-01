# Qwen3.6-27B SimpleMC: canonical empty-history interpretability results

## Scope and prompt validation

This is the canonical mechanistic analysis for the empty historical assistant
response. Baseline, Game, and Neutral use explicit raw Qwen ChatML and the same
empty no-thinking scaffold. Game and Neutral contain no `[redacted]` token.

On 128 independently checked questions, the rendered text and token IDs were
identical across all three conditions through the first assistant decision
boundary. A fresh, unrestricted, batch-size-one forward pass reproduced the
saved Baseline first-token answer on 128/128 questions. Thus condition-specific
information cannot appear before the feedback is introduced.

## Final-position answer trajectories

The main JLens figure aligns the four answer choices by their fixed Baseline
rank separately for every question. Each curve is centered across A-D within
question before paired condition differences and confidence intervals are
computed.

- Game minus Baseline remains close to zero through approximately layer 49.
- Beginning around layers 50-52, Game creates a clean inverse-rank
  redistribution: it lowers the original winner most, leaves the runner-up
  approximately unchanged, and raises ranks 3 and 4, with the largest increase
  for rank 4.
- Neutral produces a substantially smaller and noisier transformation.
- Consequently, Game minus Neutral isolates a strong late rank-ordered effect.

Selected Game-minus-Neutral JLens contrasts:

| Readout | Original winner | Runner-up | Rank 3 | Rank 4 |
|---:|---:|---:|---:|---:|
| L48 | -0.075 | +0.009 | +0.041 | +0.024 |
| L52 | -0.332 | +0.050 | +0.124 | +0.157 |
| L56 | -1.182 | -0.002 | +0.444 | +0.740 |
| L60 | -1.312 | -0.101 | +0.471 | +0.942 |
| L64 | -0.853 | +0.067 | +0.290 | +0.496 |

The final-readout JLens balanced accuracy is 98.2% for Baseline, 93.2% for
Game, and 97.9% for Neutral. The answer-letter readout is therefore reliable
where the late redistribution appears. A separate option-text-content readout
is much weaker, but it shows the same qualitative late ordering: at L64,
Game-minus-Neutral is -0.344, -0.004, +0.108, and +0.240 for fixed Baseline
ranks 1 through 4.

## Instruction semantics and their transmission

The unrestricted JLens explorer asks which vocabulary tokens each residual
state points toward; it does not preselect a switch direction. The important
sequence is:

- At the final decision position, mild redo/alternative semantics first become
  visible around L33 (`again`, `another`, `instead`). By L36-L40, `other`,
  `another`, `new`, and `instead` dominate. The contrast is strongest around
  L44-L47 (`other`, `remaining`, `alternative`, `instead`).
- At the user prompt's action keyword, the strongest late-40s concepts are
  `other`, `different`, `exclude`, `new`, and `alternative`.
- At the end of the feedback sentence, `exclude` is strongest around L44-L48;
  by L53-L56 the representation points overwhelmingly toward `alternative`.
- At the end of the repeated question, a smaller representation progresses
  from `again/reconsider` near L48 to `different/new/revised` near L56.

The Game-minus-Neutral switch-versus-repeat concept contrast at the final
decision peaks at L47 (2.290, 95% CI [2.225, 2.355]), a few layers before the
answer-rank redistribution becomes large. However, this global instruction
signal barely predicts which individual Game trials actually switch (AUC
0.536; prior-letter-controlled macro AUC 0.458, 95% CI [0.391, 0.526]). It is
therefore best interpreted as a representation of the instruction, not as a
complete trial-level switching mechanism.

As required by causal masking and the shared prefix, Game-minus-Neutral is
exactly zero at the end of the first question, the first assistant decision,
the empty historical response, and the shared system prompt. The condition
contrast is strongest at the action keyword (peak L46), then the feedback end
(peak L53), and is transmitted more weakly to the repeated-question end and
the final decision position.

## Comparison with the earlier `[redacted]` analysis

Removing `[redacted]` makes the result cleaner rather than qualitatively
different. The final-decision condition signal is larger (peak 2.290 versus
1.723), and the late inverse-rank redistribution is stronger. For example, at
L60 Game-minus-Neutral changes from approximately
[-1.115, -0.027, +0.448, +0.695] with `[redacted]` to
[-1.312, -0.101, +0.471, +0.942] with the empty history.

At the same time, the condition signal becomes less predictive of individual
switches. That combination strengthens the interpretation that JLens is
showing a robust, general instruction representation followed by a distinct
question-specific answer redistribution.

## Canonical artifacts

- Fixed-rank JLens contrasts:
  `outputs/mechanistic/qwen36_27b_jlens_empty_history/analysis/preserved_figures/jlens_fixed_rank_contrasts.png`
- JLens condition and prior-position overview:
  `outputs/mechanistic/qwen36_27b_jlens_empty_history/analysis/preserved_figures/jlens_condition_representations.png`
- Interactive unrestricted-token explorer, with English glosses and four
  question-dependent `[Baseline rank 1]` through `[Baseline rank 4]`
  pseudo-tokens shown on the same score scale as ordinary vocabulary tokens:
  `outputs/mechanistic/qwen36_27b_jlens_empty_history/analysis/jlens_unrestricted_token_explorer.html`
- Interactive fixed-rank trajectories:
  `outputs/mechanistic/qwen36_27b_jlens_empty_history/analysis/rank_contrasts/jlens_fixed_rank_contrasts.html`
- Interactive answer-letter and option-content explorer:
  `outputs/mechanistic/qwen36_27b_jlens_empty_history_answer_content/analysis/jlens_answer_representation_explorer.html`
- Cross-fitted probe rank trajectories:
  `outputs/mechanistic/qwen36_27b_simplemc_empty_history_residuals/analysis/trajectories/probe_mechanism/simplemc_probe_rank_trajectories.png`
- Probe accuracy by layer:
  `outputs/mechanistic/qwen36_27b_simplemc_empty_history_residuals/analysis/trajectories/probe_mechanism/simplemc_probe_final_answer_accuracy.png`
- Exact first-answer validation:
  `outputs/mechanistic/qwen36_27b_jlens_empty_history/analysis/first_answer_exact_verification.json`
