# Llama 3.1 405B observational mechinterp plan

## Scope

This is a model-specific refinement of `MECHINTERP_PLAN.md`, not a replacement for it. It preserves the original redacted Second Chance paradigm and the comparison hierarchy of Game versus ordinary baseline, with neutral as the strategic-specificity control.

The first pass is observational only. It uses the native logit lens and blockwise A–D updates. Probes, component attribution, activation patching, and steering are deferred unless the native lens is genuinely ambiguous.

## Why this model calls for a refined test

Llama 3.1 405B passed the paper's four behavioral tests on both SimpleMC and TriviaMC. Unlike the smaller Qwen results, its final A–D behavior is more consistent with selective movement away from the leading candidate than with broad entropy-increasing flattening. Behavioral switching is also concentrated on trials with smaller baseline top-two margins.

The leading candidates are therefore:

1. **Thresholded current-leader suppression:** once a candidate leads by enough, the Game instruction recruits an update against whichever option currently leads.
2. **Targeted original-winner suppression:** the Game instruction reconstructs and selectively penalizes the ordinary baseline winner, independent of the momentary leader.
3. **Generic compression plus perturbation:** Game block updates shrink the entire centered A–D evidence vector, with switching arising from the residual perturbation.
4. **Runner-up boosting:** a distinct positive update favors the baseline runner-up over ranks 3 and 4.
5. **Correctness-directed recomputation:** the trajectory moves toward the objectively correct option rather than merely away from the prior winner.

## Collection

For every usable SimpleMC and TriviaMC trial, collect baseline, Game, and neutral prompts using the same FP8 checkpoint, Llama chat template, prompts, and greedy answer-only generation as the validated self-hosted behavioral runs.

At the final answer-prediction position:

- capture the embedding readout and the true post-block residual stream after all 126 blocks;
- apply the model's actual final RMSNorm;
- apply the canonical and leading-space A–D unembedding rows;
- retain complete A–D pseudo-logits at 127 readouts;
- require the final native-lens choice to agree with the generated A–D answer.

The implementation uses vLLM's native compressed-tensors path because the generic Hugging Face loader incorrectly attempted to dequantize the FP8 checkpoint and exceeded GPU memory. Single-prompt scheduling is used for activation capture so vLLM's internal asynchronous prompt packing cannot misassociate trials and residual rows.

## Primary analyses

All scores are centered within question and readout before interpreting relative evidence.

1. Define each trial's fixed baseline ranks from its final baseline A–D scores.
2. Plot the baseline winner, runner-up, and lower-option trajectories in baseline, Game, and neutral.
3. Plot Game-minus-baseline and neutral-minus-baseline changes in:
   - A–D spread;
   - baseline-winner advantage over the other three;
   - runner-up advantage over ranks 3 and 4.
4. For every block, measure the update to the candidate currently leading immediately before that block, relative to the other three options.
5. Project every centered A–D block update onto the negative current evidence vector. A positive coefficient is the directly measured generic-compression component.
6. Fit the preregistered observational signatures already implemented in `mechanistic/hypothesis_analysis.py` using held-out questions: null, generic compression, fixed prior-winner penalty, thresholded current-leader penalty, and thresholded compression.
7. Event-align trials around held-out-thresholded emergence of the final baseline winner and ask whether Game-specific winner suppression follows that event.
8. Keep all trials as primary. Report Game-switch, Game-no-switch, neutral-switch, and neutral-no-switch strata only as secondary descriptions.

Every aggregate trajectory will include trial-bootstrap 95% confidence intervals. SimpleMC and TriviaMC will be analyzed separately before any cross-dataset synthesis.

## Discriminating predictions

- **Thresholded current-leader suppression:** the Game-minus-baseline relative update to the current leader becomes more negative only in high pre-block leader-margin bins, and the thresholded model wins out of sample.
- **Targeted original-winner suppression:** a fixed baseline-winner predictor outperforms dynamic-leader and compression predictors, including before or after the dynamic leader changes identity.
- **Generic compression:** the negative-evidence-vector projection explains the differential update across all four options, with no remaining selective current-leader penalty.
- **Runner-up boosting:** the baseline runner-up gains relative to ranks 3 and 4; centered rank-2 movement alone does not count.
- **Correctness-directed recomputation:** the objectively correct option gains relative to other options even when it is not the baseline runner-up.

These signatures can distinguish trajectory shapes and localize when the behavior is constructed. They do not by themselves establish a causal circuit. A causal follow-up is warranted only if this pass identifies a stable, interpretable layer range and candidate computation.
