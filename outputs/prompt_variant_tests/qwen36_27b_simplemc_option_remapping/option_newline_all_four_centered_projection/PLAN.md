# Corrected centered all-four option-newline projection

## Question

Does the affine candidate-value signal decoded at the four first-presentation
option-closing newlines causally contribute to relative option scoring, and is
that contribution specific to Game revision?

## Correction to the invalid predecessor

The predecessor projected the raw RMS-normalized residual along the probe
gradient. The probe, however, was defined only after subtracting a
displayed-letter-specific discovery mean. That predecessor therefore removed
large static A/B/C/D components and lacked a same-path cached-K/V identity
control. It is invalid for the intended causal question.

For displayed letter `l`, normalized residual `z`, fitted mean `mu_l`, and unit
score-gradient direction `u`, the corrected projection is

`z' = z - ((z - mu_l) · u)u`.

This sets only the centered candidate-value coordinate to zero while retaining
the fitted static letter mean along that direction.

## Frozen execution

- All 500 canonical remapped SimpleMC questions.
- Frozen 251-question discovery and 249-question confirmation split.
- Historical physical batches of four and SDPA implementation.
- Canonical action-matched `incorrect` Game and `lost` Neutral prompts.
- All four first-presentation option-closing newline positions.
- Every ordinary-attention block (4, 8, ..., 64), using its immediately
  preceding residual readout (3, 7, ..., 63).

## Modes

1. `natural`: no K/V replacement.
2. `identity_kv`: add the cached projected-minus-unprojected K/V delta, which
   is exactly zero. This exercises the same hook path and must reproduce
   `natural` exactly before full launch.
3. `project_centered`: add only the cached projected-minus-unprojected K/V
   delta to the model's live K/V entries. This cancels float16 cache
   reconstruction error instead of replacing live K/V with an approximation.

The exact one-cohort benchmark must verify literal newline anchors, prompt
hashes, natural output reproduction, identity-versus-natural A-D logits and
choices, near-zero post-projection centered scores, residual dose, complete
forward count, runtime, and projected cost. The full run may launch only if the
identity path is exactly neutral in A-D logits and choices.

## Prespecified analysis

Analyze discovery and confirmation separately, for all, W1 != W2 conflict, and
W1 = W2 no-conflict questions. Report Game and Neutral effects and their
difference-in-differences for W1/W2 choice, switching, W1-minus-W2 margin,
centered W1 evidence, A-D entropy/spread, and any answer change. The held-out
primary endpoint is the Game-minus-Neutral interaction in W1 choice on conflict
trials. Save one canonical PNG and preserve compact machine-readable outputs.

## Outcome

Complete on all 500 questions. The zero-delta identity path was bit-exact:
maximum A-D logit difference 0.0 and zero choice changes across 1,000
condition-question comparisons. On 136 held-out conflict questions, the
centered projection changed W1 choice by +1.5 points in Game and -0.7 in
Neutral (interaction +2.2, 95% CI [-2.2, +6.6]). W1-minus-W2 margin changed by
+0.020 logits in Game and -0.002 in Neutral (interaction +0.023, 95% CI
[-0.003, +0.049]); the discovery interaction was only +0.002. The decoded
one-dimensional coordinate affects some option scoring but is not established
as the condition-specific semantic suppression mechanism.
