# Existing-data audit: evaluation state to final decision

## Bottom line

No new model run was used. The saved tensors show that the evaluation-period
state is not merely generic noise. In the global deletion, the intact Game
write has its most negative centered effect on W1 in both conflict and
non-conflict trials. On non-conflict trials its rank-resolved effects are
-0.335 [-0.405, -0.271] for W1, +0.082 [+0.029, +0.140] for rank 2,
+0.124 [+0.069, +0.179] for rank 3, and +0.129 [+0.088, +0.171] for rank 4.
The entropy increase is therefore principally a consequence of depressing the
dominant W1 candidate, not evidence of isotropic noise.

The action-closing period is representationally downstream of that state. Its
JLens Game-minus-Neutral W1-advantage contrast is -0.242 [-0.313, -0.171] at L56 on
conflict trials but +0.107 [+0.016, +0.197] on non-conflict trials. It is also
-0.194 [-0.262, -0.125]
on trials where Game eventually switches, versus
+0.101 [+0.001, +0.203]
where it repeats W1. Thus the final-period A-D representation is behaviorally
aligned, not merely a condition-average vocabulary curiosity.

The evaluation-period source trace remains large in norm at the action period,
but Panels C-D show that its **direct W1-aligned component there is weak**. Its
per-question action-period direct readout has Spearman correlation only
+0.013
with the final causal W1 effect on conflict trials. The available state is
therefore high-dimensional: the explicit answer-targeted representation in the
complete action-period residual cannot be equated with a directly propagated
W1 vector from the earlier write.

![Canonical transmission figure](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_action_matched_period_transmission.png)

## Panel guide

- **A:** question-level JLens A-D evidence at both feedback periods. Negative
  values mean Game represents W1 less strongly than Matched Neutral.
- **B:** the norm of the exact evaluation-period GLA source trace at successive
  downstream prompt positions. This shows persistence, not answer identity.
- **C-D:** for each GLA and later position, the Game-minus-Neutral direct
  W1-versus-other contribution of the evaluation-period source trace. These are
  direct readouts, not additive whole-model causal effects.
- **E-F:** the actual causal effect of the intact evaluation-period write on
  final centered logits, computed as natural minus globally ablated. These are
  the decisive targeting panels.

## Targeting versus flattening

On conflict trials, Game's intact period-write effects by original Baseline
rank are W1 -0.228 [-0.269, -0.190], rank 2 +0.012 [-0.029, +0.055], rank 3
+0.078 [+0.042, +0.114], and rank 4 +0.138 [+0.107, +0.170]. On non-conflict trials
the corresponding Game effects are W1 -0.335 [-0.405, -0.271], rank 2
+0.082 [+0.029, +0.140], rank 3 +0.124 [+0.069, +0.179], and rank 4
+0.129 [+0.088, +0.171]. The W1 effect is directionally the strongest negative
effect in both subsets. This supports a targeted-W1 operation whose
distributional consequence is flattening when W1 is the current winner.

Matched Neutral does change the raw A-D logits. Its non-conflict centered rank effects, however, are W1
-0.004 [-0.035, +0.028], rank 2 +0.004 [-0.024, +0.031], rank 3
-0.019 [-0.045, +0.009], and rank 4 +0.019 [-0.012, +0.052]. This is why causal
comparisons must distinguish common offsets from candidate redistribution.
Crucially, these centered effects also resolve the apparently
large raw Neutral W1-logit response reported earlier: most of it is a common
A-D offset (-0.384 [-0.430, -0.334] for natural minus ablated on
non-conflict trials), which cannot change the relative A-D distribution. Once
that offset is removed, Neutral has essentially no rank-specific effect.

## What the final action period contributes

The current files establish three things:

1. The action period contains a strong readable exclusion policy in the mean
   full-vocabulary JLens.
2. Question-specific A-D evidence there is negative for W1 on conflict and
   eventual-switch trials but reverses or disappears on non-conflict and
   repeat trials (Panel A and `summary.json`).
3. The evaluation-period causal write remains present there in high-dimensional
   norm, but its directly readable W1 component is weak (Panels B-D).

They do **not** establish that the action-period update is necessary or
sufficient. The earlier standard-prompt all-layer residual replacement at the
final feedback period restored 13.6% of the winner-advantage gap, 23.4% of the
spread gap, and 26.0% of the entropy gap, but changed net switching by only
-1.2 percentage points with a confidence interval spanning zero. The current
action-matched final-period token-state swaps were likewise small. Thus the
best existing conclusion is that the action period is a behaviorally aligned
downstream decision state, not yet a demonstrated causal bottleneck.

## Limits

- Full-vocabulary JLens states were retained only as across-question means, so
  `exclude`-token strength cannot be correlated with individual switching
  without another run.
- Summed per-GLA source traces are descriptive direct readouts. Only the global
  write deletion and bidirectional transplant are whole-model causal tests.
- Intermediate-state collection slightly changes low-order SDPA numerics; the
  separate period-JLens report documents the 94.4%/96.6% A-D argmax agreement.

## Files

- Machine-readable summary: [summary.json](summary.json)
- Figure: [qwen36_action_matched_period_transmission.png](/Users/christopherackerman/repos/secondchance_interp/figures/qwen36_action_matched_period_transmission.png)
