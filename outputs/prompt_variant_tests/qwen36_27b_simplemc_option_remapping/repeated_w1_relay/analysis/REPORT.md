# Repeated-W1 downstream relay localization

## Bottom line

This experiment asks whether Game's suppression of W1 is carried by ordinary attention from the repeated W1 option line into intermediate, pre-final states. It is not. Blocking those reads makes W1 *less* likely, so this pathway normally supports W1 reinstatement rather than suppressing it.

On held-out conflict trials, blocking every later pre-final read from the repeated W1 line changed Game W1 choice by -11.76 [-17.65, -6.62] percentage points and W1−W2 margin by -0.62 [-0.75, -0.49] logits. The corresponding discovery effects were -21.90 [-28.47, -16.06] points and -0.57 [-0.69, -0.45] logits.

Neutral depended even more strongly on this pathway: the held-out changes were -30.15 [-38.24, -22.79] points and -1.01 [-1.21, -0.83] logits. Thus the lesion increased Game-minus-Neutral W1 choice by +18.38 [+10.29, +26.47] points and the W1−W2 margin by +0.40 [+0.27, +0.52] logits—not because it recovered W1 in Game, but because it removed substantially more W1 reinstatement from Neutral.

This rules out the prespecified hypothesis that a pre-final repeated-W1 relay carries an active anti-W1 signal in Game. The gated depth-band run was therefore not performed. Together with the earlier original-line→repeated-line lesion, the cleaner interpretation is differential reinstatement: the repeated line provides pro-W1 evidence in both conditions, while Game weakens or negatively contextualizes that evidence relative to Neutral.

## Held-out conflict decomposition

- Later repeated-option queries, Game W1 choice: +2.21 [+0.00, +5.15] points; margin: -0.00 [-0.04, +0.04] logits.
- Post-options pre-final queries, Game W1 choice: -11.03 [-16.91, -5.88] points; margin: -0.60 [-0.74, -0.48] logits.
- Post-options W1-line lesion minus matched-line lesion, Game W1 choice: -18.38 [-25.00, -12.50] points; margin: -0.79 [-0.94, -0.64] logits.

## Held-out no-conflict context

- All later pre-final reads, Game W1 choice: -38.05 [-46.90, -29.20] points.
- All later pre-final reads, Neutral W1 choice: -49.56 [-58.41, -40.71] points.

## Validation

Same-batch natural choices matched the trusted run on `988/1000` condition-question outputs (98.8%).
Maximum trusted-logit discrepancy: `0.125`. All causal effects use the same-batch natural companion.
Maximum absolute intervention-induced A-D logit change: `4.412989`.

Canonical figure: `figures/qwen36_repeated_w1_relay.png`.
