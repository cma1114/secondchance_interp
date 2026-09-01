# Mixer 56 historical-endpoint edge ablation

On the frozen 249-question SimpleMC confirmation set, remove the single
historical-answer-endpoint to final-decision attention edge from every head in
Mixer 56. The selected attention logits are set to negative infinity before
softmax, so the endpoint contributes exactly zero and the remaining sources are
renormalized normally. No residual activation, other source position, other
query position, or other layer is changed.

Run natural and ablated forwards separately in Game and Neutral. The primary
outcomes are the change in switching away from the matched live self-hosted
Baseline answer and the Game-minus-Neutral difference in that causal effect.
Secondary outcomes are changes in the Baseline-winner logit contrast, its A-D
probability, A-D entropy, and the rate at which the intervention changes the
natural final answer.
