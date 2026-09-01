# Grouped feedback-source crossover

## Question and intervention

The seven-token policy suffix was divided into two prespecified groups: the feedback sentence `incorrect/lost | .` and the following identical instruction `Choose | the | answer | again | .`. Each group and their complete union were reciprocally crossed between Game and Neutral across all 16 ordinary-attention layers and all 48 GLA layers. Source-token outputs were preserved, so the intervention changes downstream ordinary-attention K/V and recurrent GLA k/v/g/β writes rather than replacing the source token's own residual.

The nonlinear interaction is `complete suffix − feedback sentence − following instruction`. Positive values indicate synergy; negative values indicate subadditivity or redundant information.

## Replication control

The new complete-suffix corrected logits differ from the established complete-suffix run by at most 0 logits (bit-exact: True).

## Confirmation results

| Task | Component | Transfer (95% CI) | Fraction of complete (95% CI) |
|---|---|---:|---:|
| Game | Feedback sentence | 0.643 [0.600, 0.683] | 69.5% [64.9%, 73.7%] |
| Game | Following instruction | 0.450 [0.413, 0.487] | 48.7% [44.8%, 52.3%] |
| Game | Separate effects summed | 1.093 [1.027, 1.158] | 118.2% [111.4%, 124.7%] |
| Game | Complete suffix | 0.925 [0.909, 0.939] | 100.0% [100.0%, 100.0%] |
| Game | Complete − sum | -0.168 [-0.229, -0.106] | -18.2% [-24.7%, -11.4%] |
| Neutral | Feedback sentence | 0.665 [0.626, 0.703] | 70.6% [66.6%, 74.7%] |
| Neutral | Following instruction | 0.575 [0.535, 0.618] | 61.0% [56.8%, 65.6%] |
| Neutral | Separate effects summed | 1.239 [1.184, 1.297] | 131.7% [125.8%, 137.3%] |
| Neutral | Complete suffix | 0.941 [0.927, 0.955] | 100.0% [100.0%, 100.0%] |
| Neutral | Complete − sum | -0.298 [-0.352, -0.244] | -31.7% [-37.3%, -25.8%] |

## Discovery results

| Task | Component | Transfer (95% CI) | Fraction of complete (95% CI) |
|---|---|---:|---:|
| Game | Feedback sentence | 0.698 [0.663, 0.737] | 74.8% [70.9%, 78.9%] |
| Game | Following instruction | 0.471 [0.441, 0.497] | 50.4% [47.3%, 53.4%] |
| Game | Separate effects summed | 1.169 [1.123, 1.217] | 125.2% [119.9%, 130.3%] |
| Game | Complete suffix | 0.933 [0.921, 0.946] | 100.0% [100.0%, 100.0%] |
| Game | Complete − sum | -0.236 [-0.283, -0.187] | -25.2% [-30.3%, -19.9%] |
| Neutral | Feedback sentence | 0.656 [0.627, 0.686] | 69.9% [67.1%, 72.9%] |
| Neutral | Following instruction | 0.531 [0.485, 0.575] | 56.6% [51.8%, 61.1%] |
| Neutral | Separate effects summed | 1.187 [1.143, 1.233] | 126.4% [122.3%, 130.8%] |
| Neutral | Complete suffix | 0.939 [0.923, 0.955] | 100.0% [100.0%, 100.0%] |
| Neutral | Complete − sum | -0.248 [-0.290, -0.209] | -26.4% [-30.8%, -22.3%] |

## Scope

The intervention covers every applicable ordinary-attention and recurrent GLA write layer. As in the established complete-suffix source crossover, it does not patch Qwen3.6's short causal GLA q/k/v convolution state; estimates therefore concern the measured outgoing K/V and delta-rule write channels.

Raw switch rates, W1 rates, bivalent changes, discovery results, and validation controls are retained in `summary.json`.
