# Action-ending-period source lesion

## Question

Does the period ending `Choose the answer again.` add causal information beyond the state already present after the evaluation clause? The intervention preserves all pre-period state and removes only this period's GLA write, later ordinary-attention reads, or both.

An effect is always **lesioned minus natural within the named condition**. Percentage-point values are changes in trial-level answer selection, not changes relative to the other condition.

## Conflict trials (W1 != W2)

| Route removed | Condition | W1 selection | W1−W2 margin | Entropy |
|---|---|---:|---:|---:|
| `gla_write` | Evaluation | +0.7 [-1.8, +3.3] pp | +0.012 [-0.008, +0.032] | +0.010 [+0.004, +0.016] bits |
| `gla_write` | Matched Neutral | +0.7 [-1.5, +2.9] pp | +0.049 [+0.030, +0.067] | +0.006 [+0.001, +0.012] bits |
| `attention_read` | Evaluation | -0.4 [-2.6, +1.8] pp | -0.016 [-0.035, +0.002] | -0.014 [-0.020, -0.008] bits |
| `attention_read` | Matched Neutral | -1.5 [-4.8, +1.8] pp | -0.033 [-0.051, -0.014] | -0.001 [-0.007, +0.004] bits |
| `joint` | Evaluation | +0.7 [-1.5, +2.9] pp | +0.010 [-0.013, +0.034] | -0.007 [-0.014, +0.001] bits |
| `joint` | Matched Neutral | -0.4 [-3.3, +2.6] pp | +0.021 [-0.001, +0.045] | +0.008 [+0.002, +0.014] bits |

## No-conflict trials (W1 = W2)

| Route removed | Condition | W1 selection | W1 centered advantage | Entropy |
|---|---|---:|---:|---:|
| `gla_write` | Evaluation | +0.4 [-2.2, +3.1] pp | -0.023 [-0.043, -0.003] | +0.007 [-0.000, +0.014] bits |
| `gla_write` | Matched Neutral | +1.3 [-0.4, +3.5] pp | +0.021 [+0.004, +0.037] | -0.007 [-0.013, -0.002] bits |
| `attention_read` | Evaluation | -0.4 [-3.1, +2.2] pp | +0.048 [+0.031, +0.066] | -0.021 [-0.027, -0.015] bits |
| `attention_read` | Matched Neutral | +2.2 [+0.0, +4.8] pp | +0.005 [-0.016, +0.025] | -0.003 [-0.010, +0.004] bits |
| `joint` | Evaluation | +2.2 [-0.4, +5.3] pp | +0.036 [+0.015, +0.058] | -0.017 [-0.024, -0.009] bits |
| `joint` | Matched Neutral | +1.3 [-1.3, +4.0] pp | +0.012 [-0.012, +0.037] | -0.006 [-0.015, +0.002] bits |

## Bottom line

The corrected ordinary-attention intervention is not a null. With the source period's own residual output preserved, blocking later reads changes Neutral conflict-trial W1-minus-W2 margin by -0.033 [-0.051, -0.014]. That direction replicates almost exactly on discovery and confirmation: -0.033 [-0.061, -0.006] and -0.033 [-0.056, -0.008].

In Game no-conflict trials, the same read blockade instead raises W1's centered advantage by +0.048 [+0.031, +0.066], again with the same sign on discovery and confirmation: +0.043 [+0.018, +0.069] and +0.054 [+0.029, +0.079]. The action-ending period therefore is genuinely read by later ordinary attention, with task- and conflict-dependent consequences.

The effects remain modest: no pooled W1-selection change excludes zero, and the joint lesion does not reproduce the main Game-versus-Neutral behavioral gap. The supported conclusion is a small causal downstream source, not a dominant policy bottleneck. The former claim of exactly zero ordinary-attention impact was an instrument artifact.

## Validation

- 500 questions: 273 conflict and 227 no-conflict.
- Maximum same-batch natural deviation from trusted logits: 0.000000.
- Source-token local output preservation: `True` (from `run_metadata.json`).
- Exact historical batches of four and SDPA execution were preserved.
- Canonical figure: `figures/qwen36_action_period_source_lesion.png`.
- All condition-specific differences, letter strata, and frozen discovery/confirmation intervals are in `summary.json`.
