# Exact remapped source contributions and feedback-token swaps

## Bottom line

The late W1 difference is not supplied by direct reads from the literal
`incorrect`, `different`, or feedback-period tokens.  On the 273
conflict trials, Mixer 56 writes +0.168 centered W1 units in
Game but +1.000 in Neutral.  Most of that
-0.832 Game-minus-Neutral difference comes from the
states over the *second presentation itself*: the repeated question and its
option-boundary states
(-0.407), the identical
final `Your choice (A, B, C, or D):` cue
(-0.314), and the repeated
option containing W1
(-0.226).

This is best described as **feedback-conditioned reprocessing of the repeated
question**.  Neutral strongly reconstructs or reinstates W1 from the second
presentation; Game does much less of that.  The source decomposition is exact
and additive for the natural mixer write, but it is a computational
decomposition rather than an independent causal ablation of each source region.

The head-level result is concentrated but not single-head.  At Mixer 56, the
four largest head contrasts are H6
(-0.412), H15
(-0.189), H0
(-0.159), and H2
(-0.150).  At Mixer 52, H23
supplies -0.090 of the
-0.175 total contrast.

## Validation

- All 500 questions ran in the exact historical batch-of-four SDPA cohorts.
- Natural A--D logits match the trusted remapped run exactly (maximum absolute error 0.0).
- Per-token/per-head contributions reconstruct the actual gated attention context with maximum absolute error 0.0625.
- Primary semantic analysis uses the 273 W1 != W2 conflict trials.

## Source contributions

Each number is the exact additive contribution of a source region to the
mixer's residual write, projected onto the canonical W1 unembedding direction
and centered across A--D. It is not an attention weight alone. The contributions
sum across tokens and heads to the complete ordinary-attention write.

### Mixer 52: largest Game--Neutral source differences

| Source region | Game | Neutral | Game - Neutral (95% CI) | Largest head |
|---|---:|---:|---:|---:|
| Repeated question/option-boundary states | +0.0056 | +0.0519 | -0.0463 [-0.0599, -0.0339] | H23 (-0.0478) |
| Second choice cue | -0.0510 | -0.0134 | -0.0376 [-0.0438, -0.0317] | H23 (-0.0270) |
| Final assistant prefix | -0.0378 | -0.0071 | -0.0306 [-0.0386, -0.0228] | H19 (-0.0106) |
| Historical assistant boundary | +0.0580 | +0.0865 | -0.0285 [-0.0345, -0.0221] | H20 (-0.0139) |
| System prompt | -0.0341 | -0.0101 | -0.0240 [-0.0301, -0.0179] | H20 (-0.0145) |
| Repeated-presentation W1 option | +0.0565 | +0.0786 | -0.0221 [-0.0321, -0.0128] | H23 (-0.0107) |
| Repeated-presentation W2 option | -0.0103 | -0.0188 | +0.0085 [+0.0063, +0.0109] | H23 (+0.0063) |
| Second answer-only instruction | -0.0030 | -0.0105 | +0.0075 [-0.0007, +0.0158] | H5 (+0.0077) |

### Mixer 56: largest Game--Neutral source differences

| Source region | Game | Neutral | Game - Neutral (95% CI) | Largest head |
|---|---:|---:|---:|---:|
| Repeated question/option-boundary states | +0.1014 | +0.5081 | -0.4067 [-0.4889, -0.3246] | H6 (-0.1583) |
| Second choice cue | -0.0687 | +0.2451 | -0.3138 [-0.3918, -0.2409] | H6 (-0.1952) |
| Repeated-presentation W1 option | +0.4490 | +0.6749 | -0.2259 [-0.2741, -0.1797] | H6 (-0.0984) |
| Repeated-presentation W2 option | -0.0664 | -0.1081 | +0.0417 [+0.0332, +0.0513] | H15 (+0.0190) |
| First question stem | +0.0058 | -0.0230 | +0.0288 [+0.0151, +0.0426] | H0 (+0.0153) |
| Final assistant prefix | +0.0166 | +0.0429 | -0.0263 [-0.0547, +0.0027] | H6 (-0.0120) |
| First choice cue | -0.0026 | -0.0211 | +0.0184 [+0.0136, +0.0235] | H15 (+0.0049) |
| Other repeated options | -0.2433 | -0.2612 | +0.0178 [-0.0003, +0.0363] | H15 (+0.0132) |

### Exact-token audit of the three largest Mixer-56 regions

The region labels above are aggregates. Decoding the saved per-token
contributions shows that their -0.946 combined contrast is distributed as
follows:

- **Repeated question/option-boundary states (-0.407):** newline after option C
  -0.249; newline after option B -0.181; newline after option A +0.023; all
  actual question-word tokens together approximately -0.0001. The earlier
  friendly label “question stem” was therefore misleading.
- **Final `Your choice (A, B, C, or D): ` cue (-0.314):** the final space token
  -0.257; literal `A` -0.040; ` (` -0.018; commas -0.007; literal ` C` +0.006;
  remaining cue tokens approximately zero.
- **Repeated W1 option (-0.226):** the answer-content tokens together -0.135;
  the displayed option-label tokens account for most of the remaining -0.091
  (especially displayed `D`, -0.061, and `C`, -0.022).

These are contributions from the *contextual states located at those tokens*.
A newline or final space can summarize the prefix preceding it; the result does
not imply that whitespace has semantic meaning in isolation.

## Causal token-state swaps

The `incorrect`/`lost` result is a corrected rerun: the predecessor runner used
unpadded character-derived positions directly against left-padded batches, so
its intervention was misaligned on shorter rows. These results add each row's
actual padding offset before collecting or replacing states.

All three single-token state swaps are small relative to the natural
17.6-percentage-point W1-selection
gap on conflict trials (Game 20.9%, Neutral
38.5%).  The largest absolute W1-selection change from any
tested token, direction, or readout is 1.83 percentage
points.  None of the three token states at one late readout is therefore the
controller of the behavioral difference.

- `evaluation`, Neutral into Game: largest W1 effect -0.0072 at readout 48 (95% CI -0.0152 to +0.0006).
- `evaluation`, Game into Neutral: largest W1 effect -0.0062 at readout 51 (95% CI -0.0119 to -0.0005).
- `action_target`, Neutral into Game: largest W1 effect -0.0068 at readout 48 (95% CI -0.0144 to +0.0004).
- `action_target`, Game into Neutral: largest W1 effect -0.0026 at readout 52 (95% CI -0.0086 to +0.0034).
- `feedback_end`, Neutral into Game: largest W1 effect +0.0036 at readout 54 (95% CI -0.0022 to +0.0095).
- `feedback_end`, Game into Neutral: largest W1 effect -0.0084 at readout 54 (95% CI -0.0140 to -0.0029).

## Prespecified no-conflict check

The source pattern is not unique to W1 != W2 trials.  When the original and
fresh-remapped winners agree (n=227), the total W1-directed
writes remain much smaller in Game:

- Mixer 52: Game +0.162, Neutral +0.253, difference -0.091 (95% CI -0.132 to -0.048).
- Mixer 56: Game +0.970, Neutral +1.916, difference -0.946 (95% CI -1.099 to -0.802).

Single-token swaps are also small on these trials.  The largest absolute
W1-selection change across every token, direction, and readout is
1.76 percentage points
(`feedback_end`, Neutral into Game, readout
49).

## Artifacts

- `figures/qwen36_simplemc_remapped_mixer_source_contributions.png`
- `figures/qwen36_simplemc_remapped_feedback_token_state_swaps.png`
- `summary.json`
