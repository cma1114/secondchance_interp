# Fixed-A selected-option K/V layer localization

## Bottom line

The direct semantic read is sharply localized to **ordinary-attention block
44**, but the complete Game-versus-Neutral difference is a cooperative
computation spanning blocks 4--48 rather than a block-44-only mechanism.

Patching only the selected-option K/V at block 44 moved the final answer toward
the donor semantic answer in both conditions, much more strongly in Neutral:

- discovery: Game -0.142 logits, Neutral -0.465, Game-minus-Neutral +0.324
  [+0.258, +0.393];
- confirmation: Game -0.174 logits, Neutral -0.592, Game-minus-Neutral +0.418
  [+0.347, +0.496].

No other individual block produced a comparably large replicated positive
Game-minus-Neutral effect. Block 28 showed a smaller replicated effect in the
opposite direction (-0.106 discovery; -0.122 confirmation), which is useful
evidence that the result is not a generic consequence of patching any K/V.

The four-block band 36--48 was independently sufficient for substantial donor
reinstatement and condition differentiation. On confirmation it moved the
semantic margin by -0.242 logits in Game and -0.790 in Neutral, a +0.548-logit
condition difference [+0.449, +0.656]. Discovery gave the same pattern
(-0.191, -0.609, and +0.418 respectively).

The leave-one-band-out results show why block 44 is not the whole mechanism.
Omitting blocks 20--32 reduced the all-layer Game-minus-Neutral effect by 1.503
logits on confirmation; omitting 4--16 reduced it by 0.985, and omitting 36--48
reduced it by 0.735. Yet the early bands were weak when transplanted alone.
They therefore act as nonlinear enabling context for the later semantic read,
not as independently sufficient semantic readers. Omitting 52--64 changed the
effect by -0.007 logits, with a confidence interval spanning zero; this final
quarter is dispensable.

The all-layer positive control replicated the preceding source-localization
experiment despite replacement-host numerical differences: Game-minus-Neutral
was +2.163 versus the prior +2.187 on discovery and +2.544 versus +2.603 on
confirmation. Thus the strongest supported mechanism is: selected-option
semantic K/V is read directly at block 44, with blocks 4--32 establishing the
condition-dependent computational context that makes Neutral reinstate it much
more than Game. This remains a fixed-literal-`A` result and does not establish
letter-general operation.

## Validation

- Discovery: 57/64 exact-regime questions.
- Identity versus prior source run maximum A-D error: 0.874945 logits.
- All-layer selected-option versus prior source run maximum A-D error: 1.37452 logits.
- Cached identity versus unsplit natural answer differences: 13.
- Confirmation: 64/73 exact-regime questions.
- Identity versus prior source run maximum A-D error: 0.996878 logits.
- All-layer selected-option versus prior source run maximum A-D error: 0.874678 logits.
- Cached identity versus unsplit natural answer differences: 12.

## Discovery layer effects

| Cell | Blocks | Game margin | Neutral margin | Game − Neutral | Game donor chosen | Neutral donor chosen |
|---|---|---:|---:|---:|---:|---:|
| identity | none | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.0 [+0.0, +0.0] pp | +0.0 [+0.0, +0.0] pp |
| block_04 | 4 | +0.041 [+0.010, +0.073] | +0.025 [-0.018, +0.065] | +0.015 [-0.029, +0.064] | +0.0 [+0.0, +0.0] pp | +1.8 [+0.0, +4.4] pp |
| block_08 | 8 | +0.003 [-0.051, +0.056] | +0.006 [-0.047, +0.058] | -0.003 [-0.061, +0.059] | +0.9 [-1.8, +3.5] pp | +3.5 [+0.9, +7.0] pp |
| block_12 | 12 | -0.021 [-0.067, +0.025] | -0.070 [-0.126, -0.021] | +0.049 [-0.011, +0.118] | +0.9 [-1.8, +4.4] pp | +4.4 [+0.9, +7.9] pp |
| block_16 | 16 | -0.007 [-0.036, +0.019] | -0.001 [-0.031, +0.028] | -0.006 [-0.040, +0.028] | +0.0 [-2.6, +2.6] pp | +0.0 [+0.0, +0.0] pp |
| block_20 | 20 | -0.017 [-0.046, +0.010] | +0.000 [-0.026, +0.027] | -0.018 [-0.050, +0.015] | -0.9 [-4.4, +2.6] pp | +0.9 [+0.0, +2.6] pp |
| block_24 | 24 | -0.003 [-0.025, +0.020] | +0.014 [-0.005, +0.034] | -0.017 [-0.045, +0.009] | +0.9 [-1.8, +4.4] pp | +1.8 [+0.0, +4.4] pp |
| block_28 | 28 | -0.063 [-0.124, +0.004] | +0.043 [-0.043, +0.134] | -0.106 [-0.171, -0.037] | -2.6 [-6.1, +0.0] pp | +1.8 [+0.0, +4.4] pp |
| block_32 | 32 | -0.004 [-0.031, +0.021] | +0.000 [-0.028, +0.029] | -0.005 [-0.039, +0.029] | -1.8 [-5.3, +1.8] pp | +1.8 [+0.0, +4.4] pp |
| block_36 | 36 | -0.033 [-0.061, -0.007] | -0.031 [-0.059, -0.005] | -0.003 [-0.029, +0.026] | +0.9 [-1.8, +4.4] pp | +0.9 [+0.0, +2.6] pp |
| block_40 | 40 | -0.025 [-0.045, -0.006] | -0.018 [-0.037, +0.001] | -0.007 [-0.029, +0.016] | -1.8 [-4.4, +0.0] pp | +0.0 [+0.0, +0.0] pp |
| block_44 | 44 | -0.142 [-0.202, -0.082] | -0.465 [-0.561, -0.375] | +0.324 [+0.258, +0.393] | +0.9 [-1.8, +4.4] pp | +7.0 [+2.6, +11.4] pp |
| block_48 | 48 | -0.013 [-0.026, +0.000] | +0.006 [-0.009, +0.021] | -0.019 [-0.038, -0.000] | +0.9 [+0.0, +2.6] pp | +0.0 [+0.0, +0.0] pp |
| block_52 | 52 | -0.003 [-0.018, +0.012] | +0.009 [-0.003, +0.021] | -0.012 [-0.033, +0.008] | +0.9 [+0.0, +2.6] pp | +0.0 [+0.0, +0.0] pp |
| block_56 | 56 | -0.019 [-0.035, -0.003] | +0.000 [-0.013, +0.013] | -0.019 [-0.039, +0.002] | -0.9 [-3.5, +1.8] pp | +0.0 [+0.0, +0.0] pp |
| block_60 | 60 | -0.005 [-0.021, +0.011] | +0.003 [-0.009, +0.014] | -0.008 [-0.028, +0.013] | -0.9 [-2.6, +0.0] pp | +0.0 [+0.0, +0.0] pp |
| block_64 | 64 | -0.003 [-0.013, +0.006] | +0.007 [-0.003, +0.016] | -0.010 [-0.024, +0.004] | +0.0 [+0.0, +0.0] pp | +0.0 [+0.0, +0.0] pp |
| band_04_16 | 4,8,12,16 | +0.179 [+0.023, +0.341] | +0.067 [-0.071, +0.208] | +0.112 [+0.011, +0.213] | +0.0 [-4.4, +5.3] pp | +5.3 [+1.8, +9.6] pp |
| band_20_32 | 20,24,28,32 | +0.097 [-0.062, +0.267] | +0.079 [-0.080, +0.236] | +0.018 [-0.102, +0.144] | -0.9 [-4.4, +2.6] pp | +7.0 [+2.6, +12.3] pp |
| band_36_48 | 36,40,44,48 | -0.191 [-0.269, -0.117] | -0.609 [-0.730, -0.493] | +0.418 [+0.339, +0.502] | +0.9 [-1.8, +4.4] pp | +6.1 [+2.6, +10.5] pp |
| band_52_64 | 52,56,60,64 | -0.012 [-0.024, +0.001] | -0.005 [-0.019, +0.010] | -0.007 [-0.025, +0.011] | -1.8 [-4.4, +0.0] pp | +0.0 [+0.0, +0.0] pp |
| without_band_04_16 | 20,24,28,32,36,40,44,48,52,56,60,64 | -0.668 [-0.909, -0.440] | -1.980 [-2.314, -1.673] | +1.313 [+1.095, +1.539] | +7.9 [+2.6, +14.0] pp | +28.1 [+19.3, +36.8] pp |
| without_band_20_32 | 4,8,12,16,36,40,44,48,52,56,60,64 | -0.307 [-0.480, -0.133] | -1.264 [-1.481, -1.063] | +0.957 [+0.782, +1.142] | +6.1 [+0.0, +13.2] pp | +18.4 [+11.4, +25.4] pp |
| without_band_36_48 | 4,8,12,16,20,24,28,32,52,56,60,64 | +0.053 [-0.241, +0.352] | -1.508 [-1.837, -1.181] | +1.561 [+1.349, +1.783] | +2.6 [-3.5, +9.6] pp | +24.6 [+16.7, +32.5] pp |
| without_band_52_64 | 4,8,12,16,20,24,28,32,36,40,44,48 | -0.725 [-1.043, -0.421] | -2.885 [-3.239, -2.542] | +2.159 [+1.897, +2.433] | +12.3 [+5.3, +19.3] pp | +43.0 [+33.3, +52.6] pp |
| all_selected_option | 4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64 | -0.711 [-1.033, -0.404] | -2.874 [-3.226, -2.535] | +2.163 [+1.902, +2.435] | +13.2 [+6.1, +21.1] pp | +41.2 [+31.6, +50.9] pp |

### Discovery leave-one-band-out necessity

| Omitted band | Game contribution | Neutral contribution | Game − Neutral contribution |
|---|---:|---:|---:|
| band_04_16 | -0.043 [-0.214, +0.127] | -0.894 [-1.072, -0.720] | +0.850 [+0.669, +1.033] |
| band_20_32 | -0.404 [-0.645, -0.171] | -1.610 [-1.867, -1.374] | +1.206 [+1.028, +1.392] |
| band_36_48 | -0.764 [-0.905, -0.630] | -1.366 [-1.572, -1.164] | +0.602 [+0.464, +0.748] |
| band_52_64 | +0.014 [-0.001, +0.030] | +0.011 [-0.004, +0.025] | +0.004 [-0.016, +0.023] |

## Confirmation layer effects

| Cell | Blocks | Game margin | Neutral margin | Game − Neutral | Game donor chosen | Neutral donor chosen |
|---|---|---:|---:|---:|---:|---:|
| identity | none | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.0 [+0.0, +0.0] pp | +0.0 [+0.0, +0.0] pp |
| block_04 | 4 | +0.054 [+0.012, +0.096] | +0.056 [+0.007, +0.106] | -0.002 [-0.061, +0.058] | -2.3 [-7.0, +3.1] pp | -0.8 [-3.9, +1.6] pp |
| block_08 | 8 | +0.074 [+0.022, +0.128] | +0.048 [+0.001, +0.096] | +0.026 [-0.040, +0.092] | +0.0 [-3.9, +3.9] pp | +0.0 [-2.3, +2.3] pp |
| block_12 | 12 | +0.055 [+0.003, +0.110] | +0.037 [-0.013, +0.086] | +0.018 [-0.053, +0.089] | +1.6 [-2.3, +6.2] pp | +0.0 [-3.1, +3.1] pp |
| block_16 | 16 | +0.008 [-0.026, +0.043] | +0.021 [-0.007, +0.050] | -0.013 [-0.051, +0.025] | +1.6 [-1.6, +4.7] pp | +0.0 [-2.3, +2.3] pp |
| block_20 | 20 | -0.023 [-0.058, +0.011] | -0.043 [-0.083, -0.006] | +0.020 [-0.017, +0.057] | +2.3 [-0.8, +5.5] pp | +0.0 [+0.0, +0.0] pp |
| block_24 | 24 | -0.007 [-0.036, +0.025] | -0.016 [-0.038, +0.008] | +0.009 [-0.021, +0.038] | +0.8 [-1.6, +3.9] pp | +0.8 [+0.0, +2.3] pp |
| block_28 | 28 | -0.050 [-0.123, +0.034] | +0.072 [-0.002, +0.148] | -0.122 [-0.177, -0.067] | +0.0 [-4.7, +4.7] pp | +0.0 [-3.1, +3.1] pp |
| block_32 | 32 | -0.003 [-0.028, +0.024] | -0.031 [-0.062, -0.004] | +0.028 [+0.001, +0.057] | +0.8 [-2.3, +3.9] pp | +0.0 [-2.3, +2.3] pp |
| block_36 | 36 | -0.031 [-0.052, -0.009] | -0.067 [-0.105, -0.035] | +0.036 [+0.007, +0.068] | +1.6 [-1.6, +4.7] pp | +0.0 [+0.0, +0.0] pp |
| block_40 | 40 | -0.010 [-0.027, +0.009] | -0.037 [-0.056, -0.019] | +0.027 [+0.005, +0.050] | +0.8 [+0.0, +2.3] pp | +0.8 [+0.0, +2.3] pp |
| block_44 | 44 | -0.174 [-0.258, -0.102] | -0.592 [-0.713, -0.480] | +0.418 [+0.347, +0.496] | +4.7 [+0.8, +9.4] pp | +6.2 [+2.3, +10.9] pp |
| block_48 | 48 | -0.013 [-0.029, +0.003] | -0.004 [-0.020, +0.012] | -0.009 [-0.030, +0.012] | +1.6 [+0.0, +3.9] pp | -0.8 [-2.3, +0.0] pp |
| block_52 | 52 | -0.012 [-0.024, +0.000] | -0.001 [-0.016, +0.013] | -0.011 [-0.030, +0.007] | +1.6 [+0.0, +3.9] pp | +0.0 [-2.3, +2.3] pp |
| block_56 | 56 | -0.003 [-0.016, +0.010] | -0.010 [-0.024, +0.004] | +0.007 [-0.013, +0.026] | +1.6 [+0.0, +3.9] pp | +0.0 [+0.0, +0.0] pp |
| block_60 | 60 | -0.011 [-0.023, +0.002] | -0.012 [-0.024, +0.000] | +0.001 [-0.016, +0.019] | +1.6 [+0.0, +3.9] pp | +0.0 [+0.0, +0.0] pp |
| block_64 | 64 | -0.009 [-0.020, +0.002] | -0.001 [-0.009, +0.008] | -0.008 [-0.022, +0.005] | -0.8 [-2.3, +0.0] pp | +0.0 [+0.0, +0.0] pp |
| band_04_16 | 4,8,12,16 | +0.275 [+0.138, +0.420] | +0.149 [+0.009, +0.294] | +0.126 [+0.020, +0.235] | +0.0 [-4.7, +4.7] pp | +0.8 [-3.9, +5.5] pp |
| band_20_32 | 20,24,28,32 | +0.196 [+0.042, +0.367] | +0.079 [-0.093, +0.238] | +0.117 [-0.031, +0.274] | +0.0 [-6.2, +7.0] pp | +2.3 [-2.3, +7.0] pp |
| band_36_48 | 36,40,44,48 | -0.242 [-0.361, -0.152] | -0.790 [-0.996, -0.624] | +0.548 [+0.449, +0.656] | +6.2 [+2.3, +10.9] pp | +8.6 [+3.9, +13.3] pp |
| band_52_64 | 52,56,60,64 | -0.022 [-0.037, -0.006] | -0.029 [-0.042, -0.015] | +0.007 [-0.014, +0.028] | +2.3 [+0.0, +5.5] pp | -0.8 [-2.3, +0.0] pp |
| without_band_04_16 | 20,24,28,32,36,40,44,48,52,56,60,64 | -0.645 [-0.844, -0.464] | -2.204 [-2.658, -1.819] | +1.559 [+1.251, +1.901] | +14.1 [+6.2, +21.9] pp | +25.8 [+18.0, +34.4] pp |
| without_band_20_32 | 4,8,12,16,36,40,44,48,52,56,60,64 | -0.259 [-0.441, -0.084] | -1.299 [-1.568, -1.053] | +1.040 [+0.878, +1.219] | +9.4 [+3.9, +14.8] pp | +17.2 [+10.9, +23.4] pp |
| without_band_36_48 | 4,8,12,16,20,24,28,32,52,56,60,64 | +0.401 [+0.125, +0.678] | -1.408 [-1.760, -1.074] | +1.809 [+1.532, +2.106] | +1.6 [-6.2, +9.4] pp | +22.7 [+14.8, +30.5] pp |
| without_band_52_64 | 4,8,12,16,20,24,28,32,36,40,44,48 | -0.536 [-0.783, -0.299] | -3.087 [-3.547, -2.664] | +2.550 [+2.151, +3.002] | +16.4 [+7.8, +25.0] pp | +40.6 [+32.0, +49.2] pp |
| all_selected_option | 4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64 | -0.548 [-0.794, -0.312] | -3.092 [-3.553, -2.669] | +2.544 [+2.149, +2.990] | +16.4 [+8.6, +24.2] pp | +40.6 [+31.2, +50.0] pp |

### Confirmation leave-one-band-out necessity

| Omitted band | Game contribution | Neutral contribution | Game − Neutral contribution |
|---|---:|---:|---:|
| band_04_16 | +0.097 [-0.146, +0.348] | -0.888 [-1.126, -0.651] | +0.985 [+0.773, +1.214] |
| band_20_32 | -0.289 [-0.557, -0.010] | -1.793 [-2.198, -1.461] | +1.503 [+1.184, +1.892] |
| band_36_48 | -0.949 [-1.108, -0.801] | -1.684 [-2.002, -1.395] | +0.735 [+0.521, +0.976] |
| band_52_64 | -0.012 [-0.026, +0.002] | -0.005 [-0.019, +0.009] | -0.007 [-0.028, +0.014] |

Canonical figure: `figures/qwen36_fixed_a_kv_layer_localization.png`.
