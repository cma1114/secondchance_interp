# Qwen switch-trial readout comparison

This is the matched-readout check motivated by the apparent Seed-OSS/Qwen difference. It applies both Qwen's original Jacobian lens and a conventional logit lens to the same cached final-decision residuals, same questions, same condition labels, same frozen first-presentation ranks, and same switch subsets. No transformer forward was rerun for this comparison.

The plotted endpoint is the displayed-letter-controlled R2-minus-R1 score. Positive means the candidate ranked second on the first presentation is more readable than the original winner. The control first subtracts the stable A/B/C/D geometry within each condition and layer; without it, Qwen's output rows create a misleading early R1 pattern. Outcome-conditioned switch panels are descriptive activation evidence, not a causal intervention.

[Canonical comparison figure](../../../../../figures/model_replications/qwen36_jlens_vs_standard_logit_lens_switch_r2_r1.png)

## Results

### SimpleMC

- **Game switch trials (n=174):** Jacobian-lens R2−R1 has its first three-layer sustained positive CI at L49; the standard logit lens does so at L50. At L52 the margins are +0.170 `[+0.119, +0.220]` and +0.177 `[+0.107, +0.250]`, respectively.
- **Neutral switch trials (n=132):** Jacobian-lens R2−R1 has its first three-layer sustained positive CI at L36; the standard logit lens does so at L51. At L52 the margins are +0.251 `[+0.186, +0.319]` and +0.306 `[+0.216, +0.397]`, respectively.

### TriviaMC difficulty-filtered

- **Game switch trials (n=91):** Jacobian-lens R2−R1 has its first three-layer sustained positive CI at L40; the standard logit lens does so at L50. At L52 the margins are +0.226 `[+0.139, +0.313]` and +0.426 `[+0.299, +0.548]`, respectively.
- **Neutral switch trials (n=54):** Jacobian-lens R2−R1 has its first three-layer sustained positive CI at L52; the standard logit lens does so at L52. At L52 the margins are +0.290 `[+0.156, +0.429]` and +0.488 `[+0.293, +0.682]`, respectively.

## Interpretation

The matched conventional readout does not turn Qwen into the Seed pattern. After displayed-letter geometry is removed, Qwen remains approximately unseparated until late, and R2 is already above R1 when a reliable switch-trial ordering becomes readable. The original Qwen conclusion was therefore not an artifact of using a Jacobian lens while Seed used a standard logit lens.

The descriptive cross-model difference remains: Seed often exposes an R1-leading output-readable state before R2 overtakes it, whereas Qwen's output-readable switch ordering appears with R2 already ahead. This does not establish that Qwen lacks earlier non-output-aligned answer information; Qwen's held-out prospective decoders recover question-specific and policy-adjusted information earlier. The evidence supports a difference in when the intermediate computation becomes aligned with each model's own output vocabulary, not a claim that either model implements a literally serial symbolic algorithm.
