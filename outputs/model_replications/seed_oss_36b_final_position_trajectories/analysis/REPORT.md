# Seed-OSS 36B final-decision trajectories by first-presentation rank

This report measures every post-block residual L1--L64 at the final prompt token immediately before Seed's second answer. Game and Neutral prompts are non-remapped and differ only at the single `incorrect`/`lost` token. R1--R4 are frozen from Seed's same-format first-presentation aggregated A--D logits.

No published or local Jacobian lens exists for this Seed revision. The figures therefore use the standard Seed logit lens: each post-block residual passes through Seed's exact final RMS norm and A--D unembedding rows. The separate held-out prospective-decoder analysis tests for answer information that is linearly present before this fixed output readout can expose it.

All-question panels are primary activation descriptions. Switch/no-switch panels are selected by the model's eventual answer and are descriptive, not causal evidence about why it switched. Confidence bands use a first-presentation-winner-letter-stratified question bootstrap. Background tint is the mean per-question cosine similarity between the layer's complete centered A--D pattern and the exact final pattern.

## SimpleMC

- **all:** Game n=500, Neutral n=500; [figure](../../../../figures/model_replications/seed_oss_36b_simplemc_nonremapped_rank_trajectories_all.png)
- **switch:** Game n=155, Neutral n=84; [figure](../../../../figures/model_replications/seed_oss_36b_simplemc_nonremapped_rank_trajectories_switch.png)
- **stay:** Game n=345, Neutral n=416; [figure](../../../../figures/model_replications/seed_oss_36b_simplemc_nonremapped_rank_trajectories_stay.png)
- **Companions:** [raw scores](../../../../figures/model_replications/seed_oss_36b_simplemc_nonremapped_rank_trajectories_raw.png) · [displayed-letter-controlled](../../../../figures/model_replications/seed_oss_36b_simplemc_nonremapped_rank_trajectories_letter_controlled.png)

At L64 on all questions, centered R1 is 2.271 in Game and 3.788 in Neutral; the paired Game-minus-Neutral difference is -1.517 `[-1.637, -1.399]`.

On eventual-switch trials, L64 R1/R2 is 0.598/1.451 in Game and 1.068/1.827 in Neutral. On no-switch trials, L64 R1 is 3.022 in Game and 4.337 in Neutral.

## TriviaMC difficulty-filtered

- **all:** Game n=500, Neutral n=500; [figure](../../../../figures/model_replications/seed_oss_36b_triviamc_nonremapped_rank_trajectories_all.png)
- **switch:** Game n=80, Neutral n=42; [figure](../../../../figures/model_replications/seed_oss_36b_triviamc_nonremapped_rank_trajectories_switch.png)
- **stay:** Game n=420, Neutral n=458; [figure](../../../../figures/model_replications/seed_oss_36b_triviamc_nonremapped_rank_trajectories_stay.png)
- **Companions:** [raw scores](../../../../figures/model_replications/seed_oss_36b_triviamc_nonremapped_rank_trajectories_raw.png) · [displayed-letter-controlled](../../../../figures/model_replications/seed_oss_36b_triviamc_nonremapped_rank_trajectories_letter_controlled.png)

At L64 on all questions, centered R1 is 6.989 in Game and 9.247 in Neutral; the paired Game-minus-Neutral difference is -2.258 `[-2.409, -2.110]`.

On eventual-switch trials, L64 R1/R2 is 1.183/2.486 in Game and 1.676/3.140 in Neutral. On no-switch trials, L64 R1 is 8.095 in Game and 9.942 in Neutral.

## Scope

The standard logit lens measures when a candidate ordering is directly readable by Seed's own output norm and A--D rows. A flat early lens trajectory does not establish absence from the residual stream; that question belongs to the held-out prospective decoders. Displayed-letter-controlled figures subtract each condition/layer's across-question mean for displayed A, B, C, and D before aligning candidates by 1P rank. Raw-score companions retain the common A--D offset, which mixes generic answer-token readiness with layer-dependent readout scale.

## Matched Qwen readout

The same conventional readout has now been applied to Qwen's cached
final-position residuals. After the identical displayed-letter control, Qwen
still shows no reliable switch-trial separation until about L50, with R2
already ahead when the ordering appears. Seed's earlier visible R1-leading
phase therefore cannot be attributed simply to Seed having been plotted with a
standard logit lens while Qwen was plotted with a Jacobian lens. This remains a
descriptive cross-model timing difference. [Qwen matched-readout report](../../../prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/standard_logit_lens/comparison/REPORT.md).
