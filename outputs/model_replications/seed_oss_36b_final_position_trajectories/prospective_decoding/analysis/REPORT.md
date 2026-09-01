# Seed-OSS 36B prospective final-answer decoding

At every final-decision post-block residual L1--L64, ridge decoders predict the exact eventual centered A--D score vector. Shared Game+Neutral, Game-only, and Neutral-only bases are fit and tuned using discovery questions only and evaluated on the frozen confirmation questions. Cross-condition evaluation asks whether Game and Neutral encode the prospective answer in a shared linear basis. A W1-matched question shuffle preserves the easiest displayed old-winner structure while destroying question-specific final geometry.

The fixed comparison is Seed's standard logit lens, because no compatible Seed Jacobian lens exists. Earlier held-out decoding than the logit lens means that the eventual answer pattern is linearly accessible before it is directly output-readable; it does not by itself establish causal use.

## SimpleMC

Discovery n=251; confirmation n=249. [figure](../../../../figures/prospective_decoding/seed_oss_36b_simplemc_prospective_answer_decoding.png)

Selected held-out mean cosine similarities:

- L24: Shared 0.023 / Matched task 0.017 / Cross-task -0.006 / Standard logit lens 0.044
- L32: Shared 0.029 / Matched task 0.027 / Cross-task -0.000 / Standard logit lens 0.040
- L40: Shared 0.682 / Matched task 0.676 / Cross-task 0.591 / Standard logit lens 0.232
- L44: Shared 0.945 / Matched task 0.946 / Cross-task 0.934 / Standard logit lens 0.876
- L48: Shared 0.954 / Matched task 0.953 / Cross-task 0.945 / Standard logit lens 0.883
- L52: Shared 0.959 / Matched task 0.957 / Cross-task 0.950 / Standard logit lens 0.883
- L56: Shared 0.956 / Matched task 0.957 / Cross-task 0.951 / Standard logit lens 0.865
- L60: Shared 0.961 / Matched task 0.961 / Cross-task 0.956 / Standard logit lens 0.851
- L64: Shared 0.978 / Matched task 0.973 / Cross-task 0.968 / Standard logit lens 1.000

On held-out eventual-switch trials, the first three-layer-sustained positive shared-decoder R2-minus-R1 interval begins at 53 in Game and 41 in Neutral.

## TriviaMC difficulty-filtered

Discovery n=250; confirmation n=250. [figure](../../../../figures/prospective_decoding/seed_oss_36b_triviamc_prospective_answer_decoding.png)

Selected held-out mean cosine similarities:

- L24: Shared -0.047 / Matched task -0.035 / Cross-task -0.036 / Standard logit lens 0.012
- L32: Shared -0.014 / Matched task -0.008 / Cross-task -0.012 / Standard logit lens 0.002
- L40: Shared 0.804 / Matched task 0.798 / Cross-task 0.742 / Standard logit lens 0.363
- L44: Shared 0.960 / Matched task 0.957 / Cross-task 0.950 / Standard logit lens 0.931
- L48: Shared 0.965 / Matched task 0.963 / Cross-task 0.957 / Standard logit lens 0.941
- L52: Shared 0.966 / Matched task 0.965 / Cross-task 0.958 / Standard logit lens 0.939
- L56: Shared 0.966 / Matched task 0.965 / Cross-task 0.959 / Standard logit lens 0.934
- L60: Shared 0.971 / Matched task 0.970 / Cross-task 0.967 / Standard logit lens 0.942
- L64: Shared 0.989 / Matched task 0.987 / Cross-task 0.983 / Standard logit lens 1.000

On held-out eventual-switch trials, the first three-layer-sustained positive shared-decoder R2-minus-R1 interval begins at 13 in Game and None in Neutral.

## All-question policy-adjusted timing

For every paired confirmation question, the shared decoder is applied to both conditions and the decoded Neutral vector is subtracted from Game. This measures when the question-specific final Game-versus-Neutral answer adjustment is linearly available without selecting on eventual switching.

### SimpleMC

The held-out policy-pattern cosine becomes persistently positive at 36. At L40 the learned cosine is 0.575 `[0.516,0.633]`, versus standard-logit-lens 0.223. The exact final Game-minus-Neutral R1--R4 effects are -1.530/+0.121/+0.450/+0.958. [figure](../../../../figures/prospective_decoding/seed_oss_36b_simplemc_policy_adjusted_prospective_decoding.png)

### TriviaMC difficulty-filtered

The held-out policy-pattern cosine becomes persistently positive at 39. At L40 the learned cosine is 0.446 `[0.378,0.514]`, versus standard-logit-lens 0.154. The exact final Game-minus-Neutral R1--R4 effects are -2.142/+0.766/+0.720/+0.656. [figure](../../../../figures/prospective_decoding/seed_oss_36b_triviamc_policy_adjusted_prospective_decoding.png)

## Scope

These are held-out linear decoding results at the exact final decision position. They establish timing and cross-condition representational accessibility, not causal mediation. Switch-conditioned margins are descriptive postselection. The completed all-layer matching-history blockade remains the causal evidence that semantic 1P recollection affects Seed's preferential Game avoidance of the old winner.
