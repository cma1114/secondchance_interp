# Historical-assistant latent-answer intervention

## Question

At the empty historical assistant turn, the learned JLens decodes the live
Baseline answer ranks in order by the mid-fifties. Mixer 56 subsequently places
about one quarter of its final-position attention on that historical turn. The
causal question is whether this answer evidence is merely decodable or is read
and used differently by Game and Neutral.

## Confirmatory design

Use the frozen 249-question SimpleMC confirmation split. For every question,
run the self-hosted Baseline with exactly the same prefix as the historical
assistant turn and define ranks from its live canonical A-D logits. At post-block
readout 55—the residual immediately available to Mixer 56—construct the
question-independent centered A-D decoder induced by the pretrained JLens map,
Qwen's final RMSNorm weights, and the four bare A-D unembedding rows.

At the final token of the empty historical assistant scaffold, run five paired
interventions in both Game and Neutral:

1. erase the Baseline winner-versus-other-options evidence;
2. erase the Baseline runner-up evidence as a semantic control;
3. erase the entire centered A-D evidence subspace;
4. swap the winner and runner-up evidence;
5. add a deterministic random update orthogonal to the A-D subspace and matched
   to the winner-erasure L2 norm.

All transformations are minimum-L2 residual updates. The identical
question-specific update is applied to Game and Neutral. Every intervention
batch includes an unpatched row, and causal logits are recentered on a
single-example natural forward to remove Qwen's batch-size numerical drift.

## Predictions

If the historical representation is ordinary answer evidence, winner erasure
should reduce winner choice and winner probability similarly in Game and
Neutral, while swapping should push both conditions toward the runner-up.

If Game reads the representation as an answer to avoid while Neutral reads it
as an answer to preserve, winner erasure should increase switching in Neutral
but decrease switching in Game. A winner/runner swap should make Neutral favor
the inserted runner and Game avoid it. The decisive statistic is therefore the
paired difference between each intervention's Game effect and Neutral effect,
not decodability alone.

Report switch rate, runner-up choice, winner and runner probabilities, accuracy,
A-D entropy, full-vocabulary top-token validity, intervention norms, achieved
JLens score changes, and question-bootstrap confidence intervals.
