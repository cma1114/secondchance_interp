# A general MCQ uncertainty direction at the second decision

## Bottom line

A frozen direction learned only from first-presentation multiple-choice entropy reappears in the second-decision residual stream and is causally coupled to the second answer ranking in both Qwen3.6-27B and Seed-OSS 36B, on both SimpleMC and TriviaMC. The cleanest causal evidence is bidirectional steering: moving in the high-uncertainty direction generally lowers the old winner relative to the other candidates at middle-to-late layers, whereas an equal-dose orthogonal random direction does not.

That effect is **not specific to Game**. It is similar in Game and Neutral for Seed and is substantially stronger in Neutral for Qwen. At the tested dose it reliably moves logits but rarely changes the discrete answer. Thus this is evidence for a general uncertainty-like ranking control available at 2P, but not evidence that Game preferential switching is driven by a special readout of that control.

This experiment does not train on, predict, or define the direction using 2P output uncertainty. It measures the activation of the frozen 1P direction at 2P and intervenes on that coordinate directly. It also does not test remembered 1P uncertainty; earlier first-decision source interventions make simple copying from that position unlikely, so *re-instantiation at 2P* is the best-supported inference, not a directly traced construction mechanism.

## Scope and validity

All estimates use the frozen confirmation questions and 10,000 paired question-bootstrap draws. Every causal cell covers L1-L64 in Game and Neutral separately and is compared with a same-norm orthogonal random direction.

- **qwen36_27b / simplemc**: n=249; identity raw/centered error 0.74921/0.374775; identity argmax agreement 96.2%; held-out L64 entropy correlation raw/controlling old-winner letter 0.816/0.817; cross-dataset L64 correlation 0.855; maximum direct answer-contrast fraction 0.67% at L63; ablation residual fraction 0.0438%; random-ablation residual fraction 0.2176%; random steering means -3.0000/3.0000; all finite=True.
- **qwen36_27b / triviamc**: n=250; identity raw/centered error 0.993031/0.845671; identity argmax agreement 99.0%; held-out L64 entropy correlation raw/controlling old-winner letter 0.836/0.838; cross-dataset L64 correlation 0.773; maximum direct answer-contrast fraction 0.19% at L32; ablation residual fraction 0.0528%; random-ablation residual fraction 0.1809%; random steering means -3.0000/3.0000; all finite=True.
- **seed_oss_36b / simplemc**: n=249; identity raw/centered error 0.5/0.4375; identity argmax agreement 97.4%; held-out L64 entropy correlation raw/controlling old-winner letter 0.532/0.537; cross-dataset L64 correlation 0.585; maximum direct answer-contrast fraction 1.73% at L52; ablation residual fraction 0.0700%; random-ablation residual fraction 0.1410%; random steering means -2.9997/2.9997; all finite=True.
- **seed_oss_36b / triviamc**: n=250; identity raw/centered error 0.75/0.625; identity argmax agreement 100.0%; held-out L64 entropy correlation raw/controlling old-winner letter 0.534/0.554; cross-dataset L64 correlation 0.486; maximum direct answer-contrast fraction 5.41% at L54; ablation residual fraction 0.0821%; random-ablation residual fraction 0.1090%; random steering means -2.9997/2.9997; all finite=True.

The old-winner-letter control is important: removing the mean projection within each displayed old-winner letter leaves the entropy correlations essentially unchanged. The frozen direction is therefore not simply a code for whether A, B, C, or D won. Its direct Euclidean overlap with the centered four-answer output subspace is also small in every cell. Seed's middle/late logit lens is semantically recognizable (`unknown`, `none`, and Chinese equivalents); Qwen's top vocabulary tokens are less clean. Cross-dataset transfer is strong for both models.

## Natural 2P activation

The left column of the figure is the projection itself, standardized by the 1P discovery distribution. It is strongly structured and frequently far from zero in all four cells. Its sign and Game-versus-Neutral ordering vary by layer, model, and dataset; there is no universal scalar level that separates Game from Neutral. This answers the activation question without redefining the target around the model's eventual 2P logits.

## Causal steering

The table reports the strongest middle-to-late (L33-L64) signed steering effect on the centered old-R1 logit. A value of -0.02 means that each +1 unit step along the frozen high-uncertainty direction lowers old R1 by 0.02 logits relative to the four-answer mean, beyond the matched random-direction effect.

| Model / dataset | Game | Neutral | Policy-specific reading |
|---|---:|---:|---|
| qwen36_27b / simplemc | L61 -0.0088 [-0.0106, -0.0071] | L48 -0.0175 [-0.0195, -0.0155] | largest late G-N: L48 +0.0128 [+0.0104, +0.0151] |
| qwen36_27b / triviamc | L50 -0.0308 [-0.0344, -0.0274] | L49 -0.0576 [-0.0633, -0.0522] | largest late G-N: L49 +0.0276 [+0.0239, +0.0313] |
| seed_oss_36b / simplemc | L43 -0.0163 [-0.0195, -0.0130] | L43 -0.0161 [-0.0191, -0.0132] | largest late G-N: L58 +0.0054 [+0.0021, +0.0088] |
| seed_oss_36b / triviamc | L42 -0.0251 [-0.0288, -0.0215] | L40 -0.0262 [-0.0290, -0.0232] | largest late G-N: L37 +0.0071 [+0.0032, +0.0109] |

At those same layers the full centered-rank vectors are not generic gain changes: old R1 moves down while lower-ranked candidates move up. The initial-margin-weighted analysis preserves and usually strengthens the same late-layer sign. Seed SimpleMC also has an early, opposite-signed L1-L9 effect; its later L37-L55 effect matches the other three cells.

The policy-specific column is decisive. Qwen's positive Game-minus-Neutral interaction means that high-uncertainty steering suppresses old R1 *less* in Game than in Neutral. Seed's Game and Neutral responses are nearly the same: the familywise Game-minus-Neutral R1 band excludes zero at no layer in SimpleMC and only at isolated L37 in TriviaMC. This is the opposite of the simple metacognitive hypothesis in which Game uniquely reads the uncertainty coordinate to decide whether to abandon its old answer.

Discrete old-W1 choice slopes at the strongest logit-effect layers are small and their ordinary 95% intervals generally include zero. The causal claim is therefore about candidate scoring, not a demonstrated change in switch rate at the frozen ±3 dose.

## Ablation and its limit

Projection ablation changes the answer computation, but it is not an edit-norm-matched comparison with the orthogonal random ablation. Both directions are unit vectors, yet ablation subtracts each state's complete natural projection. At the layers with the largest old-R1 effects, the mean absolute uncertainty-coordinate edits are about 57–562 residual units, versus about 3–50 for the random coordinate. Its very large, non-monotonic late effects—especially in Seed—therefore establish that this aggressive coordinate removal changes the output, but they do not isolate a physiological effect size or direction. The ±3 bidirectional steering comparison is truly dose-matched and is the primary signed causal result.

## Conclusion

The narrow hypothesis succeeds: a dataset-general 1P MCQ uncertainty axis is naturally present again at the 2P decision position, and changing that axis causally changes the final candidate ranking in two architectures and two datasets. The stronger metacognitive interpretation does not: the causal effect is shared with Neutral and, in Qwen, is stronger there. The measured axis is therefore best described as a general uncertainty-like control on answer ranking that both conditions can use, not the condition-specific trigger for preferential Game switching.

The complete machine-readable layerwise estimates, pointwise intervals, familywise simultaneous bands, all four rank effects, W1-minus-W2 margins, choice effects, initial-margin-weighted estimates, and paired Game-minus-Neutral contrasts are in `summary.json`.

Interpretation must distinguish natural activation from causal use. A projection alone is activation evidence; an uncertainty intervention that differs from the orthogonal random control is causal evidence. A null remains bounded because a single direction may be redundant with other uncertainty directions.
