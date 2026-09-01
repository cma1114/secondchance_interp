# Qwen3.6-27B: exclusion transmission from feedback to decision

## Question

The unrestricted JLens explorer showed an exclusion-like vocabulary cluster at
the period ending the Second Chance feedback sentence (`exclude`, `restrict`,
and related tokens) beginning around readout 41. At the final decision position,
change- and alternative-related tokens became prominent shortly afterward. This
experiment tested whether those observations reflect transmission through the
feedback-ending token.

## Design

The analysis used the fixed 128-question SimpleMC sample stratified by generated
Baseline letter and Game switching status. At the period ending the complete
feedback sentence and at the final decision position, every post-block residual
was read with the pretrained Qwen3.6-27B Jacobian lens.

Two fixed vocabulary families were defined before examining per-trial
transmission:

- exclusion: inflections of `exclude`, `restrict`, `ban`, `reject`, and
  `eliminate`;
- alternative selection: inflections of `instead`, `other`, `another`,
  `alternative`, `change`, `retry`, and `different`.

The observational test related the paired Game-minus-Neutral exclusion readout
at feedback end to the paired alternative-selection readout at decision,
controlling for the Baseline answer letter.

The causal test replaced the feedback-ending period's post-block residual with
the paired same-question residual from the other condition. The complete
5,120-dimensional residual was replaced separately over readouts 1--16, 17--32,
33--40, 41--48, 49--64, and all 64 readouts. Both Neutral-into-Game and
Game-into-Neutral replacements were run. A narrower intervention also replaced
only the one-dimensional JLens exclusion coordinate over readouts 41--48.

The target measures were the decision-position alternative-family JLens readout
over readouts 44--50, the final prior-answer logit margin, A--D spread, and the
argmax A--D choice. All causal comparisons are paired with natural execution in
the same self-hosted run. Natural A--D choices agreed with the cached experiment
on 96.1% of Game and 100% of Neutral trials.

## Observational result

The average exclusion and alternative representations are condition-specific,
but their question-to-question strengths are not linked in the focal windows:

- correlation between feedback exclusion at readouts 41--48 and decision
  alternative content at readouts 44--50: **r = -0.073**, 95% bootstrap CI
  **[-0.251, 0.110]**;
- feedback exclusion predicting Game switching, within Baseline letter:
  **AUC = 0.440**, 95% CI **[0.334, 0.545]**.

Thus, semantic similarity between the condition-average readouts is not by
itself evidence that a larger exclusion signal causes a larger downstream
alternative signal on the same question.

## Causal result: transmission occurs earlier than the readable `exclude` signal

The natural Game-minus-Neutral decision alternative gap in this sample is
**2.905 JLens logit units**. Complete feedback-end residual replacements changed
that representation as follows:

| Feedback-end source window | Neutral into Game | Game into Neutral |
|---|---:|---:|
| L1--16 | -0.076 | +0.036 |
| L17--32 | **-0.300** | **+0.095** |
| L33--40 | **-0.143** | **+0.075** |
| L41--48 | +0.009 | +0.008 |
| L49--64 | +0.0004 | -0.0006 |
| All layers | **-0.376** | **+0.159** |

The confidence intervals for the L1--40 and all-layer representational effects
exclude zero. Window effects are not additive because later processing can
restore or transform a perturbation.

Neutralizing the Game feedback-ending residual over all layers therefore removes
about **12.9%** of the natural alternative-representation gap. Injecting the Game
residual into Neutral adds about **5.5%** of that gap. The strongest disjoint
window is L17--32, with additional transmission over L33--40. By contrast, the
L41--48 window where `exclude` becomes plainly JLens-readable has essentially no
causal effect on the later alternative readout.

The one-dimensional exclusion-coordinate replacement at L41--48 likewise
changed the later Game alternative score by only **-0.0026**, under 0.1% of the
natural gap. Its behavioral effects were tiny and not in the predicted pattern.

The most natural interpretation is that the feedback-ending token carries a
condition-sensitive representation through early and middle layers, especially
L17--40, which contributes to the later alternative-selection representation.
The transparent `exclude` vocabulary readout at L41+ is a late description of
that computation, not the causal message being transmitted at those layers.

## It does not explain behavioral switching

Despite the reproducible representational mediation, all-layer feedback-end
replacement did not materially mediate final switching:

- Neutral into Game: switch-rate change **0.000**, 95% CI
  **[-0.035, +0.035]**;
- Game into Neutral: switch-rate change **-0.016**, 95% CI
  **[-0.045, +0.012]**.

Neutral-into-Game changed 9/128 A--D choices, but these comprised three new
switches, three prevented switches, and three changes between alternatives, for
zero net switching effect. Game-into-Neutral changed 5/128 choices.

The all-layer Neutral-into-Game patch increased the prior-answer margin by
**+0.108 logits**, 95% CI **[+0.075, +0.145]**, compared with a natural
Game-minus-Neutral margin gap of -1.348 logits. Thus the feedback-ending pathway
contributes modestly to later representation and prior-answer disadvantage, but
it is not the primary mechanism producing the model's answer changes.

This null behavioral result does not prove that feedback information is
irrelevant. The system prefix, `incorrect`, `different answer`, and other prompt
positions provide redundant routes. It establishes that transmission through
the feedback-ending period is neither necessary nor sufficient for Second
Chance switching under the present intervention.

## Artifacts

- Canonical observational figure:
  `outputs/mechanistic/qwen36_27b_jlens_transmission/analysis/preserved_figures/jlens_exclusion_transmission.png`
- Canonical causal figure:
  `outputs/mechanistic/qwen36_27b_exclusion_bridge_intervention/analysis/preserved_figures/exclusion_bridge_causal_test.png`
- Observational summary:
  `outputs/mechanistic/qwen36_27b_jlens_transmission/analysis/transmission_summary.json`
- Causal summary:
  `outputs/mechanistic/qwen36_27b_exclusion_bridge_intervention/analysis/causal_summary.json`
- Causal effects table:
  `outputs/mechanistic/qwen36_27b_exclusion_bridge_intervention/analysis/causal_effects.csv`
- Collection and intervention code:
  `mechanistic/jlens_transmission_collect.py`,
  `mechanistic/analyze_jlens_transmission.py`,
  `mechanistic/run_jlens_exclusion_bridge_intervention.py`, and
  `mechanistic/analyze_jlens_exclusion_bridge_intervention.py`.

Vast instance 46566562 was stopped, not destroyed, after retrieval.
