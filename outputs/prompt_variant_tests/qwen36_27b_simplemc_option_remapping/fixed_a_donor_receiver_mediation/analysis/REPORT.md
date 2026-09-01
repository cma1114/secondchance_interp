# Fixed-A donor-to-repeated-option mediation

> **Historical two-mapping result.** A later fixed-A calibration using the
> complete 24-ordering B--D pipeline did not reproduce Game donor transfer and
> found that the nonmatching control blockade removed more Neutral transfer
> than the matching blockade. Use the
> [same-pipeline calibration](../../fixed_a_full24_calibration/analysis/REPORT.md)
> for the canonical conclusion.

Positive values mean that transplanting the first selected-option line moves the final answer toward the donor history's semantic answer.

## Discovery

- Game: open transfer +0.711 [+0.405, +1.037]; matching-blocked transfer +0.409 [+0.125, +0.698]; matching mediation +0.302 [+0.199, +0.406]; matching-specific mediation +0.357 [+0.217, +0.504].
- Neutral: open transfer +2.874 [+2.523, +3.232]; matching-blocked transfer +2.289 [+1.995, +2.614]; matching mediation +0.585 [+0.466, +0.713]; matching-specific mediation +0.566 [+0.365, +0.762].
- Game minus Neutral: open transfer -2.163 [-2.426, -1.902]; matching-blocked transfer -1.880 [-2.115, -1.648]; matching mediation -0.283 [-0.371, -0.198]; matching-specific mediation -0.209 [-0.345, -0.062].

Validation: `{"all_position_counts_positive": true, "donor_open_vs_prior_selected_line_answer_agreement": 0.9409090909090909, "donor_open_vs_prior_selected_line_mean_abs_logit_error": 0.1483223936774514, "excluded_non_A_first_decision": 7, "historical_rows": 64, "prior_selected_line_comparable_rows": 55, "recipient_open_vs_natural_answer_agreement": 0.9429824561403509, "recipient_open_vs_natural_mean_abs_logit_error": 0.13578643506033378, "valid_rows": 57}`

## Confirmation

- Game: open transfer +0.548 [+0.311, +0.797]; matching-blocked transfer +0.260 [+0.034, +0.506]; matching mediation +0.288 [+0.204, +0.370]; matching-specific mediation +0.450 [+0.290, +0.613].
- Neutral: open transfer +3.092 [+2.682, +3.550]; matching-blocked transfer +2.508 [+2.191, +2.849]; matching mediation +0.584 [+0.455, +0.737]; matching-specific mediation +0.516 [+0.295, +0.752].
- Game minus Neutral: open transfer -2.544 [-2.981, -2.146]; matching-blocked transfer -2.248 [-2.594, -1.912]; matching mediation -0.296 [-0.440, -0.177]; matching-specific mediation -0.065 [-0.283, +0.172].

Validation: `{"all_position_counts_positive": true, "donor_open_vs_prior_selected_line_answer_agreement": 0.9590163934426229, "donor_open_vs_prior_selected_line_mean_abs_logit_error": 0.1689089966602013, "excluded_non_A_first_decision": 9, "historical_rows": 73, "prior_selected_line_comparable_rows": 61, "recipient_open_vs_natural_answer_agreement": 0.953125, "recipient_open_vs_natural_mean_abs_logit_error": 0.15346391312777996, "valid_rows": 64}`
