from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("Game", "Neutral", "Game_minus_Neutral")


def _align(
    values: np.ndarray,
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            new_letter = mappings[qid]["original_to_new"][original]
            out[..., qi, original_index] = values[
                ..., qi, LETTERS.index(new_letter)
            ]
    return out


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _question_center_features(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def _fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    discovery: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.linalg.lstsq(
        features[discovery].reshape(-1, features.shape[-1]),
        target[discovery].reshape(-1),
        rcond=None,
    )[0]
    prediction = np.einsum("qcf,f->qc", features, coefficients)
    return prediction, coefficients


def _r2(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    residual = target[mask] - prediction[mask]
    denominator = np.square(target[mask]).sum()
    return float(1.0 - np.square(residual).sum() / denominator)


def _bootstrap_r2_difference(
    target: np.ndarray,
    base_prediction: np.ndarray,
    extended_prediction: np.ndarray,
    confirmation: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> list[float]:
    indices = np.flatnonzero(confirmation)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(indices, len(indices), replace=True)
        denominator = np.square(target[sampled]).sum()
        base_sse = np.square(target[sampled] - base_prediction[sampled]).sum()
        extended_sse = np.square(
            target[sampled] - extended_prediction[sampled]
        ).sum()
        values[draw] = (base_sse - extended_sse) / denominator
    return np.quantile(values, (0.025, 0.975)).tolist()


def _bootstrap_coefficients(
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(mask)

    def fit(sampled: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(
            features[sampled].reshape(-1, features.shape[-1]),
            target[sampled].reshape(-1),
            rcond=None,
        )[0]

    point = fit(indices)
    samples = np.empty((draws, features.shape[-1]), dtype=float)
    for draw in range(draws):
        samples[draw] = fit(rng.choice(indices, len(indices), replace=True))
    return point, np.quantile(samples, (0.025, 0.975), axis=0)


def _interval(point: float, interval: list[float]) -> dict[str, Any]:
    return {"mean": float(point), "ci": [float(x) for x in interval]}


def _format_interval(value: dict[str, Any]) -> str:
    return (
        f"{value['mean']:+.3f} "
        f"[{value['ci'][0]:+.3f}, {value['ci'][1]:+.3f}]"
    )


def analyze(args: argparse.Namespace) -> None:
    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not arrays["completed"].astype(bool).all():
        raise RuntimeError("All-candidate checkpoint is incomplete")

    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500:
        raise RuntimeError(f"Expected 500 questions, found {len(qids)}")

    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    discovery_ids = set(
        json.loads(args.discovery_plan.read_text())["question_ids"]
    )
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    confirmation = ~discovery
    if discovery.sum() != 251 or confirmation.sum() != 249:
        raise RuntimeError("Frozen discovery/confirmation split is not 251/249")

    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.asarray(
        [[LETTERS.index(value) for value in row] for row in rank_contents],
        dtype=int,
    ).T
    w1_indices = rank_indices[0]

    old_score = _center(arrays["baseline_logits"].astype(float))
    remapped_results = json.loads(args.remapped_baseline.read_text())["results"]
    current_raw = np.asarray(
        [remapped_results[qid]["aggregated_ad_logits"] for qid in qids],
        dtype=float,
    )
    current_score = _center(_align(current_raw, qids, mappings))
    final_logits = _center(
        _align(arrays["natural_logits"].astype(float), qids, mappings)
    )

    matched = _align(arrays["matched_logits"].astype(float), qids, mappings)
    control = _align(arrays["control_logits"].astype(float), qids, mappings)
    rows = np.arange(len(qids))
    matching_effect = np.empty((2, len(qids), 4), dtype=float)
    for condition in range(2):
        for rank in range(4):
            indices = rank_indices[rank]
            matched_advantage = matched[condition, rank, rows, indices] - (
                matched[condition, rank].sum(1)
                - matched[condition, rank, rows, indices]
            ) / 3.0
            control_advantage = control[condition, rank, rows, indices] - (
                control[condition, rank].sum(1)
                - control[condition, rank, rows, indices]
            ) / 3.0
            matching_effect[condition, rows, indices] = (
                matched_advantage - control_advantage
            )
    matching_effect = _center(matching_effect)

    old_mean = float(old_score[discovery].mean())
    old_sd = float(old_score[discovery].std())
    current_mean = float(current_score[discovery].mean())
    current_sd = float(current_score[discovery].std())
    old_standardized = (old_score - old_mean) / old_sd
    current_standardized = (current_score - current_mean) / current_sd

    first_position = np.zeros((len(qids), 4, 3), dtype=float)
    second_position = np.zeros((len(qids), 4, 3), dtype=float)
    winner = np.zeros((len(qids), 4, 1), dtype=float)
    winner[rows, w1_indices, 0] = 1.0
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            if original_index < 3:
                first_position[qi, original_index, original_index] = 1.0
            new_index = LETTERS.index(
                mappings[qid]["original_to_new"][original]
            )
            if new_index < 3:
                second_position[qi, original_index, new_index] = 1.0

    position_controls = _question_center_features(
        np.concatenate([first_position, second_position], axis=2)
    )

    def feature(*columns: np.ndarray) -> np.ndarray:
        return _question_center_features(np.concatenate(columns, axis=2))

    old = old_standardized[..., None]
    current = current_standardized[..., None]
    interaction = (old_standardized * current_standardized)[..., None]
    old_polynomial = np.stack(
        [old_standardized, old_standardized**2, old_standardized**3], axis=2
    )
    current_polynomial = np.stack(
        [
            current_standardized,
            current_standardized**2,
            current_standardized**3,
        ],
        axis=2,
    )
    cross_polynomial = np.stack(
        [
            old_standardized * current_standardized,
            old_standardized * current_standardized**2,
            old_standardized**2 * current_standardized,
        ],
        axis=2,
    )

    models = {
        "linear_current": feature(position_controls, current),
        "linear_both": feature(position_controls, current, old),
        "linear_interaction": feature(
            position_controls, current, old, interaction
        ),
        "linear_interaction_w1": feature(
            position_controls, current, old, interaction, winner
        ),
        "flex_current": feature(position_controls, current_polynomial),
        "flex_both": feature(
            position_controls, current_polynomial, old_polynomial
        ),
        "flex_interaction": feature(
            position_controls,
            current_polynomial,
            old_polynomial,
            cross_polynomial,
        ),
        "flex_interaction_w1": feature(
            position_controls,
            current_polynomial,
            old_polynomial,
            cross_polynomial,
            winner,
        ),
    }
    comparisons = (
        ("linear_current", "linear_both", "add_old_score_linear"),
        ("linear_both", "linear_interaction", "add_score_interaction_linear"),
        (
            "linear_interaction",
            "linear_interaction_w1",
            "add_W1_linear",
        ),
        ("flex_current", "flex_both", "add_old_score_flexible"),
        ("flex_both", "flex_interaction", "add_score_interaction_flexible"),
        (
            "flex_interaction",
            "flex_interaction_w1",
            "add_W1_flexible",
        ),
    )

    targets = {
        "final_centered_logits": (
            final_logits[0],
            final_logits[1],
            final_logits[0] - final_logits[1],
        ),
        "matching_specific_lesion_effect": (
            matching_effect[0],
            matching_effect[1],
            matching_effect[0] - matching_effect[1],
        ),
    }
    summary: dict[str, Any] = {
        "definitions": {
            "old_score": "first-presentation candidate-centered Baseline A-D logit",
            "current_score": "fresh remapped-presentation candidate-centered Baseline A-D logit with no Second Chance history",
            "matching_specific_lesion_effect": "candidate advantage after matching-edge lesion minus cyclic nonmatching-edge control; positive means the intact semantic match opposed the candidate",
            "models": "all models contain question-centered first- and second-presentation displayed-position controls",
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int(confirmation.sum()),
            "old_score_discovery_sd": old_sd,
            "current_score_discovery_sd": current_sd,
        },
        "endpoints": {},
    }
    rng = np.random.default_rng(args.seed)
    for endpoint, condition_targets in targets.items():
        endpoint_record: dict[str, Any] = {}
        for condition, target in zip(CONDITIONS, condition_targets):
            predictions: dict[str, np.ndarray] = {}
            condition_record: dict[str, Any] = {"heldout_r2": {}, "increments": {}}
            for model_name, model_features in models.items():
                prediction, _ = _fit_predict(model_features, target, discovery)
                predictions[model_name] = prediction
                condition_record["heldout_r2"][model_name] = _r2(
                    target, prediction, confirmation
                )
            for base_name, extended_name, label in comparisons:
                point = (
                    condition_record["heldout_r2"][extended_name]
                    - condition_record["heldout_r2"][base_name]
                )
                interval = _bootstrap_r2_difference(
                    target,
                    predictions[base_name],
                    predictions[extended_name],
                    confirmation,
                    rng,
                    args.draws,
                )
                condition_record["increments"][label] = _interval(
                    point, interval
                )

            coefficient_features = models["linear_interaction_w1"]
            coefficient_names = (
                "first_position_A",
                "first_position_B",
                "first_position_C",
                "second_position_A",
                "second_position_B",
                "second_position_C",
                "current_score",
                "old_score",
                "old_by_current",
                "W1",
            )
            condition_record["coefficients"] = {}
            for split_name, mask in (
                ("discovery", discovery),
                ("confirmation", confirmation),
            ):
                point, intervals = _bootstrap_coefficients(
                    coefficient_features,
                    target,
                    mask,
                    rng,
                    args.draws,
                )
                condition_record["coefficients"][split_name] = {
                    name: _interval(point[index], intervals[:, index].tolist())
                    for index, name in enumerate(coefficient_names)
                }
            endpoint_record[condition] = condition_record
        summary["endpoints"][endpoint] = endpoint_record

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    final = summary["endpoints"]["final_centered_logits"]
    causal = summary["endpoints"]["matching_specific_lesion_effect"]
    lines = [
        "# Existing-data old-score/current-score integration analysis",
        "",
        "## Bottom line",
        "",
        "Fresh second-presentation evidence is the strongest single predictor of the final candidate ranking, but first-presentation evidence adds held-out predictive information in both Game and Neutral. The increment is larger in Neutral. A simple old-by-current interaction has a small effect in the linear specification but does not robustly improve the flexible model. A W1 term remains most clearly in the Game-minus-Neutral contrast.",
        "",
        "The causal matching-edge endpoint is more diagnostic. After controlling current score and both displayed positions, stronger old evidence predicts a more suppressive matching effect in Game relative to Neutral on both frozen splits. Current second-presentation evidence independently predicts Game's matching-edge effect. The separate Neutral old-score coefficient points toward support but is uncertain. Thus the semantic match is not merely carrying a condition-independent old score: its causal use depends on both historical and current candidate evidence.",
        "",
        "This is a predictive decomposition of saved causal and natural outputs. It does not identify which vectors encode either score and does not replace a same-semantic/different-score transplant.",
        "",
        "## Held-out predictive increments",
        "",
        "All values are increases in confirmation-set R² from adding the named feature family to the nested discovery-fitted model.",
        "",
        "| Endpoint | Condition | Add old score (linear) | Add old score (flexible) | Add old×current (linear) | Add old×current (flexible) | Add W1 after flexible scores |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for endpoint_name, record in (
        ("Final centered logits", final),
        ("Matching-edge lesion", causal),
    ):
        for condition in CONDITIONS:
            increments = record[condition]["increments"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        endpoint_name,
                        condition.replace("Game_minus_Neutral", "Game − Neutral"),
                        _format_interval(increments["add_old_score_linear"]),
                        _format_interval(increments["add_old_score_flexible"]),
                        _format_interval(increments["add_score_interaction_linear"]),
                        _format_interval(increments["add_score_interaction_flexible"]),
                        _format_interval(increments["add_W1_flexible"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Linear causal coefficients",
            "",
            "Scores are standardized using discovery data. Positive lesion coefficients mean that the intact match becomes more opposing as the predictor increases.",
            "",
            "| Split | Condition | Old score | Current score | Old×current | W1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("discovery", "confirmation"):
        for condition in CONDITIONS:
            coefficients = causal[condition]["coefficients"][split]
            lines.append(
                "| "
                + " | ".join(
                    [
                        split.capitalize(),
                        condition.replace("Game_minus_Neutral", "Game − Neutral"),
                        _format_interval(coefficients["old_score"]),
                        _format_interval(coefficients["current_score"]),
                        _format_interval(coefficients["old_by_current"]),
                        _format_interval(coefficients["W1"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The results support a two-evidence account as a serious leading hypothesis: a repeated candidate is evaluated using both its first-presentation evidence and its fresh second-presentation evidence. They do not establish that the matching source value itself contains the old score; the old-score dependence could enter through the receiver query or another state correlated with first-pass evidence.",
            "",
            "The W1 increment also survives flexible score terms in the held-out condition difference, but the gain is small. It could reflect a categorical winner variable or remaining nonlinear score structure. A causal rank manipulation is required to distinguish them.",
            "",
            "## Validation",
            "",
            f"- Questions: {len(qids)}.",
            f"- Frozen discovery/confirmation split: {int(discovery.sum())}/{int(confirmation.sum())}.",
            "- Natural logits and causal lesions are taken from the exact canonical all-candidate run; no new model inference was used.",
            "- Every model includes question-centered controls for the candidate's displayed position in both presentations.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
