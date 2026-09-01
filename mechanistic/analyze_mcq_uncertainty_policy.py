from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


LAYERS = 64
CONDITIONS = ("game", "neutral")
SCENARIOS = (
    "identity",
    "uncertainty_ablation",
    "uncertainty_steer_negative",
    "uncertainty_steer_positive",
    "random_ablation",
    "random_steer_negative",
    "random_steer_positive",
)
_BOOTSTRAP_COUNT_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _bootstrap_indices(n: int, draws: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def _bootstrap_means(
    values: np.ndarray, indices: np.ndarray, weights: np.ndarray | None = None,
    chunk_size: int = 256,
) -> np.ndarray:
    """Question-bootstrap means via a cached draw-by-question count matrix."""
    values = np.asarray(values, dtype=np.float64)
    del chunk_size  # Retained for backward-compatible tests/callers.
    cache_key = id(indices)
    cached = _BOOTSTRAP_COUNT_CACHE.get(cache_key)
    if cached is None or cached[0] is not indices:
        draws, n = indices.shape
        counts = np.zeros((draws, n), dtype=np.float64)
        np.add.at(
            counts,
            (np.repeat(np.arange(draws), n), indices.reshape(-1)),
            1.0,
        )
        _BOOTSTRAP_COUNT_CACHE[cache_key] = (indices, counts)
    else:
        counts = cached[1]
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        numerator = np.tensordot(values * weights, counts, axes=([-1], [-1]))
        denominator = counts @ weights
        return numerator / denominator
    return np.tensordot(values, counts, axes=([-1], [-1])) / indices.shape[1]


def _summary(
    values: np.ndarray, indices: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        estimate = values.mean(axis=-1)
        bootstrap = _bootstrap_means(values, indices)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        estimate = (values * weights).sum(axis=-1) / weights.sum()
        bootstrap = _bootstrap_means(values, indices, weights)
    low, high = np.quantile(bootstrap, (0.025, 0.975), axis=-1)
    return {
        "mean": np.asarray(estimate).tolist(),
        "ci95_low": np.asarray(low).tolist(),
        "ci95_high": np.asarray(high).tolist(),
    }


def _simultaneous_curve(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    """Question-bootstrap familywise band for a layers-by-questions curve."""
    values = np.asarray(values, dtype=np.float64)
    estimate = values.mean(axis=1)
    bootstrap = _bootstrap_means(values, indices).T
    radius = float(np.quantile(np.max(np.abs(bootstrap - estimate[None]), axis=1), 0.95))
    return {
        "mean": estimate.tolist(),
        "simultaneous95_low": (estimate - radius).tolist(),
        "simultaneous95_high": (estimate + radius).tolist(),
        "radius": radius,
    }


def _center(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=-1, keepdims=True)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _remove_group_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    groups = np.asarray(groups)
    for group in np.unique(groups):
        selected = groups == group
        result[selected] -= result[selected].mean()
    return result


def _ranked(values: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    # values: layers, questions, candidates
    return np.take_along_axis(values, ranks[None], axis=-1)


def _choice_rank(logits: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    choices = np.argmax(logits, axis=-1)
    return (ranks[None] == choices[..., None]).argmax(axis=-1)


def _effects(logits: np.ndarray, ranks: np.ndarray) -> dict[str, np.ndarray]:
    identity_index = SCENARIOS.index("identity")
    identity = logits[:, identity_index]
    identity_centered_rank = _ranked(_center(identity), ranks)
    identity_margin = _ranked(identity, ranks)[..., 0] - _ranked(identity, ranks)[..., 1]
    identity_w1 = (np.argmax(identity, axis=-1) == ranks[None, :, 0]).astype(np.float64)
    identity_choice_rank = _choice_rank(identity, ranks)
    result: dict[str, np.ndarray] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        current = logits[:, scenario_index]
        current_ranked = _ranked(_center(current), ranks)
        current_raw_rank = _ranked(current, ranks)
        current_margin = current_raw_rank[..., 0] - current_raw_rank[..., 1]
        current_w1 = (np.argmax(current, axis=-1) == ranks[None, :, 0]).astype(np.float64)
        current_choice_rank = _choice_rank(current, ranks)
        result[f"{scenario}:rank_centered"] = current_ranked - identity_centered_rank
        result[f"{scenario}:w1_minus_w2"] = current_margin - identity_margin
        result[f"{scenario}:w1_choice"] = current_w1 - identity_w1
        for rank in range(4):
            result[f"{scenario}:choice_R{rank + 1}"] = (
                (current_choice_rank == rank).astype(np.float64)
                - (identity_choice_rank == rank).astype(np.float64)
            )
    return result


def _curve_from_summary(summary: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = summary[key]
    return (
        np.asarray(value["mean"], dtype=np.float64),
        np.asarray(value["ci95_low"], dtype=np.float64),
        np.asarray(value["ci95_high"], dtype=np.float64),
    )


def _analyze_cell(
    model_slug: str, dataset_spec: dict[str, Any], direction_root: Path,
    draws: int, seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dataset_slug = dataset_spec["slug"]
    projection_path = direction_root / f"{dataset_slug}_projections.npz"
    intervention_root = Path(
        dataset_spec.get(
            "intervention_output",
            Path(dataset_spec["output"]).parent / "intervention",
        )
    )
    intervention_path = intervention_root / "results.npz"
    metadata_path = intervention_path.parent / "run_metadata.json"
    with np.load(projection_path, allow_pickle=False) as loaded:
        projection = {key: loaded[key] for key in loaded.files}
    with np.load(intervention_path, allow_pickle=False) as loaded:
        run = {key: loaded[key] for key in loaded.files}
    with np.load(Path(dataset_spec["output"]) / "results.npz", allow_pickle=False) as loaded:
        states = {key: loaded[key] for key in loaded.files}
    metadata = json.loads(metadata_path.read_text())
    direction_fit = json.loads((direction_root / "summary.json").read_text())[
        "datasets"
    ][dataset_slug]
    cross_key = (
        "cross_to_triviamc" if dataset_slug == "simplemc"
        else "cross_to_simplemc"
    )
    direction_lens = json.loads(
        (intervention_root / "direction_lens.json").read_text()
    )
    answer_fractions = {
        int(layer): float(values["centered_answer_subspace_fraction"])
        for layer, values in direction_lens.items()
        if layer != "_metadata"
    }
    maximum_answer_layer = max(answer_fractions, key=answer_fractions.get)
    selected_lens_layers = {}
    for layer in (32, 40, 48, 56, 64):
        current = direction_lens[str(layer)]
        selected_lens_layers[str(layer)] = {
            "positive_tokens": current["positive_tokens"][:5],
            "negative_tokens": current["negative_tokens"][:5],
            "centered_answer_subspace_fraction": current[
                "centered_answer_subspace_fraction"
            ],
            "shared_answer_mean_cosine": current["shared_answer_mean_cosine"],
        }
    qids = run["question_ids"].astype(str).tolist()
    if run["conditions"].astype(str).tolist() != list(CONDITIONS):
        raise RuntimeError("Intervention condition ordering changed")
    if run["scenarios"].astype(str).tolist() != list(SCENARIOS):
        raise RuntimeError("Intervention scenario ordering changed")
    if not run["completed"].all() or not np.isfinite(run["logits"]).all():
        raise RuntimeError(f"{model_slug}/{dataset_slug}: intervention incomplete or non-finite")
    projection_index = {
        qid: index for index, qid in enumerate(projection["question_ids"].astype(str))
    }
    try:
        selected = np.asarray([projection_index[qid] for qid in qids], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError("Projection and intervention questions disagree") from error
    confirmation_mask = projection["confirmation_mask"][selected]
    if not confirmation_mask.all():
        raise RuntimeError("Intervention contains a non-confirmation question")
    ranks = np.asarray(run["rank_order"], dtype=np.int64)
    projection_qids = projection["question_ids"].astype(str).tolist()
    if states["question_ids"].astype(str).tolist() != projection_qids:
        raise RuntimeError("State and projection question ordering changed")
    all_old_winner = np.asarray(states["rank_order"][:, 0], dtype=np.int64)
    confirmation_all = np.asarray(projection["confirmation_mask"], dtype=bool)
    discovery_all = np.asarray(projection["discovery_mask"], dtype=bool)
    first_entropy = np.asarray(projection["first_entropy"], dtype=np.float64)
    first_projection = np.asarray(projection["first_projection"], dtype=np.float64)
    raw_entropy_r: list[float] = []
    old_winner_controlled_entropy_r: list[float] = []
    for layer in range(LAYERS):
        current_projection = first_projection[confirmation_all, layer]
        current_entropy = first_entropy[confirmation_all]
        current_winner = all_old_winner[confirmation_all]
        raw_entropy_r.append(_corr(current_projection, current_entropy))
        old_winner_controlled_entropy_r.append(
            _corr(
                _remove_group_means(current_projection, current_winner),
                _remove_group_means(current_entropy, current_winner),
            )
        )
    discovery_rows = np.flatnonzero(discovery_all)
    discovery_order = discovery_rows[
        np.argsort(first_entropy[discovery_all], kind="stable")
    ]
    quartile_count = max(1, int(np.floor(len(discovery_order) * 0.25)))
    low_rows = discovery_order[:quartile_count]
    high_rows = discovery_order[-quartile_count:]
    direction_validity = {
        "heldout_entropy_projection_r": raw_entropy_r,
        "heldout_entropy_projection_r_controlling_old_winner_letter": (
            old_winner_controlled_entropy_r
        ),
        "discovery_low_entropy_old_winner_letter_counts": np.bincount(
            all_old_winner[low_rows], minlength=4
        ).tolist(),
        "discovery_high_entropy_old_winner_letter_counts": np.bincount(
            all_old_winner[high_rows], minlength=4
        ).tolist(),
    }
    margins = np.maximum(np.asarray(run["first_top_two_margin"], dtype=np.float64), 0.0)
    weights = margins / max(float(margins.mean()), 1e-12)
    bootstrap = _bootstrap_indices(len(qids), draws, seed)
    projection_summary: dict[str, Any] = {}
    curves: dict[str, np.ndarray] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        for source in ("within", "cross"):
            values = np.asarray(
                projection[f"second_projection_{source}_z"][condition_index, selected],
                dtype=np.float64,
            ).T
            pointwise = _summary(values, bootstrap)
            simultaneous = _simultaneous_curve(values, bootstrap)
            projection_summary[f"{condition}_{source}"] = {
                **pointwise,
                **{
                    key: value for key, value in simultaneous.items()
                    if key != "mean"
                },
            }
            curves[f"projection_{condition}_{source}"] = values
    projection_summary["game_minus_neutral"] = {}
    for source in ("within", "cross"):
        values = (
            np.asarray(
                projection[f"second_projection_{source}_z"][0, selected],
                dtype=np.float64,
            )
            - np.asarray(
                projection[f"second_projection_{source}_z"][1, selected],
                dtype=np.float64,
            )
        ).T
        projection_summary["game_minus_neutral"][source] = {
            **_summary(values, bootstrap),
            "simultaneous95": _simultaneous_curve(values, bootstrap),
        }

    logits = np.asarray(run["logits"], dtype=np.float64)
    causal_summary: dict[str, Any] = {}
    contrast_summary: dict[str, Any] = {}
    effects_by_condition: dict[str, dict[str, np.ndarray]] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        effects = _effects(logits[condition_index], ranks)
        effects_by_condition[condition] = effects
        causal_summary[condition] = {}
        for scenario in SCENARIOS[1:]:
            causal_summary[condition][scenario] = {}
            rank_effect = effects[f"{scenario}:rank_centered"]
            for rank in range(4):
                causal_summary[condition][scenario][f"centered_R{rank + 1}"] = {
                    "absolute": _summary(rank_effect[..., rank], bootstrap),
                    "initial_margin_weighted": _summary(
                        rank_effect[..., rank], bootstrap, weights
                    ),
                }
            for endpoint in ("w1_minus_w2", "w1_choice"):
                values = effects[f"{scenario}:{endpoint}"]
                causal_summary[condition][scenario][endpoint] = {
                    "absolute": _summary(values, bootstrap),
                    "initial_margin_weighted": _summary(values, bootstrap, weights),
                }
            for rank in range(4):
                values = effects[f"{scenario}:choice_R{rank + 1}"]
                causal_summary[condition][scenario][f"choice_R{rank + 1}"] = {
                    "absolute": _summary(values, bootstrap),
                    "initial_margin_weighted": _summary(values, bootstrap, weights),
                }

        pairs = {
            "ablation_minus_random": ("uncertainty_ablation", "random_ablation"),
            "negative_steering_minus_random": (
                "uncertainty_steer_negative", "random_steer_negative"
            ),
            "positive_steering_minus_random": (
                "uncertainty_steer_positive", "random_steer_positive"
            ),
        }
        contrast_summary[condition] = {}
        for contrast, (uncertainty, random) in pairs.items():
            contrast_summary[condition][contrast] = {}
            for endpoint in ("w1_minus_w2", "w1_choice"):
                values = effects[f"{uncertainty}:{endpoint}"] - effects[f"{random}:{endpoint}"]
                contrast_summary[condition][contrast][endpoint] = {
                    "absolute": _summary(values, bootstrap),
                    "initial_margin_weighted": _summary(values, bootstrap, weights),
                    "absolute_simultaneous95": _simultaneous_curve(values, bootstrap),
                }
            rank_values = (
                effects[f"{uncertainty}:rank_centered"]
                - effects[f"{random}:rank_centered"]
            )
            for rank in range(4):
                values = rank_values[..., rank]
                contrast_summary[condition][contrast][f"centered_R{rank + 1}"] = {
                    "absolute": _summary(values, bootstrap),
                    "initial_margin_weighted": _summary(values, bootstrap, weights),
                    "absolute_simultaneous95": _simultaneous_curve(values, bootstrap),
                }
            if contrast == "ablation_minus_random":
                curves[f"ablation_{condition}_uncertainty"] = effects[
                    "uncertainty_ablation:rank_centered"
                ][..., 0]
                curves[f"ablation_{condition}_random"] = effects[
                    "random_ablation:rank_centered"
                ][..., 0]

        uncertainty_slope = (
            effects["uncertainty_steer_positive:rank_centered"][..., 0]
            - effects["uncertainty_steer_negative:rank_centered"][..., 0]
        ) / 6.0
        random_slope = (
            effects["random_steer_positive:rank_centered"][..., 0]
            - effects["random_steer_negative:rank_centered"][..., 0]
        ) / 6.0
        contrast_summary[condition]["signed_steering_slope"] = {
            "uncertainty": _summary(uncertainty_slope, bootstrap),
            "random": _summary(random_slope, bootstrap),
            "uncertainty_minus_random": _summary(
                uncertainty_slope - random_slope, bootstrap
            ),
            "uncertainty_minus_random_simultaneous95": _simultaneous_curve(
                uncertainty_slope - random_slope, bootstrap
            ),
        }
        contrast_summary[condition]["signed_steering_slope_by_rank"] = {}
        for rank in range(4):
            uncertainty_rank_slope = (
                effects["uncertainty_steer_positive:rank_centered"][..., rank]
                - effects["uncertainty_steer_negative:rank_centered"][..., rank]
            ) / 6.0
            random_rank_slope = (
                effects["random_steer_positive:rank_centered"][..., rank]
                - effects["random_steer_negative:rank_centered"][..., rank]
            ) / 6.0
            contrast_summary[condition]["signed_steering_slope_by_rank"][
                f"centered_R{rank + 1}"
            ] = {
                "uncertainty": _summary(uncertainty_rank_slope, bootstrap),
                "random": _summary(random_rank_slope, bootstrap),
                "uncertainty_minus_random": _summary(
                    uncertainty_rank_slope - random_rank_slope, bootstrap
                ),
                "uncertainty_minus_random_simultaneous95": _simultaneous_curve(
                    uncertainty_rank_slope - random_rank_slope, bootstrap
                ),
            }
        contrast_summary[condition]["signed_steering_slope_by_endpoint"] = {}
        for endpoint in ("w1_minus_w2", "w1_choice"):
            uncertainty_endpoint_slope = (
                effects[f"uncertainty_steer_positive:{endpoint}"]
                - effects[f"uncertainty_steer_negative:{endpoint}"]
            ) / 6.0
            random_endpoint_slope = (
                effects[f"random_steer_positive:{endpoint}"]
                - effects[f"random_steer_negative:{endpoint}"]
            ) / 6.0
            contrast_summary[condition]["signed_steering_slope_by_endpoint"][
                endpoint
            ] = {
                "uncertainty": _summary(uncertainty_endpoint_slope, bootstrap),
                "random": _summary(random_endpoint_slope, bootstrap),
                "uncertainty_minus_random": _summary(
                    uncertainty_endpoint_slope - random_endpoint_slope, bootstrap
                ),
                "uncertainty_minus_random_simultaneous95": _simultaneous_curve(
                    uncertainty_endpoint_slope - random_endpoint_slope, bootstrap
                ),
            }
        curves[f"steering_{condition}_uncertainty"] = uncertainty_slope
        curves[f"steering_{condition}_random"] = random_slope

    condition_difference: dict[str, Any] = {
        "definition": (
            "paired (Game intervention-minus-identity) minus "
            "(Neutral intervention-minus-identity)"
        ),
        "matched_random_contrasts": {},
    }
    pairs = {
        "ablation_minus_random": ("uncertainty_ablation", "random_ablation"),
        "negative_steering_minus_random": (
            "uncertainty_steer_negative", "random_steer_negative"
        ),
        "positive_steering_minus_random": (
            "uncertainty_steer_positive", "random_steer_positive"
        ),
    }
    game_effects = effects_by_condition["game"]
    neutral_effects = effects_by_condition["neutral"]
    for contrast, (uncertainty, random) in pairs.items():
        current: dict[str, Any] = {}
        for endpoint in ("w1_minus_w2", "w1_choice"):
            values = (
                game_effects[f"{uncertainty}:{endpoint}"]
                - game_effects[f"{random}:{endpoint}"]
                - neutral_effects[f"{uncertainty}:{endpoint}"]
                + neutral_effects[f"{random}:{endpoint}"]
            )
            current[endpoint] = {
                "absolute": _summary(values, bootstrap),
                "initial_margin_weighted": _summary(values, bootstrap, weights),
                "absolute_simultaneous95": _simultaneous_curve(values, bootstrap),
            }
        rank_values = (
            game_effects[f"{uncertainty}:rank_centered"]
            - game_effects[f"{random}:rank_centered"]
            - neutral_effects[f"{uncertainty}:rank_centered"]
            + neutral_effects[f"{random}:rank_centered"]
        )
        for rank in range(4):
            values = rank_values[..., rank]
            current[f"centered_R{rank + 1}"] = {
                "absolute": _summary(values, bootstrap),
                "initial_margin_weighted": _summary(values, bootstrap, weights),
                "absolute_simultaneous95": _simultaneous_curve(values, bootstrap),
            }
        condition_difference["matched_random_contrasts"][contrast] = current

    condition_difference["signed_steering_slope_by_rank"] = {}
    for rank in range(4):
        game_uncertainty_slope = (
            game_effects["uncertainty_steer_positive:rank_centered"][..., rank]
            - game_effects["uncertainty_steer_negative:rank_centered"][..., rank]
        ) / 6.0
        game_random_slope = (
            game_effects["random_steer_positive:rank_centered"][..., rank]
            - game_effects["random_steer_negative:rank_centered"][..., rank]
        ) / 6.0
        neutral_uncertainty_slope = (
            neutral_effects["uncertainty_steer_positive:rank_centered"][..., rank]
            - neutral_effects["uncertainty_steer_negative:rank_centered"][..., rank]
        ) / 6.0
        neutral_random_slope = (
            neutral_effects["random_steer_positive:rank_centered"][..., rank]
            - neutral_effects["random_steer_negative:rank_centered"][..., rank]
        ) / 6.0
        values = (
            game_uncertainty_slope - game_random_slope
            - neutral_uncertainty_slope + neutral_random_slope
        )
        condition_difference["signed_steering_slope_by_rank"][
            f"centered_R{rank + 1}"
        ] = {
            "absolute": _summary(values, bootstrap),
            "initial_margin_weighted": _summary(values, bootstrap, weights),
            "absolute_simultaneous95": _simultaneous_curve(values, bootstrap),
        }
    condition_difference["signed_steering_slope_by_endpoint"] = {}
    for endpoint in ("w1_minus_w2", "w1_choice"):
        game_uncertainty_slope = (
            game_effects[f"uncertainty_steer_positive:{endpoint}"]
            - game_effects[f"uncertainty_steer_negative:{endpoint}"]
        ) / 6.0
        game_random_slope = (
            game_effects[f"random_steer_positive:{endpoint}"]
            - game_effects[f"random_steer_negative:{endpoint}"]
        ) / 6.0
        neutral_uncertainty_slope = (
            neutral_effects[f"uncertainty_steer_positive:{endpoint}"]
            - neutral_effects[f"uncertainty_steer_negative:{endpoint}"]
        ) / 6.0
        neutral_random_slope = (
            neutral_effects[f"random_steer_positive:{endpoint}"]
            - neutral_effects[f"random_steer_negative:{endpoint}"]
        ) / 6.0
        values = (
            game_uncertainty_slope - game_random_slope
            - neutral_uncertainty_slope + neutral_random_slope
        )
        condition_difference["signed_steering_slope_by_endpoint"][endpoint] = {
            "absolute": _summary(values, bootstrap),
            "initial_margin_weighted": _summary(values, bootstrap, weights),
            "absolute_simultaneous95": _simultaneous_curve(values, bootstrap),
        }

    return (
        {
            "model": model_slug,
            "dataset": dataset_slug,
            "n_confirmation": len(qids),
            "bootstrap_draws": draws,
            "first_margin_mean": float(margins.mean()),
            "direction_validity": direction_validity,
            "direction_generalization": {
                "within_dataset_l64_entropy_projection_r": direction_fit[
                    "within_dataset"
                ][-1]["mean_diff_entropy_projection_r"],
                "cross_dataset_l64_entropy_projection_r": direction_fit[cross_key][
                    -1
                ]["mean_diff_entropy_projection_r"],
                "maximum_centered_answer_subspace_fraction": answer_fractions[
                    maximum_answer_layer
                ],
                "maximum_centered_answer_subspace_fraction_layer": (
                    maximum_answer_layer
                ),
                "selected_logit_lens_layers": selected_lens_layers,
            },
            "projection": projection_summary,
            "causal": causal_summary,
            "matched_random_contrasts": contrast_summary,
            "condition_difference": condition_difference,
            "validity": {
                "completed": bool(run["completed"].all()),
                "all_finite": bool(np.isfinite(run["logits"]).all()),
                "identity_trusted_max_abs_error": metadata[
                    "identity_trusted_max_abs_error"
                ],
                "identity_trusted_centered_max_abs_error": metadata[
                    "identity_trusted_centered_max_abs_error"
                ],
                "identity_trusted_argmax_agreement": metadata[
                    "identity_trusted_argmax_agreement"
                ],
                "identity_trusted_old_winner_choice_agreement": metadata[
                    "identity_trusted_old_winner_choice_agreement"
                ],
                "identity_layer_spread_max_abs_error": metadata[
                    "identity_layer_spread_max_abs_error"
                ],
                "duplicate_padding_max_abs_error": metadata[
                    "duplicate_padding_max_abs_error"
                ],
                "ablation_post_projection_max_abs": metadata[
                    "ablation_post_projection_max_abs"
                ],
                "ablation_post_projection_relative_to_pre_max": metadata[
                    "ablation_post_projection_relative_to_pre_max"
                ],
                "negative_steering_mean_shift": metadata[
                    "negative_steering_mean_shift"
                ],
                "positive_steering_mean_shift": metadata[
                    "positive_steering_mean_shift"
                ],
                "orthogonal_random_uncertainty_projection_shift_max_abs": metadata[
                    "orthogonal_random_uncertainty_projection_shift_max_abs"
                ],
                "random_ablation_post_projection_relative_to_pre_max": metadata[
                    "random_ablation_post_projection_relative_to_pre_max"
                ],
                "random_negative_steering_mean_shift": metadata[
                    "random_negative_steering_mean_shift"
                ],
                "random_positive_steering_mean_shift": metadata[
                    "random_positive_steering_mean_shift"
                ],
            },
        },
        curves,
    )


def _plot(cells: list[tuple[dict[str, Any], dict[str, np.ndarray]]], path: Path) -> None:
    import matplotlib.pyplot as plt

    layers = np.arange(1, LAYERS + 1)
    fig, axes = plt.subplots(len(cells), 3, figsize=(14.5, 3.1 * len(cells)), sharex=True)
    if len(cells) == 1:
        axes = axes[None]
    colors = {"game": "#b5392f", "neutral": "#2566a8"}
    for row, (cell, curves) in enumerate(cells):
        label = f"{cell['model']} — {cell['dataset']}"
        first_curve = curves["projection_game_within"]
        plot_bootstrap = _bootstrap_indices(
            first_curve.shape[1], 2000, 20260903 + row
        )
        for condition in CONDITIONS:
            values = curves[f"projection_{condition}_within"]
            mean = values.mean(axis=1)
            low, high = np.quantile(
                _bootstrap_means(values, plot_bootstrap), (0.025, 0.975), axis=1
            )
            axes[row, 0].plot(layers, mean, color=colors[condition], label=condition.title())
            axes[row, 0].fill_between(layers, low, high, color=colors[condition], alpha=0.18)
        axes[row, 0].axhline(0, color="#777777", lw=0.8)
        axes[row, 0].set_ylabel(label + "\n1P-standardized projection")
        for column, prefix, title in (
            (1, "ablation", "Ablation effect on old R1"),
            (2, "steering", "Signed steering slope on old R1"),
        ):
            for condition in CONDITIONS:
                for kind, linestyle, alpha in (
                    ("uncertainty", "-", 1.0), ("random", "--", 0.65)
                ):
                    values = curves[f"{prefix}_{condition}_{kind}"]
                    mean = values.mean(axis=1)
                    low, high = np.quantile(
                        _bootstrap_means(values, plot_bootstrap),
                        (0.025, 0.975), axis=1
                    )
                    axes[row, column].plot(
                        layers, mean, color=colors[condition], linestyle=linestyle,
                        alpha=alpha, label=f"{condition.title()} {kind}"
                    )
                    axes[row, column].fill_between(
                        layers, low, high, color=colors[condition], alpha=0.10
                    )
            axes[row, column].axhline(0, color="#777777", lw=0.8)
            axes[row, column].set_ylabel("Change in centered old-R1 logit")
            if row == 0:
                axes[row, column].set_title(title)
        if row == 0:
            axes[row, 0].set_title("Natural 2P uncertainty-vector activation")
    for axis in axes[-1]:
        axis.set_xlabel("Layer")
        axis.set_xticks([1, *range(8, 65, 8)])
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=7)
    axes[0, 2].legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle(
        "Recomputed MCQ uncertainty at the second decision\n"
        "solid = frozen 1P uncertainty direction; dashed = orthogonal random control",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def analyze(spec_paths: list[Path], output: Path, figure: Path, draws: int) -> None:
    cells: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for model_index, spec_path in enumerate(spec_paths):
        payload = json.loads(spec_path.read_text())
        direction_root = Path(payload["direction_output"])
        for dataset_index, dataset_spec in enumerate(payload["datasets"]):
            cells.append(
                _analyze_cell(
                    payload["model_name"], dataset_spec, direction_root, draws,
                    20260903 + model_index * 1000 + dataset_index * 100,
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "recomputed MCQ uncertainty at the second decision",
        "evidence_classes": {
            "projection": "activation/decoding: direct residual projection onto a frozen 1P direction",
            "intervention": "causal intervention: 2P decision-position post-block residual ablation/steering",
        },
        "cells": [cell for cell, _curves in cells],
    }
    _atomic_json(output / "summary.json", summary)
    _plot(cells, figure)

    lines = [
        "# A general MCQ uncertainty direction at the second decision",
        "",
        "## Bottom line",
        "",
        "A frozen direction learned only from first-presentation multiple-choice entropy "
        "reappears in the second-decision residual stream and is causally coupled to the "
        "second answer ranking in both Qwen3.6-27B and Seed-OSS 36B, on both SimpleMC and "
        "TriviaMC. The cleanest causal evidence is bidirectional steering: moving in the "
        "high-uncertainty direction generally lowers the old winner relative to the other "
        "candidates at middle-to-late layers, whereas an equal-dose orthogonal random "
        "direction does not.",
        "",
        "That effect is **not specific to Game**. It is similar in Game and Neutral for "
        "Seed and is substantially stronger in Neutral for Qwen. At the tested dose it "
        "reliably moves logits but rarely changes the discrete answer. Thus this is evidence "
        "for a general uncertainty-like ranking control available at 2P, but not evidence "
        "that Game preferential switching is driven by a special readout of that control.",
        "",
        "This experiment does not train on, predict, or define the direction using 2P "
        "output uncertainty. It measures the activation of the frozen 1P direction at 2P "
        "and intervenes on that coordinate directly. It also does not test remembered 1P "
        "uncertainty; earlier first-decision source interventions make simple copying from "
        "that position unlikely, so *re-instantiation at 2P* is the best-supported inference, "
        "not a directly traced construction mechanism.",
        "",
        "## Scope and validity",
        "",
        f"All estimates use the frozen confirmation questions and {draws:,} paired "
        "question-bootstrap draws. Every causal cell covers L1-L64 in Game and Neutral "
        "separately and is compared with a same-norm orthogonal random direction.",
        "",
    ]
    for cell, _curves in cells:
        validity = cell["validity"]
        direction_validity = cell["direction_validity"]
        generalization = cell["direction_generalization"]
        lines.extend(
            [
                f"- **{cell['model']} / {cell['dataset']}**: n={cell['n_confirmation']}; "
                f"identity raw/centered error "
                f"{validity['identity_trusted_max_abs_error']:.6g}/"
                f"{validity['identity_trusted_centered_max_abs_error']:.6g}; "
                f"identity argmax agreement "
                f"{100 * validity['identity_trusted_argmax_agreement']:.1f}%; "
                f"held-out L64 entropy correlation raw/controlling old-winner letter "
                f"{direction_validity['heldout_entropy_projection_r'][-1]:.3f}/"
                f"{direction_validity['heldout_entropy_projection_r_controlling_old_winner_letter'][-1]:.3f}; "
                f"cross-dataset L64 correlation "
                f"{generalization['cross_dataset_l64_entropy_projection_r']:.3f}; "
                f"maximum direct answer-contrast fraction "
                f"{100 * generalization['maximum_centered_answer_subspace_fraction']:.2f}% "
                f"at L{generalization['maximum_centered_answer_subspace_fraction_layer']}; "
                f"ablation residual fraction "
                f"{validity['ablation_post_projection_relative_to_pre_max']:.4%}; "
                f"random-ablation residual fraction "
                f"{validity['random_ablation_post_projection_relative_to_pre_max']:.4%}; "
                f"random steering means "
                f"{validity['random_negative_steering_mean_shift']:.4f}/"
                f"{validity['random_positive_steering_mean_shift']:.4f}; "
                f"all finite={validity['all_finite']}.",
            ]
        )
    lines.extend(
        [
            "",
            "The old-winner-letter control is important: removing the mean projection "
            "within each displayed old-winner letter leaves the entropy correlations "
            "essentially unchanged. The frozen direction is therefore not simply a code "
            "for whether A, B, C, or D won. Its direct Euclidean overlap with the centered "
            "four-answer output subspace is also small in every cell. Seed's middle/late "
            "logit lens is semantically recognizable (`unknown`, `none`, and Chinese "
            "equivalents); Qwen's top vocabulary tokens are less clean. Cross-dataset "
            "transfer is strong for both models.",
            "",
            "## Natural 2P activation",
            "",
            "The left column of the figure is the projection itself, standardized by the "
            "1P discovery distribution. It is strongly structured and frequently far from "
            "zero in all four cells. Its sign and Game-versus-Neutral ordering vary by "
            "layer, model, and dataset; there is no universal scalar level that separates "
            "Game from Neutral. This answers the activation question without redefining "
            "the target around the model's eventual 2P logits.",
            "",
            "## Causal steering",
            "",
            "The table reports the strongest middle-to-late (L33-L64) signed steering "
            "effect on the centered old-R1 logit. A value of -0.02 means that each +1 unit "
            "step along the frozen high-uncertainty direction lowers old R1 by 0.02 logits "
            "relative to the four-answer mean, beyond the matched random-direction effect.",
            "",
            "| Model / dataset | Game | Neutral | Policy-specific reading |",
            "|---|---:|---:|---|",
        ]
    )
    for cell, _curves in cells:
        condition_fields = []
        for condition in CONDITIONS:
            result = cell["matched_random_contrasts"][condition][
                "signed_steering_slope"
            ]["uncertainty_minus_random"]
            means = np.asarray(result["mean"], dtype=np.float64)
            layer_index = 32 + int(np.argmax(np.abs(means[32:])))
            condition_fields.append(
                f"L{layer_index + 1} {means[layer_index]:+.4f} "
                f"[{result['ci95_low'][layer_index]:+.4f}, "
                f"{result['ci95_high'][layer_index]:+.4f}]"
            )
        interaction = cell["condition_difference"]["signed_steering_slope_by_rank"][
            "centered_R1"
        ]["absolute"]
        interaction_mean = np.asarray(interaction["mean"], dtype=np.float64)
        interaction_index = 32 + int(np.argmax(np.abs(interaction_mean[32:])))
        interaction_text = (
            f"largest late G-N: L{interaction_index + 1} "
            f"{interaction_mean[interaction_index]:+.4f} "
            f"[{interaction['ci95_low'][interaction_index]:+.4f}, "
            f"{interaction['ci95_high'][interaction_index]:+.4f}]"
        )
        lines.append(
            f"| {cell['model']} / {cell['dataset']} | {condition_fields[0]} | "
            f"{condition_fields[1]} | {interaction_text} |"
        )
    lines.extend(
        [
            "",
            "At those same layers the full centered-rank vectors are not generic gain "
            "changes: old R1 moves down while lower-ranked candidates move up. The "
            "initial-margin-weighted analysis preserves and usually strengthens the same "
            "late-layer sign. Seed SimpleMC also has an early, opposite-signed L1-L9 effect; "
            "its later L37-L55 effect matches the other three cells.",
            "",
            "The policy-specific column is decisive. Qwen's positive Game-minus-Neutral "
            "interaction means that high-uncertainty steering suppresses old R1 *less* in "
            "Game than in Neutral. Seed's Game and Neutral responses are nearly the same: "
            "the familywise Game-minus-Neutral R1 band excludes zero at no layer in "
            "SimpleMC and only at isolated L37 in TriviaMC. This is the opposite of the "
            "simple metacognitive hypothesis in which Game uniquely reads the uncertainty "
            "coordinate to decide whether to abandon its old answer.",
            "",
            "Discrete old-W1 choice slopes at the strongest logit-effect layers are small "
            "and their ordinary 95% intervals generally include zero. The causal claim is "
            "therefore about candidate scoring, not a demonstrated change in switch rate "
            "at the frozen ±3 dose.",
            "",
            "## Ablation and its limit",
            "",
            "Projection ablation changes the answer computation, but it is not an "
            "edit-norm-matched comparison with the orthogonal random ablation. Both "
            "directions are unit vectors, yet ablation subtracts each state's complete "
            "natural projection. At the layers with the largest old-R1 effects, the mean "
            "absolute uncertainty-coordinate edits are about 57–562 residual units, versus "
            "about 3–50 for the random coordinate. Its very large, non-monotonic late "
            "effects—especially in Seed—therefore establish that this aggressive coordinate "
            "removal changes the output, but they do not isolate a physiological effect size "
            "or direction. The ±3 bidirectional steering comparison is truly dose-matched "
            "and is the primary signed causal result.",
            "",
            "## Conclusion",
            "",
            "The narrow hypothesis succeeds: a dataset-general 1P MCQ uncertainty axis is "
            "naturally present again at the 2P decision position, and changing that axis "
            "causally changes the final candidate ranking in two architectures and two "
            "datasets. The stronger metacognitive interpretation does not: the causal "
            "effect is shared with Neutral and, in Qwen, is stronger there. The measured "
            "axis is therefore best described as a general uncertainty-like control on "
            "answer ranking that both conditions can use, not the condition-specific "
            "trigger for preferential Game switching.",
            "",
            "The complete machine-readable layerwise estimates, pointwise intervals, "
            "familywise simultaneous bands, all four rank effects, W1-minus-W2 margins, "
            "choice effects, initial-margin-weighted estimates, and paired Game-minus-Neutral "
            "contrasts are in `summary.json`.",
            "",
            "Interpretation must distinguish natural activation from causal use. A projection "
            "alone is activation evidence; an uncertainty intervention that differs from the "
            "orthogonal random control is causal evidence. A null remains bounded because a "
            "single direction may be redundant with other uncertainty directions.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "figure": str(figure), "cells": len(cells)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specs", type=Path, nargs="+", default=[
            Path("outputs/model_replications/mcq_uncertainty_policy/specs/qwen36_27b.json"),
            Path("outputs/model_replications/mcq_uncertainty_policy/specs/seed_oss_36b.json"),
        ]
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/model_replications/mcq_uncertainty_policy/analysis"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("figures/model_replications/mcq_uncertainty_policy.png"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    analyze(args.specs, args.output, args.figure, args.bootstrap_draws)


if __name__ == "__main__":
    main()
