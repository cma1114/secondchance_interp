from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .compression_robustness_analysis import (
    _aligned_rank_contrasts,
    _cross_fitted_flexible_residuals,
)
from .data import load_activation_dataset
from .perturbation_analysis import _style
from .probes import stratified_folds
from .trajectory_analysis import centered


OUTCOME_LABELS = {
    "runner_switch": "Game-only switch to runner",
    "any_switch": "Any Game-only switch",
    "lower_switch": "Game-only switch to rank 3/4",
}


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=-1)


def _aligned(logits: np.ndarray, order: np.ndarray) -> np.ndarray:
    if logits.ndim == 2:
        return np.take_along_axis(logits, order, axis=-1)
    return np.take_along_axis(logits, order[:, None, :], axis=-1)


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, float]:
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    model.fit(scaler.transform(train_x), train_y)
    probability = model.predict_proba(scaler.transform(test_x))[:, 1]
    return probability, float(model.coef_[0, -1])


def _fit_predict_calibrated(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, float]:
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, max_iter=3000)
    model.fit(scaler.transform(train_x), train_y)
    probability = model.predict_proba(scaler.transform(test_x))[:, 1]
    return probability, float(model.coef_[0, -1])


def _control_features(
    logits: np.ndarray,
    order: np.ndarray,
    baseline_correct: np.ndarray,
    layer: int,
) -> np.ndarray:
    baseline_final = _aligned(logits[:, 0, -1], order)
    neutral_layer = _aligned(logits[:, 2, layer], order)
    winner_letter = np.eye(4)[order[:, 0], :3]
    return np.column_stack(
        [
            baseline_final[:, :3],
            baseline_final[:, 0] - baseline_final[:, 1],
            baseline_final.std(axis=-1),
            _entropy(baseline_final),
            baseline_correct.astype(float),
            winner_letter,
            neutral_layer[:, :3],
            neutral_layer[:, 0] - neutral_layer[:, 1],
            neutral_layer[:, 1] - neutral_layer[:, 2:].mean(axis=-1),
            neutral_layer.std(axis=-1),
        ]
    )


def _fixed_layer_cv(
    controls: dict[int, np.ndarray],
    signal: np.ndarray,
    ids: np.ndarray,
    labels: np.ndarray,
    layers: list[int],
    seed: int,
) -> dict:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = {}
    for layer in layers:
        control_prediction = np.empty(len(ids))
        augmented_prediction = np.empty(len(ids))
        coefficients = []
        for train_local, test_local in splitter.split(ids, labels):
            train, test = ids[train_local], ids[test_local]
            control_prediction[test_local], _ = _fit_predict(
                controls[layer][train], labels[train_local], controls[layer][test]
            )
            augmented_prediction[test_local], coefficient = _fit_predict(
                np.column_stack([controls[layer][train], signal[train, layer]]),
                labels[train_local],
                np.column_stack([controls[layer][test], signal[test, layer]]),
            )
            coefficients.append(coefficient)
        rows[layer] = {
            "control_auc": float(roc_auc_score(labels, control_prediction)),
            "augmented_auc": float(roc_auc_score(labels, augmented_prediction)),
            "delta_auc": float(
                roc_auc_score(labels, augmented_prediction)
                - roc_auc_score(labels, control_prediction)
            ),
            "mean_standardized_signal_coefficient": float(np.mean(coefficients)),
        }
    return rows


def _nested_predictions(
    controls: dict[int, np.ndarray],
    signal: np.ndarray,
    ids: np.ndarray,
    labels: np.ndarray,
    candidate_layers: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    control_prediction = np.empty(len(ids))
    augmented_prediction = np.empty(len(ids))
    selected_layers = []
    coefficients = []
    for outer_fold, (train_local, test_local) in enumerate(outer.split(ids, labels)):
        train_ids, test_ids = ids[train_local], ids[test_local]
        train_labels = labels[train_local]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 100 + outer_fold)
        layer_scores = []
        for layer in candidate_layers:
            inner_control = np.empty(len(train_ids))
            inner_augmented = np.empty(len(train_ids))
            for inner_train_local, inner_test_local in inner.split(train_ids, train_labels):
                inner_train, inner_test = train_ids[inner_train_local], train_ids[inner_test_local]
                inner_control[inner_test_local], _ = _fit_predict(
                    controls[layer][inner_train],
                    train_labels[inner_train_local],
                    controls[layer][inner_test],
                )
                inner_augmented[inner_test_local], _ = _fit_predict(
                    np.column_stack([controls[layer][inner_train], signal[inner_train, layer]]),
                    train_labels[inner_train_local],
                    np.column_stack([controls[layer][inner_test], signal[inner_test, layer]]),
                )
            layer_scores.append(
                roc_auc_score(train_labels, inner_augmented)
                - roc_auc_score(train_labels, inner_control)
            )
        selected = candidate_layers[int(np.argmax(layer_scores))]
        selected_layers.append(selected)
        control_prediction[test_local], _ = _fit_predict(
            controls[selected][train_ids], train_labels, controls[selected][test_ids]
        )
        augmented_prediction[test_local], coefficient = _fit_predict(
            np.column_stack([controls[selected][train_ids], signal[train_ids, selected]]),
            train_labels,
            np.column_stack([controls[selected][test_ids], signal[test_ids, selected]]),
        )
        coefficients.append(coefficient)
    return control_prediction, augmented_prediction, selected_layers, coefficients


def _auc_inference(
    labels: np.ndarray,
    control_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    control_auc = float(roc_auc_score(labels, control_prediction))
    augmented_auc = float(roc_auc_score(labels, augmented_prediction))
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, 3))
    for repetition in range(repetitions):
        sample = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        control = roc_auc_score(labels[sample], control_prediction[sample])
        augmented = roc_auc_score(labels[sample], augmented_prediction[sample])
        draws[repetition] = control, augmented, augmented - control
    return {
        "control_auc": control_auc,
        "control_ci": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
        "augmented_auc": augmented_auc,
        "augmented_ci": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
        "delta_auc": augmented_auc - control_auc,
        "delta_ci": np.quantile(draws[:, 2], [0.025, 0.975]).tolist(),
    }


def _nested_calibrated_predictions(
    controls: dict[int, np.ndarray],
    signal: np.ndarray,
    ids: np.ndarray,
    labels: np.ndarray,
    candidate_layers: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[float]]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    null_prediction = np.empty(len(ids))
    control_prediction = np.empty(len(ids))
    augmented_prediction = np.empty(len(ids))
    selected_layers = []
    coefficients = []
    for outer_fold, (train_local, test_local) in enumerate(outer.split(ids, labels)):
        train_ids, test_ids = ids[train_local], ids[test_local]
        train_labels = labels[train_local]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 100 + outer_fold)
        layer_scores = []
        for layer in candidate_layers:
            inner_control = np.empty(len(train_ids))
            inner_augmented = np.empty(len(train_ids))
            for inner_train_local, inner_test_local in inner.split(train_ids, train_labels):
                inner_train, inner_test = train_ids[inner_train_local], train_ids[inner_test_local]
                inner_control[inner_test_local], _ = _fit_predict_calibrated(
                    controls[layer][inner_train],
                    train_labels[inner_train_local],
                    controls[layer][inner_test],
                )
                inner_augmented[inner_test_local], _ = _fit_predict_calibrated(
                    np.column_stack([controls[layer][inner_train], signal[inner_train, layer]]),
                    train_labels[inner_train_local],
                    np.column_stack([controls[layer][inner_test], signal[inner_test, layer]]),
                )
            layer_scores.append(
                log_loss(train_labels, inner_control, labels=[0, 1])
                - log_loss(train_labels, inner_augmented, labels=[0, 1])
            )
        selected = candidate_layers[int(np.argmax(layer_scores))]
        selected_layers.append(selected)
        null_prediction[test_local] = train_labels.mean()
        control_prediction[test_local], _ = _fit_predict_calibrated(
            controls[selected][train_ids], train_labels, controls[selected][test_ids]
        )
        augmented_prediction[test_local], coefficient = _fit_predict_calibrated(
            np.column_stack([controls[selected][train_ids], signal[train_ids, selected]]),
            train_labels,
            np.column_stack([controls[selected][test_ids], signal[test_ids, selected]]),
        )
        coefficients.append(coefficient)
    return null_prediction, control_prediction, augmented_prediction, selected_layers, coefficients


def _variation_metrics(
    labels: np.ndarray,
    null_prediction: np.ndarray,
    control_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
) -> dict[str, float]:
    eps = 1e-9
    null_prediction = np.clip(null_prediction, eps, 1 - eps)
    control_prediction = np.clip(control_prediction, eps, 1 - eps)
    augmented_prediction = np.clip(augmented_prediction, eps, 1 - eps)
    null_log_loss = float(log_loss(labels, null_prediction, labels=[0, 1]))
    control_log_loss = float(log_loss(labels, control_prediction, labels=[0, 1]))
    augmented_log_loss = float(log_loss(labels, augmented_prediction, labels=[0, 1]))
    null_brier = float(brier_score_loss(labels, null_prediction))
    control_brier = float(brier_score_loss(labels, control_prediction))
    augmented_brier = float(brier_score_loss(labels, augmented_prediction))

    def tjur(probability: np.ndarray) -> float:
        return float(probability[labels == 1].mean() - probability[labels == 0].mean())

    def nagelkerke(probability: np.ndarray) -> float:
        ll_model = float(np.sum(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
        ll_null = float(np.sum(labels * np.log(null_prediction) + (1 - labels) * np.log(1 - null_prediction)))
        cox_snell = 1 - np.exp((2 / len(labels)) * (ll_null - ll_model))
        maximum = 1 - np.exp((2 / len(labels)) * ll_null)
        return float(cox_snell / maximum) if maximum > 0 else 0.0

    return {
        "null_log_loss": null_log_loss,
        "control_log_loss": control_log_loss,
        "augmented_log_loss": augmented_log_loss,
        "control_cross_entropy_r2": 1 - control_log_loss / null_log_loss,
        "augmented_cross_entropy_r2": 1 - augmented_log_loss / null_log_loss,
        "incremental_cross_entropy_r2": (control_log_loss - augmented_log_loss) / null_log_loss,
        "partial_log_loss_reduction": (control_log_loss - augmented_log_loss) / control_log_loss,
        "control_brier_r2": 1 - control_brier / null_brier,
        "augmented_brier_r2": 1 - augmented_brier / null_brier,
        "incremental_brier_r2": (control_brier - augmented_brier) / null_brier,
        "control_tjur_r2": tjur(control_prediction),
        "augmented_tjur_r2": tjur(augmented_prediction),
        "incremental_tjur_r2": tjur(augmented_prediction) - tjur(control_prediction),
        "control_nagelkerke_r2": nagelkerke(control_prediction),
        "augmented_nagelkerke_r2": nagelkerke(augmented_prediction),
        "incremental_nagelkerke_r2": nagelkerke(augmented_prediction) - nagelkerke(control_prediction),
    }


def _variation_inference(
    labels: np.ndarray,
    null_prediction: np.ndarray,
    control_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    observed = _variation_metrics(labels, null_prediction, control_prediction, augmented_prediction)
    keys = [key for key in observed if key.startswith("incremental_") or key == "partial_log_loss_reduction"]
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    draws = {key: np.empty(repetitions) for key in keys}
    for repetition in range(repetitions):
        sample = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        metrics = _variation_metrics(
            labels[sample],
            null_prediction[sample],
            control_prediction[sample],
            augmented_prediction[sample],
        )
        for key in keys:
            draws[key][repetition] = metrics[key]
    observed["bootstrap_ci"] = {
        key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in draws.items()
    }
    return observed


def _group_trajectory(signal: np.ndarray, group_ids: dict[str, np.ndarray], layers: list[int], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for name, ids in group_ids.items():
        values = signal[ids][:, layers]
        draws = np.empty((2000, len(layers)))
        for repetition in range(len(draws)):
            draws[repetition] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
        result[name] = {
            "n": len(ids),
            "mean": values.mean(axis=0).tolist(),
            "ci_low": np.quantile(draws, 0.025, axis=0).tolist(),
            "ci_high": np.quantile(draws, 0.975, axis=0).tolist(),
        }
    return result


def _plot(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    layers = np.asarray(summary["layers"])
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.05))
    group_styles = {
        "neither": ("Neither switches", "#777777", "-"),
        "runner_switch": ("Game switches to runner", "#0072B2", "-"),
        "lower_switch": ("Game switches to rank 3/4", "#D55E00", "--"),
    }
    for group, (label, color, style) in group_styles.items():
        row = summary["group_trajectory"][group]
        mean, low, high = (np.asarray(row[key]) for key in ("mean", "ci_low", "ci_high"))
        axes[0].fill_between(layers, low, high, color=color, alpha=0.13, linewidth=0)
        axes[0].plot(layers, mean, color=color, ls=style, lw=1.45, label=f"{label} (n={row['n']})")
    axes[0].set_title("A  Candidate runner signal", loc="left", fontweight="bold")
    axes[0].set_ylabel("Game minus Neutral flexible residual\n(natural-logit units)")
    axes[0].legend(frameon=False, fontsize=7.2, loc="upper left")

    primary = summary["outcomes"]["runner_switch"]["fixed_layer"]
    control = np.asarray([primary[str(layer)]["control_auc"] for layer in layers])
    augmented = np.asarray([primary[str(layer)]["augmented_auc"] for layer in layers])
    axes[1].plot(layers, control, color="#777777", lw=1.45, label="Baseline + Neutral controls")
    axes[1].plot(layers, augmented, color="#0072B2", lw=1.45, label="Controls + runner signal")
    axes[1].axhline(0.5, color="#777777", ls=":", lw=0.8)
    axes[1].set_title("B  Held-out runner-switch prediction", loc="left", fontweight="bold")
    axes[1].set_ylabel("ROC AUC")
    axes[1].legend(frameon=False, fontsize=7.2, loc="lower right")

    outcome_names = ("runner_switch", "any_switch", "lower_switch")
    x = np.arange(3)
    means = np.asarray([summary["outcomes"][name]["nested"]["delta_auc"] for name in outcome_names])
    lows = np.asarray([summary["outcomes"][name]["nested"]["delta_ci"][0] for name in outcome_names])
    highs = np.asarray([summary["outcomes"][name]["nested"]["delta_ci"][1] for name in outcome_names])
    axes[2].bar(x, means, 0.55, color="#009E73")
    axes[2].errorbar(x, means, yerr=np.vstack([means - lows, highs - means]), fmt="none", ecolor="#333333", capsize=2, lw=0.8)
    axes[2].set_xticks(x, ("To runner", "Any switch", "To ranks 3/4"), rotation=18, ha="right")
    axes[2].set_title("C  Nested layer selection", loc="left", fontweight="bold")
    axes[2].set_ylabel("Incremental held-out AUC")

    for axis in axes:
        axis.axhline(0, color="#555555", lw=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.7)
        axis.set_axisbelow(True)
    axes[0].set_xlabel("Residual readout")
    axes[1].set_xlabel("Residual readout")
    figure.tight_layout(w_pad=1.25)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"runner_predictive_bridge.{suffix}", bbox_inches="tight")
    plt.close(figure)


def analyze(input_dir: str | Path, output_dir: str | Path, bootstrap: int, seed: int) -> dict:
    data = load_activation_dataset(input_dir, ["baseline", "incorrect", "neutral"])
    logits = centered(data.logits)
    n, _, n_layers, _ = logits.shape
    final_baseline = logits[:, 0, -1]
    order = np.argsort(-final_baseline, axis=-1)
    winner, runner = order[:, 0], order[:, 1]
    choices = np.argmax(logits[:, :, -1], axis=-1)
    game_switch = choices[:, 1] != winner
    neutral_switch = choices[:, 2] != winner
    neither = ~game_switch & ~neutral_switch
    game_only = game_switch & ~neutral_switch
    runner_switch = game_only & (choices[:, 1] == runner)
    lower_switch = game_only & (choices[:, 1] != runner)
    baseline_correct = np.asarray(
        [bool(data.metadata[(qid, "baseline")]["baseline_correct"]) for qid in data.question_ids]
    )

    layers = list(range(max(0, n_layers - 25), n_layers))
    candidate_layers = list(range(max(0, n_layers - 21), n_layers - 8))
    compression_folds = stratified_folds(winner, 5, seed)
    signal = np.full((n, n_layers), np.nan)
    for layer in layers:
        baseline_layer = logits[:, 0, layer]
        condition_residuals = {}
        for condition_index, condition in ((1, "incorrect"), (2, "neutral")):
            condition_residuals[condition] = _cross_fitted_flexible_residuals(
                baseline_layer,
                logits[:, condition_index, layer] - baseline_layer,
                order,
                compression_folds,
                excluded_rank=1,
                alpha=1.0,
                n_knots=6,
            )
        signal[:, layer] = (
            _aligned_rank_contrasts(condition_residuals["incorrect"], order)[:, 1]
            - _aligned_rank_contrasts(condition_residuals["neutral"], order)[:, 1]
        )

    controls = {
        layer: _control_features(logits, order, baseline_correct, layer) for layer in layers
    }
    definitions = {
        "runner_switch": (neither | runner_switch, runner_switch),
        "any_switch": (neither | game_only, game_only),
        "lower_switch": (neither | lower_switch, lower_switch),
    }
    summary = {
        "n_questions": n,
        "layers": layers,
        "candidate_layers_nested": candidate_layers,
        "groups": {
            "neither": int(neither.sum()),
            "game_only": int(game_only.sum()),
            "runner_switch": int(runner_switch.sum()),
            "lower_switch": int(lower_switch.sum()),
            "neutral_only": int((~game_switch & neutral_switch).sum()),
            "both_switch": int((game_switch & neutral_switch).sum()),
        },
        "controls": [
            "final baseline ordered A-D scores",
            "baseline winner-runner margin, spread, entropy, and correctness",
            "baseline winner letter",
            "same-layer Neutral ordered A-D scores, winner margin, runner-vs-lower contrast, and spread",
        ],
        "group_trajectory": _group_trajectory(
            signal,
            {
                "neither": np.flatnonzero(neither),
                "runner_switch": np.flatnonzero(runner_switch),
                "lower_switch": np.flatnonzero(lower_switch),
            },
            layers,
            seed,
        ),
        "outcomes": {},
    }
    csv_rows = []
    for outcome_index, (name, (include, positive)) in enumerate(definitions.items()):
        ids = np.flatnonzero(include)
        labels = positive[ids].astype(int)
        fixed = _fixed_layer_cv(controls, signal, ids, labels, layers, seed + outcome_index)
        control_pred, augmented_pred, selected, coefficients = _nested_predictions(
            controls,
            signal,
            ids,
            labels,
            candidate_layers,
            seed + 20 + outcome_index,
        )
        nested = _auc_inference(labels, control_pred, augmented_pred, bootstrap, seed + 50 + outcome_index)
        nested.update(
            {
                "n": len(ids),
                "positives": int(labels.sum()),
                "selected_layers": selected,
                "mean_standardized_signal_coefficient": float(np.mean(coefficients)),
                "fold_signal_coefficients": coefficients,
            }
        )
        calibrated_null, calibrated_control, calibrated_augmented, calibrated_layers, calibrated_coefficients = (
            _nested_calibrated_predictions(
                controls,
                signal,
                ids,
                labels,
                candidate_layers,
                seed + 120 + outcome_index,
            )
        )
        calibrated = _variation_inference(
            labels,
            calibrated_null,
            calibrated_control,
            calibrated_augmented,
            bootstrap,
            seed + 150 + outcome_index,
        )
        calibrated.update(
            {
                "selected_layers": calibrated_layers,
                "mean_standardized_signal_coefficient": float(np.mean(calibrated_coefficients)),
                "fold_signal_coefficients": calibrated_coefficients,
            }
        )
        summary["outcomes"][name] = {
            "label": OUTCOME_LABELS[name],
            "fixed_layer": {str(layer): row for layer, row in fixed.items()},
            "nested": nested,
            "calibrated_nested": calibrated,
        }
        for layer, row in fixed.items():
            csv_rows.append({"outcome": name, "layer": layer, **row})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runner_predictive_bridge.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        output / "runner_predictive_bridge.npz",
        signal=signal,
        order=order,
        choices=choices,
        baseline_correct=baseline_correct,
    )
    with (output / "runner_predictive_bridge.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    _plot(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether a pre-output runner residual predicts later Game switching")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()
