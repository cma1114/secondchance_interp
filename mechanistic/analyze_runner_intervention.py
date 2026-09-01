from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .io import read_metadata, shard_path


def _available(root: Path, scenario: str) -> set[str]:
    return {path.stem for path in (root / "shards" / scenario).glob("*.npz")}


def _load(root: Path, scenarios: list[str]) -> tuple[list[str], dict[str, np.ndarray], dict]:
    qids = sorted(set.intersection(*[_available(root, scenario) for scenario in scenarios]))
    if not qids:
        raise FileNotFoundError("No questions are complete across all intervention scenarios")
    logits, metadata = {}, {}
    for scenario in scenarios:
        rows = []
        for qid in qids:
            with np.load(shard_path(root, scenario, qid), allow_pickle=False) as source:
                rows.append(source["final_canonical_logits"].astype(np.float64))
                metadata[(scenario, qid)] = read_metadata(source)
        logits[scenario] = np.asarray(rows)
    return qids, logits, metadata


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=-1, keepdims=True)
    return -np.sum(probability * np.log2(np.clip(probability, 1e-12, 1.0)), axis=-1)


def _bootstrap_sum(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> list[float]:
    draws = np.empty(repetitions)
    n = len(values)
    for index in range(repetitions):
        draws[index] = values[rng.integers(0, n, n)].sum()
    return np.quantile(draws, [0.025, 0.975]).tolist()


def _effect(
    values: np.ndarray,
    natural_gap: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    effect_draws = np.empty(repetitions)
    fraction_draws = []
    for index in range(repetitions):
        sample = rng.integers(0, n, n)
        effect = values[sample].sum()
        denominator = natural_gap[sample].sum()
        effect_draws[index] = effect
        if denominator > 0:
            fraction_draws.append(effect / denominator)
    return {
        "count_effect": float(values.sum()),
        "rate_effect": float(values.mean()),
        "count_ci": np.quantile(effect_draws, [0.025, 0.975]).tolist(),
        "rate_ci": (np.quantile(effect_draws, [0.025, 0.975]) / n).tolist(),
        "fraction_of_natural_net_runner_gap": float(values.sum() / natural_gap.sum()),
        "fraction_ci": np.quantile(fraction_draws, [0.025, 0.975]).tolist(),
    }


def _paired_count(values: np.ndarray, repetitions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(repetitions)
    for index in range(repetitions):
        draws[index] = values[rng.integers(0, n, n)].sum()
    return {
        "count": float(values.sum()),
        "count_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "rate": float(values.mean()),
        "rate_ci": (np.quantile(draws, [0.025, 0.975]) / n).tolist(),
    }


def _plot(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ("Targeted\nrunner", "Rank 3\ncontrol", "Orthogonal\ncontrol")
    game_counts = [
        summary["scenario_metrics"][name]["runner_count"]
        for name in (
            "game_remove_x0p5",
            "game_remove_rank3_control_x0p5",
            "game_remove_orthogonal_control_x0p5",
        )
    ]
    neutral_counts = [
        summary["scenario_metrics"][name]["runner_count"]
        for name in (
            "neutral_add_x0p5",
            "neutral_add_rank3_control_x0p5",
            "neutral_add_orthogonal_control_x0p5",
        )
    ]
    closure_rows = [
        summary["primary"]["x0p5"]["combined_gap_closure"],
        summary["controls"]["rank3_control_x0p5"]["combined_gap_closure"],
        summary["controls"]["orthogonal_control_x0p5"]["combined_gap_closure"],
        summary["primary"]["x1"]["combined_gap_closure"],
    ]
    closure_labels = ("Targeted 0.5×", "Rank 3 0.5×", "Orthogonal 0.5×", "Targeted 1×")

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.35))
    x = np.arange(3)
    axes[0].axhline(summary["natural"]["game_runner_count"], color="#0072B2", lw=0.9, alpha=0.7)
    axes[0].axhline(summary["natural"]["neutral_runner_count"], color="#D55E00", lw=0.9, alpha=0.7)
    axes[0].plot(x, game_counts, "o", color="#0072B2", label="Game (natural = 51)")
    axes[0].plot(x, neutral_counts, "s", color="#D55E00", label="Neutral (natural = 26)")
    for index, value in enumerate(game_counts):
        axes[0].text(index, value - 1.2, str(value), ha="center", va="top", fontsize=8)
    for index, value in enumerate(neutral_counts):
        axes[0].text(index, value + 1.1, str(value), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Final runner-up choices")
    axes[0].set_title("A  Matched 0.5× interventions", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="center right")

    y = np.arange(len(closure_rows))
    means = np.asarray([row["count_effect"] for row in closure_rows])
    lows = np.asarray([row["count_ci"][0] for row in closure_rows])
    highs = np.asarray([row["count_ci"][1] for row in closure_rows])
    colors = ["#0072B2", "#999999", "#666666", "#56B4E9"]
    axes[1].axvline(0, color="#555555", lw=0.8)
    axes[1].errorbar(
        means,
        y,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="none",
        ecolor="#444444",
        capsize=2.5,
        lw=0.9,
    )
    axes[1].scatter(means, y, color=colors, s=28, zorder=3)
    axes[1].set_yticks(y, closure_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Natural net runner gap closed (trials)")
    axes[1].set_title("B  No specific or dose-dependent closure", loc="left", fontweight="bold")
    for index, (mean, low, high) in enumerate(zip(means, lows, highs)):
        axes[1].text(high + 0.35, index, f"{mean:+.0f} [{low:+.0f}, {high:+.0f}]", va="center", fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.55)
        axis.set_axisbelow(True)
    figure.tight_layout(w_pad=2.0)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"runner_intervention_results.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def analyze(input_dir: str, output_dir: str, bootstrap: int, seed: int) -> dict:
    root = Path(input_dir)
    run_metadata = json.loads((root / "run_metadata.json").read_text())
    scenarios = run_metadata["scenarios"]
    qids, logits, metadata = _load(root, scenarios)
    choices = {scenario: values.argmax(axis=-1) for scenario, values in logits.items()}
    winner = np.asarray([
        "ABCD".index(metadata[("baseline_natural", qid)]["winner_letter"]) for qid in qids
    ])
    runner = np.asarray([
        "ABCD".index(metadata[("baseline_natural", qid)]["runner_letter"]) for qid in qids
    ])
    correct = np.asarray([
        "ABCD".index(metadata[("baseline_natural", qid)]["correct_answer"]) for qid in qids
    ])
    natural_game_runner = (choices["game_natural"] == runner).astype(float)
    natural_neutral_runner = (choices["neutral_natural"] == runner).astype(float)
    natural_gap = natural_game_runner - natural_neutral_runner
    natural_gap_count = float(natural_gap.sum())
    if natural_gap_count <= 0:
        raise RuntimeError("Natural run does not reproduce a positive net runner-switch gap")

    scenario_rows = []
    scenario_summary = {}
    canonical_tokens = {
        int(item[1])
        for entries in run_metadata["resolved_answer_tokens"].values()
        for item in entries
    }
    for scenario in scenarios:
        scenario_choice = choices[scenario]
        full_top = [metadata[(scenario, qid)]["full_vocab_top_token_id"] for qid in qids]
        row = {
            "scenario": scenario,
            "n": len(qids),
            "change_count": int((scenario_choice != winner).sum()),
            "change_rate": float((scenario_choice != winner).mean()),
            "runner_count": int((scenario_choice == runner).sum()),
            "runner_rate": float((scenario_choice == runner).mean()),
            "accuracy": float((scenario_choice == correct).mean()),
            "mean_ad_entropy_bits": float(_entropy(logits[scenario]).mean()),
            "noncanonical_full_vocab_top_count": int(sum(token not in canonical_tokens for token in full_top)),
        }
        scenario_rows.append(row)
        scenario_summary[scenario] = row

    primary = {}
    strengths = [float(value) for value in run_metadata["config"]["strengths"]]
    for index, strength in enumerate(strengths):
        suffix = f"x{strength:g}".replace(".", "p")
        game = f"game_remove_{suffix}"
        neutral = f"neutral_add_{suffix}"
        necessity_values = natural_game_runner - (choices[game] == runner)
        sufficiency_values = (choices[neutral] == runner) - natural_neutral_runner
        combined_values = necessity_values + sufficiency_values
        primary[suffix] = {
            "necessity": _effect(necessity_values, natural_gap, bootstrap, seed + 10 * index),
            "sufficiency": _effect(sufficiency_values, natural_gap, bootstrap, seed + 10 * index + 1),
            "combined_gap_closure": _effect(combined_values, natural_gap, bootstrap, seed + 10 * index + 2),
            "post_intervention_net_runner_gap_count": int(
                (choices[game] == runner).sum() - (choices[neutral] == runner).sum()
            ),
        }

    controls = {}
    control_names = (
        "rank3_control_x0p5",
        "orthogonal_control_x0p5",
        "rank3_control_x1",
        "orthogonal_control_x1",
        "early_control_x1",
    )
    for index, control in enumerate(control_names):
        game = f"game_remove_{control}"
        neutral = f"neutral_add_{control}"
        necessity_values = natural_game_runner - (choices[game] == runner)
        sufficiency_values = (choices[neutral] == runner) - natural_neutral_runner
        controls[control] = {
            "necessity": _effect(necessity_values, natural_gap, bootstrap, seed + 100 + index * 3),
            "sufficiency": _effect(sufficiency_values, natural_gap, bootstrap, seed + 101 + index * 3),
            "combined_gap_closure": _effect(
                necessity_values + sufficiency_values,
                natural_gap,
                bootstrap,
                seed + 102 + index * 3,
            ),
        }

    specificity = {}
    primary_necessity = natural_game_runner - (choices["game_remove_x0p5"] == runner)
    primary_sufficiency = (choices["neutral_add_x0p5"] == runner) - natural_neutral_runner
    for index, control in enumerate(("rank3_control", "orthogonal_control")):
        control_necessity = natural_game_runner - (
            choices[f"game_remove_{control}_x0p5"] == runner
        )
        control_sufficiency = (
            choices[f"neutral_add_{control}_x0p5"] == runner
        ) - natural_neutral_runner
        specificity[control] = {
            "necessity_targeted_minus_control": _paired_count(
                primary_necessity - control_necessity, bootstrap, seed + 300 + index * 3
            ),
            "sufficiency_targeted_minus_control": _paired_count(
                primary_sufficiency - control_sufficiency, bootstrap, seed + 301 + index * 3
            ),
            "combined_targeted_minus_control": _paired_count(
                primary_necessity
                + primary_sufficiency
                - control_necessity
                - control_sufficiency,
                bootstrap,
                seed + 302 + index * 3,
            ),
        }

    baseline_add_runner = (choices["baseline_add_x1"] == runner).astype(float)
    baseline_natural_runner = (choices["baseline_natural"] == runner).astype(float)
    baseline_control = {
        "additional_runner_choices": int((baseline_add_runner - baseline_natural_runner).sum()),
        "additional_changes": int(
            (choices["baseline_add_x1"] != winner).sum()
            - (choices["baseline_natural"] != winner).sum()
        ),
        "accuracy_change": float(
            (choices["baseline_add_x1"] == correct).mean()
            - (choices["baseline_natural"] == correct).mean()
        ),
    }

    calibration = {}
    for scenario in scenarios:
        if scenario.endswith("natural"):
            continue
        target_values = []
        achieved_values = []
        norms = []
        for qid in qids:
            row = metadata[(scenario, qid)]
            achieved_values.append(row["achieved_native_lens_contrast_change"])
            norms.append(row["residual_delta_l2"])
            target_values.append(row["requested_native_lens_contrast_change"])
        target_values = np.asarray(target_values)
        achieved_values = np.asarray(achieved_values)
        calibration[scenario] = {
            "mean_absolute_target_error": float(np.abs(achieved_values - target_values).mean()),
            "max_absolute_target_error": float(np.abs(achieved_values - target_values).max()),
            "mean_residual_delta_l2": float(np.mean(norms)),
        }

    summary = {
        "n_questions": len(qids),
        "natural": {
            "baseline_choice_disagreements_with_frozen_order": int(
                (choices["baseline_natural"] != winner).sum()
            ),
            "game_runner_count": int(natural_game_runner.sum()),
            "neutral_runner_count": int(natural_neutral_runner.sum()),
            "net_runner_gap_count": int(natural_gap_count),
            "net_runner_gap_rate": float(natural_gap.mean()),
            "net_runner_gap_count_ci": _bootstrap_sum(
                natural_gap, bootstrap, np.random.default_rng(seed)
            ),
        },
        "primary": primary,
        "controls": controls,
        "specificity": specificity,
        "baseline_control": baseline_control,
        "scenario_metrics": scenario_summary,
        "calibration": calibration,
        "estimand": (
            "Runner choice means final canonical A-D argmax equals the baseline-defined "
            "runner-up. Necessity is natural Game runner choice minus intervened Game "
            "runner choice; sufficiency is intervened Neutral minus natural Neutral."
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runner_intervention_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "runner_intervention_scenarios.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scenario_rows[0].keys())
        writer.writeheader()
        writer.writerows(scenario_rows)
    _plot(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze runner residual necessity and sufficiency")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()
