from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("Game", "Neutral")
SOURCE_GROUPS = {
    "1P option lines": (3, 4, 5, 6),
    "feedback suffix": (11, 12, 13, 14, 15, 16, 17),
    "2P answer instruction": (18,),
    "2P question stem": (19,),
    "2P option lines": (20, 21, 22, 23),
    "2P choice cue/query": (24,),
    "final assistant prefix": (25,),
}


def _correlation(values: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(values, dtype=np.float64).reshape(-1)
    right = np.asarray(target, dtype=np.float64).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else 0.0


def _question_center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _bootstrap_correlation(
    values: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> tuple[float, float, float]:
    point = _correlation(values, target)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float32)
    for draw in range(draws):
        rows = rng.integers(0, values.shape[0], size=values.shape[0])
        samples[draw] = _correlation(values[rows], target[rows])
    low, high = np.quantile(samples, (0.025, 0.975))
    return point, float(low), float(high)


def analyze(args: argparse.Namespace) -> None:
    source = np.load(args.source_writes, allow_pickle=False)
    scores = np.load(args.score_projections, allow_pickle=False)
    components = np.load(args.components, allow_pickle=False)

    if not np.array_equal(source["question_ids"], scores["question_ids"]):
        raise RuntimeError("Source and score question order differs")
    if not np.array_equal(source["question_ids"], components["question_ids"]):
        raise RuntimeError("Source and component question order differs")
    if not np.array_equal(source["discovery"], scores["discovery"]):
        raise RuntimeError("Frozen split differs between inputs")
    if tuple(str(value) for value in source["source_names"])[:7] != (
        "system_and_header",
        "first_task_instruction",
        "first_question_stem",
        "first_R1_line",
        "first_R2_line",
        "first_R3_line",
        "first_R4_line",
    ):
        raise RuntimeError("Source labels changed")

    discovery = source["discovery"].astype(bool)
    confirmation = ~discovery
    if (int(discovery.sum()), int(confirmation.sum())) != (251, 249):
        raise RuntimeError("Expected the frozen 251/249 split")
    layers = source["ordinary_layers"].astype(int)
    fresh_target = _question_center(scores["fresh_unique"].astype(np.float32))
    target_names = [str(value) for value in components["target_names"]]
    component_names = [str(value) for value in components["component_names"]]
    fresh_component_index = target_names.index("fresh_unique")

    # decoded_delta is condition x ordinary-layer x question x source x target x output.
    raw_source = source["decoded_delta"][:, :, :, :, 1].astype(np.float32)
    grouped = np.empty(
        (len(CONDITIONS), len(SOURCE_GROUPS), len(layers), len(discovery), 4),
        dtype=np.float32,
    )
    group_names = list(SOURCE_GROUPS)
    for group_index, positions in enumerate(SOURCE_GROUPS.values()):
        grouped[:, group_index] = np.take(raw_source, positions, axis=3).sum(axis=3)
    grouped = _question_center(grouped)

    result: dict[str, object] = {
        "question": "Which cached exact final-query source writes carry fresh 2P score?",
        "evidence_label": (
            "Held-out source-write decoding; source groups are sums of individually "
            "decoded source contributions and are not causal group interventions."
        ),
        "questions": 500,
        "discovery": 251,
        "confirmation": 249,
        "ordinary_layers": layers.tolist(),
        "source_groups": group_names,
        "source_trajectories": {},
        "component_trajectories": {},
    }

    for condition_index, condition in enumerate(CONDITIONS):
        condition_rows: dict[str, object] = {}
        for group_index, group_name in enumerate(group_names):
            discovery_correlations = np.asarray(
                [
                    _correlation(grouped[condition_index, group_index, layer, discovery], fresh_target[discovery])
                    for layer in range(len(layers))
                ],
                dtype=np.float32,
            )
            confirmation_correlations = np.asarray(
                [
                    _correlation(grouped[condition_index, group_index, layer, confirmation], fresh_target[confirmation])
                    for layer in range(len(layers))
                ],
                dtype=np.float32,
            )
            selected_index = int(np.argmax(discovery_correlations))
            selected = _bootstrap_correlation(
                grouped[condition_index, group_index, selected_index, confirmation],
                fresh_target[confirmation],
                seed=args.seed + 100 * condition_index + group_index,
                draws=args.bootstrap_draws,
            )
            condition_rows[group_name] = {
                "discovery_correlation": discovery_correlations.tolist(),
                "confirmation_correlation": confirmation_correlations.tolist(),
                "discovery_selected_layer": int(layers[selected_index]),
                "discovery_selected_correlation": float(discovery_correlations[selected_index]),
                "heldout_at_selected_layer": {
                    "mean": selected[0],
                    "ci_low": selected[1],
                    "ci_high": selected[2],
                },
            }
        result["source_trajectories"][condition] = condition_rows

    # Complete final-position component writes are already decoded for every layer.
    component_delta = components["decoded_delta"][:, :, :, :, fresh_component_index].astype(np.float32)
    component_delta = _question_center(component_delta)
    for condition_index, condition in enumerate(CONDITIONS):
        condition_rows = {}
        for component_index, component_name in enumerate(component_names):
            discovery_correlations = []
            confirmation_correlations = []
            for layer in range(64):
                values = component_delta[condition_index, :, layer, component_index]
                discovery_correlations.append(_correlation(values[discovery], fresh_target[discovery]))
                confirmation_correlations.append(_correlation(values[confirmation], fresh_target[confirmation]))
            selected_layer = int(np.argmax(discovery_correlations))
            heldout = _bootstrap_correlation(
                component_delta[condition_index, confirmation, selected_layer, component_index],
                fresh_target[confirmation],
                seed=args.seed + 1000 + 100 * condition_index + component_index,
                draws=args.bootstrap_draws,
            )
            condition_rows[component_name] = {
                "discovery_correlation": discovery_correlations,
                "confirmation_correlation": confirmation_correlations,
                "discovery_selected_layer": selected_layer + 1,
                "discovery_selected_correlation": discovery_correlations[selected_layer],
                "heldout_at_selected_layer": {
                    "mean": heldout[0],
                    "ci_low": heldout[1],
                    "ci_high": heldout[2],
                },
            }
        result["component_trajectories"][condition] = condition_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    colors = {
        "2P question stem": "#9467bd",
        "2P option lines": "#1f77b4",
        "2P choice cue/query": "#ff7f0e",
        "1P option lines": "#7f7f7f",
        "feedback suffix": "#d62728",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True, sharey=True)
    for condition_index, condition in enumerate(CONDITIONS):
        axis = axes[condition_index]
        for group_name, color in colors.items():
            row = result["source_trajectories"][condition][group_name]
            axis.plot(layers, row["confirmation_correlation"], label=group_name, color=color, linewidth=2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(condition)
        axis.set_xlabel("Layer")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out correlation with unique fresh 2P score")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("Which final-attention source writes carry fresh 2P evidence?")
    figure.tight_layout()
    figure.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps({
        "output": str(args.output_dir / "summary.json"),
        "figure": str(args.figure),
        "discovery": int(discovery.sum()),
        "confirmation": int(confirmation.sum()),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("outputs/prompt_variant_tests/qwen36_27b_simplemc_option_remapping/second_presentation_residual_workspace/final_position_program")
    parser.add_argument("--source-writes", type=Path, default=root / "source_writes/final_query_source_writes.npz")
    parser.add_argument("--score-projections", type=Path, default=root / "score/final_score_projections.npz")
    parser.add_argument("--components", type=Path, default=root / "components/final_component_trajectory.npz")
    parser.add_argument("--output-dir", type=Path, default=root / "fresh_source_map")
    parser.add_argument("--figure", type=Path, default=Path("figures/qwen36_final_fresh_score_sources.png"))
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
