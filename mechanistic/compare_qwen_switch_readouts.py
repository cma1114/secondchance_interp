from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TASKS = ("Game", "Neutral")
READOUTS = (
    ("jlens_scores", "Jacobian lens", "#6f4aa8"),
    ("logit_lens_scores", "Standard logit lens", "#d06435"),
)
SELECTED_LAYERS = (32, 40, 44, 48, 50, 52, 54, 56, 60, 64)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _bootstrap(
    values: np.ndarray,
    strata: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = values.mean(axis=0)
    sampled_means = np.empty((draws, values.shape[1]), dtype=np.float32)
    unique = np.unique(strata)
    for draw in range(draws):
        sample = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(strata == stratum),
                    size=int(np.sum(strata == stratum)),
                    replace=True,
                )
                for stratum in unique
            ]
        )
        sampled_means[draw] = values[sample].mean(axis=0)
    low, high = np.quantile(sampled_means, (0.025, 0.975), axis=0)
    return point, low, high


def _first_sustained_positive(low: np.ndarray, run: int = 3) -> int | None:
    positive = low > 0
    for start in range(len(low) - run + 1):
        if np.all(positive[start : start + run]):
            return start + 1
    return None


def _controlled_rank_scores(
    raw_scores: np.ndarray,
    direct: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    scores = raw_scores.astype(np.float32).copy()
    scores[:, :, -1] = direct
    centered = scores - scores.mean(axis=-1, keepdims=True)
    controlled = centered - centered.mean(axis=1, keepdims=True)
    return np.take_along_axis(
        controlled,
        np.broadcast_to(order[None, :, None, :], controlled.shape),
        axis=-1,
    )


def analyze(
    specs_path: Path,
    output_dir: Path,
    figure_path: Path,
    draws: int,
    seed: int,
) -> None:
    specs = json.loads(specs_path.read_text())["datasets"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    layers = np.arange(1, 65)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0), sharex=True, sharey=True)
    summary: dict[str, Any] = {
        "analysis": "qwen_switch_jlens_vs_standard_logit_lens",
        "evidence_class": "activation/readout; switch panels are descriptive outcome postselection",
        "displayed_letter_control": (
            "Within each condition and layer, subtract the across-question centered score "
            "for each displayed A-D letter before aligning candidates by frozen 1P rank."
        ),
        "uncertainty": (
            "First-presentation-winner-letter-stratified question bootstrap, 95% percentile CI."
        ),
        "bootstrap_draws": draws,
        "datasets": [],
    }
    for dataset_index, spec in enumerate(specs):
        data = _load(Path(spec["results"]))
        direct = data["direct_logits"].astype(np.float32)
        order = data["rank_order"].astype(np.int64)
        first_winner = order[:, 0]
        switch = np.argmax(direct, axis=-1) != first_winner[None]
        dataset_record: dict[str, Any] = {
            "dataset": spec["name"],
            "n_questions": int(len(order)),
            "conditions": {},
        }
        controlled_by_readout = {
            key: _controlled_rank_scores(data[key], direct, order)
            for key, _, _ in READOUTS
        }
        for task_index, task in enumerate(TASKS):
            axis = axes[dataset_index, task_index]
            mask = switch[task_index]
            condition_record: dict[str, Any] = {
                "n_switch": int(mask.sum()),
                "readouts": {},
            }
            for readout_index, (key, label, color) in enumerate(READOUTS):
                aligned = controlled_by_readout[key][task_index, mask]
                margin = aligned[:, :, 1] - aligned[:, :, 0]
                point, low, high = _bootstrap(
                    margin,
                    first_winner[mask],
                    draws,
                    np.random.default_rng(
                        seed + 10000 * dataset_index + 1000 * task_index + 100 * readout_index
                    ),
                )
                axis.plot(layers, point, color=color, linewidth=2.35, label=label)
                axis.fill_between(layers, low, high, color=color, alpha=0.17, linewidth=0)
                first_positive = int(np.flatnonzero(low > 0)[0] + 1) if np.any(low > 0) else None
                condition_record["readouts"][key] = {
                    "label": label,
                    "mean": point.tolist(),
                    "ci_low": low.tolist(),
                    "ci_high": high.tolist(),
                    "first_positive_ci_layer": first_positive,
                    "first_three_layer_sustained_positive_ci_layer": _first_sustained_positive(low),
                    "selected_layers": {
                        str(layer): {
                            "mean": float(point[layer - 1]),
                            "ci_low": float(low[layer - 1]),
                            "ci_high": float(high[layer - 1]),
                        }
                        for layer in SELECTED_LAYERS
                    },
                }
            axis.axhline(0, color="#555555", linewidth=0.9)
            axis.set_title(f"{spec['display_name']} — {task} switch trials (n={int(mask.sum())})", fontsize=11.2, weight="bold")
            axis.set_xlim(0.5, 64.5)
            axis.set_xticks(np.arange(5, 65, 5))
            axis.set_xticks(np.arange(1, 65), minor=True)
            axis.tick_params(axis="x", which="minor", length=1.8, color="#aaaaaa")
            axis.grid(axis="x", which="major", color="#e5e5e5", linewidth=0.45)
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
            dataset_record["conditions"][task.lower()] = condition_record
        summary["datasets"].append(dataset_record)
    for axis in axes[-1]:
        axis.set_xlabel("Final-decision readout layer")
    for axis in axes[:, 0]:
        axis.set_ylabel("Displayed-letter-controlled R2 − R1 score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        "Qwen3.6-27B switch trials: matched Jacobian-lens and standard-logit-lens timing",
        y=1.02,
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Positive values mean the eventual alternative R2 is more output-readable than the first-presentation winner R1; shading is a stratified 95% bootstrap CI.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.91, hspace=0.28, wspace=0.10)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary["figure"] = str(figure_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Qwen switch-trial readout comparison",
        "",
        "This is the matched-readout check motivated by the apparent Seed-OSS/Qwen difference. It applies both Qwen's original Jacobian lens and a conventional logit lens to the same cached final-decision residuals, same questions, same condition labels, same frozen first-presentation ranks, and same switch subsets. No transformer forward was rerun for this comparison.",
        "",
        "The plotted endpoint is the displayed-letter-controlled R2-minus-R1 score. Positive means the candidate ranked second on the first presentation is more readable than the original winner. The control first subtracts the stable A/B/C/D geometry within each condition and layer; without it, Qwen's output rows create a misleading early R1 pattern. Outcome-conditioned switch panels are descriptive activation evidence, not a causal intervention.",
        "",
        f"[Canonical comparison figure](../../../../../{figure_path})",
        "",
        "## Results",
        "",
    ]
    for dataset in summary["datasets"]:
        lines.extend([f"### {dataset['dataset']}", ""])
        for task in ("game", "neutral"):
            record = dataset["conditions"][task]
            j = record["readouts"]["jlens_scores"]
            s = record["readouts"]["logit_lens_scores"]
            lines.append(
                f"- **{task.title()} switch trials (n={record['n_switch']}):** Jacobian-lens R2−R1 has its first three-layer sustained positive CI at L{j['first_three_layer_sustained_positive_ci_layer']}; the standard logit lens does so at L{s['first_three_layer_sustained_positive_ci_layer']}. At L52 the margins are {j['selected_layers']['52']['mean']:+.3f} `[{j['selected_layers']['52']['ci_low']:+.3f}, {j['selected_layers']['52']['ci_high']:+.3f}]` and {s['selected_layers']['52']['mean']:+.3f} `[{s['selected_layers']['52']['ci_low']:+.3f}, {s['selected_layers']['52']['ci_high']:+.3f}]`, respectively."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The matched conventional readout does not turn Qwen into the Seed pattern. After displayed-letter geometry is removed, Qwen remains approximately unseparated until late, and R2 is already above R1 when a reliable switch-trial ordering becomes readable. The original Qwen conclusion was therefore not an artifact of using a Jacobian lens while Seed used a standard logit lens.",
            "",
            "The descriptive cross-model difference remains: Seed often exposes an R1-leading output-readable state before R2 overtakes it, whereas Qwen's output-readable switch ordering appears with R2 already ahead. This does not establish that Qwen lacks earlier non-output-aligned answer information; Qwen's held-out prospective decoders recover question-specific and policy-adjusted information earlier. The evidence supports a difference in when the intermediate computation becomes aligned with each model's own output vocabulary, not a claim that either model implements a literally serial symbolic algorithm.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    analyze(args.specs, args.output_dir, args.figure, args.draws, args.seed)


if __name__ == "__main__":
    main()
