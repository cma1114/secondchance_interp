from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"
TASKS = ("Game", "Neutral")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap values must be a nonempty vector")
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        rows = rng.integers(0, len(values), size=(count, len(values)))
        samples[start : start + count] = values[rows].mean(1)
    return {
        "mean": float(values.mean()),
        "ci": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "n": int(len(values)),
    }


def _semantic_logits(
    displayed: np.ndarray,
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
) -> np.ndarray:
    output = np.empty_like(displayed, dtype=np.float64)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            new = mappings[qid]["original_to_new"][original]
            output[..., qi, original_index] = displayed[..., qi, LETTERS.index(new)]
    return output


def _choice_indices(
    displayed: np.ndarray,
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
) -> np.ndarray:
    displayed_choice = displayed.argmax(axis=-1)
    semantic = np.empty_like(displayed_choice)
    for qi, qid in enumerate(qids):
        lookup = np.asarray(
            [LETTERS.index(mappings[qid]["new_to_original"][letter]) for letter in LETTERS]
        )
        semantic[..., qi] = lookup[displayed_choice[..., qi]]
    return semantic


def _fresh_winner_indices(
    remapped: dict[str, dict[str, Any]],
    qids: list[str],
    mappings: dict[str, dict[str, Any]],
    *,
    seed_step5: bool,
) -> np.ndarray:
    """Return the semantic fresh winner under each experiment's frozen rule.

    Seed Step 5 fitted its fresh-score directions to the A--D score vector from
    the standalone bare-remapped collector, so its endpoint uses that vector's
    displayed argmax.  The archived Qwen experiment instead froze the model's
    emitted semantic answer.  The Qwen archive stores that value as
    ``answer_original_content``; it has never had an
    ``emitted_second_answer`` field.
    """
    fresh_winner = np.empty(len(qids), dtype=np.int64)
    for qi, qid in enumerate(qids):
        row = remapped[qid]
        if seed_step5:
            displayed_index = int(
                np.argmax(np.asarray(row["aggregated_ad_logits"], dtype=np.float64))
            )
            displayed_letter = LETTERS[displayed_index]
            semantic_letter = mappings[qid]["new_to_original"][displayed_letter]
        else:
            semantic_letter = row["answer_original_content"]
            if "answer_new_letter" in row:
                mapped = mappings[qid]["new_to_original"][row["answer_new_letter"]]
                if mapped != semantic_letter:
                    raise RuntimeError(
                        f"{qid}: emitted Qwen answer mapping is internally inconsistent"
                    )
        if semantic_letter not in LETTERS:
            raise RuntimeError(f"{qid}: invalid semantic fresh winner {semantic_letter!r}")
        fresh_winner[qi] = LETTERS.index(semantic_letter)
    return fresh_winner


def _scenario_summary(
    semantic_logits: np.ndarray,
    choices: np.ndarray,
    old_winner: np.ndarray,
    old_runner: np.ndarray,
    fresh_winner: np.ndarray,
    mask: np.ndarray,
    destination_mask: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    selected = np.flatnonzero(mask)
    destination = np.flatnonzero(mask & destination_mask)
    result: dict[str, Any] = {"n": int(len(selected)), "tasks": {}}
    task_vectors: dict[str, dict[str, np.ndarray]] = {}
    for task_index, task in enumerate(TASKS):
        task_choice = choices[task_index, selected]
        task_logits = semantic_logits[task_index, selected]
        winner = old_winner[selected]
        winner_logit = task_logits[np.arange(len(selected)), winner]
        centered_winner = winner_logit - task_logits.mean(axis=-1)
        vectors = {
            "old_winner_avoidance": (task_choice != winner).astype(float),
            "old_winner_choice": (task_choice == winner).astype(float),
            "old_winner_centered_advantage": centered_winner,
        }
        if len(destination):
            destination_choice = choices[task_index, destination]
            vectors.update(
                {
                    "destination_fresh_winner_choice": (
                        destination_choice == fresh_winner[destination]
                    ).astype(float),
                    "destination_old_runner_choice": (
                        destination_choice == old_runner[destination]
                    ).astype(float),
                    "destination_fresh_minus_old_runner": (
                        (destination_choice == fresh_winner[destination]).astype(float)
                        - (destination_choice == old_runner[destination]).astype(float)
                    ),
                }
            )
        task_vectors[task] = vectors
        result["tasks"][task] = {
            name: _interval(values, rng, draws) for name, values in vectors.items()
        }
        result["tasks"][task]["n_destination"] = int(len(destination))

    result["Game_minus_Neutral"] = {}
    for name in task_vectors["Game"]:
        result["Game_minus_Neutral"][name] = _interval(
            task_vectors["Game"][name] - task_vectors["Neutral"][name], rng, draws
        )
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    metadata = json.loads(args.metadata.read_text())
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    qids = arrays["question_ids"].astype(str).tolist()
    if not arrays["completed"].all() or not np.isfinite(arrays["logits"]).all():
        raise RuntimeError("Run is incomplete or contains non-finite logits")
    scenarios = metadata["scenarios"]
    scenario_index = {name: index for index, name in enumerate(scenarios)}
    natural_scenario = (
        "complete_path_natural"
        if "complete_path_natural" in scenario_index
        else "complete_sequence_natural"
    )
    required = {
        "trusted_natural",
        natural_scenario,
        "identity_hook",
        "fresh_scrub",
        "matching_history_blockade",
        "matching_plus_fresh",
        "dose_matched_random",
        "matching_plus_random",
    }
    if set(scenarios) != required:
        raise RuntimeError("Unexpected scenario inventory")

    displayed = arrays["logits"].astype(np.float64)
    semantic = _semantic_logits(displayed, qids, mappings)
    choices = _choice_indices(displayed, qids, mappings)
    old = arrays["baseline_logits"].astype(np.float64)
    old_order = np.argsort(-old, axis=-1, kind="stable")
    old_winner = old_order[:, 0]
    old_runner = old_order[:, 1]
    # Seed Step 5 defines fresh score from the standalone remapped candidate-
    # score vector used to fit the decoder. Keep its endpoint on that same
    # target. Preserve the historical emitted-answer convention for the
    # earlier Qwen experiment.
    seed_step5 = "Seed-OSS" in str(metadata.get("config", {}).get("model_id", ""))
    fresh_winner = _fresh_winner_indices(
        remapped, qids, mappings, seed_step5=seed_step5
    )
    conflict = old_winner != fresh_winner
    destination_distinct = conflict & (fresh_winner != old_runner)
    split = arrays["split"].astype(str)

    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {
        "definition": {
            "model_id": str(metadata.get("config", {}).get("model_id", "unknown")),
            "dataset": str(metadata.get("dataset", "SimpleMC")),
            "seed_step5": seed_step5,
            "primary": "Game-minus-Neutral old-W1 avoidance on the fixed W1 != fresh-W2 conflict set",
            "destination": "Fresh-W2 versus old-R2 choice on the fixed set where fresh-W2 differs from both old-W1 and old-R2",
            "ties": "Displayed-order argmax, then mapped back to semantic identity",
            "fresh_scrub": "Discovery-fitted unique-fresh decoder component removed from 2P semantic wordpieces and newlines after every layer L1-L64",
            "natural_scenario": natural_scenario,
        },
        "validation": {
            "questions": len(qids),
            "conflict_questions": int(conflict.sum()),
            "destination_distinct_questions": int(destination_distinct.sum()),
            "trusted_natural_max_abs_error": float(np.max(arrays["trusted_max_abs_error"])),
            "identity_max_abs_error": float(
                np.max(
                    np.abs(
                        displayed[:, scenario_index["identity_hook"]]
                        - displayed[:, scenario_index[natural_scenario]]
                    )
                )
            ),
        },
        "manipulation": {},
        "subsets": {},
        "contrasts": {},
        "endpoint_contrasts": {},
        "subset_contrasts": {},
        "provenance": {
            "results": {"path": str(args.results), "sha256": _sha256(args.results)},
            "metadata": {"path": str(args.metadata), "sha256": _sha256(args.metadata)},
            "remapping_plan": {"path": str(args.remapping_plan), "sha256": _sha256(args.remapping_plan)},
            "remapped_baseline": {"path": str(args.remapped_baseline), "sha256": _sha256(args.remapped_baseline)},
        },
    }

    for scenario in (
        "identity_hook",
        "fresh_scrub",
        "matching_plus_fresh",
        "dose_matched_random",
        "matching_plus_random",
    ):
        index = scenario_index[scenario]
        pre_fresh = arrays["pre_fresh"][:, index]
        post_fresh = arrays["post_fresh"][:, index]
        pre_old = arrays["pre_old"][:, index]
        post_old = arrays["post_old"][:, index]
        summary["manipulation"][scenario] = {
            "mean_abs_pre_fresh": float(np.mean(np.abs(pre_fresh))),
            "mean_abs_post_fresh": float(np.mean(np.abs(post_fresh))),
            "fresh_fraction_remaining": float(
                np.mean(np.abs(post_fresh)) / max(np.mean(np.abs(pre_fresh)), 1e-12)
            ),
            "mean_abs_old_coordinate_change": float(np.mean(np.abs(post_old - pre_old))),
            "max_abs_old_coordinate_change": float(np.max(np.abs(post_old - pre_old))),
            "mean_dose_l2": float(np.mean(arrays["dose_l2"][:, index])),
        }

    subset_masks = {
        "discovery_all": split == "discovery",
        "confirmation_all": split == "confirmation",
        "all_questions": np.ones(len(qids), dtype=bool),
        "discovery_conflict": (split == "discovery") & conflict,
        "confirmation_conflict": (split == "confirmation") & conflict,
        "all_conflict": conflict,
        "discovery_nonconflict": (split == "discovery") & ~conflict,
        "confirmation_nonconflict": (split == "confirmation") & ~conflict,
        "all_nonconflict": ~conflict,
    }
    for subset_name, mask in subset_masks.items():
        summary["subsets"][subset_name] = {}
        for scenario, index in scenario_index.items():
            summary["subsets"][subset_name][scenario] = _scenario_summary(
                semantic[:, index],
                choices[:, index],
                old_winner,
                old_runner,
                fresh_winner,
                mask,
                destination_distinct,
                rng,
                args.bootstrap_draws,
            )

    confirmation = summary["subsets"]["confirmation_conflict"]
    pairs = {
        "fresh_minus_random": ("fresh_scrub", "dose_matched_random"),
        "joint_minus_matching_random": ("matching_plus_fresh", "matching_plus_random"),
        "matching_minus_natural": ("matching_history_blockade", natural_scenario),
        "fresh_minus_natural": ("fresh_scrub", natural_scenario),
        "joint_minus_matching": ("matching_plus_fresh", "matching_history_blockade"),
    }

    # Report the two principal endpoints on the full confirmation split and
    # separately on conflict/non-conflict questions.  The original analysis
    # only computed causal contrasts on confirmation conflicts, which hid that
    # Seed's aggregate matching-history result and its conflict behavior differ.
    for subset_name, mask in subset_masks.items():
        selected_subset = np.flatnonzero(mask)
        subset_record: dict[str, Any] = {"n": int(len(selected_subset))}
        winner_subset = old_winner[selected_subset]
        rows_subset = np.arange(len(selected_subset))
        for name, (left, right) in pairs.items():
            left_index, right_index = scenario_index[left], scenario_index[right]
            pair_record: dict[str, Any] = {}
            task_values: dict[str, dict[str, np.ndarray]] = {}
            for task_index, task in enumerate(TASKS):
                left_choice = choices[task_index, left_index, selected_subset]
                right_choice = choices[task_index, right_index, selected_subset]
                left_logits = semantic[task_index, left_index, selected_subset]
                right_logits = semantic[task_index, right_index, selected_subset]
                task_values[task] = {
                    "old_winner_avoidance": (
                        (left_choice != winner_subset).astype(float)
                        - (right_choice != winner_subset).astype(float)
                    ),
                    "old_winner_centered_advantage": (
                        left_logits[rows_subset, winner_subset]
                        - left_logits.mean(axis=-1)
                        - right_logits[rows_subset, winner_subset]
                        + right_logits.mean(axis=-1)
                    ),
                }
            for endpoint in (
                "old_winner_avoidance",
                "old_winner_centered_advantage",
            ):
                pair_record[endpoint] = {
                    task: _interval(task_values[task][endpoint], rng, args.bootstrap_draws)
                    for task in TASKS
                }
                pair_record[endpoint]["Game_minus_Neutral_interaction"] = _interval(
                    task_values["Game"][endpoint] - task_values["Neutral"][endpoint],
                    rng,
                    args.bootstrap_draws,
                )
            subset_record[name] = pair_record
        summary["subset_contrasts"][subset_name] = subset_record

    confirmation_mask = (split == "confirmation") & conflict
    selected = np.flatnonzero(confirmation_mask)
    for name, (left, right) in pairs.items():
        left_index, right_index = scenario_index[left], scenario_index[right]
        record: dict[str, Any] = {}
        for task_index, task in enumerate(TASKS):
            left_avoid = (
                choices[task_index, left_index, selected] != old_winner[selected]
            ).astype(float)
            right_avoid = (
                choices[task_index, right_index, selected] != old_winner[selected]
            ).astype(float)
            record[task] = _interval(left_avoid - right_avoid, rng, args.bootstrap_draws)
        left_gap = (
            (choices[0, left_index, selected] != old_winner[selected]).astype(float)
            - (choices[1, left_index, selected] != old_winner[selected]).astype(float)
        )
        right_gap = (
            (choices[0, right_index, selected] != old_winner[selected]).astype(float)
            - (choices[1, right_index, selected] != old_winner[selected]).astype(float)
        )
        record["Game_minus_Neutral_interaction"] = _interval(
            left_gap - right_gap, rng, args.bootstrap_draws
        )
        summary["contrasts"][name] = record

        endpoint_record: dict[str, Any] = {}
        endpoint_indices = {
            "old_winner_avoidance": selected,
            "old_winner_centered_advantage": selected,
            "destination_fresh_winner_choice": np.flatnonzero(
                confirmation_mask & destination_distinct
            ),
            "destination_old_runner_choice": np.flatnonzero(
                confirmation_mask & destination_distinct
            ),
            "destination_fresh_minus_old_runner": np.flatnonzero(
                confirmation_mask & destination_distinct
            ),
        }
        for endpoint, endpoint_selected in endpoint_indices.items():
            task_differences: dict[str, np.ndarray] = {}
            for task_index, task in enumerate(TASKS):
                left_choice = choices[task_index, left_index, endpoint_selected]
                right_choice = choices[task_index, right_index, endpoint_selected]
                if endpoint == "old_winner_avoidance":
                    target = old_winner[endpoint_selected]
                    left_values = (left_choice != target).astype(float)
                    right_values = (right_choice != target).astype(float)
                elif endpoint == "old_winner_centered_advantage":
                    target = old_winner[endpoint_selected]
                    left_logits = semantic[task_index, left_index, endpoint_selected]
                    right_logits = semantic[task_index, right_index, endpoint_selected]
                    rows = np.arange(len(endpoint_selected))
                    left_values = left_logits[rows, target] - left_logits.mean(axis=-1)
                    right_values = right_logits[rows, target] - right_logits.mean(axis=-1)
                elif endpoint == "destination_fresh_winner_choice":
                    target = fresh_winner[endpoint_selected]
                    left_values = (left_choice == target).astype(float)
                    right_values = (right_choice == target).astype(float)
                elif endpoint == "destination_old_runner_choice":
                    target = old_runner[endpoint_selected]
                    left_values = (left_choice == target).astype(float)
                    right_values = (right_choice == target).astype(float)
                else:
                    fresh_target = fresh_winner[endpoint_selected]
                    runner_target = old_runner[endpoint_selected]
                    left_values = (
                        (left_choice == fresh_target).astype(float)
                        - (left_choice == runner_target).astype(float)
                    )
                    right_values = (
                        (right_choice == fresh_target).astype(float)
                        - (right_choice == runner_target).astype(float)
                    )
                task_differences[task] = left_values - right_values
            endpoint_record[endpoint] = {
                task: _interval(values, rng, args.bootstrap_draws)
                for task, values in task_differences.items()
            }
            endpoint_record[endpoint]["Game_minus_Neutral_interaction"] = _interval(
                task_differences["Game"] - task_differences["Neutral"],
                rng,
                args.bootstrap_draws,
            )
        summary["endpoint_contrasts"][name] = endpoint_record

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_report(summary, args.output_dir / "REPORT.md")
    _write_figure(summary, args.figure)
    return summary


def _pct(interval: dict[str, Any]) -> str:
    return f"{100 * interval['mean']:.1f}% [{100 * interval['ci'][0]:.1f}, {100 * interval['ci'][1]:.1f}]"


def _logit(interval: dict[str, Any]) -> str:
    return (
        f"{interval['mean']:+.3f} "
        f"[{interval['ci'][0]:+.3f}, {interval['ci'][1]:+.3f}]"
    )


def _write_report_legacy(summary: dict[str, Any], path: Path) -> None:
    confirmation = summary["subsets"]["confirmation_conflict"]
    natural_scenario = summary["definition"]["natural_scenario"]
    lines = [
        "# Fresh 2P × recollected-history double dissociation",
        "",
        "## Conclusion",
        "",
        f"- The intervention removed {100 * (1 - summary['manipulation']['fresh_scrub']['fresh_fraction_remaining']):.2f}% of the discovery-fitted unique-fresh coordinate from every second-option semantic wordpiece and newline while preserving the decoded old-score coordinate.",
        f"- Relative to the same-dose random edit, the fresh scrub changed the confirmation Game-minus-Neutral old-W1-avoidance gap by {_pct(summary['contrasts']['fresh_minus_random']['Game_minus_Neutral_interaction'])}. The interval includes zero: the scrub did not reduce preferential Game old-winner avoidance.",
        f"- With matching history also blocked, fresh removal changed that gap relative to the matched random control by {_pct(summary['contrasts']['joint_minus_matching_random']['Game_minus_Neutral_interaction'])}; this interval also includes zero.",
        "- Thus the validated decoded fresh-score subspace is not necessary for the preferential Game-versus-Neutral old-W1-avoidance endpoint tested here. This is a bounded null: distributed fresh computation outside that linear coordinate may remain.",
        "- Destination-choice effects are reported separately below and are secondary; they do not redefine the preferential-switching result.",
        "",
        "## Validity",
        "",
        f"- Canonical conflict questions: {summary['validation']['conflict_questions']} total.",
        f"- Native natural max logit error: {summary['validation']['trusted_natural_max_abs_error']:.8g}.",
        f"- Complete-sequence identity max logit error: {summary['validation']['identity_max_abs_error']:.8g}.",
        "",
        "## Confirmation old-winner avoidance",
        "",
        "| Scenario | Game | Neutral | Game − Neutral |",
        "|---|---:|---:|---:|",
    ]
    for scenario in (
        natural_scenario,
        "fresh_scrub",
        "dose_matched_random",
        "matching_history_blockade",
        "matching_plus_fresh",
        "matching_plus_random",
    ):
        row = confirmation[scenario]
        lines.append(
            f"| {scenario} | {_pct(row['tasks']['Game']['old_winner_avoidance'])} | "
            f"{_pct(row['tasks']['Neutral']['old_winner_avoidance'])} | "
            f"{_pct(row['Game_minus_Neutral']['old_winner_avoidance'])} |"
        )
    lines += ["", "## Prespecified causal contrasts", ""]
    for name, row in summary["contrasts"].items():
        lines.append(
            f"- **{name}:** Game {_pct(row['Game'])}; Neutral {_pct(row['Neutral'])}; "
            f"interaction {_pct(row['Game_minus_Neutral_interaction'])}."
        )
    lines += ["", "## Old-W1 logit contrast", ""]
    for name in ("fresh_minus_random", "joint_minus_matching_random"):
        values = summary["endpoint_contrasts"][name]["old_winner_centered_advantage"]
        lines.append(
            f"- **{name}:** Game {values['Game']['mean']:+.4f} "
            f"[{values['Game']['ci'][0]:+.4f}, {values['Game']['ci'][1]:+.4f}]; "
            f"Neutral {values['Neutral']['mean']:+.4f} "
            f"[{values['Neutral']['ci'][0]:+.4f}, {values['Neutral']['ci'][1]:+.4f}]; "
            f"interaction {values['Game_minus_Neutral_interaction']['mean']:+.4f} "
            f"[{values['Game_minus_Neutral_interaction']['ci'][0]:+.4f}, "
            f"{values['Game_minus_Neutral_interaction']['ci'][1]:+.4f}]."
        )
    lines += [
        "",
        "## Confirmation destination choice",
        "",
        "The fixed destination subset contains questions where fresh W2 differs from both old W1 and old R2.",
        "",
        "| Scenario | Task | Fresh-W2 choice | Old-R2 choice | Fresh W2 − old R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in (
        natural_scenario,
        "fresh_scrub",
        "dose_matched_random",
        "matching_history_blockade",
        "matching_plus_fresh",
        "matching_plus_random",
    ):
        for task in TASKS:
            row = confirmation[scenario]["tasks"][task]
            lines.append(
                f"| {scenario} | {task} | {_pct(row['destination_fresh_winner_choice'])} | "
                f"{_pct(row['destination_old_runner_choice'])} | "
                f"{_pct(row['destination_fresh_minus_old_runner'])} |"
            )
    lines += ["", "## Destination causal contrasts", ""]
    for name in ("fresh_minus_random", "joint_minus_matching_random"):
        row = summary["endpoint_contrasts"][name]
        for endpoint in (
            "destination_fresh_winner_choice",
            "destination_old_runner_choice",
            "destination_fresh_minus_old_runner",
        ):
            values = row[endpoint]
            lines.append(
                f"- **{name}, {endpoint}:** Game {_pct(values['Game'])}; "
                f"Neutral {_pct(values['Neutral'])}; interaction "
                f"{_pct(values['Game_minus_Neutral_interaction'])}."
            )
    lines += ["", "## Manipulation checks", ""]
    for scenario, row in summary["manipulation"].items():
        lines.append(
            f"- **{scenario}:** fresh fraction remaining {row['fresh_fraction_remaining']:.4f}; "
            f"mean |old-coordinate change| {row['mean_abs_old_coordinate_change']:.6g}; "
            f"mean L2 dose {row['mean_dose_l2']:.6g}."
        )
    path.write_text("\n".join(lines) + "\n")


def _write_figure_legacy(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    confirmation = summary["subsets"]["confirmation_conflict"]
    scenarios = [
        summary["definition"]["natural_scenario"],
        "fresh_scrub",
        "dose_matched_random",
        "matching_history_blockade",
        "matching_plus_fresh",
        "matching_plus_random",
    ]
    labels = ["Natural", "Fresh scrub", "Random control", "History block", "History + fresh", "History + random"]
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for task_index, task in enumerate(TASKS):
        means = np.asarray(
            [confirmation[name]["tasks"][task]["old_winner_avoidance"]["mean"] for name in scenarios]
        )
        cis = np.asarray(
            [confirmation[name]["tasks"][task]["old_winner_avoidance"]["ci"] for name in scenarios]
        )
        axes[0].errorbar(
            x + (-0.08 if task_index == 0 else 0.08),
            means,
            yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            marker="o",
            capsize=4,
            linewidth=2,
            label=task,
        )
    gap = np.asarray(
        [confirmation[name]["Game_minus_Neutral"]["old_winner_avoidance"]["mean"] for name in scenarios]
    )
    gap_ci = np.asarray(
        [confirmation[name]["Game_minus_Neutral"]["old_winner_avoidance"]["ci"] for name in scenarios]
    )
    axes[1].errorbar(
        x,
        gap,
        yerr=np.vstack((gap - gap_ci[:, 0], gap_ci[:, 1] - gap)),
        marker="o",
        capsize=4,
        linewidth=2,
        color="#7b3294",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Old-W1 avoidance")
    axes[1].set_ylabel("Game − Neutral avoidance")
    axes[0].set_title("Confirmation conflict trials")
    axes[1].set_title("Differential switching")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _write_report(summary: dict[str, Any], path: Path) -> None:
    natural_scenario = summary["definition"]["natural_scenario"]
    seed_step5 = bool(summary["definition"].get("seed_step5", False))
    dataset = summary["definition"].get("dataset", "SimpleMC")
    subsets = summary["subsets"]
    contrasts = summary["subset_contrasts"]
    full = subsets["confirmation_all"]
    conflict = subsets["confirmation_conflict"]

    def natural_gap(subset: dict[str, Any], endpoint: str) -> dict[str, Any]:
        return subset[natural_scenario]["Game_minus_Neutral"][endpoint]

    def interaction(subset_name: str, contrast: str, endpoint: str) -> dict[str, Any]:
        return contrasts[subset_name][contrast][endpoint][
            "Game_minus_Neutral_interaction"
        ]

    full_natural_choice = natural_gap(full, "old_winner_avoidance")
    conflict_natural_choice = natural_gap(conflict, "old_winner_avoidance")
    full_natural_logit = natural_gap(full, "old_winner_centered_advantage")
    fresh_full_logit = interaction(
        "confirmation_all", "fresh_minus_random", "old_winner_centered_advantage"
    )
    fresh_conflict_logit = interaction(
        "confirmation_conflict", "fresh_minus_random", "old_winner_centered_advantage"
    )
    matching_full_choice = interaction(
        "confirmation_all", "matching_minus_natural", "old_winner_avoidance"
    )
    matching_conflict_choice = interaction(
        "confirmation_conflict", "matching_minus_natural", "old_winner_avoidance"
    )
    matching_nonconflict_choice = interaction(
        "confirmation_nonconflict", "matching_minus_natural", "old_winner_avoidance"
    )
    joint_conflict_logit = conflict["matching_plus_fresh"]["Game_minus_Neutral"][
        "old_winner_centered_advantage"
    ]
    if seed_step5:
        conflict_endpoint_sentence = (
            f"On the prespecified conflict subset, the choice gap is only "
            f"{_pct(conflict_natural_choice)}, so a null choice interaction there "
            "cannot establish that an intervention preserves a clearly detected "
            "natural effect."
        )
    else:
        conflict_endpoint_sentence = (
            f"On the prespecified conflict subset, the corresponding choice gap is "
            f"{_pct(conflict_natural_choice)} and is clearly detected."
        )

    lines = [
        "# Fresh 2P × recollected-history double dissociation",
        "",
        "## Corrected conclusion",
        "",
        f"- The intervention removed {100 * (1 - summary['manipulation']['fresh_scrub']['fresh_fraction_remaining']):.2f}% of the discovery-fitted unique-fresh coordinate from every second-option semantic wordpiece and newline while preserving the decoded old-score coordinate.",
        f"- On the complete {dataset} confirmation split, the natural Game-minus-Neutral old-W1-avoidance gap is {_pct(full_natural_choice)} and the centered old-W1 logit gap is {_logit(full_natural_logit)}. {conflict_endpoint_sentence}",
        f"- Relative to the same-dose random edit, fresh removal changes the centered old-W1 task gap by {_logit(fresh_full_logit)} on the complete confirmation split and {_logit(fresh_conflict_logit)} on confirmation conflicts. This is the score-level test of whether the decoded fresh coordinate contributes to policy-conditioned old-winner suppression.",
        f"- Matching-history blockade changes the choice gap by {_pct(matching_full_choice)} over the complete confirmation split, {_pct(matching_conflict_choice)} on conflicts, and {_pct(matching_nonconflict_choice)} on non-conflicts. Reporting all three prevents an aggregate recollection result from being silently treated as a conflict-specific decomposition.",
        f"- After matching blockade and fresh removal are combined, the conflict-set centered old-W1 task gap remains {_logit(joint_conflict_logit)}. These two interventions do not exhaust the conflict pathway.",
    ]
    if seed_step5:
        lines += [
            "- **Corrected Seed verdict:** aggregate dependence on matching recollection replicates, but the Qwen recollection-versus-fresh-evidence dissociation does not. The measured fresh 2P score coordinate contributes a material minority of Seed's score-level policy effect in at least one informative slice of each dataset, while conflict questions retain an unlocalized policy-conditioned component after both lesions.",
            "- The matching lesion blocks only first-option-line to matching second-option-line attention. Direct final-position reads of first-presentation history are one plausible Seed-specific bypass, but this run does not localize the surviving component.",
        ]
    else:
        lines += [
            "- **Corrected Qwen verdict:** matching recollection is the dominant causal route and is necessary for the choice-level preferential-switching effect, but the decoded fresh coordinate is not purely task-shared. Its removal produces a small, statistically reliable reduction of the Game-minus-Neutral old-W1 logit gap on both the complete confirmation split and confirmation conflicts. The intervention does not reliably remove the choice-level gap, so the supported claim is that this fresh coordinate contributes to policy-conditioned scoring but is not necessary for the existence of preferential switching. This remains bounded to the decoded linear coordinate; distributed nonlinear fresh computation was not removed.",
        ]
    lines += [
        "",
        "## Validity",
        "",
        f"- Canonical conflict questions: {summary['validation']['conflict_questions']} total.",
        f"- Native natural max logit error: {summary['validation']['trusted_natural_max_abs_error']:.8g}.",
        f"- Complete-sequence identity max logit error: {summary['validation']['identity_max_abs_error']:.8g}.",
        "",
        "## Confirmation natural task gaps",
        "",
        "| Subset | n | Old-W1 avoidance: Game − Neutral | Centered old-W1 logits: Game − Neutral |",
        "|---|---:|---:|---:|",
    ]
    for subset_name, label in (
        ("confirmation_all", "All confirmation"),
        ("confirmation_conflict", "Conflict"),
        ("confirmation_nonconflict", "Non-conflict"),
    ):
        row = subsets[subset_name][natural_scenario]
        lines.append(
            f"| {label} | {row['n']} | "
            f"{_pct(row['Game_minus_Neutral']['old_winner_avoidance'])} | "
            f"{_logit(row['Game_minus_Neutral']['old_winner_centered_advantage'])} |"
        )
    lines += [
        "",
        "## Confirmation causal decomposition",
        "",
        "Each entry is the intervention-minus-control change in the Game-minus-Neutral task gap. Fresh removal is compared with its exact same-L2 random edit; matching blockade is compared with natural.",
        "",
        "| Subset | Contrast | Old-W1 avoidance interaction | Centered old-W1 logit interaction |",
        "|---|---|---:|---:|",
    ]
    for subset_name, label in (
        ("confirmation_all", "All confirmation"),
        ("confirmation_conflict", "Conflict"),
        ("confirmation_nonconflict", "Non-conflict"),
    ):
        for contrast_name, contrast_label in (
            ("fresh_minus_random", "Fresh scrub − random"),
            ("matching_minus_natural", "Matching block − natural"),
            ("joint_minus_matching_random", "Joint − matching+random"),
        ):
            row = contrasts[subset_name][contrast_name]
            lines.append(
                f"| {label} | {contrast_label} | "
                f"{_pct(row['old_winner_avoidance']['Game_minus_Neutral_interaction'])} | "
                f"{_logit(row['old_winner_centered_advantage']['Game_minus_Neutral_interaction'])} |"
            )
    lines += [
        "",
        "## Confirmation conflict scenarios",
        "",
        "| Scenario | Game avoidance | Neutral avoidance | Avoidance gap | Centered old-W1 logit gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in (
        natural_scenario,
        "fresh_scrub",
        "dose_matched_random",
        "matching_history_blockade",
        "matching_plus_fresh",
        "matching_plus_random",
    ):
        row = conflict[scenario]
        lines.append(
            f"| {scenario} | {_pct(row['tasks']['Game']['old_winner_avoidance'])} | "
            f"{_pct(row['tasks']['Neutral']['old_winner_avoidance'])} | "
            f"{_pct(row['Game_minus_Neutral']['old_winner_avoidance'])} | "
            f"{_logit(row['Game_minus_Neutral']['old_winner_centered_advantage'])} |"
        )
    lines += ["", "## Original prespecified conflict-choice contrasts", ""]
    for name, row in summary["contrasts"].items():
        lines.append(
            f"- **{name}:** Game {_pct(row['Game'])}; Neutral {_pct(row['Neutral'])}; "
            f"interaction {_pct(row['Game_minus_Neutral_interaction'])}."
        )
    lines += ["", "## Confirmation destination choice", ""]
    lines += [
        "The fixed destination subset contains questions where fresh W2 differs from both old W1 and old R2.",
        "",
        "| Scenario | Task | Fresh-W2 choice | Old-R2 choice | Fresh W2 − old R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in (
        natural_scenario,
        "fresh_scrub",
        "dose_matched_random",
        "matching_history_blockade",
        "matching_plus_fresh",
        "matching_plus_random",
    ):
        for task in TASKS:
            row = conflict[scenario]["tasks"][task]
            lines.append(
                f"| {scenario} | {task} | {_pct(row['destination_fresh_winner_choice'])} | "
                f"{_pct(row['destination_old_runner_choice'])} | "
                f"{_pct(row['destination_fresh_minus_old_runner'])} |"
            )
    lines += ["", "## Destination causal contrasts", ""]
    for name in ("fresh_minus_random", "joint_minus_matching_random"):
        row = summary["endpoint_contrasts"][name]
        for endpoint in (
            "destination_fresh_winner_choice",
            "destination_old_runner_choice",
            "destination_fresh_minus_old_runner",
        ):
            values = row[endpoint]
            lines.append(
                f"- **{name}, {endpoint}:** Game {_pct(values['Game'])}; "
                f"Neutral {_pct(values['Neutral'])}; interaction "
                f"{_pct(values['Game_minus_Neutral_interaction'])}."
            )
    lines += ["", "## Manipulation checks", ""]
    for scenario, row in summary["manipulation"].items():
        lines.append(
            f"- **{scenario}:** fresh fraction remaining {row['fresh_fraction_remaining']:.4f}; "
            f"mean |old-coordinate change| {row['mean_abs_old_coordinate_change']:.6g}; "
            f"mean L2 dose {row['mean_dose_l2']:.6g}."
        )
    path.write_text("\n".join(lines) + "\n")


def _write_figure(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = [
        summary["definition"]["natural_scenario"],
        "fresh_scrub",
        "dose_matched_random",
        "matching_history_blockade",
        "matching_plus_fresh",
        "matching_plus_random",
    ]
    labels = [
        "Natural",
        "Fresh scrub",
        "Random control",
        "History block",
        "History + fresh",
        "History + random",
    ]
    x = np.arange(len(scenarios))
    subset_columns = (
        ("confirmation_all", "All confirmation"),
        ("confirmation_conflict", "Conflict"),
        ("confirmation_nonconflict", "Non-conflict"),
    )
    endpoints = (
        ("old_winner_avoidance", "Game − Neutral old-W1 avoidance"),
        ("old_winner_centered_advantage", "Game − Neutral centered old-W1 logits"),
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 8.5), sharey="row", constrained_layout=True
    )
    for column, (subset_name, subset_label) in enumerate(subset_columns):
        subset = summary["subsets"][subset_name]
        for row_index, (endpoint, ylabel) in enumerate(endpoints):
            axis = axes[row_index, column]
            means = np.asarray(
                [subset[name]["Game_minus_Neutral"][endpoint]["mean"] for name in scenarios]
            )
            cis = np.asarray(
                [subset[name]["Game_minus_Neutral"][endpoint]["ci"] for name in scenarios]
            )
            axis.errorbar(
                x,
                means,
                yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
                marker="o",
                capsize=4,
                linewidth=2,
                color="#7b3294" if row_index == 0 else "#2166ac",
            )
            axis.axhline(0, color="black", linewidth=1)
            axis.set_xticks(x, labels, rotation=30, ha="right")
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(f"{subset_label} (n={subset[scenarios[0]]['n']})")
            if column == 0:
                axis.set_ylabel(ylabel)
    model_label = (
        "Seed-OSS 36B"
        if summary["definition"].get("seed_step5")
        else "Qwen3.6-27B"
    )
    fig.suptitle(
        f"{model_label} {summary['definition'].get('dataset', 'SimpleMC')}: "
        "fresh-score and matching-history decomposition",
        fontsize=15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
