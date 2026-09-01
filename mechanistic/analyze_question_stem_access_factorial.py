from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .semantic_mapping import (
    align_displayed_logits_to_semantic,
    displayed_argmax_to_semantic_indices,
)


TASKS = ("Game", "Neutral")
SCENARIOS = (
    "trusted_natural",
    "identity_monitor",
    "block_first_stem",
    "block_second_stem",
    "block_both_stems",
)
GROUPS = ("semantic_wordpieces", "option_newlines")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _interval(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> dict[str, Any]:
    selected = np.asarray(values, dtype=np.float64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError(f"Expected a nonempty question vector, got {selected.shape}")
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 400):
        count = min(400, draws - start)
        rows = rng.integers(0, len(selected), size=(count, len(selected)))
        samples[start : start + count] = selected[rows].mean(1)
    return {
        "n": int(len(selected)),
        "mean": float(selected.mean()),
        "ci": [float(x) for x in np.quantile(samples, (0.025, 0.975))],
    }


def _ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if num.shape != den.shape or num.ndim != 1 or len(num) == 0:
        raise ValueError("Ratio inputs must be same-length question vectors")
    point_den = float(den.sum())
    if abs(point_den) < 1e-10:
        return {"n": int(len(num)), "ratio": float("nan"), "ci": [float("nan")] * 2}
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 400):
        count = min(400, draws - start)
        rows = rng.integers(0, len(num), size=(count, len(num)))
        num_sum = num[rows].sum(1)
        den_sum = den[rows].sum(1)
        samples[start : start + count] = np.divide(
            num_sum,
            den_sum,
            out=np.full(count, np.nan),
            where=np.abs(den_sum) > 1e-10,
        )
    finite = samples[np.isfinite(samples)]
    return {
        "n": int(len(num)),
        "ratio": float(num.sum() / point_den),
        "ci": [float(x) for x in np.quantile(finite, (0.025, 0.975))],
    }


def _rank_target(values: np.ndarray, rank_indices: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for qi in range(len(values)):
        output[qi] = values[qi, rank_indices[qi]]
    return output


def _scenario_vectors(
    semantic_logits: np.ndarray,
    semantic_choices: np.ndarray,
    old_winner: np.ndarray,
    fresh_winner: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.arange(len(old_winner))
    w1 = semantic_logits[rows, old_winner]
    w2 = semantic_logits[rows, fresh_winner]
    return {
        "old_winner_avoidance": (semantic_choices != old_winner).astype(float),
        "old_winner_choice": (semantic_choices == old_winner).astype(float),
        "fresh_winner_choice": (semantic_choices == fresh_winner).astype(float),
        "old_winner_centered_advantage": w1 - semantic_logits.mean(-1),
        "old_winner_minus_fresh_winner_logit": w1 - w2,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    arrays = _load(args.results)
    metadata = json.loads(args.metadata.read_text())
    if arrays["scenario_ids"].astype(str).tolist() != list(SCENARIOS):
        raise RuntimeError("Scenario inventory differs from the frozen plan")
    if len(arrays["question_ids"]) != 500 or not arrays["completed"].all():
        raise RuntimeError("A complete 500-question checkpoint is required")
    for key in ("baseline_logits", "logits"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"Non-finite values in {key}")
    if not np.isfinite(arrays["fresh_coordinates"][:, 1:]).all():
        raise RuntimeError("Fresh-coordinate manipulation audit is incomplete")
    if not np.isfinite(arrays["old_coordinates"][:, 1:]).all():
        raise RuntimeError("Old-coordinate manipulation audit is incomplete")

    qids = arrays["question_ids"].astype(str).tolist()
    mapping_lookup = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    mappings = [mapping_lookup[qid] for qid in qids]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    score = _load(args.score_projections)
    if score["question_ids"].astype(str).tolist() != qids:
        raise RuntimeError("Score projections and causal run use different question order")

    rank_indices = np.asarray(
        [[LETTERS.index(letter) for letter in row] for row in arrays["rank_letters"].astype(str)],
        dtype=np.int64,
    )
    old_winner = rank_indices[:, 0]
    fresh_winner = np.asarray(
        [LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids],
        dtype=np.int64,
    )
    conflict = old_winner != fresh_winner
    discovery = arrays["split"].astype(str) == "discovery"
    confirmation = ~discovery
    if [int(discovery.sum()), int(confirmation.sum()), int(conflict.sum())] != [251, 249, 273]:
        raise RuntimeError("Frozen discovery/confirmation or conflict population changed")

    displayed = arrays["logits"].astype(np.float64)
    semantic = align_displayed_logits_to_semantic(displayed, mappings)
    choices = displayed_argmax_to_semantic_indices(displayed, mappings)
    scenario_index = {name: index for index, name in enumerate(SCENARIOS)}
    natural = scenario_index["trusted_natural"]
    identity = scenario_index["identity_monitor"]

    fresh_target = _rank_target(score["fresh_unique"].astype(float), rank_indices)
    old_target = _rank_target(score["old_unique"].astype(float), rank_indices)
    fresh_target -= fresh_target.mean(-1, keepdims=True)
    old_target -= old_target.mean(-1, keepdims=True)
    fresh_coordinates = arrays["fresh_coordinates"].astype(np.float64)
    old_coordinates = arrays["old_coordinates"].astype(np.float64)
    fresh_alignment = np.einsum("tsqlgr,qr->tsqlg", fresh_coordinates, fresh_target)
    old_alignment = np.einsum("tsqlgr,qr->tsqlg", old_coordinates, old_target)

    rng = np.random.default_rng(args.seed)
    splits = {
        "discovery_conflict": discovery & conflict,
        "confirmation_conflict": confirmation & conflict,
        "all_conflict": conflict,
    }
    summary: dict[str, Any] = {
        "question": "Is direct ordinary-attention rereading of the original or repeated question wording required for later Second Chance computation?",
        "scope": {
            "first_stem_blockade": "Every query from the feedback token onward is prevented from reading 1P question-stem/separator K/V at every ordinary-attention layer.",
            "second_stem_blockade": "Every 2P option-line, post-list cue/query, and final-prefix query is prevented from reading causally prior 2P question-stem/separator K/V at every ordinary-attention layer.",
            "untouched": "First-pass computation, both sets of option lines, all GLA recurrence, and question information already embedded in other token states.",
            "ordinary_layers_one_based": metadata["ordinary_attention_layers_one_based"],
        },
        "validation": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "canonical_conflicts": 273,
            "trusted_natural_max_abs_error": float(np.max(arrays["trusted_max_abs_error"])),
            "identity_max_abs_error": float(np.max(np.abs(displayed[:, identity] - displayed[:, natural]))),
            "all_outputs_finite": True,
            "first_source_count_range": [int(arrays["first_source_count"].min()), int(arrays["first_source_count"].max())],
            "second_source_count_range": [int(arrays["second_source_count"].min()), int(arrays["second_source_count"].max())],
            "max_abs_intervention_change": {
                name: float(np.max(np.abs(displayed[:, index] - displayed[:, identity])))
                for name, index in scenario_index.items() if index >= 2
            },
        },
        "behavior": {},
        "contrasts": {},
        "manipulation": {},
        "provenance": {
            "results": {"path": str(args.results), "sha256": _sha256(args.results)},
            "metadata": {"path": str(args.metadata), "sha256": _sha256(args.metadata)},
            "remapping_plan": {"path": str(args.remapping_plan), "sha256": _sha256(args.remapping_plan)},
            "remapped_baseline": {"path": str(args.remapped_baseline), "sha256": _sha256(args.remapped_baseline)},
            "score_projections": {"path": str(args.score_projections), "sha256": _sha256(args.score_projections)},
        },
    }

    vectors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split_name, mask in splits.items():
        selected = np.flatnonzero(mask)
        summary["behavior"][split_name] = {"n": int(mask.sum()), "scenarios": {}}
        vectors[split_name] = {}
        for scenario, si in scenario_index.items():
            scenario_record: dict[str, Any] = {"tasks": {}}
            vectors[split_name][scenario] = {}
            for ti, task in enumerate(TASKS):
                all_vectors = _scenario_vectors(
                    semantic[ti, si], choices[ti, si], old_winner, fresh_winner
                )
                task_vectors = {name: value[selected] for name, value in all_vectors.items()}
                vectors[split_name][scenario][task] = task_vectors
                scenario_record["tasks"][task] = {
                    name: _interval(value, rng, args.draws)
                    for name, value in task_vectors.items()
                }
            scenario_record["Game_minus_Neutral"] = {
                name: _interval(
                    vectors[split_name][scenario]["Game"][name]
                    - vectors[split_name][scenario]["Neutral"][name],
                    rng,
                    args.draws,
                )
                for name in vectors[split_name][scenario]["Game"]
            }
            summary["behavior"][split_name]["scenarios"][scenario] = scenario_record

        summary["contrasts"][split_name] = {}
        for scenario in SCENARIOS[2:]:
            record: dict[str, Any] = {"tasks": {}}
            for task in TASKS:
                record["tasks"][task] = {
                    endpoint: _interval(
                        vectors[split_name][scenario][task][endpoint]
                        - vectors[split_name]["identity_monitor"][task][endpoint],
                        rng,
                        args.draws,
                    )
                    for endpoint in vectors[split_name][scenario][task]
                }
            record["Game_minus_Neutral_interaction"] = {
                endpoint: _interval(
                    (
                        vectors[split_name][scenario]["Game"][endpoint]
                        - vectors[split_name][scenario]["Neutral"][endpoint]
                    )
                    - (
                        vectors[split_name]["identity_monitor"]["Game"][endpoint]
                        - vectors[split_name]["identity_monitor"]["Neutral"][endpoint]
                    ),
                    rng,
                    args.draws,
                )
                for endpoint in vectors[split_name][scenario]["Game"]
            }
            summary["contrasts"][split_name][scenario] = record

        factorial: dict[str, Any] = {"tasks": {}}
        for task in TASKS:
            factorial["tasks"][task] = {
                endpoint: _interval(
                    vectors[split_name]["block_both_stems"][task][endpoint]
                    - vectors[split_name]["block_first_stem"][task][endpoint]
                    - vectors[split_name]["block_second_stem"][task][endpoint]
                    + vectors[split_name]["identity_monitor"][task][endpoint],
                    rng,
                    args.draws,
                )
                for endpoint in vectors[split_name]["identity_monitor"][task]
            }
        summary["contrasts"][split_name]["two_by_two_interaction"] = factorial

    for split_name, mask in {"discovery": discovery, "confirmation": confirmation}.items():
        selected = np.flatnonzero(mask)
        split_record: dict[str, Any] = {}
        for ti, task in enumerate(TASKS):
            task_record: dict[str, Any] = {}
            for scenario in SCENARIOS[1:]:
                si = scenario_index[scenario]
                scenario_record: dict[str, Any] = {}
                for gi, group in enumerate(GROUPS):
                    layers: list[dict[str, Any]] = []
                    for layer in range(64):
                        layers.append(
                            {
                                "layer": layer + 1,
                                "fresh_alignment": _interval(
                                    fresh_alignment[ti, si, selected, layer, gi],
                                    rng,
                                    args.draws,
                                ),
                                "fresh_alignment_identity": _interval(
                                    fresh_alignment[ti, identity, selected, layer, gi],
                                    rng,
                                    args.draws,
                                ),
                                "fresh_alignment_change": _interval(
                                    fresh_alignment[ti, si, selected, layer, gi]
                                    - fresh_alignment[ti, identity, selected, layer, gi],
                                    rng,
                                    args.draws,
                                ),
                                "fresh_fraction_of_identity": _ratio_interval(
                                    fresh_alignment[ti, si, selected, layer, gi],
                                    fresh_alignment[ti, identity, selected, layer, gi],
                                    rng,
                                    args.draws,
                                ),
                                "old_fraction_of_identity": _ratio_interval(
                                    old_alignment[ti, si, selected, layer, gi],
                                    old_alignment[ti, identity, selected, layer, gi],
                                    rng,
                                    args.draws,
                                ),
                                "old_alignment_change": _interval(
                                    old_alignment[ti, si, selected, layer, gi]
                                    - old_alignment[ti, identity, selected, layer, gi],
                                    rng,
                                    args.draws,
                                ),
                            }
                        )
                    scenario_record[group] = layers
                task_record[scenario] = scenario_record
            split_record[task] = task_record
        summary["manipulation"][split_name] = split_record

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_report(summary, args.output_dir / "REPORT.md")
    _write_figure(summary, args.figure)
    return summary


def _pct(row: dict[str, Any]) -> str:
    return f"{100*row['mean']:.1f}% [{100*row['ci'][0]:.1f}, {100*row['ci'][1]:.1f}]"


def _num(row: dict[str, Any]) -> str:
    return f"{row['mean']:+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]"


def _write_report(summary: dict[str, Any], path: Path) -> None:
    confirmation = summary["behavior"]["confirmation_conflict"]["scenarios"]
    contrasts = summary["contrasts"]["confirmation_conflict"]
    discovery_contrasts = summary["contrasts"]["discovery_conflict"]
    lines = [
        "# Original/repeated question-stem access factorial",
        "",
        "## Conclusion",
        "",
        "- Direct ordinary-attention rereading of the original question wording after the first answer is **not necessary** for preferential Game switching. Blocking it changes held-out Game old-W1 avoidance by "
        f"{_pct(contrasts['block_first_stem']['tasks']['Game']['old_winner_avoidance'])}, while increasing Neutral avoidance by "
        f"{_pct(contrasts['block_first_stem']['tasks']['Neutral']['old_winner_avoidance'])}. The same asymmetry appears in discovery (Game "
        f"{_pct(discovery_contrasts['block_first_stem']['tasks']['Game']['old_winner_avoidance'])}; Neutral "
        f"{_pct(discovery_contrasts['block_first_stem']['tasks']['Neutral']['old_winner_avoidance'])}).",
        "- Direct rereading of the repeated question wording contributes to the decoded fresh-evidence state at the 2P semantic tokens, but it is not required for choosing the fresh W2. Its blockade reduces held-out semantic-token fresh alignment at L40/L48 in both tasks, yet fresh-W2 choice increases in Game by "
        f"{_pct(contrasts['block_second_stem']['tasks']['Game']['fresh_winner_choice'])} and in Neutral by "
        f"{_pct(contrasts['block_second_stem']['tasks']['Neutral']['fresh_winner_choice'])}.",
        "- Blocking both question copies shrinks the held-out Game-minus-Neutral avoidance gap from "
        f"{_pct(confirmation['identity_monitor']['Game_minus_Neutral']['old_winner_avoidance'])} to "
        f"{_pct(confirmation['block_both_stems']['Game_minus_Neutral']['old_winner_avoidance'])}. This happens because Neutral becomes much more switch-prone (+"
        f"{100*contrasts['block_both_stems']['tasks']['Neutral']['old_winner_avoidance']['mean']:.1f} points), not because Game loses its old-winner avoidance.",
        "- The best-supported interpretation is therefore that direct question-stem access is a stabilizing/reconsideration input—especially under Neutral—not the source of Game's strategic old-winner suppression. The intervention does not erase question information already embedded in option states or GLA memory, so it does not rule out every distributed form of recomputation.",
        "",
        "## What was tested",
        "",
        summary["scope"]["first_stem_blockade"],
        "",
        summary["scope"]["second_stem_blockade"],
        "",
        "This is a causal test of direct ordinary-attention rereading of question words. It does not erase question information already stored elsewhere and it leaves GLA recurrent memory untouched.",
        "",
        "## Confirmation conflict behavior",
        "",
        "| Scenario | Game avoids old W1 | Neutral avoids old W1 | Game - Neutral |",
        "|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS[1:]:
        row = confirmation[scenario]
        lines.append(
            f"| {scenario} | {_pct(row['tasks']['Game']['old_winner_avoidance'])} | "
            f"{_pct(row['tasks']['Neutral']['old_winner_avoidance'])} | "
            f"{_pct(row['Game_minus_Neutral']['old_winner_avoidance'])} |"
        )
    lines += ["", "## Causal changes from identity", ""]
    for scenario in SCENARIOS[2:]:
        row = contrasts[scenario]
        lines.append(
            f"- **{scenario}:** avoidance change Game {_pct(row['tasks']['Game']['old_winner_avoidance'])}; "
            f"Neutral {_pct(row['tasks']['Neutral']['old_winner_avoidance'])}; "
            f"task interaction {_pct(row['Game_minus_Neutral_interaction']['old_winner_avoidance'])}. "
            f"Game W1-minus-fresh-W2 logit change {_num(row['tasks']['Game']['old_winner_minus_fresh_winner_logit'])}."
        )
    lines += [
        "",
        "## Fresh-W2 choice on confirmation conflicts",
        "",
        "W2 is the semantic candidate selected by the one-pass remapped baseline. These changes distinguish a directed move toward the freshly favored candidate from arbitrary switching to some other option.",
        "",
        "| Blockade | Game change | Neutral change |",
        "|---|---:|---:|",
    ]
    for scenario in SCENARIOS[2:]:
        row = contrasts[scenario]
        lines.append(
            f"| {scenario} | {_pct(row['tasks']['Game']['fresh_winner_choice'])} | "
            f"{_pct(row['tasks']['Neutral']['fresh_winner_choice'])} |"
        )
    lines += [
        "",
        "## Fresh-evidence coordinate inside repeated options",
        "",
        "The table reports the absolute change in held-out fresh-evidence alignment at the 2P semantic wordpieces. Unlike a ratio, this remains defined when natural alignment is near zero.",
        "",
        "| Task | Blockade | L31 | L40 | L48 | L64 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    manipulation = summary["manipulation"]["confirmation"]
    for task in TASKS:
        for scenario in SCENARIOS[2:]:
            rows = manipulation[task][scenario]["semantic_wordpieces"]
            values = [rows[layer - 1]["fresh_alignment_change"]["mean"] for layer in (31, 40, 48, 64)]
            lines.append(
                f"| {task} | {scenario} | " + " | ".join(f"{value:+.3f}" for value in values) + " |"
            )
    lines += [
        "",
        "## Validity",
        "",
        f"- Natural reproduction maximum error: {summary['validation']['trusted_natural_max_abs_error']:.9g}.",
        f"- Identity-monitor maximum error: {summary['validation']['identity_max_abs_error']:.9g}.",
        f"- Canonical conflicts: {summary['validation']['canonical_conflicts']}.",
        "- The complete layerwise fresh/old coordinate manipulation audit is in `summary.json` and the canonical figure.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_figure(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Game": "#d95f02", "Neutral": "#1b7fb8"}
    scenarios = list(SCENARIOS[2:])
    labels = ["block 1P", "block 2P", "block both"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    confirmation = summary["contrasts"]["confirmation_conflict"]
    x = np.arange(len(scenarios))
    for ti, task in enumerate(TASKS):
        rows = [confirmation[s]["tasks"][task]["old_winner_avoidance"] for s in scenarios]
        values = np.asarray([r["mean"] for r in rows])
        cis = np.asarray([r["ci"] for r in rows])
        axes[0].bar(
            x + (ti - 0.5) * 0.36,
            values,
            0.36,
            yerr=np.stack((values - cis[:, 0], cis[:, 1] - values)),
            capsize=3,
            color=colors[task],
            label=task,
        )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Change in probability of leaving old W1")
    axes[0].set_title("A  Behavioral effect")
    axes[0].legend()

    manipulation = summary["manipulation"]["confirmation"]
    for ai, group in enumerate(GROUPS, start=1):
        for task in TASKS:
            for scenario, linestyle in (("block_first_stem", "--"), ("block_second_stem", ":"), ("block_both_stems", "-")):
                rows = manipulation[task][scenario][group]
                axes[ai].plot(
                    [r["layer"] for r in rows],
                    [r["fresh_alignment_change"]["mean"] for r in rows],
                    color=colors[task],
                    linestyle=linestyle,
                    alpha=0.9,
                    label=f"{task}, {scenario.replace('block_', '').replace('_stem', '')}",
                )
        axes[ai].axhline(0, color="black", lw=0.8)
        axes[ai].set_xlabel("Layer")
        axes[ai].set_ylabel("Change in fresh-evidence alignment")
        axes[ai].set_title(f"{'B' if ai == 1 else 'C'}  {group.replace('_', ' ')}")
        axes[ai].legend(fontsize=7, ncol=2)
    fig.suptitle("Does later computation require rereading either question stem?")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--score-projections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
