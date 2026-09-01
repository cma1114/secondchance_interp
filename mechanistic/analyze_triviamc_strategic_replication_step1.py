from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"
TASKS = ("Game", "Neutral")
CONDITION_FILES = {
    "Game": "incorrect_again_results.json",
    "Neutral": "lost_again_results.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=-1, keepdims=True)


def _entropy_bits(values: np.ndarray) -> np.ndarray:
    probabilities = _softmax(values)
    return -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=-1)


def _interval(
    values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float | list[float]]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    samples = np.zeros((draws, values.shape[1]), dtype=float)
    for group in groups:
        chosen = rng.choice(group, size=(draws, len(group)), replace=True)
        samples += values[chosen].sum(axis=1)
    samples /= len(values)
    mean = values.mean(axis=0)
    low, high = np.quantile(samples, (0.025, 0.975), axis=0)
    if squeeze:
        return {"mean": float(mean[0]), "ci95": [float(low[0]), float(high[0])]}
    return {
        "mean": [float(value) for value in mean],
        "ci95_low": [float(value) for value in low],
        "ci95_high": [float(value) for value in high],
    }


def _fmt(entry: dict, scale: float = 1.0, digits: int = 1, signed: bool = False) -> str:
    sign = "+" if signed else ""
    low, high = entry["ci95"]
    return (
        f"{entry['mean'] * scale:{sign}.{digits}f} "
        f"[{low * scale:{sign}.{digits}f}, {high * scale:{sign}.{digits}f}]"
    )


def _vector_fmt(entry: dict, scale: float = 1.0, digits: int = 3) -> list[str]:
    return [
        f"{mean * scale:+.{digits}f} [{low * scale:+.{digits}f}, {high * scale:+.{digits}f}]"
        for mean, low, high in zip(entry["mean"], entry["ci95_low"], entry["ci95_high"])
    ]


def analyze(
    baseline_path: Path,
    run_dir: Path,
    remapping_path: Path,
    split_path: Path,
    output_dir: Path,
    figure_path: Path,
    seed: int,
    draws: int,
) -> dict:
    baseline_payload = json.loads(baseline_path.read_text())
    baseline = baseline_payload["results"]
    remapping_payload = json.loads(remapping_path.read_text())
    remapping = {row["question_id"]: row for row in remapping_payload["rows"]}
    qids = [row["question_id"] for row in remapping_payload["rows"]]
    split_payload = json.loads(split_path.read_text())
    discovery_ids = set(split_payload["discovery_question_ids"])
    confirmation_ids = set(split_payload["confirmation_question_ids"])
    if discovery_ids & confirmation_ids or discovery_ids | confirmation_ids != set(qids):
        raise ValueError("Frozen split does not exactly partition the remapping questions")

    payloads = {
        task: json.loads((run_dir / filename).read_text())
        for task, filename in CONDITION_FILES.items()
    }
    for task, payload in payloads.items():
        if set(payload["results"]) != set(qids):
            raise ValueError(f"{task} result IDs do not match the remapping plan")
        if not payload.get("complete") or payload.get("n_results") != len(qids):
            raise ValueError(f"{task} output is not complete")

    first_logits = np.asarray(
        [baseline[qid]["aggregated_ad_logits"] for qid in qids], dtype=float
    )
    if not np.isfinite(first_logits).all():
        raise ValueError("Baseline contains non-finite logits")
    # Stable displayed-order tie handling before any semantic remapping.
    first_order = np.argsort(-first_logits, axis=1, kind="stable")
    first_rank = np.empty_like(first_order)
    first_rank[np.arange(len(qids))[:, None], first_order] = np.arange(4)
    first_choice = np.asarray([baseline[qid]["answer"] for qid in qids])
    first_top_token = np.asarray([
        baseline[qid]["full_vocab_top_token"].strip() for qid in qids
    ])
    if not np.all(np.isin(first_choice, list(LETTERS))):
        raise ValueError("Baseline aggregated answers are not all A-D")
    if not np.all(np.isin(first_top_token, list(LETTERS))):
        raise ValueError("Baseline unrestricted top tokens are not all A-D")

    aligned_first = np.take_along_axis(first_logits, first_order, axis=1)
    first_centered = aligned_first - aligned_first.mean(axis=1, keepdims=True)
    first_probabilities = np.take_along_axis(_softmax(first_logits), first_order, axis=1)
    strata = first_choice.copy()

    final_logits: dict[str, np.ndarray] = {}
    final_centered: dict[str, np.ndarray] = {}
    final_probabilities: dict[str, np.ndarray] = {}
    final_content: dict[str, np.ndarray] = {}
    final_letter: dict[str, np.ndarray] = {}
    aggregated_final_content: dict[str, np.ndarray] = {}
    aggregated_final_letter: dict[str, np.ndarray] = {}
    semantic_switch: dict[str, np.ndarray] = {}
    old_letter_avoidance: dict[str, np.ndarray] = {}
    aggregated_semantic_switch: dict[str, np.ndarray] = {}
    aggregated_old_letter_avoidance: dict[str, np.ndarray] = {}
    choice_rank: dict[str, np.ndarray] = {}
    entropy: dict[str, np.ndarray] = {}
    prompt_pair_exact = np.ones(len(qids), dtype=bool)

    rows = {
        task: [payloads[task]["results"][qid] for qid in qids]
        for task in TASKS
    }
    for qi in range(len(qids)):
        game_prompt = rows["Game"][qi]["rendered_prompt"]
        neutral_prompt = rows["Neutral"][qi]["rendered_prompt"]
        game_sentence = "Your answer was incorrect. Choose the answer again."
        neutral_sentence = "Your answer was lost. Choose the answer again."
        prompt_pair_exact[qi] = (
            game_prompt.count(game_sentence) == 1
            and neutral_prompt.count(neutral_sentence) == 1
            and game_prompt.replace(game_sentence, neutral_sentence) == neutral_prompt
        )

    for task in TASKS:
        displayed = np.asarray([row["aggregated_ad_logits"] for row in rows[task]], dtype=float)
        if not np.isfinite(displayed).all():
            raise ValueError(f"{task} contains non-finite logits")
        final_letter[task] = np.asarray([row["answer_new_letter"] for row in rows[task]])
        final_content[task] = np.asarray([row["answer_original_content"] for row in rows[task]])
        aggregated_final_letter[task] = np.asarray([
            row["aggregated_ad_answer_new_letter"] for row in rows[task]
        ])
        aggregated_final_content[task] = np.asarray([
            row["aggregated_ad_answer_original_content"] for row in rows[task]
        ])
        if not np.all(np.isin(final_letter[task], list(LETTERS))):
            raise ValueError(f"{task} unrestricted answers are not all A-D")
        aligned_original = np.empty_like(displayed)
        for qi, qid in enumerate(qids):
            original_to_new = remapping[qid]["original_to_new"]
            aligned_original[qi] = [
                displayed[qi, LETTERS.index(original_to_new[original])]
                for original in LETTERS
            ]
        final_logits[task] = np.take_along_axis(aligned_original, first_order, axis=1)
        final_centered[task] = final_logits[task] - final_logits[task].mean(axis=1, keepdims=True)
        final_probabilities[task] = _softmax(final_logits[task])
        semantic_switch[task] = final_content[task] != first_choice
        old_letter_avoidance[task] = final_letter[task] != first_choice
        aggregated_semantic_switch[task] = aggregated_final_content[task] != first_choice
        aggregated_old_letter_avoidance[task] = aggregated_final_letter[task] != first_choice
        original_index = np.asarray([LETTERS.index(value) for value in final_content[task]])
        choice_rank[task] = first_rank[np.arange(len(qids)), original_index]
        entropy[task] = _entropy_bits(displayed)

    split_masks = {
        "all": np.ones(len(qids), dtype=bool),
        "discovery": np.asarray([qid in discovery_ids for qid in qids]),
        "confirmation": np.asarray([qid in confirmation_ids for qid in qids]),
    }
    rng = np.random.default_rng(seed)
    split_results: dict[str, dict] = {}
    for split_name, mask in split_masks.items():
        local_strata = strata[mask]
        entry: dict[str, object] = {
            "n": int(mask.sum()),
            "first_rank_raw_logits": _interval(aligned_first[mask], local_strata, rng, draws),
            "first_rank_probabilities": _interval(first_probabilities[mask], local_strata, rng, draws),
            "tasks": {},
        }
        for task in TASKS:
            task_entry = {
                "semantic_switch": _interval(semantic_switch[task][mask], local_strata, rng, draws),
                "old_letter_avoidance": _interval(old_letter_avoidance[task][mask], local_strata, rng, draws),
                "aggregated_semantic_switch": _interval(
                    aggregated_semantic_switch[task][mask], local_strata, rng, draws
                ),
                "aggregated_old_letter_avoidance": _interval(
                    aggregated_old_letter_avoidance[task][mask], local_strata, rng, draws
                ),
                "entropy_bits": _interval(entropy[task][mask], local_strata, rng, draws),
                "final_rank_raw_logits": _interval(final_logits[task][mask], local_strata, rng, draws),
                "raw_paired_change": _interval(
                    final_logits[task][mask] - aligned_first[mask], local_strata, rng, draws
                ),
                "centered_paired_change": _interval(
                    final_centered[task][mask] - first_centered[mask], local_strata, rng, draws
                ),
                "final_rank_probabilities": _interval(
                    final_probabilities[task][mask], local_strata, rng, draws
                ),
                "final_choice_by_old_rank": _interval(
                    np.column_stack([choice_rank[task][mask] == rank for rank in range(4)]),
                    local_strata,
                    rng,
                    draws,
                ),
            }
            entry["tasks"][task] = task_entry
        entry["game_minus_neutral"] = {
            "semantic_switch": _interval(
                semantic_switch["Game"][mask].astype(float)
                - semantic_switch["Neutral"][mask].astype(float),
                local_strata,
                rng,
                draws,
            ),
            "old_letter_avoidance": _interval(
                old_letter_avoidance["Game"][mask].astype(float)
                - old_letter_avoidance["Neutral"][mask].astype(float),
                local_strata,
                rng,
                draws,
            ),
            "aggregated_semantic_switch": _interval(
                aggregated_semantic_switch["Game"][mask].astype(float)
                - aggregated_semantic_switch["Neutral"][mask].astype(float),
                local_strata,
                rng,
                draws,
            ),
            "aggregated_old_letter_avoidance": _interval(
                aggregated_old_letter_avoidance["Game"][mask].astype(float)
                - aggregated_old_letter_avoidance["Neutral"][mask].astype(float),
                local_strata,
                rng,
                draws,
            ),
            "centered_final_rank_evidence": _interval(
                final_centered["Game"][mask] - final_centered["Neutral"][mask],
                local_strata,
                rng,
                draws,
            ),
            "entropy_bits": _interval(
                entropy["Game"][mask] - entropy["Neutral"][mask],
                local_strata,
                rng,
                draws,
            ),
        }
        split_results[split_name] = entry

    summary = {
        "status": "complete",
        "design": {
            "dataset": "TriviaMC_difficulty_filtered",
            "questions": len(qids),
            "conditions": {
                "Game": "Your answer was incorrect. Choose the answer again.",
                "Neutral": "Your answer was lost. Choose the answer again.",
            },
            "remapping": "balanced four-option derangement",
            "remapping_seed": remapping_payload["seed"],
            "split_seed": split_payload["seed"],
            "choice_rule": (
                "aggregated A-D first-presentation W1/ranks; unrestricted final top token "
                "primary; stable displayed A-D argmax is the secondary final choice"
            ),
        },
        "validation": {
            "baseline_complete": bool(baseline_payload.get("complete")),
            "baseline_n": int(baseline_payload.get("n_results", 0)),
            "baseline_unrestricted_ad_fraction": float(np.mean(np.isin(first_top_token, list(LETTERS)))),
            "baseline_aggregated_vs_unrestricted_agreement": float(np.mean(first_choice == first_top_token)),
            "paired_prompts_differ_only_in_incorrect_lost": bool(prompt_pair_exact.all()),
            "game_neutral_prompt_pair_count": int(prompt_pair_exact.sum()),
            "all_logits_finite": True,
            "game_ad_probability_mass_mean": float(np.mean([row["ad_probability_mass"] for row in rows["Game"]])),
            "neutral_ad_probability_mass_mean": float(np.mean([row["ad_probability_mass"] for row in rows["Neutral"]])),
        },
        "results": split_results,
        "provenance": {
            "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "game": {"path": str(run_dir / CONDITION_FILES["Game"]), "sha256": _sha256(run_dir / CONDITION_FILES["Game"])},
            "neutral": {"path": str(run_dir / CONDITION_FILES["Neutral"]), "sha256": _sha256(run_dir / CONDITION_FILES["Neutral"])},
            "remapping": {"path": str(remapping_path), "sha256": _sha256(remapping_path)},
            "split": {"path": str(split_path), "sha256": _sha256(split_path)},
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with (output_dir / "trial_table.csv").open("w", newline="") as stream:
        fields = [
            "question_id", "split", "first_answer", "game_answer_content",
            "neutral_answer_content", "game_semantic_switch", "neutral_semantic_switch",
            "game_old_letter_avoidance", "neutral_old_letter_avoidance",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for qi, qid in enumerate(qids):
            writer.writerow({
                "question_id": qid,
                "split": "discovery" if qid in discovery_ids else "confirmation",
                "first_answer": first_choice[qi],
                "game_answer_content": final_content["Game"][qi],
                "neutral_answer_content": final_content["Neutral"][qi],
                "game_semantic_switch": bool(semantic_switch["Game"][qi]),
                "neutral_semantic_switch": bool(semantic_switch["Neutral"][qi]),
                "game_old_letter_avoidance": bool(old_letter_avoidance["Game"][qi]),
                "neutral_old_letter_avoidance": bool(old_letter_avoidance["Neutral"][qi]),
            })

    confirmation = split_results["confirmation"]
    ranks = np.arange(1, 5)
    colors = {"Game": "#c44e52", "Neutral": "#4c72b0"}
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))

    ax = axes[0, 0]
    for ti, task in enumerate(TASKS):
        values = [
            confirmation["tasks"][task]["semantic_switch"],
            confirmation["tasks"][task]["old_letter_avoidance"],
        ]
        x = np.arange(2) + (ti - 0.5) * 0.28
        means = [value["mean"] * 100 for value in values]
        errors = np.asarray([
            [value["mean"] - value["ci95"][0], value["ci95"][1] - value["mean"]]
            for value in values
        ]).T * 100
        ax.bar(x, means, width=0.28, color=colors[task], label=task)
        ax.errorbar(x, means, yerr=errors, fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(np.arange(2), ["Semantic switch", "Old-letter avoidance"])
    ax.set_ylabel("Questions (%)")
    ax.set_title("Behavioral dissociation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    first = confirmation["first_rank_probabilities"]
    ax.plot(ranks, np.asarray(first["mean"]) * 100, color="#555555", marker="o", label="First presentation")
    for task in TASKS:
        values = confirmation["tasks"][task]["final_rank_probabilities"]
        ax.plot(ranks, np.asarray(values["mean"]) * 100, color=colors[task], marker="o", label=f"{task} final")
        ax.fill_between(ranks, np.asarray(values["ci95_low"]) * 100, np.asarray(values["ci95_high"]) * 100, color=colors[task], alpha=0.15)
    ax.set_xticks(ranks, ["W1", "W2", "W3", "W4"])
    ax.set_ylabel("Mean per-question A-D probability (%)")
    ax.set_title("Old-rank probability profile")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for task in TASKS:
        values = confirmation["tasks"][task]["centered_paired_change"]
        mean = np.asarray(values["mean"])
        ax.plot(ranks, mean, color=colors[task], marker="o", label=task)
        ax.fill_between(ranks, values["ci95_low"], values["ci95_high"], color=colors[task], alpha=0.15)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xticks(ranks, ["W1", "W2", "W3", "W4"])
    ax.set_ylabel("Final − first centered logit")
    ax.set_title("Rank-specific evidence change")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    for task in TASKS:
        values = confirmation["tasks"][task]["final_choice_by_old_rank"]
        mean = np.asarray(values["mean"]) * 100
        ax.plot(ranks, mean, color=colors[task], marker="o", label=task)
        ax.fill_between(ranks, np.asarray(values["ci95_low"]) * 100, np.asarray(values["ci95_high"]) * 100, color=colors[task], alpha=0.15)
    ax.set_xticks(ranks, ["W1", "W2", "W3", "W4"])
    ax.set_ylabel("Final choices (%)")
    ax.set_title("Chosen old rank")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    figure.suptitle("Qwen3.6-27B TriviaMC strategic-switching replication — confirmation")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=190)
    plt.close(figure)

    lines = [
        "# TriviaMC strategic-switching replication — Step 1",
        "",
        "## Design",
        "",
        "The same pinned Qwen3.6-27B model was run on all 500 questions in the existing difficulty-filtered TriviaMC manifest. The first presentation is unchanged. In the second presentation every answer content moves to a different A-D letter using a frozen balanced derangement. Game and Neutral differ only at `incorrect` versus `lost` in `Your answer was ... Choose the answer again.` Unrestricted top-token choices are primary; aggregated A-D logits provide the continuous rank analysis.",
        "",
        "## Validation",
        "",
        f"- Questions: 500; discovery: {split_results['discovery']['n']}; confirmation: {split_results['confirmation']['n']}.",
        f"- Baseline unrestricted A-D fraction: {summary['validation']['baseline_unrestricted_ad_fraction']:.1%}.",
        f"- Baseline aggregated/unrestricted agreement: {summary['validation']['baseline_aggregated_vs_unrestricted_agreement']:.1%}.",
        f"- All 500 paired prompts differ only at `incorrect`/`lost`: {summary['validation']['paired_prompts_differ_only_in_incorrect_lost']}.",
        f"- All logits finite: {summary['validation']['all_logits_finite']}.",
        "",
        "## Behavioral result",
        "",
        "| Split | Game semantic switch | Neutral semantic switch | Game − Neutral | Game old-letter avoidance | Neutral old-letter avoidance | Game − Neutral |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("discovery", "confirmation", "all"):
        cell = split_results[split_name]
        lines.append(
            f"| {split_name.title()} | {_fmt(cell['tasks']['Game']['semantic_switch'], 100)}% | "
            f"{_fmt(cell['tasks']['Neutral']['semantic_switch'], 100)}% | "
            f"{_fmt(cell['game_minus_neutral']['semantic_switch'], 100, signed=True)} pp | "
            f"{_fmt(cell['tasks']['Game']['old_letter_avoidance'], 100)}% | "
            f"{_fmt(cell['tasks']['Neutral']['old_letter_avoidance'], 100)}% | "
            f"{_fmt(cell['game_minus_neutral']['old_letter_avoidance'], 100, signed=True)} pp |"
        )

    lines += [
        "",
        "## First-to-final old-rank transformation",
        "",
        "The tables below use the untouched confirmation half. Candidates are ordered by their first-presentation aggregated A-D logits. Raw changes preserve the common four-logit shift; centered changes remove it within question.",
        "",
        "| Old rank | First raw logit | Game final raw logit | Game raw change | Game centered change | Neutral final raw logit | Neutral raw change | Neutral centered change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    first_raw = confirmation["first_rank_raw_logits"]
    game_raw = confirmation["tasks"]["Game"]["final_rank_raw_logits"]
    neutral_raw = confirmation["tasks"]["Neutral"]["final_rank_raw_logits"]
    game_change = confirmation["tasks"]["Game"]["raw_paired_change"]
    neutral_change = confirmation["tasks"]["Neutral"]["raw_paired_change"]
    game_centered = confirmation["tasks"]["Game"]["centered_paired_change"]
    neutral_centered = confirmation["tasks"]["Neutral"]["centered_paired_change"]
    vector_cells = {
        "first": _vector_fmt(first_raw), "game_raw": _vector_fmt(game_raw),
        "neutral_raw": _vector_fmt(neutral_raw), "game_change": _vector_fmt(game_change),
        "neutral_change": _vector_fmt(neutral_change), "game_centered": _vector_fmt(game_centered),
        "neutral_centered": _vector_fmt(neutral_centered),
    }
    for rank in range(4):
        lines.append(
            f"| W{rank + 1} | {vector_cells['first'][rank]} | {vector_cells['game_raw'][rank]} | "
            f"{vector_cells['game_change'][rank]} | {vector_cells['game_centered'][rank]} | "
            f"{vector_cells['neutral_raw'][rank]} | {vector_cells['neutral_change'][rank]} | "
            f"{vector_cells['neutral_centered'][rank]} |"
        )
    lines += [
        "",
        "### Mean per-question A-D probabilities",
        "",
        "| Old rank | First presentation | Game final | Neutral final |",
        "|---|---:|---:|---:|",
    ]
    probability_cells = {
        "first": _vector_fmt(confirmation["first_rank_probabilities"], 100, 1),
        "game": _vector_fmt(confirmation["tasks"]["Game"]["final_rank_probabilities"], 100, 1),
        "neutral": _vector_fmt(confirmation["tasks"]["Neutral"]["final_rank_probabilities"], 100, 1),
    }
    for rank in range(4):
        lines.append(
            f"| W{rank + 1} | {probability_cells['first'][rank]}% | "
            f"{probability_cells['game'][rank]}% | {probability_cells['neutral'][rank]}% |"
        )
    lines += [
        "",
        "### Aggregated A-D choice robustness",
        "",
        "| Split | Game semantic switch | Neutral semantic switch | Game − Neutral | Game old-letter avoidance | Neutral old-letter avoidance | Game − Neutral |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("discovery", "confirmation", "all"):
        cell = split_results[split_name]
        lines.append(
            f"| {split_name.title()} | {_fmt(cell['tasks']['Game']['aggregated_semantic_switch'], 100)}% | "
            f"{_fmt(cell['tasks']['Neutral']['aggregated_semantic_switch'], 100)}% | "
            f"{_fmt(cell['game_minus_neutral']['aggregated_semantic_switch'], 100, signed=True)} pp | "
            f"{_fmt(cell['tasks']['Game']['aggregated_old_letter_avoidance'], 100)}% | "
            f"{_fmt(cell['tasks']['Neutral']['aggregated_old_letter_avoidance'], 100)}% | "
            f"{_fmt(cell['game_minus_neutral']['aggregated_old_letter_avoidance'], 100, signed=True)} pp |"
        )
    game_minus_neutral = confirmation["game_minus_neutral"]
    relative_rank = _vector_fmt(game_minus_neutral["centered_final_rank_evidence"])
    lines += [
        "",
        "## Conclusion",
        "",
        f"On the untouched confirmation half, Game leaves the semantic first-presentation winner on "
        f"{_fmt(confirmation['tasks']['Game']['semantic_switch'], 100)}% of questions versus "
        f"{_fmt(confirmation['tasks']['Neutral']['semantic_switch'], 100)}% in Neutral, a paired "
        f"Game-minus-Neutral difference of {_fmt(game_minus_neutral['semantic_switch'], 100, signed=True)} percentage points. "
        "The same direction appears on discovery and under the secondary aggregated A-D choice rule.",
        "",
        f"The literal old-letter result goes the other way: Game-minus-Neutral old-letter avoidance is "
        f"{_fmt(game_minus_neutral['old_letter_avoidance'], 100, signed=True)} points on confirmation. "
        "The extra Game switching therefore follows the earlier winner's semantic content after it moves, not its old A-D character.",
        "",
        "The continuous final evidence is also rank-shaped rather than an equal perturbation to all four candidates. "
        f"Relative to Neutral, Game's centered final evidence for W1/W2/W3/W4 is "
        f"**{relative_rank[0]} / {relative_rank[1]} / {relative_rank[2]} / {relative_rank[3]} logits** on confirmation. "
        f"Game also has {_fmt(game_minus_neutral['entropy_bits'], signed=True, digits=3)} more bits of A-D entropy. "
        "Thus TriviaMC reproduces the qualitative behavioral target: Game is more uncertain overall, but the task difference is specifically concentrated in suppressing the semantic old winner and redistributing relative evidence toward all three alternatives. The effect is smaller behaviorally than the canonical SimpleMC remapping gap, so later causal steps should be treated as a cross-dataset replication rather than assumed to have the original magnitude.",
    ]
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Step 1 is behavioral and descriptive. A positive semantic-switch difference together with a rank-shaped Game transformation argues against equal undirected candidate noise, but it does not establish causal recollection. That question belongs to the gated Step 2 matching-versus-wrong history blockade.",
        "",
        f"Canonical figure: `{figure_path}`.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TriviaMC strategic replication Step 1")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remapping", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(
        args.baseline, args.run_dir, args.remapping, args.split,
        args.output_dir, args.figure, args.seed, args.draws,
    )
    print(json.dumps({"status": result["status"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
