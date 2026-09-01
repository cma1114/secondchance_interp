from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


LETTERS = "ABCD"


def _interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "ci": [None, None]}
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _entropy(values: np.ndarray) -> np.ndarray:
    probabilities = _softmax(values)
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(axis=-1)


def _load(root: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays = dict(np.load(root / "results.npz", allow_pickle=False))
    plan = json.loads((root / "donor_plan.json").read_text())["rows"]
    qids = arrays["question_ids"].astype(str).tolist()
    if [row["question_id"] for row in plan] != qids:
        raise ValueError("Result and donor-plan question order differs")
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result set: {root}")
    return arrays, plan


def _align_by_content(
    values: np.ndarray, donor_plan: list[dict[str, Any]]
) -> np.ndarray:
    output = np.empty_like(values)
    for qi, row in enumerate(donor_plan):
        mapping = row["second_mapping"]["original_to_new"]
        for ci, content in enumerate(LETTERS):
            letter_index = LETTERS.index(mapping[content])
            output[..., qi, ci] = values[..., qi, letter_index]
    return output


def _semantic_answers(values: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    """Argmax semantic logits using displayed A-D order to resolve exact ties."""
    output = np.empty(values.shape[:-1], dtype=np.int64)
    maxima = values.max(axis=-1, keepdims=True)
    for qi, row in enumerate(rows):
        mapping = row["second_mapping"]["original_to_new"]
        displayed_indices = np.asarray(
            [LETTERS.index(mapping[content]) for content in LETTERS]
        )
        candidates = np.where(values[..., qi, :] == maxima[..., qi, :], displayed_indices, 99)
        output[..., qi] = candidates.argmin(axis=-1)
    return output


def _target_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    w1 = np.asarray([LETTERS.index(row["recipient_winner_content"]) for row in rows])
    donor = np.asarray([LETTERS.index(row["donor"]["winner_content"]) for row in rows])
    literal = np.asarray(
        [
            LETTERS.index(row["donor"]["literal_letter_content_in_second"])
            for row in rows
        ]
    )
    primary = np.asarray(
        [row["primary_letter_decoupled_changed_winner"] for row in rows], dtype=bool
    )
    same_winner = donor == w1
    return w1, donor, literal, primary, same_winner


def _split_metrics(
    root: Path, draws: int, seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays, rows = _load(root)
    readouts = arrays["readouts"].astype(int)
    natural = _align_by_content(arrays["natural_logits"], rows)
    patched = _align_by_content(arrays["donor_patched_logits"], rows)
    identity = _align_by_content(arrays["identity_patched_logits"], rows)
    w1, donor, literal, primary, same_winner = _target_arrays(rows)
    q = np.arange(len(rows))
    natural_suppression = natural[1] - natural[0]
    patched_suppression = patched[1] - patched[0]
    identity_suppression = identity[1] - identity[0]
    delta = patched_suppression - natural_suppression[None, :, :]
    identity_delta = identity_suppression - natural_suppression[None, :, :]

    semantic = delta[:, q, donor] - delta[:, q, w1]
    literal_effect = delta[:, q, literal] - delta[:, q, w1]
    semantic_vs_literal = delta[:, q, donor] - delta[:, q, literal]
    identity_semantic = identity_delta[:, q, donor] - identity_delta[:, q, w1]

    natural_answers = _semantic_answers(natural, rows)
    patched_answers = _semantic_answers(patched, rows)
    identity_answers = _semantic_answers(identity, rows)
    natural_game_target = (
        (natural_answers[0] == w1).astype(float)
        - (natural_answers[0] == donor).astype(float)
    )
    patched_game_target = (
        (patched_answers[0] == w1[None, :]).astype(float)
        - (patched_answers[0] == donor[None, :]).astype(float)
    )
    behavioral_transfer = patched_game_target - natural_game_target[None, :]

    rng = np.random.default_rng(seed)
    layer_rows = []
    for li, readout in enumerate(readouts):
        layer_rows.append(
            {
                "readout": int(readout),
                "semantic_target_transfer": _interval(semantic[li, primary], rng, draws),
                "literal_letter_transfer": _interval(
                    literal_effect[li, primary], rng, draws
                ),
                "semantic_minus_literal": _interval(
                    semantic_vs_literal[li, primary], rng, draws
                ),
                "behavioral_target_transfer": _interval(
                    behavioral_transfer[li, primary], rng, draws
                ),
                "identity_semantic_transfer": _interval(
                    identity_semantic[li, primary], rng, draws
                ),
                "same_winner_answer_change_game": _interval(
                    (patched_answers[0, li, same_winner] != natural_answers[0, same_winner]).astype(float),
                    rng,
                    draws,
                ),
                "identity_answer_change_game": _interval(
                    (identity_answers[0, li] != natural_answers[0]).astype(float),
                    rng,
                    draws,
                ),
            }
        )

    summary = {
        "root": str(root),
        "n": len(rows),
        "n_primary_letter_decoupled_changed_winner": int(primary.sum()),
        "n_same_winner_control": int(same_winner.sum()),
        "readouts": readouts.tolist(),
        "donor_mapping_counts": {
            str(index): int(
                sum(row["donor"]["mapping_index"] == index for row in rows)
            )
            for index in range(1, 4)
        },
        "layerwise": layer_rows,
        "max_identity_source_error_norm": float(
            np.nanmax(arrays["identity_source_error_norm"])
        ),
    }
    cache = {
        "natural": natural,
        "patched": patched,
        "identity": identity,
        "w1": w1,
        "donor": donor,
        "literal": literal,
        "primary": primary,
        "same_winner": same_winner,
    }
    return summary, cache, rows


def _rate_test(successes: int, n: int) -> dict[str, Any]:
    return {
        "successes": int(successes),
        "n": int(n),
        "rate": float(successes / n) if n else None,
        "exact_one_sided_p": (
            float(binomtest(successes, n, 1 / 3, alternative="greater").pvalue)
            if n
            else None
        ),
    }


def _battery(
    cache: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    layer_index: int,
    baseline_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())["results"]
    manifest = {
        row["id"]: row for row in json.loads(manifest_path.read_text())["questions"]
    }
    qids = [row["question_id"] for row in rows]
    w1 = cache["w1"]
    baseline_runner = np.asarray(
        [
            int(np.argsort(-np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float))[1])
            for qid in qids
        ]
    )
    correct = np.asarray([LETTERS.index(manifest[qid]["correct_answer"]) for qid in qids])
    baseline_correct = w1 == correct

    output = {}
    for scenario, values in (
        ("natural", cache["natural"]),
        ("donor_patch", cache["patched"][:, layer_index]),
        ("identity_patch", cache["identity"][:, layer_index]),
    ):
        scenario_row = {}
        answers = _semantic_answers(values, rows)
        entropies = _entropy(values)
        for condition_index, condition in enumerate(("game", "neutral")):
            changed = answers[condition_index] != w1
            wrong_changed = (~baseline_correct) & changed
            scenario_row[condition] = {
                "accuracy": float(np.mean(answers[condition_index] == correct)),
                "change_rate": float(changed.mean()),
                "entropy_bits": float(entropies[condition_index].mean()),
                "accincor_changed_baseline_wrong": _rate_test(
                    int(np.sum((answers[condition_index] == correct) & wrong_changed)),
                    int(wrong_changed.sum()),
                ),
                "second_choice_among_changed": _rate_test(
                    int(np.sum((answers[condition_index] == baseline_runner) & changed)),
                    int(changed.sum()),
                ),
            }
        game_change = scenario_row["game"]["change_rate"]
        neutral_change = scenario_row["neutral"]["change_rate"]
        scenario_row["lift"] = {
            "absolute": float(game_change - neutral_change),
            "normalized": float(
                (game_change - neutral_change) / max(1e-12, 1 - neutral_change)
            ),
        }
        output[scenario] = scenario_row
    return output


def _fmt(interval: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    if interval["mean"] is None:
        return "n/a"
    return (
        f"{interval['mean'] * scale:+.{digits}f} "
        f"[{interval['ci'][0] * scale:+.{digits}f}, "
        f"{interval['ci'][1] * scale:+.{digits}f}]"
    )


def analyze(
    discovery_root: Path,
    confirmation_root: Path | None,
    baseline_path: Path,
    manifest_path: Path,
    output: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    discovery, discovery_cache, discovery_rows = _split_metrics(
        discovery_root, draws, seed
    )
    selected = max(
        discovery["layerwise"],
        key=lambda row: row["semantic_target_transfer"]["mean"],
    )
    selected_readout = int(selected["readout"])
    confirmation = confirmation_cache = confirmation_rows = None
    if confirmation_root is not None:
        confirmation, confirmation_cache, confirmation_rows = _split_metrics(
            confirmation_root, draws, seed + 1
        )
        if selected_readout not in confirmation["readouts"]:
            raise ValueError("Confirmation does not contain selected discovery readout")
        confirmation_index = confirmation["readouts"].index(selected_readout)
        confirmation["selected_layer_result"] = confirmation["layerwise"][confirmation_index]
        confirmation["battery"] = _battery(
            confirmation_cache,
            confirmation_rows,
            confirmation_index,
            baseline_path,
            manifest_path,
        )

    summary = {
        "definitions": {
            "semantic_target_transfer": (
                "Patch-induced increase in Game-specific suppression of the donor semantic "
                "winner relative to the recipient winner W1. Positive supports target transfer."
            ),
            "literal_letter_transfer": (
                "Same calculation for the content occupying the donor's old literal answer "
                "letter in the fixed second presentation."
            ),
            "suppression": "Neutral centered A-D logit minus Game centered A-D logit.",
        },
        "selected_readout": selected_readout,
        "selection_rule": "Maximum discovery semantic-target transfer.",
        "discovery": discovery,
        "confirmation": confirmation,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "selected_readout.json").write_text(
        json.dumps(
            {
                "selected_readout": selected_readout,
                "selection_rule": summary["selection_rule"],
                "discovery_result": selected,
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        "# Cross-order first-decision residual patching",
        "",
        "The complete first-decision residual was patched from a different option order while the second presentation was held fixed. No learned direction or semantic projection defines the intervention.",
        "",
        "Discrete answers resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. Continuous quantities are invariant to this tie rule.",
        "",
        f"Discovery: **{discovery['n']}** questions; primary letter-decoupled different-winner pairs: **{discovery['n_primary_letter_decoupled_changed_winner']}**.",
        f"The frozen selection rule chose post-block readout **{selected_readout}**.",
        "",
        "## Discovery-selected result",
        "",
        f"- Semantic target transfer: {_fmt(selected['semantic_target_transfer'])} logits",
        f"- Literal-letter transfer: {_fmt(selected['literal_letter_transfer'])} logits",
        f"- Semantic minus literal: {_fmt(selected['semantic_minus_literal'])} logits",
        f"- Behavioral target transfer: {_fmt(selected['behavioral_target_transfer'], 100, 1)} percentage points",
        f"- Identity-patch semantic transfer: {_fmt(selected['identity_semantic_transfer'])} logits",
    ]
    if confirmation is not None:
        confirm = confirmation["selected_layer_result"]
        lines.extend(
            [
                "",
                "## Frozen confirmation",
                "",
                f"Confirmation: **{confirmation['n']}** questions; primary pairs: **{confirmation['n_primary_letter_decoupled_changed_winner']}**.",
                "",
                f"- Semantic target transfer: {_fmt(confirm['semantic_target_transfer'])} logits",
                f"- Literal-letter transfer: {_fmt(confirm['literal_letter_transfer'])} logits",
                f"- Semantic minus literal: {_fmt(confirm['semantic_minus_literal'])} logits",
                f"- Behavioral target transfer: {_fmt(confirm['behavioral_target_transfer'], 100, 1)} percentage points",
                f"- Identity-patch semantic transfer: {_fmt(confirm['identity_semantic_transfer'])} logits",
                "",
                "## Complete behavioral battery at the selected readout",
                "",
                "| Scenario | Game accuracy | Game change | Neutral change | Normalized lift | Game AccIncor | Game second choice | Game entropy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for scenario, label in (
            ("natural", "Natural"),
            ("donor_patch", "Donor patch"),
            ("identity_patch", "Identity patch"),
        ):
            row = confirmation["battery"][scenario]
            acc = row["game"]["accincor_changed_baseline_wrong"]
            second = row["game"]["second_choice_among_changed"]
            lines.append(
                f"| {label} | {row['game']['accuracy']:.1%} | {row['game']['change_rate']:.1%} | "
                f"{row['neutral']['change_rate']:.1%} | {row['lift']['normalized']:.3f} | "
                f"{acc['successes']}/{acc['n']} = {acc['rate']:.1%} | "
                f"{second['successes']}/{second['n']} = {second['rate']:.1%} | "
                f"{row['game']['entropy_bits']:.3f} bits |"
            )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")

    _plot(summary, output)
    return summary


def _plot(summary: dict[str, Any], output: Path) -> None:
    discovery = summary["discovery"]
    layers = np.asarray(discovery["readouts"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for key, label, color in (
        ("semantic_target_transfer", "Donor semantic winner", "#348ce8"),
        ("literal_letter_transfer", "Donor literal letter", "#ef7f32"),
    ):
        rows = [row[key] for row in discovery["layerwise"]]
        means = np.asarray([row["mean"] for row in rows])
        low = np.asarray([row["ci"][0] for row in rows])
        high = np.asarray([row["ci"][1] for row in rows])
        ax.plot(layers, means, color=color, label=label)
        ax.fill_between(layers, low, high, color=color, alpha=0.16)
    ax.axhline(0, color="#555", linewidth=1)
    ax.axvline(summary["selected_readout"], color="#777", linestyle="--")
    ax.set_title("A  Discovery target-transfer scan")
    ax.set_xlabel("Post-block residual readout")
    ax.set_ylabel("Suppression-target transfer (logit units)")
    ax.legend(frameon=False)

    ax = axes[1]
    confirmation = summary.get("confirmation")
    source = (
        confirmation["selected_layer_result"]
        if confirmation is not None
        else next(
            row
            for row in discovery["layerwise"]
            if row["readout"] == summary["selected_readout"]
        )
    )
    keys = (
        "semantic_target_transfer",
        "literal_letter_transfer",
        "semantic_minus_literal",
        "identity_semantic_transfer",
    )
    labels = ("Semantic", "Literal letter", "Semantic − literal", "Identity control")
    x = np.arange(len(keys))
    means = np.asarray([source[key]["mean"] for key in keys])
    low = np.asarray([source[key]["ci"][0] for key in keys])
    high = np.asarray([source[key]["ci"][1] for key in keys])
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([means - low, high - means]),
        fmt="o",
        linestyle="none",
        capsize=4,
        color="#348ce8",
    )
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Effect (logit units)")
    ax.set_title(
        "B  Frozen confirmation" if confirmation is not None else "B  Discovery-selected readout"
    )
    fig.suptitle("Does the first-decision residual carry the remembered semantic winner?")
    fig.tight_layout()
    fig.savefig(output / "first_decision_cross_order_patching.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    analyze(
        args.discovery,
        args.confirmation,
        args.baseline,
        args.manifest,
        args.output,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
