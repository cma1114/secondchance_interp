from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


LETTERS = "ABCD"
CONDITIONS = ("game", "neutral")
TENSOR_NAMES = ("key", "value", "g", "beta")


def _interval(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> dict[str, Any]:
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
    return -(probabilities * np.log2(np.clip(probabilities, 1e-12, None))).sum(-1)


def _load(root: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays = dict(np.load(root / "results.npz", allow_pickle=False))
    rows = json.loads((root / "donor_plan.json").read_text())["rows"]
    qids = arrays["question_ids"].astype(str).tolist()
    if [row["question_id"] for row in rows] != qids:
        raise ValueError(f"Question ordering differs in {root}")
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result set: {root}")
    return arrays, rows


def _align_by_content(values: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    output = np.empty_like(values)
    for qi, row in enumerate(rows):
        mapping = row["second_mapping"]["original_to_new"]
        for ci, content in enumerate(LETTERS):
            output[..., qi, ci] = values[..., qi, LETTERS.index(mapping[content])]
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


def _targets(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
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
    return w1, donor, literal, primary


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _historical_validation(
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    historical_root: Path,
) -> dict[str, Any]:
    output = {}
    qids = arrays["question_ids"].astype(str).tolist()
    for ci, filename in enumerate(("incorrect_results.json", "neutral_results.json")):
        historical = json.loads((historical_root / filename).read_text())["results"]
        expected = np.asarray(
            [historical[qid]["aggregated_ad_logits"] for qid in qids], dtype=float
        )
        delta = np.abs(arrays["natural_logits"][ci].astype(float) - expected)
        output[CONDITIONS[ci]] = {
            "n": len(qids),
            "exact_rows": int(np.all(delta == 0, axis=1).sum()),
            "max_absolute_logit_difference": float(delta.max()),
        }
    identity_delta = np.abs(
        arrays["identity_patched_logits"].astype(float)
        - arrays["natural_logits"].astype(float)
    )
    output["identity_patch"] = {
        "exact_rows": int(np.all(identity_delta == 0, axis=-1).sum()),
        "n_condition_rows": int(np.prod(identity_delta.shape[:2])),
        "max_absolute_logit_difference": float(identity_delta.max()),
    }
    return output


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
    values: np.ndarray,
    rows: list[dict[str, Any]],
    baseline_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())["results"]
    manifest = {
        row["id"]: row for row in json.loads(manifest_path.read_text())["questions"]
    }
    qids = [row["question_id"] for row in rows]
    w1, _, _, _ = _targets(rows)
    correct = np.asarray([LETTERS.index(manifest[qid]["correct_answer"]) for qid in qids])
    runner = np.asarray(
        [
            int(
                np.argsort(
                    -np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=float),
                    kind="stable",
                )[1]
            )
            for qid in qids
        ]
    )
    baseline_correct = w1 == correct
    answers = _semantic_answers(values, rows)
    entropies = _entropy(values)
    output: dict[str, Any] = {}
    for ci, condition in enumerate(CONDITIONS):
        changed = answers[ci] != w1
        wrong_changed = (~baseline_correct) & changed
        output[condition] = {
            "accuracy": float(np.mean(answers[ci] == correct)),
            "change_rate": float(changed.mean()),
            "entropy_bits": float(entropies[ci].mean()),
            "accincor_changed_baseline_wrong": _rate_test(
                int(np.sum((answers[ci] == correct) & wrong_changed)),
                int(wrong_changed.sum()),
            ),
            "second_choice_among_changed": _rate_test(
                int(np.sum((answers[ci] == runner) & changed)), int(changed.sum())
            ),
        }
    game_change = output["game"]["change_rate"]
    neutral_change = output["neutral"]["change_rate"]
    output["lift"] = {
        "absolute": float(game_change - neutral_change),
        "normalized": float(
            (game_change - neutral_change) / max(1e-12, 1 - neutral_change)
        ),
    }
    return output


def _split(
    root: Path,
    baseline_path: Path,
    manifest_path: Path,
    historical_root: Path,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays, rows = _load(root)
    natural = _align_by_content(arrays["natural_logits"], rows)
    patched = _align_by_content(arrays["donor_patched_logits"], rows)
    identity = _align_by_content(arrays["identity_patched_logits"], rows)
    w1, donor, literal, primary = _targets(rows)
    q = np.arange(len(rows))
    natural_centered = _center(natural)
    patched_centered = _center(patched)
    delta = patched_centered - natural_centered

    semantic_margin_change = (
        (patched[:, q, w1] - patched[:, q, donor])
        - (natural[:, q, w1] - natural[:, q, donor])
    )
    literal_margin_change = (
        (patched[:, q, w1] - patched[:, q, literal])
        - (natural[:, q, w1] - natural[:, q, literal])
    )
    semantic_minus_literal = semantic_margin_change - literal_margin_change
    policy_divergence = semantic_margin_change[0] - semantic_margin_change[1]

    natural_answers = _semantic_answers(natural, rows)
    patched_answers = _semantic_answers(patched, rows)
    identity_answers = _semantic_answers(identity, rows)
    choice_contrast_natural = (
        (natural_answers == w1[None, :]).astype(float)
        - (natural_answers == donor[None, :]).astype(float)
    )
    choice_contrast_patched = (
        (patched_answers == w1[None, :]).astype(float)
        - (patched_answers == donor[None, :]).astype(float)
    )
    choice_contrast_change = choice_contrast_patched - choice_contrast_natural
    behavioral_policy_divergence = choice_contrast_change[0] - choice_contrast_change[1]

    rng = np.random.default_rng(seed)
    conditions: dict[str, Any] = {}
    for ci, condition in enumerate(CONDITIONS):
        conditions[condition] = {
            "w1_minus_donor_margin_change": _interval(
                semantic_margin_change[ci, primary], rng, draws
            ),
            "w1_minus_literal_margin_change": _interval(
                literal_margin_change[ci, primary], rng, draws
            ),
            "semantic_minus_literal_change": _interval(
                semantic_minus_literal[ci, primary], rng, draws
            ),
            "donor_centered_logit_change": _interval(
                delta[ci, q, donor][primary], rng, draws
            ),
            "w1_centered_logit_change": _interval(
                delta[ci, q, w1][primary], rng, draws
            ),
            "donor_selection_change": _interval(
                (
                    (patched_answers[ci] == donor).astype(float)
                    - (natural_answers[ci] == donor).astype(float)
                )[primary],
                rng,
                draws,
            ),
            "w1_selection_change": _interval(
                (
                    (patched_answers[ci] == w1).astype(float)
                    - (natural_answers[ci] == w1).astype(float)
                )[primary],
                rng,
                draws,
            ),
            "choice_contrast_change": _interval(
                choice_contrast_change[ci, primary], rng, draws
            ),
        }

    layer_indices = arrays["gla_layer_indices"].astype(int)
    delta_norm = arrays["donor_target_delta_norm"][:, primary].mean(axis=(0, 1))
    target_norm = arrays["target_write_norm"][:, primary].mean(axis=(0, 1))
    relative = delta_norm / np.clip(target_norm, 1e-12, None)
    activation = {
        "gla_layer_indices_zero_based": layer_indices.tolist(),
        "tensor_names": list(TENSOR_NAMES),
        "mean_donor_target_delta_norm": delta_norm.tolist(),
        "mean_target_norm": target_norm.tolist(),
        "mean_relative_delta": relative.tolist(),
        "peak_relative_delta": {
            name: {
                "model_layer_zero_based": int(layer_indices[np.argmax(relative[:, ti])]),
                "relative_delta": float(np.max(relative[:, ti])),
            }
            for ti, name in enumerate(TENSOR_NAMES)
        },
    }

    summary = {
        "root": str(root),
        "n_questions": len(rows),
        "n_primary_letter_decoupled_changed_winner": int(primary.sum()),
        "conditions": conditions,
        "game_minus_neutral_w1_minus_donor_margin_change": _interval(
            policy_divergence[primary], rng, draws
        ),
        "game_minus_neutral_behavioral_choice_contrast_change": _interval(
            behavioral_policy_divergence[primary], rng, draws
        ),
        "activation": activation,
        "historical_validation": _historical_validation(
            arrays, rows, historical_root
        ),
        "battery": {
            "natural": _battery(natural, rows, baseline_path, manifest_path),
            "donor_patch": _battery(patched, rows, baseline_path, manifest_path),
            "identity_patch": _battery(identity, rows, baseline_path, manifest_path),
        },
    }
    cache = {
        "primary": primary,
        "semantic_margin_change": semantic_margin_change,
        "choice_contrast_change": choice_contrast_change,
        "policy_divergence": policy_divergence,
        "behavioral_policy_divergence": behavioral_policy_divergence,
    }
    return summary, cache


def _fmt(interval: dict[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    if interval["mean"] is None:
        return "n/a"
    return (
        f"{interval['mean'] * scale:+.{digits}f} "
        f"[{interval['ci'][0] * scale:+.{digits}f}, "
        f"{interval['ci'][1] * scale:+.{digits}f}]"
    )


def _pooled(
    discovery_cache: dict[str, np.ndarray],
    confirmation_cache: dict[str, np.ndarray],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    output = {}
    for key in (
        "policy_divergence",
        "behavioral_policy_divergence",
    ):
        values = np.concatenate(
            [
                discovery_cache[key][discovery_cache["primary"]],
                confirmation_cache[key][confirmation_cache["primary"]],
            ]
        )
        output[key] = _interval(values, rng, draws)
    for ci, condition in enumerate(CONDITIONS):
        values = np.concatenate(
            [
                discovery_cache["semantic_margin_change"][ci, discovery_cache["primary"]],
                confirmation_cache["semantic_margin_change"][ci, confirmation_cache["primary"]],
            ]
        )
        output[f"{condition}_semantic_margin_change"] = _interval(values, rng, draws)
        behavior = np.concatenate(
            [
                discovery_cache["choice_contrast_change"][ci, discovery_cache["primary"]],
                confirmation_cache["choice_contrast_change"][ci, confirmation_cache["primary"]],
            ]
        )
        output[f"{condition}_choice_contrast_change"] = _interval(
            behavior, rng, draws
        )
    return output


def _plot(summary: dict[str, Any], output: Path) -> None:
    splits = ("discovery", "confirmation")
    colors = {"game": "#348ce8", "neutral": "#ef7f32"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    ax = axes[0]
    positions = np.arange(2)
    width = 0.34
    for ci, condition in enumerate(CONDITIONS):
        intervals = [
            summary[split]["conditions"][condition]["w1_minus_donor_margin_change"]
            for split in splits
        ]
        means = np.asarray([row["mean"] for row in intervals])
        low = means - np.asarray([row["ci"][0] for row in intervals])
        high = np.asarray([row["ci"][1] for row in intervals]) - means
        ax.bar(
            positions + (ci - 0.5) * width,
            means,
            width,
            color=colors[condition],
            label=condition.title(),
            yerr=np.vstack([low, high]),
            capsize=4,
        )
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xticks(positions, ["Discovery", "Confirmation"])
    ax.set_ylabel("Change in W1 − donor-answer margin (logits)")
    ax.set_title("A  Does policy follow transplanted memory?")
    ax.legend(frameon=False)

    ax = axes[1]
    for ci, condition in enumerate(CONDITIONS):
        intervals = [
            summary[split]["conditions"][condition]["choice_contrast_change"]
            for split in splits
        ]
        means = np.asarray([row["mean"] for row in intervals]) * 100
        low = means - np.asarray([row["ci"][0] for row in intervals]) * 100
        high = np.asarray([row["ci"][1] for row in intervals]) * 100 - means
        ax.bar(
            positions + (ci - 0.5) * width,
            means,
            width,
            color=colors[condition],
            label=condition.title(),
            yerr=np.vstack([low, high]),
            capsize=4,
        )
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xticks(positions, ["Discovery", "Confirmation"])
    ax.set_ylabel("Change in P(W1 choice) − P(donor choice), pp")
    ax.set_title("B  Behavioral target transfer")

    ax = axes[2]
    activation = summary["confirmation"]["activation"]
    layers = np.asarray(activation["gla_layer_indices_zero_based"]) + 1
    relative = np.asarray(activation["mean_relative_delta"])
    for ti, name in enumerate(TENSOR_NAMES):
        ax.plot(layers, relative[:, ti], label=name)
    ax.set_xlabel("Model block")
    ax.set_ylabel("‖donor − target‖ / ‖target‖")
    ax.set_title("C  Size of transplanted GLA writes")
    ax.legend(frameon=False)

    fig.suptitle("First-answer-boundary GLA semantic-memory rewrite", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "first_boundary_gla_memory_rewrite.png", dpi=220)
    plt.close(fig)


def analyze(
    discovery_root: Path,
    confirmation_root: Path,
    baseline_path: Path,
    manifest_path: Path,
    historical_root: Path,
    output: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    discovery, discovery_cache = _split(
        discovery_root,
        baseline_path,
        manifest_path,
        historical_root,
        draws,
        seed,
    )
    confirmation, confirmation_cache = _split(
        confirmation_root,
        baseline_path,
        manifest_path,
        historical_root,
        draws,
        seed + 1,
    )
    pooled = _pooled(discovery_cache, confirmation_cache, draws, seed + 2)
    summary = {
        "definitions": {
            "W1": "semantic answer selected in the original first presentation",
            "donor_answer": (
                "different semantic answer selected under the frozen alternative "
                "first-option mapping"
            ),
            "margin_change": (
                "Donor-patched minus natural change in the final centered-logit margin "
                "W1 minus donor answer. Positive means the donor answer is relatively "
                "disfavored."
            ),
            "policy_divergence": (
                "Game margin change minus Neutral margin change. Positive supports the "
                "same transplanted memory being avoided in Game but retained in Neutral."
            ),
        },
        "intervention": (
            "At the exact first-answer-boundary span and all 48 GLA layers, replace "
            "key, value, decay gate g, and beta with same-question tensors from an "
            "alternative first presentation that produced a different semantic answer."
        ),
        "discovery": discovery,
        "confirmation": confirmation,
        "pooled": pooled,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# First-answer-boundary GLA semantic-memory rewrite",
        "",
        summary["intervention"],
        "",
        "Discrete answers resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. Continuous quantities are invariant to this tie rule.",
        "",
        "Positive W1-minus-donor margin change means the transplanted donor answer became relatively less favored. The key prediction is positive in Game, negative in Neutral, and therefore a positive Game-minus-Neutral policy divergence.",
        "",
        "## Primary causal result",
        "",
        "| Split | Primary pairs | Game W1−donor change | Neutral W1−donor change | Game−Neutral policy divergence | Behavioral divergence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("discovery", "Discovery"), ("confirmation", "Confirmation")):
        row = summary[key]
        lines.append(
            f"| {label} | {row['n_primary_letter_decoupled_changed_winner']} | "
            f"{_fmt(row['conditions']['game']['w1_minus_donor_margin_change'])} | "
            f"{_fmt(row['conditions']['neutral']['w1_minus_donor_margin_change'])} | "
            f"{_fmt(row['game_minus_neutral_w1_minus_donor_margin_change'])} | "
            f"{_fmt(row['game_minus_neutral_behavioral_choice_contrast_change'], 100, 1)} pp |"
        )
    lines.append(
        f"| Pooled | — | {_fmt(pooled['game_semantic_margin_change'])} | "
        f"{_fmt(pooled['neutral_semantic_margin_change'])} | "
        f"{_fmt(pooled['policy_divergence'])} | "
        f"{_fmt(pooled['behavioral_policy_divergence'], 100, 1)} pp |"
    )
    lines.extend(
        [
            "",
            "## Content and letter specificity",
            "",
            "| Split | Condition | Semantic margin change | Literal-letter-control change | Semantic minus literal | Donor selection change | W1 selection change |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split_key, split_label in (("discovery", "Discovery"), ("confirmation", "Confirmation")):
        for condition in CONDITIONS:
            row = summary[split_key]["conditions"][condition]
            lines.append(
                f"| {split_label} | {condition.title()} | "
                f"{_fmt(row['w1_minus_donor_margin_change'])} | "
                f"{_fmt(row['w1_minus_literal_margin_change'])} | "
                f"{_fmt(row['semantic_minus_literal_change'])} | "
                f"{_fmt(row['donor_selection_change'], 100, 1)} pp | "
                f"{_fmt(row['w1_selection_change'], 100, 1)} pp |"
            )
    lines.extend(
        [
            "",
            "## Standard behavioral checks on confirmation",
            "",
            "| Scenario | Game accuracy | Game change | Neutral change | Normalized lift | Game AccIncor | Game second choice | Game entropy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, label in (
        ("natural", "Natural"),
        ("donor_patch", "Donor memory patch"),
        ("identity_patch", "Identity patch"),
    ):
        row = confirmation["battery"][scenario]
        acc = row["game"]["accincor_changed_baseline_wrong"]
        second = row["game"]["second_choice_among_changed"]
        lines.append(
            f"| {label} | {row['game']['accuracy']:.1%} | "
            f"{row['game']['change_rate']:.1%} | {row['neutral']['change_rate']:.1%} | "
            f"{row['lift']['normalized']:.3f} | "
            f"{acc['successes']}/{acc['n']} = {acc['rate']:.1%} | "
            f"{second['successes']}/{second['n']} = {second['rate']:.1%} | "
            f"{row['game']['entropy_bits']:.3f} bits |"
        )
    lines.extend(["", "## Numerical validation", ""])
    for split_key, split_label in (("discovery", "Discovery"), ("confirmation", "Confirmation")):
        validation = summary[split_key]["historical_validation"]
        lines.append(
            f"- {split_label}: natural logits exactly match history on "
            f"{validation['game']['exact_rows']}/{validation['game']['n']} Game and "
            f"{validation['neutral']['exact_rows']}/{validation['neutral']['n']} Neutral rows; "
            f"identity patches exactly match on "
            f"{validation['identity_patch']['exact_rows']}/"
            f"{validation['identity_patch']['n_condition_rows']} condition-rows."
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    _plot(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    analyze(
        args.discovery,
        args.confirmation,
        args.baseline,
        args.manifest,
        args.historical_root,
        args.output,
        args.draws,
        args.seed,
    )


if __name__ == "__main__":
    main()
