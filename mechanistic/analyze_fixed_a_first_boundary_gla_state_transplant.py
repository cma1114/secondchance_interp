from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _load(root: Path) -> dict[str, np.ndarray]:
    arrays = dict(np.load(root / "results.npz", allow_pickle=False))
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete result: {root}")
    return arrays


def _metrics(root: Path, draws: int, seed: int) -> dict[str, Any]:
    arrays = _load(root)
    natural = arrays["natural_logits"]
    identity = arrays["identity_state_logits"]
    patched = arrays["cross_state_logits"]
    x = np.asarray(["ABCD".index(v) for v in arrays["x_second_letter"].astype(str)])
    y = np.asarray(["ABCD".index(v) for v in arrays["y_second_letter"].astype(str)])
    q = np.arange(len(x))

    natural_x = natural[1, q, x] - natural[0, q, x] - natural[3, q, x] + natural[2, q, x]
    natural_y = natural[3, q, y] - natural[2, q, y] - natural[1, q, y] + natural[0, q, y]
    natural_targeting = 0.5 * (natural_x + natural_y)

    game_x = (
        patched[0, 0, q, x] - patched[0, 0, q, y]
        - identity[0, q, x] + identity[0, q, y]
    )
    game_y = (
        patched[0, 2, q, y] - patched[0, 2, q, x]
        - identity[2, q, y] + identity[2, q, x]
    )
    neutral_x = (
        patched[1, 1, q, x] - patched[1, 1, q, y]
        - identity[1, q, x] + identity[1, q, y]
    )
    neutral_y = (
        patched[1, 3, q, y] - patched[1, 3, q, x]
        - identity[3, q, y] + identity[3, q, x]
    )
    game = 0.5 * (game_x + game_y)
    neutral = 0.5 * (neutral_x + neutral_y)
    divergence = game - neutral

    identity_answers = identity.argmax(axis=-1)
    patched_answers = patched.argmax(axis=-1)
    game_selection = 0.5 * (
        (patched_answers[0, 0] == x).astype(float)
        - (patched_answers[0, 0] == y).astype(float)
        - (identity_answers[0] == x).astype(float)
        + (identity_answers[0] == y).astype(float)
        + (patched_answers[0, 2] == y).astype(float)
        - (patched_answers[0, 2] == x).astype(float)
        - (identity_answers[2] == y).astype(float)
        + (identity_answers[2] == x).astype(float)
    )
    neutral_selection = 0.5 * (
        (patched_answers[1, 1] == x).astype(float)
        - (patched_answers[1, 1] == y).astype(float)
        - (identity_answers[1] == x).astype(float)
        + (identity_answers[1] == y).astype(float)
        + (patched_answers[1, 3] == y).astype(float)
        - (patched_answers[1, 3] == x).astype(float)
        - (identity_answers[3] == y).astype(float)
        + (identity_answers[3] == x).astype(float)
    )

    rng = np.random.default_rng(seed)
    relative = arrays["cross_state_delta_norm"] / np.maximum(
        1e-12,
        0.5 * (
            arrays["recipient_state_norm"][[0, 1]].transpose(0, 2, 1)
            + arrays["recipient_state_norm"][[2, 3]].transpose(0, 2, 1)
        ).transpose(0, 2, 1),
    )
    return {
        "root": str(root),
        "n": int(len(x)),
        "natural_semantic_targeting": _interval(natural_targeting, rng, draws),
        "game_margin_transfer": _interval(game, rng, draws),
        "neutral_margin_transfer": _interval(neutral, rng, draws),
        "game_minus_neutral_transfer": _interval(divergence, rng, draws),
        "game_selection_transfer": _interval(game_selection, rng, draws),
        "neutral_selection_transfer": _interval(neutral_selection, rng, draws),
        "max_identity_vs_natural_ad_logit_difference": float(np.max(np.abs(identity - natural))),
        "identity_vs_natural_answer_differences": {
            cell: int(np.sum(identity_answers[index] != natural.argmax(axis=-1)[index]))
            for index, cell in enumerate(("evaluation_x", "neutral_x", "evaluation_y", "neutral_y"))
        },
        "mean_relative_cross_semantic_state_difference": float(np.nanmean(relative)),
    }


def _trusted_validation(root: Path, trusted_root: Path) -> dict[str, Any]:
    current = _load(root)
    trusted = dict(np.load(trusted_root / "results.npz", allow_pickle=False))
    lookup = {
        qid: index for index, qid in enumerate(trusted["question_ids"].astype(str))
    }
    indices = np.asarray([lookup[qid] for qid in current["question_ids"].astype(str)])
    reference = trusted["natural_logits"][:, indices]
    delta = current["natural_logits"] - reference
    return {
        "mean_absolute_ad_logit_difference": float(np.mean(np.abs(delta))),
        "max_absolute_ad_logit_difference": float(np.max(np.abs(delta))),
        "cell_level_answer_differences": int(
            np.sum(current["natural_logits"].argmax(axis=-1) != reference.argmax(axis=-1))
        ),
        "cell_level_answers": int(reference.shape[0] * reference.shape[1]),
    }


def _fmt(value: dict[str, Any], scale: float = 1.0, suffix: str = "") -> str:
    return (
        f"{scale * value['mean']:+.3f} "
        f"[{scale * value['ci'][0]:+.3f}, {scale * value['ci'][1]:+.3f}]{suffix}"
    )


def analyze(
    discovery_root: Path,
    confirmation_root: Path,
    trusted_root: Path | None,
    output_dir: Path,
    draws: int,
) -> None:
    payload = {
        "definitions": {
            "game_margin_transfer": "Change, relative to recipient-state reinsertion, in the recipient-answer minus donor-answer margin after cross-semantic GLA-state transplantation. Positive means Game redirects suppression toward the donor semantic answer.",
            "neutral_margin_transfer": "The same signed margin change in Neutral. Negative means Neutral redirects repetition toward the donor semantic answer.",
        },
        "discovery": _metrics(discovery_root, draws, 1703),
        "confirmation": _metrics(confirmation_root, draws, 2703),
    }
    shard_roots = {
        name: discovery_root.parent / name
        for name in (
            "discovery_shard0",
            "discovery_shard1",
            "confirmation_shard0",
            "confirmation_shard1",
        )
    }
    payload["shard_sensitivity"] = {
        name: _metrics(root, draws, 3703 + index)
        for index, (name, root) in enumerate(shard_roots.items())
        if (root / "results.npz").exists()
    }
    if trusted_root is not None:
        payload["trusted_natural_validation"] = {
            "discovery": _trusted_validation(discovery_root, trusted_root / "discovery"),
            "confirmation": _trusted_validation(confirmation_root, trusted_root / "confirmation"),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    report = [
        "# Fixed-A accumulated GLA-state transplant",
        "",
        "Both histories produce the literal first answer `A`, but `A` names different semantic answers X and Y. At all 48 GLA layers jointly, the complete accumulated recurrent state immediately after that answer boundary is exchanged X↔Y. Everything visible after the boundary is held fixed.",
        "",
        "Because splitting the recurrent kernel has a numerical effect, every causal estimate compares cross-semantic transplantation with reinserting the recipient's own state through the identical segmented computation.",
        "",
        "## Results",
        "",
        "| Split | Natural semantic-history effect | Game target transfer | Neutral target transfer | Game − Neutral | Game selection transfer | Neutral selection transfer |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("discovery", "confirmation"):
        row = payload[split]
        report.append(
            f"| {split.title()} | {_fmt(row['natural_semantic_targeting'], suffix=' logits')} | "
            f"{_fmt(row['game_margin_transfer'], suffix=' logits')} | "
            f"{_fmt(row['neutral_margin_transfer'], suffix=' logits')} | "
            f"{_fmt(row['game_minus_neutral_transfer'], suffix=' logits')} | "
            f"{_fmt(row['game_selection_transfer'], 100, ' pp')} | "
            f"{_fmt(row['neutral_selection_transfer'], 100, ' pp')} |"
        )
    report.extend([
        "",
        "## Interpretation rule",
        "",
        "If the first-answer GLA memory carries semantic identity, the held-out result should be positive in Game (the donor semantic answer becomes the one suppressed) and negative in Neutral (the donor answer becomes the one repeated). A null result despite substantial donor–recipient state differences would rule out the complete accumulated GLA state at this boundary as a clean transplantable carrier under this intervention.",
        "",
        "## Validation",
        "",
    ])
    for split in ("discovery", "confirmation"):
        row = payload[split]
        report.extend([
            f"- {split.title()} maximum identity-reinsertion versus unsplit A–D logit difference: {row['max_identity_vs_natural_ad_logit_difference']:.3f}.",
            f"- {split.title()} mean cross-semantic GLA-state difference relative to recipient-state norm: {100 * row['mean_relative_cross_semantic_state_difference']:.2f}%.",
        ])
    if "trusted_natural_validation" in payload:
        report.extend([
            "",
            "The replacement hosts did not reproduce the trusted natural logits bit-for-bit, despite using the same batch-of-four prompts and Transformers version:",
            "",
            "| Split | Mean absolute A-D logit difference | Maximum difference | Answer differences |",
            "|---|---:|---:|---:|",
        ])
        for split in ("discovery", "confirmation"):
            value = payload["trusted_natural_validation"][split]
            report.append(
                f"| {split.title()} | {value['mean_absolute_ad_logit_difference']:.3f} | "
                f"{value['max_absolute_ad_logit_difference']:.3f} | "
                f"{value['cell_level_answer_differences']}/{value['cell_level_answers']} |"
            )
        report.extend([
            "",
            "The natural semantic-history interaction nevertheless reproduced closely. Every transplant effect is a within-host comparison against identity-state reinsertion under the identical segmented kernel, so the causal contrast does not compare raw logits across hosts.",
        ])
    report.extend([
        "",
        "## Host/shard sensitivity",
        "",
        "| Shard | N | Game target transfer | Neutral target transfer | Game − Neutral |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, value in payload["shard_sensitivity"].items():
        report.append(
            f"| {name} | {value['n']} | {_fmt(value['game_margin_transfer'], suffix=' logits')} | "
            f"{_fmt(value['neutral_margin_transfer'], suffix=' logits')} | "
            f"{_fmt(value['game_minus_neutral_transfer'], suffix=' logits')} |"
        )
    report.extend([
        "",
        "## Bottom line",
        "",
        "The natural fixed-A semantic-history effect is robust, but the accumulated GLA-state transplant does **not** produce a replicated semantic-target transfer. The held-out Game effect is small and uncertain, Neutral moves in the same rather than the predicted opposite direction, the Game-minus-Neutral contrast is essentially zero, and the shard estimates are heterogeneous. Thus the complete accumulated GLA state at the first-answer boundary is not a clean portable carrier of `which semantic answer I chose`. This leaves intact the earlier finding that disrupting boundary GLA processing affects switching; it means that contribution is not explained by a transplantable semantic-answer state in the GLAs alone.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--trusted-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    analyze(
        args.discovery,
        args.confirmation,
        args.trusted_root,
        args.output_dir,
        args.draws,
    )


if __name__ == "__main__":
    main()
