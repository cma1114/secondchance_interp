from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("game", "neutral")


def interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "ci": [None, None]}
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def align(values: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    output = np.empty_like(values)
    for qi, row in enumerate(rows):
        mapping = row["second_mapping"]["original_to_new"]
        for content in LETTERS:
            output[:, qi, LETTERS.index(content)] = values[
                :, qi, LETTERS.index(mapping[content])
            ]
    return output


def semantic_answers(values: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
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


def analyze_split(root: Path, draws: int, seed: int) -> dict[str, Any]:
    arrays = dict(np.load(root / "results.npz", allow_pickle=False))
    rows = json.loads((root / "donor_plan.json").read_text())["rows"]
    if not np.all(arrays["completed"]):
        raise ValueError(f"Incomplete split: {root}")
    rng = np.random.default_rng(seed)
    natural = align(arrays["natural_logits"], rows)
    identity = align(arrays["identity_state_logits"], rows)
    different = align(arrays["different_winner_state_logits"], rows)
    same = align(arrays["same_winner_state_logits"], rows)
    q = np.arange(len(rows))
    w1 = np.asarray([LETTERS.index(row["recipient_winner_content"]) for row in rows])
    donor = np.asarray([
        LETTERS.index(row["different_winner_donor"]["winner_content"]) for row in rows
    ])
    primary = np.asarray([
        row["primary_letter_decoupled_changed_winner"] for row in rows
    ], dtype=bool)
    has_same = arrays["has_same_winner_control"].astype(bool)

    semantic_margin = (
        (different[:, q, w1] - different[:, q, donor])
        - (identity[:, q, w1] - identity[:, q, donor])
    )[:, primary]
    natural_answer = semantic_answers(natural, rows)
    identity_answer = semantic_answers(identity, rows)
    different_answer = semantic_answers(different, rows)
    same_answer = semantic_answers(same, rows)
    result: dict[str, Any] = {
        "root": str(root),
        "n": len(rows),
        "n_primary": int(primary.sum()),
        "n_same_winner_control": int(has_same.sum()),
        "identity_vs_unsplit": {
            "max_absolute_logit_difference": float(np.max(np.abs(identity - natural))),
            "answer_differences": {
                condition: int(np.sum(identity_answer[ci] != natural_answer[ci]))
                for ci, condition in enumerate(CONDITIONS)
            },
        },
        "different_winner_transplant": {},
        "same_winner_control": {},
    }
    for ci, condition in enumerate(CONDITIONS):
        donor_choice_change = (
            (different_answer[ci] == donor).astype(float)
            - (identity_answer[ci] == donor).astype(float)
        )[primary]
        w1_choice_change = (
            (different_answer[ci] == w1).astype(float)
            - (identity_answer[ci] == w1).astype(float)
        )[primary]
        result["different_winner_transplant"][condition] = {
            "w1_minus_donor_margin_change": interval(
                semantic_margin[ci], rng, draws
            ),
            "donor_choice_rate_change": interval(donor_choice_change, rng, draws),
            "w1_choice_rate_change": interval(w1_choice_change, rng, draws),
            "any_answer_change_rate": float(np.mean(
                different_answer[ci, primary] != identity_answer[ci, primary]
            )),
        }
        same_delta = same[ci, has_same] - identity[ci, has_same]
        same_centered = same_delta - same_delta.mean(-1, keepdims=True)
        result["same_winner_control"][condition] = {
            "centered_ad_rms_change": interval(
                np.sqrt(np.mean(same_centered**2, axis=-1)), rng, draws
            ),
            "any_answer_change_rate": float(np.mean(
                same_answer[ci, has_same] != identity_answer[ci, has_same]
            )),
        }
    result["different_winner_transplant"]["game_minus_neutral"] = {
        "w1_minus_donor_margin_change": interval(
            semantic_margin[0] - semantic_margin[1], rng, draws
        )
    }
    relative = arrays["different_state_delta_norm"] / np.maximum(
        arrays["recipient_state_norm"], 1e-12
    )
    result["state_difference_norms"] = {
        "mean_relative_norm_by_gla_layer": np.nanmean(relative, axis=0).tolist(),
        "mean_relative_norm_all_layers": float(np.nanmean(relative)),
    }
    return result


def fmt(value: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{value['mean'] * scale:+.3f} "
        f"[{value['ci'][0] * scale:+.3f}, {value['ci'][1] * scale:+.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    discovery = analyze_split(args.discovery, args.draws, 20260812)
    confirmation = analyze_split(args.confirmation, args.draws, 20260813)
    summary = {
        "definitions": {
            "primary_endpoint": (
                "Change caused by different-winner state transplantation, relative "
                "to recipient-state reinsertion, in the recipient W1 minus donor "
                "semantic-winner logit margin. Positive values mean the donor winner "
                "is suppressed relative to W1."
            ),
            "game_minus_neutral": (
                "Primary endpoint in Game minus the same endpoint in Neutral."
            ),
        },
        "discovery": discovery,
        "confirmation": confirmation,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# First-boundary accumulated GLA-state transplant",
        "",
        "The visible recipient prompt and second presentation are unchanged. At all "
        "48 GLA layers, the accumulated recurrent matrix state immediately after the "
        "first-answer boundary is replaced. Effects are measured relative to reinserting "
        "the recipient's own state, because splitting the kernel itself is not numerically "
        "identical to the unsplit pass.",
        "",
        "Discrete answers resolve exact ties in displayed A-D order before mapping the winning letter back to semantic content. Continuous quantities are invariant to this tie rule.",
        "",
        "## Primary semantic-transfer result",
        "",
        "Positive values mean transplantation makes the model suppress the donor's "
        "semantic winner relative to the recipient's original first answer.",
        "",
        "| Split | Game | Neutral | Game minus Neutral |",
        "|---|---:|---:|---:|",
    ]
    for name, split in (("Discovery", discovery), ("Confirmation", confirmation)):
        diff = split["different_winner_transplant"]
        lines.append(
            f"| {name} | {fmt(diff['game']['w1_minus_donor_margin_change'])} | "
            f"{fmt(diff['neutral']['w1_minus_donor_margin_change'])} | "
            f"{fmt(diff['game_minus_neutral']['w1_minus_donor_margin_change'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A selective semantic-memory transfer requires a replicated positive Game effect "
        "that exceeds Neutral and the same-winner mapping control. If this pattern is absent, "
        "the full accumulated GLA state at this boundary is not a clean, transplantable "
        "representation of the first semantic answer, even if disrupting GLA writes there "
        "affects later suppression.",
        "",
        "## Numerical control",
        "",
        f"Discovery identity reinsertion versus unsplit maximum logit difference: "
        f"{discovery['identity_vs_unsplit']['max_absolute_logit_difference']:.3f}. "
        f"Confirmation: {confirmation['identity_vs_unsplit']['max_absolute_logit_difference']:.3f}. "
        "Accordingly, unsplit natural logits are validation references only, not the causal "
        "counterfactual used for the reported effect.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
