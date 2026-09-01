from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_fixed_a_donor_receiver_mediation import SCENARIOS


def _load(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "results.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"].astype(bool)):
        raise RuntimeError(f"Incomplete result: {root}")
    if arrays["scenarios"].astype(str).tolist() != list(SCENARIOS):
        raise RuntimeError("Unexpected scenarios")
    return arrays


def _margin(logits: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.arange(len(x)); out = np.empty((4, len(x)), dtype=float)
    out[0] = logits[0, q, y] - logits[0, q, x]
    out[1] = logits[1, q, y] - logits[1, q, x]
    out[2] = logits[2, q, x] - logits[2, q, y]
    out[3] = logits[3, q, x] - logits[3, q, y]
    return out


def _choice(logits: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    answer = logits.argmax(-1); out = np.empty((4, len(x)), dtype=float)
    out[0] = answer[0] == y; out[1] = answer[1] == y
    out[2] = answer[2] == x; out[3] = answer[3] == x
    return out


def _condition(values: np.ndarray, condition: str) -> np.ndarray:
    rows = (0, 2) if condition == "Game" else (1, 3)
    return 0.5 * (values[rows[0]] + values[rows[1]])


def _ci(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return {"n": len(values), "mean": float(values.mean()), "ci": np.quantile(means, (.025, .975)).tolist()}


def summarize(root: Path, seed: int, draws: int) -> dict[str, Any]:
    arrays = _load(root)
    valid = arrays["first_decision_valid"].astype(bool)
    logits = arrays["scenario_logits"][:, :, valid].astype(float)
    letters = arrays["literal_letters"][valid].astype(str)
    x = np.array(["ABCD".index(value) for value in arrays["x_second_letter"][valid].astype(str)])
    y = np.array(["ABCD".index(value) for value in arrays["y_second_letter"][valid].astype(str)])
    margins = np.stack([_margin(row, x, y) for row in logits])
    choices = np.stack([_choice(row, x, y) for row in logits])
    scenario = {name: index for index, name in enumerate(SCENARIOS)}
    summary: dict[str, Any] = {"letters": {}, "validation": {}}
    analyzed_letters = "".join(letter for letter in "ABCD" if letter in set(letters.tolist()))
    for li, letter in enumerate(analyzed_letters):
        mask = letters == letter
        letter_rows: dict[str, Any] = {}
        for ci, condition in enumerate(("Game", "Neutral")):
            open_transfer = _condition(
                margins[scenario["donor_open"]] - margins[scenario["recipient_open"]], condition
            )[mask]
            matching_transfer = _condition(
                margins[scenario["donor_matching_block"]] - margins[scenario["recipient_matching_block"]], condition
            )[mask]
            control_transfer = _condition(
                margins[scenario["donor_control_block"]] - margins[scenario["recipient_control_block"]], condition
            )[mask]
            open_choice = _condition(
                choices[scenario["donor_open"]] - choices[scenario["recipient_open"]], condition
            )[mask]
            matching_choice = _condition(
                choices[scenario["donor_matching_block"]] - choices[scenario["recipient_matching_block"]], condition
            )[mask]
            raw = {
                "open_transfer": open_transfer,
                "matching_blocked_transfer": matching_transfer,
                "control_blocked_transfer": control_transfer,
                "matching_mediation": open_transfer - matching_transfer,
                "matching_specific_mediation": control_transfer - matching_transfer,
                "open_donor_choice": open_choice,
                "donor_choice_mediation": open_choice - matching_choice,
            }
            letter_rows[condition] = {
                metric: _ci(values, seed + li * 1000 + ci * 100 + mi, draws)
                for mi, (metric, values) in enumerate(raw.items())
            }
        summary["letters"][letter] = letter_rows
    summary["validation"] = {
        "planned_pairs": int(len(valid)),
        "exact_valid_pairs": int(valid.sum()),
        "valid_by_letter": {letter: int(np.sum(letters == letter)) for letter in analyzed_letters},
        "all_source_counts_positive": bool(np.all(arrays["source_position_counts"][:, valid] > 0)),
        "all_matching_counts_positive": bool(np.all(arrays["matching_query_counts"][:, valid] > 0)),
        "all_control_counts_positive": bool(np.all(arrays["control_query_counts"][:, valid] > 0)),
    }
    return summary


def _fmt(row: dict[str, Any], scale: float = 1) -> str:
    if row["mean"] is None: return "n/a"
    return f"{row['mean']*scale:+.3f} [{row['ci'][0]*scale:+.3f}, {row['ci'][1]*scale:+.3f}]"


def plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    letters = summary["design"]["letters"]
    fig, axes = plt.subplots(2, len(letters), figsize=(4.7 * len(letters), 8), constrained_layout=True, sharey=True, squeeze=False)
    for row_index, split in enumerate(("discovery", "confirmation")):
        for column, letter in enumerate(letters):
            axis = axes[row_index, column]
            x = np.arange(3)
            for offset, condition, color in ((-.12, "Game", "#2878b5"), (.12, "Neutral", "#e07a2f")):
                rows = summary[split]["letters"].get(letter, {}).get(condition)
                if not rows or rows["open_transfer"]["mean"] is None: continue
                metrics = ("open_transfer", "matching_blocked_transfer", "control_blocked_transfer")
                means = np.array([rows[m]["mean"] for m in metrics])
                cis = np.array([rows[m]["ci"] for m in metrics])
                axis.errorbar(x + offset, means, yerr=np.vstack((means-cis[:,0], cis[:,1]-means)), fmt="o", capsize=3, color=color, label=condition)
            axis.axhline(0, color="#888", lw=1)
            axis.set_xticks(x, ("Open", "Match blocked", "Control blocked"), rotation=18, ha="right")
            axis.set_title(f"{split.title()}: fixed {letter}")
            axis.grid(axis="y", alpha=.18)
    axes[0,0].legend(frameon=False); axes[0,0].set_ylabel("Donor-semantic margin transfer")
    axes[1,0].set_ylabel("Donor-semantic margin transfer")
    fig.suptitle("Selected-line semantic transfer and matching-line mediation")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--transplant-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8202026)
    args = parser.parse_args()
    transplant = json.loads(args.transplant_summary.read_text())
    summary = {
        "design": {
            "letters": transplant.get("letters", sorted(transplant["discovery"]["letters"])),
            "gated_letters": transplant["gated_letters_for_mediation"],
            "positive_transfer": "movement toward donor semantic answer",
            "matching_specific_mediation": "nonmatching-control-blocked transfer minus matching-blocked transfer",
        },
        "transplant": transplant,
        "discovery": summarize(args.discovery, args.seed, args.draws),
        "confirmation": summarize(args.confirmation, args.seed + 1, args.draws),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.figure is not None:
        plot(summary, args.figure)
    letters = summary["design"]["letters"]
    label = "/".join(letters)
    lines = [f"# Fixed-{label} selected-line semantic transfer and mediation", ""]
    for split in ("discovery", "confirmation"):
        lines += [f"## {split.title()}", ""]
        for letter in letters:
            for condition in ("Game", "Neutral"):
                row = summary[split]["letters"].get(letter, {}).get(condition)
                if not row or row["open_transfer"]["mean"] is None: continue
                lines.append(
                    f"- Fixed {letter}, {condition}: open {_fmt(row['open_transfer'])}; "
                    f"matching blocked {_fmt(row['matching_blocked_transfer'])}; "
                    f"matching-specific mediation {_fmt(row['matching_specific_mediation'])}; "
                    f"donor-choice transfer {_fmt(row['open_donor_choice'],100)} pp."
                )
        lines += ["", f"Validation: `{json.dumps(summary[split]['validation'], sort_keys=True)}`", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
