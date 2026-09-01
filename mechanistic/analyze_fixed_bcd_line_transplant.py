from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCENARIOS = ("recipient_open", "donor_open")


def _load(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "results.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"].astype(bool)):
        raise RuntimeError(f"Incomplete results: {root}")
    if arrays["scenarios"].astype(str).tolist() != list(SCENARIOS):
        raise RuntimeError("Unexpected transplant scenarios")
    return arrays


def _oriented_margin(logits: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.arange(len(x))
    result = np.empty((4, len(x)), dtype=float)
    result[0] = logits[0, q, y] - logits[0, q, x]
    result[1] = logits[1, q, y] - logits[1, q, x]
    result[2] = logits[2, q, x] - logits[2, q, y]
    result[3] = logits[3, q, x] - logits[3, q, y]
    return result


def _oriented_choice(logits: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    answer = logits.argmax(axis=-1)
    result = np.empty((4, len(x)), dtype=float)
    result[0] = answer[0] == y
    result[1] = answer[1] == y
    result[2] = answer[2] == x
    result[3] = answer[3] == x
    return result


def _condition(values: np.ndarray, name: str) -> np.ndarray:
    rows = (0, 2) if name == "Game" else (1, 3)
    return 0.5 * (values[rows[0]] + values[rows[1]])


def _interval(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"n": int(len(values)), "mean": float(values.mean()), "ci": np.quantile(means, (.025, .975)).tolist()}


def _ordered_letters(values: np.ndarray) -> str:
    present = set(values.astype(str).tolist())
    return "".join(letter for letter in "ABCD" if letter in present)


def _split(root: Path, seed: int, draws: int) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    arrays = _load(root)
    valid = arrays["first_decision_valid"].astype(bool)
    logits = arrays["scenario_logits"][:, :, valid].astype(float)
    letters = arrays["literal_letters"][valid].astype(str)
    x = np.array(["ABCD".index(value) for value in arrays["x_second_letter"][valid].astype(str)])
    y = np.array(["ABCD".index(value) for value in arrays["y_second_letter"][valid].astype(str)])
    margins = np.stack([_oriented_margin(row, x, y) for row in logits])
    choices = np.stack([_oriented_choice(row, x, y) for row in logits])
    transfer_margin = margins[1] - margins[0]
    transfer_choice = choices[1] - choices[0]
    raw: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, Any] = {"letters": {}, "validation": {}}
    analyzed_letters = _ordered_letters(letters)
    for li, letter in enumerate(analyzed_letters):
        mask = letters == letter
        rows: dict[str, np.ndarray] = {}
        for condition in ("Game", "Neutral"):
            rows[condition] = _condition(transfer_margin[:, mask], condition)
            rows[f"{condition}_choice"] = _condition(transfer_choice[:, mask], condition)
        rows["Pooled"] = 0.5 * (rows["Game"] + rows["Neutral"])
        rows["Game_minus_Neutral"] = rows["Game"] - rows["Neutral"]
        raw[letter] = rows
        summary["letters"][letter] = {
            "Game_margin": _interval(rows["Game"], seed + li * 100, draws),
            "Neutral_margin": _interval(rows["Neutral"], seed + li * 100 + 1, draws),
            "Pooled_margin": _interval(rows["Pooled"], seed + li * 100 + 2, draws),
            "Game_minus_Neutral_margin": _interval(rows["Game_minus_Neutral"], seed + li * 100 + 3, draws),
            "Game_donor_choice": _interval(rows["Game_choice"], seed + li * 100 + 4, draws),
            "Neutral_donor_choice": _interval(rows["Neutral_choice"], seed + li * 100 + 5, draws),
        }
    eligible_errors = arrays["cached_identity_max_abs_error"][valid]
    summary["validation"] = {
        "planned_pairs": int(len(valid)),
        "exact_valid_pairs": int(valid.sum()),
        "valid_by_letter": {letter: int(np.sum(letters == letter)) for letter in analyzed_letters},
        "all_source_counts_positive": bool(np.all(arrays["source_position_counts"][:, valid] > 0)),
        "max_cached_identity_error": float(np.nanmax(eligible_errors)) if len(eligible_errors) else None,
        "mean_cached_identity_error": float(np.nanmean(eligible_errors)) if len(eligible_errors) else None,
    }
    return summary, raw


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    if row["mean"] is None:
        return "n/a"
    return f"{row['mean']*scale:+.3f} [{row['ci'][0]*scale:+.3f}, {row['ci'][1]*scale:+.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8202026)
    args = parser.parse_args()
    discovery, _ = _split(args.discovery, args.seed, args.draws)
    confirmation, _ = _split(args.confirmation, args.seed + 1, args.draws)
    analyzed_letters = "".join(
        letter for letter in "ABCD"
        if letter in discovery["letters"] or letter in confirmation["letters"]
    )
    gated = [
        letter for letter in analyzed_letters
        if discovery["letters"][letter]["Pooled_margin"]["mean"] is not None
        and discovery["letters"][letter]["Pooled_margin"]["mean"] > 0
    ]
    summary = {
        "design": {
            "positive": "transplant moves the final decision toward donor semantic content",
            "discovery_gate": "positive pooled Game/Neutral donor-semantic margin transfer",
        },
        "letters": list(analyzed_letters),
        "gated_letters_for_mediation": gated,
        "discovery": discovery,
        "confirmation": confirmation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "mediation_gate.json").write_text(json.dumps({"letters": gated}, indent=2) + "\n")
    label = "/".join(analyzed_letters)
    lines = [f"# Fixed-{label} whole-line semantic transfer", ""]
    for split in ("discovery", "confirmation"):
        lines += [f"## {split.title()}", ""]
        rows = summary[split]["letters"]
        for letter in analyzed_letters:
            lines.append(
                f"- {letter}: Game margin {_fmt(rows[letter]['Game_margin'])}; "
                f"Neutral {_fmt(rows[letter]['Neutral_margin'])}; pooled {_fmt(rows[letter]['Pooled_margin'])}; "
                f"Game donor choice {_fmt(rows[letter]['Game_donor_choice'], 100)} pp; "
                f"Neutral {_fmt(rows[letter]['Neutral_donor_choice'], 100)} pp."
            )
        lines += ["", f"Validation: `{json.dumps(summary[split]['validation'], sort_keys=True)}`", ""]
    lines += ["## Mediation gate", "", f"Letters passing discovery: {', '.join(gated) if gated else 'none'}.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
