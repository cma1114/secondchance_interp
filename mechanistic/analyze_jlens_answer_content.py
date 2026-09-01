from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .answer_emergence_figures import Z_975, macro_mean_and_se
from .data import decision_letter, load_activation_dataset
from .jlens_collect import CONDITIONS, POSITION_CONDITIONS


RANKS = ("Original winner", "Original runner-up", "Original rank 3", "Original rank 4")
CONDITION_LABELS = {
    "baseline": "Baseline",
    "incorrect": "Game",
    "neutral": "Neutral",
    "game_minus_neutral": "Game minus Neutral",
}


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def answer_letter_scores(scores: np.ndarray, layout: list[dict]) -> np.ndarray:
    values = []
    for letter in "ABCD":
        indices = [index for index, row in enumerate(layout) if row["family"] == f"answer_{letter}"]
        if not indices:
            raise ValueError(f"Missing JLens tokens for answer {letter}")
        values.append(_logsumexp(scores[..., indices]))
    return np.stack(values, axis=-1)


def baseline_rank_order(data) -> tuple[np.ndarray, np.ndarray]:
    """Generated Baseline answer first; remaining options ordered by its final logits."""
    prior = []
    for qid in data.question_ids:
        letter = decision_letter(data.metadata[(qid, "baseline")])
        if letter not in "ABCD":
            raise ValueError(f"Non-A-D Baseline output for {qid}: {letter!r}")
        prior.append("ABCD".index(letter))
    prior = np.asarray(prior, dtype=np.int64)
    final_logits = data.condition("baseline")[:, -1]
    order = np.empty((len(prior), 4), dtype=np.int64)
    order[:, 0] = prior
    for index, winner in enumerate(prior):
        others = [option for option in range(4) if option != winner]
        order[index, 1:] = sorted(others, key=lambda option: final_logits[index, option], reverse=True)
    return order, prior


def _output_labels(data, condition: str) -> np.ndarray:
    labels = []
    for qid in data.question_ids:
        letter = decision_letter(data.metadata[(qid, condition)])
        if letter not in "ABCD":
            raise ValueError(f"Non-A-D {condition} output for {qid}: {letter!r}")
        labels.append("ABCD".index(letter))
    return np.asarray(labels, dtype=np.int64)


def _accuracy_trajectories(scores: np.ndarray, target: np.ndarray) -> tuple[list[float], list[float]]:
    prediction = scores.argmax(axis=-1)
    accuracy = np.mean(prediction == target[:, None], axis=0)
    balanced = []
    for layer in range(scores.shape[1]):
        recalls = [np.mean(prediction[target == label, layer] == label) for label in range(4) if np.any(target == label)]
        balanced.append(float(np.mean(recalls)))
    return np.round(accuracy, 4).tolist(), np.round(balanced, 4).tolist()


def _align_and_center(scores: np.ndarray, order: np.ndarray) -> np.ndarray:
    centered = scores.astype(np.float64) - scores.astype(np.float64).mean(axis=-1, keepdims=True)
    return np.take_along_axis(centered, order[:, None, :], axis=-1)


def _summarize(scores: np.ndarray, order: np.ndarray, strata: np.ndarray) -> list[dict]:
    aligned = _align_and_center(scores, order)
    rows = []
    for rank, label in enumerate(RANKS):
        mean, se = macro_mean_and_se(aligned[:, :, rank], strata)
        half = Z_975 * se
        rows.append({
            "rank": label,
            "mean": mean,
            "ci_low": mean - half,
            "ci_high": mean + half,
        })
    return rows


def _json_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "rank": row["rank"],
            "mean": np.round(row["mean"], 4).tolist(),
            "ci_low": np.round(row["ci_low"], 4).tolist(),
            "ci_high": np.round(row["ci_high"], 4).tolist(),
        }
        for row in rows
    ]


def analyze(
    jlens_root: str | Path,
    content_root: str | Path,
    residual_root: str | Path,
    output_root: str | Path,
) -> dict:
    jlens_root = Path(jlens_root)
    content_root = Path(content_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        letter_final = answer_letter_scores(cached["final_scores"].astype(np.float64), layout)
        letter_position = answer_letter_scores(cached["position_scores"].astype(np.float64), layout)
        qids = cached["question_ids"].astype(str).tolist()
        position_qids = cached["position_question_ids"].astype(str).tolist()
        anchors = cached["anchors"].astype(str).tolist()
        availability = cached["position_availability"].astype(bool)
    with np.load(content_root / "option_content_scores.npz", allow_pickle=False) as cached:
        content_final = cached["final_scores"].astype(np.float64)
        content_position = cached["position_scores"].astype(np.float64)
        if cached["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Letter and option-content question orders differ")
        if cached["position_question_ids"].astype(str).tolist() != position_qids:
            raise ValueError("Letter and option-content position samples differ")
        if cached["anchors"].astype(str).tolist() != anchors:
            raise ValueError("Letter and option-content anchors differ")

    data = load_activation_dataset(residual_root, list(CONDITIONS))
    if data.question_ids != qids:
        raise ValueError("JLens scores and activation shards have different question order")
    order, prior = baseline_rank_order(data)
    generated = {condition: _output_labels(data, condition) for condition in CONDITIONS}
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    position_indices = np.asarray([qid_to_index[qid] for qid in position_qids], dtype=np.int64)
    position_order = order[position_indices]
    position_prior = prior[position_indices]

    payload = {
        "layers": list(range(1, 65)),
        "ranks": list(RANKS),
        "anchors": anchors,
        "conditions": CONDITION_LABELS,
        "readouts": {},
        "notes": {
            "ranking": (
                "Rank 1 is the actually generated Baseline answer. The other three are ordered by their "
                "final Baseline A-D logits. This fixed ranking is used in every condition and layer."
            ),
            "letter": "Log-sum-exp over the valid bare and leading-space token variants for A, B, C, and D.",
            "content": (
                "Mean JLens vocabulary score over question-specific, option-distinctive alphanumeric tokens "
                "from each answer text."
            ),
            "centering": "Each trial is centered across its four options before rank alignment and averaging.",
            "ci": "95% normal intervals for an equal-weight macro-average over original-answer letters.",
        },
    }
    csv_rows = []
    final_sources = {"letter": letter_final, "content": content_final}
    position_sources = {"letter": letter_position, "content": content_position}
    condition_index = {condition: index for index, condition in enumerate(CONDITIONS)}

    for readout in ("letter", "content"):
        payload["readouts"][readout] = {}
        for anchor_index, anchor in enumerate(anchors):
            payload["readouts"][readout][anchor] = {}
            conditions = ["incorrect", "neutral", "game_minus_neutral"]
            if anchor == "decision":
                conditions.insert(0, "baseline")
            for condition in conditions:
                if condition in CONDITION_LABELS and condition not in ("baseline", "game_minus_neutral"):
                    ci = POSITION_CONDITIONS.index(condition)
                    if anchor != "decision" and not availability[ci, anchor_index]:
                        continue
                if condition == "baseline":
                    scores = final_sources[readout][condition_index["baseline"]]
                    local_order, local_prior, n = order, prior, len(qids)
                    target = generated["baseline"]
                elif condition == "game_minus_neutral":
                    if anchor == "decision":
                        scores = (
                            final_sources[readout][condition_index["incorrect"]]
                            - final_sources[readout][condition_index["neutral"]]
                        )
                        local_order, local_prior, n = order, prior, len(qids)
                    else:
                        if not availability[:, anchor_index].all():
                            continue
                        scores = position_sources[readout][0, :, anchor_index] - position_sources[readout][1, :, anchor_index]
                        local_order, local_prior, n = position_order, position_prior, len(position_qids)
                    target = None
                elif anchor == "decision":
                    scores = final_sources[readout][condition_index[condition]]
                    local_order, local_prior, n = order, prior, len(qids)
                    target = generated[condition]
                else:
                    ci = POSITION_CONDITIONS.index(condition)
                    scores = position_sources[readout][ci, :, anchor_index]
                    local_order, local_prior, n = position_order, position_prior, len(position_qids)
                    target = generated[condition][position_indices]

                rows = _summarize(scores, local_order, local_prior)
                entry = {"n": n, "series": _json_rows(rows)}
                if target is not None:
                    accuracy, balanced = _accuracy_trajectories(scores, target)
                    entry["accuracy_vs_condition_output"] = accuracy
                    entry["balanced_accuracy_vs_condition_output"] = balanced
                payload["readouts"][readout][anchor][condition] = entry
                for row in rows:
                    for layer, mean, low, high in zip(
                        payload["layers"], row["mean"], row["ci_low"], row["ci_high"]
                    ):
                        csv_rows.append({
                            "readout": readout,
                            "anchor": anchor,
                            "condition": condition,
                            "rank": row["rank"],
                            "layer": layer,
                            "mean": float(mean),
                            "ci_low": float(low),
                            "ci_high": float(high),
                            "n": n,
                        })

    (output / "answer_representation_trajectories.json").write_text(json.dumps(payload, separators=(",", ":")))
    with (output / "answer_representation_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    diagnostics = {}
    for readout in ("letter", "content"):
        diagnostics[readout] = {}
        for condition in ("baseline", "incorrect", "neutral", "game_minus_neutral"):
            row = payload["readouts"][readout]["decision"][condition]
            diagnostics[readout][condition] = {
                "n": row["n"],
                "layer_48": {series["rank"]: series["mean"][47] for series in row["series"]},
                "layer_56": {series["rank"]: series["mean"][55] for series in row["series"]},
                "layer_64": {series["rank"]: series["mean"][63] for series in row["series"]},
            }
            if "balanced_accuracy_vs_condition_output" in row:
                diagnostics[readout][condition]["balanced_accuracy_layer_64"] = row[
                    "balanced_accuracy_vs_condition_output"
                ][63]
    (output / "answer_representation_summary.json").write_text(json.dumps(diagnostics, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rank-aligned JLens letter and option-content readouts")
    parser.add_argument("--jlens-root", required=True)
    parser.add_argument("--content-root", required=True)
    parser.add_argument("--residual-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.jlens_root, args.content_root, args.residual_root, args.output)


if __name__ == "__main__":
    main()
