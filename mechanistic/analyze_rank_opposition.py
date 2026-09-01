from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_jlens_answer_content import answer_letter_scores, baseline_rank_order
from .answer_emergence_figures import Z_975, macro_mean_and_se
from .component_causal_metrics import aggregate_mean, bootstrap, center
from .data import load_activation_dataset
from .io import read_metadata, shard_path


RANK_AXIS = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
RANK_AXIS_DENOMINATOR = float(np.sum(RANK_AXIS**2))
CONDITIONS = {"incorrect": "Game", "neutral": "Neutral"}


def _macro_weights(labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in range(4):
        count = int(np.sum(labels == label))
        if count == 0:
            raise ValueError(f"No trials with Baseline answer {'ABCD'[label]}")
        weights[labels == label] = 1.0 / (4.0 * count)
    return weights


def _macro_energy(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Equal-letter mean of per-trial squared L2 energy, preserving other axes."""
    energy = np.sum(values**2, axis=-1)
    return np.mean(
        np.stack([energy[labels == label].mean(axis=0) for label in range(4)]),
        axis=0,
    )


def _rank_slope(aligned_delta: np.ndarray) -> np.ndarray:
    return np.sum(aligned_delta * RANK_AXIS, axis=-1) / RANK_AXIS_DENOMINATOR


def _rank_r2(aligned_delta: np.ndarray, labels: np.ndarray) -> np.ndarray:
    slope = _rank_slope(aligned_delta)
    fitted = slope[..., None] * RANK_AXIS
    denominator = np.maximum(_macro_energy(aligned_delta, labels), 1e-12)
    return 1.0 - _macro_energy(aligned_delta - fitted, labels) / denominator


def _mean_ci(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, se = macro_mean_and_se(values, labels)
    return mean, mean - Z_975 * se, mean + Z_975 * se


def analyze_jlens(jlens_root: Path, residual_root: Path) -> dict:
    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        scores = answer_letter_scores(cached["final_scores"].astype(np.float64), layout)
        qids = cached["question_ids"].astype(str).tolist()
        conditions = cached["conditions"].astype(str).tolist()
    scores = center(scores)

    data = load_activation_dataset(residual_root, ["baseline", "incorrect", "neutral"])
    if data.question_ids != qids:
        raise ValueError("JLens answer-letter and activation question orders differ")
    order, prior = baseline_rank_order(data)
    weights = _macro_weights(prior)
    baseline = scores[conditions.index("baseline")]

    payload = {
        "layers": list(range(1, scores.shape[2] + 1)),
        "n_questions": len(qids),
        "rank_axis": RANK_AXIS.tolist(),
        "conditions": {},
    }
    for condition, label in CONDITIONS.items():
        values = scores[conditions.index(condition)]
        delta = values - baseline
        aligned = np.take_along_axis(delta, order[:, None, :], axis=-1)
        slope = _rank_slope(aligned)
        cosine = np.sum(aligned * RANK_AXIS, axis=-1) / np.maximum(
            np.linalg.norm(aligned, axis=-1) * np.linalg.norm(RANK_AXIS), 1e-12
        )
        positive = (slope > 0).astype(np.float64)
        monotone = np.all(np.diff(aligned, axis=-1) >= 0, axis=-1).astype(np.float64)

        baseline_energy = np.sum(baseline**2, axis=-1)
        delta_dot_baseline = np.sum(delta * baseline, axis=-1)
        numerator = np.sum(weights[:, None] * delta_dot_baseline, axis=0)
        denominator = np.maximum(np.sum(weights[:, None] * baseline_energy, axis=0), 1e-12)
        opposition = -numerator / denominator
        residual = delta + opposition[None, :, None] * baseline
        delta_energy = np.maximum(_macro_energy(delta, prior), 1e-12)
        baseline_opposition_r2 = 1.0 - _macro_energy(residual, prior) / delta_energy

        entry = {
            "label": label,
            "rank_axis_r2": np.round(_rank_r2(aligned, prior), 6).tolist(),
            "baseline_opposition_coefficient": np.round(opposition, 6).tolist(),
            "baseline_opposition_r2": np.round(baseline_opposition_r2, 6).tolist(),
        }
        for name, metric in {
            "rank_opposed_slope": slope,
            "rank_axis_cosine": cosine,
            "fraction_rank_opposed": positive,
            "fraction_strictly_monotone": monotone,
        }.items():
            mean, low, high = _mean_ci(metric, prior)
            entry[name] = {
                "mean": np.round(mean, 6).tolist(),
                "ci_low": np.round(low, 6).tolist(),
                "ci_high": np.round(high, 6).tolist(),
            }
        payload["conditions"][condition] = entry

    delta = scores[conditions.index("incorrect")] - scores[conditions.index("neutral")]
    aligned = np.take_along_axis(delta, order[:, None, :], axis=-1)
    slope = _rank_slope(aligned)
    cosine = np.sum(aligned * RANK_AXIS, axis=-1) / np.maximum(
        np.linalg.norm(aligned, axis=-1) * np.linalg.norm(RANK_AXIS), 1e-12
    )
    baseline_energy = np.sum(baseline**2, axis=-1)
    numerator = np.sum(weights[:, None] * np.sum(delta * baseline, axis=-1), axis=0)
    denominator = np.maximum(np.sum(weights[:, None] * baseline_energy, axis=0), 1e-12)
    opposition = -numerator / denominator
    residual = delta + opposition[None, :, None] * baseline
    entry = {
        "label": "Game minus Neutral",
        "rank_axis_r2": np.round(_rank_r2(aligned, prior), 6).tolist(),
        "baseline_opposition_coefficient": np.round(opposition, 6).tolist(),
        "baseline_opposition_r2": np.round(
            1.0 - _macro_energy(residual, prior) / np.maximum(_macro_energy(delta, prior), 1e-12), 6
        ).tolist(),
    }
    for name, metric in {
        "rank_opposed_slope": slope,
        "rank_axis_cosine": cosine,
        "fraction_rank_opposed": (slope > 0).astype(np.float64),
        "fraction_strictly_monotone": np.all(np.diff(aligned, axis=-1) >= 0, axis=-1).astype(np.float64),
    }.items():
        mean, low, high = _mean_ci(metric, prior)
        entry[name] = {
            "mean": np.round(mean, 6).tolist(),
            "ci_low": np.round(low, 6).tolist(),
            "ci_high": np.round(high, 6).tolist(),
        }
    payload["conditions"]["game_minus_neutral"] = entry
    return payload


def _load(root: Path, group: str, qids: list[str], key: str = "final_canonical_logits") -> np.ndarray:
    return np.asarray(
        [np.load(shard_path(root, group, qid), allow_pickle=False)[key] for qid in qids],
        dtype=np.float64,
    )


def _baseline_order(ranking_root: Path, qids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    order = np.empty((len(qids), 4), dtype=np.int64)
    winners = np.empty(len(qids), dtype=np.int64)
    for index, qid in enumerate(qids):
        with np.load(shard_path(ranking_root, "baseline", qid), allow_pickle=False) as cached:
            metadata = read_metadata(cached)
            rank_logits = cached["canonical_logits"][-1].astype(np.float64)
        # Match the exact self-hosted residual run used by the JLens figure.
        answer = metadata.get("full_vocab_top_token", "").strip()
        if answer not in "ABCD":
            answer = "ABCD"[int(np.argmax(rank_logits))]
        if answer not in "ABCD":
            raise ValueError(f"Missing generated Baseline answer for {qid}: {answer!r}")
        winner = "ABCD".index(answer)
        winners[index] = winner
        others = [option for option in range(4) if option != winner]
        others.sort(key=lambda option: rank_logits[option], reverse=True)
        order[index] = [winner, *others]
    return order, winners


def _final_rank_opposition(values: np.ndarray, baseline: np.ndarray, order: np.ndarray) -> np.ndarray:
    delta = center(values) - center(baseline)
    aligned = np.take_along_axis(delta, order, axis=-1)
    return _rank_slope(aligned)


def analyze_causal_split(
    natural_root: Path,
    ranking_root: Path,
    patch_root: Path,
    plan_path: Path,
    samples: int,
    seed: int,
) -> dict:
    plan = json.loads(plan_path.read_text())
    planned_qids = plan.get("question_ids", plan.get("confirmation_question_ids", []))
    qids = [
        qid
        for qid in planned_qids
        if all(
            shard_path(patch_root, group, qid).exists()
            for group in ["natural_game", "natural_neutral", *[row["id"] for row in plan["scenarios"]]]
        )
    ]
    if not qids:
        raise FileNotFoundError(f"No complete causal questions for {plan_path}")

    baseline = _load(natural_root, "baseline", qids)
    natural = {
        "incorrect": _load(patch_root, "natural_game", qids),
        "neutral": _load(patch_root, "natural_neutral", qids),
    }
    order, winners = _baseline_order(ranking_root, qids)
    natural_opposition = {
        condition: _final_rank_opposition(values, baseline, order)
        for condition, values in natural.items()
    }
    natural_gap = natural_opposition["incorrect"] - natural_opposition["neutral"]
    natural_gap_aligned = np.take_along_axis(
        center(natural["incorrect"]) - center(natural["neutral"]), order, axis=-1
    )
    natural_gap_vector = np.mean(
        np.stack([natural_gap_aligned[winners == label].mean(axis=0) for label in range(4)]),
        axis=0,
    )
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in plan["scenarios"]:
        values = _load(patch_root, scenario["id"], qids)
        target = scenario["target_condition"]
        direction = "neutral_into_game" if target == "incorrect" else "game_into_neutral"
        sign = -1.0 if direction == "neutral_into_game" else 1.0
        patched_opposition = _final_rank_opposition(values, baseline, order)
        mediation = sign * (patched_opposition - natural_opposition[target])

        natural_centered = center(natural[target])
        patch_centered = center(values)
        game_like_write = sign * (patch_centered - natural_centered)
        aligned_write = np.take_along_axis(game_like_write, order, axis=-1)
        target_info = scenario["targets"][0] if len(scenario["targets"]) == 1 else None
        entry = {
            "scenario": scenario["id"],
            "direction": direction,
            "component": None if target_info is None else target_info["component"],
            "kind": None if target_info is None else target_info["kind"],
            "layer": None if target_info is None else target_info["layer"],
            "n_targets": len(scenario["targets"]),
            "rank_writes": [],
        }
        for aggregation in ("dataset", "letter_macro"):
            mean, low, high = bootstrap(mediation, winners, aggregation, samples, rng)
            gap = aggregate_mean(natural_gap, winners, aggregation)
            entry[aggregation] = {
                "mediation_mean": mean,
                "mediation_ci_low": low,
                "mediation_ci_high": high,
                "natural_game_minus_neutral_gap": gap,
                "fraction_gap_mediated": mean / gap if abs(gap) > 1e-12 else None,
            }
        for rank in range(4):
            mean, low, high = bootstrap(aligned_write[:, rank], winners, "letter_macro", samples, rng)
            entry["rank_writes"].append({
                "rank": rank + 1,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            })
        write_vector = np.asarray([row["mean"] for row in entry["rank_writes"]])
        gap_norm = float(np.linalg.norm(natural_gap_vector))
        write_norm = float(np.linalg.norm(write_vector))
        entry["rank_vector_geometry"] = {
            "natural_gap_vector": natural_gap_vector.tolist(),
            "game_like_write_vector": write_vector.tolist(),
            "cosine_with_natural_gap": (
                float(np.dot(write_vector, natural_gap_vector) / (write_norm * gap_norm))
                if write_norm > 1e-12 and gap_norm > 1e-12 else None
            ),
            "norm_ratio": write_norm / gap_norm if gap_norm > 1e-12 else None,
            "gap_vector_variance_removed": (
                1.0 - float(np.sum((natural_gap_vector - write_vector) ** 2)) / float(np.sum(natural_gap_vector**2))
                if gap_norm > 1e-12 else None
            ),
            "by_baseline_answer": {},
        }
        for label in range(4):
            mask = winners == label
            letter_gap = natural_gap_aligned[mask].mean(axis=0)
            letter_write = aligned_write[mask].mean(axis=0)
            letter_gap_norm = float(np.linalg.norm(letter_gap))
            letter_write_norm = float(np.linalg.norm(letter_write))
            gap_slope = float(_rank_slope(letter_gap))
            write_slope = float(_rank_slope(letter_write))
            entry["rank_vector_geometry"]["by_baseline_answer"]["ABCD"[label]] = {
                "n": int(mask.sum()),
                "natural_gap_vector": letter_gap.tolist(),
                "game_like_write_vector": letter_write.tolist(),
                "slope_fraction_mediated": write_slope / gap_slope if abs(gap_slope) > 1e-12 else None,
                "cosine_with_natural_gap": (
                    float(np.dot(letter_write, letter_gap) / (letter_write_norm * letter_gap_norm))
                    if letter_write_norm > 1e-12 and letter_gap_norm > 1e-12 else None
                ),
                "gap_vector_variance_removed": (
                    1.0 - float(np.sum((letter_gap - letter_write) ** 2)) / float(np.sum(letter_gap**2))
                    if letter_gap_norm > 1e-12 else None
                ),
            }
        rows.append(entry)

    return {
        "n_questions": len(qids),
        "planned_questions": len(planned_qids),
        "baseline_winner_counts": {"ABCD"[label]: int(np.sum(winners == label)) for label in range(4)},
        "natural": {
            condition: {
                aggregation: aggregate_mean(values, winners, aggregation)
                for aggregation in ("dataset", "letter_macro")
            }
            for condition, values in natural_opposition.items()
        },
        "natural_gap": {
            aggregation: aggregate_mean(natural_gap, winners, aggregation)
            for aggregation in ("dataset", "letter_macro")
        },
        "natural_gap_rank_vector": natural_gap_vector.tolist(),
        "scenarios": rows,
    }


def _write_csv(payload: dict, output: Path) -> None:
    rows = []
    for condition, entry in payload["jlens"]["conditions"].items():
        for index, layer in enumerate(payload["jlens"]["layers"]):
            row = {"section": "jlens", "condition": condition, "layer": layer}
            for metric in (
                "rank_opposed_slope", "rank_axis_cosine", "fraction_rank_opposed",
                "fraction_strictly_monotone",
            ):
                row[metric] = entry[metric]["mean"][index]
                row[f"{metric}_ci_low"] = entry[metric]["ci_low"][index]
                row[f"{metric}_ci_high"] = entry[metric]["ci_high"][index]
            row["rank_axis_r2"] = entry["rank_axis_r2"][index]
            row["baseline_opposition_coefficient"] = entry["baseline_opposition_coefficient"][index]
            row["baseline_opposition_r2"] = entry["baseline_opposition_r2"][index]
            rows.append(row)
    fieldnames = list(rows[0])
    with (output / "jlens_rank_opposition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    causal_rows = []
    for split in payload["causal"]:
        for entry in payload["causal"][split]["scenarios"]:
            for aggregation in ("dataset", "letter_macro"):
                causal_rows.append({
                    "split": split,
                    "scenario": entry["scenario"],
                    "direction": entry["direction"],
                    "component": entry["component"],
                    "kind": entry["kind"],
                    "layer": entry["layer"],
                    "n_targets": entry["n_targets"],
                    "aggregation": aggregation,
                    **entry[aggregation],
                })
    with (output / "causal_rank_opposition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(causal_rows[0]))
        writer.writeheader()
        writer.writerows(causal_rows)


def _write_report(payload: dict, output: Path) -> None:
    jlens = payload["jlens"]
    lines = [
        "# Rank-opposed Second Chance transformation",
        "",
        "The rank-opposed slope projects each same-question, centered condition-minus-Baseline four-option vector onto `[-1.5, -0.5, +0.5, +1.5]`, after fixing ranks from the generated Baseline answer and final Baseline logits. Positive values mean progressively less suppression or more boosting from Baseline rank 1 through rank 4.",
        "",
        "## JLens answer-letter trajectory",
        "",
        "| Layer | Game slope | Neutral slope | Game - Neutral slope | Game trial fraction > 0 | Neutral trial fraction > 0 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in (40, 44, 48, 52, 56, 60, 64):
        index = layer - 1
        game = jlens["conditions"]["incorrect"]
        neutral = jlens["conditions"]["neutral"]
        contrast = jlens["conditions"]["game_minus_neutral"]
        lines.append(
            f"| {layer} | {game['rank_opposed_slope']['mean'][index]:+.4f} | "
            f"{neutral['rank_opposed_slope']['mean'][index]:+.4f} | "
            f"{contrast['rank_opposed_slope']['mean'][index]:+.4f} | "
            f"{game['fraction_rank_opposed']['mean'][index]:.3f} | "
            f"{neutral['fraction_rank_opposed']['mean'][index]:.3f} |"
        )
    lines.extend([
        "",
        "The rank-axis R2 asks how much of the full four-option change energy lies on this single ordered axis. The Baseline-opposition coefficient instead fits the full question-specific same-layer Baseline JLens vector; it tests a stronger negative-feedback claim than the rank-average plot alone.",
        "",
        "## Existing causal component sweep",
        "",
    ])
    for split in payload["causal"]:
        causal = payload["causal"][split]
        individual = [row for row in causal["scenarios"] if row["n_targets"] == 1]
        individual.sort(key=lambda row: row["letter_macro"]["mediation_mean"], reverse=True)
        lines.extend([
            f"### {split.title()} ({causal['n_questions']} questions)",
            "",
            f"Natural Game-minus-Neutral rank-opposition gap: {causal['natural_gap']['letter_macro']:+.4f} (equal-letter macro-average).",
            "",
            "| Component | Direction | Mediated slope | Fraction of natural gap |",
            "|---|---|---:|---:|",
        ])
        for row in individual[:12]:
            metric = row["letter_macro"]
            lines.append(
                f"| {row['component']} | {row['direction']} | "
                f"{metric['mediation_mean']:+.4f} [{metric['mediation_ci_low']:+.4f}, {metric['mediation_ci_high']:+.4f}] | "
                f"{metric['fraction_gap_mediated']:.1%} |"
            )
        lines.append("")
        groups = [row for row in causal["scenarios"] if row["n_targets"] > 1]
        if groups:
            lines.extend([
                "| Joint patch | Direction | Fraction of slope gap | Cosine with natural rank vector | Rank-vector variance removed |",
                "|---|---|---:|---:|---:|",
            ])
            for row in groups:
                geometry = row["rank_vector_geometry"]
                lines.append(
                    f"| {row['scenario'].split('__', 1)[1]} | {row['direction']} | "
                    f"{row['letter_macro']['fraction_gap_mediated']:.1%} | "
                    f"{geometry['cosine_with_natural_gap']:.3f} | "
                    f"{geometry['gap_vector_variance_removed']:.1%} |"
                )
            lines.append("")
    lines.extend([
        "## Interpretation limits",
        "",
        "The JLens trajectory is observational and guides localization. The component table is causal for final-position component replacement, but discovery-only components require held-out reciprocal confirmation before being treated as established mediators.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def analyze(args: argparse.Namespace) -> dict:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    causal = {
        "discovery": analyze_causal_split(
            args.natural_root, args.residual_root, args.discovery_patch_root, args.discovery_plan,
            args.bootstrap_samples, args.seed,
        ),
        "confirmation": analyze_causal_split(
            args.natural_root, args.residual_root, args.confirmation_patch_root, args.confirmation_plan,
            args.bootstrap_samples, args.seed + 1,
        ),
    }
    if args.rank_confirmation_patch_root is not None or args.rank_confirmation_plan is not None:
        if args.rank_confirmation_patch_root is None or args.rank_confirmation_plan is None:
            raise ValueError("Both rank-confirmation arguments must be supplied together")
        causal["rank_confirmation"] = analyze_causal_split(
            args.natural_root, args.residual_root, args.rank_confirmation_patch_root, args.rank_confirmation_plan,
            args.bootstrap_samples, args.seed + 2,
        )
    payload = {
        "definition": {
            "rank_axis": RANK_AXIS.tolist(),
            "rank_opposed_slope": "Projection of the centered condition-minus-Baseline vector onto the fixed Baseline-rank axis.",
            "positive_direction": "More negative change for Baseline rank 1 than rank 2 than rank 3 than rank 4.",
            "causal_mediation": "Game-like rank-opposition removed by Neutral-into-Game patch or induced by Game-into-Neutral patch.",
        },
        "jlens": analyze_jlens(args.jlens_root, args.residual_root),
        "causal": causal,
    }
    (output / "rank_opposition.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_csv(payload, output)
    _write_report(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the rank-opposed Second Chance transformation")
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument("--discovery-patch-root", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--confirmation-patch-root", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--rank-confirmation-patch-root", type=Path)
    parser.add_argument("--rank-confirmation-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
