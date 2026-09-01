from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .analyze_jlens_answer_content import answer_letter_scores, baseline_rank_order
from .component_causal_metrics import bootstrap, center
from .data import load_activation_dataset
from .io import shard_path


def _load(root: Path, group: str, qids: list[str]) -> np.ndarray:
    return np.asarray([
        np.load(shard_path(root, group, qid), allow_pickle=False)["final_canonical_logits"]
        for qid in qids
    ], dtype=np.float64)


def analyze(
    jlens_root: Path,
    residual_root: Path,
    sublayer_root: Path,
    patch_root: Path,
    plan_path: Path,
    output: Path,
    samples: int,
    seed: int,
) -> dict:
    plan = json.loads(plan_path.read_text())
    individual = [row for row in plan["scenarios"] if len(row["targets"]) == 1]
    planned = plan.get("question_ids", plan.get("confirmation_question_ids", []))
    qids = [
        qid for qid in planned
        if all(shard_path(patch_root, group, qid).exists() for group in [
            "natural_game", "natural_neutral", *[row["id"] for row in individual]
        ])
    ]
    if not qids:
        raise FileNotFoundError("No complete individual-component questions")

    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        all_qids = cached["question_ids"].astype(str).tolist()
        conditions = cached["conditions"].astype(str).tolist()
        scores = answer_letter_scores(cached["final_scores"].astype(np.float64), layout)
    qid_index = {qid: index for index, qid in enumerate(all_qids)}
    indices = np.asarray([qid_index[qid] for qid in qids], dtype=np.int64)
    game_scores = scores[conditions.index("incorrect"), indices]

    data = load_activation_dataset(residual_root, ["baseline", "incorrect", "neutral"])
    _order, all_winners = baseline_rank_order(data)
    data_index = {qid: index for index, qid in enumerate(data.question_ids)}
    winners = all_winners[np.asarray([data_index[qid] for qid in qids], dtype=np.int64)]
    natural = {
        "incorrect": _load(patch_root, "natural_game", qids),
        "neutral": _load(patch_root, "natural_neutral", qids),
    }
    game_boundaries = np.asarray([
        np.load(shard_path(sublayer_root, "incorrect", qid), allow_pickle=False)["boundary_canonical_logits"]
        for qid in qids
    ], dtype=np.float64)

    rng = np.random.default_rng(seed)
    rows = []
    for scenario in individual:
        target = scenario["target_condition"]
        direction = "neutral_into_game" if target == "incorrect" else "game_into_neutral"
        sign = -1.0 if direction == "neutral_into_game" else 1.0
        intervened = _load(patch_root, scenario["id"], qids)
        game_like_write = sign * (center(intervened) - center(natural[target]))
        component = scenario["targets"][0]
        layer = int(component["layer"])
        incoming_index = max(layer - 1, 0)
        incoming = game_scores[:, incoming_index]
        leader = np.argmax(incoming, axis=-1)
        ordered = np.sort(incoming, axis=-1)
        margin = ordered[:, -1] - ordered[:, -2]
        leader_write = game_like_write[np.arange(len(qids)), leader]
        suppression = -leader_write
        other_write = (game_like_write.sum(axis=-1) - leader_write) / 3.0
        mean, low, high = bootstrap(suppression, winners, "letter_macro", samples, rng)
        fraction, fraction_low, fraction_high = bootstrap(
            (suppression > 0).astype(np.float64), winners, "letter_macro", samples, rng
        )
        cuts = np.quantile(margin, [0.25, 0.5, 0.75])
        bins = np.digitize(margin, cuts)
        quartiles = []
        for index in range(4):
            mask = bins == index
            quartiles.append({
                "quartile": index + 1,
                "n": int(mask.sum()),
                "margin_mean": float(margin[mask].mean()),
                "suppression_mean": float(suppression[mask].mean()),
                "fraction_suppressed": float(np.mean(suppression[mask] > 0)),
            })
        boundary_index = 0 if component["kind"] == "mixer" else 1
        logit_incoming = game_boundaries[:, layer, boundary_index]
        logit_leader = np.argmax(logit_incoming, axis=-1)
        logit_ordered = np.sort(logit_incoming, axis=-1)
        logit_margin = logit_ordered[:, -1] - logit_ordered[:, -2]
        logit_suppression = -game_like_write[np.arange(len(qids)), logit_leader]
        logit_mean, logit_low, logit_high = bootstrap(
            logit_suppression, winners, "letter_macro", samples, rng
        )
        logit_fraction, logit_fraction_low, logit_fraction_high = bootstrap(
            (logit_suppression > 0).astype(np.float64), winners, "letter_macro", samples, rng
        )
        rows.append({
            "scenario": scenario["id"],
            "direction": direction,
            "component": component["component"],
            "kind": component["kind"],
            "zero_indexed_layer": layer,
            "incoming_jlens_readout": incoming_index + 1,
            "n_questions": len(qids),
            "leader_suppression_mean": mean,
            "leader_suppression_ci_low": low,
            "leader_suppression_ci_high": high,
            "fraction_leader_suppressed": fraction,
            "fraction_leader_suppressed_ci_low": fraction_low,
            "fraction_leader_suppressed_ci_high": fraction_high,
            "mean_other_option_write": float(other_write.mean()),
            "margin_pearson_r": float(pearsonr(margin, suppression).statistic),
            "margin_spearman_r": float(spearmanr(margin, suppression).statistic),
            "margin_quartiles": quartiles,
            "logit_lens_leader_suppression_mean": logit_mean,
            "logit_lens_leader_suppression_ci_low": logit_low,
            "logit_lens_leader_suppression_ci_high": logit_high,
            "logit_lens_fraction_leader_suppressed": logit_fraction,
            "logit_lens_fraction_leader_suppressed_ci_low": logit_fraction_low,
            "logit_lens_fraction_leader_suppressed_ci_high": logit_fraction_high,
            "logit_lens_margin_pearson_r": float(pearsonr(logit_margin, logit_suppression).statistic),
            "logit_lens_margin_spearman_r": float(spearmanr(logit_margin, logit_suppression).statistic),
        })

    payload = {
        "definition": (
            "For each component, identify the leading A-D JLens answer-letter score immediately before its "
            "block in the natural Game computation, and independently identify the leader by directly "
            "unembedding the component's exact input residual. A positive suppression value means the "
            "component's Game-like causal effect lowers that option's final A-D logit."
        ),
        "n_questions": len(qids),
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "leader_dependence.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    flat = [{key: value for key, value in row.items() if key != "margin_quartiles"} for row in rows]
    with (output / "leader_dependence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    lines = [
        "# Current-leader dependence of confirmed component effects",
        "",
        payload["definition"],
        "",
        "| Component | Direction | JLens leader suppressed | JLens margin r | Logit-lens leader suppressed | Logit-lens margin r |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: value["leader_suppression_mean"], reverse=True):
        lines.append(
            f"| {row['component']} | {row['direction']} | "
            f"{row['fraction_leader_suppressed']:.1%} | {row['margin_pearson_r']:+.3f} | "
            f"{row['logit_lens_fraction_leader_suppressed']:.1%} | {row['logit_lens_margin_pearson_r']:+.3f} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether causal components suppress the current Game leader")
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--sublayer-root", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        args.jlens_root, args.residual_root, args.sublayer_root, args.patch_root, args.plan,
        args.output, args.bootstrap_samples, args.seed,
    )


if __name__ == "__main__":
    main()
