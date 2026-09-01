from __future__ import annotations

import argparse
import json
from pathlib import Path


GLA_LAYERS = [index for index in range(64) if (index + 1) % 4 != 0]
SOURCES = {
    "evaluation_period": ([4], "evaluation period"),
    "action_clause": ([5, 6, 7, 8, 9], "action clause"),
}


def _scenario(
    scenario_id: str,
    kind: str,
    label: str,
    tokens: list[int],
    layers: list[int],
    **extra,
) -> dict:
    return {
        "id": scenario_id,
        "kind": kind,
        "token_pair": label,
        "feedback_token_indices_zero_based": tokens,
        "layers_zero_based": layers,
        **extra,
    }


def build(discovery_summary: Path, confirmation_source: Path, output: Path) -> None:
    discovery = json.loads(discovery_summary.read_text())
    confirmation = json.loads(confirmation_source.read_text())
    effects = discovery["effects"]
    scenarios: list[dict] = []
    selections = {}

    for source_id, (tokens, label) in SOURCES.items():
        rows = []
        for layer in GLA_LAYERS:
            block = layer + 1
            record = effects[f"{source_id}__block_{block:02d}"]["Game"][
                "winner_advantage"
            ]
            rows.append(
                {
                    "layer": layer,
                    "block": block,
                    "estimate": record["estimate"],
                    "ci": record["ci"],
                }
            )
        positive = sorted(
            [row for row in rows if row["ci"][0] > 0],
            key=lambda row: row["estimate"],
            reverse=True,
        )
        negative = sorted(
            [row for row in rows if row["ci"][1] < 0],
            key=lambda row: row["estimate"],
        )
        selections[source_id] = {
            "positive_blocks": [row["block"] for row in positive],
            "negative_blocks": [row["block"] for row in negative],
            "criterion": (
                "Discovery within-Game winner-advantage ablation effect has a "
                "95% bootstrap CI entirely above or below zero, respectively."
            ),
        }

        scenarios.append(
            _scenario(f"{source_id}__all_gla", "all_gla", label, tokens, GLA_LAYERS)
        )
        for row in rows:
            scenarios.append(
                _scenario(
                    f"{source_id}__block_{row['block']:02d}",
                    "exact_layer",
                    label,
                    tokens,
                    [row["layer"]],
                    human_block=row["block"],
                    discovery_winner_advantage=row["estimate"],
                    discovery_ci=row["ci"],
                )
            )

        sets = {
            "top2_positive": positive[:2],
            "top4_positive": positive[:4],
            "all_discovery_positive": positive,
            "top2_negative": negative[:2],
            "all_discovery_negative": negative,
        }
        for set_id, selected in sets.items():
            if not selected:
                continue
            layers = [row["layer"] for row in selected]
            scenarios.append(
                _scenario(
                    f"{source_id}__joint_{set_id}",
                    "joint_selected_layers",
                    label,
                    tokens,
                    layers,
                    selected_human_blocks=[row["block"] for row in selected],
                )
            )

        for set_id, selected in {
            "positive": positive,
            "negative": negative,
        }.items():
            if not selected:
                continue
            retained = {row["layer"] for row in selected}
            ablated = [layer for layer in GLA_LAYERS if layer not in retained]
            scenarios.append(
                _scenario(
                    f"{source_id}__all_except_discovery_{set_id}",
                    "leave_selected_layers_natural",
                    label,
                    tokens,
                    ablated,
                    natural_human_blocks=[row["block"] for row in selected],
                )
            )

    payload = {
        "status": "frozen_winner_margin_confirmation",
        "question_ids": confirmation["question_ids"],
        "scenarios": scenarios,
        "selection": selections,
        "primary_condition": "incorrect",
        "primary_metric": (
            "Within-Game change in frozen-Baseline winner advantage: ablation minus natural"
        ),
        "notes": (
            "All 48 GLA blocks are confirmed individually for both sources. Joint and "
            "rescue sets are frozen from the 251-question discovery half only."
        ),
        "discovery_summary": str(discovery_summary),
        "confirmation_question_source": str(confirmation_source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-summary", type=Path, required=True)
    parser.add_argument("--confirmation-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.discovery_summary, args.confirmation_source, args.output)


if __name__ == "__main__":
    main()
