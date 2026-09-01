from __future__ import annotations

import argparse
import json
from pathlib import Path


GLA_LAYERS = [index for index in range(64) if (index + 1) % 4 != 0]
WINDOWS = [
    [index for index in GLA_LAYERS if start <= index + 1 <= start + 7]
    for start in range(1, 65, 8)
]


def _scenario(
    scenario_id: str,
    kind: str,
    token_pair: str,
    token_indices: list[int],
    layers: list[int],
    **extra,
) -> dict:
    return {
        "id": scenario_id,
        "kind": kind,
        "token_pair": token_pair,
        "feedback_token_indices_zero_based": token_indices,
        "layers_zero_based": layers,
        **extra,
    }


def build(source_plan: Path, output: Path) -> None:
    source = json.loads(source_plan.read_text())
    qids = source["question_ids"]
    scenarios: list[dict] = []
    sources = {
        "evaluation_period": ([4], "evaluation period"),
        "action_clause": ([5, 6, 7, 8, 9], "action clause"),
    }

    for source_id, (tokens, label) in sources.items():
        scenarios.append(_scenario(
            f"{source_id}__all_gla", "all_gla_replication", label, tokens, GLA_LAYERS
        ))
        for layer in GLA_LAYERS:
            block = layer + 1
            scenarios.append(_scenario(
                f"{source_id}__block_{block:02d}",
                "exact_layer",
                label,
                tokens,
                [layer],
                human_block=block,
            ))
        for window_index, window in enumerate(WINDOWS):
            start = 1 + 8 * window_index
            end = start + 7
            retained = [layer for layer in GLA_LAYERS if layer not in window]
            scenarios.append(_scenario(
                f"{source_id}__all_except_blocks_{start:02d}_{end:02d}",
                "leave_one_window_out",
                label,
                tokens,
                retained,
                omitted_human_blocks=[start, end],
            ))
            prefix = [layer for layer in GLA_LAYERS if layer <= max(window)]
            suffix = [layer for layer in GLA_LAYERS if layer >= min(window)]
            scenarios.append(_scenario(
                f"{source_id}__prefix_through_block_{end:02d}",
                "cumulative_prefix",
                label,
                tokens,
                prefix,
                human_blocks=[1, end],
            ))
            scenarios.append(_scenario(
                f"{source_id}__suffix_from_block_{start:02d}",
                "cumulative_suffix",
                label,
                tokens,
                suffix,
                human_blocks=[start, 64],
            ))

    action_tokens = [5, 6, 7, 8, 9]
    action_labels = ["Choose", "a/the", "different/answer", "answer/again", "period"]
    for omitted, omitted_label in zip(action_tokens, action_labels, strict=True):
        retained = [index for index in action_tokens if index != omitted]
        scenarios.append(_scenario(
            f"action_clause__all_tokens_except_{omitted:02d}",
            "leave_one_token_unablated",
            f"action clause except {omitted_label}",
            retained,
            GLA_LAYERS,
            unablated_token_index_zero_based=omitted,
            unablated_token_label=omitted_label,
        ))

    payload = {
        "status": "source_layer_discovery",
        "question_ids": qids,
        "scenarios": scenarios,
        "source_question_split": str(source_plan),
        "notes": (
            "No full-feedback interventions. Period and action-clause sources are "
            "decomposed separately. Human block numbers are one-based."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare source-specific GLA layer decomposition")
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_plan, args.output)


if __name__ == "__main__":
    main()
