from __future__ import annotations

import argparse
import json
from pathlib import Path


GAME_TOKENS = ("Your", "answer", "was", "incorrect", ".", "Choose", "a", "different", "answer", ".")
NEUTRAL_TOKENS = ("Your", "answer", "was", "lost", ".", "Choose", "the", "answer", "again", ".")


def prepare(
    confirmation_plan: Path,
    manifest: Path,
    output: Path,
) -> dict:
    confirmation_data = json.loads(confirmation_plan.read_text())
    confirmation = set(
        confirmation_data.get(
            "question_ids", confirmation_data.get("confirmation_question_ids", [])
        )
    )
    if not confirmation:
        raise ValueError("Confirmation plan has no question IDs")
    questions = json.loads(manifest.read_text())["questions"]
    all_ids = [question["id"] for question in questions]
    discovery = [question_id for question_id in all_ids if question_id not in confirmation]
    if len(discovery) != 251 or len(confirmation) != 249:
        raise ValueError(
            f"Expected frozen 251/249 split; found {len(discovery)}/{len(confirmation)}"
        )

    # Qwen3.6 uses ordinary attention at human-numbered blocks 4, 8, ..., 64.
    # Every other model layer is GLA.  Windows are defined by eight consecutive
    # human-numbered blocks, yielding six GLA layers per window.
    gdn_layers = [index for index in range(64) if (index + 1) % 4 != 0]
    windows = []
    for start in range(0, 64, 8):
        layers = [layer for layer in gdn_layers if start <= layer < start + 8]
        windows.append(
            {
                "id": f"blocks_{start + 1:02d}_{start + 8:02d}",
                "human_blocks": [start + 1, start + 8],
                "layers_zero_based": layers,
            }
        )
    scenarios = []
    for token_index, (game, neutral) in enumerate(zip(GAME_TOKENS, NEUTRAL_TOKENS)):
        token_id = f"token_{token_index + 1:02d}"
        token_pair = f"{game}<->{neutral}"
        scenarios.append(
            {
                "id": f"{token_id}__all_gla",
                "kind": "all_gla_token",
                "token_pair": token_pair,
                "feedback_token_indices_zero_based": [token_index],
                "layers_zero_based": gdn_layers,
            }
        )
        for window in windows:
            scenarios.append(
                {
                    "id": f"{token_id}__{window['id']}",
                    "kind": "token_window",
                    "token_pair": token_pair,
                    "window": window["id"],
                    "human_blocks": window["human_blocks"],
                    "feedback_token_indices_zero_based": [token_index],
                    "layers_zero_based": window["layers_zero_based"],
                }
            )
    plan = {
        "status": "discovery_only",
        "split_source": str(confirmation_plan),
        "question_ids": discovery,
        "confirmation_question_ids": sorted(confirmation),
        "tokens": {"Game": GAME_TOKENS, "Neutral": NEUTRAL_TOKENS},
        "gdn_layers_zero_based": gdn_layers,
        "windows": windows,
        "scenarios": scenarios,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen GLA token-layer discovery grid")
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.confirmation_plan, args.manifest, args.output)
    print(f"Prepared {len(plan['scenarios'])} scenarios on {len(plan['question_ids'])} questions")


if __name__ == "__main__":
    main()

