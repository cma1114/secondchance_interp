from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(screen_path: Path, output_path: Path) -> None:
    screen = np.load(screen_path, allow_pickle=False)
    layers = screen["layer_indices"].astype(int) + 1
    sources = screen["source_labels"].astype(str).tolist()
    roles = screen["receiver_roles"].astype(str).tolist()
    discovery = screen["discovery_write_contrast"].astype(np.float32)
    confirmation = screen["confirmation_write_contrast"].astype(np.float32)
    attention = screen["mean_attention"].astype(np.float32)

    rows = []
    for layer_index, layer in enumerate(layers):
        for source_index, source in enumerate(sources):
            for role_index, role in enumerate(roles):
                first = discovery[layer_index, source_index, role_index]
                second = confirmation[layer_index, source_index, role_index]
                denominator = np.linalg.norm(first) * np.linalg.norm(second)
                rows.append(
                    {
                        "layer": int(layer),
                        "source": source,
                        "receiver_role": role,
                        "discovery_contrast_rms": float(np.sqrt(np.mean(first * first))),
                        "confirmation_contrast_rms": float(np.sqrt(np.mean(second * second))),
                        "discovery_confirmation_cosine": float(
                            np.dot(first, second) / max(float(denominator), 1e-12)
                        ),
                        "attention": {
                            "discovery_game": float(
                                attention[0, 0, layer_index, source_index, role_index]
                            ),
                            "discovery_neutral": float(
                                attention[0, 1, layer_index, source_index, role_index]
                            ),
                            "confirmation_game": float(
                                attention[1, 0, layer_index, source_index, role_index]
                            ),
                            "confirmation_neutral": float(
                                attention[1, 1, layer_index, source_index, role_index]
                            ),
                        },
                    }
                )
    result = {
        "definition": (
            "Full ordinary-attention-layer trajectory of exact feedback-token-specific "
            "Game-minus-Neutral residual writes into every retained 2P receiver role."
        ),
        "layers": layers.tolist(),
        "sources": sources,
        "receiver_roles": roles,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.screen, args.output)


if __name__ == "__main__":
    main()
