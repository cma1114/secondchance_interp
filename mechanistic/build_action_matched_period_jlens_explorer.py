from __future__ import annotations

import argparse
from pathlib import Path

from .build_jlens_token_explorer import build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--position-audit", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    build(
        args.source, args.output, args.top_n, False,
        args.glossary, args.position_audit,
    )
    text = args.output.read_text()
    replacements = {
        "Game minus Neutral": "Evaluation minus Matched Neutral",
        '<option value="incorrect">Game</option>': '<option value="incorrect">Evaluation</option>',
        '<option value="neutral">Neutral</option>': '<option value="neutral">Matched Neutral</option>',
        "Game-pointing tokens": "Evaluation-pointing tokens",
        "Neutral-pointing tokens": "Matched-Neutral-pointing tokens",
        "Game − Neutral JLens score": "Evaluation − Matched Neutral JLens score",
        "Game minus Neutral JLens vocabulary contrasts": "Evaluation minus Matched Neutral JLens vocabulary contrasts",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
