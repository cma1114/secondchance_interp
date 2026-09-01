from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .io import shard_path


def analyze(results: Path, output: Path) -> list[dict]:
    metadata = json.loads((results / "run_metadata.json").read_text())
    qids = metadata["question_ids"]
    layers = metadata["selected_user_facing_layers_one_based"]
    values = {}
    for condition in ("incorrect", "neutral"):
        values[condition] = np.asarray([
            np.load(shard_path(results, condition, qid), allow_pickle=False)[
                "endpoint_attention"
            ].astype(np.float64)
            for qid in qids
        ])
    module_rows = []
    head_rows = []
    for layer_index, layer in enumerate(layers):
        game_heads = values["incorrect"][:, layer_index].mean(axis=0)
        neutral_heads = values["neutral"][:, layer_index].mean(axis=0)
        for head in range(len(game_heads)):
            head_rows.append({
                "mixer": layer,
                "head": head,
                "game_mean_attention": float(game_heads[head]),
                "neutral_mean_attention": float(neutral_heads[head]),
                "game_minus_neutral": float(game_heads[head] - neutral_heads[head]),
            })
        game_module = values["incorrect"][:, layer_index].mean(axis=1)
        neutral_module = values["neutral"][:, layer_index].mean(axis=1)
        game_best = int(game_heads.argmax())
        neutral_best = int(neutral_heads.argmax())
        module_rows.append({
            "mixer": layer,
            "game_mean_per_head": float(game_module.mean()),
            "neutral_mean_per_head": float(neutral_module.mean()),
            "game_minus_neutral": float((game_module - neutral_module).mean()),
            "strongest_game_head": game_best,
            "strongest_game_head_attention": float(game_heads[game_best]),
            "strongest_neutral_head": neutral_best,
            "strongest_neutral_head_attention": float(neutral_heads[neutral_best]),
            "game_heads_above_5_percent": int((game_heads >= 0.05).sum()),
            "neutral_heads_above_5_percent": int((neutral_heads >= 0.05).sum()),
        })
    module_rows.sort(key=lambda row: row["game_mean_per_head"], reverse=True)
    head_rows.sort(key=lambda row: row["game_mean_attention"], reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("late_endpoint_attention_modules.csv", module_rows),
                       ("late_endpoint_attention_heads.csv", head_rows)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "n": len(qids),
        "module_rows": module_rows,
        "top_20_heads_by_game_attention": head_rows[:20],
    }
    (output / "late_endpoint_attention_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    lines = [
        "# Late attention to the first-answer generation endpoint",
        "",
        f"Frozen held-out SimpleMC questions: **{len(qids)}**.",
        "Values are mean final-query attention to the exact historical first-answer",
        "generation endpoint, averaged over questions and then over all 24 heads.",
        "",
        "| Mixer | Game | Neutral | Game − Neutral | Strongest Game head | Heads >5% (Game) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in module_rows:
        lines.append(
            f"| {row['mixer']} | {100*row['game_mean_per_head']:.2f}% | "
            f"{100*row['neutral_mean_per_head']:.2f}% | "
            f"{100*row['game_minus_neutral']:+.2f} pp | "
            f"H{row['strongest_game_head']} ({100*row['strongest_game_head_attention']:.2f}%) | "
            f"{row['game_heads_above_5_percent']} |"
        )
    (output / "LATE_ENDPOINT_ATTENTION_REPORT.md").write_text("\n".join(lines) + "\n")
    return module_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.output)


if __name__ == "__main__":
    main()

