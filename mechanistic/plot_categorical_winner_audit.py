from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(args: argparse.Namespace) -> None:
    data = json.loads(args.summary.read_text())
    labels = ["Linear score", "Cubic score", "Flexible score\n+ competitor gap"]
    keys = ["linear_score", "score_cubic", "score_gap_spline"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    x = np.arange(len(keys))
    for offset, split, color, marker in [
        (-0.09, "discovery", "#2878B5", "o"),
        (+0.09, "confirmation", "#D95F02", "s"),
    ]:
        rows = [data["models"][key][f"{split}_W1_term"] for key in keys]
        means = np.asarray([row["mean"] for row in rows])
        cis = np.asarray([row["ci"] for row in rows])
        axes[0].errorbar(
            x + offset, means, yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
            fmt=marker, color=color, capsize=4, linewidth=1.8, markersize=6,
            label=split.capitalize(),
        )
    axes[0].axhline(0, color="0.35", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Additional R1 lesion effect (logits)")
    axes[0].set_title("A  Separate winner term depends on score model")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="0.88", linewidth=0.8)

    near = data["near_ties"]
    tie_keys = ["0.25", "0.5", "1.0"]
    rows = [near[key] for key in tie_keys]
    means = np.asarray([row["mean"] for row in rows])
    cis = np.asarray([row["ci"] for row in rows])
    axes[1].errorbar(
        np.arange(3), means,
        yerr=np.vstack((means - cis[:, 0], cis[:, 1] - means)),
        fmt="o", color="#5B4B9A", capsize=4, linewidth=1.8, markersize=6,
    )
    axes[1].axhline(0, color="0.35", linewidth=1)
    axes[1].set_xticks(np.arange(3), ["≤0.25\n(n=67)", "≤0.5\n(n=106)", "≤1.0\n(n=164)"])
    axes[1].set_xlabel("Maximum first-pass R1–R2 score gap (logits)")
    axes[1].set_ylabel("R1 minus R2 lesion effect (logits)")
    axes[1].set_title("B  Near-tie comparisons remain uncertain")
    axes[1].grid(axis="y", color="0.88", linewidth=0.8)

    fig.suptitle("A separate categorical first-pass-winner effect is not established", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
