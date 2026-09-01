from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("baseline", "incorrect", "neutral")
CONDITION_LABELS = {
    "baseline": "Baseline",
    "incorrect": "Second Chance",
    "neutral": "Neutral",
}
BOUNDARIES = {"mixer": (0, 1), "mlp": (1, 2)}


def _load(path: Path) -> np.ndarray:
    with np.load(path) as shard:
        return shard["boundary_canonical_logits"].astype(np.float64)


def _letter_macro(values: np.ndarray, winner: np.ndarray) -> np.ndarray:
    return np.mean([values[winner == letter].mean(axis=0) for letter in range(4)], axis=0)


def _interval(
    values: np.ndarray,
    winner: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = _letter_macro(values, winner)
    strata = [np.flatnonzero(winner == letter) for letter in range(4)]
    draws = np.empty((bootstrap, 4), dtype=np.float64)
    for draw in range(bootstrap):
        sampled = [values[rng.choice(index, len(index), replace=True)].mean(axis=0) for index in strata]
        draws[draw] = np.mean(sampled, axis=0)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return mean, low, high


def _plot(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"baseline": "#333333", "incorrect": "#0072B2", "neutral": "#D55E00"}
    components = list(dict.fromkeys(row["component"] for row in rows))
    figure, axes = plt.subplots(1, len(components), figsize=(4.5 * len(components), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(4)
    width = 0.24
    for axis, component in zip(axes, components):
        for offset, condition in enumerate(CONDITIONS):
            selected = [
                row for row in rows
                if row["component"] == component and row["condition"] == condition
            ]
            selected.sort(key=lambda row: row["rank"])
            mean = np.asarray([row["mean"] for row in selected])
            low = np.asarray([row["ci_low"] for row in selected])
            high = np.asarray([row["ci_high"] for row in selected])
            location = x + (offset - 1) * width
            axis.bar(location, mean, width, color=colors[condition], label=CONDITION_LABELS[condition])
            axis.errorbar(location, mean, yerr=np.vstack([mean - low, high - mean]), fmt="none",
                          ecolor=colors[condition], elinewidth=0.8, capsize=2)
        axis.axhline(0, color="#666666", linewidth=0.7)
        axis.set_xticks(x, ["Winner", "Runner-up", "Rank 3", "Rank 4"])
        axis.set_title(component)
        axis.set_xlabel("Option aligned by final Baseline rank")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Immediate centered A-D pseudo-logit write")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output / f"selected_component_writes.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyze(
    input_dir: str | Path,
    output_dir: str | Path,
    components: list[tuple[str, int]],
    bootstrap: int,
    seed: int,
) -> None:
    root = Path(input_dir) / "shards"
    qids = sorted(path.stem for path in (root / "baseline").glob("*.npz"))
    if not qids:
        raise ValueError(f"No Baseline shards found under {root}")
    for condition in CONDITIONS:
        missing = [qid for qid in qids if not (root / condition / f"{qid}.npz").exists()]
        if missing:
            raise ValueError(f"{condition} is missing {len(missing)} shards")

    baseline_final = np.stack([_load(root / "baseline" / f"{qid}.npz")[-1, -1] for qid in qids])
    order = np.argsort(-baseline_final, axis=-1)
    winner = order[:, 0]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for kind, layer in components:
        before, after = BOUNDARIES[kind]
        component = f"{'Mixer' if kind == 'mixer' else 'MLP'} {layer}"
        for condition in CONDITIONS:
            writes = []
            for index, qid in enumerate(qids):
                lens = _load(root / condition / f"{qid}.npz")
                pre = lens[layer, before] - lens[layer, before].mean()
                post = lens[layer, after] - lens[layer, after].mean()
                writes.append((post - pre)[order[index]])
            values = np.stack(writes)
            mean, low, high = _interval(values, winner, bootstrap, rng)
            for rank in range(4):
                rows.append({
                    "component": component,
                    "kind": kind,
                    "layer": layer,
                    "condition": condition,
                    "rank": rank + 1,
                    "mean": float(mean[rank]),
                    "ci_low": float(low[rank]),
                    "ci_high": float(high[rank]),
                    "n_questions": len(qids),
                })

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selected_component_writes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "selected_component_writes.json").write_text(json.dumps(rows, indent=2))
    _plot(rows, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank-resolved natural writes of selected sublayers")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--component", action="append", required=True, help="KIND:LAYER, e.g. mixer:62")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    components = []
    for value in args.component:
        kind, layer = value.split(":", 1)
        if kind not in BOUNDARIES:
            raise ValueError(f"Unknown component kind: {kind}")
        components.append((kind, int(layer)))
    analyze(args.input, args.output, components, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
