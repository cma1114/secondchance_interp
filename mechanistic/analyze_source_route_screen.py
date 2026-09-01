from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .component_causal_metrics import RANK_AXIS, RANK_AXIS_DENOMINATOR, center
from .io import shard_path


def _load(root: Path, group: str, qids: list[str], key: str) -> np.ndarray:
    values = []
    for qid in qids:
        with np.load(shard_path(root, group, qid), allow_pickle=False) as data:
            values.append(data[key])
    return np.asarray(values, dtype=np.float64)


def _route_slopes(direct: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    # direct: question, source, head, A-D
    order = np.argsort(-center(baseline), axis=-1)
    centered = center(direct)
    aligned = np.take_along_axis(
        centered,
        order[:, None, None, :],
        axis=-1,
    )
    return np.sum(aligned * RANK_AXIS, axis=-1) / RANK_AXIS_DENOMINATOR


def analyze(
    screen_root: Path,
    discovery_plan_path: Path,
    confirmation_plan_path: Path,
    output: Path,
    routes_per_component: int,
) -> dict:
    discovery = json.loads(discovery_plan_path.read_text())
    confirmation = json.loads(confirmation_plan_path.read_text())
    metadata = json.loads((screen_root / "run_metadata.json").read_text())
    sources = list(metadata["source_names"])
    planned = discovery["question_ids"]
    qids = [
        qid
        for qid in planned
        if all(shard_path(screen_root, group, qid).exists() for group in ("baseline", "incorrect", "neutral"))
    ]
    if not qids:
        raise FileNotFoundError("No complete source-route screen questions")
    baseline = _load(screen_root, "baseline", qids, "final_canonical_logits")

    specs = (
        (
            "attention",
            "attention_route_direct_ad",
            "attention_mass",
            None,
            None,
            "decision__mixer_l55",
        ),
        (
            "gdn",
            "gdn_route_direct_ad",
            None,
            "gdn_beta",
            "gdn_retention",
            "decision__mixer_l62",
        ),
    )
    rows = []
    selected = []
    plot_values = {}
    auxiliary_values = {}
    for kind, direct_key, mass_key, beta_key, retention_key, component in specs:
        direct = {
            condition: _load(screen_root, condition, qids, direct_key)
            for condition in ("incorrect", "neutral")
        }
        slopes = {
            condition: _route_slopes(values, baseline)
            for condition, values in direct.items()
        }
        differential = slopes["incorrect"] - slopes["neutral"]
        mean_differential = differential.mean(axis=0)
        plot_values[kind] = mean_differential
        mass = None
        if mass_key is not None:
            mass = {
                condition: _load(screen_root, condition, qids, mass_key)
                for condition in ("incorrect", "neutral")
            }
        beta = None
        if beta_key is not None:
            beta = {
                condition: _load(screen_root, condition, qids, beta_key)
                for condition in ("incorrect", "neutral")
            }
        retention = None
        if retention_key is not None:
            retention = {
                condition: _load(screen_root, condition, qids, retention_key)
                for condition in ("incorrect", "neutral")
            }
        auxiliary_values[kind] = (
            (mass["incorrect"] - mass["neutral"]).mean(axis=0)
            if mass is not None
            else (beta["incorrect"] - beta["neutral"]).mean(axis=0)
        )
        for source_index, source in enumerate(sources):
            for head in range(mean_differential.shape[1]):
                row = {
                    "kind": kind,
                    "component": component,
                    "source": source,
                    "head": head,
                    "n_questions": len(qids),
                    "game_rank_slope": float(slopes["incorrect"][:, source_index, head].mean()),
                    "neutral_rank_slope": float(slopes["neutral"][:, source_index, head].mean()),
                    "game_minus_neutral_rank_slope": float(mean_differential[source_index, head]),
                    "game_attention_mass": None if mass is None else float(mass["incorrect"][:, source_index, head].mean()),
                    "neutral_attention_mass": None if mass is None else float(mass["neutral"][:, source_index, head].mean()),
                    "game_beta": None if beta is None else float(beta["incorrect"][:, source_index, head].mean()),
                    "neutral_beta": None if beta is None else float(beta["neutral"][:, source_index, head].mean()),
                    "game_retention": None if retention is None else float(retention["incorrect"][:, source_index, head].mean()),
                    "neutral_retention": None if retention is None else float(retention["neutral"][:, source_index, head].mean()),
                }
                rows.append(row)
        eligible = [
            row for row in rows
            if row["kind"] == kind and row["game_minus_neutral_rank_slope"] > 0
        ]
        eligible.sort(
            key=lambda row: row["game_minus_neutral_rank_slope"], reverse=True
        )
        selected.extend(eligible[:routes_per_component])

    output.mkdir(parents=True, exist_ok=True)
    with (output / "source_route_screen.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    target_lookup = {
        target["component"]: target for target in confirmation["targets"]
    }
    route_plan = {
        "stage": "heldout_conditional_source_route_confirmation",
        "question_ids": confirmation["question_ids"],
        "component_targets": confirmation["targets"],
        "selected_routes": selected,
        "source_names": sources,
        "selection_rule": (
            f"On {len(qids)} discovery questions, select the top "
            f"{routes_per_component} positive Game-minus-Neutral immediate "
            "ordered-rank writes separately for Mixer 56 attention routes and "
            "Mixer 63 DeltaNet routes. Confirmation uses only untouched questions."
        ),
        "screen_root": str(screen_root),
        "discovery_plan": str(discovery_plan_path),
        "source_components": {
            "attention": target_lookup["decision__mixer_l55"],
            "gdn": target_lookup["decision__mixer_l62"],
        },
    }
    (output / "confirmation_plan.json").write_text(
        json.dumps(route_plan, indent=2, sort_keys=True)
    )
    _plot(plot_values, auxiliary_values, sources, selected, output)
    summary = {
        "complete": len(qids) == len(planned),
        "n_discovery": len(qids),
        "routes_per_component": routes_per_component,
        "selected_routes": selected,
    }
    (output / "source_route_screen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return route_plan


def _plot(
    values: dict[str, np.ndarray],
    auxiliary: dict[str, np.ndarray],
    sources: list[str],
    selected: list[dict],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(19, 17), constrained_layout=True)
    for column, (kind, title) in enumerate(zip(
        ("attention", "gdn"),
        ("Mixer 56 ordinary-attention routes", "Mixer 63 DeltaNet memory routes"),
    )):
        axis = axes[0, column]
        matrix = values[kind]
        bound = np.percentile(np.abs(matrix), 99)
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-bound,
            vmax=bound,
        )
        axis.set_yticks(range(len(sources)), labels=sources)
        axis.set_xlabel("Head")
        axis.set_ylabel("Exhaustive prompt source span")
        axis.set_title(title + "\nGame−Neutral immediate ordered-rank write")
        for row in selected:
            if row["kind"] == kind:
                axis.scatter(
                    row["head"],
                    sources.index(row["source"]),
                    marker="s",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.2,
                )
        fig.colorbar(image, ax=axis, shrink=0.75)
        axis = axes[1, column]
        matrix = auxiliary[kind]
        bound = max(float(np.percentile(np.abs(matrix), 99)), 1e-8)
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-bound,
            vmax=bound,
        )
        axis.set_yticks(range(len(sources)), labels=sources)
        axis.set_xlabel("Head")
        axis.set_ylabel("Exhaustive prompt source span")
        axis.set_title(
            (
                "Game−Neutral final-query attention mass"
                if kind == "attention"
                else "Game−Neutral recurrent beta write strength"
            ),
            loc="left",
        )
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(output / "source_route_screen.png", dpi=220)
    fig.savefig(output / "source_route_screen.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and select prompt-source routes")
    parser.add_argument("--screen-root", required=True)
    parser.add_argument("--discovery-plan", required=True)
    parser.add_argument("--confirmation-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--routes-per-component", type=int, default=8)
    args = parser.parse_args()
    plan = analyze(
        Path(args.screen_root),
        Path(args.discovery_plan),
        Path(args.confirmation_plan),
        Path(args.output),
        args.routes_per_component,
    )
    print(f"Selected {len(plan['selected_routes'])} routes")


if __name__ == "__main__":
    main()
