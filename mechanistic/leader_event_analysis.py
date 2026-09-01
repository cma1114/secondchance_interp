from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import load_activation_dataset
from .trajectory_analysis import centered


CONDITIONS = ["baseline", "incorrect", "neutral"]


def _bootstrap_curve(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trial-bootstrap a questions x layers array, allowing NaNs at event edges."""
    mean = np.nanmean(values, axis=0)
    lo = np.empty(values.shape[1]); hi = np.empty(values.shape[1])
    for layer in range(values.shape[1]):
        x = values[:, layer]
        x = x[np.isfinite(x)]
        if not len(x):
            lo[layer] = hi[layer] = np.nan
            continue
        boot = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
        lo[layer], hi[layer] = np.quantile(boot, [0.025, 0.975])
    return mean, lo, hi


def _relative(values: np.ndarray, choices: np.ndarray) -> np.ndarray:
    selected = np.take_along_axis(values, choices[..., None], axis=-1)[..., 0]
    return selected - (values.sum(axis=-1) - selected) / 3.0


def _compression(z: np.ndarray) -> np.ndarray:
    """Positive values mean a block update points toward flattening A-D evidence."""
    x = z[:, :-1]
    update = z[:, 1:] - z[:, :-1]
    return -np.sum(update * x, axis=-1) / np.maximum(np.sum(x * x, axis=-1), 1e-9)


def _crossfit_emergence(baseline: np.ndarray, question_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Find a held-out, margin-thresholded emergence of the final baseline winner."""
    n, layers, _ = baseline.shape
    final_winner = np.argmax(baseline[:, -1], axis=-1)
    leader = np.argmax(baseline, axis=-1)
    ordered = np.sort(baseline, axis=-1)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    folds = np.asarray([
        int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2 for qid in question_ids
    ])

    # The threshold is the median margin at unthresholded stable emergence in
    # the opposite fold. This calibrates scale without peeking at an evaluated
    # trial's own trajectory.
    stable = np.full(n, layers - 1, dtype=int)
    for i in range(n):
        for layer in range(layers):
            end = min(layers, layer + 4)
            if np.all(leader[i, layer:end] == final_winner[i]):
                stable[i] = layer
                break
    thresholds = np.zeros(2)
    for fold in (0, 1):
        train = np.where(folds != fold)[0]
        thresholds[fold] = np.median(margin[train, stable[train]])

    event = np.full(n, layers - 1, dtype=int)
    for i in range(n):
        tau = thresholds[folds[i]]
        for layer in range(layers):
            end = min(layers, layer + 4)
            if (
                leader[i, layer] == final_winner[i]
                and margin[i, layer] >= tau
                and np.all(leader[i, layer:end] == final_winner[i])
            ):
                event[i] = layer
                break
    return event, thresholds


def _event_align(values: np.ndarray, events: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    out = np.full((len(values), len(offsets)), np.nan)
    for i, event in enumerate(events):
        layers = event + offsets
        valid = (layers >= 0) & (layers < values.shape[1])
        out[i, valid] = values[i, layers[valid]]
    return out


def _write_curves(path: Path, curves: dict[str, np.ndarray], x_name: str, xs: np.ndarray, rng, n_boot: int) -> None:
    rows = []
    for metric, values in curves.items():
        mean, lo, hi = _bootstrap_curve(values, rng, n_boot)
        count = np.sum(np.isfinite(values), axis=0)
        for x, m, a, b, n in zip(xs, mean, lo, hi, count):
            rows.append({"metric": metric, x_name: int(x), "mean": m, "ci_low": a, "ci_high": b, "n": int(n)})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def analyze(input_dir: str, behavioral_dir: str, output_dir: str, n_boot: int, seed: int) -> dict:
    data = load_activation_dataset(input_dir, CONDITIONS)
    z = centered(data.logits)
    n, _, layers, _ = z.shape
    baseline, game, neutral = z[:, 0], z[:, 1], z[:, 2]
    order = np.argsort(-baseline[:, -1], axis=-1)
    winner = order[:, 0]
    runner = order[:, 1]
    winner_layers = np.broadcast_to(winner[:, None], (n, layers))

    winner_adv = np.stack([_relative(z[:, ci], winner_layers) for ci in range(3)], axis=1)
    # Runner-up boosting must be measured against baseline ranks 3 and 4, not
    # against a mean that includes the winner; otherwise winner suppression
    # mechanically masquerades as runner boosting.
    runner_adv = np.empty((n, 3, layers), dtype=z.dtype)
    for ci in range(3):
        ranked = np.take_along_axis(z[:, ci], np.broadcast_to(order[:, None, :], (n, layers, 4)), axis=-1)
        runner_adv[:, ci] = ranked[:, :, 1] - ranked[:, :, 2:].mean(axis=-1)
    spread = z.max(axis=-1) - z.min(axis=-1)
    compression = np.stack([_compression(z[:, ci]) for ci in range(3)], axis=1)

    # Blockwise penalty to the candidate currently leading in Game. Negative is
    # suppression. The same candidate is evaluated in the paired reference.
    game_leader = np.argmax(game[:, :-1], axis=-1)
    game_update = game[:, 1:] - game[:, :-1]
    base_update = baseline[:, 1:] - baseline[:, :-1]
    neutral_update = neutral[:, 1:] - neutral[:, :-1]
    leader_game = _relative(game_update, game_leader)
    leader_base = _relative(base_update, game_leader)
    leader_neutral = _relative(neutral_update, game_leader)

    fixed_curves = {
        "game_minus_baseline_spread": spread[:, 1] - spread[:, 0],
        "neutral_minus_baseline_spread": spread[:, 2] - spread[:, 0],
        "game_minus_neutral_spread": spread[:, 1] - spread[:, 2],
        "game_minus_baseline_winner_advantage": winner_adv[:, 1] - winner_adv[:, 0],
        "neutral_minus_baseline_winner_advantage": winner_adv[:, 2] - winner_adv[:, 0],
        "game_minus_neutral_winner_advantage": winner_adv[:, 1] - winner_adv[:, 2],
        "game_minus_baseline_runner_advantage": runner_adv[:, 1] - runner_adv[:, 0],
        "neutral_minus_baseline_runner_advantage": runner_adv[:, 2] - runner_adv[:, 0],
        "game_minus_neutral_runner_advantage": runner_adv[:, 1] - runner_adv[:, 2],
    }
    block_curves = {
        "game_minus_baseline_compression": compression[:, 1] - compression[:, 0],
        "neutral_minus_baseline_compression": compression[:, 2] - compression[:, 0],
        "game_minus_baseline_current_leader_update": leader_game - leader_base,
        "game_minus_neutral_current_leader_update": leader_game - leader_neutral,
    }

    events, thresholds = _crossfit_emergence(baseline, data.question_ids)
    offsets = np.arange(-15, 31)
    event_curves = {
        key: _event_align(values, events, offsets)
        for key, values in fixed_curves.items()
    }

    # Margin bins make the thresholded-leader prediction visually testable.
    ordered_game = np.sort(game[:, :-1], axis=-1)
    game_margin = ordered_game[:, :, -1] - ordered_game[:, :, -2]
    leader_effect = leader_game - leader_base
    flat_margin = game_margin.ravel()
    edges = np.quantile(flat_margin, np.linspace(0, 1, 6))
    margin_rows = []
    rng = np.random.default_rng(seed)
    for b in range(5):
        mask = (flat_margin >= edges[b]) & (flat_margin <= edges[b + 1] if b == 4 else flat_margin < edges[b + 1])
        mask_2d = mask.reshape(game_margin.shape)
        trial_values = np.asarray([
            leader_effect[i, mask_2d[i]].mean() if np.any(mask_2d[i]) else np.nan
            for i in range(n)
        ])
        x = trial_values[np.isfinite(trial_values)]
        boot = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        margin_rows.append({
            "bin": b + 1, "margin_low": edges[b], "margin_high": edges[b + 1],
            "mean_game_minus_baseline_leader_update": x.mean(), "ci_low": lo, "ci_high": hi,
            "n_trials": len(x), "n_observations": int(mask.sum()),
        })

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    _write_curves(out / "fixed_layer_curves.csv", fixed_curves, "readout", np.arange(layers), rng, n_boot)
    _write_curves(out / "block_update_curves.csv", block_curves, "block_from", np.arange(layers - 1), rng, n_boot)
    _write_curves(out / "event_aligned_curves.csv", event_curves, "offset", offsets, rng, n_boot)
    with (out / "leader_margin_bins.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=margin_rows[0].keys()); writer.writeheader(); writer.writerows(margin_rows)

    game_results = json.loads((Path(behavioral_dir) / "game_results.json").read_text())["results"]
    neutral_results = json.loads((Path(behavioral_dir) / "neutral_results.json").read_text())["results"]
    game_switch = np.asarray([bool(game_results[q]["answer_changed"]) for q in data.question_ids])
    neutral_switch = np.asarray([bool(neutral_results[q]["answer_changed"]) for q in data.question_ids])
    switch_rows = []
    for label, mask in (("game_switch", game_switch), ("game_no_switch", ~game_switch),
                        ("neutral_switch", neutral_switch), ("neutral_no_switch", ~neutral_switch)):
        condition = 1 if label.startswith("game") else 2
        switch_rows.append({
            "stratum": label, "n": int(mask.sum()),
            "final_winner_advantage_minus_baseline": float((winner_adv[mask, condition, -1] - winner_adv[mask, 0, -1]).mean()),
            "final_spread_minus_baseline": float((spread[mask, condition, -1] - spread[mask, 0, -1]).mean()),
            "mean_last10_compression_minus_baseline": float((compression[mask, condition, -10:] - compression[mask, 0, -10:]).mean()),
        })
    with (out / "switch_strata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=switch_rows[0].keys()); writer.writeheader(); writer.writerows(switch_rows)

    summary = {
        "n_questions": n,
        "n_readouts": layers,
        "crossfit_emergence_thresholds": thresholds.tolist(),
        "median_emergence_readout": float(np.median(events)),
        "final_effects": {key: float(values[:, -1].mean()) for key, values in fixed_curves.items()},
        "strongest_game_leader_penalty_block_vs_baseline": int(np.argmin((leader_game - leader_base).mean(axis=0))),
        "strongest_game_compression_block_vs_baseline": int(np.argmax((compression[:, 1] - compression[:, 0]).mean(axis=0))),
        "leader_margin_bins": margin_rows,
        "switch_strata": switch_rows,
        "caveat": "Logit-lens scores are pseudo-logits at intermediate readouts; these are observational signatures, not causal circuit identification.",
    }
    (out / "leader_event_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(fixed_curves, block_curves, event_curves, offsets, margin_rows, out, n_boot, seed)
    return summary


def _plot(fixed, block, event, offsets, margin_rows, out: Path, n_boot: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    rng = np.random.default_rng(seed + 1000)
    layers = np.arange(next(iter(fixed.values())).shape[1])
    for key, color in (("game_minus_baseline_spread", "#b2182b"), ("neutral_minus_baseline_spread", "#2166ac")):
        mean, lo, hi = _bootstrap_curve(fixed[key], rng, n_boot)
        axes[0, 0].plot(layers, mean, label=key.replace("_", " "), color=color)
        axes[0, 0].fill_between(layers, lo, hi, color=color, alpha=.16)
    axes[0, 0].set(title="A–D spread relative to baseline", xlabel="Residual readout", ylabel="Pseudo-logit difference")
    for key, color in (
        ("game_minus_baseline_winner_advantage", "#b2182b"),
        ("neutral_minus_baseline_winner_advantage", "#2166ac"),
        ("game_minus_neutral_winner_advantage", "#1b7837"),
    ):
        mean, lo, hi = _bootstrap_curve(fixed[key], rng, n_boot)
        axes[0, 1].plot(layers, mean, label=key.replace("_", " "), color=color)
        axes[0, 1].fill_between(layers, lo, hi, color=color, alpha=.16)
    axes[0, 1].set(title="Original winner advantage: paired contrasts", xlabel="Residual readout", ylabel="Pseudo-logit difference")
    for key, color in (
        ("game_minus_baseline_winner_advantage", "#b2182b"),
        ("neutral_minus_baseline_winner_advantage", "#2166ac"),
        ("game_minus_neutral_winner_advantage", "#1b7837"),
    ):
        mean, lo, hi = _bootstrap_curve(event[key], rng, n_boot)
        axes[1, 0].plot(offsets, mean, label=key.replace("_", " "), color=color)
        axes[1, 0].fill_between(offsets, lo, hi, color=color, alpha=.16)
    axes[1, 0].axvline(0, color="black", lw=.8, ls="--")
    axes[1, 0].set(title="Event-aligned to baseline winner emergence", xlabel="Readouts from emergence", ylabel="Pseudo-logit difference")
    x = [r["bin"] for r in margin_rows]; y = [r["mean_game_minus_baseline_leader_update"] for r in margin_rows]
    lo = [r["ci_low"] for r in margin_rows]; hi = [r["ci_high"] for r in margin_rows]
    axes[1, 1].errorbar(x, y, yerr=[np.asarray(y)-np.asarray(lo), np.asarray(hi)-np.asarray(y)], fmt="o-", color="#7b3294")
    axes[1, 1].axhline(0, color="black", lw=.8)
    axes[1, 1].set(title="Current-leader update by pre-block margin", xlabel="Game leader-margin quintile", ylabel="Game − baseline relative update")
    for ax in axes.flat:
        ax.axhline(0, color="black", lw=.5, alpha=.6)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "leader_dynamics.png", dpi=220)
    fig.savefig(out / "leader_dynamics.svg")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze leader emergence and Second Chance block updates")
    p.add_argument("--input", required=True)
    p.add_argument("--behavioral", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(json.dumps(analyze(args.input, args.behavioral, args.output, args.bootstrap, args.seed), indent=2))


if __name__ == "__main__":
    main()
