from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LETTERS = "ABCD"
CURRENT = ("low", "high")
SCENARIOS = ("identity_cached", "d_line_kv", "identity_trajectory", "d_closing_trajectory", "full_history")
BASELINES = {"d_line_kv": 0, "d_closing_trajectory": 2, "full_history": 2}
CONDITION_CELLS = {"Game": (0, 2), "Neutral": (1, 3)}


def _ci(values: np.ndarray, rng: np.random.Generator, samples: int = 5000) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "low": None, "high": None, "n": 0}
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    boot = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high), "n": int(len(values))}


def _ratio_ci(num: np.ndarray, den: np.ndarray, rng: np.random.Generator, samples: int = 5000) -> dict:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    keep = np.isfinite(num) & np.isfinite(den)
    num, den = num[keep], den[keep]
    if not len(num) or abs(den.mean()) < 1e-8:
        return {"ratio_of_means": None, "low": None, "high": None, "n": int(len(num))}
    indices = rng.integers(0, len(num), size=(samples, len(num)))
    bnum = num[indices].mean(axis=1)
    bden = den[indices].mean(axis=1)
    valid = np.abs(bden) > 1e-8
    boot = bnum[valid] / bden[valid]
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"ratio_of_means": float(num.mean() / den.mean()), "low": float(low), "high": float(high), "n": int(len(num))}


def _target_centered(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    centered = logits - logits.mean(axis=-1, keepdims=True)
    indices = np.asarray([LETTERS.index(value) for value in targets], dtype=np.int64)
    prefix = centered.shape[:-2]
    n = centered.shape[-2]
    flat = centered.reshape((-1, n, 4))
    selected = np.stack([row[np.arange(n), indices] for row in flat], axis=0)
    return selected.reshape((*prefix, n))


def _transfer(target: np.ndarray, current: int, scenario: int, baseline: int, high_cell: int, low_cell: int) -> np.ndarray:
    identity = target[current, baseline]
    changed = target[current, scenario]
    low_shift = changed[low_cell] - identity[low_cell]
    high_shift = changed[high_cell] - identity[high_cell]
    return 0.5 * (low_shift - high_shift)


def analyze(discovery_path: Path, confirmation_path: Path, cohort_path: Path, output: Path, figure: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(cohort_path.read_text())
    cohort_rows = {row["question_id"]: row for row in cohort["rows"]}
    rng = np.random.default_rng(20260821)
    summary: dict = {"splits": {}, "validation": {}}
    plot_rows = []
    for split, path in (("discovery", discovery_path), ("confirmation", confirmation_path)):
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if not arrays["completed"].all():
            raise ValueError(f"{split} run is incomplete")
        keep = arrays["exact_eligible"].astype(bool)
        if not keep.any():
            raise ValueError(f"{split} has no exact-eligible rows")
        targets = arrays["semantic_targets"].astype(str)[keep]
        kept_qids = arrays["question_ids"].astype(str)[keep]
        for qid, target_name in zip(kept_qids, targets):
            row = cohort_rows[qid]
            if any(row[f"current_{name}_new_to_original"]["D"] != target_name for name in CURRENT):
                raise ValueError("The semantic target is not displayed at D in both current strata")
        final = arrays["final_logits"][:, :, :, keep, :].astype(np.float64)
        # The semantic target varies across original A-D identities, but the
        # frozen design displays it at literal D in both second presentations.
        target = _target_centered(final, np.full(len(targets), "D"))
        split_result: dict = {
            "n_frozen": int(len(keep)),
            "n_exact": int(keep.sum()),
            "actual_old_score_gap": _ci(arrays["actual_old_score_gap"][keep], rng),
            "conditions": {},
            "winner_crossing": {},
        }
        for condition, (high_cell, low_cell) in CONDITION_CELLS.items():
            condition_result = {"current": {}}
            transfers_by_current = {}
            for current_index, current_name in enumerate(CURRENT):
                current_result = {"scenarios": {}}
                for scenario in ("d_line_kv", "d_closing_trajectory", "full_history"):
                    scenario_index = SCENARIOS.index(scenario)
                    baseline_index = BASELINES[scenario]
                    history = target[current_index, baseline_index, high_cell] - target[current_index, baseline_index, low_cell]
                    values = _transfer(target, current_index, scenario_index, baseline_index, high_cell, low_cell)
                    transfers_by_current[(current_name, scenario)] = values
                    current_result["scenarios"][scenario] = {
                        "old_state_transfer": _ci(values, rng),
                        "fraction_of_complete_history": _ratio_ci(values, history, rng),
                    }
                    plot_rows.append((split, condition, current_name, scenario, current_result["scenarios"][scenario]["old_state_transfer"]))
                condition_result["current"][current_name] = current_result
            condition_result["current_high_minus_low_modulation"] = {
                scenario: _ci(transfers_by_current[("high", scenario)] - transfers_by_current[("low", scenario)], rng)
                for scenario in ("d_line_kv", "d_closing_trajectory", "full_history")
            }
            split_result["conditions"][condition] = condition_result

        winner = arrays["winner_crossing"].astype(bool)[keep]
        for condition, (high_cell, low_cell) in CONDITION_CELLS.items():
            split_result["winner_crossing"][condition] = {}
            for current_index, current_name in enumerate(CURRENT):
                split_result["winner_crossing"][condition][current_name] = {
                    scenario: _ci(_transfer(target, current_index, SCENARIOS.index(scenario), BASELINES[scenario], high_cell, low_cell)[winner], rng)
                    for scenario in ("d_line_kv", "d_closing_trajectory", "full_history")
                }
        summary["splits"][split] = split_result
        summary["validation"][split] = {
            "all_complete": bool(arrays["completed"].all()),
            "all_exact_rows_used": int(keep.sum()),
            "ordinary_layer_counts": sorted(set(arrays["ordinary_layer_count"][keep].astype(int).tolist())),
            "max_full_history_decision_error": float(np.nanmax(arrays["full_history_decision_max_error"][keep])),
            "max_full_history_final_error": float(np.nanmax(arrays["full_history_final_max_error"][:, keep])),
            "max_trajectory_identity_decision_error": float(np.nanmax(arrays["trajectory_identity_decision_max_error"][keep])),
            "max_trajectory_identity_final_error": float(np.nanmax(arrays["trajectory_identity_final_max_error"][:, keep])),
            "model_calls": sorted(set(arrays["model_calls"][keep].astype(int).tolist())),
        }

    for split, path in (("discovery", discovery_path), ("confirmation", confirmation_path)):
        with np.load(path, allow_pickle=False) as loaded:
            keep = loaded["exact_eligible"].astype(bool)
            targets = loaded["semantic_targets"].astype(str)[keep]
            target = _target_centered(loaded["final_logits"][:, :, :, keep, :].astype(np.float64), np.full(len(targets), "D"))
        interaction = {}
        for current_index, current_name in enumerate(CURRENT):
            interaction[current_name] = {}
            for scenario in ("d_line_kv", "d_closing_trajectory", "full_history"):
                scenario_index = SCENARIOS.index(scenario)
                baseline_index = BASELINES[scenario]
                game = _transfer(target, current_index, scenario_index, baseline_index, 0, 2)
                neutral = _transfer(target, current_index, scenario_index, baseline_index, 1, 3)
                interaction[current_name][scenario] = _ci(game - neutral, rng)
        summary["splits"][split]["game_minus_neutral_transfer"] = interaction

    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    colors = {"d_line_kv": "#2878B5", "d_closing_trajectory": "#E07A1F", "full_history": "#555555"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for row_index, condition in enumerate(("Game", "Neutral")):
        for col_index, split in enumerate(("discovery", "confirmation")):
            ax = axes[row_index, col_index]
            width = 0.22
            for scenario_index, scenario in enumerate(("d_line_kv", "d_closing_trajectory", "full_history")):
                rows = [r for r in plot_rows if r[0] == split and r[1] == condition and r[3] == scenario]
                rows.sort(key=lambda r: CURRENT.index(r[2]))
                x = np.arange(2) + (scenario_index - 1) * width
                means = [r[4]["mean"] for r in rows]
                lows = [m - r[4]["low"] for m, r in zip(means, rows)]
                highs = [r[4]["high"] - m for m, r in zip(means, rows)]
                ax.errorbar(x, means, yerr=[lows, highs], marker="o", capsize=4, linewidth=2, color=colors[scenario], label=scenario.replace("_", " "))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(np.arange(2), ["low current", "high current"])
            ax.set_title(f"{condition} — {split}")
            if col_index == 0:
                ax.set_ylabel("Transferred old-high minus old-low\ntarget evidence (logits)")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Where first-pass candidate value is carried", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    def fmt(row: dict) -> str:
        if row["mean"] is None:
            return "n/a"
        return f"{row['mean']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]"

    lines = [
        "# D-line old-score transfer and D-closing-state crossover",
        "",
        "## Bottom line",
        "",
        "This report keeps the local candidate-line and global D-closing-state hypotheses separate. The table reports direct within-task effects; Game-minus-Neutral contrasts follow only after those effects.",
        "",
    ]
    for split in ("discovery", "confirmation"):
        lines += [f"## {split.title()}", "", "| Task | Current evidence | Complete D-line K/V | D-closing state | Complete history |", "|---|---|---:|---:|---:|"]
        for condition in ("Game", "Neutral"):
            for current in CURRENT:
                scenarios = summary["splits"][split]["conditions"][condition]["current"][current]["scenarios"]
                lines.append(f"| {condition} | {current} | {fmt(scenarios['d_line_kv']['old_state_transfer'])} | {fmt(scenarios['d_closing_trajectory']['old_state_transfer'])} | {fmt(scenarios['full_history']['old_state_transfer'])} |")
        lines += [
            "",
            f"Exact eligible: {summary['splits'][split]['n_exact']}/{summary['splits'][split]['n_frozen']}.",
            "The frozen screen chose high/low histories using the earlier 24-permutation forward path. Exact eligibility was then rechecked inside the cached causal cohort and required the target's measured centered first-decision score to remain higher in the nominated high history than in the nominated low history. Rows lost here are numerical/path-regime reversals of that ordering, not intervention-outcome exclusions.",
            "",
        ]
    lines += [
        "## Interpretation rules",
        "",
        "- A replicating D-line K/V effect means the text-identical final candidate line carries causally usable information about its old value.",
        "- A replicating D-closing-state effect means that one closing token carries portable old-history information. Because D is also the target's own line, the target-logit effect alone does not distinguish a local target state from a global comparison summary; that distinction requires the four-candidate transfer-vector analysis.",
        "- Current-high versus current-low differences test whether either old state is combined non-additively with fresh evidence.",
        "- A complete-history effect alone confirms history dependence but does not localize the information.",
        "",
        "## Validation",
        "",
        f"- Discovery: {json.dumps(summary['validation']['discovery'], sort_keys=True)}",
        f"- Confirmation: {json.dumps(summary['validation']['confirmation'], sort_keys=True)}",
        "",
        f"Canonical figure: `{figure}`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.discovery, args.confirmation, args.cohort, args.output, args.figure)


if __name__ == "__main__":
    main()
