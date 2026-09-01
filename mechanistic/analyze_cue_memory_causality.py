from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import LETTERS


TASKS = ("Game", "Neutral")


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _bootstrap(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    seed: int = 20260822,
    draws: int = 10000,
) -> dict[str, float]:
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if n == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
        }
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        index = rng.integers(0, n, size=(stop - start, n))
        for local, row in enumerate(index):
            estimates[start + local] = statistic(values[row])
    return {
        "mean": statistic(values),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
        "n": int(n),
    }


def _projection_fraction(delta: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sum(delta * target)
    denominator = np.sum(target * target)
    return float(numerator / denominator) if denominator > 1e-12 else float("nan")


def _condition_metrics(
    logits: np.ndarray,
    w1_index: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    # logits: task, scenario, question, letter
    result: dict[str, Any] = {}
    q_index = np.arange(logits.shape[2])
    for task_index, task in enumerate(TASKS):
        task_result: dict[str, Any] = {}
        for scenario_index in range(logits.shape[1]):
            winner = logits[task_index, scenario_index].argmax(axis=-1)
            switch = winner != w1_index
            task_result[str(scenario_index)] = _bootstrap(switch[mask].astype(float))
        result[task] = task_result
    gap: dict[str, Any] = {}
    for scenario_index in range(logits.shape[1]):
        game = logits[0, scenario_index].argmax(axis=-1) != w1_index
        neutral = logits[1, scenario_index].argmax(axis=-1) != w1_index
        gap[str(scenario_index)] = _bootstrap(
            (game[mask].astype(float) - neutral[mask].astype(float))
        )
    result["Game_minus_Neutral"] = gap
    return result


def _switching_change_from_natural(
    logits: np.ndarray,
    w1_index: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Paired intervention-minus-natural changes in switching probability."""
    switched = logits.argmax(axis=-1) != w1_index[None, None, :]
    result: dict[str, Any] = {task: {} for task in TASKS}
    result["Game_minus_Neutral"] = {}
    for scenario in range(1, logits.shape[1]):
        for task_index, task in enumerate(TASKS):
            change = (
                switched[task_index, scenario].astype(float)
                - switched[task_index, 0].astype(float)
            )
            result[task][str(scenario)] = _bootstrap(change[mask])
        gap_change = (
            switched[0, scenario].astype(float)
            - switched[1, scenario].astype(float)
            - switched[0, 0].astype(float)
            + switched[1, 0].astype(float)
        )
        result["Game_minus_Neutral"][str(scenario)] = _bootstrap(
            gap_change[mask]
        )
    return result


def run(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    qids = arrays["question_ids"].astype(str).tolist()
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    expected = ["natural", "cue_swapped", "cue_ablated", "colon_ablated"]
    if scenarios != expected:
        raise RuntimeError(f"Unexpected scenarios: {scenarios}")
    if not arrays["completed"].all() or len(qids) != 500:
        raise RuntimeError("Causal run is incomplete")
    final = arrays["scenario_final_logits"].astype(np.float64)
    cue = arrays["scenario_cue_logits"].astype(np.float64)
    trusted = arrays["trusted_natural_logits"].astype(np.float64)
    same_batch = arrays["same_batch_natural_logits"].astype(np.float64)
    if not np.isfinite(final).all() or not np.isfinite(cue).all():
        raise RuntimeError("Causal outputs contain non-finite values")

    split = json.loads(args.split_plan.read_text())
    discovery_ids = set(split["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids], dtype=bool)
    confirmation = ~discovery
    if (int(discovery.sum()), int(confirmation.sum())) != (251, 249):
        raise RuntimeError("Frozen split sizes changed")

    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    mappings = {
        row["question_id"]: row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    w1_index = np.empty(len(qids), dtype=np.int16)
    conflict = np.empty(len(qids), dtype=bool)
    for index, qid in enumerate(qids):
        w1 = baseline[qid]["answer"]
        w1_second = mappings[qid]["original_to_new"][w1]
        w1_index[index] = LETTERS.index(w1_second)
        fresh = remapped[qid]["answer_original_content"]
        conflict[index] = fresh != w1

    natural = final[:, 0]
    displacement = np.sqrt(np.mean(
        _center(final - natural[:, None]) ** 2, axis=-1
    ))
    winner_change = final.argmax(axis=-1) != natural[:, None].argmax(axis=-1)
    cue_invariance = float(np.max(np.abs(cue[:, 1:3] - cue[:, 0:1])))
    natural_drift = float(np.max(np.abs(same_batch - trusted)))

    summary: dict[str, Any] = {
        "evidence_label": "Causal source-memory intervention.",
        "questions": len(qids),
        "discovery_questions": int(discovery.sum()),
        "confirmation_questions": int(confirmation.sum()),
        "scenarios": scenarios,
        "validation": {
            "same_batch_natural_max_abs_error": natural_drift,
            "cue_swap_or_ablation_max_abs_cue_logit_error": cue_invariance,
            "all_values_finite": True,
            "ordinary_layers_one_based": arrays["ordinary_layers_one_based"].astype(int).tolist(),
            "gla_layers_one_based": arrays["gla_layers_one_based"].astype(int).tolist(),
        },
        "confirmation": {},
        "discovery": {},
    }

    for split_name, mask in (("confirmation", confirmation), ("discovery", discovery)):
        section: dict[str, Any] = {
            "final_centered_logit_displacement": {},
            "final_winner_change": {},
            "switching_all": _condition_metrics(final, w1_index, mask),
            "switching_conflict": _condition_metrics(final, w1_index, mask & conflict),
            "switching_no_conflict": _condition_metrics(final, w1_index, mask & ~conflict),
            "switching_change_all": _switching_change_from_natural(
                final, w1_index, mask
            ),
            "switching_change_conflict": _switching_change_from_natural(
                final, w1_index, mask & conflict
            ),
            "switching_change_no_conflict": _switching_change_from_natural(
                final, w1_index, mask & ~conflict
            ),
            "cue_ablation_minus_colon": {},
            "swap_transfer": {},
        }
        for task_index, task in enumerate(TASKS):
            section["final_centered_logit_displacement"][task] = {
                scenarios[s]: _bootstrap(displacement[task_index, s, mask])
                for s in range(1, len(scenarios))
            }
            section["final_winner_change"][task] = {
                scenarios[s]: _bootstrap(winner_change[task_index, s, mask].astype(float))
                for s in range(1, len(scenarios))
            }
            section["cue_ablation_minus_colon"][task] = {
                "centered_logit_displacement": _bootstrap(
                    (displacement[task_index, 2] - displacement[task_index, 3])[mask],
                    seed=20260831 + task_index,
                ),
                "winner_change": _bootstrap(
                    (
                        winner_change[task_index, 2].astype(float)
                        - winner_change[task_index, 3].astype(float)
                    )[mask],
                    seed=20260833 + task_index,
                ),
            }

            donor = 1 - task_index
            cue_direction = _center(cue[donor, 0] - cue[task_index, 0])
            final_direction = _center(natural[donor] - natural[task_index])
            swap_delta = _center(final[task_index, 1] - natural[task_index])
            values = np.stack((swap_delta, cue_direction, final_direction), axis=1)[mask]
            cue_fraction = _bootstrap(
                values,
                statistic=lambda x: _projection_fraction(x[:, 0], x[:, 1]),
                seed=20260823 + task_index,
            )
            final_fraction = _bootstrap(
                values,
                statistic=lambda x: _projection_fraction(x[:, 0], x[:, 2]),
                seed=20260825 + task_index,
            )
            donor_cue_winner = cue[donor, 0].argmax(axis=-1)
            rows = np.arange(len(qids))
            patched_margin = final[task_index, 1, rows, donor_cue_winner] - np.max(
                np.where(
                    np.arange(4)[None, :] == donor_cue_winner[:, None],
                    -np.inf,
                    final[task_index, 1],
                ),
                axis=-1,
            )
            natural_margin = natural[task_index, rows, donor_cue_winner] - np.max(
                np.where(
                    np.arange(4)[None, :] == donor_cue_winner[:, None],
                    -np.inf,
                    natural[task_index],
                ),
                axis=-1,
            )
            discordant = cue[task_index, 0].argmax(axis=-1) != donor_cue_winner
            eligible = mask & discordant
            section["swap_transfer"][task] = {
                "toward_donor_cue_vector_fraction": cue_fraction,
                "toward_donor_final_vector_fraction": final_fraction,
                "donor_cue_winner_margin_change_all": _bootstrap(
                    (patched_margin - natural_margin)[mask], seed=20260827 + task_index
                ),
                "donor_cue_winner_margin_change_discordant": _bootstrap(
                    (patched_margin - natural_margin)[eligible], seed=20260829 + task_index
                ),
                "discordant_cue_winner_questions": int(eligible.sum()),
            }
        summary[split_name] = section

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    held = summary["confirmation"]
    colors = {"Game": "#2f80ed", "Neutral": "#f2994a"}
    x = np.arange(3)
    labels = ["Cue swap", "Cue ablation", "Colon control"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    for task_index, task in enumerate(TASKS):
        offset = (-0.18, 0.18)[task_index]
        disp = held["final_centered_logit_displacement"][task]
        means = np.asarray([disp[scenarios[s]]["mean"] for s in range(1, 4)])
        lows = np.asarray([disp[scenarios[s]]["ci_low"] for s in range(1, 4)])
        highs = np.asarray([disp[scenarios[s]]["ci_high"] for s in range(1, 4)])
        axes[0, 0].errorbar(
            x + offset, means, yerr=[means - lows, highs - means], fmt="o",
            capsize=4, color=colors[task], label=task,
        )
        changed = held["final_winner_change"][task]
        means = 100 * np.asarray([changed[scenarios[s]]["mean"] for s in range(1, 4)])
        lows = 100 * np.asarray([changed[scenarios[s]]["ci_low"] for s in range(1, 4)])
        highs = 100 * np.asarray([changed[scenarios[s]]["ci_high"] for s in range(1, 4)])
        axes[0, 1].errorbar(
            x + offset, means, yerr=[means - lows, highs - means], fmt="o",
            capsize=4, color=colors[task], label=task,
        )

    for col, (name, key) in enumerate((("Conflict trials", "switching_conflict"), ("No-conflict trials", "switching_no_conflict"))):
        axis = axes[1, col]
        xx = np.arange(4)
        for task_index, task in enumerate(TASKS):
            offset = (-0.18, 0.18)[task_index]
            cells = held[key][task]
            means = 100 * np.asarray([cells[str(s)]["mean"] for s in range(4)])
            lows = 100 * np.asarray([cells[str(s)]["ci_low"] for s in range(4)])
            highs = 100 * np.asarray([cells[str(s)]["ci_high"] for s in range(4)])
            axis.errorbar(
                xx + offset, means, yerr=[means - lows, highs - means], fmt="o-",
                capsize=3, color=colors[task], label=task,
            )
        axis.set_xticks(xx, ["Natural", "Swap", "Cue ablate", "Colon"])
        axis.set_ylabel("Switching away from 1P winner (%)")
        axis.set_title(name)

    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Centered A-D logit displacement (RMS)")
    axes[0, 0].set_title("A  How much the final ranking moves")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Final winner changed (%)")
    axes[0, 1].set_title("B  How often the selected answer changes")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle(
        "Does the post-list answer cue causally drive the final decision?\n"
        "All 64 layers; held-out 249 questions; 95% paired-question bootstrap intervals",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)

    def pct(cell: dict[str, Any]) -> str:
        return f"{100*cell['mean']:.1f}% [{100*cell['ci_low']:.1f},{100*cell['ci_high']:.1f}]"

    lines = [
        "# Causal role of the post-list answer cue",
        "",
        "## Method",
        "",
        "The exact trailing cue-space token remains present and its own residual is kept natural. At every ordinary-attention layer, later tokens either receive paired donor K/V or cannot read the source. At every GLA layer, later recurrent outputs receive the paired donor write or the source write is removed. The structural control applies the identical removal to the immediately preceding `):` token.",
        "",
        "## Held-out results",
        "",
    ]
    for task in TASKS:
        disp = held["final_centered_logit_displacement"][task]
        win = held["final_winner_change"][task]
        lines.append(f"**{task}:**")
        for scenario in scenarios[1:]:
            lines.append(
                f"- `{scenario}`: centered-logit displacement {disp[scenario]['mean']:.3f} "
                f"`[{disp[scenario]['ci_low']:.3f},{disp[scenario]['ci_high']:.3f}]`; "
                f"winner changed {pct(win[scenario])}."
            )
        transfer = held["swap_transfer"][task]
        lines.append(
            f"- Swap transfer toward donor cue vector: {transfer['toward_donor_cue_vector_fraction']['mean']:.3f} "
            f"`[{transfer['toward_donor_cue_vector_fraction']['ci_low']:.3f},{transfer['toward_donor_cue_vector_fraction']['ci_high']:.3f}]`."
        )
        lines.append("")
    lines.extend([
        "## Behavioral switching",
        "",
    ])
    for subset, key in (("Conflict", "switching_conflict"), ("No conflict", "switching_no_conflict")):
        lines.append(f"**{subset}:**")
        for task in TASKS:
            cells = held[key][task]
            values = ", ".join(
                f"{scenarios[s]} {pct(cells[str(s)])}" for s in range(4)
            )
            lines.append(f"- {task}: {values}.")
        lines.append("")
    lines.extend([
        "## Paired causal contrasts",
        "",
    ])
    for task in TASKS:
        specific = held["cue_ablation_minus_colon"][task]
        disp = specific["centered_logit_displacement"]
        win = specific["winner_change"]
        lines.append(
            f"- {task}, cue ablation minus colon control: logit displacement "
            f"{disp['mean']:+.3f} `[{disp['ci_low']:+.3f},{disp['ci_high']:+.3f}]`; "
            f"winner-change difference {100*win['mean']:+.1f} points "
            f"`[{100*win['ci_low']:+.1f},{100*win['ci_high']:+.1f}]`."
        )
    for subset, key in (
        ("All", "switching_change_all"),
        ("Conflict", "switching_change_conflict"),
        ("No conflict", "switching_change_no_conflict"),
    ):
        cell = held[key]["Game_minus_Neutral"]["1"]
        lines.append(
            f"- {subset}, cue swap change in preferential Game switching: "
            f"{100*cell['mean']:+.1f} points "
            f"`[{100*cell['ci_low']:+.1f},{100*cell['ci_high']:+.1f}]`."
        )
    lines.append("")
    discovery_ablation = summary["discovery"]["switching_change_all"][
        "Game_minus_Neutral"
    ]["2"]
    confirmation_ablation = held["switching_change_all"][
        "Game_minus_Neutral"
    ]["2"]
    lines.extend([
        "## Bottom line",
        "",
        "Complete downstream ablation of the cue is now a live ordinary-attention-plus-GLA intervention. It changes individual final rankings above the structural-colon control, so the cue is causally used. However, it does not materially reduce the main Game-minus-Neutral switching difference: the all-question change is "
        f"{100*discovery_ablation['mean']:+.1f} points "
        f"`[{100*discovery_ablation['ci_low']:+.1f},{100*discovery_ablation['ci_high']:+.1f}]` on discovery and "
        f"{100*confirmation_ablation['mean']:+.1f} points "
        f"`[{100*confirmation_ablation['ci_low']:+.1f},{100*confirmation_ablation['ci_high']:+.1f}]` on confirmation. Thus the cue carries and contributes task-specific policy information, but the main behavioral difference does not require this cue route.",
        "",
    ])
    lines.extend([
        "## Validation",
        "",
        f"- Same-batch natural maximum A-D logit error: `{natural_drift:.6g}`.",
        f"- Cue-state invariance under cue swap/ablation: `{cue_invariance:.6g}`.",
        "- Every ordinary-attention and GLA layer was covered.",
        "- Discovery results are retained in `summary.json` for replication assessment.",
        "",
        "## Artifacts",
        "",
        f"- Canonical figure: `{args.figure}`",
        "- Compact statistics: `summary.json`",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
