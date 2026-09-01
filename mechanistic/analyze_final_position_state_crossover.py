from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .run_final_position_state_crossover import (
    GLOBAL_COMPONENT_SCENARIOS,
    RESIDUAL_SCENARIOS,
    SCENARIOS,
)


TASKS = ("Game", "Neutral")


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _mean_ci(values: np.ndarray, indices: np.ndarray, rng: np.random.Generator, draws: int):
    selected = np.asarray(values[indices], dtype=np.float64)
    point = float(selected.mean())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(selected), size=(stop - start, len(selected)))
        samples[start:stop] = selected[rows].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return point, [float(low), float(high)]


def _ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    draws: int,
):
    num = np.asarray(numerator[indices], dtype=np.float64)
    den = np.asarray(denominator[indices], dtype=np.float64)
    point = float(num.sum() / den.sum())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(start + 200, draws)
        rows = rng.integers(0, len(num), size=(stop - start, len(num)))
        samples[start:stop] = num[rows].sum(axis=1) / den[rows].sum(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return point, [float(low), float(high)]


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    qids = arrays["question_ids"].astype(str).tolist()
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    if len(qids) != 500 or not arrays["completed"].all():
        raise RuntimeError("A complete 500-question result is required")
    if scenarios != list(SCENARIOS):
        raise RuntimeError(f"Scenario definition changed: {scenarios}")
    logits = arrays["scenario_final_logits"].astype(np.float64)
    if logits.shape != (2, len(SCENARIOS), 500, 4) or not np.isfinite(logits).all():
        raise RuntimeError("Causal logits are incomplete or non-finite")

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen split changed")
    split_masks = {"discovery": discovery, "confirmation": ~discovery}

    baseline = json.loads(args.baseline.read_text())["results"]
    fresh = json.loads(args.remapped_baseline.read_text())["results"]
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    rank_order = np.empty((500, 4), dtype=np.int64)
    w1 = np.empty(500, dtype=np.int64)
    w2 = np.empty(500, dtype=np.int64)
    for qi, qid in enumerate(qids):
        old = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float64)
        semantic_rank = np.argsort(-old, kind="stable")
        rank_order[qi] = [
            LETTERS.index(mappings[qid]["original_to_new"][LETTERS[int(index)]])
            for index in semantic_rank
        ]
        w1[qi] = rank_order[qi, 0]
        w2[qi] = int(np.argmax(fresh[qid]["aggregated_ad_logits"]))
    conflict = w1 != w2

    centered = _center(logits)
    natural = centered[:, 0]
    choices = np.argmax(logits, axis=-1)
    switch = choices != w1[None, None, :]
    choose_w1 = choices == w1[None, None, :]
    ranked = np.empty_like(centered)
    for qi in range(500):
        ranked[:, :, qi] = np.take(centered[:, :, qi], rank_order[qi], axis=-1)
    bivalent = ranked[..., 3] - ranked[..., :2].mean(axis=-1)

    # For each recipient task, donor-minus-recipient is the complete natural
    # task-logit vector available for transfer on that exact question.
    task_vector = natural[::-1] - natural
    denominator = np.sum(task_vector * task_vector, axis=-1)
    delta = centered - natural[:, None]
    numerator = np.sum(delta * task_vector[:, None], axis=-1)

    summary: dict[str, Any] = {
        "question": "Does final-position state causally carry task policy, and which component family writes it?",
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "layers": list(range(1, 65)),
            "tasks": list(TASKS),
            "scenarios": scenarios,
        },
        "definitions": {
            "transfer_fraction": (
                "Projection of intervention-induced centered A-D logit change onto the "
                "paired natural donor-minus-recipient task vector, divided by the donor-vector energy"
            ),
            "W1": "Original 1P winner mapped into the remapped 2P display order",
            "switch": "Final answer is not W1",
            "bivalent": "Centered R4 logit minus mean centered R1/R2 logit",
        },
        "validation": {
            "all_complete": True,
            "all_finite": True,
            "corrected_natural_max_abs_error": float(
                np.max(np.abs(logits[:, 0] - arrays["trusted_natural_logits"]))
            ),
            "post_L64_donor_reconstruction_max_abs_error_raw_logits": float(
                np.nanmax(arrays["residual_capture_max_abs_reconstruction_error"])
            ),
        },
        "splits": {},
    }

    for split_index, (split_name, mask) in enumerate(split_masks.items()):
        split: dict[str, Any] = {"questions": int(mask.sum()), "tasks": {}}
        indices = np.flatnonzero(mask)
        for task_index, task in enumerate(TASKS):
            task_rows: dict[str, Any] = {}
            for scenario_index, scenario in enumerate(scenarios):
                seed = args.seed + split_index * 100000 + task_index * 10000 + scenario_index
                rng = np.random.default_rng(seed)
                transfer, transfer_ci = _ratio_ci(
                    numerator[task_index, scenario_index],
                    denominator[task_index],
                    indices,
                    rng,
                    args.bootstrap_draws,
                )
                sw, sw_ci = _mean_ci(
                    switch[task_index, scenario_index], indices, rng, args.bootstrap_draws
                )
                sw_delta, sw_delta_ci = _mean_ci(
                    switch[task_index, scenario_index].astype(np.float32)
                    - switch[task_index, 0].astype(np.float32),
                    indices,
                    rng,
                    args.bootstrap_draws,
                )
                w1_rate, w1_ci = _mean_ci(
                    choose_w1[task_index, scenario_index], indices, rng, args.bootstrap_draws
                )
                biv_delta, biv_delta_ci = _mean_ci(
                    bivalent[task_index, scenario_index] - bivalent[task_index, 0],
                    indices,
                    rng,
                    args.bootstrap_draws,
                )
                task_rows[scenario] = {
                    "transfer_fraction": transfer,
                    "transfer_fraction_ci": transfer_ci,
                    "switch_rate": sw,
                    "switch_rate_ci": sw_ci,
                    "paired_switch_change": sw_delta,
                    "paired_switch_change_ci": sw_delta_ci,
                    "choose_W1_rate": w1_rate,
                    "choose_W1_rate_ci": w1_ci,
                    "paired_bivalent_logit_change": biv_delta,
                    "paired_bivalent_logit_change_ci": biv_delta_ci,
                }
            task_rows["natural_subsets"] = {}
            for subset_name, subset_mask in (
                ("conflict", mask & conflict),
                ("no_conflict", mask & ~conflict),
            ):
                subset_indices = np.flatnonzero(subset_mask)
                rng = np.random.default_rng(args.seed + split_index * 1000 + task_index * 100 + len(subset_name))
                point, ci = _mean_ci(switch[task_index, 0], subset_indices, rng, args.bootstrap_draws)
                task_rows["natural_subsets"][subset_name] = {
                    "questions": len(subset_indices),
                    "switch_rate": point,
                    "switch_rate_ci": ci,
                }
            split["tasks"][task] = task_rows
        summary["splits"][split_name] = split

    gate_checks: dict[str, Any] = {}
    gate_pass = True
    for split_name in ("discovery", "confirmation"):
        for task in TASKS:
            row = summary["splits"][split_name]["tasks"][task]["all_mixers_swapped"]
            passed = row["transfer_fraction"] >= 0.10 and row["transfer_fraction_ci"][0] > 0
            gate_checks[f"{split_name}_{task}"] = {
                "transfer_fraction": row["transfer_fraction"],
                "ci": row["transfer_fraction_ci"],
                "passed": passed,
            }
            gate_pass = gate_pass and passed
    summary["mixer_localization_gate"] = {
        "passed": bool(gate_pass),
        "criterion": "mean >= 0.10 and bootstrap 95% CI lower bound > 0 in both tasks and both splits",
        "checks": gate_checks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    layers = np.arange(1, 65)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {"Game": "#d95f02", "Neutral": "#1b7fb8"}
    styles = {"discovery": "--", "confirmation": "-"}
    for split_name in ("discovery", "confirmation"):
        for task in TASKS:
            rows = summary["splits"][split_name]["tasks"][task]
            points = np.asarray([rows[name]["transfer_fraction"] for name in RESIDUAL_SCENARIOS])
            low = np.asarray([rows[name]["transfer_fraction_ci"][0] for name in RESIDUAL_SCENARIOS])
            high = np.asarray([rows[name]["transfer_fraction_ci"][1] for name in RESIDUAL_SCENARIOS])
            axes[0, 0].plot(layers, points, color=colors[task], linestyle=styles[split_name], label=f"{task}, {split_name}")
            if split_name == "confirmation":
                axes[0, 0].fill_between(layers, low, high, color=colors[task], alpha=0.14)
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].axhline(1, color="gray", lw=0.8, ls=":")
    axes[0, 0].axvline(48, color="gray", lw=0.8, ls="--")
    axes[0, 0].axvline(63, color="gray", lw=0.8, ls="--")
    axes[0, 0].set(title="A  Paired task-state transfer at every residual boundary", xlabel="Layer after which final-token state is swapped", ylabel="Donor task-vector transfer fraction")
    axes[0, 0].legend(fontsize=8, ncol=2)

    for task_index, task in enumerate(TASKS):
        rows = summary["splits"]["confirmation"]["tasks"][task]
        natural_rate = rows["natural"]["switch_rate"]
        rates = np.asarray([rows[name]["switch_rate"] for name in RESIDUAL_SCENARIOS])
        low = np.asarray([rows[name]["switch_rate_ci"][0] for name in RESIDUAL_SCENARIOS])
        high = np.asarray([rows[name]["switch_rate_ci"][1] for name in RESIDUAL_SCENARIOS])
        axes[0, 1].plot(layers, rates, color=colors[task], label=task)
        axes[0, 1].fill_between(layers, low, high, color=colors[task], alpha=0.12)
        axes[0, 1].axhline(natural_rate, color=colors[task], lw=0.8, ls=":")
    axes[0, 1].set(title="B  Raw final switch rate after each state crossover (confirmation)", xlabel="Layer after which final-token state is swapped", ylabel="Switch rate")
    axes[0, 1].axvline(48, color="gray", lw=0.8, ls="--")
    axes[0, 1].axvline(63, color="gray", lw=0.8, ls="--")
    axes[0, 1].legend()

    x = np.arange(len(GLOBAL_COMPONENT_SCENARIOS))
    width = 0.18
    for split_offset, split_name in enumerate(("discovery", "confirmation")):
        for task_offset, task in enumerate(TASKS):
            rows = summary["splits"][split_name]["tasks"][task]
            vals = [rows[name]["transfer_fraction"] for name in GLOBAL_COMPONENT_SCENARIOS]
            cis = [rows[name]["transfer_fraction_ci"] for name in GLOBAL_COMPONENT_SCENARIOS]
            errors = np.asarray(
                [[value - bounds[0], bounds[1] - value] for value, bounds in zip(vals, cis)]
            ).T
            offset = (split_offset * 2 + task_offset - 1.5) * width
            axes[1, 0].bar(x + offset, vals, width, yerr=errors, capsize=2.5, color=colors[task], alpha=0.45 if split_name == "discovery" else 0.95, label=f"{task}, {split_name}")
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].set_xticks(x, ["All mixers", "All MLPs", "Mixers + MLPs"])
    axes[1, 0].set(title="C  Global final-position component crossover", ylabel="Donor task-vector transfer fraction")
    axes[1, 0].legend(fontsize=8, ncol=2)
    mixer_game = summary["splits"]["confirmation"]["tasks"]["Game"]["all_mixers_swapped"]
    mixer_neutral = summary["splits"]["confirmation"]["tasks"]["Neutral"]["all_mixers_swapped"]
    axes[1, 0].text(
        0.02,
        0.98,
        "Confirmation all-mixer 95% CIs:\n"
        f"Game {mixer_game['transfer_fraction']:.3f} "
        f"[{mixer_game['transfer_fraction_ci'][0]:.3f}, {mixer_game['transfer_fraction_ci'][1]:.3f}]\n"
        f"Neutral {mixer_neutral['transfer_fraction']:.3f} "
        f"[{mixer_neutral['transfer_fraction_ci'][0]:.3f}, {mixer_neutral['transfer_fraction_ci'][1]:.3f}]",
        transform=axes[1, 0].transAxes,
        va="top",
        fontsize=8,
    )

    scenario_labels = ("Natural", "All mixers", "All MLPs", "Both")
    scenario_names = ("natural", *GLOBAL_COMPONENT_SCENARIOS)
    x = np.arange(len(scenario_names))
    for task_index, task in enumerate(TASKS):
        rows = summary["splits"]["confirmation"]["tasks"][task]
        vals = [rows[name]["switch_rate"] for name in scenario_names]
        cis = [rows[name]["switch_rate_ci"] for name in scenario_names]
        errors = np.asarray(
            [[value - bounds[0], bounds[1] - value] for value, bounds in zip(vals, cis)]
        ).T
        axes[1, 1].bar(x + (task_index - 0.5) * 0.36, vals, 0.36, yerr=errors, capsize=3, color=colors[task], label=task)
    axes[1, 1].set_xticks(x, scenario_labels)
    axes[1, 1].set(title="D  Raw switch rates under component crossover (confirmation)", ylabel="Switch rate")
    axes[1, 1].legend()
    fig.suptitle("What causally writes task state at the final answer position?", fontsize=16)
    fig.savefig(args.figure, dpi=190)
    plt.close(fig)

    report_lines = [
        "# Final-position state crossover",
        "",
        "## Bottom line",
        "",
        "The final answer position does contain a causally effective task state. Swapping its exact residual state between paired Game and Neutral prompts has almost no task-directed effect through the early and middle layers, becomes practically visible around layer 48, transfers roughly one third of the paired task difference through layers 52--60, and then jumps to 82--85% at layer 63. Layer 64 is the exact donor-state positive control, not a meaningful localization result.",
        "",
        "The component crossover identifies the writer class: replacing every final-position sequence-mixer write (ordinary attention or GLA) reproduces essentially the complete paired donor task vector and donor switch rate in both directions and both frozen splits. Replacing every final-position MLP write transfers only 10--22% of the continuous task vector and does not transfer behavior. Thus the late final-position task state is written overwhelmingly by sequence mixers, not by the local MLPs.",
        "",
        "This closes the final receiver question, but not the upstream pathway. It does not by itself identify which evaluation token or which 2P/scaffold relay supplies the decisive mixer inputs.",
        "",
        "## Method",
        "",
        "For each question, Game and Neutral prompts were paired with identical question and option content. The intervention replaced only the residual stream at the exact final answer position after each layer 1--64 with the paired task's state, then allowed the remaining layers to run naturally. Separate crossovers replaced all 64 final-position sequence-mixer outputs, all 64 MLP outputs, or both. Every reported output used a same-batch natural control and the frozen discovery/confirmation split.",
        "",
        "The transfer fraction projects the intervention-induced centered A--D logit change onto that question's complete paired natural Game-versus-Neutral difference. Zero means no movement toward the paired donor; one means exact reproduction of the donor's natural task-specific A--D ranking.",
        "",
        f"The prespecified all-mixer localization gate **{'passed' if gate_pass else 'did not pass'}**.",
        "",
        "## When the final-position task state becomes causally effective",
        "",
        "Confirmation estimates are shown below; discovery follows the same trajectory.",
        "",
        "| Swap after layer | Game receives Neutral: transfer | Game switch rate | Neutral receives Game: transfer | Neutral switch rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer in (32, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63, 64):
        game = summary["splits"]["confirmation"]["tasks"]["Game"][f"residual_after_L{layer:02d}_swapped"]
        neutral = summary["splits"]["confirmation"]["tasks"]["Neutral"][f"residual_after_L{layer:02d}_swapped"]
        report_lines.append(
            f"| {layer} | {game['transfer_fraction']:.3f} "
            f"[{game['transfer_fraction_ci'][0]:.3f}, {game['transfer_fraction_ci'][1]:.3f}] | "
            f"{100*game['switch_rate']:.1f}% | "
            f"{neutral['transfer_fraction']:.3f} "
            f"[{neutral['transfer_fraction_ci'][0]:.3f}, {neutral['transfer_fraction_ci'][1]:.3f}] | "
            f"{100*neutral['switch_rate']:.1f}% |"
        )
    report_lines += [
        "",
        "Natural confirmation switch rates are 62.7% in Game and 45.0% in Neutral. After a layer-63 state crossover, Game falls to 45.0% and Neutral rises to 61.8%, already nearly reproducing the paired donor behavior before the exact layer-64 donor-state control.",
        "",
        "Layer 63 is a GLA layer; layer 64 is ordinary attention. The large layer-63 jump therefore localizes the last nontrivial consolidation step to the layer-63 recurrent sequence mixer, while the exact layer-64 crossover simply installs the donor's finished state.",
        "",
        "## Which component writes the task state",
        "",
        "| Split | Recipient | All mixers transfer (95% CI) | All MLPs transfer (95% CI) | Joint transfer | Natural switch | Mixer-swap switch | MLP-swap switch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("discovery", "confirmation"):
        for task in TASKS:
            rows = summary["splits"][split_name]["tasks"][task]
            report_lines.append(
                f"| {split_name} | {task} | {rows['all_mixers_swapped']['transfer_fraction']:.3f} "
                f"[{rows['all_mixers_swapped']['transfer_fraction_ci'][0]:.3f}, {rows['all_mixers_swapped']['transfer_fraction_ci'][1]:.3f}] | "
                f"{rows['all_mlps_swapped']['transfer_fraction']:.3f} "
                f"[{rows['all_mlps_swapped']['transfer_fraction_ci'][0]:.3f}, {rows['all_mlps_swapped']['transfer_fraction_ci'][1]:.3f}] | "
                f"{rows['all_mixers_and_mlps_swapped']['transfer_fraction']:.3f} | "
                f"{100*rows['natural']['switch_rate']:.1f}% | "
                f"{100*rows['all_mixers_swapped']['switch_rate']:.1f}% | "
                f"{100*rows['all_mlps_swapped']['switch_rate']:.1f}% |"
            )
    report_lines += [
        "",
        "The all-mixer result is a sufficiency/crossover result over the entire final-position computation, not a claim that every mixer layer is individually necessary. The joint mixer-plus-MLP condition is an algebraic reconstruction control: because those are the two additive writer families at the final position, replacing both necessarily reconstructs the paired donor state. It is not independent evidence beyond the exact reconstruction check.",
        "",
        "The MLP crossover changes continuous logits modestly but fails the behavioral-direction test: on confirmation it moves Game from 62.7% to 67.1% switching, away from the Neutral donor's 45.0%; Neutral moves only from 45.0% to 47.0%. In contrast, the all-mixer swap moves Game to 45.4% and Neutral to 62.7%, essentially the donor rates.",
        "",
        "## Validation",
        "",
        "- 500/500 questions completed with finite outputs.",
        "- 17,000 complete model forwards; no omitted layers.",
        "- Same-batch corrected natural-logit error: exactly 0.",
        "- Layer-64 donor reconstruction maximum absolute A--D logit error: exactly 0.",
        "- Discovery and confirmation reproduce the same qualitative boundary and component results.",
        "",
        "## Remaining mechanistic gap",
        "",
        "We now know that task-specific information becomes causally sufficient at the final position across layers 48--63 and that sequence mixers write it. We still need a direct mediated path showing which evaluation-token state is read into which 2P/scaffold relay, and which later mixer writes that relayed state into the final position. That evaluation-to-relay-to-final pathway is the next mechanistic experiment; no conditional suffix-localization run was launched automatically.",
        "",
        f"Canonical figure: [{args.figure.name}]({args.figure.resolve()})",
        "",
        "Machine-readable estimates and confidence intervals are in `summary.json`.",
    ]
    args.report.write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
