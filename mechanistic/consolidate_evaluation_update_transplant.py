from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"


def interval(values: np.ndarray, strata: np.ndarray, rng, draws=20_000):
    values = np.asarray(values, dtype=float)
    samples = np.zeros(draws)
    for label in np.unique(strata):
        group = np.flatnonzero(strata == label)
        chosen = rng.choice(group, (draws, len(group)), replace=True)
        samples += values[chosen].sum(axis=1)
    samples /= len(values)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def ratio_interval(
    numerator: np.ndarray, denominator: np.ndarray, strata: np.ndarray, rng, draws=20_000
):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    num_samples = np.zeros(draws)
    den_samples = np.zeros(draws)
    for label in np.unique(strata):
        group = np.flatnonzero(strata == label)
        chosen = rng.choice(group, (draws, len(group)), replace=True)
        num_samples += numerator[chosen].sum(axis=1)
        den_samples += denominator[chosen].sum(axis=1)
    samples = np.divide(
        num_samples, den_samples,
        out=np.full(draws, np.nan),
        where=den_samples != 0,
    )
    low, high = np.nanquantile(samples, (0.025, 0.975))
    return {
        "mean": float(numerator.sum() / denominator.sum()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def fmt(cell, scale=1.0, digits=3):
    return (
        f"{cell['mean'] * scale:+.{digits}f} "
        f"[{cell['ci_low'] * scale:+.{digits}f}, "
        f"{cell['ci_high'] * scale:+.{digits}f}]"
    )


def entropy(logits):
    shifted = logits - logits.max(-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(-1, keepdims=True)
    return -(p * np.log2(np.maximum(p, 1e-300))).sum(-1)


def load_split(path, baseline, remapped_baseline, plan, split_name):
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    qids = arrays["question_ids"].astype(str).tolist()
    scenarios = arrays["scenario_ids"].astype(str).tolist()

    def align(values):
        output = np.empty_like(values)
        for qi, qid in enumerate(qids):
            original_to_new = plan[qid]["original_to_new"]
            output[..., qi, :] = np.stack(
                [
                    values[..., qi, LETTERS.index(original_to_new[original])]
                    for original in LETTERS
                ],
                axis=-1,
            )
        return output

    natural = align(arrays["trusted_natural_logits"].astype(float))
    patched = align(arrays["patched_logits"].astype(float))
    w1 = np.asarray([baseline[qid]["answer"] for qid in qids])
    w2 = np.asarray([remapped_baseline[qid]["answer_original_content"] for qid in qids])
    conflict = w1 != w2
    w1i = np.asarray([LETTERS.index(x) for x in w1])
    w2i = np.asarray([LETTERS.index(x) for x in w2])
    qi = np.arange(len(qids))
    mapping_rows = [plan[qid] for qid in qids]
    natural_answer = displayed_argmax_to_semantic_indices(
        arrays["trusted_natural_logits"], mapping_rows
    )
    patched_answer = displayed_argmax_to_semantic_indices(
        arrays["patched_logits"], mapping_rows
    )

    def margin(values):
        return values[..., qi, w1i] - values[..., qi, w2i]

    effects = {}
    for si, scenario in enumerate(scenarios):
        effects[scenario] = {
            "e2n_margin": margin(natural[1]) - margin(patched[0, si]),
            "n2e_margin": margin(patched[1, si]) - margin(natural[0]),
            "e2n_w1": (natural_answer[1] == w1i).astype(float)
            - (patched_answer[0, si] == w1i).astype(float),
            "n2e_w1": (patched_answer[1, si] == w1i).astype(float)
            - (natural_answer[0] == w1i).astype(float),
            "e2n_entropy": entropy(patched[0, si]) - entropy(natural[1]),
            "n2e_entropy": entropy(natural[0]) - entropy(patched[1, si]),
        }
    natural_effect = {
        "margin": margin(natural[1]) - margin(natural[0]),
        "w1": (natural_answer[1] == w1i).astype(float)
        - (natural_answer[0] == w1i).astype(float),
        "entropy": entropy(natural[0]) - entropy(natural[1]),
    }
    return {
        "split": split_name,
        "qids": qids,
        "w1": w1,
        "conflict": conflict,
        "scenarios": scenarios,
        "effects": effects,
        "natural_effect": natural_effect,
    }


def analyze(args):
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    plan = {row["question_id"]: row for row in json.loads(args.remapping_plan.read_text())["rows"]}
    gate = [
        load_split(args.discovery_gate, baseline, remapped, plan, "discovery"),
        load_split(args.confirmation_gate, baseline, remapped, plan, "confirmation"),
    ]
    blocks = [
        load_split(args.discovery_blocks, baseline, remapped, plan, "discovery"),
        load_split(args.confirmation_blocks, baseline, remapped, plan, "confirmation"),
    ]
    bands = [
        load_split(args.discovery_bands, baseline, remapped, plan, "discovery"),
        load_split(args.confirmation_bands, baseline, remapped, plan, "confirmation"),
    ]
    rng = np.random.default_rng(args.seed)

    def combined(source, getter):
        values, strata = [], []
        for split in source:
            mask = split["conflict"]
            values.append(np.asarray(getter(split))[mask])
            strata.extend([f"{split['split']}:{letter}" for letter in split["w1"][mask]])
        return np.concatenate(values), np.asarray(strata)

    summary = {"n_conflict": int(sum(x["conflict"].sum() for x in gate))}
    for key in ("margin", "w1", "entropy"):
        values, strata = combined(gate, lambda x, k=key: x["natural_effect"][k])
        summary[f"natural_{key}"] = interval(values, strata, rng)
    scenario = "evaluation_period_all_gla"
    for direction in ("e2n", "n2e"):
        for metric in ("margin", "w1", "entropy"):
            values, strata = combined(
                gate, lambda x, d=direction, m=metric: x["effects"][scenario][f"{d}_{m}"]
            )
            summary[f"{direction}_{metric}"] = interval(values, strata, rng)
    e2n_w1_values, e2n_w1_strata = combined(
        gate, lambda x: x["effects"][scenario]["e2n_w1"]
    )
    natural_w1_values, natural_w1_strata = combined(
        gate, lambda x: x["natural_effect"]["w1"]
    )
    if not np.array_equal(e2n_w1_strata, natural_w1_strata):
        raise RuntimeError("W1 ratio strata changed")
    e2n_margin_values, e2n_margin_strata = combined(
        gate, lambda x: x["effects"][scenario]["e2n_margin"]
    )
    natural_margin_values, natural_margin_strata = combined(
        gate, lambda x: x["natural_effect"]["margin"]
    )
    if not np.array_equal(e2n_margin_strata, natural_margin_strata):
        raise RuntimeError("Margin ratio strata changed")
    summary["e2n_fraction_natural_w1"] = ratio_interval(
        e2n_w1_values, natural_w1_values, e2n_w1_strata, rng
    )
    summary["e2n_fraction_natural_margin"] = ratio_interval(
        e2n_margin_values, natural_margin_values, e2n_margin_strata, rng
    )
    band_scenario = args.band_scenario
    selected_blocks = [int(x) for x in args.blocks.split(",") if x.strip()]
    band_values, band_strata = combined(
        bands,
        lambda x: 0.5
        * (
            x["effects"][band_scenario]["e2n_margin"]
            + x["effects"][band_scenario]["n2e_margin"]
        ),
    )
    bidirectional_gate_values, bidirectional_gate_strata = combined(
        gate,
        lambda x: 0.5
        * (
            x["effects"][scenario]["e2n_margin"]
            + x["effects"][scenario]["n2e_margin"]
        ),
    )
    if not np.array_equal(band_strata, bidirectional_gate_strata):
        raise RuntimeError("Band ratio strata changed")
    summary["band_fraction_bidirectional_margin"] = ratio_interval(
        band_values, bidirectional_gate_values, band_strata, rng
    )
    band_e2n_values, band_e2n_strata = combined(
        bands, lambda x: x["effects"][band_scenario]["e2n_margin"]
    )
    if not np.array_equal(band_e2n_strata, natural_margin_strata):
        raise RuntimeError("Band-to-natural ratio strata changed")
    summary["band_fraction_natural_margin"] = ratio_interval(
        band_e2n_values, natural_margin_values, band_e2n_strata, rng
    )
    discovery_conflict = gate[0]["conflict"]
    summary["discovery_reverse_margin"] = interval(
        gate[0]["effects"][scenario]["n2e_margin"][discovery_conflict],
        gate[0]["w1"][discovery_conflict],
        rng,
    )

    block_summary = {}
    for block in selected_blocks:
        individual = f"block_{block}"
        loo = f"all_gla_except_block_{block}"
        cell = {}
        for direction in ("e2n", "n2e"):
            individual_values, strata = combined(
                blocks, lambda x, d=direction, s=individual: x["effects"][s][f"{d}_margin"]
            )
            deletion_parts, deletion_labels = [], []
            for gate_split, block_split in zip(gate, blocks):
                mask = block_split["conflict"]
                deletion_parts.append(
                    (
                        gate_split["effects"][scenario][f"{direction}_margin"]
                        - block_split["effects"][loo][f"{direction}_margin"]
                    )[mask]
                )
                deletion_labels.extend(
                    f"{block_split['split']}:{letter}"
                    for letter in block_split["w1"][mask]
                )
            deletion_values = np.concatenate(deletion_parts)
            deletion_strata = np.asarray(deletion_labels)
            cell[f"{direction}_individual_margin"] = interval(individual_values, strata, rng)
            cell[f"{direction}_deletion_delta"] = interval(
                deletion_values, deletion_strata, rng
            )
        individual_average, strata = combined(
            blocks,
            lambda x, s=individual: 0.5
            * (x["effects"][s]["e2n_margin"] + x["effects"][s]["n2e_margin"]),
        )
        deletion_parts, deletion_labels = [], []
        for gate_split, block_split in zip(gate, blocks):
            mask = block_split["conflict"]
            deletion_parts.append(
                (
                    0.5
                    * (
                        gate_split["effects"][scenario]["e2n_margin"]
                        + gate_split["effects"][scenario]["n2e_margin"]
                        - block_split["effects"][loo]["e2n_margin"]
                        - block_split["effects"][loo]["n2e_margin"]
                    )
                )[mask]
            )
            deletion_labels.extend(
                f"{block_split['split']}:{letter}"
                for letter in block_split["w1"][mask]
            )
        deletion_average = np.concatenate(deletion_parts)
        deletion_strata = np.asarray(deletion_labels)
        cell["bidirectional_individual_margin"] = interval(individual_average, strata, rng)
        cell["bidirectional_deletion_delta"] = interval(
            deletion_average, deletion_strata, rng
        )
        block_summary[str(block)] = cell
    summary["blocks"] = block_summary
    summary["band_scenario"] = band_scenario
    summary["selected_blocks"] = selected_blocks

    split_summary = {}
    for gate_split, band_split in zip(gate, bands):
        mask = gate_split["conflict"]
        strata = gate_split["w1"][mask]
        split_summary[gate_split["split"]] = {
            "all_gla_bidirectional_margin": interval(
                0.5 * (
                    gate_split["effects"][scenario]["e2n_margin"]
                    + gate_split["effects"][scenario]["n2e_margin"]
                )[mask], strata, rng
            ),
            "band_bidirectional_margin": interval(
                0.5 * (
                    band_split["effects"][band_scenario]["e2n_margin"]
                    + band_split["effects"][band_scenario]["n2e_margin"]
                )[mask], strata, rng
            ),
        }
    summary["splits"] = split_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Output-preserved evaluation-period GLA update: consolidated result",
        "",
        "This is the canonical corrected analysis. The intervention copies only the evaluation-period GLA recurrent-memory update; it restores the source token's local output to the target-natural value, preventing donor information from also escaping through the period token's residual, ordinary-attention K/V, MLP, or short convolutional path.",
        "",
        "Primary analysis uses all 273 W1 != W2 questions across the frozen discovery and confirmation splits, with the two frozen splits also reported separately.",
        "",
        "## Portable state",
        "",
        f"The natural Evaluation-minus-Matched-Neutral W1-minus-W2 margin difference is {fmt(summary['natural_margin'])}. Copying the Evaluation period's recurrent update into Neutral transfers {fmt(summary['e2n_margin'])}; the reverse Neutral-into-Evaluation transplant transfers {fmt(summary['n2e_margin'])} in the opposite causal direction.",
        f"The corresponding displayed-answer W1-selection transfers are {fmt(summary['e2n_w1'], 100, 1)} and {fmt(summary['n2e_w1'], 100, 1)} percentage points. The Evaluation-to-Neutral estimate is {fmt(summary['e2n_fraction_natural_w1'], 100, 1)}% of the natural W1-selection difference, but its interval is necessarily wide because answer selection is discrete.",
        f"The Evaluation update also transfers {fmt(summary['e2n_entropy'])} bits of A-D entropy; this route is therefore not a pure scalar W1-suppression channel.",
        "",
        "The split asymmetry is material: the all-GLA bidirectional margin transfer is "
        f"{fmt(split_summary['discovery']['all_gla_bidirectional_margin'])} in discovery and {fmt(split_summary['confirmation']['all_gla_bidirectional_margin'])} in confirmation. Both frozen splits pass the prespecified joint gate, but only confirmation has both directional intervals above zero. The route is replicated, while its magnitude is heterogeneous.",
        "",
        "## Localization",
        "",
        f"Only `{band_scenario}` passed the frozen discovery screen and replicated on confirmation. It contains the tested GLA blocks {', '.join(map(str, selected_blocks))}. Its bidirectional margin transfer is {fmt(split_summary['discovery']['band_bidirectional_margin'])} in discovery and {fmt(split_summary['confirmation']['band_bidirectional_margin'])} in confirmation. Pooled, it is {fmt(summary['band_fraction_bidirectional_margin'], 100, 1)}% of the all-GLA bidirectional transfer. The band is thus a substantial sufficient carrier, but not the whole recurrent route.",
        "",
        "| GLA block | Alone: bidirectional margin transfer | Deleting it from all-GLA: loss of transfer |",
        "|---:|---:|---:|",
    ]
    for block, cell in block_summary.items():
        lines.append(
            f"| {block} | {fmt(cell['bidirectional_individual_margin'])} | "
            f"{fmt(cell['bidirectional_deletion_delta'])} |"
        )
    lines += [
        "",
        "Blocks 26 and 27 each show small positive block-alone effects after pooling, but neither is independently positive on both frozen splits. More importantly, every leave-one-block-out intervention retains nearly the complete all-GLA transfer. No individual block is a necessary bottleneck. The portable update is distributed and redundant at single-block resolution, with its clearest jointly sufficient carrier in blocks 25–32.",
        "",
        "## What the correction changes",
        "",
        "The historical transplant changed both the recurrent memory update and the period token's own downstream-visible residual, producing much larger headline effects. The corrected output-preserved result establishes a real persistent GLA-memory route, but not that this route alone explains most of the behavioral task difference. The remaining corrected source-trace and relay controls test where the evaluation information travels outside this route.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")

    if args.figure is not None:
        labels = ["Discovery\nall GLA", "Confirmation\nall GLA", "Discovery\n25–32", "Confirmation\n25–32"]
        cells = [
            split_summary["discovery"]["all_gla_bidirectional_margin"],
            split_summary["confirmation"]["all_gla_bidirectional_margin"],
            split_summary["discovery"]["band_bidirectional_margin"],
            split_summary["confirmation"]["band_bidirectional_margin"],
        ]
        means = np.asarray([x["mean"] for x in cells])
        yerr = np.asarray([[x["mean"] - x["ci_low"] for x in cells], [x["ci_high"] - x["mean"] for x in cells]])
        fig, ax = plt.subplots(figsize=(9, 5.4))
        colors = ["#355C7D", "#355C7D", "#C06C84", "#C06C84"]
        ax.bar(np.arange(4), means, color=colors, alpha=0.9)
        ax.errorbar(np.arange(4), means, yerr=yerr, fmt="none", ecolor="black", capsize=5, lw=1.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(4), labels)
        ax.set_ylabel("Bidirectional W1−W2 margin transfer (logits)")
        ax.set_title("Output-preserved policy update is causal and distributed")
        ax.text(0.01, -0.18, "Bars: mean; whiskers: stratified paired-bootstrap 95% CI. Source-token residual remains target-natural.", transform=ax.transAxes, fontsize=9)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-gate", type=Path, required=True)
    parser.add_argument("--confirmation-gate", type=Path, required=True)
    parser.add_argument("--discovery-blocks", type=Path, required=True)
    parser.add_argument("--confirmation-blocks", type=Path, required=True)
    parser.add_argument("--discovery-bands", type=Path, required=True)
    parser.add_argument("--confirmation-bands", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band-scenario", default="blocks_25_32")
    parser.add_argument("--blocks", default="25,26,27,29,30,31")
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--seed", type=int, default=20260814)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
