from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import LETTERS
from .run_feedback_policy_layer_decomposition import BANDS_ONE_BASED, _scenario_specs


TASKS = ("Game", "Neutral")


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def _bootstrap_mean(
    values: np.ndarray, indices: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, list[float]]:
    selected = np.asarray(values[indices], dtype=np.float64)
    point = float(selected.mean())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(draws, start + 200)
        rows = rng.integers(0, len(selected), size=(stop - start, len(selected)))
        samples[start:stop] = selected[rows].mean(axis=1)
    return point, [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def _bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, list[float]]:
    num = np.asarray(numerator[indices], dtype=np.float64)
    den = np.asarray(denominator[indices], dtype=np.float64)
    point = float(num.sum() / den.sum())
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        stop = min(draws, start + 200)
        rows = rng.integers(0, len(num), size=(stop - start, len(num)))
        samples[start:stop] = num[rows].sum(axis=1) / den[rows].sum(axis=1)
    return point, [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def _load_rank_order(
    qids: list[str], baseline_path: Path, mapping_path: Path
) -> np.ndarray:
    baseline = json.loads(baseline_path.read_text())["results"]
    mappings = {
        str(row["question_id"]): row
        for row in json.loads(mapping_path.read_text())["rows"]
    }
    order = np.empty((len(qids), 4), dtype=np.int64)
    for qi, qid in enumerate(qids):
        old = np.asarray(baseline[qid]["aggregated_ad_logits"], dtype=np.float64)
        ranks = np.argsort(-old, kind="stable")
        order[qi] = [
            LETTERS.index(mappings[qid]["original_to_new"][LETTERS[int(index)]])
            for index in ranks
        ]
    return order


def analyze(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500 or not arrays["completed"].astype(bool).all():
        raise RuntimeError("A complete 500-question checkpoint is required")
    scenarios = arrays["scenario_ids"].astype(str).tolist()
    expected = [row[0] for row in _scenario_specs(list(range(3, 64, 4)), [
        layer for layer in range(64) if layer % 4 != 3
    ])]
    if scenarios != expected:
        raise RuntimeError("Scenario inventory changed")
    scenario_index = {name: index for index, name in enumerate(scenarios)}

    logits = arrays["scenario_final_logits"].astype(np.float64)
    raw = arrays["scenario_final_logits_raw"].astype(np.float64)
    if logits.shape != (2, len(scenarios), 500, 4) or not np.isfinite(logits).all():
        raise RuntimeError("Causal output is incomplete or non-finite")
    natural_index = scenario_index["natural"]
    corrected_error = float(np.max(np.abs(
        logits[:, natural_index] - arrays["trusted_natural_logits"].astype(np.float64)
    )))
    same_batch_error = float(np.max(np.abs(
        raw[:, natural_index] - arrays["same_batch_natural_logits"].astype(np.float64)
    )))

    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.asarray([qid in discovery_ids for qid in qids])
    if [int(discovery.sum()), int((~discovery).sum())] != [251, 249]:
        raise RuntimeError("Frozen split changed")
    split_masks = {"discovery": discovery, "confirmation": ~discovery}

    centered = _center(logits)
    natural = centered[:, natural_index]
    donor_vector = natural[::-1] - natural
    denominator = np.sum(donor_vector * donor_vector, axis=-1)
    delta = centered - natural[:, None]
    transfer_numerator = np.sum(delta * donor_vector[:, None], axis=-1)

    rank_order = _load_rank_order(qids, args.baseline, args.remapping_plan)
    w1 = rank_order[:, 0]
    choices = np.argmax(logits, axis=-1)
    switch = choices != w1[None, None]
    choose_w1 = choices == w1[None, None]
    ranked = np.empty_like(centered)
    for qi in range(len(qids)):
        ranked[:, :, qi] = np.take(centered[:, :, qi], rank_order[qi], axis=-1)
    bivalent = ranked[..., 3] - ranked[..., :2].mean(axis=-1)

    summary: dict[str, Any] = {
        "question": "At which layers do feedback-suffix writes causally transmit task policy?",
        "coverage": {
            "questions": 500,
            "discovery": 251,
            "confirmation": 249,
            "tasks": list(TASKS),
            "layers": "all L1-L64",
            "bands": [f"L{start}-{stop}" for start, stop in BANDS_ONE_BASED],
            "carrier_families": ["ordinary attention", "GLA"],
        },
        "validation": {
            "all_complete": True,
            "all_finite": True,
            "corrected_natural_max_abs_error": corrected_error,
            "same_batch_natural_max_abs_error": same_batch_error,
        },
        "definitions": {
            "transfer_fraction": (
                "Projection of the intervention-induced centered A-D logit change "
                "onto that same question's paired natural donor-task minus recipient-task vector"
            ),
            "band_sufficiency": "Transfer when only this eight-layer band receives donor feedback-suffix writes",
            "band_necessity_loss": "All-layer transfer minus transfer when this band alone remains natural",
            "bivalent_change": "Change in R4 minus mean(R1,R2), relative to the task's natural run",
        },
        "splits": {},
    }

    for split_number, (split_name, mask) in enumerate(split_masks.items()):
        indices = np.flatnonzero(mask)
        split_record: dict[str, Any] = {"questions": int(mask.sum()), "tasks": {}}
        for ti, task in enumerate(TASKS):
            task_rows: dict[str, Any] = {}
            for si, scenario in enumerate(scenarios):
                rng = np.random.default_rng(args.seed + split_number * 100000 + ti * 10000 + si)
                transfer, transfer_ci = _bootstrap_ratio(
                    transfer_numerator[ti, si], denominator[ti], indices, rng,
                    args.bootstrap_draws,
                )
                sw, sw_ci = _bootstrap_mean(switch[ti, si], indices, rng, args.bootstrap_draws)
                wr, wr_ci = _bootstrap_mean(choose_w1[ti, si], indices, rng, args.bootstrap_draws)
                bv, bv_ci = _bootstrap_mean(
                    bivalent[ti, si] - bivalent[ti, natural_index], indices, rng,
                    args.bootstrap_draws,
                )
                task_rows[scenario] = {
                    "transfer_fraction": transfer,
                    "transfer_fraction_ci": transfer_ci,
                    "switch_rate": sw,
                    "switch_rate_ci": sw_ci,
                    "choose_W1_rate": wr,
                    "choose_W1_rate_ci": wr_ci,
                    "bivalent_change": bv,
                    "bivalent_change_ci": bv_ci,
                }

            all_si = scenario_index["all_layers_swapped"]
            necessity: dict[str, Any] = {}
            for bi, (start, stop) in enumerate(BANDS_ONE_BASED):
                except_si = scenario_index[f"all_except_band_{start:02d}_{stop:02d}"]
                num = transfer_numerator[ti, all_si] - transfer_numerator[ti, except_si]
                rng = np.random.default_rng(
                    args.seed + 500000 + split_number * 100000 + ti * 10000 + bi
                )
                point, ci = _bootstrap_ratio(
                    num, denominator[ti], indices, rng, args.bootstrap_draws
                )
                necessity[f"band_{start:02d}_{stop:02d}"] = {
                    "transfer_loss": point, "transfer_loss_ci": ci
                }
            split_record["tasks"][task] = {
                "scenarios": task_rows,
                "band_necessity": necessity,
            }
        summary["splits"][split_name] = split_record

    selected: list[str] = []
    disc = summary["splits"]["discovery"]["tasks"]
    for start, stop in BANDS_ONE_BASED:
        key = f"band_{start:02d}_{stop:02d}"
        sufficient = any(
            disc[task]["scenarios"][f"{key}_only"]["transfer_fraction_ci"][0] > 0
            for task in TASKS
        )
        necessary = any(
            disc[task]["band_necessity"][key]["transfer_loss_ci"][0] > 0
            for task in TASKS
        )
        if sufficient or necessary:
            selected.append(key)
    summary["discovery_selected_bands"] = selected

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    labels = [f"{start}-{stop}" for start, stop in BANDS_ONE_BASED]
    x = np.arange(len(labels))
    colors = {"Game": "#d95f02", "Neutral": "#1b7fb8"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ti, task in enumerate(TASKS):
        rows = summary["splits"]["confirmation"]["tasks"][task]
        suff = np.asarray([
            rows["scenarios"][f"band_{start:02d}_{stop:02d}_only"]["transfer_fraction"]
            for start, stop in BANDS_ONE_BASED
        ])
        suff_ci = np.asarray([
            rows["scenarios"][f"band_{start:02d}_{stop:02d}_only"]["transfer_fraction_ci"]
            for start, stop in BANDS_ONE_BASED
        ])
        need = np.asarray([
            rows["band_necessity"][f"band_{start:02d}_{stop:02d}"]["transfer_loss"]
            for start, stop in BANDS_ONE_BASED
        ])
        need_ci = np.asarray([
            rows["band_necessity"][f"band_{start:02d}_{stop:02d}"]["transfer_loss_ci"]
            for start, stop in BANDS_ONE_BASED
        ])
        axis = axes[0, ti]
        axis.errorbar(x - .09, suff, yerr=np.stack((suff-suff_ci[:, 0], suff_ci[:, 1]-suff)),
                      marker="o", capsize=3, label="band alone")
        axis.errorbar(x + .09, need, yerr=np.stack((need-need_ci[:, 0], need_ci[:, 1]-need)),
                      marker="s", capsize=3, label="loss without band")
        axis.axhline(0, color="black", lw=.8)
        axis.set(title=f"{'AB'[ti]}  {task}: eight-layer bands", ylabel="Donor-task transfer fraction")
        axis.set_xticks(x, labels, rotation=35)
        axis.legend()

    for ti, task in enumerate(TASKS):
        rows = summary["splits"]["confirmation"]["tasks"][task]["scenarios"]
        names = [f"prefix_through_{stop:02d}" for _start, stop in BANDS_ONE_BASED[:-1]] + ["all_layers_swapped"]
        values = np.asarray([rows[name]["transfer_fraction"] for name in names])
        cis = np.asarray([rows[name]["transfer_fraction_ci"] for name in names])
        axes[1, 0].errorbar(
            np.arange(8), values,
            yerr=np.stack((values-cis[:, 0], cis[:, 1]-values)), marker="o",
            capsize=3, color=colors[task], label=task,
        )
    axes[1, 0].axhline(0, color="black", lw=.8)
    axes[1, 0].set_xticks(np.arange(8), [str(stop) for _start, stop in BANDS_ONE_BASED])
    axes[1, 0].set(title="C  Cumulative donor prefix (confirmation)", xlabel="Donor writes through layer", ylabel="Transfer fraction")
    axes[1, 0].legend()

    carrier_names = ["ordinary_all_swapped", "gla_all_swapped", "all_layers_swapped"]
    carrier_labels = ["ordinary", "GLA", "both"]
    for ti, task in enumerate(TASKS):
        rows = summary["splits"]["confirmation"]["tasks"][task]["scenarios"]
        values = np.asarray([rows[name]["transfer_fraction"] for name in carrier_names])
        cis = np.asarray([rows[name]["transfer_fraction_ci"] for name in carrier_names])
        axes[1, 1].bar(
            np.arange(3) + (ti-.5)*.36, values, .36,
            yerr=np.stack((values-cis[:, 0], cis[:, 1]-values)), capsize=3,
            color=colors[task], label=task,
        )
    axes[1, 1].axhline(0, color="black", lw=.8)
    axes[1, 1].set_xticks(np.arange(3), carrier_labels)
    axes[1, 1].set(title="D  Carrier-family crossover (confirmation)", ylabel="Transfer fraction")
    axes[1, 1].legend()
    fig.suptitle("Where the feedback policy is transmitted downstream")
    fig.savefig(args.figure, dpi=190)
    plt.close(fig)

    lines = [
        "# Feedback-policy layer decomposition", "",
        "This analysis crosses the downstream writes of the complete `incorrect/lost . Choose the answer again .` suffix between paired Game and Neutral prompts. The suffix tokens' own residual states remain natural. All 64 layers are covered. Game and Neutral are analyzed separately.", "",
        f"Discovery-selected bands: {', '.join(selected) if selected else 'none'}.", "",
        "The complete estimates, confidence intervals, switch rates, W1 rates, and bivalent rank changes are in `summary.json`.", "",
        f"Canonical figure: [{args.figure.name}]({args.figure.resolve()})",
    ]
    args.report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=8232026)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
