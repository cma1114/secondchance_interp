from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LETTERS = "ABCD"
CONDITIONS = ("Game", "Neutral")
COLORS = {"W1": "#2589f5", "W2": "#ef7d32", "Other two": "#53bd73"}


def _aligned(values: np.ndarray, qids: list[str], plan: dict,
             question_axis: int) -> np.ndarray:
    result = np.empty_like(values)
    for qi, qid in enumerate(qids):
        semantic_to_displayed = [
            LETTERS.index(plan[qid]["original_to_new"][original_letter])
            for original_letter in LETTERS
        ]
        index = [slice(None)] * values.ndim
        index[question_axis] = qi
        result[tuple(index)] = values[tuple(index)][..., semantic_to_displayed]
    return result


def _interval(values: np.ndarray, strata: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    boot = np.empty((5000,) + values.shape[1:], dtype=float)
    for draw in range(len(boot)):
        picked = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        boot[draw] = values[picked].mean(axis=0)
    return values.mean(axis=0), *np.quantile(boot, (0.025, 0.975), axis=0)


def _pick(values: np.ndarray, identities: np.ndarray) -> np.ndarray:
    return np.stack([values[row, ..., identity] for row, identity in enumerate(identities)])


def analyze(results_path: Path, original_path: Path, remapped_path: Path,
            trusted_game_path: Path, trusted_neutral_path: Path, plan_path: Path,
            output: Path, figure_swap: Path, figure_sublayer: Path) -> None:
    arrays = np.load(results_path)
    if not bool(arrays["completed"].all()):
        raise RuntimeError("Run is incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    original = json.loads(original_path.read_text())["results"]
    remapped = json.loads(remapped_path.read_text())["results"]
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]
    plan = {row["question_id"]: row for row in json.loads(plan_path.read_text())["rows"]}
    trusted_logits = np.asarray([
        [trusted[ci][qid]["aggregated_ad_logits"] for qid in qids]
        for ci in range(2)
    ])
    max_error = float(np.max(np.abs(arrays["natural_logits"] - trusted_logits)))
    if max_error != 0.0:
        raise RuntimeError(f"Natural exact-match validation failed: {max_error}")

    w1 = np.asarray([LETTERS.index(original[qid]["answer"]) for qid in qids])
    w2 = np.asarray([LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids])
    conflict = w1 != w2
    selected = np.flatnonzero(conflict)
    strata = w1[selected]
    w1s, w2s = w1[selected], w2[selected]
    source_readouts = np.arange(48, 57)

    natural = _aligned(arrays["natural_logits"].astype(float), qids, plan, 1)
    patched = _aligned(arrays["patched_logits"].astype(float), qids, plan, 2)
    natural -= natural.mean(axis=-1, keepdims=True)
    patched -= patched.mean(axis=-1, keepdims=True)
    natural_q = natural[:, selected]
    patched_q = patched[:, :, selected]

    effects = np.empty((2, len(selected), len(source_readouts), 3), dtype=float)
    choice_effects = np.empty((2, len(selected), len(source_readouts)), dtype=float)
    for ci in range(2):
        for ri in range(len(source_readouts)):
            delta = patched_q[ci, ri] - natural_q[ci]
            effects[ci, :, ri, 0] = _pick(delta, w1s)
            effects[ci, :, ri, 1] = _pick(delta, w2s)
            for row in range(len(selected)):
                others = [x for x in range(4) if x not in (w1s[row], w2s[row])]
                effects[ci, row, ri, 2] = delta[row, others].mean()
            natural_choice = natural_q[ci].argmax(axis=-1)
            patched_choice = patched_q[ci, ri].argmax(axis=-1)
            choice_effects[ci, :, ri] = (
                (patched_choice == w1s).astype(float)
                - (natural_choice == w1s).astype(float)
            ) * 100

    output.mkdir(parents=True, exist_ok=True)
    figure_swap.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "n_total": len(qids), "n_conflict": int(conflict.sum()),
        "natural_exact_match_max_abs_error": max_error,
        "source_readouts": source_readouts.tolist(), "directions": {},
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ci, direction in enumerate(("Neutral `lost` into Game `incorrect`", "Game `incorrect` into Neutral `lost`")):
        summary["directions"][direction] = {}
        ax = axes[0, ci]
        for metric, name in enumerate(("W1", "W2", "Other two")):
            mean, low, high = _interval(effects[ci, :, :, metric], strata, 20260814 + ci * 10 + metric)
            ax.plot(source_readouts, mean, marker="o", lw=2, color=COLORS[name], label=name)
            ax.fill_between(source_readouts, low, high, color=COLORS[name], alpha=.18)
            summary["directions"][direction][name] = {
                str(layer): {"mean": float(m), "ci": [float(lo), float(hi)]}
                for layer, m, lo, hi in zip(source_readouts, mean, low, high)
            }
        ax.axhline(0, color="#777", ls="--", lw=1)
        ax.set_title(f"{chr(65 + ci)}  {direction}")
        ax.set_ylabel("Change in centered final evidence")
        ax.legend(frameon=False)

        ax = axes[1, ci]
        mean, low, high = _interval(choice_effects[ci], strata, 20260914 + ci)
        ax.errorbar(source_readouts, mean, yerr=np.stack((mean-low, high-mean)),
                    fmt="o", capsize=3, color=COLORS["W1"])
        ax.axhline(0, color="#777", ls="--", lw=1)
        ax.set_title(f"{chr(67 + ci)}  Effect on choosing W1")
        ax.set_ylabel("Change in W1 selection (percentage points)")
        summary["directions"][direction]["W1_selection_pp"] = {
            str(layer): {"mean": float(m), "ci": [float(lo), float(hi)]}
            for layer, m, lo, hi in zip(source_readouts, mean, low, high)
        }
    for ax in axes[1]:
        ax.set_xlabel("Source residual readout swapped")
    for ax in axes.ravel():
        ax.set_xticks(source_readouts)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(
        "Does the late incorrect/lost token state control semantic W1 suppression?\n"
        f"SimpleMC remapped conflict trials (n={len(selected)}); paired 95% bootstrap CIs",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(figure_swap, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Natural sublayer trajectory, aligned to semantic content and centered.
    sub = _aligned(arrays["sublayer_scores"].astype(float), qids, plan, 1)
    sub -= sub.mean(axis=-1, keepdims=True)
    sub = sub[:, selected]
    w1_values = np.stack([_pick(sub[ci], w1s) for ci in range(2)])
    # batch, block, boundary -> flatten the actual sequential boundaries.
    x = np.arange(15)
    labels = [f"pre {block}" if boundary == 0 else f"mix {block}" if boundary == 1 else f"MLP {block}"
              for block in range(52, 57) for boundary in range(3)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    for ci, condition in enumerate(CONDITIONS):
        values = w1_values[ci].reshape(len(selected), -1)
        mean, low, high = _interval(values, strata, 20261014 + ci)
        axes[0].plot(x, mean, marker="o", lw=2, label=condition)
        axes[0].fill_between(x, low, high, alpha=.18)
    axes[0].axhline(0, color="#777", ls="--", lw=1)
    axes[0].set_title("A  W1 evidence within blocks 52–56")
    axes[0].set_ylabel("Centered ordinary-logit-lens evidence")
    axes[0].legend(frameon=False)
    diff = (w1_values[0] - w1_values[1]).reshape(len(selected), -1)
    mean, low, high = _interval(diff, strata, 20261114)
    axes[1].plot(x, mean, marker="o", lw=2, color=COLORS["W1"])
    axes[1].fill_between(x, low, high, color=COLORS["W1"], alpha=.18)
    axes[1].axhline(0, color="#777", ls="--", lw=1)
    axes[1].set_title("B  Game minus Neutral W1 evidence")
    axes[1].set_ylabel("Paired centered-evidence difference")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(
        "Which sublayer creates the late semantic W1 divergence?\n"
        f"SimpleMC remapped conflict trials (n={len(selected)}); paired 95% bootstrap CIs",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(figure_sublayer, dpi=180, bbox_inches="tight")
    plt.close(fig)

    sub_summary = {}
    mean, low, high = _interval(diff, strata, 20261114)
    for label, m, lo, hi in zip(labels, mean, low, high):
        sub_summary[label] = {"mean": float(m), "ci": [float(lo), float(hi)]}
    summary["sublayer_game_minus_neutral_w1"] = sub_summary

    # Report the same W1 trajectory separately on W1=W2 trials.  The primary
    # figure remains the preregistered conflict analysis, but this split is
    # necessary to determine whether the localized late computation generalizes
    # beyond trials where the two standalone presentations favor different
    # semantic answers.
    subgroup_summary = {}
    for subgroup_name, subgroup_mask in (
        ("conflict_w1_ne_w2", conflict),
        ("no_conflict_w1_eq_w2", ~conflict),
    ):
        subgroup_idx = np.flatnonzero(subgroup_mask)
        subgroup_w1 = w1[subgroup_idx]
        subgroup_strata = subgroup_w1
        subgroup_sub = _aligned(arrays["sublayer_scores"].astype(float), qids, plan, 1)
        subgroup_sub -= subgroup_sub.mean(axis=-1, keepdims=True)
        subgroup_sub = subgroup_sub[:, subgroup_idx]
        subgroup_w1_values = np.stack([
            _pick(subgroup_sub[ci], subgroup_w1) for ci in range(2)
        ])
        subgroup_diff = (subgroup_w1_values[0] - subgroup_w1_values[1]).reshape(
            len(subgroup_idx), -1
        )
        mean, low, high = _interval(
            subgroup_diff, subgroup_strata,
            20261214 if subgroup_name.startswith("conflict") else 20261314,
        )
        trajectory = {
            label: {"mean": float(m), "ci": [float(lo), float(hi)]}
            for label, m, lo, hi in zip(labels, mean, low, high)
        }
        increments = {}
        for block_i, block in enumerate(range(52, 57)):
            base = block_i * 3
            increments[f"Mixer {block}"] = float(mean[base + 1] - mean[base])
            increments[f"MLP {block}"] = float(mean[base + 2] - mean[base + 1])

        # W1 effect and W1-selection effect of the bidirectional token-state
        # swaps on this subgroup.  Alignment is semantic, as above.
        subgroup_natural = natural[:, subgroup_idx]
        subgroup_patched = patched[:, :, subgroup_idx]
        swap_summary = {}
        for ci, direction in enumerate((
            "Neutral `lost` into Game `incorrect`",
            "Game `incorrect` into Neutral `lost`",
        )):
            delta = subgroup_patched[ci] - subgroup_natural[ci][None, ...]
            # source-readout, question, answer -> question, source-readout
            w1_delta = np.stack([
                delta[:, row, identity] for row, identity in enumerate(subgroup_w1)
            ])
            natural_choice = subgroup_natural[ci].argmax(axis=-1)
            patched_choice = subgroup_patched[ci].argmax(axis=-1)
            choice_delta = (
                (patched_choice == subgroup_w1[None, :]).astype(float)
                - (natural_choice[None, :] == subgroup_w1[None, :]).astype(float)
            ).T * 100
            w1_mean, w1_low, w1_high = _interval(
                w1_delta, subgroup_strata, 20261414 + ci + len(subgroup_idx)
            )
            ch_mean, ch_low, ch_high = _interval(
                choice_delta, subgroup_strata, 20261514 + ci + len(subgroup_idx)
            )
            swap_summary[direction] = {
                "W1": {
                    str(layer): {"mean": float(m), "ci": [float(lo), float(hi)]}
                    for layer, m, lo, hi in zip(source_readouts, w1_mean, w1_low, w1_high)
                },
                "W1_selection_pp": {
                    str(layer): {"mean": float(m), "ci": [float(lo), float(hi)]}
                    for layer, m, lo, hi in zip(source_readouts, ch_mean, ch_low, ch_high)
                },
            }
        subgroup_summary[subgroup_name] = {
            "n": int(len(subgroup_idx)),
            "sublayer_game_minus_neutral_w1": trajectory,
            "incremental_changes": increments,
            "feedback_token_swaps": swap_summary,
        }
    summary["conflict_split"] = subgroup_summary
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""# Remapped W1 feedback-state swap and sublayer localization

## Design

- All 500 frozen SimpleMC questions; primary semantic analysis on the {len(selected)} W1 != W2 conflict trials.
- Exact historical batch-of-four SDPA cohorts.
- At readouts 48--56, replace the complete post-block state of Game's `incorrect` token with Neutral's paired `lost` state, or vice versa, and let that one replacement propagate through the rest of the model.
- In the unmodified passes, decode the final-decision residual before the mixer, after the mixer, and after the MLP in blocks 52--56.
- Natural A--D logits reproduce the trusted remapped run exactly (maximum absolute error {max_error}).

## Artifacts

- `{figure_swap}`
- `{figure_sublayer}`
- `summary.json`
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--original-baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-swap", type=Path, required=True)
    parser.add_argument("--figure-sublayer", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.original_baseline, args.remapped_baseline,
            args.trusted_game, args.trusted_neutral, args.plan, args.output,
            args.figure_swap, args.figure_sublayer)


if __name__ == "__main__":
    main()
