from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run_remapped_source_contributions_and_swaps import (
    CONDITIONS,
    GROUP_NAMES,
    LETTERS,
    MIXER_LAYERS,
    SOURCE_READOUTS,
    SWAP_NAMES,
)


COLORS = {"W1": "#2589f5", "W2": "#ef7d32", "Other": "#53bd73"}
FRIENDLY_GROUPS = {
    "other_structure": "Chat structure",
    "system": "System prompt",
    "first_instruction": "First answer-only instruction",
    "first_question_stem": "First question stem",
    "first_option_w1": "First-presentation W1 option",
    "first_option_w2": "First-presentation W2 option",
    "first_option_w1w2": "First option (W1 = W2)",
    "first_option_other": "Other first-presentation options",
    "first_choice_cue": "First choice cue",
    "historical_assistant": "Historical assistant boundary",
    "second_instruction": "Second answer-only instruction",
    "second_question_stem": "Repeated question/option-boundary states",
    "second_option_w1": "Repeated-presentation W1 option",
    "second_option_w2": "Repeated-presentation W2 option",
    "second_option_w1w2": "Repeated option (W1 = W2)",
    "second_option_other": "Other repeated options",
    "second_choice_cue": "Second choice cue",
    "final_assistant_prefix": "Final assistant prefix",
}


def _friendly(name: str, feedback_labels: dict[int, str]) -> str:
    if name.startswith("feedback_slot_"):
        slot = int(name.rsplit("_", 1)[1])
        return feedback_labels.get(slot, f"Feedback token slot {slot}")
    return FRIENDLY_GROUPS.get(name, name.replace("_", " ").title())


def _interval(values: np.ndarray, strata: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    boot = np.empty((5000,) + values.shape[1:], dtype=float)
    for draw in range(len(boot)):
        picked = np.concatenate([
            rng.choice(group, len(group), replace=True) for group in groups
        ])
        boot[draw] = values[picked].mean(axis=0)
    return values.mean(axis=0), *np.quantile(boot, (0.025, 0.975), axis=0)


def _semantic_order(qid: str, plan: dict[str, Any]) -> list[int]:
    return [
        LETTERS.index(plan[qid]["original_to_new"][original])
        for original in LETTERS
    ]


def _feedback_labels(metadata: dict[str, Any]) -> dict[int, str]:
    # Both prompts are token matched.  These slot labels show the actual paired
    # textual roles and make the heatmap readable without requiring a tokenizer.
    game = ["Your", "answer", "was", "incorrect", ".", "Choose", "a", "different", "answer", "."]
    neutral = ["Your", "answer", "was", "lost", ".", "Choose", "the", "answer", "again", "."]
    labels = {}
    for slot, (left, right) in enumerate(zip(game, neutral)):
        labels[slot] = f"Feedback {left}" if left == right else f"Feedback {left} / {right}"
    return labels


def analyze(
    run_dir: Path,
    original_baseline_path: Path,
    remapped_baseline_path: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    plan_path: Path,
    output: Path,
    figure_sources: Path,
    figure_swaps: Path,
) -> None:
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    cohort_paths = sorted((run_dir / "cohorts").glob("cohort_*.npz"))
    if len(cohort_paths) != 125:
        raise RuntimeError(f"Expected 125 cohorts, found {len(cohort_paths)}")
    original = json.loads(original_baseline_path.read_text())["results"]
    remapped = json.loads(remapped_baseline_path.read_text())["results"]
    plan = {row["question_id"]: row for row in json.loads(plan_path.read_text())["rows"]}
    trusted = [
        json.loads(trusted_game_path.read_text())["results"],
        json.loads(trusted_neutral_path.read_text())["results"],
    ]

    qids: list[str] = []
    natural_parts = []
    patched_parts = []
    route_rows = []
    attention_rows = []
    context_errors = []
    # route rows: one array per question, [condition, mixer, group, head, answer]
    for path in cohort_paths:
        with np.load(path, allow_pickle=False) as shard:
            shard_qids = shard["question_ids"].astype(str).tolist()
            qids.extend(shard_qids)
            natural_parts.append(shard["natural_logits"].astype(float))
            patched_parts.append(shard["patched_logits"].astype(float))
            context_errors.append(shard["source_context_max_error"].astype(float))
            direct = shard["source_direct_ad"].astype(float)
            attention = shard["source_attention"].astype(float)
            token_groups = shard["token_groups"]
            # [condition, batch, mixer, token, head, answer]
            for row in range(len(shard_qids)):
                route = np.zeros(
                    (2, len(MIXER_LAYERS), len(GROUP_NAMES), direct.shape[-2], 4),
                    dtype=np.float32,
                )
                mass = np.zeros(
                    (2, len(MIXER_LAYERS), len(GROUP_NAMES), direct.shape[-2]),
                    dtype=np.float32,
                )
                for ci in range(2):
                    codes = token_groups[ci, row]
                    valid = codes >= 0
                    for group in np.unique(codes[valid]):
                        positions = np.flatnonzero(codes == group)
                        route[ci, :, group] = direct[ci, row][:, positions].sum(axis=1)
                        mass[ci, :, group] = attention[ci, row][:, positions].sum(axis=1)
                order = _semantic_order(shard_qids[row], plan)
                route_rows.append(route[..., order])
                attention_rows.append(mass)

    # Stored cohort order is condition, batch, ...; concatenate on batch.
    natural = np.concatenate(natural_parts, axis=1)  # condition, question, answer
    patched = np.concatenate(patched_parts, axis=3)  # swap, condition, readout, question, answer
    routes = np.stack(route_rows)  # question, condition, mixer, group, head, semantic answer
    attention_mass = np.stack(attention_rows)
    max_context_error = float(np.max(np.stack(context_errors)))
    if qids != list(original):
        # The result dictionaries preserve manifest order in the trusted run,
        # but do not depend on it; this catches accidental shard reordering.
        if set(qids) != set(original):
            raise RuntimeError("Run and Baseline question sets differ")

    trusted_logits = np.asarray([
        [trusted[ci][qid]["aggregated_ad_logits"] for qid in qids]
        for ci in range(2)
    ])
    natural_error = float(np.max(np.abs(natural - trusted_logits)))
    if natural_error != 0.0:
        raise RuntimeError(f"Natural logits do not match trusted run: {natural_error}")

    # Align natural and patched final logits to semantic answer content.
    natural_sem = np.empty_like(natural)
    patched_sem = np.empty_like(patched)
    displayed_to_semantic = np.empty((len(qids), 4), dtype=np.int8)
    for qi, qid in enumerate(qids):
        order = _semantic_order(qid, plan)
        for semantic_index, displayed_index in enumerate(order):
            displayed_to_semantic[qi, displayed_index] = semantic_index
        natural_sem[:, qi] = natural[:, qi][:, order]
        patched_sem[:, :, :, qi] = patched[:, :, :, qi][:, :, :, order]
    natural_sem -= natural_sem.mean(axis=-1, keepdims=True)
    patched_sem -= patched_sem.mean(axis=-1, keepdims=True)

    w1 = np.asarray([LETTERS.index(original[qid]["answer"]) for qid in qids])
    w2 = np.asarray([
        LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids
    ])
    conflict = w1 != w2
    selected = np.flatnonzero(conflict)
    strata = w1[selected]

    # Causal state swaps on conflict trials.
    swap_effects = np.empty(
        (len(SWAP_NAMES), 2, len(selected), len(SOURCE_READOUTS), 3), dtype=float
    )
    choice_effects = np.empty(
        (len(SWAP_NAMES), 2, len(selected), len(SOURCE_READOUTS)), dtype=float
    )
    for si in range(len(SWAP_NAMES)):
        for ci in range(2):
            for ri in range(len(SOURCE_READOUTS)):
                delta = patched_sem[si, ci, ri, selected] - natural_sem[ci, selected]
                for row, qi in enumerate(selected):
                    swap_effects[si, ci, row, ri, 0] = delta[row, w1[qi]]
                    swap_effects[si, ci, row, ri, 1] = delta[row, w2[qi]]
                    others = [x for x in range(4) if x not in (w1[qi], w2[qi])]
                    swap_effects[si, ci, row, ri, 2] = delta[row, others].mean()
                # Resolve exact ties in the model's displayed A--D order, then
                # translate the selected letter to semantic-content order.
                natural_displayed = natural[ci, selected].argmax(axis=-1)
                patched_displayed = patched[si, ci, ri, selected].argmax(axis=-1)
                natural_choice = displayed_to_semantic[
                    selected, natural_displayed
                ]
                patched_choice = displayed_to_semantic[
                    selected, patched_displayed
                ]
                choice_effects[si, ci, :, ri] = (
                    (patched_choice == w1[selected]).astype(float)
                    - (natural_choice == w1[selected]).astype(float)
                ) * 100

    output.mkdir(parents=True, exist_ok=True)
    figure_sources.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "n_total": len(qids),
        "n_conflict": int(conflict.sum()),
        "natural_exact_match_max_abs_error": natural_error,
        "source_context_reconstruction_max_abs_error": max_context_error,
        "state_swaps": {},
        "source_contributions": {},
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    for si, swap_name in enumerate(SWAP_NAMES):
        summary["state_swaps"][swap_name] = {}
        for ci, direction in enumerate(("Neutral into Game", "Game into Neutral")):
            ax = axes[si, ci]
            summary["state_swaps"][swap_name][direction] = {}
            for mi, metric in enumerate(("W1", "W2", "Other")):
                mean, low, high = _interval(
                    swap_effects[si, ci, :, :, mi], strata,
                    20260820 + si * 20 + ci * 5 + mi,
                )
                ax.plot(SOURCE_READOUTS, mean, marker="o", lw=2,
                        color=COLORS[metric], label=metric)
                ax.fill_between(SOURCE_READOUTS, low, high,
                                color=COLORS[metric], alpha=.20)
                summary["state_swaps"][swap_name][direction][metric] = {
                    str(layer): {"mean": float(m), "ci": [float(lo), float(hi)]}
                    for layer, m, lo, hi in zip(SOURCE_READOUTS, mean, low, high)
                }
            mean, low, high = _interval(
                choice_effects[si, ci], strata, 20260920 + si * 10 + ci
            )
            summary["state_swaps"][swap_name][direction]["W1_selection_pp"] = {
                str(layer): {"mean": float(m), "ci": [float(lo), float(hi)]}
                for layer, m, lo, hi in zip(SOURCE_READOUTS, mean, low, high)
            }
            ax.axhline(0, color="#777", ls="--", lw=1)
            ax.set_title(f"{swap_name.replace('_', ' ').title()}: {direction}")
            ax.set_ylabel("Change in centered final evidence")
            ax.grid(axis="y", alpha=.2)
            if si == 0:
                ax.legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Post-block residual readout swapped")
        ax.set_xticks(SOURCE_READOUTS)
    fig.suptitle(
        "Which feedback-token states causally distinguish Game from Neutral?\n"
        f"Remapped SimpleMC conflict trials (n={len(selected)}); paired 95% bootstrap CIs",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, .955))
    fig.savefig(figure_swaps, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Exact additive W1 contribution of every group/head. Center each route's
    # A-D projection before selecting W1.
    routes_centered = routes - routes.mean(axis=-1, keepdims=True)
    w1_routes = np.stack([
        routes_centered[qi, ..., w1[qi]] for qi in range(len(qids))
    ])
    conflict_routes = w1_routes[selected]
    conflict_mass = attention_mass[selected]
    feedback_labels = _feedback_labels(metadata)
    group_names = [_friendly(name, feedback_labels) for name in GROUP_NAMES]
    # Sum over heads: question, condition, mixer, group.
    group_writes = conflict_routes.sum(axis=-1)
    differences = group_writes[:, 0] - group_writes[:, 1]
    mean_diff = differences.mean(axis=0)  # mixer, group
    importance = np.max(np.abs(mean_diff), axis=0)
    candidates = [index for index in range(len(GROUP_NAMES)) if importance[index] > 1e-8]
    top_groups = sorted(candidates, key=lambda index: importance[index], reverse=True)[:16]
    # Display strongest at top.
    top_groups = top_groups[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    y = np.arange(len(top_groups))
    for li, layer in enumerate(MIXER_LAYERS):
        for ci, condition in enumerate(("Game", "Neutral")):
            values = group_writes[:, ci, li, top_groups]
            mean, low, high = _interval(values, strata, 20261020 + li * 10 + ci)
            offset = (-0.18 if ci == 0 else 0.18)
            axes[li].errorbar(
                mean, y + offset,
                xerr=np.stack((mean - low, high - mean)),
                fmt="o", capsize=3,
                color="#2589f5" if ci == 0 else "#ef7d32",
                label=condition,
            )
        axes[li].axvline(0, color="#777", ls="--", lw=1)
        axes[li].set_title(f"Mixer {layer + 1}: centered W1 residual write")
        axes[li].set_xlabel("Direct canonical-unembedding contribution")
        axes[li].grid(axis="x", alpha=.2)
        axes[li].legend(frameon=False)
    axes[0].set_yticks(y, [group_names[index] for index in top_groups])
    fig.suptitle(
        "Where do Mixers 52 and 56 obtain their W1-directed writes?\n"
        f"Exact additive token/head decomposition; conflict trials n={len(selected)}; 95% CIs",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(figure_sources, dpi=180, bbox_inches="tight")
    plt.close(fig)

    source_summary = {}
    for li, layer in enumerate(MIXER_LAYERS):
        rows = []
        for group in range(len(GROUP_NAMES)):
            g = group_writes[:, 0, li, group]
            n = group_writes[:, 1, li, group]
            d = g - n
            dm, dl, dh = _interval(d[:, None], strata, 20261120 + li * 100 + group)
            head_diff = conflict_routes[:, 0, li, group] - conflict_routes[:, 1, li, group]
            head_mean = head_diff.mean(axis=0)
            rows.append({
                "group": GROUP_NAMES[group],
                "label": group_names[group],
                "game_mean": float(g.mean()),
                "neutral_mean": float(n.mean()),
                "game_minus_neutral": float(d.mean()),
                "game_minus_neutral_ci": [float(dl[0]), float(dh[0])],
                "attention_mass_game": float(conflict_mass[:, 0, li, group].sum(axis=-1).mean()),
                "attention_mass_neutral": float(conflict_mass[:, 1, li, group].sum(axis=-1).mean()),
                "largest_head": int(np.argmax(np.abs(head_mean))),
                "largest_head_game_minus_neutral": float(head_mean[np.argmax(np.abs(head_mean))]),
            })
        source_summary[f"Mixer {layer + 1}"] = sorted(
            rows, key=lambda row: abs(row["game_minus_neutral"]), reverse=True
        )
    summary["source_contributions"] = source_summary
    # Full W1-directed mixer writes and their head decomposition.
    head_writes = conflict_routes.sum(axis=3)  # question, condition, mixer, head
    head_summary: dict[str, Any] = {}
    for li, layer in enumerate(MIXER_LAYERS):
        game_heads = head_writes[:, 0, li].mean(axis=0)
        neutral_heads = head_writes[:, 1, li].mean(axis=0)
        difference_heads = game_heads - neutral_heads
        ranked_heads = np.argsort(np.abs(difference_heads))[::-1]
        head_summary[f"Mixer {layer + 1}"] = {
            "game_total": float(game_heads.sum()),
            "neutral_total": float(neutral_heads.sum()),
            "game_minus_neutral": float(difference_heads.sum()),
            "heads": [
                {
                    "head": int(head),
                    "game": float(game_heads[head]),
                    "neutral": float(neutral_heads[head]),
                    "game_minus_neutral": float(difference_heads[head]),
                }
                for head in ranked_heads
            ],
        }
    summary["head_totals"] = head_summary

    # Prespecified no-conflict check. Here W1 == W2, so there is one focal
    # semantic answer rather than distinct historical and fresh winners.
    no_conflict = np.flatnonzero(~conflict)
    no_conflict_strata = w1[no_conflict]
    no_conflict_routes = w1_routes[no_conflict]
    no_conflict_group_writes = no_conflict_routes.sum(axis=-1)
    no_conflict_summary: dict[str, Any] = {
        "n": int(len(no_conflict)), "source_contributions": {}, "state_swaps": {}
    }
    for li, layer in enumerate(MIXER_LAYERS):
        total = no_conflict_group_writes[:, :, li].sum(axis=-1)
        difference = total[:, 0] - total[:, 1]
        dm, dl, dh = _interval(
            difference[:, None], no_conflict_strata, 20261220 + li
        )
        group_difference = (
            no_conflict_group_writes[:, 0, li]
            - no_conflict_group_writes[:, 1, li]
        ).mean(axis=0)
        strongest = np.argsort(np.abs(group_difference))[::-1][:5]
        no_conflict_summary["source_contributions"][f"Mixer {layer + 1}"] = {
            "game_total": float(total[:, 0].mean()),
            "neutral_total": float(total[:, 1].mean()),
            "game_minus_neutral": float(dm[0]),
            "game_minus_neutral_ci": [float(dl[0]), float(dh[0])],
            "strongest_group_differences": [
                {
                    "group": GROUP_NAMES[index],
                    "label": group_names[index],
                    "game_minus_neutral": float(group_difference[index]),
                }
                for index in strongest
            ],
        }
    for si, swap_name in enumerate(SWAP_NAMES):
        no_conflict_summary["state_swaps"][swap_name] = {}
        for ci, direction in enumerate(("Neutral into Game", "Game into Neutral")):
            w1_effect = np.empty((len(no_conflict), len(SOURCE_READOUTS)))
            choice_effect = np.empty_like(w1_effect)
            for ri in range(len(SOURCE_READOUTS)):
                delta = (
                    patched_sem[si, ci, ri, no_conflict]
                    - natural_sem[ci, no_conflict]
                )
                w1_effect[:, ri] = delta[
                    np.arange(len(no_conflict)), w1[no_conflict]
                ]
                natural_displayed = natural[ci, no_conflict].argmax(axis=-1)
                patched_displayed = patched[
                    si, ci, ri, no_conflict
                ].argmax(axis=-1)
                natural_choice = displayed_to_semantic[
                    no_conflict, natural_displayed
                ]
                patched_choice = displayed_to_semantic[
                    no_conflict, patched_displayed
                ]
                choice_effect[:, ri] = 100 * (
                    (patched_choice == w1[no_conflict]).astype(float)
                    - (natural_choice == w1[no_conflict]).astype(float)
                )
            wm, wl, wh = _interval(
                w1_effect, no_conflict_strata, 20261320 + si * 10 + ci
            )
            cm, cl, ch = _interval(
                choice_effect, no_conflict_strata, 20261420 + si * 10 + ci
            )
            no_conflict_summary["state_swaps"][swap_name][direction] = {
                "W1": {
                    str(layer): {"mean": float(m), "ci": [float(lo), float(hi)]}
                    for layer, m, lo, hi in zip(SOURCE_READOUTS, wm, wl, wh)
                },
                "W1_selection_pp": {
                    str(layer): {"mean": float(m), "ci": [float(lo), float(hi)]}
                    for layer, m, lo, hi in zip(SOURCE_READOUTS, cm, cl, ch)
                },
            }
    summary["no_conflict"] = no_conflict_summary
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def top_rows(layer: str, count: int = 8) -> str:
        return "\n".join(
            f"| {row['label']} | {row['game_mean']:+.4f} | {row['neutral_mean']:+.4f} | "
            f"{row['game_minus_neutral']:+.4f} [{row['game_minus_neutral_ci'][0]:+.4f}, "
            f"{row['game_minus_neutral_ci'][1]:+.4f}] | H{row['largest_head']} "
            f"({row['largest_head_game_minus_neutral']:+.4f}) |"
            for row in source_summary[layer][:count]
        )

    swap_lines = []
    for swap_name in SWAP_NAMES:
        for direction in ("Neutral into Game", "Game into Neutral"):
            rows = summary["state_swaps"][swap_name][direction]["W1"]
            layer, result = max(rows.items(), key=lambda item: abs(item[1]["mean"]))
            swap_lines.append(
                f"- `{swap_name}`, {direction}: largest W1 effect {result['mean']:+.4f} "
                f"at readout {layer} (95% CI {result['ci'][0]:+.4f} to {result['ci'][1]:+.4f})."
            )

    no_conflict_source_lines = []
    for layer in ("Mixer 52", "Mixer 56"):
        row = no_conflict_summary["source_contributions"][layer]
        no_conflict_source_lines.append(
            f"- {layer}: Game {row['game_total']:+.3f}, Neutral "
            f"{row['neutral_total']:+.3f}, difference "
            f"{row['game_minus_neutral']:+.3f} (95% CI "
            f"{row['game_minus_neutral_ci'][0]:+.3f} to "
            f"{row['game_minus_neutral_ci'][1]:+.3f})."
        )
    no_conflict_swap_max = max(
        (
            abs(result["mean"]), swap_name, direction, layer, result
        )
        for swap_name, directions in no_conflict_summary["state_swaps"].items()
        for direction, metrics in directions.items()
        for layer, result in metrics["W1_selection_pp"].items()
    )

    natural_content_choice = np.empty((2, len(qids)), dtype=np.int8)
    for ci in range(2):
        displayed = natural[ci].argmax(axis=-1)
        natural_content_choice[ci] = displayed_to_semantic[
            np.arange(len(qids)), displayed
        ]
    natural_w1_rates = [
        float((natural_content_choice[ci, selected] == w1[selected]).mean() * 100)
        for ci in range(2)
    ]
    max_conflict_choice = float(np.max(np.abs(choice_effects.mean(axis=2))))
    m52 = head_summary["Mixer 52"]
    m56 = head_summary["Mixer 56"]
    m56_groups = {row["group"]: row for row in source_summary["Mixer 56"]}
    leading_56 = m56["heads"][:4]
    leading_52 = m52["heads"][0]

    report = f"""# Exact remapped source contributions and feedback-token swaps

## Bottom line

The late W1 difference is not supplied by direct reads from the literal
`incorrect`, `different`, or feedback-period tokens.  On the {len(selected)}
conflict trials, Mixer 56 writes {m56['game_total']:+.3f} centered W1 units in
Game but {m56['neutral_total']:+.3f} in Neutral.  Most of that
{m56['game_minus_neutral']:+.3f} Game-minus-Neutral difference comes from the
states over the *second presentation itself*: the repeated question and its
option-boundary states
({m56_groups['second_question_stem']['game_minus_neutral']:+.3f}), the identical
final `Your choice (A, B, C, or D):` cue
({m56_groups['second_choice_cue']['game_minus_neutral']:+.3f}), and the repeated
option containing W1
({m56_groups['second_option_w1']['game_minus_neutral']:+.3f}).

This is best described as **feedback-conditioned reprocessing of the repeated
question**.  Neutral strongly reconstructs or reinstates W1 from the second
presentation; Game does much less of that.  The source decomposition is exact
and additive for the natural mixer write, but it is a computational
decomposition rather than an independent causal ablation of each source region.

The head-level result is concentrated but not single-head.  At Mixer 56, the
four largest head contrasts are H{leading_56[0]['head']}
({leading_56[0]['game_minus_neutral']:+.3f}), H{leading_56[1]['head']}
({leading_56[1]['game_minus_neutral']:+.3f}), H{leading_56[2]['head']}
({leading_56[2]['game_minus_neutral']:+.3f}), and H{leading_56[3]['head']}
({leading_56[3]['game_minus_neutral']:+.3f}).  At Mixer 52, H{leading_52['head']}
supplies {leading_52['game_minus_neutral']:+.3f} of the
{m52['game_minus_neutral']:+.3f} total contrast.

## Validation

- All {len(qids)} questions ran in the exact historical batch-of-four SDPA cohorts.
- Natural A--D logits match the trusted remapped run exactly (maximum absolute error {natural_error}).
- Per-token/per-head contributions reconstruct the actual gated attention context with maximum absolute error {max_context_error:.6g}.
- Primary semantic analysis uses the {len(selected)} W1 != W2 conflict trials.

## Source contributions

Each number is the exact additive contribution of a source region to the
mixer's residual write, projected onto the canonical W1 unembedding direction
and centered across A--D. It is not an attention weight alone. The contributions
sum across tokens and heads to the complete ordinary-attention write.

### Mixer 52: largest Game--Neutral source differences

| Source region | Game | Neutral | Game - Neutral (95% CI) | Largest head |
|---|---:|---:|---:|---:|
{top_rows('Mixer 52')}

### Mixer 56: largest Game--Neutral source differences

| Source region | Game | Neutral | Game - Neutral (95% CI) | Largest head |
|---|---:|---:|---:|---:|
{top_rows('Mixer 56')}

## Causal token-state swaps

The `incorrect`/`lost` result is a corrected rerun: the predecessor runner used
unpadded character-derived positions directly against left-padded batches, so
its intervention was misaligned on shorter rows. These results add each row's
actual padding offset before collecting or replacing states.

All three single-token state swaps are small relative to the natural
{natural_w1_rates[1] - natural_w1_rates[0]:.1f}-percentage-point W1-selection
gap on conflict trials (Game {natural_w1_rates[0]:.1f}%, Neutral
{natural_w1_rates[1]:.1f}%).  The largest absolute W1-selection change from any
tested token, direction, or readout is {max_conflict_choice:.2f} percentage
points.  None of the three token states at one late readout is therefore the
controller of the behavioral difference.

{chr(10).join(swap_lines)}

## Prespecified no-conflict check

The source pattern is not unique to W1 != W2 trials.  When the original and
fresh-remapped winners agree (n={len(no_conflict)}), the total W1-directed
writes remain much smaller in Game:

{chr(10).join(no_conflict_source_lines)}

Single-token swaps are also small on these trials.  The largest absolute
W1-selection change across every token, direction, and readout is
{no_conflict_swap_max[0]:.2f} percentage points
(`{no_conflict_swap_max[1]}`, {no_conflict_swap_max[2]}, readout
{no_conflict_swap_max[3]}).

## Artifacts

- `{figure_sources}`
- `{figure_swaps}`
- `summary.json`
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-sources", type=Path, required=True)
    parser.add_argument("--figure-swaps", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        args.run_dir,
        args.original_baseline,
        args.remapped_baseline,
        args.trusted_game,
        args.trusted_neutral,
        args.plan,
        args.output,
        args.figure_sources,
        args.figure_swaps,
    )


if __name__ == "__main__":
    main()
