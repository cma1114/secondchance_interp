from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = ("Game", "Neutral")


def _role_block_summary(
    values: np.ndarray,
    role_codes: np.ndarray,
    role_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    # values: condition, block, question, position
    conditions, blocks, questions, _positions = values.shape
    means = np.full((conditions, blocks, role_count), np.nan, dtype=float)
    coverage = np.zeros((conditions, role_count), dtype=float)
    for ci in range(conditions):
        for role in range(role_count):
            per_question: list[tuple[int, float]] = []
            present = 0
            for qi in range(questions):
                mask = role_codes[ci, qi] == role
                if not np.any(mask):
                    continue
                present += 1
                for bi in range(blocks):
                    finite = values[ci, bi, qi, mask].astype(float)
                    finite = finite[np.isfinite(finite)]
                    if finite.size:
                        per_question.append((bi, float(finite.max())))
            coverage[ci, role] = present / questions
            for bi in range(blocks):
                selected = [value for block, value in per_question if block == bi]
                if selected:
                    means[ci, bi, role] = float(np.mean(selected))
    return means, coverage


def _candidate_plan(
    role_names: list[str],
    blocks: np.ndarray,
    means: np.ndarray,
    coverage: np.ndarray,
) -> dict[str, Any]:
    eligible_roles = [
        index
        for index, name in enumerate(role_names)
        if name not in {
            "ignored_before_all_first_options",
            "final_decision_query_known_null",
        }
        and min(coverage[:, index]) == 1.0
        and np.isfinite(means[:, :, index]).all()
    ]
    pairs = []
    for role in eligible_roles:
        for bi, block in enumerate(blocks):
            raw = float(np.max(means[:, bi, role]))
            contrast = float(abs(means[0, bi, role] - means[1, bi, role]))
            pairs.append((role, bi, int(block), raw, contrast))
    by_raw = sorted(pairs, key=lambda row: (-row[3], -row[4], row[2], role_names[row[0]]))
    by_contrast = sorted(pairs, key=lambda row: (-row[4], -row[3], row[2], role_names[row[0]]))

    selected_pairs: list[tuple[int, int, int, float, float, str]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for reason, rows, limit in (("largest_source_specific_write", by_raw, 3), ("largest_game_neutral_change", by_contrast, 3)):
        added = 0
        for role, bi, block, raw, contrast in rows:
            key = (role, bi)
            if key in seen_pairs:
                continue
            selected_pairs.append((role, bi, block, raw, contrast, reason))
            seen_pairs.add(key)
            added += 1
            if added == limit:
                break

    # For the two strongest receiver roles, include the earliest block whose
    # source-specific write reaches half that role's maximum. This prevents a
    # pure magnitude screen from looking only at late readout layers.
    role_raw = {
        role: float(np.nanmax(means[:, :, role])) for role in eligible_roles
    }
    for role in sorted(eligible_roles, key=lambda r: (-role_raw[r], role_names[r]))[:2]:
        threshold = 0.5 * role_raw[role]
        for bi, block in enumerate(blocks):
            if float(np.max(means[:, bi, role])) >= threshold:
                key = (role, bi)
                if key not in seen_pairs:
                    selected_pairs.append(
                        (
                            role,
                            bi,
                            int(block),
                            float(np.max(means[:, bi, role])),
                            float(abs(means[0, bi, role] - means[1, bi, role])),
                            "earliest_half_maximum_onset",
                        )
                    )
                    seen_pairs.add(key)
                break

    role_rows = []
    for role in eligible_roles:
        raw = float(np.nanmax(means[:, :, role]))
        contrast = float(np.nanmax(np.abs(means[0, :, role] - means[1, :, role])))
        role_rows.append((role, raw, contrast))
    by_role_raw = sorted(role_rows, key=lambda row: (-row[1], -row[2], role_names[row[0]]))
    by_role_contrast = sorted(role_rows, key=lambda row: (-row[2], -row[1], role_names[row[0]]))
    selected_roles: list[tuple[int, float, float, str]] = []
    seen_roles: set[int] = set()
    for reason, rows, limit in (("role_largest_source_specific_write", by_role_raw, 2), ("role_largest_game_neutral_change", by_role_contrast, 2)):
        added = 0
        for role, raw, contrast in rows:
            if role in seen_roles:
                continue
            selected_roles.append((role, raw, contrast, reason))
            seen_roles.add(role)
            added += 1
            if added == limit:
                break

    candidates = []
    for role, _bi, block, raw, contrast, reason in selected_pairs:
        candidates.append(
            {
                "id": f"block_{block:02d}__{role_names[role]}",
                "role": role_names[role],
                "blocks": [block],
                "screen_source_specific_write_norm": raw,
                "screen_abs_game_neutral_change": contrast,
                "selection_reason": reason,
            }
        )
    for role, raw, contrast, reason in selected_roles:
        candidates.append(
            {
                "id": f"all_04_48__{role_names[role]}",
                "role": role_names[role],
                "blocks": [int(value) for value in blocks if int(value) <= 48],
                "screen_source_specific_write_norm": raw,
                "screen_abs_game_neutral_change": contrast,
                "selection_reason": reason,
            }
        )
    return {
        "selection_split": "frozen 251-question discovery set only",
        "minimum_role_coverage_each_condition": 1.0,
        "source_metric": "per-question maximum norm of W1-line gated projected write minus matched-line write within receiver role",
        "candidates": candidates,
    }


def analyze(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("Receiver screen is incomplete")
    natural_error = float(
        np.max(
            np.abs(
                arrays["same_batch_natural_logits"]
                - arrays["trusted_natural_logits"]
            )
        )
    )
    if natural_error != 0.0:
        raise RuntimeError(f"Natural logits failed exact reproduction: {natural_error}")

    role_names = arrays["role_names"].astype(str).tolist()
    blocks = arrays["ordinary_blocks"].astype(int)
    means, coverage = _role_block_summary(
        arrays["difference_write_norm"], arrays["role_codes"], len(role_names)
    )
    plan = _candidate_plan(role_names, blocks, means, coverage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )

    rows = []
    for role, name in enumerate(role_names):
        if min(coverage[:, role]) < 1.0 or not np.isfinite(means[:, :, role]).all():
            continue
        for bi, block in enumerate(blocks):
            rows.append(
                {
                    "role": name,
                    "block": int(block),
                    "game": float(means[0, bi, role]),
                    "neutral": float(means[1, bi, role]),
                    "absolute_difference": float(abs(means[0, bi, role] - means[1, bi, role])),
                    "coverage_game": float(coverage[0, role]),
                    "coverage_neutral": float(coverage[1, role]),
                }
            )
    (args.output_dir / "screen_summary.json").write_text(
        json.dumps(
            {
                "natural_logits_max_abs_error": natural_error,
                "rows": rows,
                "candidate_plan": plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    selected_roles = []
    for candidate in plan["candidates"]:
        if candidate["role"] not in selected_roles:
            selected_roles.append(candidate["role"])
    # Add the strongest remaining roles to make the heatmap useful without
    # creating additional causal candidates.
    strongest = sorted(
        range(len(role_names)),
        key=lambda role: -float(np.nanmax(means[:, :, role]))
        if np.isfinite(means[:, :, role]).any()
        else float("inf"),
    )
    for role in strongest:
        if min(coverage[:, role]) < 1.0:
            continue
        if role_names[role] not in selected_roles:
            selected_roles.append(role_names[role])
        if len(selected_roles) >= 18:
            break
    role_indices = [role_names.index(name) for name in selected_roles]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    matrices = (
        means[0][:, role_indices].T,
        means[1][:, role_indices].T,
        np.abs(means[0][:, role_indices] - means[1][:, role_indices]).T,
    )
    titles = ("Game", "Neutral", "Absolute Game–Neutral change")
    vmax = max(float(np.nanmax(matrix)) for matrix in matrices)
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax, cmap="magma")
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(range(len(blocks)), blocks, rotation=45)
        axis.set_xlabel("Ordinary-attention block")
        axis.set_yticks(range(len(selected_roles)), selected_roles)
    fig.colorbar(image, ax=axes, shrink=0.7, label="Source-specific projected write norm")
    fig.suptitle("Where do downstream queries read the first answer's semantic option line?", fontsize=15, fontweight="bold")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Canonical remapped receiver source-write screen",
        "",
        "This is a discovery-only observational screen. It nominates exact receiver/block candidates for causal edge ablation; attention or write magnitude is not itself a mechanism.",
        "",
        f"Trusted natural logits reproduced exactly (maximum absolute error `{natural_error}`).",
        "",
        "## Frozen causal candidates",
        "",
        "| Candidate | Receiver role | Blocks | Screen criterion | Source-specific norm | Absolute Game-Neutral change |",
        "|---|---|---|---|---:|---:|",
    ]
    for candidate in plan["candidates"]:
        blocks_text = ",".join(str(value) for value in candidate["blocks"])
        lines.append(
            f"| `{candidate['id']}` | `{candidate['role']}` | {blocks_text} | {candidate['selection_reason']} | {candidate['screen_source_specific_write_norm']:.4f} | {candidate['screen_abs_game_neutral_change']:.4f} |"
        )
    lines += ["", f"Figure: `{args.figure}`.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
