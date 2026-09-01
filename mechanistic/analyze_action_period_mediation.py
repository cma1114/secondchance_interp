from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_evaluation_targeting_vs_compression import _bootstrap_model
from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
SCENARIOS = ("residual_trajectory", "gla_state", "joint")


def _entropy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=-1, keepdims=True)
    return -np.sum(p * np.log2(np.maximum(p, 1e-300)), axis=-1)


def _align(values: np.ndarray, qids: list[str], mappings: dict[str, dict]) -> np.ndarray:
    out = np.empty_like(values)
    for qi, qid in enumerate(qids):
        original_to_new = mappings[qid]["original_to_new"]
        for oi, original in enumerate(LETTERS):
            ni = LETTERS.index(original_to_new[original])
            out[..., qi, oi] = values[..., qi, ni]
    return out


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([
        rng.choice(group, size=len(group), replace=True)
        for group in (np.flatnonzero(labels == label) for label in np.unique(labels))
    ])


def _interval(values: np.ndarray, labels: np.ndarray, seed: int, draws: int = 5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = values[_stratified_indices(labels, rng)].mean()
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def _transfer_interval(
    transfer: np.ndarray,
    gap: np.ndarray,
    labels: np.ndarray,
    seed: int,
    draws: int = 5000,
):
    rng = np.random.default_rng(seed)
    fraction = np.empty(draws, dtype=float)
    transfer_samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = _stratified_indices(labels, rng)
        numerator = transfer[selected].mean()
        denominator = gap[selected].mean()
        transfer_samples[draw] = numerator
        fraction[draw] = numerator / denominator if abs(denominator) > 1e-12 else np.nan
    return {
        "natural_gap": _interval(gap, labels, seed + 1, draws),
        "signed_transfer": {
            "mean": float(transfer.mean()),
            "ci_low": float(np.quantile(transfer_samples, 0.025)),
            "ci_high": float(np.quantile(transfer_samples, 0.975)),
        },
        "fraction_of_natural_gap": {
            "mean": float(transfer.mean() / gap.mean()) if abs(gap.mean()) > 1e-12 else None,
            "ci_low": float(np.nanquantile(fraction, 0.025)),
            "ci_high": float(np.nanquantile(fraction, 0.975)),
        },
    }


def _candidate(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows = np.arange(len(indices))
    return values[rows, indices]


def _metrics(
    values: np.ndarray,
    w1i: np.ndarray,
    w2i: np.ndarray,
    choice: np.ndarray,
) -> dict[str, np.ndarray]:
    centered = values - values.mean(axis=-1, keepdims=True)
    w1 = _candidate(centered, w1i)
    w2 = _candidate(centered, w2i)
    return {
        "w1_advantage": 4.0 / 3.0 * w1,
        "w1_minus_w2_margin": w1 - w2,
        "w1_selection": (choice == w1i).astype(float),
        "w2_selection": (choice == w2i).astype(float),
        "entropy_bits": _entropy(values),
        "ad_spread_sd": centered.std(axis=-1),
    }


def _fmt(row: dict, scale: float = 1.0, digits: int = 3) -> str:
    return (
        f"{row['mean'] * scale:+.{digits}f} "
        f"[{row['ci_low'] * scale:+.{digits}f}, {row['ci_high'] * scale:+.{digits}f}]"
    )


def analyze(args) -> None:
    with np.load(args.results, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if not np.all(arrays["completed"]):
        raise RuntimeError("Action-period run is incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    mappings = {
        row["question_id"]: row for row in json.loads(args.remapping_plan.read_text())["rows"]
    }
    baseline = json.loads(args.baseline.read_text())["results"]
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    discovery = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    w1 = np.asarray([
        baseline[qid]["answer"] for qid in qids
    ])
    w2 = np.asarray([remapped[qid]["answer_original_content"] for qid in qids])
    w1i = np.asarray([LETTERS.index(value) for value in w1])
    w2i = np.asarray([LETTERS.index(value) for value in w2])
    conflict = w1 != w2
    discovery_mask = np.asarray([qid in discovery for qid in qids])

    natural = _align(arrays["trusted_natural_logits"].astype(float), qids, mappings)
    patched = _align(arrays["patched_logits"].astype(float), qids, mappings)
    mapping_rows = [mappings[qid] for qid in qids]
    natural_choice = displayed_argmax_to_semantic_indices(
        arrays["trusted_natural_logits"], mapping_rows
    )
    patched_choice = displayed_argmax_to_semantic_indices(
        arrays["patched_logits"], mapping_rows
    )
    natural_metrics = [
        _metrics(natural[ci], w1i, w2i, natural_choice[ci]) for ci in range(2)
    ]
    patched_metrics = [
        [
            _metrics(patched[ci, si], w1i, w2i, patched_choice[ci, si])
            for si in range(len(SCENARIOS))
        ]
        for ci in range(2)
    ]

    masks = {
        "all": np.ones(len(qids), dtype=bool),
        "conflict_W1_not_equal_W2": conflict,
        "no_conflict_W1_equal_W2": ~conflict,
        "conflict_W1_A": conflict & (w1 == "A"),
        "conflict_W1_B_to_D": conflict & (w1 != "A"),
        "no_conflict_W1_A": (~conflict) & (w1 == "A"),
        "no_conflict_W1_B_to_D": (~conflict) & (w1 != "A"),
        "discovery": discovery_mask,
        "confirmation": ~discovery_mask,
        "discovery_conflict": discovery_mask & conflict,
        "confirmation_conflict": (~discovery_mask) & conflict,
        "discovery_no_conflict": discovery_mask & (~conflict),
        "confirmation_no_conflict": (~discovery_mask) & (~conflict),
    }
    # Positive signed transfer always means movement toward the other condition.
    gap_sign = {
        "w1_advantage": +1,
        "w1_minus_w2_margin": +1,
        "w1_selection": +1,
        "w2_selection": -1,
        "entropy_bits": -1,
        "ad_spread_sd": +1,
    }
    summary = {
        "definitions": {
            "W1": "Semantic answer selected on the original first-presentation Baseline.",
            "W2": "Semantic answer selected by a fresh Baseline under the remapped second presentation.",
            "residual_trajectory": "Other-condition complete post-block residual at the action-closing period, clamped at all 64 blocks.",
            "gla_state": "Other-condition accumulated recurrent matrix state immediately after the action-closing period in all 48 GLAs.",
            "joint": "Both action-period interventions together.",
            "signed_transfer": "Movement from the recipient condition toward the donor condition. Positive is successful causal transfer.",
        },
        "validation": {
            "questions": len(qids),
            "conflict": int(conflict.sum()),
            "no_conflict": int((~conflict).sum()),
            "max_abs_same_batch_natural_minus_trusted": float(np.max(np.abs(
                arrays["same_batch_natural_logits"] - arrays["trusted_natural_logits"]
            ))),
            "max_abs_identity_state_minus_same_batch_natural": float(np.max(np.abs(
                arrays["identity_state_logits"] - arrays["same_batch_natural_logits"]
            ))),
        },
        "subsets": {},
        "compression_targeting": {},
        "state_and_residual_magnitude": {},
    }

    for subset_index, (name, mask) in enumerate(masks.items()):
        if not np.any(mask):
            continue
        labels = w1[mask]
        cell = {"n": int(mask.sum()), "scenarios": {}}
        for si, scenario in enumerate(SCENARIOS):
            scenario_cell = {"neutral_into_evaluation": {}, "evaluation_into_neutral": {}}
            for mi, (metric, sign) in enumerate(gap_sign.items()):
                # For W1 metrics/spread, natural gap is Neutral minus Evaluation.
                # For W2/entropy, `sign=-1` reverses it so positive retains the
                # same donor-transfer interpretation.
                gap = sign * (natural_metrics[1][metric] - natural_metrics[0][metric])
                n2e = sign * (
                    patched_metrics[0][si][metric] - natural_metrics[0][metric]
                )
                e2n = sign * (
                    natural_metrics[1][metric] - patched_metrics[1][si][metric]
                )
                scenario_cell["neutral_into_evaluation"][metric] = _transfer_interval(
                    n2e[mask], gap[mask], labels, 10000 + subset_index * 1000 + si * 100 + mi
                )
                scenario_cell["evaluation_into_neutral"][metric] = _transfer_interval(
                    e2n[mask], gap[mask], labels, 20000 + subset_index * 1000 + si * 100 + mi
                )
            cell["scenarios"][scenario] = scenario_cell
        summary["subsets"][name] = cell

    # Describe how much of the natural Evaluation-vs-Neutral transformation
    # remains after Neutral state is inserted into Evaluation. This reuses the
    # same constrained compression-plus-W1 model as the upstream deletion audit.
    centered_natural = natural - natural.mean(axis=-1, keepdims=True)
    centered_patched = patched - patched.mean(axis=-1, keepdims=True)
    target = np.full((len(qids), 4), -1.0 / 3.0)
    target[np.arange(len(qids)), w1i] = 1.0
    for name, mask in {
        "conflict": conflict,
        "no_conflict": ~conflict,
        "conflict_W1_A": conflict & (w1 == "A"),
        "conflict_W1_B_to_D": conflict & (w1 != "A"),
    }.items():
        labels = w1[mask]
        rows = {
            "natural_Evaluation_minus_Neutral": _bootstrap_model(
                centered_natural[1, mask],
                centered_natural[0, mask] - centered_natural[1, mask],
                target[mask], labels, 30000, 2000,
            )
        }
        for si, scenario in enumerate(SCENARIOS):
            rows[f"Neutral_into_Evaluation_{scenario}"] = _bootstrap_model(
                centered_natural[1, mask],
                centered_patched[0, si, mask] - centered_natural[1, mask],
                target[mask], labels, 30100 + si, 2000,
            )
        summary["compression_targeting"][name] = rows

    for target_ci, direction in ((0, "neutral_into_evaluation"), (1, "evaluation_into_neutral")):
        summary["state_and_residual_magnitude"][direction] = {
            "mean_gla_state_delta_norm_by_layer": arrays["state_delta_norm"][target_ci].mean(axis=0).tolist(),
            "mean_action_residual_delta_norm_by_layer": arrays["residual_delta_norm"][target_ci].mean(axis=0).tolist(),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    primary = summary["subsets"]["conflict_W1_not_equal_W2"]
    no_conflict = summary["subsets"]["no_conflict_W1_equal_W2"]
    lines = [
        "# Action-closing-period causal mediation",
        "",
        "Positive transfer means that replacing the recipient's action-period state moves it toward the donor condition. The table uses the W1-minus-W2 margin on conflict trials; percentages are fractions of the natural Evaluation-versus-Matched-Neutral gap.",
        "",
        "| Intervention | Direction | Conflict margin transfer | Fraction of natural gap | No-conflict W1-selection transfer |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for direction, label in (
            ("neutral_into_evaluation", "Neutral → Evaluation"),
            ("evaluation_into_neutral", "Evaluation → Neutral"),
        ):
            margin = primary["scenarios"][scenario][direction]["w1_minus_w2_margin"]
            selection = no_conflict["scenarios"][scenario][direction]["w1_selection"]
            lines.append(
                f"| `{scenario}` | {label} | {_fmt(margin['signed_transfer'])} | "
                f"{_fmt(margin['fraction_of_natural_gap'], 100, 1)}% | "
                f"{_fmt(selection['signed_transfer'], 100, 1)} pp |"
            )
    lines += [
        "",
        "## Validation",
        "",
        f"- Questions: {len(qids)} ({int(conflict.sum())} conflict; {int((~conflict).sum())} no conflict).",
        f"- Maximum same-batch natural deviation from trusted logits: {summary['validation']['max_abs_same_batch_natural_minus_trusted']:.6f} logits.",
        f"- Maximum recipient-state identity deviation from same-batch natural: {summary['validation']['max_abs_identity_state_minus_same_batch_natural']:.6f} logits. State effects are corrected against this identity pass.",
        "",
        "Full condition, subset, entropy, spread, compression, targeting, and magnitude results are in `summary.json`.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--remapping-plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--remapped-baseline", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
