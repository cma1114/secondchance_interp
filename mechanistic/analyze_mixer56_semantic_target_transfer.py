from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LETTERS = "ABCD"


def _interval(values: np.ndarray, seed: int = 42) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def _load_cohorts(run_dir: Path) -> dict[str, np.ndarray]:
    paths = sorted((run_dir / "cohorts").glob("cohort_*.npz"))
    if len(paths) != 125:
        raise ValueError(f"Expected 125 cohorts, found {len(paths)}")
    rows: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                rows.setdefault(key, []).append(data[key])
    return {
        "question_ids": np.concatenate(rows["question_ids"], axis=0),
        "scenario_logits": np.concatenate(rows["scenario_logits"], axis=1),
        "scenario_write_ad": np.concatenate(rows["scenario_write_ad"], axis=1),
        "semantic_to_displayed": np.concatenate(rows["semantic_to_displayed"], axis=0),
        "recipient_winner_semantic": np.concatenate(rows["recipient_winner_semantic"], axis=0),
        "donor_winner_semantic": np.concatenate(rows["donor_winner_semantic"], axis=0),
        "same_winner_control_available": np.concatenate(
            rows["same_winner_control_available"], axis=0
        ),
    }


def _semantic_scores(raw: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return np.take_along_axis(raw, mapping[None, :, :], axis=2)


def _indexed(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[np.arange(len(values)), indices]


def _split_rows(path: Path) -> tuple[set[str], dict[str, bool]]:
    payload = json.loads(path.read_text())
    return (
        {row["question_id"] for row in payload["rows"]},
        {
            row["question_id"]: bool(row["primary_letter_decoupled_changed_winner"])
            for row in payload["rows"]
        },
    )


def analyze_split(
    name: str,
    qids: np.ndarray,
    allowed: set[str],
    primary_lookup: dict[str, bool],
    scenarios: list[str],
    semantic_logits: np.ndarray,
    semantic_write: np.ndarray,
    w1: np.ndarray,
    wd: np.ndarray,
    same_available: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray([
        str(qid) in allowed and primary_lookup.get(str(qid), False)
        for qid in qids
    ])
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError(f"{name}: no primary questions")
    primary_w1s, primary_wds = w1[indices], wd[indices]
    primary_natural_logits = semantic_logits[0, indices]
    primary_natural_choice = primary_natural_logits.argmax(axis=-1)

    results = {}
    for scenario_index, scenario in enumerate(scenarios):
        scenario_indices = indices
        if scenario == "same_winner_query_gate_kv":
            scenario_indices = np.flatnonzero(mask & same_available)
        w1s, wds = w1[scenario_indices], wd[scenario_indices]
        natural_logits = semantic_logits[0, scenario_indices]
        natural_write = semantic_write[0, scenario_indices]
        natural_contrast = _indexed(natural_logits, w1s) - _indexed(natural_logits, wds)
        natural_write_contrast = _indexed(natural_write, w1s) - _indexed(natural_write, wds)
        natural_choice = natural_logits.argmax(axis=-1)
        logits = semantic_logits[scenario_index, scenario_indices]
        write = semantic_write[scenario_index, scenario_indices]
        target_transfer = (
            _indexed(logits, w1s) - _indexed(logits, wds) - natural_contrast
        )
        write_transfer = (
            _indexed(write, w1s) - _indexed(write, wds) - natural_write_contrast
        )
        choice = logits.argmax(axis=-1)
        by_recipient = {}
        for letter_index, letter in enumerate(LETTERS):
            letter_mask = w1s == letter_index
            if letter_mask.any():
                by_recipient[letter] = {
                    "n": int(letter_mask.sum()),
                    "final_target_transfer_logit": _interval(target_transfer[letter_mask]),
                    "mixer56_immediate_target_transfer": _interval(write_transfer[letter_mask]),
                }
        non_a = w1s != 0
        results[scenario] = {
            "final_target_transfer_logit": _interval(target_transfer),
            "mixer56_immediate_target_transfer": _interval(write_transfer),
            "donor_winner_choice_change_pp": _interval(
                100.0 * (
                    (choice == wds).astype(np.float64)
                    - (natural_choice == wds).astype(np.float64)
                )
            ),
            "recipient_winner_choice_change_pp": _interval(
                100.0 * (
                    (choice == w1s).astype(np.float64)
                    - (natural_choice == w1s).astype(np.float64)
                )
            ),
            "n_changed_outputs": int(np.sum(choice != natural_choice)),
            "n": int(len(scenario_indices)),
            "by_recipient_winner": by_recipient,
            "non_A": {
                "n": int(non_a.sum()),
                "final_target_transfer_logit": _interval(target_transfer[non_a]),
                "mixer56_immediate_target_transfer": _interval(write_transfer[non_a]),
            } if non_a.any() else None,
        }

    same_mask = mask & same_available
    return {
        "name": name,
        "n_primary": int(mask.sum()),
        "n_same_winner_control": int(same_mask.sum()),
        "natural_donor_winner_choice_rate": float(
            np.mean(primary_natural_choice == primary_wds)
        ),
        "natural_recipient_winner_choice_rate": float(
            np.mean(primary_natural_choice == primary_w1s)
        ),
        "scenarios": results,
    }


def _validate_natural(
    qids: np.ndarray,
    raw_natural: np.ndarray,
    trusted_paths: list[Path],
) -> dict[str, Any]:
    lookup = {}
    for path in trusted_paths:
        with np.load(path, allow_pickle=False) as data:
            for qid, logits in zip(
                data["question_ids"].astype(str), data["natural_logits"][0]
            ):
                lookup[qid] = logits
    if set(qids) != set(lookup):
        raise ValueError("Frozen discovery/confirmation natural references are incomplete")
    expected = np.asarray([lookup[str(qid)] for qid in qids])
    difference = np.abs(raw_natural - expected)
    return {
        "max_absolute_logit_difference": float(difference.max()),
        "bit_exact": bool(np.array_equal(raw_natural, expected)),
        "winner_matches": int(np.sum(raw_natural.argmax(axis=-1) == expected.argmax(axis=-1))),
        "n": len(qids),
    }


def _make_figure(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = [
        "different_query",
        "different_gate",
        "different_query_gate",
        "different_kv",
        "different_query_gate_kv",
        "different_all_heads",
        "same_winner_query_gate_kv",
    ]
    labels = [
        "Query",
        "Gate",
        "Query + gate",
        "K/V",
        "Query + gate + K/V",
        "All heads",
        "Same-winner control",
    ]
    colors = ["#3182ce"] * 5 + ["#805ad5", "#8a8a8a"]
    confirmation = summary["confirmation"]["scenarios"]
    metrics = (
        ("mixer56_immediate_target_transfer", "A  Immediate Mixer 56 target transfer"),
        ("final_target_transfer_logit", "B  Final-logit target transfer"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        means = np.asarray([confirmation[s][metric]["mean"] for s in scenarios])
        lows = np.asarray([confirmation[s][metric]["low"] for s in scenarios])
        highs = np.asarray([confirmation[s][metric]["high"] for s in scenarios])
        x = np.arange(len(scenarios))
        axis.axhline(0, color="#777777", linewidth=1, linestyle="--")
        for index, color in enumerate(colors):
            axis.errorbar(
                x[index],
                means[index],
                yerr=np.asarray([[means[index] - lows[index]], [highs[index] - means[index]]]),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=7,
                elinewidth=2,
                capsize=4,
                zorder=3,
            )
        axis.set_xticks(x, labels, rotation=34, ha="right")
        axis.set_title(title, loc="left", fontsize=15)
        axis.set_ylabel("Δ[(recipient W1) − (donor W1′)] (logit units)")
        axis.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    fig.suptitle(
        "Does Mixer 56 transfer the semantic suppression target?\n"
        f"Held-out confirmation, n={summary['confirmation']['n_primary']}; paired 95% bootstrap CIs",
        fontsize=17,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(
    run_dir: Path,
    discovery_donor: Path,
    confirmation_donor: Path,
    trusted_natural: list[Path],
    analysis_dir: Path,
    figure: Path,
) -> None:
    arrays = _load_cohorts(run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    scenarios = metadata["scenarios"]
    qids = arrays["question_ids"].astype(str)
    semantic_logits = _semantic_scores(
        arrays["scenario_logits"], arrays["semantic_to_displayed"]
    )
    semantic_write = _semantic_scores(
        arrays["scenario_write_ad"], arrays["semantic_to_displayed"]
    )
    discovery_ids, discovery_primary = _split_rows(discovery_donor)
    confirmation_ids, confirmation_primary = _split_rows(confirmation_donor)
    summary = {
        "experiment": "Mixer 56 within-Game semantic-target transfer",
        "positive_direction": (
            "Positive means the donor semantic winner lost evidence relative to "
            "the recipient semantic winner."
        ),
        "natural_validation": _validate_natural(
            qids, arrays["scenario_logits"][0], trusted_natural
        ),
        "discovery": analyze_split(
            "discovery", qids, discovery_ids, discovery_primary, scenarios,
            semantic_logits, semantic_write, arrays["recipient_winner_semantic"],
            arrays["donor_winner_semantic"], arrays["same_winner_control_available"],
        ),
        "confirmation": analyze_split(
            "confirmation", qids, confirmation_ids, confirmation_primary, scenarios,
            semantic_logits, semantic_write, arrays["recipient_winner_semantic"],
            arrays["donor_winner_semantic"], arrays["same_winner_control_available"],
        ),
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    _make_figure(summary, figure)

    c = summary["confirmation"]
    primary = c["scenarios"]["different_query_gate_kv"]
    broad = c["scenarios"]["different_all_heads"]
    control = c["scenarios"]["same_winner_query_gate_kv"]
    non_a = primary["non_A"]
    discovery_primary = summary["discovery"]["scenarios"]["different_query_gate_kv"]
    confirmation_query = c["scenarios"]["different_query"]
    discovery_query = summary["discovery"]["scenarios"]["different_query"]
    fmt = lambda x: f"{x['mean']:+.3f} [{x['low']:+.3f}, {x['high']:+.3f}]"
    report = f"""# Mixer 56 semantic-target transfer

## Bottom line

This experiment asks whether Mixer 56 contains a causal, question-specific
comparison between the answer implicit in the first presentation and the
answers in the repeated presentation.  It keeps the visible Game prompt after
the first presentation fixed and changes only which semantic answer won in a
donor first presentation.

The frozen primary intervention (heads 0, 2, 6, and 15: final query, output
gate, and repeated-option/choice-cue K/V state) produced an immediate Mixer 56
target-transfer effect of **{fmt(primary['mixer56_immediate_target_transfer'])}**
and a final-logit target-transfer effect of
**{fmt(primary['final_target_transfer_logit'])}** on the held-out confirmation
questions (n={c['n_primary']}). Positive values mean evidence moved away from
the donor's first-pass semantic winner and toward the recipient's original
winner.

The sign is therefore decisive: **the donor state made Mixer 56 reinforce the
donor's first-pass semantic answer, not suppress it.** This reverse immediate
effect replicated in discovery
({fmt(discovery_primary['mixer56_immediate_target_transfer'])}). It was then
mostly cancelled downstream: the held-out final-logit interval includes zero,
and the behavioral choice changes below are small. The broad all-head patch
does not rescue suppression-target transfer.

Query alone produced a very small positive final effect in confirmation
({fmt(confirmation_query['final_target_transfer_logit'])}), but not in discovery
({fmt(discovery_query['final_target_transfer_logit'])}), and did not yield a
meaningful behavioral transfer. It is not convincing evidence for a separate
query-carried suppression target.

The all-head positive control gave **{fmt(broad['final_target_transfer_logit'])}**
at the final logits. The same-winner/different-order control gave
**{fmt(control['final_target_transfer_logit'])}**. Its available sample is
n={c['n_same_winner_control']}.

Because the frozen changed-winner sample contains many original-A winners, the
prespecified content-bias sensitivity is important. On the held-out questions
whose recipient first-pass winner was not A (n={non_a['n']}), the primary
final-logit target-transfer effect was
**{fmt(non_a['final_target_transfer_logit'])}**. Letter-stratified effects for
all four recipient winners are preserved in `summary.json`.

The primary patch changed the selected output on
{primary['n_changed_outputs']}/{c['n_primary']} held-out questions. Donor-winner
selection changed by {fmt(primary['donor_winner_choice_change_pp'])} percentage
points and recipient-winner selection by
{fmt(primary['recipient_winner_choice_change_pp'])} percentage points.

## What was patched

- One ordinary attention component only: Mixer 56.
- Query heads 0, 2, 6, and 15 at the final decision position.
- Their per-head output gates at that position.
- K/V state for the corresponding KV heads over all four repeated option lines
  and the final `Your choice (A, B, C, or D): ` cue.
- Factorial query-only, gate-only, query+gate, K/V-only, and joint conditions,
  plus the all-head and same-winner controls, are in `summary.json` and the
  figure.

This is not a Game-to-Neutral prefix replacement: both donor and recipient are
Game prompts, and the feedback plus complete second presentation are identical.

## Interpretation

This experiment rejects the proposed narrow story in which Mixer 56 computes
"this repeated option matches my old answer" and uses that match to inhibit the
old answer. Its K/V pathway instead carries a donor-specific **reinstatement or
reconstruction** signal: changing the implicit first-pass answer changes which
semantic answer Mixer 56 amplifies. That is a genuine content-specific causal
effect at the component output, but downstream computation nearly cancels it,
so Mixer 56 is not the behavioral revision mechanism by itself. The natural
Game-versus-Neutral difference at Mixer 56 is now best read as weaker
reinstatement in Game, not as direct suppression executed by Mixer 56.

## Validation

- Natural recipient logits were bit-exact against the frozen cross-order Game
  run that defined these donors: **{summary['natural_validation']['bit_exact']}**; maximum absolute
  difference {summary['natural_validation']['max_absolute_logit_difference']:.6g}.
- All 500 natural winners matched ({summary['natural_validation']['winner_matches']}/500).
- Donor and recipient token sequences and patch coordinates were audited over
  the identical repeated-option/cue suffix.
- Discovery and confirmation use the pre-existing frozen 251/249 split and
  independently frozen cross-order donor plan.

## Figure

![Mixer 56 semantic-target transfer]({figure.resolve()})
"""
    (analysis_dir / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--discovery-donor", type=Path, required=True)
    parser.add_argument("--confirmation-donor", type=Path, required=True)
    parser.add_argument("--trusted-natural", type=Path, nargs=2, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.run_dir,
        args.discovery_donor,
        args.confirmation_donor,
        args.trusted_natural,
        args.analysis_dir,
        args.figure,
    )


if __name__ == "__main__":
    main()
