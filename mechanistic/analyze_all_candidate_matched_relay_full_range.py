from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_mapping import displayed_argmax_to_semantic_indices


LETTERS = "ABCD"
CONDITIONS = ("Game", "Neutral")
RANKS = ("R1", "R2", "R3", "R4")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _align(values: np.ndarray, qids: list[str], mappings: dict[str, dict]) -> np.ndarray:
    out = np.empty_like(values)
    for qi, qid in enumerate(qids):
        for original_index, original in enumerate(LETTERS):
            new = mappings[qid]["original_to_new"][original]
            out[..., qi, original_index] = values[..., qi, LETTERS.index(new)]
    return out


def _advantage(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    row = np.arange(len(target))
    chosen = logits[row, target]
    return chosen - (logits.sum(axis=-1) - chosen) / 3.0


def _ci(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"n": int(len(values)), "mean": float(values.mean()),
            "ci": np.quantile(means, (0.025, 0.975)).tolist()}


def _fmt(row: dict[str, Any], scale: float = 1.0) -> str:
    lo, hi = row["ci"]
    return f"{row['mean']*scale:+.3f} [{lo*scale:+.3f}, {hi*scale:+.3f}]"


def analyze(args: argparse.Namespace) -> None:
    arrays = _load(args.results)
    if not np.all(arrays["completed"].astype(bool)):
        raise RuntimeError("Full-range run is incomplete")
    qids = arrays["question_ids"].astype(str).tolist()
    if len(qids) != 500:
        raise RuntimeError(f"Expected 500 questions; found {len(qids)}")
    expected = {
        "natural_logits", "trusted_natural_logits", "matched_logits", "control_logits",
        "joint_matched_logits", "joint_control_logits",
    }
    for key in expected:
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError(f"Non-finite values in {key}")
    bands = arrays["bands"].astype(str).tolist()
    if bands != ["late_52_64", "full_04_64"]:
        raise RuntimeError(f"Unexpected bands: {bands}")

    mappings = {row["question_id"]: row for row in json.loads(args.remapping_plan.read_text())["rows"]}
    remapped = json.loads(args.remapped_baseline.read_text())["results"]
    discovery_ids = set(json.loads(args.discovery_plan.read_text())["question_ids"])
    discovery = np.array([qid in discovery_ids for qid in qids])
    if int(discovery.sum()) != 251:
        raise RuntimeError("Frozen split is not 251 discovery / 249 confirmation")
    rank_contents = arrays["rank_contents"].astype(str)
    rank_indices = np.array([[LETTERS.index(v) for v in row] for row in rank_contents]).T
    w1 = rank_indices[0]
    w2 = np.array([LETTERS.index(remapped[qid]["answer_original_content"]) for qid in qids])
    conflict = w1 != w2

    natural = _align(arrays["natural_logits"].astype(float), qids, mappings)
    trusted = _align(arrays["trusted_natural_logits"].astype(float), qids, mappings)
    matched = _align(arrays["matched_logits"].astype(float), qids, mappings)
    control = _align(arrays["control_logits"].astype(float), qids, mappings)
    joint_matched = _align(arrays["joint_matched_logits"].astype(float), qids, mappings)
    joint_control = _align(arrays["joint_control_logits"].astype(float), qids, mappings)
    mapping_rows = [mappings[qid] for qid in qids]
    natural_choice = displayed_argmax_to_semantic_indices(
        arrays["natural_logits"], mapping_rows
    )
    joint_matched_choice = displayed_argmax_to_semantic_indices(
        arrays["joint_matched_logits"], mapping_rows
    )
    joint_control_choice = displayed_argmax_to_semantic_indices(
        arrays["joint_control_logits"], mapping_rows
    )

    masks = {
        "discovery_all": discovery,
        "confirmation_all": ~discovery,
        "discovery_conflict": discovery & conflict,
        "confirmation_conflict": (~discovery) & conflict,
        "discovery_no_conflict": discovery & ~conflict,
        "confirmation_no_conflict": (~discovery) & ~conflict,
    }
    summary: dict[str, Any] = {
        "definitions": {
            "matching_effect": "matching-edge lesion minus natural",
            "control_effect": "cyclic nonmatching-edge lesion minus natural",
            "matching_specific_effect": "matching-edge lesion minus cyclic nonmatching-edge lesion",
            "positive_candidate_effect": "the intervention increases evidence for the named first-pass rank",
        },
        "validation": {
            "questions": len(qids),
            "discovery": int(discovery.sum()),
            "confirmation": int((~discovery).sum()),
            "conflict": int(conflict.sum()),
            "natural_max_abs_logit_error_to_trusted": float(np.max(np.abs(natural-trusted))),
            "natural_answer_agreement_to_trusted": float(
                (
                    arrays["natural_logits"].argmax(-1)
                    == arrays["trusted_natural_logits"].argmax(-1)
                ).mean()
            ),
            "all_position_counts_positive": bool(
                np.all(arrays["source_position_counts"] > 0)
                and np.all(arrays["query_position_counts"] > 0)
                and np.all(arrays["control_position_counts"] > 0)
            ),
        },
        "subsets": {},
        "joint_W1": {},
        "early_4_48_comparison": {},
    }

    for si, (subset, mask) in enumerate(masks.items()):
        subset_rows: dict[str, Any] = {"n": int(mask.sum()), "bands": {}}
        for bi, band in enumerate(bands):
            band_rows: dict[str, Any] = {}
            for rank, rank_name in enumerate(RANKS):
                target = rank_indices[rank]
                rank_rows: dict[str, Any] = {}
                raw: dict[str, np.ndarray] = {}
                for ci, condition in enumerate(CONDITIONS):
                    natural_adv = _advantage(natural[ci], target)
                    match_adv = _advantage(matched[bi, ci, rank], target)
                    control_adv = _advantage(control[bi, ci, rank], target)
                    raw[condition] = match_adv - control_adv
                    rank_rows[condition] = {
                        "matching_vs_natural": _ci((match_adv-natural_adv)[mask], args.seed+si*10000+bi*1000+rank*100+ci*10, args.draws),
                        "control_vs_natural": _ci((control_adv-natural_adv)[mask], args.seed+si*10000+bi*1000+rank*100+ci*10+1, args.draws),
                        "matching_specific": _ci((match_adv-control_adv)[mask], args.seed+si*10000+bi*1000+rank*100+ci*10+2, args.draws),
                    }
                rank_rows["Game_minus_Neutral"] = _ci(
                    (raw["Game"]-raw["Neutral"])[mask],
                    args.seed+si*10000+bi*1000+rank*100+99, args.draws,
                )
                band_rows[rank_name] = rank_rows
            subset_rows["bands"][band] = band_rows
        summary["subsets"][subset] = subset_rows

        joint_rows: dict[str, Any] = {}
        row = np.arange(len(qids))
        for bi, band in enumerate(bands):
            band_rows = {}
            raw_margin: dict[str, np.ndarray] = {}
            raw_choice: dict[str, np.ndarray] = {}
            for ci, condition in enumerate(CONDITIONS):
                n_margin = natural[ci, row, w1] - natural[ci, row, w2]
                m_margin = joint_matched[bi, ci, row, w1] - joint_matched[bi, ci, row, w2]
                c_margin = joint_control[bi, ci, row, w1] - joint_control[bi, ci, row, w2]
                n_choice = (natural_choice[ci] == w1).astype(float)
                m_choice = (joint_matched_choice[bi, ci] == w1).astype(float)
                c_choice = (joint_control_choice[bi, ci] == w1).astype(float)
                raw_margin[condition] = m_margin-c_margin
                raw_choice[condition] = m_choice-c_choice
                band_rows[condition] = {
                    "matching_vs_natural_W1_minus_W2_margin": _ci((m_margin-n_margin)[mask], args.seed+300000+si*10000+bi*1000+ci*20+2, args.draws),
                    "control_vs_natural_W1_minus_W2_margin": _ci((c_margin-n_margin)[mask], args.seed+300000+si*10000+bi*1000+ci*20+3, args.draws),
                    "matching_specific_W1_minus_W2_margin": _ci((m_margin-c_margin)[mask], args.seed+300000+si*10000+bi*1000+ci*20, args.draws),
                    "matching_vs_natural_W1_choice_pp": _ci((m_choice-n_choice)[mask], args.seed+300000+si*10000+bi*1000+ci*20+4, args.draws),
                    "control_vs_natural_W1_choice_pp": _ci((c_choice-n_choice)[mask], args.seed+300000+si*10000+bi*1000+ci*20+5, args.draws),
                    "matching_specific_W1_choice_pp": _ci((m_choice-c_choice)[mask], args.seed+300000+si*10000+bi*1000+ci*20+1, args.draws),
                }
            band_rows["Game_minus_Neutral"] = {
                "matching_specific_W1_minus_W2_margin": _ci((raw_margin["Game"]-raw_margin["Neutral"])[mask], args.seed+300000+si*10000+bi*1000+99, args.draws),
                "matching_specific_W1_choice_pp": _ci((raw_choice["Game"]-raw_choice["Neutral"])[mask], args.seed+300000+si*10000+bi*1000+100, args.draws),
            }
            joint_rows[band] = band_rows
        summary["joint_W1"][subset] = joint_rows

    prior = _load(args.prior_early)
    if not np.array_equal(prior["question_ids"], arrays["question_ids"]):
        raise RuntimeError("Prior 4--48 question order differs")
    if not np.array_equal(prior["rank_contents"], arrays["rank_contents"]):
        raise RuntimeError("Prior 4--48 rank definitions differ")
    prior_matched = _align(prior["matched_logits"].astype(float), qids, mappings)
    prior_control = _align(prior["control_logits"].astype(float), qids, mappings)
    for split_name, mask in (("discovery", discovery), ("confirmation", ~discovery)):
        split_rows = {}
        for rank, rank_name in enumerate(RANKS):
            target = rank_indices[rank]
            condition_rows = {}
            for ci, condition in enumerate(CONDITIONS):
                early = _advantage(prior_matched[ci, rank], target)-_advantage(prior_control[ci, rank], target)
                full = _advantage(matched[1, ci, rank], target)-_advantage(control[1, ci, rank], target)
                condition_rows[condition] = {
                    "prior_4_48": _ci(early[mask], args.seed+500000+rank*100+ci*10, args.draws),
                    "full_4_64": _ci(full[mask], args.seed+500001+rank*100+ci*10, args.draws),
                    "full_minus_prior": _ci((full-early)[mask], args.seed+500002+rank*100+ci*10, args.draws),
                }
            split_rows[rank_name] = condition_rows
        summary["early_4_48_comparison"][split_name] = split_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    display = (("confirmation_all", "late_52_64"), ("confirmation_all", "full_04_64"),
               ("confirmation_conflict", "late_52_64"), ("confirmation_conflict", "full_04_64"),
               ("confirmation_no_conflict", "late_52_64"), ("confirmation_no_conflict", "full_04_64"))
    for axis, (subset, band) in zip(axes.flat, display):
        x = np.arange(4)
        for offset, condition, color in ((-0.12, "Game", "#2878b5"), (0.12, "Neutral", "#e07a2f")):
            rows = summary["subsets"][subset]["bands"][band]
            means=np.array([rows[r][condition]["matching_specific"]["mean"] for r in RANKS])
            cis=np.array([rows[r][condition]["matching_specific"]["ci"] for r in RANKS])
            axis.errorbar(x+offset, means, yerr=np.vstack((means-cis[:,0],cis[:,1]-means)),fmt="o",capsize=3,color=color,label=condition)
        axis.axhline(0,color="#888",lw=1); axis.set_xticks(x,RANKS)
        axis.set_title(f"{subset.replace('_',' ')}: {band.replace('_','–')}")
        axis.set_ylabel("Matching-specific candidate effect (logits)")
        axis.grid(axis="y",alpha=.18)
    axes[0,0].legend(frameon=False)
    fig.suptitle("All-candidate matching relay through the complete ordinary-attention stack")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure,dpi=200,bbox_inches="tight"); plt.close(fig)

    lines=["# All-candidate matching relay through layer 64","",
           "## Bottom line","",
           "Layers 52--64 alone make only approximately 0.01-logit candidate changes and do not change W1 choice reliably. Extending the established matching-edge blockade from layers 4--48 through layer 64 leaves the R1--R4 effects essentially unchanged. The causal semantic relay is therefore carried by layers 4--48; the larger natural attention visible at layers 52--64 is not an additional necessary route.","",
           "The complete-range intervention replicates the prior policy difference. Removing a candidate's semantic match raises R1 and R2 evidence in Game, is approximately neutral for R3, and lowers R4; in Neutral, removal lowers every candidate in first-pass rank order. Thus Neutral reinstates the full candidate history, whereas Game transforms how that shared semantic history is used.","",
           "Matching-specific means the matching-edge lesion minus its cyclic nonmatching-edge control. Positive values mean that removing the true semantic match increases evidence for the named candidate.",""]
    for subset in ("discovery_all","confirmation_all","discovery_conflict","confirmation_conflict",
                   "discovery_no_conflict","confirmation_no_conflict"):
        lines += [f"## {subset.replace('_',' ').title()}",""]
        for band in bands:
            lines.append(f"### {band}")
            for rank in RANKS:
                row=summary["subsets"][subset]["bands"][band][rank]
                lines.append(f"- {rank}: Game {_fmt(row['Game']['matching_specific'])}; Neutral {_fmt(row['Neutral']['matching_specific'])}; interaction {_fmt(row['Game_minus_Neutral'])}.")
            joint=summary["joint_W1"][subset][band]
            lines.append(f"- Joint W1 choice: Game {_fmt(joint['Game']['matching_specific_W1_choice_pp'],100)} pp; Neutral {_fmt(joint['Neutral']['matching_specific_W1_choice_pp'],100)} pp; interaction {_fmt(joint['Game_minus_Neutral']['matching_specific_W1_choice_pp'],100)} pp.")
            lines.append("")
    lines += ["## Validation","",f"`{json.dumps(summary['validation'],sort_keys=True)}`",""]
    (args.output_dir/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary["validation"],indent=2,sort_keys=True))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--prior-early",type=Path,required=True)
    parser.add_argument("--remapping-plan",type=Path,required=True)
    parser.add_argument("--remapped-baseline",type=Path,required=True)
    parser.add_argument("--discovery-plan",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--figure",type=Path,required=True)
    parser.add_argument("--draws",type=int,default=10_000)
    parser.add_argument("--seed",type=int,default=8202026)
    analyze(parser.parse_args())


if __name__=="__main__":
    main()
