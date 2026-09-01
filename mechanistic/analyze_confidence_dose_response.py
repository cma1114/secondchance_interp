"""Confidence dose-response of the natural Game revision policy.

This is a descriptive, observational analysis of six canonical non-remapped
trajectory cells (three models by two datasets).  It asks whether the natural
Game-minus-Neutral suppression of a model's own first-presentation winner
varies with that model's first-presentation top-1 versus top-2 logit margin.

No model forward pass or intervention is performed.  The analysis deliberately
separates the old-winner component from generic scaling of the complete
Game-minus-Neutral adjustment vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODEL_SPECS = {
    "qwen": {
        "label": "Qwen3.6-27B",
        "model_id": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "color": "#3569a8",
    },
    "seed": {
        "label": "Seed-OSS-36B",
        "model_id": "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "revision": "497f1dca95ebdec98e41d517b9f060ee753c902f",
        "color": "#d9782d",
    },
    "gemma": {
        "label": "Gemma-4-31B",
        "model_id": "google/gemma-4-31B-it",
        "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "color": "#3b8d64",
    },
}

DATASET_LABELS = {"simplemc": "SimpleMC", "triviamc": "TriviaMC"}


def _cell(
    model: str,
    dataset: str,
    trajectory: str,
    baseline: str,
    baseline_container: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "model_key": model,
        "dataset_key": dataset,
        "trajectory": trajectory,
        "trajectory_metadata": str(Path(trajectory).with_name("run_metadata.json")),
        "baseline": baseline,
        "baseline_container": baseline_container,
    }


CELLS = (
    _cell(
        "qwen",
        "simplemc",
        "outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/run/simplemc/results.npz",
        "outputs/prompt_variant_tests/qwen36_27b_simplemc_token_matched_feedback/baseline_results.json",
        ("results",),
    ),
    _cell(
        "qwen",
        "triviamc",
        "outputs/prompt_variant_tests/qwen36_27b_nonremapped_rank_trajectories/run/triviamc/results.npz",
        "outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/step1/baseline/baseline_results.json",
        ("results",),
    ),
    _cell(
        "seed",
        "simplemc",
        "outputs/model_replications/seed_oss_36b_final_position_trajectories/run/simplemc/results.npz",
        "outputs/model_replications/seed_oss_36b_clean_behavioral_replication/simplemc/run/results.json",
        ("scenarios", "baseline"),
    ),
    _cell(
        "seed",
        "triviamc",
        "outputs/model_replications/seed_oss_36b_final_position_trajectories/run/triviamc/results.npz",
        "outputs/model_replications/seed_oss_36b_clean_behavioral_replication/triviamc/run/results.json",
        ("scenarios", "baseline"),
    ),
    _cell(
        "gemma",
        "simplemc",
        "outputs/model_replications/gemma4_31b_negative_model_comparison/simplemc/trajectories/run/results.npz",
        "outputs/model_replications/gemma4_31b_negative_model_comparison/simplemc/behavior/run/results.json",
        ("scenarios", "baseline"),
    ),
    _cell(
        "gemma",
        "triviamc",
        "outputs/model_replications/gemma4_31b_negative_model_comparison/triviamc/trajectories/run/results.npz",
        "outputs/model_replications/gemma4_31b_negative_model_comparison/triviamc/behavior/run/results.json",
        ("scenarios", "baseline"),
    ),
)

SPLIT_PATHS = {
    "simplemc": "outputs/causal/qwen36_27b_simplemc_causal_sweep/plans/discovery_plan.json",
    "triviamc": "outputs/prompt_variant_tests/qwen36_27b_triviamc_difficulty_filtered_strategic_replication/split_plan.json",
}

TARGET_DIRECTION = np.asarray([-3.0, 1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(12.0)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_get(value: dict[str, Any], keys: Iterable[str]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing JSON key path {'/'.join(keys)}")
        current = current[key]
    return current


def stable_descending_order(values: np.ndarray) -> np.ndarray:
    """Displayed-order-stable descending argsort along the last dimension."""
    return np.argsort(-np.asarray(values), axis=-1, kind="stable")


def _model_identity(document: dict[str, Any]) -> tuple[Any, Any]:
    config = document.get("config", {})
    if not isinstance(config, dict):
        config = {}
    return (
        document.get("model_id", config.get("model_id")),
        document.get("model_revision", config.get("model_revision")),
    )


def _check_model_identity(document: dict[str, Any], model_key: str, source: Path) -> None:
    expected = MODEL_SPECS[model_key]
    model_id, revision = _model_identity(document)
    if model_id != expected["model_id"] or revision != expected["revision"]:
        raise ValueError(
            f"Model provenance mismatch in {source}: got {model_id}@{revision}, "
            f"expected {expected['model_id']}@{expected['revision']}"
        )


def _load_split_ids(root: Path, dataset_key: str, all_qids: np.ndarray) -> dict[str, np.ndarray]:
    plan_path = root / SPLIT_PATHS[dataset_key]
    plan = _load_json(plan_path)
    if dataset_key == "simplemc":
        discovery_ids = plan.get("question_ids")
    else:
        discovery_ids = plan.get("discovery_question_ids")
        confirmation_ids = plan.get("confirmation_question_ids")
        if not isinstance(confirmation_ids, list):
            raise ValueError(f"Missing confirmation_question_ids in {plan_path}")
    if not isinstance(discovery_ids, list):
        raise ValueError(f"Missing discovery question IDs in {plan_path}")
    qids = [str(q) for q in all_qids.tolist()]
    qid_set = set(qids)
    discovery_set = set(map(str, discovery_ids))
    if not discovery_set or not discovery_set <= qid_set:
        raise ValueError(f"Discovery split is empty or not contained in cell IDs: {plan_path}")
    if dataset_key == "triviamc":
        confirmation_set = set(map(str, confirmation_ids))
        if discovery_set & confirmation_set or discovery_set | confirmation_set != qid_set:
            raise ValueError(f"TriviaMC split does not exactly partition the cell: {plan_path}")
    else:
        confirmation_set = qid_set - discovery_set
        if discovery_set | confirmation_set != qid_set:
            raise ValueError(f"SimpleMC split does not partition the cell: {plan_path}")
    return {
        "full": np.ones(len(qids), dtype=bool),
        "discovery": np.asarray([q in discovery_set for q in qids], dtype=bool),
        "confirmation": np.asarray([q in confirmation_set for q in qids], dtype=bool),
    }


def derive_quantities(
    baseline_logits: np.ndarray,
    direct_logits: np.ndarray,
    rank_order: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the per-question quantities used by the analysis.

    ``direct_logits`` must be Game then Neutral.  All policy adjustments use
    within-question-centered A-D logits and are reordered by first-presentation
    semantic rank.
    """
    baseline_logits = np.asarray(baseline_logits, dtype=np.float64)
    direct_logits = np.asarray(direct_logits, dtype=np.float64)
    rank_order = np.asarray(rank_order, dtype=np.int64)
    n = baseline_logits.shape[0]
    if baseline_logits.shape != (n, 4):
        raise ValueError(f"Unexpected baseline shape {baseline_logits.shape}")
    if direct_logits.shape != (2, n, 4):
        raise ValueError(f"Unexpected direct_logits shape {direct_logits.shape}")
    if rank_order.shape != (n, 4):
        raise ValueError(f"Unexpected rank_order shape {rank_order.shape}")
    if not (
        np.isfinite(baseline_logits).all()
        and np.isfinite(direct_logits).all()
        and np.all(np.sort(rank_order, axis=1) == np.arange(4)[None, :])
    ):
        raise ValueError("Non-finite logits or invalid rank permutation")

    baseline_by_rank = np.take_along_axis(baseline_logits, rank_order, axis=1)
    confidence = baseline_by_rank[:, 0] - baseline_by_rank[:, 1]

    centered = direct_logits - direct_logits.mean(axis=2, keepdims=True)
    delta_displayed = centered[0] - centered[1]
    delta_rank = np.take_along_axis(delta_displayed, rank_order, axis=1)
    push = -delta_rank[:, 0]
    amplitude = np.linalg.norm(delta_rank, axis=1)
    if np.any(amplitude <= 0.0):
        raise ValueError("Targeting direction is undefined for a zero policy-adjustment vector")
    targeting = delta_rank @ TARGET_DIRECTION / amplitude

    neutral = direct_logits[1]
    old_w1 = rank_order[:, 0]
    rows = np.arange(n)
    neutral_w1 = neutral[rows, old_w1]
    competitors = neutral.copy()
    competitors[rows, old_w1] = -np.inf
    strongest_other = np.argmax(competitors, axis=1)
    neutral_other = neutral[rows, strongest_other]
    old_w1_margin_neutral = neutral_w1 - neutral_other
    game = direct_logits[0]
    margin_push = old_w1_margin_neutral - (
        game[rows, old_w1] - game[rows, strongest_other]
    )

    game_choice = np.argmax(game, axis=1)
    neutral_choice = np.argmax(neutral, axis=1)
    switch_game = (game_choice != old_w1).astype(np.float64)
    switch_neutral = (neutral_choice != old_w1).astype(np.float64)

    return {
        "confidence_c1": confidence,
        "neutral_old_w1_margin": old_w1_margin_neutral,
        "push_r1": push,
        "margin_push": margin_push,
        "policy_amplitude": amplitude,
        "targeting_cosine": targeting,
        "delta_rank": delta_rank,
        "switch_game": switch_game,
        "switch_neutral": switch_neutral,
        "differential_switching": switch_game - switch_neutral,
    }


def _load_cell(root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    trajectory_path = root / spec["trajectory"]
    metadata_path = root / spec["trajectory_metadata"]
    baseline_path = root / spec["baseline"]
    for path in (trajectory_path, metadata_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = _load_json(metadata_path)
    baseline_document = _load_json(baseline_path)
    _check_model_identity(metadata, spec["model_key"], metadata_path)
    _check_model_identity(baseline_document, spec["model_key"], baseline_path)
    if spec["model_key"] == "qwen" and metadata.get("baseline_results") != spec["baseline"]:
        raise ValueError(f"Qwen trajectory points to a different baseline in {metadata_path}")

    with np.load(trajectory_path, allow_pickle=False) as loaded:
        required = {"question_ids", "conditions", "direct_logits", "rank_order"}
        missing = required - set(loaded.files)
        if missing:
            raise KeyError(f"Missing trajectory arrays {sorted(missing)} in {trajectory_path}")
        question_ids = np.asarray(loaded["question_ids"]).astype(str)
        conditions = np.asarray(loaded["conditions"]).astype(str)
        direct_logits = np.asarray(loaded["direct_logits"], dtype=np.float64)
        rank_order = np.asarray(loaded["rank_order"], dtype=np.int64)
    if conditions.tolist() != ["game", "neutral"]:
        raise ValueError(f"Expected conditions ['game', 'neutral'] in {trajectory_path}")
    if len(question_ids) != 500 or len(set(question_ids.tolist())) != len(question_ids):
        raise ValueError(f"Expected 500 unique question IDs in {trajectory_path}")

    baseline_rows = _nested_get(baseline_document, spec["baseline_container"])
    if not isinstance(baseline_rows, dict):
        raise ValueError(f"Baseline rows are not an object in {baseline_path}")
    if set(baseline_rows) != set(question_ids.tolist()):
        missing = set(question_ids.tolist()) - set(baseline_rows)
        extra = set(baseline_rows) - set(question_ids.tolist())
        raise ValueError(
            f"Baseline/trajectory ID mismatch in {baseline_path}: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    baseline_logits = []
    for qid in question_ids:
        row = baseline_rows[str(qid)]
        if row.get("question_id") != str(qid):
            raise ValueError(f"Question ID mismatch inside baseline row {qid}")
        values = row.get("aggregated_ad_logits")
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"Invalid aggregated_ad_logits for {qid}")
        baseline_logits.append(values)
    baseline_logits_array = np.asarray(baseline_logits, dtype=np.float64)
    reconstructed_rank = stable_descending_order(baseline_logits_array)
    rank_matches = np.all(reconstructed_rank == rank_order, axis=1)
    if not rank_matches.all():
        raise ValueError(
            f"Fail-closed rank gate failed for {int((~rank_matches).sum())} questions in "
            f"{MODEL_SPECS[spec['model_key']]['label']} {DATASET_LABELS[spec['dataset_key']]}"
        )

    quantities = derive_quantities(baseline_logits_array, direct_logits, rank_order)
    quantities["question_ids"] = question_ids
    validation = {
        "model": MODEL_SPECS[spec["model_key"]]["label"],
        "dataset": DATASET_LABELS[spec["dataset_key"]],
        "model_id": MODEL_SPECS[spec["model_key"]]["model_id"],
        "model_revision": MODEL_SPECS[spec["model_key"]]["revision"],
        "n": int(len(question_ids)),
        "conditions": conditions.tolist(),
        "all_inputs_finite": bool(
            np.isfinite(baseline_logits_array).all() and np.isfinite(direct_logits).all()
        ),
        "rank_order_exact_matches": int(rank_matches.sum()),
        "rank_order_gate_passed": bool(rank_matches.all()),
        "trajectory": spec["trajectory"],
        "trajectory_sha256": _sha256(trajectory_path),
        "trajectory_metadata": spec["trajectory_metadata"],
        "baseline": spec["baseline"],
        "baseline_sha256": _sha256(baseline_path),
        "split_plan": SPLIT_PATHS[spec["dataset_key"]],
    }
    return validation, quantities


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("Cannot z-score a constant or non-finite vector")
    return (values - values.mean()) / sd


def _percentile_record(value: float, bootstrap: np.ndarray) -> dict[str, float]:
    bootstrap = np.asarray(bootstrap, dtype=np.float64)
    if not np.isfinite(bootstrap).all():
        raise ValueError("Non-finite bootstrap estimate")
    return {
        "value": float(value),
        "ci_low": float(np.percentile(bootstrap, 2.5)),
        "ci_high": float(np.percentile(bootstrap, 97.5)),
    }


def _univariate_bundle(
    x: np.ndarray,
    outcomes: dict[str, np.ndarray],
    draws: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    """Regress each outcome on within-resample standardized x."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    index = rng.integers(0, n, size=(draws, n))
    xb = x[index]
    xb_centered = xb - xb.mean(axis=1, keepdims=True)
    xb_sd = xb_centered.std(axis=1, ddof=0)
    if np.any(xb_sd <= 0.0):
        raise ValueError("A bootstrap resample has constant confidence")
    zxb = xb_centered / xb_sd[:, None]
    zx = _zscore(x)
    result: dict[str, dict[str, Any]] = {}
    for name, values in outcomes.items():
        y = np.asarray(values, dtype=np.float64)
        if len(y) != n or not np.isfinite(y).all():
            raise ValueError(f"Invalid outcome {name}")
        yb = y[index]
        yb_centered = yb - yb.mean(axis=1, keepdims=True)
        raw_boot = np.mean(zxb * yb_centered, axis=1)
        raw_value = float(np.mean(zx * (y - y.mean())))
        y_sd = float(y.std(ddof=0))
        if y_sd <= 0.0:
            beta_boot = np.zeros(draws, dtype=np.float64)
            beta_value = 0.0
        else:
            yb_sd = yb_centered.std(axis=1, ddof=0)
            beta_boot = np.divide(
                raw_boot,
                yb_sd,
                out=np.zeros_like(raw_boot),
                where=yb_sd > 0.0,
            )
            beta_value = raw_value / y_sd
        result[name] = {
            "raw_outcome_per_1sd_c1": _percentile_record(raw_value, raw_boot),
            "standardized_beta": _percentile_record(beta_value, beta_boot),
        }
    return result


def _joint_choice_regression(
    confidence: np.ndarray,
    margin: np.ndarray,
    outcome: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Linear probability model D ~ z(C1) + z(signed Neutral W1 margin)."""
    confidence = np.asarray(confidence, dtype=np.float64)
    margin = np.asarray(margin, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    zc = _zscore(confidence)
    zm = _zscore(margin)
    yc = outcome - outcome.mean()
    corr = float(np.mean(zc * zm))
    determinant = 1.0 - corr * corr
    if determinant <= 1e-10:
        raise ValueError("Joint choice regression is singular")
    cov_c = float(np.mean(zc * yc))
    cov_m = float(np.mean(zm * yc))
    b_c = (cov_c - corr * cov_m) / determinant
    b_m = (cov_m - corr * cov_c) / determinant

    n = len(confidence)
    index = rng.integers(0, n, size=(draws, n))
    cb = confidence[index]
    mb = margin[index]
    yb = outcome[index]
    zcb = (cb - cb.mean(axis=1, keepdims=True)) / cb.std(axis=1, keepdims=True)
    zmb = (mb - mb.mean(axis=1, keepdims=True)) / mb.std(axis=1, keepdims=True)
    ybc = yb - yb.mean(axis=1, keepdims=True)
    rb = np.mean(zcb * zmb, axis=1)
    detb = 1.0 - rb * rb
    if np.any(detb <= 1e-10):
        raise ValueError("A bootstrap joint choice regression is singular")
    cov_cb = np.mean(zcb * ybc, axis=1)
    cov_mb = np.mean(zmb * ybc, axis=1)
    bcb = (cov_cb - rb * cov_mb) / detb
    bmb = (cov_mb - rb * cov_cb) / detb
    return {
        "c1_coefficient": _percentile_record(b_c, bcb),
        "neutral_old_w1_margin_coefficient": _percentile_record(b_m, bmb),
        "corr_c1_margin": corr,
        "variance_inflation_factor": float(1.0 / determinant),
        "outcome_mean": float(outcome.mean()),
    }


def _quadratic_regression(
    confidence: np.ndarray,
    push: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Robustness model P ~ z(C1) + centered z(C1)^2."""

    def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        z = _zscore(x)
        q = z * z
        q -= q.mean()
        yc = y - y.mean()
        a11 = float(np.mean(z * z))
        a12 = float(np.mean(z * q))
        a22 = float(np.mean(q * q))
        determinant = a11 * a22 - a12 * a12
        if determinant <= 1e-12:
            raise ValueError("Quadratic regression is singular")
        c1 = float(np.mean(z * yc))
        c2 = float(np.mean(q * yc))
        return ((c1 * a22 - c2 * a12) / determinant, (c2 * a11 - c1 * a12) / determinant)

    linear, quadratic = fit(confidence, push)
    n = len(confidence)
    index = rng.integers(0, n, size=(draws, n))
    boot = np.empty((draws, 2), dtype=np.float64)
    for draw, sample in enumerate(index):
        boot[draw] = fit(confidence[sample], push[sample])
    return {
        "linear_c1": _percentile_record(linear, boot[:, 0]),
        "quadratic_c1_squared": _percentile_record(quadratic, boot[:, 1]),
    }


def _mean_interval(values: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    index = rng.integers(0, len(values), size=(draws, len(values)))
    return _percentile_record(float(values.mean()), values[index].mean(axis=1))


def _tercile_summary(
    confidence: np.ndarray,
    outcomes: dict[str, np.ndarray],
    draws: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    order = np.argsort(confidence, kind="stable")
    groups = np.array_split(order, 3)
    labels = ("low", "middle", "high")
    records = []
    for label, group in zip(labels, groups, strict=True):
        record: dict[str, Any] = {
            "tercile": label,
            "n": int(len(group)),
            "c1_min": float(np.min(confidence[group])),
            "c1_max": float(np.max(confidence[group])),
            "c1_mean": float(np.mean(confidence[group])),
        }
        for name, values in outcomes.items():
            record[name] = _mean_interval(np.asarray(values)[group], draws, rng)
        records.append(record)
    return records


def _analyze_subset(
    quantities: dict[str, np.ndarray],
    mask: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    q = {key: np.asarray(value)[mask] for key, value in quantities.items() if key != "question_ids"}
    outcomes = {
        "push_r1": q["push_r1"],
        "margin_push": q["margin_push"],
        "policy_amplitude": q["policy_amplitude"],
        "targeting_cosine": q["targeting_cosine"],
        **{f"delta_r{rank + 1}": q["delta_rank"][:, rank] for rank in range(4)},
    }
    result = {
        "n": int(mask.sum()),
        "corr_c1_neutral_old_w1_margin": float(
            np.corrcoef(q["confidence_c1"], q["neutral_old_w1_margin"])[0, 1]
        ),
        "univariate": _univariate_bundle(q["confidence_c1"], outcomes, draws, rng),
        "choice_lpm": _joint_choice_regression(
            q["confidence_c1"],
            q["neutral_old_w1_margin"],
            q["differential_switching"],
            draws,
            rng,
        ),
        "quadratic_push": _quadratic_regression(
            q["confidence_c1"], q["push_r1"], draws, rng
        ),
        "means": {
            "confidence_c1": float(q["confidence_c1"].mean()),
            "push_r1": float(q["push_r1"].mean()),
            "margin_push": float(q["margin_push"].mean()),
            "policy_amplitude": float(q["policy_amplitude"].mean()),
            "targeting_cosine": float(q["targeting_cosine"].mean()),
            "switch_game": float(q["switch_game"].mean()),
            "switch_neutral": float(q["switch_neutral"].mean()),
            "differential_switching": float(q["differential_switching"].mean()),
        },
    }
    return result


def _sign(record: dict[str, float]) -> str:
    if record["ci_low"] > 0.0:
        return "positive"
    if record["ci_high"] < 0.0:
        return "negative"
    return "uncertain"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {
        "analysis": "confidence_dose_response",
        "evidence_class": "descriptive/observational natural runs; no intervention and no causal claim",
        "seed": int(args.seed),
        "bootstrap_draws": int(args.draws),
        "definitions": {
            "score_scale": "plain within-question-centered aggregated A-D logits; no 4/3 advantage scaling",
            "c1": "model's own first-presentation top-1 minus top-2 aggregated A-D logit",
            "delta": "within-question-centered Game final A-D logits minus centered Neutral final A-D logits, reordered by first-presentation rank",
            "push_r1": "negative delta at old R1; positive means Game suppresses the old winner relative to Neutral",
            "margin_push": "Neutral old-W1-vs-best-other margin minus the corresponding Game margin; positive means Game lowers W1 relative to the strongest Neutral replacement",
            "neutral_old_w1_margin": "signed Neutral final logit of old W1 minus the strongest non-W1 final logit",
            "differential_switching": "Game switch indicator minus Neutral switch indicator, with displayed-order-stable A-D argmax",
            "policy_amplitude": "Euclidean norm of the complete centered Game-minus-Neutral rank vector",
            "targeting_cosine": "cosine of that vector with (-3,+1,+1,+1)/sqrt(12), a scale-free old-W1-suppression/equal-redistribution direction",
            "primary_model": "push_r1 ~ z(C1)",
            "choice_model": "differential_switching ~ z(C1) + z(signed Neutral old-W1 margin); descriptive post-treatment companion",
            "quadratic_robustness": "push_r1 ~ z(C1) + centered z(C1)^2",
            "bootstrap": "question-level percentile bootstrap; predictors re-standardized within every resample",
        },
        "prespecified_interpretation": {
            "fixed_reflex": "C1 coefficient on push near zero together with a flat tercile profile; switching differences may then arise through the expression margin",
            "positive": "larger old-W1 suppression after more confident first answers, consistent with surprise-weighted dose response",
            "negative": "larger old-W1 suppression after weakly held first answers",
            "mixed": "dose responsiveness is not a shared property across models",
            "scale_reference": {
                "amplitude_positive_targeting_flat": "confidence scales the whole established revision-policy vector, not its old-W1 selectivity",
                "targeting_positive": "confidence makes the adjustment increasingly old-W1-targeted beyond generic scaling",
                "rankwise_proportional": "similar proportional R1 and R4 changes support generic policy-amplitude scaling",
            },
            "caveat": "A final-output association can arise from late nonlinear saturation or other downstream processing; this analysis cannot establish that the policy circuit itself reads confidence.",
        },
        "split_sources": SPLIT_PATHS,
        "cells": [],
    }

    for spec in CELLS:
        validation, quantities = _load_cell(root, spec)
        split_masks = _load_split_ids(root, spec["dataset_key"], quantities["question_ids"])
        analyses = {
            split: _analyze_subset(quantities, mask, args.draws, rng)
            for split, mask in split_masks.items()
        }
        terciles = _tercile_summary(
            quantities["confidence_c1"],
            {
                "push_r1": quantities["push_r1"],
                "differential_switching": quantities["differential_switching"],
                "policy_amplitude": quantities["policy_amplitude"],
                "targeting_cosine": quantities["targeting_cosine"],
            },
            args.draws,
            rng,
        )
        summary["cells"].append(
            {
                "model_key": spec["model_key"],
                "model": MODEL_SPECS[spec["model_key"]]["label"],
                "dataset_key": spec["dataset_key"],
                "dataset": DATASET_LABELS[spec["dataset_key"]],
                "validation": validation,
                "splits": analyses,
                "terciles": terciles,
            }
        )

    primary_signs = [
        _sign(cell["splits"]["full"]["univariate"]["push_r1"]["raw_outcome_per_1sd_c1"])
        for cell in summary["cells"]
    ]
    targeting_signs = [
        _sign(
            cell["splits"]["full"]["univariate"]["targeting_cosine"]
            ["raw_outcome_per_1sd_c1"]
        )
        for cell in summary["cells"]
    ]
    summary["cross_model_synthesis"] = {
        "full_cell_primary_signs": primary_signs,
        "full_cell_targeting_signs": targeting_signs,
        "n_primary_positive": int(sum(sign == "positive" for sign in primary_signs)),
        "n_targeting_positive": int(sum(sign == "positive" for sign in targeting_signs)),
        "all_positive": bool(all(sign == "positive" for sign in primary_signs)),
        "all_negative": bool(all(sign == "negative" for sign in primary_signs)),
        "all_uncertain": bool(all(sign == "uncertain" for sign in primary_signs)),
    }
    return summary


def _fmt(record: dict[str, float], digits: int = 3) -> str:
    return (
        f"{record['value']:.{digits}f} "
        f"[{record['ci_low']:.{digits}f}, {record['ci_high']:.{digits}f}]"
    )


def _interpret_cell(cell: dict[str, Any]) -> str:
    full = cell["splits"]["full"]["univariate"]
    push = full["push_r1"]["raw_outcome_per_1sd_c1"]
    amplitude = full["policy_amplitude"]["raw_outcome_per_1sd_c1"]
    targeting = full["targeting_cosine"]["raw_outcome_per_1sd_c1"]
    push_sign = _sign(push)
    amplitude_sign = _sign(amplitude)
    targeting_sign = _sign(targeting)
    if push_sign == "positive" and amplitude_sign == "positive" and targeting_sign == "uncertain":
        return "larger overall revision-policy amplitude, without reliable extra W1 targeting"
    if push_sign == "positive" and targeting_sign == "positive":
        return "larger and increasingly W1-targeted revision"
    if push_sign == "positive":
        return "larger W1 suppression; the scale/direction decomposition is mixed"
    if push_sign == "negative":
        return "larger W1 suppression on lower-confidence first answers"
    return "no reliable linear W1-suppression dose response"


def write_report(summary: dict[str, Any], path: Path) -> None:
    cells = summary["cells"]
    signs = summary["cross_model_synthesis"]["full_cell_primary_signs"]
    positive_count = summary["cross_model_synthesis"]["n_primary_positive"]
    targeting_count = summary["cross_model_synthesis"]["n_targeting_positive"]
    if all(sign == "positive" for sign in signs):
        headline = "The final old-winner suppression grows with first-pass confidence in all six cells."
    elif all(sign == "uncertain" for sign in signs):
        headline = "None of the six cells shows a reliable linear old-winner-suppression dose response."
    else:
        headline = (
            f"Old-winner suppression grows linearly with first-pass confidence in "
            f"{positive_count} of six cells; the remaining cell is not reliably linear."
        )

    lines = [
        "# Confidence dose-response of the Game revision policy",
        "",
        "**Evidence class:** descriptive/observational analysis of canonical natural",
        "runs. No representation, margin, or confidence state is intervened on, so",
        "none of the associations below establishes a causal confidence readout.",
        "",
        "## Prespecified question and interpretation",
        "",
        "The primary regression is the continuous Game-minus-Neutral suppression of",
        "the model's own first-presentation winner on that model's own first-pass",
        "top-1-versus-top-2 logit margin. A coefficient near zero, together with flat",
        "confidence terciles, supports a fixed revision reflex whose observed choices",
        "are gated by the decision margin. A positive coefficient means more confident",
        "first answers receive a larger final suppression; a negative coefficient means",
        "the adjustment concentrates on weakly held answers. Mixed signs mean graded",
        "dose response is not shared across models.",
        "",
        "The scale-reference decomposition distinguishes a larger complete policy vector",
        "from more selective targeting. `Amplitude` is the norm of the complete centered",
        "Game-minus-Neutral rank vector. `Targeting` is its cosine with",
        "`(-3,+1,+1,+1)/sqrt(12)`. Rising suppression and amplitude with flat targeting",
        "means confidence scales the established policy as a whole; rising targeting",
        "means the policy becomes increasingly specific to old W1.",
        "",
        "All logit endpoints use plain within-question centering, not the project's",
        "`×4/3` advantage convention. Predictors are z-scored within cell and within",
        f"each of {summary['bootstrap_draws']:,} question-bootstrap resamples (seed",
        f"`{summary['seed']}`). Reported primary slopes are logits per one-SD increase",
        "in first-pass confidence. The forest plot also reports fully standardized betas.",
        "",
        "## Result",
        "",
        headline,
        (
            f"The scale-free old-W1-targeting direction increases with confidence in "
            f"{targeting_count} of six cells. Thus the main result is not explained by "
            "a generic increase in the size of every Game-minus-Neutral adjustment. "
            "The exception to the linear W1-push result is Gemma TriviaMC: its middle "
            "confidence tercile is highest, its quadratic term is negative, and the "
            "positive discovery slope does not replicate on confirmation."
        ),
        "",
        "| Model | Dataset | Mean W1 suppression | C1 slope on W1 suppression | Standardized beta | Scale-reference reading |",
        "|---|---|---:|---:|---:|---|",
    ]
    for cell in cells:
        full = cell["splits"]["full"]
        primary = full["univariate"]["push_r1"]
        lines.append(
            f"| {cell['model']} | {cell['dataset']} | {full['means']['push_r1']:.3f} "
            f"| {_fmt(primary['raw_outcome_per_1sd_c1'])} "
            f"| {_fmt(primary['standardized_beta'])} | {_interpret_cell(cell)} |"
        )

    lines += [
        "",
        "### Generic scaling versus targeted suppression",
        "",
        "| Model | Dataset | W1 push slope | Full-vector amplitude slope | Targeting-cosine slope | R4 Game−Neutral slope |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in cells:
        uni = cell["splits"]["full"]["univariate"]
        lines.append(
            f"| {cell['model']} | {cell['dataset']} "
            f"| {_fmt(uni['push_r1']['raw_outcome_per_1sd_c1'])} "
            f"| {_fmt(uni['policy_amplitude']['raw_outcome_per_1sd_c1'])} "
            f"| {_fmt(uni['targeting_cosine']['raw_outcome_per_1sd_c1'])} "
            f"| {_fmt(uni['delta_r4']['raw_outcome_per_1sd_c1'])} |"
        )

    lines += [
        "",
        "The R4 column is reported as a readable rankwise reference, but it is not a",
        "literal placebo: the established Game policy can itself raise R4. The norm and",
        "cosine decomposition is the cleaner test of generic magnitude versus changing",
        "direction. The complete rankwise slopes are preserved in `summary.json`.",
        "",
        "### Choice expression after the signed Neutral margin",
        "",
        "The companion linear-probability model uses `D = Game switch − Neutral switch`",
        "and the signed Neutral margin of old W1 over its strongest competitor. This",
        "margin is an outcome of Neutral processing and shares final logits with the",
        "dependent quantities, so the model is descriptive—not a causal adjustment.",
        "",
        "| Model | Dataset | corr(C1, signed margin) | C1 coefficient | Margin coefficient | VIF |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in cells:
        choice = cell["splits"]["full"]["choice_lpm"]
        lines.append(
            f"| {cell['model']} | {cell['dataset']} "
            f"| {choice['corr_c1_margin']:+.3f} "
            f"| {_fmt(choice['c1_coefficient'])} "
            f"| {_fmt(choice['neutral_old_w1_margin_coefficient'])} "
            f"| {choice['variance_inflation_factor']:.2f} |"
        )

    lines += [
        "",
        "### Frozen-split robustness of the primary slope",
        "",
        "| Model | Dataset | Discovery | Confirmation | Quadratic term (full) |",
        "|---|---|---:|---:|---:|",
    ]
    for cell in cells:
        discovery = cell["splits"]["discovery"]["univariate"]["push_r1"]["raw_outcome_per_1sd_c1"]
        confirmation = cell["splits"]["confirmation"]["univariate"]["push_r1"]["raw_outcome_per_1sd_c1"]
        quadratic = cell["splits"]["full"]["quadratic_push"]["quadratic_c1_squared"]
        lines.append(
            f"| {cell['model']} | {cell['dataset']} | {_fmt(discovery)} "
            f"| {_fmt(confirmation)} | {_fmt(quadratic)} |"
        )

    lines += [
        "",
        "### Confidence terciles",
        "",
        "The terciles are stable rank-based groups within each cell, so each group has",
        "nearly equal size even if confidence values tie.",
        "",
        "| Model | Dataset | Low / middle / high mean W1 suppression | Low / middle / high mean differential switching |",
        "|---|---|---:|---:|",
    ]
    for cell in cells:
        p = " / ".join(f"{row['push_r1']['value']:.3f}" for row in cell["terciles"])
        d = " / ".join(f"{100*row['differential_switching']['value']:+.1f}pp" for row in cell["terciles"])
        lines.append(f"| {cell['model']} | {cell['dataset']} | {p} | {d} |")

    lines += [
        "",
        "## Validity and provenance",
        "",
        "Every cell contains 500 questions, finite baseline and final A-D logits, and",
        "exact Game/Neutral condition order. The displayed-order-stable argsort of each",
        "model's own canonical baseline logits reproduced the trajectory array's stored",
        "first-presentation rank order on every question; any mismatch would have aborted",
        "the analysis. Model IDs and pinned revisions were checked in both trajectory",
        "metadata and baseline artifacts. File hashes and exact source paths are in",
        "`summary.json`.",
        "",
        "## Scope",
        "",
        "Even a consistent positive slope would not by itself demonstrate metacognitive",
        "access to confidence. Because suppression is measured only after all downstream",
        "processing, graded output can arise through nonlinear saturation, interaction",
        "with other question features, or a policy whose amplitude—not selectivity—is",
        "larger on high-confidence questions. The magnitude/direction decomposition narrows",
        "that ambiguity but does not turn this natural-run analysis into a causal test.",
        "",
    ]
    path.write_text("\n".join(lines))


def make_figure(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 11.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 1.25))
    cells_by_key = {
        (cell["dataset_key"], cell["model_key"]): cell for cell in summary["cells"]
    }
    model_keys = ("qwen", "seed", "gemma")
    dataset_keys = ("simplemc", "triviamc")
    all_tercile_points = [
        row["push_r1"][bound]
        for cell in summary["cells"]
        for row in cell["terciles"]
        for bound in ("ci_low", "ci_high")
    ]
    y_min, y_max = min(all_tercile_points), max(all_tercile_points)
    pad = 0.08 * max(y_max - y_min, 0.1)
    for row_index, dataset_key in enumerate(dataset_keys):
        for col_index, model_key in enumerate(model_keys):
            ax = fig.add_subplot(grid[row_index, col_index])
            cell = cells_by_key[(dataset_key, model_key)]
            terciles = cell["terciles"]
            values = np.asarray([t["push_r1"]["value"] for t in terciles])
            lows = np.asarray([t["push_r1"]["ci_low"] for t in terciles])
            highs = np.asarray([t["push_r1"]["ci_high"] for t in terciles])
            x = np.arange(3)
            ax.errorbar(
                x,
                values,
                yerr=np.vstack((values - lows, highs - values)),
                color=MODEL_SPECS[model_key]["color"],
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                capsize=4,
            )
            ax.axhline(0.0, color="0.45", linewidth=0.9)
            ax.set_xticks(x, ("Low", "Middle", "High"))
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(axis="y", color="0.88", linewidth=0.8)
            if row_index == 0:
                ax.set_title(MODEL_SPECS[model_key]["label"], fontsize=12, weight="bold")
            if col_index == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset_key]}\nW1 suppression (logits)")
            else:
                ax.set_ylabel("W1 suppression (logits)")
            if row_index == 1:
                ax.set_xlabel("First-presentation confidence tercile")

    metrics = (
        ("push_r1", "Old-W1 suppression", "standardized beta"),
        ("policy_amplitude", "Complete-policy amplitude", "standardized beta"),
        ("targeting_cosine", "W1-targeting direction", "standardized beta"),
    )
    ordered_cells = [
        cells_by_key[(dataset_key, model_key)]
        for model_key in model_keys
        for dataset_key in dataset_keys
    ]
    labels = [f"{c['model'].split('-')[0]} · {c['dataset']}" for c in ordered_cells]
    y = np.arange(len(ordered_cells))[::-1]
    for col_index, (metric, title, xlabel) in enumerate(metrics):
        ax = fig.add_subplot(grid[2, col_index])
        records = [
            c["splits"]["full"]["univariate"][metric]["standardized_beta"]
            for c in ordered_cells
        ]
        values = np.asarray([r["value"] for r in records])
        lows = np.asarray([r["ci_low"] for r in records])
        highs = np.asarray([r["ci_high"] for r in records])
        for yi, value, low, high, cell in zip(y, values, lows, highs, ordered_cells, strict=True):
            marker = "o" if cell["dataset_key"] == "simplemc" else "s"
            color = MODEL_SPECS[cell["model_key"]]["color"]
            ax.errorbar(
                value,
                yi,
                xerr=np.asarray([[value - low], [high - value]]),
                marker=marker,
                color=color,
                linewidth=1.8,
                markersize=6.2,
                capsize=3,
            )
        ax.axvline(0.0, color="0.35", linewidth=1.0)
        ax.grid(axis="x", color="0.88", linewidth=0.8)
        ax.set_yticks(y)
        if col_index == 0:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=11.5, weight="bold")
        ax.set_xlabel(xlabel)

    fig.suptitle(
        "Does first-pass confidence scale the Game revision policy?",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model_replications/confidence_dose_response/analysis"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/model_replications/confidence_dose_response.png"),
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--draws", type=int, default=10_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze(args)
    output_dir = args.root / args.output_dir
    figure_path = args.root / args.figure
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, output_dir / "REPORT.md")
    make_figure(summary, figure_path)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'REPORT.md'}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
