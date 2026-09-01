from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .all_trial_figures import _style
from .data import load_activation_dataset


Z975 = 1.959963984540054


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = values.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def _family(scores: np.ndarray, layout: list[dict], family: str) -> np.ndarray:
    concepts: dict[str, list[int]] = {}
    for index, row in enumerate(layout):
        if row["family"] == family:
            concepts.setdefault(row["concept"], []).append(index)
    values = [_logsumexp(scores[..., indices]) for indices in concepts.values()]
    return np.stack(values, axis=-1).mean(axis=-1)


def _labels(data, qids: list[str], condition: str) -> np.ndarray:
    result = []
    for qid in qids:
        token = data.metadata[(qid, condition)]["full_vocab_top_token"].strip()
        result.append("ABCD".index(token))
    return np.asarray(result, dtype=int)


def _macro_summary(values: np.ndarray, strata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.stack([values[strata == label].mean(axis=0) for label in range(4)])
    mean = groups.mean(axis=0)
    se = groups.std(axis=0, ddof=1) / np.sqrt(4)
    return mean, Z975 * se


def _residualize(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    result = values.copy()
    for label in range(4):
        mask = strata == label
        result[mask] -= result[mask].mean(axis=0, keepdims=True)
    return result


def _corr_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = (source - source.mean(0)) / source.std(0, ddof=1)
    target = (target - target.mean(0)) / target.std(0, ddof=1)
    return source.T @ target / (len(source) - 1)


def _macro_auc(target: np.ndarray, values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    result = np.empty(values.shape[1])
    for layer in range(values.shape[1]):
        result[layer] = np.mean([
            roc_auc_score(target[strata == label], values[strata == label, layer])
            for label in range(4)
        ])
    return result


def _bootstrap_window(source: np.ndarray, target: np.ndarray, switched: np.ndarray, strata: np.ndarray, seed=42):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    source_window = source[:, 40:48].mean(axis=1)  # displayed readouts 41--48
    target_window = target[:, 43:50].mean(axis=1)  # displayed readouts 44--50
    source_r = _residualize(source_window[:, None], strata)[:, 0]
    target_r = _residualize(target_window[:, None], strata)[:, 0]
    observed_corr = float(np.corrcoef(source_r, target_r)[0, 1])
    observed_auc = float(np.mean([
        roc_auc_score(switched[strata == label], source_window[strata == label]) for label in range(4)
    ]))
    correlations = []
    aucs = []
    for _ in range(5000):
        chosen = np.concatenate([
            rng.choice(np.flatnonzero(strata == label), np.sum(strata == label), replace=True)
            for label in range(4)
        ])
        boot_strata = strata[chosen]
        s = source_window[chosen]
        t = target_window[chosen]
        s_r = _residualize(s[:, None], boot_strata)[:, 0]
        t_r = _residualize(t[:, None], boot_strata)[:, 0]
        correlations.append(np.corrcoef(s_r, t_r)[0, 1])
        letter_aucs = []
        for label in range(4):
            mask = boot_strata == label
            if len(np.unique(switched[chosen][mask])) == 2:
                letter_aucs.append(roc_auc_score(switched[chosen][mask], s[mask]))
        if letter_aucs:
            aucs.append(np.mean(letter_aucs))
    return {
        "source_feedback_exclusion_readouts": [41, 48],
        "target_decision_alternative_readouts": [44, 50],
        "prior_letter_residualized_correlation": observed_corr,
        "correlation_ci_95": [float(x) for x in np.quantile(correlations, (0.025, 0.975))],
        "source_macro_auc_predicting_game_switch": observed_auc,
        "auc_ci_95": [float(x) for x in np.quantile(aucs, (0.025, 0.975))],
    }


def analyze(transmission_root: Path, residual_root: Path, output: Path) -> dict:
    with np.load(transmission_root / "transmission_scores.npz", allow_pickle=False) as saved:
        scores = saved["scores"].astype(np.float64)
        qids = saved["question_ids"].astype(str).tolist()
        positions = saved["positions"].astype(str).tolist()
    layout = json.loads((transmission_root / "token_layout.json").read_text())
    data = load_activation_dataset(residual_root, ["baseline", "incorrect", "neutral"])
    prior = _labels(data, qids, "baseline")
    game = _labels(data, qids, "incorrect")
    switched = game != prior
    feedback = positions.index("feedback_end")
    decision = positions.index("decision")
    exclusion = _family(scores, layout, "exclusion")
    alternative = _family(scores, layout, "alternative")
    source = exclusion[0, :, feedback] - exclusion[1, :, feedback]
    target = alternative[0, :, decision] - alternative[1, :, decision]
    source_r = _residualize(source, prior)
    target_r = _residualize(target, prior)
    correlations = _corr_matrix(source_r, target_r)
    source_auc = _macro_auc(switched, source, prior)
    target_auc = _macro_auc(switched, target, prior)
    primary = _bootstrap_window(source, target, switched, prior)

    trajectory_rows = []
    summaries = {}
    for name, values in {
        "feedback_exclusion_game": exclusion[0, :, feedback],
        "feedback_exclusion_neutral": exclusion[1, :, feedback],
        "feedback_exclusion_game_minus_neutral": source,
        "decision_alternative_game": alternative[0, :, decision],
        "decision_alternative_neutral": alternative[1, :, decision],
        "decision_alternative_game_minus_neutral": target,
    }.items():
        mean, half = _macro_summary(values, prior)
        summaries[name] = (mean, half)
        for layer, value, width in zip(range(1, 65), mean, half):
            trajectory_rows.append({
                "metric": name, "readout": layer, "mean": float(value),
                "ci_low": float(value - width), "ci_high": float(value + width),
            })

    output.mkdir(parents=True, exist_ok=True)
    with (output / "transmission_trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory_rows[0]))
        writer.writeheader(); writer.writerows(trajectory_rows)
    np.savez_compressed(
        output / "transmission_statistics.npz",
        source_feedback_exclusion=source,
        target_decision_alternative=target,
        correlation_matrix=correlations,
        source_switch_auc=source_auc,
        target_switch_auc=target_auc,
        prior=prior,
        switched=switched,
        question_ids=np.asarray(qids),
    )

    _style()
    import matplotlib.pyplot as plt

    layers = np.arange(1, 65)
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.4))
    for key, label, color, style in (
        ("feedback_exclusion_game", "Game", "#0072B2", "-"),
        ("feedback_exclusion_neutral", "Neutral", "#D55E00", "--"),
    ):
        mean, half = summaries[key]
        axes[0, 0].fill_between(layers, mean-half, mean+half, color=color, alpha=.14, linewidth=0)
        axes[0, 0].plot(layers, mean, color=color, linestyle=style, label=label)
    axes[0, 0].set_title("A  Exclusion readout at feedback end", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("JLens-estimated logit family")
    axes[0, 0].legend(frameon=False)
    for key, label, color, style in (
        ("decision_alternative_game", "Game", "#0072B2", "-"),
        ("decision_alternative_neutral", "Neutral", "#D55E00", "--"),
    ):
        mean, half = summaries[key]
        axes[0, 1].fill_between(layers, mean-half, mean+half, color=color, alpha=.14, linewidth=0)
        axes[0, 1].plot(layers, mean, color=color, linestyle=style, label=label)
    axes[0, 1].set_title("B  Alternative readout at decision", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("JLens-estimated logit family")
    axes[0, 1].legend(frameon=False)
    image = axes[1, 0].imshow(correlations.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-.5, vmax=.5, extent=(.5,64.5,.5,64.5))
    axes[1, 0].plot([1,64],[1,64], color="#555555", linewidth=.7)
    axes[1, 0].set_title("C  Trial-level source–target correlation", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Feedback exclusion readout")
    axes[1, 0].set_ylabel("Decision alternative readout")
    fig.colorbar(image, ax=axes[1, 0], fraction=.046, pad=.03, label="Correlation controlling prior letter")
    axes[1, 1].plot(layers, source_auc, label="Feedback exclusion", color="#009E73")
    axes[1, 1].plot(layers, target_auc, label="Decision alternative", color="#7B3294")
    axes[1, 1].axhline(.5, color="#555555", linewidth=.7)
    axes[1, 1].set_title("D  Association with Game switching", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Residual readout")
    axes[1, 1].set_ylabel("Within-letter macro AUC")
    axes[1, 1].legend(frameon=False)
    for axis in (axes[0,0], axes[0,1], axes[1,1]):
        axis.set_xlim(1,64); axis.set_xticks(np.arange(8,65,8)); axis.grid(axis="y", color="#DDDDDD", linewidth=.5)
        axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("Qwen3.6-27B SimpleMC: feedback-to-decision representational bridge", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.96), w_pad=1.4, h_pad=1.7)
    figure_dir = output / "preserved_figures"; figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "jlens_exclusion_transmission.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_questions": len(qids),
        "sample_note": "fixed 128-question sample stratified by baseline letter x Game switch status",
        "primary_window_test": primary,
        "source_peak_mean_layer": int(np.argmax(summaries["feedback_exclusion_game_minus_neutral"][0]) + 1),
        "target_peak_mean_layer": int(np.argmax(summaries["decision_alternative_game_minus_neutral"][0]) + 1),
        "source_peak_switch_auc_layer": int(np.argmax(source_auc) + 1),
        "source_peak_switch_auc": float(np.max(source_auc)),
        "target_peak_switch_auc_layer": int(np.argmax(target_auc) + 1),
        "target_peak_switch_auc": float(np.max(target_auc)),
    }
    (output / "transmission_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transmission-root", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.transmission_root, args.residual_root, args.output), indent=2))


if __name__ == "__main__":
    main()
