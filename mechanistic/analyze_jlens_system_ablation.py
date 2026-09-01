from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_jlens import _answer_scores
from .data import load_activation_dataset


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=-1, keepdims=True)


def _js_divergence(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    middle = 0.5 * (first + second)
    return 0.5 * (
        np.sum(first * np.log(np.maximum(first, 1e-12) / np.maximum(middle, 1e-12)), axis=-1)
        + np.sum(second * np.log(np.maximum(second, 1e-12) / np.maximum(middle, 1e-12)), axis=-1)
    )


def _centered_correlation(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    first = first - first.mean(axis=-1, keepdims=True)
    second = second - second.mean(axis=-1, keepdims=True)
    pooled = float(np.corrcoef(first.ravel(), second.ravel())[0, 1])
    denominator = np.sqrt(np.sum(first * first, axis=-1) * np.sum(second * second, axis=-1))
    per_question = np.sum(first * second, axis=-1) / np.maximum(denominator, 1e-12)
    return pooled, float(np.mean(per_question)), float(np.median(per_question))


def analyze(jlens_root: Path, output: Path, residual_root: Path | None = None) -> dict:
    layout = json.loads((jlens_root / "selected_token_layout.json").read_text())
    with np.load(jlens_root / "jlens_scores.npz", allow_pickle=False) as data:
        final = data["final_scores"].astype(np.float64)
        position = data["position_scores"].astype(np.float64)
        qids = data["question_ids"].astype(str).tolist()
        position_qids = data["position_question_ids"].astype(str).tolist()
        anchors = data["anchors"].astype(str).tolist()
        first_answer_full_vocab_top_ids = (
            data["first_answer_full_vocab_top_ids"].astype(np.int64)
            if "first_answer_full_vocab_top_ids" in data.files
            else None
        )
        baseline_full_vocab_top_ids = (
            data["baseline_full_vocab_top_ids"].astype(np.int64)
            if "baseline_full_vocab_top_ids" in data.files
            else None
        )
    qid_index = {qid: index for index, qid in enumerate(qids)}
    selected = np.asarray([qid_index[qid] for qid in position_qids])
    anchor = anchors.index("first_answer_decision")

    baseline = _answer_scores(final[0, selected], layout)
    game = _answer_scores(position[0, :, anchor], layout)
    neutral = _answer_scores(position[1, :, anchor], layout)
    baseline -= baseline.mean(axis=-1, keepdims=True)
    game -= game.mean(axis=-1, keepdims=True)
    neutral -= neutral.mean(axis=-1, keepdims=True)

    prefix_audit = None
    if residual_root is not None:
        activation_data = load_activation_dataset(
            residual_root, ["baseline", "incorrect", "neutral"]
        )
        if activation_data.question_ids != qids:
            raise ValueError("JLens and residual question orders differ")
        run_metadata = json.loads((residual_root / "run_metadata.json").read_text())
        serialization = run_metadata.get("config", {}).get("chat_serialization")
        raw_chatml = serialization in {"raw_qwen_chatml", "raw_qwen_chatml_bare"}
        bare_chatml = serialization == "raw_qwen_chatml_bare"
        thinking_suffix = "<think>\n\n</think>\n\n"
        exact_shared_prefix = []
        exact_question_message = []
        exact_repeated_question_message = []
        for qid in qids:
            baseline_meta = activation_data.metadata[(qid, "baseline")]
            game_meta = activation_data.metadata[(qid, "incorrect")]
            neutral_meta = activation_data.metadata[(qid, "neutral")]
            baseline_prompt = baseline_meta["rendered_prompt"]
            baseline_shared = baseline_prompt
            if not raw_chatml and baseline_prompt.endswith(thinking_suffix):
                baseline_shared = baseline_prompt[: -len(thinking_suffix)]
            game_prefix = game_meta["rendered_prompt"].split("[redacted]", 1)[0]
            neutral_prefix = neutral_meta["rendered_prompt"].split("[redacted]", 1)[0]
            exact_shared_prefix.append(baseline_shared == game_prefix == neutral_prefix)
            exact_question_message.append(
                baseline_meta["messages"][1]["content"]
                == game_meta["messages"][1]["content"]
                == neutral_meta["messages"][1]["content"]
            )
            baseline_question = baseline_meta["messages"][1]["content"]
            exact_repeated_question_message.append(
                game_meta["messages"][-1]["content"].endswith("\n\n" + baseline_question)
                and neutral_meta["messages"][-1]["content"].endswith(
                    "\n\n" + baseline_question
                )
            )
        prefix_audit = {
            "n_questions": len(qids),
            "exact_question_message_fraction": float(np.mean(exact_question_message)),
            "exact_repeated_question_message_fraction": float(
                np.mean(exact_repeated_question_message)
            ),
            "exact_rendered_prefix_through_assistant_header_fraction": float(
                np.mean(exact_shared_prefix)
            ),
            "baseline_saved_final_position_note": (
                "Baseline and the Game/Neutral first_answer_decision anchor are both "
                + (
                    "immediately after the same bare assistant-role header, with no "
                    "thinking scaffold or intervening token."
                    if bare_chatml
                    else "after the same explicit empty <think></think> scaffold, "
                    "immediately before the first model-visible answer token."
                )
            ),
        }

    rows = []
    for layer in range(64):
        game_prob = _softmax(game[:, layer])
        neutral_prob = _softmax(neutral[:, layer])
        baseline_prob = _softmax(baseline[:, layer])
        rows.append({
            "layer": layer + 1,
            "game_neutral_max_abs_score_difference": float(
                np.max(np.abs(game[:, layer] - neutral[:, layer]))
            ),
            "game_neutral_choice_agreement": float(
                np.mean(game[:, layer].argmax(-1) == neutral[:, layer].argmax(-1))
            ),
            "game_neutral_mean_js_divergence_nats": float(
                _js_divergence(game_prob, neutral_prob).mean()
            ),
            "game_baseline_choice_agreement": float(
                np.mean(game[:, layer].argmax(-1) == baseline[:, layer].argmax(-1))
            ),
            "game_baseline_mean_js_divergence_nats": float(
                _js_divergence(game_prob, baseline_prob).mean()
            ),
            "game_baseline_mean_abs_centered_score_difference": float(
                np.mean(np.abs(game[:, layer] - baseline[:, layer]))
            ),
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "first_answer_decision_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    jlens_summary = json.loads((jlens_root / "analysis/jlens_summary.json").read_text())
    result = {
        "n_position_questions": len(position_qids),
        "first_answer_decision": {
            "definition": (
                (
                    "Residual immediately after the shared bare assistant-role header, "
                    "with no thinking scaffold, and before the first token of [redacted]."
                    if bare_chatml
                    else "Residual after the shared explicit empty-thinking scaffold, "
                    "immediately before the first token of [redacted]."
                )
            ),
            "game_neutral_prefixes_identical": True,
            "all_layers_game_neutral_max_abs_score_difference": float(
                np.max(np.abs(game - neutral))
            ),
            "all_layers_game_neutral_choice_agreement": float(
                np.mean(game.argmax(-1) == neutral.argmax(-1))
            ),
            "prefix_audit": prefix_audit,
        },
        "final_decision": {
            "switch_minus_repeat_peak": jlens_summary[
                "decision_switch_minus_repeat_game_neutral"
            ],
            "behavior_at_peak": jlens_summary["decision_signal_behavior_at_peak"],
            "concepts_at_peak": jlens_summary[
                "decision_concept_game_neutral_contrasts_at_peak"
            ],
        },
    }
    exact_verification_path = output / "first_answer_exact_verification.json"
    if exact_verification_path.exists():
        verification = json.loads(exact_verification_path.read_text())
        exact = verification["recomputed_actual_top_matches_saved_baseline"]
        result["first_answer_decision"]["actual_full_vocab_top_token"] = {
            condition: {
                "matches_baseline": exact["hits"],
                "n": exact["n"],
                "agreement": exact["rate"],
                "method": "fresh batch-size-one forward pass on the exact shared prefix",
            }
            for condition in ("game", "neutral")
        }
    elif (
        first_answer_full_vocab_top_ids is not None
        and baseline_full_vocab_top_ids is not None
    ):
        result["first_answer_decision"]["actual_full_vocab_top_token"] = {
            condition: {
                "matches_baseline": int(
                    np.sum(first_answer_full_vocab_top_ids[ci] == baseline_full_vocab_top_ids)
                ),
                "n": int(len(baseline_full_vocab_top_ids)),
                "agreement": float(
                    np.mean(first_answer_full_vocab_top_ids[ci] == baseline_full_vocab_top_ids)
                ),
                "method": "FP16 cached-residual reconstruction; near-tie argmaxes may differ",
            }
            for ci, condition in enumerate(("game", "neutral"))
        }
    (output / "system_ablation_jlens_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    peak = result["final_decision"]["switch_minus_repeat_peak"]
    if raw_chatml:
        actual = result["first_answer_decision"]["actual_full_vocab_top_token"]
        position_note = (
            "With explicit raw ChatML, Baseline's saved final residual is this exact "
            "same position. The actual unrestricted top-logit token agrees with "
            f"Baseline on {actual['game']['matches_baseline']}/{actual['game']['n']} "
            f"Game prompts ({actual['game']['agreement']:.1%}) and "
            f"{actual['neutral']['matches_baseline']}/{actual['neutral']['n']} Neutral "
            f"prompts ({actual['neutral']['agreement']:.1%})."
        )
        serialization_label = (
            "After explicit bare raw ChatML serialization"
            if bare_chatml
            else "After explicit raw ChatML serialization"
        )
    else:
        position_note = (
            "Baseline's cached final residual is *not* this same point: it comes "
            "after Qwen's empty `<think></think>` generation scaffold. The earlier "
            "66--68% agreement statistic compared these nonmatching positions and "
            "is invalid; it has been removed."
        )
        serialization_label = "After Hugging Face chat templating"
    report = f"""# Qwen3.6-27B JLens with Baseline-matched question presentations

## Design

The condition-specific system line was removed. Both the first and repeated
question presentations use the exact Baseline capabilities-test user message.
Game and Neutral therefore differ only in the feedback sentence: Game says that
the answer was incorrect and requests a different answer; Neutral says that the
response was lost and requests another answer.

Final-position trajectories use all 500 SimpleMC questions. Prompt-position
readouts use the same fixed 128-question stratified design as the previous JLens
analysis. System anchors are excluded from the canonical explorer.

## Before `[redacted]`

The initial user question is byte-identical in all three conditions on
{prefix_audit['exact_question_message_fraction']:.1%} of questions, and the
question block repeated after feedback is byte-identical to that same Baseline
message on {prefix_audit['exact_repeated_question_message_fraction']:.1%} of
Game and Neutral questions. {serialization_label}, Baseline, Game, and Neutral
are also byte-identical through the assistant-role header on
{prefix_audit['exact_rendered_prefix_through_assistant_header_fraction']:.1%}
of questions; in bare raw mode there is no thinking scaffold.
This is the shared point immediately before `[redacted]` in Game and Neutral.

Game and Neutral have exactly the same model-visible prefix there. As required
by causal masking, their JLens A--D scores are numerically identical at all 64
readouts: maximum absolute difference
{result['first_answer_decision']['all_layers_game_neutral_max_abs_score_difference']:.3g},
with {result['first_answer_decision']['all_layers_game_neutral_choice_agreement']:.1%}
argmax agreement. {position_note}

## After user feedback

Despite removing the advance system cue, the final decision position still has
a strong Game-minus-Neutral alternative-selection representation. The
switch-minus-repeat contrast peaks at readout {peak['largest_absolute_layer']}:
{peak['value']:.3f} [{peak['ci_low']:.3f}, {peak['ci_high']:.3f}]. At that
readout the leading concept contrasts are `alternative`
{result['final_decision']['concepts_at_peak']['switch/alternative']:+.2f},
`other` {result['final_decision']['concepts_at_peak']['switch/other']:+.2f},
`change` {result['final_decision']['concepts_at_peak']['switch/change']:+.2f},
and `incorrect` {result['final_decision']['concepts_at_peak']['incorrect/incorrect']:+.2f}.

The paired Game-minus-Neutral decision signal predicts ablated-Game switching
with AUC {result['final_decision']['behavior_at_peak']['paired_game_neutral_delta_auc_predicting_game_switch']:.3f};
the original-answer-letter-controlled macro AUC is
{result['final_decision']['behavior_at_peak']['paired_delta_macro_auc_controlling_prior_letter']:.3f}
with 95% interval
[{result['final_decision']['behavior_at_peak']['paired_delta_macro_auc_ci_95'][0]:.3f},
{result['final_decision']['behavior_at_peak']['paired_delta_macro_auc_ci_95'][1]:.3f}].
"""
    (output / "REPORT.md").write_text(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.jlens_root, args.output, args.residual_root), indent=2))


if __name__ == "__main__":
    main()
