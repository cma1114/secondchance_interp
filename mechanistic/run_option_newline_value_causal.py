from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .collect_contextual_option_representations import ANCHORS, _positions
from .config import ExperimentConfig
from .io import atomic_save_npz
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import prompt_hash
from .run_first_decision_cross_order_patching import _aggregate_logits
from .run_semantic_binding_module_factorial import (
    _forward,
    _hidden,
    _messages,
    _remap_question,
    _replace_hidden,
)


LETTERS = "ABCD"
CONDITIONS = ("incorrect_again", "lost_again")
CONDITION_NAMES = ("game", "neutral")
MODES = ("natural", "chosen_sham", "devalue", "opposite")
TARGET_READOUTS = tuple(range(33, 57))


def _target_scores(
    residual_cache: Path,
    cache_results: Path,
    probe_results: Path,
    eligible_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    residuals = np.load(residual_cache, mmap_mode="r")
    with np.load(cache_results, allow_pickle=False) as loaded:
        qids = loaded["question_ids"].astype(str).tolist()
        completed = loaded["completed"].astype(bool)
    if not completed.all():
        raise ValueError("Option-newline residual cache is incomplete")
    qindex = {qid: index for index, qid in enumerate(qids)}
    with np.load(probe_results, allow_pickle=False) as loaded:
        weights = loaded["weights"].astype(np.float32)
        means = loaded["letter_means"].astype(np.float32)
        scales = loaded["scales"].astype(np.float32)
    if residuals.shape[2:] != (64, 4, weights.shape[1]):
        raise ValueError("Residual cache and probe shapes disagree")

    chosen = np.empty((len(eligible_rows), 64), dtype=np.float32)
    unchosen = np.empty_like(chosen)
    for index, row in enumerate(eligible_rows):
        qid = row["question_id"]
        qi = qindex[qid]
        letter = LETTERS.index(row["w1_displayed_letter"])
        chosen_mapping = int(row["chosen_mapping_index"])
        unchosen_mapping = int(row["unchosen_mapping_index"])
        if chosen_mapping != 0:
            raise ValueError(f"{qid}: chosen mapping is not identity")
        for destination, mapping in (
            (chosen, chosen_mapping),
            (unchosen, unchosen_mapping),
        ):
            values = np.asarray(
                residuals[mapping, qi, :, letter, :], dtype=np.float32
            )
            destination[index] = np.einsum(
                "ld,ld->l",
                (values - means[:, letter]) / scales,
                weights,
                optimize=True,
            )
    return chosen, unchosen, qindex


def _initialize(
    path: Path,
    rows: list[dict[str, Any]],
    chosen_targets: np.ndarray,
    unchosen_targets: np.ndarray,
) -> dict[str, np.ndarray]:
    qids = [row["question_id"] for row in rows]
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses another question order")
        return arrays
    n = len(rows)
    shape = (len(CONDITIONS), len(MODES), n)
    layer_shape = shape + (64,)
    return {
        "question_ids": np.asarray(qids),
        "split": np.asarray([row["split"] for row in rows]),
        "w1_original": np.asarray([row["w1_original_content"] for row in rows]),
        "w1_displayed_letter": np.asarray(
            [row["w1_displayed_letter"] for row in rows]
        ),
        "unchosen_mapping_index": np.asarray(
            [int(row["unchosen_mapping_index"]) for row in rows], dtype=np.int64
        ),
        "chosen_target_score": chosen_targets.astype(np.float32),
        "unchosen_target_score": unchosen_targets.astype(np.float32),
        "completed": np.zeros(n, dtype=bool),
        "logits": np.full(shape + (4,), np.nan, dtype=np.float32),
        "pre_score": np.full(layer_shape, np.nan, dtype=np.float32),
        "post_score": np.full(layer_shape, np.nan, dtype=np.float32),
        "residual_norm": np.full(layer_shape, np.nan, dtype=np.float32),
        "dose_l2": np.full(layer_shape, np.nan, dtype=np.float32),
        "trusted_max_abs_error": np.full(
            (len(CONDITIONS), n), np.nan, dtype=np.float32
        ),
        "trusted_choice_match": np.zeros((len(CONDITIONS), n), dtype=bool),
    }


class OptionValueClamp:
    """Clamp one probe coordinate at each selected row's W1 option newline."""

    def __init__(
        self,
        parts: Any,
        positions: list[int],
        letters: list[int],
        selected_rows: list[int],
        weights: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
        chosen_targets: np.ndarray,
        unchosen_targets: np.ndarray,
        mode: str,
    ) -> None:
        import torch

        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.selected_rows = tuple(int(row) for row in selected_rows)
        self.positions = torch.as_tensor(positions, dtype=torch.long)
        self.letters = torch.as_tensor(letters, dtype=torch.long)
        device = model_input_device(parts)
        self.weights = torch.from_numpy(weights).float().to(device)
        self.means = torch.from_numpy(means).float().to(device)
        self.scales = torch.from_numpy(scales).float().to(device)
        self.gradient = self.weights / self.scales
        self.gradient_norm_sq = self.gradient.square().sum(dim=-1)
        self.chosen_targets = torch.from_numpy(chosen_targets).float().to(device)
        self.unchosen_targets = torch.from_numpy(unchosen_targets).float().to(device)
        n_layers = len(parts.layers)
        batch = len(positions)
        self.pre = torch.full((n_layers, batch), float("nan"))
        self.post = torch.full((n_layers, batch), float("nan"))
        self.norm = torch.full((n_layers, batch), float("nan"))
        self.dose = torch.zeros((n_layers, batch))
        # Only the prespecified causal band is needed.  Some Qwen hybrid
        # blocks emit invalid historical-token outputs at earlier readouts in
        # long full-conversation forwards even while final logits and the
        # later historical states remain finite.  Observing those irrelevant
        # early outputs made the experiment fail nondeterministically.
        target_layers = [readout - 1 for readout in TARGET_READOUTS]
        self.handles = [
            parts.layers[index].register_forward_hook(self._hook(index))
            for index in target_layers
        ]

    def _score(self, index: int, current: Any) -> Any:
        means = self.means[
            index, self.letters.to(self.means.device)
        ].to(current.device)
        scales = self.scales[index].to(current.device)
        weights = self.weights[index].to(current.device)
        return ((current.float() - means) / scales).mul(
            weights
        ).sum(dim=-1)

    def _hook(self, index: int):
        def intervene(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            hidden = _hidden(output)
            positions = self.positions.to(hidden.device)
            if int(positions.min()) < 0 or int(positions.max()) >= hidden.shape[1]:
                raise RuntimeError(
                    f"Layer {index}: positions {positions.tolist()} outside "
                    f"hidden shape {tuple(hidden.shape)}"
                )
            rows_all = torch.arange(hidden.shape[0], device=hidden.device)
            current = hidden[rows_all, positions].float()
            score = self._score(index, current)
            if not torch.isfinite(score).all():
                normalized = (
                    current
                    - self.means[index, self.letters.to(self.means.device)].to(
                        current.device
                    )
                ) / self.scales[index].to(current.device)
                raise RuntimeError(
                    f"Non-finite probe score in mode={self.mode} layer={index}; "
                    f"hidden_finite={bool(torch.isfinite(current).all())} "
                    f"hidden_finite_by_row="
                    f"{torch.isfinite(current).all(dim=-1).tolist()} "
                    f"hidden_nonfinite_counts_by_row="
                    f"{(~torch.isfinite(current)).sum(dim=-1).tolist()} "
                    f"hidden_absmax={float(current.abs().max())} "
                    f"normalized_finite={bool(torch.isfinite(normalized).all())} "
                    f"normalized_absmax={float(normalized.abs().max())} "
                    f"weight_absmax={float(self.weights[index].abs().max())}"
                )
            self.pre[index] = score.detach().cpu()
            self.norm[index] = current.norm(dim=-1).detach().cpu()
            if self.mode == "natural" or (index + 1) not in TARGET_READOUTS:
                self.post[index] = score.detach().cpu()
                # A PyTorch forward hook should return None when it is only
                # observing.  Returning the original output is normally
                # equivalent, but Qwen's output-capturing wrapper can treat a
                # non-None hook result as a replacement and corrupt later
                # hybrid-attention blocks.
                return None

            rows = torch.as_tensor(self.selected_rows, device=hidden.device)
            target_rows = torch.as_tensor(
                self.selected_rows, device=self.chosen_targets.device
            )
            if self.mode == "chosen_sham":
                # Exact zero-dose repeat-forward control.  Using the cached
                # typical-chosen score here introduces a small real clamp on
                # the live full-prompt state and therefore is not a sham.
                target = score[rows].detach()
            elif self.mode == "devalue":
                target = self.unchosen_targets[index, target_rows]
            else:
                target = (
                    2 * self.chosen_targets[index, target_rows]
                    - self.unchosen_targets[index, target_rows]
                )
            target = target.to(hidden.device)
            gradient = self.gradient[index].to(hidden.device)
            gradient_norm_sq = self.gradient_norm_sq[index].to(hidden.device)
            coefficient = (target - score[rows]) / gradient_norm_sq
            update = coefficient[:, None] * gradient[None, :]
            changed = hidden.clone()
            old = current[rows]
            changed[rows, positions[rows]] = (old + update).to(hidden.dtype)
            actual = changed[rows, positions[rows]].float()
            post = self._score(index, changed[rows_all, positions].float())
            self.post[index] = post.detach().cpu()
            self.dose[index, list(self.selected_rows)] = (
                (actual - old).norm(dim=-1).detach().cpu()
            )
            return _replace_hidden(output, changed)

        return intervene

    def arrays(self) -> dict[str, np.ndarray]:
        target_layers = np.asarray(TARGET_READOUTS, dtype=np.int64) - 1
        target_pre = self.pre.numpy()[target_layers]
        if not np.all(np.isfinite(target_pre)):
            missing = np.argwhere(~np.isfinite(target_pre))
            raise RuntimeError(
                "Not every target-layer hook recorded a finite pre-score; first "
                f"target-index/row pairs={missing[:20].tolist()} total={len(missing)}"
            )
        return {
            "pre_score": self.pre.numpy(),
            "post_score": self.post.numpy(),
            "residual_norm": self.norm.numpy(),
            "dose_l2": self.dose.numpy(),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "OptionValueClamp":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _trusted(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload["results"]
    if len(rows) != 500:
        raise ValueError(f"Trusted result is incomplete: {path}")
    return rows


def run(
    config_path: Path,
    eligible_path: Path,
    second_mapping_path: Path,
    baseline_path: Path,
    residual_cache: Path,
    cache_results: Path,
    probe_results: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    split: str,
    max_cohorts: int | None,
) -> None:
    import torch
    import transformers

    config = ExperimentConfig.load(config_path)
    if config.prompt_mode != "baseline_matched_empty_history":
        raise ValueError("Requires baseline_matched_empty_history")
    if config.feedback_variant != "token_matched_test":
        raise ValueError("Requires token_matched_test feedback")
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("Requires raw Qwen ChatML")
    if config.attn_implementation != "sdpa" or int(config.batch_size) != 4:
        raise ValueError("Requires historical batch-four SDPA")
    if split not in {"discovery", "confirmation"}:
        raise ValueError("Unknown split")

    eligible_all = json.loads(eligible_path.read_text())["rows"]
    rows = [
        row
        for row in eligible_all
        if row["split"] == split and row["w1_displayed_letter"] != "A"
    ]
    expected = 74 if split == "discovery" else 71
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} eligible rows, found {len(rows)}")
    rows.sort(key=lambda row: row["question_id"])
    row_lookup = {row["question_id"]: row for row in rows}

    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    all_qids = [row["id"] for row in manifest]
    questions = {row["id"]: row for row in manifest}
    baseline = json.loads(baseline_path.read_text())["results"]
    second_rows = {
        row["question_id"]: row
        for row in json.loads(second_mapping_path.read_text())["rows"]
    }
    trusted = {
        "game": _trusted(trusted_game_path),
        "neutral": _trusted(trusted_neutral_path),
    }

    chosen_all, unchosen_all, cache_qindex = _target_scores(
        residual_cache, cache_results, probe_results, rows
    )
    del cache_qindex
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, rows, chosen_all, unchosen_all)
    output_index = {
        qid: index for index, qid in enumerate(arrays["question_ids"].astype(str))
    }

    with np.load(probe_results, allow_pickle=False) as loaded:
        weights = loaded["weights"].astype(np.float32)
        means = loaded["letter_means"].astype(np.float32)
        scales = loaded["scales"].astype(np.float32)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()
    completed_cohorts = 0
    pending = {
        qid for qid in row_lookup if not arrays["completed"][output_index[qid]]
    }
    total_cohorts = sum(
        bool(set(all_qids[start : start + 4]) & pending)
        for start in range(0, len(all_qids), 4)
    )
    cohort_durations: list[float] = []

    for start in range(0, len(all_qids), 4):
        group_qids = all_qids[start : start + 4]
        targets = [qid for qid in group_qids if qid in pending]
        if not targets:
            continue
        cohort_started = time.monotonic()
        selected_rows = [group_qids.index(qid) for qid in targets]
        group_letters = [LETTERS.index(baseline[qid]["answer"]) for qid in group_qids]
        group_chosen = np.zeros((64, len(group_qids)), dtype=np.float32)
        group_unchosen = np.zeros_like(group_chosen)
        for qid in targets:
            source = output_index[qid]
            destination = group_qids.index(qid)
            group_chosen[:, destination] = chosen_all[source]
            group_unchosen[:, destination] = unchosen_all[source]

        for condition_index, (condition, condition_name) in enumerate(
            zip(CONDITIONS, CONDITION_NAMES)
        ):
            prompts = []
            unpadded_positions = []
            lengths = []
            for qid in group_qids:
                question = questions[qid]
                second = _remap_question(
                    question, second_rows[qid]["new_to_original"]
                )
                prompt = render_chat(
                    processor,
                    _messages(config, question, second, condition),
                    config.disable_thinking,
                    config.chat_serialization,
                )
                positions, _audit = _positions(tokenizer, prompt, question)
                line_position = positions[
                    ANCHORS.index(f"line_end_{baseline[qid]['answer']}")
                ]
                ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                if tokenizer.decode([ids[line_position]]) != "\n":
                    raise RuntimeError(f"{qid}: W1 position is not a newline")
                if qid in targets:
                    expected_hash = trusted[condition_name][qid]["prompt_hash"]
                    if prompt_hash(prompt) != expected_hash:
                        raise RuntimeError(
                            f"{qid}: {condition_name} prompt differs from trusted run"
                        )
                prompts.append(prompt)
                unpadded_positions.append(line_position)
                lengths.append(len(ids))

            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            width = int(input_ids.shape[1])
            padded_positions = [
                int(position) + width - int(length)
                for position, length in zip(unpadded_positions, lengths)
            ]
            if completed_cohorts == 0:
                print(
                    "option-value position audit "
                    + json.dumps(
                        {
                            "condition": condition_name,
                            "width": width,
                            "lengths": lengths,
                            "unpadded_positions": unpadded_positions,
                            "padded_positions": padded_positions,
                            "mask_at_positions": [
                                int(attention_mask[row, position])
                                for row, position in enumerate(padded_positions)
                            ],
                            "tokens_at_positions": [
                                tokenizer.decode([int(input_ids[row, position])])
                                for row, position in enumerate(padded_positions)
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            for mode_index, mode in enumerate(MODES):
                with OptionValueClamp(
                    parts,
                    padded_positions,
                    group_letters,
                    selected_rows,
                    weights,
                    means,
                    scales,
                    group_chosen,
                    group_unchosen,
                    mode,
                ) as hook:
                    output = _forward(model, parts, input_ids, attention_mask)
                    if not torch.isfinite(output.logits).all():
                        raise RuntimeError(
                            f"Non-finite final logits for {condition_name}/{mode}"
                        )
                    hook_arrays = hook.arrays()
                final = output.logits.detach().float()
                final = final[:, 0] if final.shape[1] == 1 else final[:, -1]
                logits = _aggregate_logits(final, variant_ids).cpu().numpy()

                for qid in targets:
                    group_row = group_qids.index(qid)
                    qi = output_index[qid]
                    arrays["logits"][condition_index, mode_index, qi] = logits[group_row]
                    arrays["pre_score"][condition_index, mode_index, qi] = (
                        hook_arrays["pre_score"][:, group_row]
                    )
                    arrays["post_score"][condition_index, mode_index, qi] = (
                        hook_arrays["post_score"][:, group_row]
                    )
                    arrays["residual_norm"][condition_index, mode_index, qi] = (
                        hook_arrays["residual_norm"][:, group_row]
                    )
                    arrays["dose_l2"][condition_index, mode_index, qi] = (
                        hook_arrays["dose_l2"][:, group_row]
                    )
                    if mode == "natural":
                        reference = np.asarray(
                            trusted[condition_name][qid]["aggregated_ad_logits"],
                            dtype=np.float32,
                        )
                        difference = float(
                            np.max(np.abs(logits[group_row] - reference))
                        )
                        arrays["trusted_max_abs_error"][condition_index, qi] = (
                            difference
                        )
                        arrays["trusted_choice_match"][condition_index, qi] = bool(
                            int(np.argmax(logits[group_row]))
                            == int(np.argmax(reference))
                        )

                if not audit_path.exists() and mode == "devalue":
                    qid = targets[0]
                    group_row = group_qids.index(qid)
                    qi = output_index[qid]
                    audit_path.write_text(
                        json.dumps(
                            {
                                "question_id": qid,
                                "condition": condition_name,
                                "historical_group_qids": group_qids,
                                "selected_rows": selected_rows,
                                "w1_letter": baseline[qid]["answer"],
                                "w1_newline_position_unpadded": unpadded_positions[
                                    group_row
                                ],
                                "w1_newline_position_padded": padded_positions[
                                    group_row
                                ],
                                "w1_token": tokenizer.decode(
                                    [int(input_ids[group_row, padded_positions[group_row]])]
                                ),
                                "prompt_hash": prompt_hash(prompts[group_row]),
                                "target_readouts": list(TARGET_READOUTS),
                                "chosen_target_readout_53": float(
                                    arrays["chosen_target_score"][qi, 52]
                                ),
                                "unchosen_target_readout_53": float(
                                    arrays["unchosen_target_score"][qi, 52]
                                ),
                                "live_pre_readout_53": float(
                                    hook_arrays["pre_score"][52, group_row]
                                ),
                                "post_readout_53": float(
                                    hook_arrays["post_score"][52, group_row]
                                ),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )

        for qid in targets:
            arrays["completed"][output_index[qid]] = True
        atomic_save_npz(result_path, **arrays)
        pending.difference_update(targets)
        completed_cohorts += 1
        duration = time.monotonic() - cohort_started
        cohort_durations.append(duration)
        elapsed = time.monotonic() - started
        remaining = total_cohorts - completed_cohorts
        eta = elapsed / completed_cohorts * remaining
        print(
            f"option-value causal {split}: {int(arrays['completed'].sum())}/"
            f"{len(rows)}; cohort={duration:.1f}s elapsed={elapsed/60:.1f}m "
            f"eta={eta/60:.1f}m",
            flush=True,
        )
        if max_cohorts is not None and completed_cohorts >= int(max_cohorts):
            print(f"Stopped after {max_cohorts} benchmark cohorts", flush=True)
            break

    metadata = {
        "experiment": "Option-newline candidate-value coordinate causal clamp",
        "config": config.as_dict(),
        "split": split,
        "n_questions": len(rows),
        "conditions": list(CONDITION_NAMES),
        "prompt_conditions": list(CONDITIONS),
        "modes": list(MODES),
        "target_readouts": list(TARGET_READOUTS),
        "intervention_position": "first-presentation W1 option-closing newline",
        "complete_model_forwards_per_cohort": len(CONDITIONS) * len(MODES),
        "total_target_cohorts": total_cohorts,
        "complete": bool(arrays["completed"].all()),
        "natural_validation": {
            "definition": (
                "Exact prompt hashes plus fresh same-host natural companions; "
                "old-host logits are diagnostic because replacement-host SDPA "
                "numerics are not bit-exact."
            ),
            "max_abs_old_host_logit_error": float(
                np.nanmax(arrays["trusted_max_abs_error"])
            ),
            "old_host_choice_agreement": float(
                arrays["trusted_choice_match"][:, arrays["completed"]].mean()
            ),
        },
        "elapsed_seconds_after_model_load": time.monotonic() - started,
        "completed_cohort_durations_seconds": cohort_durations,
        "resolved_answer_tokens": resolved,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eligible-pairs", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--residual-cache", type=Path, required=True)
    parser.add_argument("--cache-results", type=Path, required=True)
    parser.add_argument("--probe-results", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("discovery", "confirmation"), required=True
    )
    parser.add_argument("--max-cohorts", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.eligible_pairs,
        args.second_mapping,
        args.baseline,
        args.residual_cache,
        args.cache_results,
        args.probe_results,
        args.trusted_game,
        args.trusted_neutral,
        args.output_dir,
        args.split,
        args.max_cohorts,
    )


if __name__ == "__main__":
    main()
