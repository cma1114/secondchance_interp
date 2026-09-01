from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from mechanistic.analyze_contextual_option_representations import _load
from mechanistic.config import ExperimentConfig
from mechanistic.modeling import get_tokenizer, load_model_and_processor, model_input_device


LETTERS = "ABCD"
ANCHORS = ("content_end", "line_end")


def _display(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _align(values: np.ndarray, metadata: dict, qids: list[str]) -> np.ndarray:
    if not metadata.get("mappings"):
        return values
    indices = np.empty((len(qids), 4), dtype=np.int64)
    for qi, qid in enumerate(qids):
        mapping = metadata["mappings"][qid]["original_to_new"]
        indices[qi] = [LETTERS.index(mapping[letter]) for letter in LETTERS]
    return np.take_along_axis(values, indices[:, :, None], axis=1)


def _sample_qids(qids: list[str], baseline_results: Path, count_per_letter: int, seed: int) -> list[str]:
    rows = json.loads(baseline_results.read_text())["results"]
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for letter in LETTERS:
        pool = np.asarray([qid for qid in qids if rows[qid]["answer"] == letter])
        if len(pool) < count_per_letter:
            raise ValueError(f"Not enough held-out Baseline-{letter} questions")
        selected.extend(rng.choice(pool, size=count_per_letter, replace=False).tolist())
    # Interleave answer letters so the explorer is not visually grouped.
    return [selected[i + 4 * j] for j in range(count_per_letter) for i in range(4)]


def _substantive_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    keep = []
    for token_id in ids:
        decoded = tokenizer.decode([int(token_id)]).strip()
        if len(decoded) >= 2 and any(character.isalnum() for character in decoded):
            keep.append(int(token_id))
    return sorted(set(keep or [int(value) for value in ids]))


def collect(
    config_path: Path,
    confirmation_roots: list[Path],
    baseline_results: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
    count_per_letter: int,
    top_k: int,
    seed: int,
) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    if len(confirmation_roots) != 4:
        raise ValueError("Exactly four confirmation mappings are required")
    sets = [_load(path) for path in confirmation_roots]
    qids = list(sets[0][1]["question_ids"])
    for _, metadata in sets[1:]:
        if list(metadata["question_ids"]) != qids:
            raise ValueError("Confirmation question order differs across mappings")
    for qid in qids:
        for original in LETTERS:
            positions = [
                original if not metadata.get("mappings")
                else metadata["mappings"][qid]["original_to_new"][original]
                for _, metadata in sets
            ]
            if set(positions) != set(LETTERS):
                raise ValueError(f"{qid}: {original} does not occupy A-D exactly once")

    selected_qids = _sample_qids(qids, baseline_results, count_per_letter, seed)
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    selected_indices = np.asarray([qid_to_index[qid] for qid in selected_qids])

    config = ExperimentConfig.load(config_path)
    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    if sorted(int(value) for value in jacobians) != list(range(63)):
        raise ValueError("JLens checkpoint does not contain source layers 0--62")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    baseline_rows = json.loads(baseline_results.read_text())["results"]
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}

    audit: dict[str, dict] = {}
    option_token_ids: list[list[list[int]]] = []
    for qid in selected_qids:
        question = questions[qid]
        option_ids = [_substantive_ids(tokenizer, question["options"][letter]) for letter in LETTERS]
        option_token_ids.append(option_ids)
        audit[qid] = {
            "question": question["question"],
            "options": question["options"],
            "correct_answer": question["correct_answer"],
            "baseline_answer": baseline_rows[qid]["answer"],
            "substantive_option_tokens": {
                letter: [
                    {"token_id": token_id, "token": _display(tokenizer.decode([token_id]))}
                    for token_id in option_ids[index]
                ]
                for index, letter in enumerate(LETTERS)
            },
        }

    # Reproduce the confirmation-set shifts used by the shuffled-content null.
    rng = np.random.default_rng(seed + 900001)
    rng.integers(1, 4, size=251)  # discovery shifts
    confirmation_shift_all = rng.integers(1, 4, size=len(qids))
    selected_shifts = confirmation_shift_all[selected_indices]

    top_tokens: dict[str, dict] = {qid: {} for qid in selected_qids}
    lexical_scores = np.zeros(
        (len(selected_qids), 2, 2, 64, 4, 4), dtype=np.float32
    )
    exact_top_hit = np.zeros((len(selected_qids), 2, 2, 64, 4), dtype=bool)

    for anchor_index, anchor in enumerate(ANCHORS):
        for layer in range(64):
            aligned = []
            for residuals, metadata in sets:
                indices = [metadata["anchors"].index(f"{anchor}_{letter}") for letter in LETTERS]
                raw = np.asarray(residuals[selected_indices, layer][:, indices], dtype=np.float32)
                aligned.append(_align(raw, metadata, selected_qids))
            average = np.mean(aligned, axis=0)
            flat = torch.from_numpy(average.reshape(-1, average.shape[-1])).to(
                device, dtype=torch.float16
            )
            methods = {"logit_lens": flat}
            if layer < 63:
                J = jacobians[layer].to(device, dtype=torch.float16)
                methods["jlens"] = flat @ J.T
                del J
            else:
                methods["jlens"] = flat

            for method_index, method in enumerate(("jlens", "logit_lens")):
                transported = methods[method]
                with torch.inference_mode():
                    normed = parts.final_norm(
                        transported.to(parts.final_norm.weight.dtype)
                    )
                    logits = parts.output_head(normed).float().reshape(
                        len(selected_qids), 4, -1
                    )
                    contrast = logits - logits.mean(dim=1, keepdim=True)
                    raw_scores, raw_ids = torch.topk(logits, k=top_k, dim=-1)
                    contrast_scores, contrast_ids = torch.topk(
                        contrast, k=top_k, dim=-1
                    )
                    bottom_values, bottom_ids = torch.topk(
                        -contrast, k=top_k, dim=-1
                    )

                for qi, qid in enumerate(selected_qids):
                    layer_row = top_tokens[qid].setdefault(str(layer + 1), {})
                    method_row = layer_row.setdefault(method, {})
                    anchor_row = method_row.setdefault(anchor, {})
                    for option_index, letter in enumerate(LETTERS):
                        def rows(scores, ids, sign=1.0):
                            return [
                                {
                                    "token_id": int(token_id),
                                    "token": _display(tokenizer.decode([int(token_id)])),
                                    "score": round(float(score) * sign, 4),
                                }
                                for score, token_id in zip(scores, ids)
                            ]

                        anchor_row[letter] = {
                            "raw_top": rows(
                                raw_scores[qi, option_index].cpu(),
                                raw_ids[qi, option_index].cpu(),
                            ),
                            "contrast_top": rows(
                                contrast_scores[qi, option_index].cpu(),
                                contrast_ids[qi, option_index].cpu(),
                            ),
                            "contrast_bottom": rows(
                                bottom_values[qi, option_index].cpu(),
                                bottom_ids[qi, option_index].cpu(),
                                -1.0,
                            ),
                        }
                        target_ids = set(option_token_ids[qi][option_index])
                        exact_top_hit[qi, anchor_index, method_index, layer, option_index] = bool(
                            target_ids & set(int(value) for value in contrast_ids[qi, option_index].cpu())
                        )
                        for content_index in range(4):
                            ids = torch.as_tensor(
                                option_token_ids[qi][content_index], device=contrast.device
                            )
                            lexical_scores[
                                qi, anchor_index, method_index, layer, option_index, content_index
                            ] = float(contrast[qi, option_index, ids].max().cpu())
                del logits, contrast, transported
            if layer == 0 or (layer + 1) % 8 == 0 or layer == 63:
                print(f"four-map lens audit {anchor}: {layer + 1}/64", flush=True)

    lexical_prediction = lexical_scores.argmax(axis=-1)
    target = np.arange(4)[None, None, None, None, :]
    lexical_correct = lexical_prediction == target
    summary = {
        "definition": (
            "JLens and native logit-lens decoding of raw option residuals averaged "
            "across four mappings in which each semantic content occupies A-D once. "
            "Contrast tokens are candidate vocabulary scores minus the within-question "
            "mean across the four option candidates."
        ),
        "split": "frozen held-out 249-question confirmation set",
        "sample_seed": seed,
        "count_per_baseline_letter": count_per_letter,
        "question_ids": selected_qids,
        "layers": list(range(1, 65)),
        "anchors": list(ANCHORS),
        "methods": ["jlens", "logit_lens"],
        "top_k": top_k,
        "shuffled_shift_by_question": {
            qid: int(shift) for qid, shift in zip(selected_qids, selected_shifts)
        },
        "audit": audit,
        "lexical_option_accuracy": {
            anchor: {
                method: lexical_correct[:, ai, mi].mean(axis=(0, 2)).tolist()
                for mi, method in enumerate(("jlens", "logit_lens"))
            }
            for ai, anchor in enumerate(ANCHORS)
        },
        "exact_target_token_top_k_rate": {
            anchor: {
                method: exact_top_hit[:, ai, mi].mean(axis=(0, 2)).tolist()
                for mi, method in enumerate(("jlens", "logit_lens"))
            }
            for ai, anchor in enumerate(ANCHORS)
        },
        "lexical_scores": lexical_scores.tolist(),
        "top_tokens": top_tokens,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirmation-roots", nargs=4, type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--count-per-letter", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    collect(
        args.config,
        args.confirmation_roots,
        args.baseline_results,
        args.output,
        args.lens_repo,
        args.lens_filename,
        args.count_per_letter,
        args.top_k,
        args.seed,
    )


if __name__ == "__main__":
    main()
