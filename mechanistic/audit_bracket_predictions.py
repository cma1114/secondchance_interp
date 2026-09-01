from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .io import shard_path
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    resolve_answer_tokens,
    tokenize_batch,
)


CONDITIONS = ("baseline", "incorrect", "neutral")


def _metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as shard:
        raw = shard["metadata"]
        if raw.shape == ():
            raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(str(raw))


def _load_existing(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["condition"], row["question_id"])] = row
    return rows


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        ranks = [int(row["left_bracket_rank"]) for row in subset]
        top_counts = Counter(row["top_token"] for row in subset)
        result[condition] = {
            "n": len(subset),
            "left_bracket_argmax": sum(rank == 1 for rank in ranks),
            "left_bracket_top4": sum(rank <= 4 for rank in ranks),
            "left_bracket_top10": sum(rank <= 10 for rank in ranks),
            "left_bracket_rank_counts_top10": {
                str(rank): int(sum(value == rank for value in ranks)) for rank in range(1, 11)
            },
            "left_bracket_rank_quantiles": {
                "median": float(np.median(ranks)),
                "p10": float(np.quantile(ranks, 0.10)),
                "p25": float(np.quantile(ranks, 0.25)),
                "minimum": int(min(ranks)),
            },
            "left_bracket_probability": _quantiles(
                [float(row["left_bracket_probability"]) for row in subset]
            ),
            "ad_probability_mass": _quantiles(
                [float(row["ad_probability_mass"]) for row in subset]
            ),
            "bracket_minus_best_ad_logit": _quantiles(
                [float(row["bracket_minus_best_ad_logit"]) for row in subset]
            ),
            "any_bracket_leading_token_in_top4": sum(
                bool(row["any_bracket_leading_token_in_top4"]) for row in subset
            ),
            "top_token_counts": dict(top_counts.most_common()),
        }
    return result


def audit(config_path: Path, output_dir: Path, batch_size: int) -> None:
    import torch

    config = ExperimentConfig.load(config_path)
    residual_root = Path(config.output_dir)
    if config.chat_serialization != "raw_qwen_chatml":
        raise ValueError("This audit is specifically for the explicit empty-think raw serializer")

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "bracket_trials.jsonl"
    existing = _load_existing(jsonl_path)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    bracket_ids = tokenizer.encode("[", add_special_tokens=False)
    if len(bracket_ids) != 1:
        raise ValueError(f"Expected '[' to be one token, got {bracket_ids}")
    bracket_id = int(bracket_ids[0])
    answer_tokens = resolve_answer_tokens(tokenizer, config.answer_variants)
    ad_ids = sorted({token_id for values in answer_tokens.values() for _, token_id in values})

    work: list[tuple[str, str, str, str]] = []
    for condition in CONDITIONS:
        condition_dir = residual_root / "shards" / condition
        for path in sorted(condition_dir.glob("*.npz")):
            metadata = _metadata(path)
            key = (condition, metadata["question_id"])
            if key not in existing:
                work.append(
                    (condition, metadata["question_id"], metadata["rendered_prompt"], metadata["prompt_hash"])
                )

    completed = len(existing)
    with jsonl_path.open("a") as stream:
        for start in range(0, len(work), batch_size):
            batch = work[start : start + batch_size]
            prompts = [item[2] for item in batch]
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids.to(model_input_device(parts)),
                    "attention_mask": attention_mask.to(model_input_device(parts)),
                    "use_cache": False,
                    "return_dict": True,
                }
                try:
                    output = model(**kwargs, logits_to_keep=1)
                except TypeError:
                    output = model(**kwargs)
            logits = output.logits.detach().float().cpu()
            if logits.shape[1] != 1:
                last_indices = attention_mask.sum(dim=1).long() - 1
                logits = logits[torch.arange(len(batch)), last_indices][:, None]
            logits = logits[:, 0]
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_ids = torch.topk(probabilities, k=10, dim=-1)
            bracket_values = logits[:, bracket_id]
            ranks = 1 + (logits > bracket_values[:, None]).sum(dim=-1)
            best_ad = logits[:, ad_ids].max(dim=-1).values

            for index, (condition, question_id, _prompt, prompt_hash) in enumerate(batch):
                token_ids = [int(value) for value in top_ids[index].tolist()]
                tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
                row = {
                    "condition": condition,
                    "question_id": question_id,
                    "prompt_hash": prompt_hash,
                    "left_bracket_token_id": bracket_id,
                    "left_bracket_rank": int(ranks[index]),
                    "left_bracket_probability": float(probabilities[index, bracket_id]),
                    "ad_probability_mass": float(probabilities[index, ad_ids].sum()),
                    "bracket_minus_best_ad_logit": float(bracket_values[index] - best_ad[index]),
                    "top_token_id": token_ids[0],
                    "top_token": tokens[0],
                    "top10": [
                        {
                            "rank": rank + 1,
                            "token_id": token_id,
                            "token": token,
                            "probability": float(top_values[index, rank]),
                        }
                        for rank, (token_id, token) in enumerate(zip(token_ids, tokens))
                    ],
                    "any_bracket_leading_token_in_top4": any(
                        token.lstrip().startswith("[") for token in tokens[:4]
                    ),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                existing[(condition, question_id)] = row
            completed += len(batch)
            if completed == len(CONDITIONS) * 500 or completed % 50 < len(batch):
                print(f"saved {completed}/{len(CONDITIONS) * 500}", flush=True)

    rows = list(existing.values())
    if len(rows) != len(CONDITIONS) * 500:
        raise RuntimeError(f"Expected 1500 completed rows, found {len(rows)}")
    summary = {
        "definition": {
            "prompt_source": "Exact rendered_prompt saved in each raw_chatml_matched activation shard.",
            "left_bracket": "The exact single token '['; token ID recorded below.",
            "rank": "One plus the number of full-vocabulary logits strictly greater than the '[' logit.",
            "probability": "Softmax probability over the complete output vocabulary.",
            "ad_probability_mass": "Sum over the configured bare and leading-space A-D token variants.",
        },
        "left_bracket_token_id": bracket_id,
        "conditions": _summarize(rows),
    }
    (output_dir / "bracket_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    with (output_dir / "bracket_trials.csv").open("w", newline="") as stream:
        fields = [
            "condition", "question_id", "left_bracket_rank", "left_bracket_probability",
            "ad_probability_mass", "bracket_minus_best_ad_logit", "top_token",
            "any_bracket_leading_token_in_top4",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: (CONDITIONS.index(value["condition"]), value["question_id"])):
            writer.writerow({field: row[field] for field in fields})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit full-vocabulary '[' ranks on saved prompts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    audit(args.config, args.output, args.batch_size)


if __name__ == "__main__":
    main()
