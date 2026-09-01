from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import load_activation_dataset
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    tokenize_batch,
)
from .prompts import build_messages, load_trials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify actual full-vocabulary logits at the matched pre-redacted prefix"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--jlens-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch

    config = ExperimentConfig.load(args.config)
    with np.load(args.jlens_root / "jlens_scores.npz", allow_pickle=False) as cached:
        question_ids = cached["position_question_ids"].astype(str).tolist()
    trials = {
        trial.question_id: trial
        for trial in load_trials(config.manifest_path, config.baseline_results_path)
    }
    data = load_activation_dataset(args.residual_root, ["baseline", "incorrect", "neutral"])
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)

    prompts = []
    audit_rows = []
    for qid in question_ids:
        trial = trials[qid]
        baseline = render_chat(
            processor,
            build_messages(trial.question, "baseline", config.prompt_mode),
            config.disable_thinking,
            config.chat_serialization,
        )
        row = {"question_id": qid}
        for condition in ("incorrect", "neutral"):
            rendered = render_chat(
                processor,
                build_messages(trial.question, condition, config.prompt_mode),
                config.disable_thinking,
                config.chat_serialization,
            )
            marker = rendered.find("[redacted]")
            if marker >= 0:
                prefix = rendered[:marker]
            elif config.prompt_mode == "baseline_matched_empty_history":
                if not rendered.startswith(baseline + "<|im_end|>\n"):
                    raise RuntimeError(
                        f"Empty historical assistant does not begin at the "
                        f"Baseline boundary for {condition}/{qid}"
                    )
                prefix = rendered[:len(baseline)]
            else:
                raise RuntimeError(
                    f"No historical-answer boundary for {condition}/{qid}"
                )
            row[f"{condition}_rendered_prefix_exact"] = prefix == baseline
            row[f"{condition}_token_prefix_exact"] = (
                tokenizer(prefix, add_special_tokens=False)["input_ids"]
                == tokenizer(baseline, add_special_tokens=False)["input_ids"]
            )
        prompts.append(baseline)
        audit_rows.append(row)

    top_ids: list[int] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, batch)
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
            logits = logits[:, -1:]
        top_ids.extend(logits[:, 0].argmax(dim=-1).tolist())
        print(f"verified {len(top_ids)}/{len(prompts)}", flush=True)

    saved_baseline_ids = [
        int(data.metadata[(qid, "baseline")]["full_vocab_top_token_id"])
        for qid in question_ids
    ]
    matches = np.asarray(top_ids) == np.asarray(saved_baseline_ids)
    result = {
        "n": len(question_ids),
        "serialization": config.chat_serialization,
        "decision_mode": config.decision_mode,
        "all_game_rendered_prefixes_exact": all(
            row["incorrect_rendered_prefix_exact"] for row in audit_rows
        ),
        "all_neutral_rendered_prefixes_exact": all(
            row["neutral_rendered_prefix_exact"] for row in audit_rows
        ),
        "all_game_token_prefixes_exact": all(
            row["incorrect_token_prefix_exact"] for row in audit_rows
        ),
        "all_neutral_token_prefixes_exact": all(
            row["neutral_token_prefix_exact"] for row in audit_rows
        ),
        "recomputed_actual_top_matches_saved_baseline": {
            "hits": int(matches.sum()),
            "n": int(len(matches)),
            "rate": float(matches.mean()),
        },
        "top_token_counts": {
            tokenizer.decode([token_id]): int(top_ids.count(token_id))
            for token_id in sorted(set(top_ids))
        },
        "trials": [
            {
                **row,
                "actual_top_token_id": int(top_id),
                "actual_top_token": tokenizer.decode([int(top_id)]),
                "saved_baseline_top_token_id": int(saved_id),
                "match": bool(match),
            }
            for row, top_id, saved_id, match in zip(
                audit_rows, top_ids, saved_baseline_ids, matches
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
