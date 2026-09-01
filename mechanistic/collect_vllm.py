from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, config_arg_parser
from .io import atomic_save_npz, json_array, shard_path
from .modeling import resolve_answer_tokens, variant_layout
from .prompts import build_messages, load_trials, prompt_hash


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _render(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def collect(config: ExperimentConfig) -> None:
    """Collect a native logit lens through the behaviorally validated vLLM path."""
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials(
        config.manifest_path,
        config.baseline_results_path,
        config.question_ids,
        config.max_questions,
        config.skip_missing_baseline,
    )

    llm = LLM(
        model=config.model_id,
        tokenizer=config.model_id,
        tensor_parallel_size=8,
        dtype=config.dtype,
        quantization="compressed-tensors",
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=max(16, config.batch_size),
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        async_scheduling=False,
        seed=config.seed,
        worker_extension_cls="mechanistic.vllm_lens.LensWorkerExtension",
    )
    tokenizer = llm.get_tokenizer()
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    selected_ids, variant_meta = variant_layout(resolved)
    canonical_positions = [
        next(i for i, item in enumerate(variant_meta) if item["letter"] == c and item["text"] == c)
        for c in "ABCD"
    ]
    install = llm.collective_rpc("install_lens_capture", args=(selected_ids,))
    rank_zero = next(item for item in install if item["tp_rank"] == 0)

    run_meta = {
        "config": config.as_dict(),
        "resolved_answer_tokens": resolved,
        "variant_layout": variant_meta,
        "n_text_layers": rank_zero["n_layers"],
        "collector": "vllm_native_final_position_logit_lens",
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        },
        "platform": platform.platform(),
        "worker_install": install,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True))
    sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    for condition in config.conditions:
        pending = [t for t in trials if not shard_path(output_dir, condition, t.question_id).exists()]
        print(f"{condition}: {len(pending)} pending / {len(trials)} total", flush=True)
        completed = 0
        for batch in _chunks(pending, config.batch_size):
            messages = [build_messages(t.question, condition, config.prompt_mode) for t in batch]
            prompts = [_render(tokenizer, m) for m in messages]
            prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
            llm.collective_rpc("begin_lens_capture", args=(len(batch),))
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            captured = llm.collective_rpc("finish_lens_capture")
            lens = next(item for item in captured if item is not None)
            variant_logits = np.asarray(lens["variant_logits"], dtype=np.float32)
            residual_norms = np.asarray(lens["residual_norms"], dtype=np.float32)
            # vLLM's V2 runner packs new prefill requests in ascending prompt-
            # length order, whereas LLM.generate returns outputs in caller order.
            # Restore caller order before pairing readouts with trials.
            packed_order = sorted(range(len(batch)), key=lambda i: prompt_lengths[i])
            inverse_order = np.argsort(np.asarray(packed_order))
            variant_logits = variant_logits[inverse_order]
            residual_norms = residual_norms[inverse_order]
            canonical_logits = variant_logits[:, :, canonical_positions]

            for i, (trial, output) in enumerate(zip(batch, outputs)):
                generated_ids = list(output.outputs[0].token_ids)
                generated_id = int(generated_ids[0]) if generated_ids else -1
                generated_text = output.outputs[0].text
                final_choice = "ABCD"[int(canonical_logits[i, -1].argmax())]
                top_variant_index = int(variant_logits[i, -1].argmax())
                top_variant_choice = variant_meta[top_variant_index]["letter"]
                top_variant_text = variant_meta[top_variant_index]["text"]
                generated_answer_choice = None
                generated_vs_lens_agree = None
                if generated_id in selected_ids:
                    generated_answer_choice = variant_meta[selected_ids.index(generated_id)]["letter"]
                    generated_vs_lens_agree = generated_answer_choice == top_variant_choice
                    if not generated_vs_lens_agree:
                        # TP reductions and the separate selected-row matmul can
                        # flip very close candidates. Preserve and audit these
                        # trials instead of censoring them or terminating a
                        # resumable long run.
                        print(
                            f"WARNING final lens/generated mismatch {trial.question_id}: "
                            f"top_variant={top_variant_choice}, generated={generated_text!r}",
                            flush=True,
                        )
                meta = {
                    "question_id": trial.question_id,
                    "condition": condition,
                    "baseline_answer": trial.baseline_answer,
                    "baseline_correct": trial.baseline_correct,
                    "correct_answer": trial.question["correct_answer"],
                    "prompt_hash": prompt_hash(prompts[i]),
                    "prompt_length": prompt_lengths[i],
                    "rendered_prompt": prompts[i],
                    "messages": messages[i],
                    "full_vocab_top_token_id": generated_id,
                    "full_vocab_top_token": generated_text,
                    "canonical_ad_choice": final_choice,
                    "top_answer_variant_choice": top_variant_choice,
                    "top_answer_variant_text": top_variant_text,
                    "canonical_variant_agree": final_choice == top_variant_choice,
                    "generated_answer_choice": generated_answer_choice,
                    "generated_vs_lens_agree": generated_vs_lens_agree,
                }
                atomic_save_npz(
                    shard_path(output_dir, condition, trial.question_id),
                    canonical_logits=canonical_logits[i],
                    variant_logits=variant_logits[i],
                    residual_norms=residual_norms[i],
                    metadata=json_array(meta),
                )
            completed += len(batch)
            print(f"{condition}: saved {completed}/{len(pending)} pending trials", flush=True)

    llm.collective_rpc("remove_lens_capture")


def main() -> None:
    args = config_arg_parser("Collect a native vLLM final-position A-D logit lens").parse_args()
    collect(ExperimentConfig.load(args.config))


if __name__ == "__main__":
    main()
