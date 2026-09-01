from __future__ import annotations

import argparse


def main() -> None:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument(
        "--revision", default="842da3794eaa0b77d5f08bae87a17459d91ff475"
    )
    parser.add_argument("--tp", action="store_true")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model_kwargs = {
        "revision": args.revision,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if args.tp:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, **model_kwargs
    ).eval()
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Answer only with the capital city: The capital of France is",
        }],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    rank = int(__import__("os").environ.get("RANK", "0"))
    print("RANK", rank, "MODEL_CLASS", type(model).__name__, flush=True)
    print("RANK", rank, "INPUT_KEYS", sorted(inputs.keys()), flush=True)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            use_cache=True,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    if rank == 0:
        print(
            "TOKENS",
            processor.tokenizer.convert_ids_to_tokens(generated.tolist()),
            flush=True,
        )
        print(
            "TEXT",
            repr(processor.decode(generated, skip_special_tokens=False)),
            flush=True,
        )


if __name__ == "__main__":
    main()
