from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import LETTERS


@dataclass
class ModelParts:
    layers: Any
    embedding: Any
    final_norm: Any
    output_head: Any


def _get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def locate_model_parts(model: Any) -> ModelParts:
    bases = (
        "model.language_model",
        "model.text_model",
        "language_model.model",
        "language_model",
        "model",
    )
    for base in bases:
        try:
            trunk = _get_path(model, base)
            layers = trunk.layers
            embedding = trunk.embed_tokens
            final_norm = trunk.norm
            break
        except (AttributeError, KeyError):
            continue
    else:
        raise RuntimeError(
            "Could not locate text decoder layers. Expected one of: " + ", ".join(bases)
        )
    head = model.get_output_embeddings()
    if head is None:
        for path in ("lm_head", "language_model.lm_head"):
            try:
                head = _get_path(model, path)
                break
            except AttributeError:
                pass
    if head is None or not hasattr(head, "weight"):
        raise RuntimeError("Could not locate the language-model output head")
    return ModelParts(layers, embedding, final_norm, head)


QWEN_EMPTY_THINKING = "<think>\n\n</think>\n\n"


def render_raw_qwen_chatml(messages: list[dict[str, str]]) -> str:
    """Serialize Qwen ChatML explicitly, without apply_chat_template.

    The empty reasoning block is written explicitly at every assistant turn.
    This keeps thinking disabled while making a live Baseline generation and
    the historical [redacted] turn token-identical up to [redacted].
    """
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported raw ChatML role: {role!r}")
        content = message["content"]
        if role == "assistant":
            content = QWEN_EMPTY_THINKING + content
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n" + QWEN_EMPTY_THINKING)
    return "".join(parts)


def render_bare_qwen_chatml(messages: list[dict[str, str]]) -> str:
    """Serialize literal Qwen ChatML without adding any thinking tokens."""
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported raw ChatML role: {role!r}")
        parts.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def render_hf_direct_assistant(processor: Any, messages: list[dict[str, str]]) -> str:
    """Use the native HF turn template but open a normal assistant response.

    Gemma 4's current ``add_generation_prompt`` helper opens its dedicated
    ``thought`` channel even when thinking is disabled.  That is unsuitable for
    answer-token logit measurements.  The native template serializes completed
    assistant messages directly after ``<|turn>model\n``.  We reproduce that
    official boundary by rendering the conversation without a generation
    prompt and appending only the model-turn opener.
    """
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("Native chat template did not return text")
    return rendered + "<|turn>model\n"


def render_chat(
    processor: Any,
    messages: list[dict[str, str]],
    disable_thinking: bool,
    serialization: str = "hf_template",
    template_kwargs: dict[str, Any] | None = None,
) -> str:
    if serialization == "raw_qwen_chatml":
        if not disable_thinking:
            raise ValueError("raw_qwen_chatml currently requires disable_thinking=True")
        return render_raw_qwen_chatml(messages)
    if serialization == "raw_qwen_chatml_bare":
        if not disable_thinking:
            raise ValueError("raw_qwen_chatml_bare requires disable_thinking=True")
        return render_bare_qwen_chatml(messages)
    if serialization == "hf_template_direct_assistant":
        if not disable_thinking:
            raise ValueError(
                "hf_template_direct_assistant requires disable_thinking=True"
            )
        if template_kwargs:
            raise ValueError(
                "hf_template_direct_assistant does not accept chat_template_kwargs"
            )
        return render_hf_direct_assistant(processor, messages)
    if serialization != "hf_template":
        raise ValueError(f"Unknown chat serialization: {serialization!r}")
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    kwargs.update(template_kwargs or {})
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def get_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def forward_runtime_kwargs(model: Any, input_ids: Any, device: Any) -> dict[str, Any]:
    """Return architecture-required kwargs for a fresh complete prompt forward."""
    import torch

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    if model_type == "gemma4":
        return {
            "use_cache": True,
            "mm_token_type_ids": torch.zeros_like(input_ids, device=device),
        }
    return {"use_cache": False}


def resolve_answer_tokens(tokenizer: Any, variants: dict[str, list[str]]) -> dict[str, list[tuple[str, int]]]:
    resolved: dict[str, list[tuple[str, int]]] = {}
    for letter in LETTERS:
        entries = []
        for text in variants[letter]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                entries.append((text, int(ids[0])))
        if not entries or entries[0][0] != letter:
            raise RuntimeError(f"Canonical answer {letter!r} is not an available single token")
        resolved[letter] = entries
    return resolved


def load_model_and_processor(config: Any) -> tuple[Any, Any, ModelParts]:
    import torch
    from transformers import AutoTokenizer

    dtype = getattr(torch, config.dtype)
    common = dict(
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    model_kwargs = dict(common, torch_dtype=dtype, device_map=config.device_map)
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    if config.model_loader == "causal_lm":
        from transformers import AutoModelForCausalLM

        processor = AutoTokenizer.from_pretrained(config.model_id, **common)
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    else:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(config.model_id, **common)
        model = AutoModelForMultimodalLM.from_pretrained(config.model_id, **model_kwargs)
    model.eval()
    return model, processor, locate_model_parts(model)


class ResidualCollector:
    """Capture embedding and post-block residuals at each prompt's final token."""

    def __init__(self, parts: ModelParts, last_indices: list[int]):
        self.last_indices = last_indices
        self.values: list[Any] = [None] * (len(parts.layers) + 1)
        self.handles = [parts.embedding.register_forward_hook(self._hook(0))]
        self.handles.extend(layer.register_forward_hook(self._hook(i + 1)) for i, layer in enumerate(parts.layers))

    def _hook(self, index: int) -> Callable:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            idx = torch.as_tensor(self.last_indices, device=hidden.device)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            self.values[index] = hidden[batch, idx].detach().to("cpu", dtype=torch.float16)
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self) -> Any:
        import torch

        if any(x is None for x in self.values):
            missing = [i for i, x in enumerate(self.values) if x is None]
            raise RuntimeError(f"Hooks did not capture layers {missing}")
        return torch.stack(self.values, dim=1)  # batch, layer, hidden


class FinalHeadContextCollector:
    """Capture each attention block's per-head context before its output projection."""

    def __init__(self, parts: ModelParts, last_indices: list[int], layer_indices: list[int] | None = None):
        self.last_indices = last_indices
        self.layer_indices = layer_indices or list(range(len(parts.layers)))
        self.values: list[Any] = [None] * len(self.layer_indices)
        self.handles = []
        for output_index, layer_index in enumerate(self.layer_indices):
            layer = parts.layers[layer_index]
            attention = getattr(layer, "self_attn", None)
            output_projection = getattr(attention, "o_proj", None)
            if output_projection is None:
                raise RuntimeError(f"Layer {layer_index} has no self_attn.o_proj module")
            self.handles.append(
                output_projection.register_forward_pre_hook(self._hook(output_index))
            )

    def _hook(self, index: int) -> Callable:
        def capture(_module: Any, inputs: Any) -> None:
            import torch

            hidden = inputs[0]
            idx = torch.as_tensor(self.last_indices, device=hidden.device)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            self.values[index] = hidden[batch, idx].detach().to("cpu", dtype=torch.float16)
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self) -> Any:
        import torch

        if any(x is None for x in self.values):
            missing = [i for i, x in enumerate(self.values) if x is None]
            raise RuntimeError(f"Hooks did not capture attention contexts for layers {missing}")
        return torch.stack(self.values, dim=1)  # batch, layer, concatenated heads


def cpu_lens(parts: ModelParts, residuals: Any, token_ids: list[int]) -> Any:
    """Apply the actual final norm and selected unembedding rows on CPU."""
    import torch

    # This is an observational readout. The copied norm still has trainable
    # parameters, so explicitly disable autograd before converting its output
    # to NumPy in the collector.
    with torch.inference_mode():
        norm = copy.deepcopy(parts.final_norm)
        # Models loaded with Accelerate's device_map attach execution hooks to
        # modules. Deepcopy preserves that hook, so a nominally CPU copy can
        # move its input back to the original GPU and then fail against its CPU
        # parameter. The lens copy is deliberately standalone and must not
        # retain model-dispatch behavior.
        if hasattr(norm, "_hf_hook"):
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(norm, recurse=True)
        norm = norm.float().cpu().eval()
        rows = parts.output_head.weight.detach()[token_ids].float().cpu()
        bias = getattr(parts.output_head, "bias", None)
        selected_bias = None if bias is None else bias.detach()[token_ids].float().cpu()
        batch, n_layers, hidden = residuals.shape
        normed = norm(residuals.float().reshape(batch * n_layers, hidden))
        logits = normed @ rows.T
        if selected_bias is not None:
            logits = logits + selected_bias
        return logits.reshape(batch, n_layers, len(token_ids))


def tokenize_batch(tokenizer: Any, prompts: list[str]) -> tuple[Any, Any, list[int]]:
    import torch

    encoded = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    lengths = [len(x) for x in encoded]
    if not all(lengths):
        raise RuntimeError("Rendered an empty prompt")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    width = max(lengths)
    # Left-pad decoder-only prompts so every example's prediction position is
    # the final physical column.  This is required when a model implements
    # ``logits_to_keep=1``: with right padding that optimization returns the
    # padded column for every shorter example in a variable-length batch.
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, ids in enumerate(encoded):
        start = width - len(ids)
        input_ids[i, start:] = torch.as_tensor(ids)
        attention_mask[i, start:] = 1
    return input_ids, attention_mask, [width - 1] * len(lengths)


def model_input_device(parts: ModelParts) -> Any:
    return parts.embedding.weight.device


def variant_layout(resolved: dict[str, list[tuple[str, int]]]) -> tuple[list[int], list[dict[str, Any]]]:
    ids, layout = [], []
    for li, letter in enumerate(LETTERS):
        for text, token_id in resolved[letter]:
            layout.append({"letter": letter, "letter_index": li, "text": text, "token_id": token_id})
            ids.append(token_id)
    return ids, layout
