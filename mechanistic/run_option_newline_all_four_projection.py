from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from .run_first_decision_cross_order_patching import _aggregate_logits, _decision_position
from .run_semantic_binding_module_factorial import (
    _messages,
    _remap_question,
)


LETTERS = "ABCD"
CONDITIONS = ("incorrect_again", "lost_again")
CONDITION_NAMES = ("game", "neutral")
MODES = ("natural", "identity_kv", "project_centered")
JOINT_MODES = (
    "natural",
    "identity_kv",
    "score_only",
    "decision_letter_only",
    "joint_score_and_letter",
)
EXECUTION_BATCH_SIZE = 4
GLA_CHUNK_SIZE = 64
CARRIER_BLOCKS = tuple(range(4, 65, 4))
CARRIER_READOUTS = tuple(block - 1 for block in CARRIER_BLOCKS)
TARGET_READOUTS = CARRIER_READOUTS


def _initialize(
    path: Path,
    qids: list[str],
    split: np.ndarray,
    modes: tuple[str, ...] = MODES,
) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        if arrays["question_ids"].astype(str).tolist() != qids:
            raise ValueError("Existing checkpoint uses another question order")
        return arrays
    n = len(qids)
    shape = (len(CONDITIONS), len(modes), n)
    layer_shape = shape + (64, 4)
    arrays = {
        "question_ids": np.asarray(qids),
        "split": split,
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
    if modes == JOINT_MODES:
        decision_shape = shape + (64,)
        arrays.update(
            {
                "decision_pre_ad_norm": np.full(
                    decision_shape, np.nan, dtype=np.float32
                ),
                "decision_post_ad_norm": np.full(
                    decision_shape, np.nan, dtype=np.float32
                ),
                "decision_residual_norm": np.full(
                    decision_shape, np.nan, dtype=np.float32
                ),
                "decision_dose_l2": np.full(
                    decision_shape, np.nan, dtype=np.float32
                ),
                "first_decision_logits": np.full((n, 4), np.nan, dtype=np.float32),
                "first_decision_matches_baseline": np.zeros(n, dtype=bool),
            }
        )
    return arrays


class EventOptionValueProjection:
    """Project the score when selected option-newline tokens are processed."""

    def __init__(
        self,
        parts: Any,
        selected_rows: list[int],
        selected_letters: list[int],
        weights: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
        mode: str,
    ) -> None:
        import torch

        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.selected_rows = tuple(int(value) for value in selected_rows)
        self.selected_letters = tuple(int(value) for value in selected_letters)
        if len(self.selected_rows) != len(self.selected_letters):
            raise ValueError("Selected rows and letters differ in length")
        self.weights = torch.from_numpy(weights).float().cpu()
        self.means = torch.from_numpy(means).float().cpu()
        self.scales = torch.from_numpy(scales).float().cpu()
        n_layers = len(parts.layers)
        selected = len(self.selected_rows)
        self.pre = torch.full((n_layers, selected), float("nan"))
        self.post = torch.full((n_layers, selected), float("nan"))
        self.norm = torch.full((n_layers, selected), float("nan"))
        self.dose = torch.zeros((n_layers, selected))
        # The probe cache calls the post-block output of block r "readout r".
        # Under Qwen's cached recurrent path, decoder-layer forward outputs can
        # be zero placeholders.  The identical residual is the input to block
        # r+1, so edit that live boundary with a pre-hook instead.
        self.handles = [
            parts.layers[readout].register_forward_pre_hook(
                self._pre_hook(readout - 1)
            )
            for readout in TARGET_READOUTS
        ]

    def _parameters(self, index: int, device: Any) -> tuple[Any, Any]:
        scale = self.scales[index].to(device)
        weight = self.weights[index].to(device)
        gradient = weight / scale
        gradient_norm = gradient.norm().clamp_min(1e-12)
        return gradient / gradient_norm, gradient_norm

    def _pre_hook(self, index: int):
        def intervene(_module: Any, inputs: Any) -> Any:
            import torch

            hidden = inputs[0]
            rows = torch.as_tensor(self.selected_rows, device=hidden.device)
            current = hidden[rows, -1].float()
            rms = current.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            normalized = current / rms
            unit_gradient, _gradient_norm = self._parameters(index, hidden.device)
            activation = (normalized * unit_gradient[None, :]).sum(dim=-1)
            if not torch.isfinite(activation).all():
                raise RuntimeError(
                    f"Non-finite option activation in mode={self.mode} layer={index}; "
                    f"hidden_finite={bool(torch.isfinite(current).all())} "
                    f"hidden_absmax={float(current.abs().max())}"
                )
            self.pre[index] = activation.detach().cpu()
            self.norm[index] = normalized.norm(dim=-1).detach().cpu()
            if self.mode == "natural":
                self.post[index] = activation.detach().cpu()
                return None

            changed_normalized = (
                normalized - activation[:, None] * unit_gradient[None, :]
            )
            changed = hidden.clone()
            changed[rows, -1] = (changed_normalized * rms).to(hidden.dtype)
            actual = changed[rows, -1].float()
            actual_rms = actual.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            actual_normalized = actual / actual_rms
            post = (actual_normalized * unit_gradient[None, :]).sum(dim=-1)
            self.post[index] = post.detach().cpu()
            self.dose[index] = (actual_normalized - normalized).norm(dim=-1).detach().cpu()
            return (changed, *inputs[1:])

        return intervene

    def arrays(self) -> dict[str, np.ndarray]:
        target_layers = np.asarray(TARGET_READOUTS, dtype=np.int64) - 1
        for name, value in (("pre", self.pre), ("post", self.post), ("norm", self.norm)):
            selected = value.numpy()[target_layers]
            if not np.all(np.isfinite(selected)):
                missing = np.argwhere(~np.isfinite(selected))
                raise RuntimeError(
                    f"Missing/non-finite {name} values at target readouts; "
                    f"first indices={missing[:20].tolist()} total={len(missing)}"
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

    def __enter__(self) -> "EventOptionValueProjection":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class FullOptionValueProjection:
    """Project all four option scores in the exact full-prompt forward."""

    def __init__(
        self,
        parts: Any,
        positions: list[list[int]],
        weights: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
        mode: str,
    ) -> None:
        import torch

        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.positions = torch.as_tensor(positions, dtype=torch.long)
        if self.positions.ndim != 2 or int(self.positions.shape[1]) != 4:
            raise ValueError("Expected four option positions per prompt")
        # Keep immutable probe parameters on CPU and transfer only the needed
        # readout slice to each sharded layer's actual device.
        self.weights = torch.from_numpy(weights).float().cpu()
        self.means = torch.from_numpy(means).float().cpu()
        self.scales = torch.from_numpy(scales).float().cpu()
        n_layers = len(parts.layers)
        batch = int(self.positions.shape[0])
        self.pre = torch.full((n_layers, batch, 4), float("nan"))
        self.post = torch.full((n_layers, batch, 4), float("nan"))
        self.norm = torch.full((n_layers, batch, 4), float("nan"))
        self.dose = torch.zeros((n_layers, batch, 4))
        self.handles = [
            parts.layers[readout].register_forward_pre_hook(
                self._pre_hook(readout - 1)
            )
            for readout in TARGET_READOUTS
        ]

    def _parameters(self, index: int, device: Any) -> tuple[Any, Any]:
        weight = self.weights[index].to(device)
        scales = self.scales[index].to(device)
        gradient = weight / scales
        gradient_norm = gradient.norm().clamp_min(1e-12)
        return gradient / gradient_norm, gradient_norm

    def _pre_hook(self, index: int):
        def intervene(_module: Any, inputs: Any) -> Any:
            import torch

            hidden = inputs[0]
            positions = self.positions.to(hidden.device)
            rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            current = hidden[rows, positions].float()
            rms = current.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            normalized = current / rms
            unit_gradient, _gradient_norm = self._parameters(index, hidden.device)
            activation = (normalized * unit_gradient[None, None, :]).sum(dim=-1)
            if not torch.isfinite(activation).all():
                raise RuntimeError(
                    f"Non-finite full-forward option activation mode={self.mode} "
                    f"readout={index + 1}; hidden_finite={bool(torch.isfinite(current).all())} "
                    f"hidden_absmax={float(current.abs().max())}"
                )
            self.pre[index] = activation.detach().cpu()
            self.norm[index] = normalized.norm(dim=-1).detach().cpu()
            if self.mode == "natural":
                self.post[index] = activation.detach().cpu()
                return None

            changed_normalized = (
                normalized
                - activation[..., None] * unit_gradient[None, None, :]
            )
            changed = hidden.clone()
            changed[rows, positions] = (changed_normalized * rms).to(hidden.dtype)
            actual = changed[rows, positions].float()
            actual_rms = actual.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            actual_normalized = actual / actual_rms
            post = (actual_normalized * unit_gradient[None, None, :]).sum(dim=-1)
            self.post[index] = post.detach().cpu()
            self.dose[index] = (actual_normalized - normalized).norm(dim=-1).detach().cpu()
            return (changed, *inputs[1:])

        return intervene

    def arrays(self) -> dict[str, np.ndarray]:
        target_layers = np.asarray(TARGET_READOUTS, dtype=np.int64) - 1
        for name, value in (("pre", self.pre), ("post", self.post), ("norm", self.norm)):
            selected = value.numpy()[target_layers]
            if not np.all(np.isfinite(selected)):
                raise RuntimeError(f"Missing/non-finite full-forward {name} values")
        return {
            "pre_score": self.pre.numpy(),
            "post_score": self.post.numpy(),
            "residual_norm": self.norm.numpy(),
            "dose_l2": self.dose.numpy(),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "FullOptionValueProjection":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class CompleteSequenceGLA:
    """Make every recurrent block return valid outputs for the whole prompt.

    Qwen's native long-sequence GLA kernel is allowed to retain placeholder
    outputs for old tokens because ordinary generation only consumes the last
    state.  A residual intervention at historical option tokens needs those
    states to remain live.  Running the same recurrence in contiguous chunks
    and carrying its exact final state forward produces all token outputs
    without resetting the recurrent memory.
    """

    def __init__(self, parts: Any, chunk_size: int = GLA_CHUNK_SIZE) -> None:
        self.chunk_size = int(chunk_size)
        self.originals: list[tuple[Any, Any]] = []
        for layer in parts.layers:
            module = getattr(layer, "linear_attn", None)
            if module is None:
                continue
            original = module.chunk_gated_delta_rule

            def wrapped(
                query: Any,
                key: Any,
                value: Any,
                *args: Any,
                _original=original,
                **kwargs: Any,
            ):
                if kwargs.get("initial_state") is not None:
                    return _original(query, key, value, *args, **kwargs)
                requested_final = bool(kwargs.get("output_final_state", False))
                state = None
                outputs = []
                for start in range(0, int(query.shape[1]), self.chunk_size):
                    end = min(start + self.chunk_size, int(query.shape[1]))
                    local = dict(kwargs)
                    local["g"] = kwargs["g"][:, start:end]
                    local["beta"] = kwargs["beta"][:, start:end]
                    local["initial_state"] = state
                    local["output_final_state"] = True
                    local["cu_seqlens"] = None
                    output, state = _original(
                        query[:, start:end],
                        key[:, start:end],
                        value[:, start:end],
                        *args,
                        **local,
                    )
                    if state is None:
                        raise RuntimeError("GLA chunk did not return recurrent state")
                    outputs.append(output)
                import torch

                return torch.cat(outputs, dim=1), state if requested_final else None

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped
        if len(self.originals) != 48:
            self.close()
            raise RuntimeError(
                f"Expected 48 recurrent GLA blocks, found {len(self.originals)}"
            )

    def close(self) -> None:
        for module, original in reversed(getattr(self, "originals", [])):
            module.chunk_gated_delta_rule = original
        self.originals = []

    def __enter__(self) -> "CompleteSequenceGLA":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class OptionValueKVProjection:
    """Replace option K/V entries with those from value-projected residuals."""

    @staticmethod
    def _materialized_weight(module: Any) -> Any:
        """Return a CPU weight even when Accelerate currently stores it as meta."""
        weight = module.weight
        if weight.device.type != "meta":
            return weight.detach().cpu()
        hook = getattr(module, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is None:
            raise RuntimeError("Meta parameter has no Accelerate offload map")
        # Accelerate's PrefixedDataset iterates fully-qualified keys but its
        # __getitem__ expects the module-local key and adds the prefix itself.
        try:
            materialized = weights_map["weight"]
        except KeyError as error:
            keys = [key for key in weights_map if str(key).endswith(".weight")]
            raise RuntimeError(
                f"Could not resolve the offloaded module weight from {keys}"
            ) from error
        if materialized.device.type == "meta":
            raise RuntimeError("Accelerate offload map returned another meta tensor")
        return materialized.detach().cpu()

    def __init__(
        self,
        parts: Any,
        positions: list[list[int]],
        residuals: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
        mode: str,
    ) -> None:
        import torch

        self.handles: list[Any] = []
        batch = len(positions)
        self.pre = torch.full((64, batch, 4), float("nan"))
        self.post = torch.full((64, batch, 4), float("nan"))
        self.norm = torch.full((64, batch, 4), float("nan"))
        self.dose = torch.full((64, batch, 4), float("nan"))
        position_tensor = torch.as_tensor(positions, dtype=torch.long)
        for block, readout in zip(CARRIER_BLOCKS, CARRIER_READOUTS):
            layer = parts.layers[block - 1]
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                self.close()
                raise RuntimeError(f"Block {block} is not ordinary attention")
            layer_index = readout - 1
            current = torch.from_numpy(
                np.asarray(residuals[:, layer_index], dtype=np.float32)
            )
            rms = current.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            normalized = current / rms
            weight = torch.from_numpy(weights[layer_index]).float()
            mean = torch.from_numpy(means[layer_index]).float()
            scale = torch.from_numpy(scales[layer_index]).float()
            direction = weight / scale
            direction = direction / direction.norm().clamp_min(1e-12)
            # The fitted value score is affine: each displayed letter was
            # centered by its discovery-set mean before fitting.  Remove only
            # the candidate-specific deviation along the score gradient, not
            # the static A/B/C/D mean component.
            activation = (
                (normalized - mean[None, :, :]) * direction[None, None, :]
            ).sum(dim=-1)
            if mode == "project_centered":
                carrier = normalized - activation[..., None] * direction[None, None, :]
            else:
                carrier = normalized
            base_raw = normalized * rms
            carrier_raw = carrier * rms
            post = (
                (carrier - mean[None, :, :]) * direction[None, None, :]
            ).sum(dim=-1)
            self.pre[layer_index] = activation.detach().cpu()
            self.post[layer_index] = post.detach().cpu()
            self.norm[layer_index] = normalized.norm(dim=-1).detach().cpu()
            self.dose[layer_index] = (carrier - normalized).norm(dim=-1).detach().cpu()
            if mode == "natural":
                continue
            with torch.inference_mode():
                norm_weight = self._materialized_weight(layer.input_layernorm).float()
                eps = float(
                    getattr(
                        layer.input_layernorm,
                        "variance_epsilon",
                        getattr(layer.input_layernorm, "eps", 1e-6),
                    )
                )
                base_variance = base_raw.float().square().mean(dim=-1, keepdim=True)
                base_input = (
                    base_raw.float()
                    * torch.rsqrt(base_variance + eps)
                    * norm_weight
                )
                carrier_variance = (
                    carrier_raw.float().square().mean(dim=-1, keepdim=True)
                )
                carrier_input = (
                    carrier_raw.float()
                    * torch.rsqrt(carrier_variance + eps)
                    * norm_weight
                )
                k_weight = self._materialized_weight(attention.k_proj)
                v_weight = self._materialized_weight(attention.v_proj)
                base_k = torch.nn.functional.linear(
                    base_input.to(k_weight.dtype), k_weight
                )
                base_v = torch.nn.functional.linear(
                    base_input.to(v_weight.dtype), v_weight
                )
                carrier_k = torch.nn.functional.linear(
                    carrier_input.to(k_weight.dtype), k_weight
                )
                carrier_v = torch.nn.functional.linear(
                    carrier_input.to(v_weight.dtype), v_weight
                )
                # Add only the causal projection delta to the live K/V output.
                # Absolute replacement would also inject float16 residual-cache
                # reconstruction error at every block.  Differencing the two
                # cache-derived projections cancels that nuisance exactly; the
                # identity mode therefore has an exactly zero K/V delta.
                delta_k = (carrier_k - base_k).detach()
                delta_v = (carrier_v - base_v).detach()

            def replace_k(
                _module: Any,
                _inputs: Any,
                output: Any,
                _delta=delta_k,
            ) -> Any:
                rows = torch.arange(batch, device=output.device)[:, None]
                positions = position_tensor.to(output.device)
                changed = output.clone()
                changed[rows, positions] = changed[rows, positions] + _delta.to(
                    device=output.device, dtype=output.dtype
                )
                return changed

            def replace_v(
                _module: Any,
                _inputs: Any,
                output: Any,
                _delta=delta_v,
            ) -> Any:
                rows = torch.arange(batch, device=output.device)[:, None]
                positions = position_tensor.to(output.device)
                changed = output.clone()
                changed[rows, positions] = changed[rows, positions] + _delta.to(
                    device=output.device, dtype=output.dtype
                )
                return changed

            self.handles.append(attention.k_proj.register_forward_hook(replace_k))
            self.handles.append(attention.v_proj.register_forward_hook(replace_v))

    def arrays(self) -> dict[str, np.ndarray]:
        layers = np.asarray(CARRIER_READOUTS, dtype=np.int64) - 1
        for value in (self.pre, self.post, self.norm, self.dose):
            if not np.all(np.isfinite(value.numpy()[layers])):
                raise RuntimeError("Missing/non-finite projected K/V audit values")
        return {
            "pre_score": self.pre.numpy(),
            "post_score": self.post.numpy(),
            "residual_norm": self.norm.numpy(),
            "dose_l2": self.dose.numpy(),
        }

    def close(self) -> None:
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []

    def __enter__(self) -> "OptionValueKVProjection":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _project_answer_identity(current: Any, basis: Any) -> tuple[Any, Any, Any]:
    """Orthogonally remove the complete centered A--D decoder subspace."""
    coefficients = current @ basis.T
    carrier = current - coefficients @ basis
    post_coefficients = carrier @ basis.T
    return carrier, coefficients, post_coefficients


class DecisionResidualCollector:
    """Collect exact first-decision inputs to every ordinary-attention block."""

    def __init__(self, parts: Any) -> None:
        self.values: dict[int, Any] = {}
        self.handles = []
        for block, readout in zip(CARRIER_BLOCKS, CARRIER_READOUTS):
            layer_index = readout - 1
            self.handles.append(
                parts.layers[block - 1].register_forward_pre_hook(
                    self._capture(layer_index)
                )
            )

    def _capture(self, layer_index: int):
        def capture(_module: Any, inputs: Any) -> None:
            hidden = inputs[0]
            self.values[layer_index] = hidden[:, -1].detach().float().cpu()

        return capture

    def arrays(self) -> np.ndarray:
        import torch

        result = torch.full((64, len(next(iter(self.values.values()))), 5120), float("nan"))
        for layer_index, value in self.values.items():
            result[layer_index] = value
        target = np.asarray(CARRIER_READOUTS, dtype=np.int64) - 1
        array = result.numpy().transpose(1, 0, 2)
        if not np.all(np.isfinite(array[:, target])):
            raise RuntimeError("Missing/non-finite first-decision residuals")
        return array

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []

    def __enter__(self) -> "DecisionResidualCollector":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class DecisionLetterKVProjection:
    """Remove A--D identity only from first-decision ordinary-attention K/V."""

    def __init__(
        self,
        parts: Any,
        positions: list[int],
        residuals: np.ndarray,
        bases: dict[int, Any],
        project: bool,
    ) -> None:
        import torch

        self.handles: list[Any] = []
        batch = len(positions)
        if residuals.shape[:2] != (batch, 64):
            raise ValueError(
                f"Unexpected decision residual shape {residuals.shape}; "
                f"expected ({batch}, 64, hidden)"
            )
        self.pre = torch.full((64, batch), float("nan"))
        self.post = torch.full((64, batch), float("nan"))
        self.norm = torch.full((64, batch), float("nan"))
        self.dose = torch.full((64, batch), float("nan"))
        position_tensor = torch.as_tensor(positions, dtype=torch.long)

        for block, readout in zip(CARRIER_BLOCKS, CARRIER_READOUTS):
            layer = parts.layers[block - 1]
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                self.close()
                raise RuntimeError(f"Block {block} is not ordinary attention")
            layer_index = readout - 1
            current = torch.from_numpy(
                np.asarray(residuals[:, layer_index], dtype=np.float32)
            )
            basis = bases[layer_index].float().cpu()
            if basis.shape != (3, current.shape[-1]):
                self.close()
                raise RuntimeError(
                    f"A-D basis at readout {readout} has shape {tuple(basis.shape)}"
                )
            projected, coefficients, post_coefficients = _project_answer_identity(
                current, basis
            )
            carrier = projected if project else current
            actual_post = post_coefficients if project else coefficients
            self.pre[layer_index] = coefficients.norm(dim=-1)
            self.post[layer_index] = actual_post.norm(dim=-1)
            self.norm[layer_index] = current.norm(dim=-1)
            self.dose[layer_index] = (carrier - current).norm(dim=-1)

            with torch.inference_mode():
                norm_weight = OptionValueKVProjection._materialized_weight(
                    layer.input_layernorm
                ).float()
                eps = float(
                    getattr(
                        layer.input_layernorm,
                        "variance_epsilon",
                        getattr(layer.input_layernorm, "eps", 1e-6),
                    )
                )

                def normalized_input(value: Any) -> Any:
                    variance = value.float().square().mean(dim=-1, keepdim=True)
                    return value.float() * torch.rsqrt(variance + eps) * norm_weight

                base_input = normalized_input(current)
                carrier_input = normalized_input(carrier)
                k_weight = OptionValueKVProjection._materialized_weight(attention.k_proj)
                v_weight = OptionValueKVProjection._materialized_weight(attention.v_proj)
                base_k = torch.nn.functional.linear(
                    base_input.to(k_weight.dtype), k_weight
                )
                base_v = torch.nn.functional.linear(
                    base_input.to(v_weight.dtype), v_weight
                )
                carrier_k = torch.nn.functional.linear(
                    carrier_input.to(k_weight.dtype), k_weight
                )
                carrier_v = torch.nn.functional.linear(
                    carrier_input.to(v_weight.dtype), v_weight
                )
                delta_k = (carrier_k - base_k).detach()
                delta_v = (carrier_v - base_v).detach()

            def add_k(
                _module: Any,
                _inputs: Any,
                output: Any,
                _delta=delta_k,
            ) -> Any:
                rows = torch.arange(batch, device=output.device)
                columns = position_tensor.to(output.device)
                changed = output.clone()
                changed[rows, columns] = changed[rows, columns] + _delta.to(
                    device=output.device, dtype=output.dtype
                )
                return changed

            def add_v(
                _module: Any,
                _inputs: Any,
                output: Any,
                _delta=delta_v,
            ) -> Any:
                rows = torch.arange(batch, device=output.device)
                columns = position_tensor.to(output.device)
                changed = output.clone()
                changed[rows, columns] = changed[rows, columns] + _delta.to(
                    device=output.device, dtype=output.dtype
                )
                return changed

            self.handles.append(attention.k_proj.register_forward_hook(add_k))
            self.handles.append(attention.v_proj.register_forward_hook(add_v))

    def arrays(self) -> dict[str, np.ndarray]:
        target = np.asarray(CARRIER_READOUTS, dtype=np.int64) - 1
        values = {
            "decision_pre_ad_norm": self.pre.numpy(),
            "decision_post_ad_norm": self.post.numpy(),
            "decision_residual_norm": self.norm.numpy(),
            "decision_dose_l2": self.dose.numpy(),
        }
        if any(not np.all(np.isfinite(value[target])) for value in values.values()):
            raise RuntimeError("Missing/non-finite decision-letter K/V audit values")
        return values

    def close(self) -> None:
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []

    def __enter__(self) -> "DecisionLetterKVProjection":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _trusted(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload["results"]
    if len(rows) != 500:
        raise ValueError(f"Trusted result is incomplete: {path}")
    return rows


def _load_decision_answer_bases(
    parts: Any,
    canonical_ids: list[int],
    lens_repo: str,
    lens_filename: str,
    readouts: tuple[int, ...] = CARRIER_READOUTS,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Build the same three-dimensional A--D spaces used by the prior lesion."""
    import torch
    from huggingface_hub import hf_hub_download

    lens_path = hf_hub_download(repo_id=lens_repo, filename=lens_filename)
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    output_weight = OptionValueKVProjection._materialized_weight(parts.output_head)
    norm_weight = OptionValueKVProjection._materialized_weight(parts.final_norm)
    gamma_rows = (
        output_weight[canonical_ids].float().cpu()
        * norm_weight.float().cpu()[None, :]
    )
    bases: dict[int, Any] = {}
    diagnostics: dict[str, Any] = {}
    for readout in readouts:
        layer_index = readout - 1
        if layer_index not in checkpoint["J"]:
            raise KeyError(f"JLens has no map for key {layer_index}")
        decoder = gamma_rows @ checkpoint["J"][layer_index].detach().float().cpu()
        centered = decoder - decoder.mean(dim=0, keepdim=True)
        _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:3].contiguous()
        bases[layer_index] = basis
        diagnostics[str(readout)] = {
            "jlens_key": int(layer_index),
            "rank": 3,
            "largest_singular_value": float(singular[0]),
            "smallest_nonzero_singular_value": float(singular[2]),
            "fourth_to_first_singular_ratio": float(singular[3] / singular[0]),
            "basis_orthogonality_max_error": float(
                (basis @ basis.T - torch.eye(3)).abs().max()
            ),
        }
    return bases, diagnostics


def _cached_chunk_forward(
    model: Any,
    parts: Any,
    input_ids: Any,
    attention_mask: Any,
    past_key_values: Any,
):
    """Process one physical token chunk, preserving the left-padded batch cache."""
    import torch

    device = model_input_device(parts)
    full_mask = attention_mask.to(device)
    position_ids = full_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(full_mask == 0, 1)
    position_ids = position_ids[:, -int(input_ids.shape[1]) :]
    with torch.inference_mode():
        return model(
            input_ids=input_ids.to(device),
            attention_mask=full_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )


def run(
    config_path: Path,
    discovery_plan_path: Path,
    second_mapping_path: Path,
    baseline_path: Path,
    probe_results: Path,
    residual_cache_dir: Path,
    trusted_game_path: Path,
    trusted_neutral_path: Path,
    output_dir: Path,
    max_cohorts: int | None,
    joint_factorial: bool = False,
    lens_repo: str = "neuronpedia/jacobian-lens",
    lens_filename: str = (
        "qwen3.6-27b/jlens/Salesforce-wikitext/"
        "Qwen3.6-27B_jacobian_lens_n1000.pt"
    ),
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

    manifest = json.loads(Path(config.manifest_path).read_text())["questions"]
    qids = [row["id"] for row in manifest]
    if len(qids) != 500:
        raise ValueError(f"Expected 500 questions, got {len(qids)}")
    questions = {row["id"]: row for row in manifest}
    discovery_ids = set(json.loads(discovery_plan_path.read_text())["question_ids"])
    if len(discovery_ids) != 251 or len(set(qids) - discovery_ids) != 249:
        raise ValueError("Expected the frozen 251/249 question split")
    split = np.asarray(
        ["discovery" if qid in discovery_ids else "confirmation" for qid in qids]
    )
    baseline = json.loads(baseline_path.read_text())["results"]
    second_rows = {
        row["question_id"]: row
        for row in json.loads(second_mapping_path.read_text())["rows"]
    }
    trusted = {
        "game": _trusted(trusted_game_path),
        "neutral": _trusted(trusted_neutral_path),
    }
    with np.load(probe_results, allow_pickle=False) as loaded:
        weights = loaded["weights"].astype(np.float32)
        means = loaded["letter_means"].astype(np.float32)
        scales = loaded["scales"].astype(np.float32)
    probe_sha256 = hashlib.sha256(probe_results.read_bytes()).hexdigest()
    if weights.shape != (64, 5120) or means.shape != (64, 4, 5120):
        raise ValueError("Unexpected probe coordinate shape")
    residual_cache = np.load(
        residual_cache_dir / "option_newline_residuals.npy", mmap_mode="r"
    )
    with np.load(residual_cache_dir / "results.npz", allow_pickle=False) as loaded:
        cache_qids = loaded["question_ids"].astype(str).tolist()
        if not loaded["completed"].all():
            raise ValueError("Option-newline residual cache is incomplete")
    if cache_qids != qids or residual_cache.shape[:4] != (6, 500, 64, 4):
        raise ValueError("Option-newline residual cache does not match questions")

    modes = JOINT_MODES if joint_factorial else MODES
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.npz"
    arrays = _initialize(result_path, qids, split, modes)
    if arrays["logits"].shape[1] != len(modes):
        raise ValueError(
            f"Existing checkpoint has {arrays['logits'].shape[1]} modes; "
            f"this run requires {len(modes)}"
        )

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    variant_ids = {
        letter: sorted({token_id for _, token_id in resolved[letter]})
        for letter in LETTERS
    }
    canonical_ids = [resolved[letter][0][1] for letter in LETTERS]
    decision_bases: dict[int, Any] | None = None
    decision_diagnostics: dict[str, Any] | None = None
    if joint_factorial:
        decision_bases, decision_diagnostics = _load_decision_answer_bases(
            parts, canonical_ids, lens_repo, lens_filename
        )
    audit_path = output_dir / "prompt_audit.json"
    started = time.monotonic()
    completed_cohorts = 0
    pending = {
        qid for qid, done in zip(qids, arrays["completed"]) if not bool(done)
    }
    total_cohorts = sum(
        bool(set(qids[start : start + EXECUTION_BATCH_SIZE]) & pending)
        for start in range(0, len(qids), EXECUTION_BATCH_SIZE)
    )
    cohort_durations: list[float] = []
    total_model_calls = 0

    for start in range(0, len(qids), EXECUTION_BATCH_SIZE):
        group_qids = qids[start : start + EXECUTION_BATCH_SIZE]
        if not set(group_qids) & pending:
            continue
        cohort_started = time.monotonic()
        decision_residuals: np.ndarray | None = None
        first_decision_ad_logits: np.ndarray | None = None
        canonical_prefix_rows: list[list[int]] | None = None

        for condition_index, (condition, condition_name) in enumerate(
            zip(CONDITIONS, CONDITION_NAMES)
        ):
            prompts: list[str] = []
            token_rows: list[list[int]] = []
            boundaries: list[int] = []
            unpadded_positions: list[list[int]] = []
            full_lengths: list[int] = []
            for qid in group_qids:
                question = questions[qid]
                second = _remap_question(
                    question, second_rows[qid]["new_to_original"]
                )
                messages = _messages(config, question, second, condition)
                prompt = render_chat(
                    processor,
                    messages,
                    config.disable_thinking,
                    config.chat_serialization,
                )
                positions, _audit = _positions(tokenizer, prompt, question)
                line_positions = [
                    positions[ANCHORS.index(f"line_end_{letter}")]
                    for letter in LETTERS
                ]
                ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                boundary, boundary_ids = _decision_position(tokenizer, prompt)
                if [int(value) for value in boundary_ids] != [int(value) for value in ids]:
                    raise RuntimeError(f"{qid}: boundary tokenization disagrees")
                if max(line_positions) >= int(boundary):
                    raise RuntimeError(f"{qid}: option newline follows first boundary")
                tokens = [tokenizer.decode([ids[position]]) for position in line_positions]
                if tokens != ["\n"] * 4:
                    raise RuntimeError(f"{qid}: option anchors are not newlines: {tokens}")
                expected_hash = trusted[condition_name][qid]["prompt_hash"]
                if prompt_hash(prompt) != expected_hash:
                    raise RuntimeError(
                        f"{qid}: {condition_name} prompt differs from trusted run"
                    )
                prompts.append(prompt)
                token_rows.append([int(value) for value in ids])
                boundaries.append(int(boundary))
                unpadded_positions.append(line_positions)
                full_lengths.append(len(ids))

            input_ids, attention_mask, last_indices = tokenize_batch(tokenizer, prompts)
            width = int(input_ids.shape[1])
            padded_positions = [
                [
                    int(position) + width - int(length)
                    for position in positions
                ]
                for positions, length in zip(unpadded_positions, full_lengths)
            ]
            events: dict[int, list[tuple[int, int]]] = {}
            for row, positions in enumerate(padded_positions):
                for letter_index, position in enumerate(positions):
                    events.setdefault(int(position), []).append((row, letter_index))
            event_columns = sorted(events)
            if not event_columns or event_columns[-1] >= width - 1:
                raise RuntimeError("Option events must precede the final decision token")
            if completed_cohorts == 0:
                print(
                    "all-four full-forward position audit "
                    + json.dumps(
                        {
                            "condition": condition_name,
                            "full_lengths": full_lengths,
                            "batch_width": width,
                            "unpadded_positions": unpadded_positions,
                            "padded_positions": padded_positions,
                            "event_columns": event_columns,
                            "mask_at_positions": [
                                [
                                    int(attention_mask[row, position])
                                    for position in padded_positions[row]
                                ]
                                for row in range(len(group_qids))
                            ],
                            "tokens_at_positions": [
                                [
                                    tokenizer.decode(
                                        [int(input_ids[row, position])]
                                    )
                                    for position in padded_positions[row]
                                ]
                                for row in range(len(group_qids))
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            prefix_rows = [
                ids[: boundary + 1] for ids, boundary in zip(token_rows, boundaries)
            ]
            suffix_rows = [
                ids[boundary + 1 :] for ids, boundary in zip(token_rows, boundaries)
            ]
            if not all(suffix_rows):
                raise RuntimeError("Expected a non-empty second-turn suffix")
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id

            def padded(
                rows_to_pad: list[list[int]], *, left: bool
            ) -> tuple[Any, Any]:
                local_width = max(len(values) for values in rows_to_pad)
                ids_tensor = torch.full(
                    (len(rows_to_pad), local_width), int(pad_id), dtype=torch.long
                )
                mask_tensor = torch.zeros(
                    (len(rows_to_pad), local_width), dtype=torch.long
                )
                for row_index, values in enumerate(rows_to_pad):
                    offset = local_width - len(values) if left else 0
                    ids_tensor[row_index, offset : offset + len(values)] = torch.as_tensor(
                        values
                    )
                    mask_tensor[row_index, offset : offset + len(values)] = 1
                return ids_tensor, mask_tensor

            prefix_ids, prefix_mask = padded(prefix_rows, left=True)
            suffix_ids, suffix_mask = padded(suffix_rows, left=False)
            prefix_width = int(prefix_ids.shape[1])
            prefix_positions = [
                [
                    int(position) + prefix_width - len(prefix_rows[row])
                    for position in positions
                ]
                for row, positions in enumerate(unpadded_positions)
            ]

            padded_boundaries = [
                int(boundary) + width - int(length)
                for boundary, length in zip(boundaries, full_lengths)
            ]
            if any(
                int(input_ids[row, position])
                != int(prefix_ids[row, prefix_width - 1])
                for row, position in enumerate(padded_boundaries)
            ):
                raise RuntimeError("Full-prompt and prefix decision tokens disagree")

            if joint_factorial:
                if condition_index == 0:
                    canonical_prefix_rows = [list(values) for values in prefix_rows]
                    with DecisionResidualCollector(parts) as collector:
                        with torch.inference_mode():
                            prefix_kwargs = dict(
                                input_ids=prefix_ids.to(model_input_device(parts)),
                                attention_mask=prefix_mask.to(model_input_device(parts)),
                                use_cache=False,
                                return_dict=True,
                            )
                            try:
                                prefix_output = model(**prefix_kwargs, logits_to_keep=1)
                            except TypeError:
                                prefix_output = model(**prefix_kwargs)
                    total_model_calls += 1
                    decision_residuals = collector.arrays()
                    prefix_final = (
                        prefix_output.logits[:, 0]
                        if int(prefix_output.logits.shape[1]) == 1
                        else prefix_output.logits[:, -1]
                    ).detach().float()
                    first_decision_ad_logits = (
                        _aggregate_logits(prefix_final, variant_ids).cpu().numpy()
                    )
                    for group_row, qid in enumerate(group_qids):
                        qi = qids.index(qid)
                        arrays["first_decision_logits"][qi] = first_decision_ad_logits[
                            group_row
                        ]
                        arrays["first_decision_matches_baseline"][qi] = bool(
                            LETTERS[int(np.argmax(first_decision_ad_logits[group_row]))]
                            == baseline[qid]["answer"]
                        )
                else:
                    if canonical_prefix_rows != [list(values) for values in prefix_rows]:
                        raise RuntimeError(
                            "Game and Neutral prefixes differ through first decision"
                        )
                if decision_residuals is None or decision_bases is None:
                    raise RuntimeError("Decision residual collection was not initialized")

            for mode_index, mode in enumerate(modes):
                question_indices = [qids.index(qid) for qid in group_qids]
                clean_residuals = np.asarray(
                    residual_cache[0, question_indices], dtype=np.float32
                )
                if joint_factorial:
                    option_mode = {
                        "natural": "natural",
                        "identity_kv": "identity_kv",
                        "score_only": "project_centered",
                        "decision_letter_only": "identity_kv",
                        "joint_score_and_letter": "project_centered",
                    }[mode]
                    project_letter = mode in {
                        "decision_letter_only",
                        "joint_score_and_letter",
                    }
                else:
                    option_mode = mode
                    project_letter = False

                with contextlib.ExitStack() as stack:
                    hook = stack.enter_context(
                        OptionValueKVProjection(
                            parts,
                            padded_positions,
                            clean_residuals,
                            weights,
                            means,
                            scales,
                            option_mode,
                        )
                    )
                    decision_hook = None
                    if joint_factorial:
                        decision_hook = stack.enter_context(
                            DecisionLetterKVProjection(
                                parts,
                                padded_boundaries,
                                decision_residuals,
                                decision_bases,
                                project_letter,
                            )
                        )
                    with torch.inference_mode():
                        kwargs = dict(
                            input_ids=input_ids.to(model_input_device(parts)),
                            attention_mask=attention_mask.to(model_input_device(parts)),
                            use_cache=False,
                            return_dict=True,
                        )
                        try:
                            output = model(**kwargs, logits_to_keep=1)
                        except TypeError:
                            output = model(**kwargs)
                total_model_calls += 1
                local = hook.arrays()
                decision_local = (
                    decision_hook.arrays() if decision_hook is not None else None
                )
                final = (
                    output.logits[:, 0]
                    if int(output.logits.shape[1]) == 1
                    else output.logits[:, -1]
                ).detach().float()
                if not torch.isfinite(final).all():
                    raise RuntimeError(
                        f"Non-finite final logits for {condition_name}/{mode}"
                    )
                logits = _aggregate_logits(final, variant_ids).cpu().numpy()

                target_layers = np.asarray(CARRIER_READOUTS) - 1
                for name in ("pre_score", "post_score", "residual_norm"):
                    target = local[name][target_layers]
                    if not np.all(np.isfinite(target)):
                        raise RuntimeError(f"Incomplete full-forward {name} capture")

                for group_row, qid in enumerate(group_qids):
                    qi = qids.index(qid)
                    arrays["logits"][condition_index, mode_index, qi] = logits[
                        group_row
                    ]
                    for name in (
                        "pre_score",
                        "post_score",
                        "residual_norm",
                        "dose_l2",
                    ):
                        arrays[name][condition_index, mode_index, qi] = local[name][
                            :, group_row
                        ]
                    if decision_local is not None:
                        for name in (
                            "decision_pre_ad_norm",
                            "decision_post_ad_norm",
                            "decision_residual_norm",
                            "decision_dose_l2",
                        ):
                            arrays[name][condition_index, mode_index, qi] = (
                                decision_local[name][:, group_row]
                            )
                    if mode == "natural":
                        reference = np.asarray(
                            trusted[condition_name][qid]["aggregated_ad_logits"],
                            dtype=np.float32,
                        )
                        arrays["trusted_max_abs_error"][condition_index, qi] = float(
                            np.max(np.abs(logits[group_row] - reference))
                        )
                        arrays["trusted_choice_match"][condition_index, qi] = bool(
                            int(np.argmax(logits[group_row]))
                            == int(np.argmax(reference))
                        )

                audit_mode = (
                    "joint_score_and_letter" if joint_factorial else "project_centered"
                )
                if not audit_path.exists() and mode == audit_mode:
                    audit_path.write_text(
                        json.dumps(
                            {
                                "question_id": group_qids[0],
                                "condition": condition_name,
                                "historical_group_qids": group_qids,
                                "option_newline_positions_unpadded": unpadded_positions[0],
                                "option_newline_positions_padded": padded_positions[0],
                                "option_tokens": [
                                    tokenizer.decode(
                                        [int(input_ids[0, position])]
                                    )
                                    for position in padded_positions[0]
                                ],
                                "physical_event_columns": event_columns,
                                "prompt_hash": prompt_hash(prompts[0]),
                                "target_readouts": list(TARGET_READOUTS),
                                "projection_geometry": (
                                    "RMS-normalized residual centered by the "
                                    "displayed-letter discovery mean"
                                ),
                                "pre_scores_readout_55": local["pre_score"][
                                    54, 0
                                ].tolist(),
                                "post_scores_readout_55": local["post_score"][
                                    54, 0
                                ].tolist(),
                                "max_abs_post_score_target_band": float(
                                    np.max(
                                        np.abs(
                                            local["post_score"][
                                                target_layers, 0
                                            ]
                                        )
                                    )
                                ),
                                "decision_position_unpadded": boundaries[0],
                                "decision_position_padded": padded_boundaries[0],
                                "decision_token": tokenizer.decode(
                                    [int(input_ids[0, padded_boundaries[0]])]
                                ),
                                "first_decision_ad_logits": (
                                    first_decision_ad_logits[0].tolist()
                                    if first_decision_ad_logits is not None
                                    else None
                                ),
                                "decision_pre_ad_norm_target_band": (
                                    decision_local["decision_pre_ad_norm"][
                                        target_layers, 0
                                    ].tolist()
                                    if decision_local is not None
                                    else None
                                ),
                                "decision_post_ad_norm_target_band": (
                                    decision_local["decision_post_ad_norm"][
                                        target_layers, 0
                                    ].tolist()
                                    if decision_local is not None
                                    else None
                                ),
                                "max_decision_post_ad_norm_target_band": (
                                    float(
                                        np.max(
                                            decision_local[
                                                "decision_post_ad_norm"
                                            ][target_layers, 0]
                                        )
                                    )
                                    if decision_local is not None
                                    else None
                                ),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )

        for qid in group_qids:
            arrays["completed"][qids.index(qid)] = True
        atomic_save_npz(result_path, **arrays)
        pending.difference_update(group_qids)
        completed_cohorts += 1
        duration = time.monotonic() - cohort_started
        cohort_durations.append(duration)
        elapsed = time.monotonic() - started
        remaining = total_cohorts - completed_cohorts
        eta = elapsed / completed_cohorts * remaining
        print(
            f"all-four option projection: {int(arrays['completed'].sum())}/500; "
            f"cohort={duration:.1f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )
        if max_cohorts is not None and completed_cohorts >= int(max_cohorts):
            print(f"Stopped after {max_cohorts} benchmark cohorts", flush=True)
            break

    completed_mask = arrays["completed"]
    metadata = {
        "experiment": (
            "Joint option-value and first-decision A-D identity projection"
            if joint_factorial
            else "Corrected centered all-four option-newline candidate-value projection"
        ),
        "config": config.as_dict(),
        "n_questions": len(qids),
        "split_counts": {
            "discovery": int((split == "discovery").sum()),
            "confirmation": int((split == "confirmation").sum()),
        },
        "conditions": list(CONDITION_NAMES),
        "prompt_conditions": list(CONDITIONS),
        "modes": list(modes),
        "target_readouts": list(TARGET_READOUTS),
        "ordinary_attention_carrier_blocks": list(CARRIER_BLOCKS),
        "carrier_input_readouts": list(CARRIER_READOUTS),
        "intervention_positions": (
            "all four first-presentation option-closing newlines plus the "
            "empty first-decision position"
            if joint_factorial
            else "all four first-presentation option-closing newlines"
        ),
        "intervention_geometry": (
            "At each ordinary-attention input readout, RMS-normalize each "
            "option-newline residual, subtract its discovery-set displayed-letter "
            "mean, and orthogonally project only that centered deviation out of "
            "the fitted candidate-value direction (weight divided by feature "
            "scale), then pass the resulting state "
            "through the model's own input norm and K/V projections and replace "
            "the four source-token entries in every ordinary-attention block 4-64. "
            "The identity_kv mode performs the same replacement without projection."
        ),
        "execution_batch_size": EXECUTION_BATCH_SIZE,
        "reference_batch_size": int(config.batch_size),
        "probe_provenance": {
            "path": str(probe_results),
            "sha256": probe_sha256,
            "weights_shape": list(weights.shape),
            "letter_means_shape": list(means.shape),
            "scales_shape": list(scales.shape),
        },
        "complete_prompt_executions_per_cohort": len(CONDITIONS) * len(modes),
        "prefix_executions_per_cohort": 1 if joint_factorial else 0,
        "model_forward_calls_total": total_model_calls,
        "mean_model_forward_calls_per_completed_cohort": (
            total_model_calls / completed_cohorts if completed_cohorts else None
        ),
        "execution": (
            "exact canonical left-padded batch-four full-prompt SDPA forward; "
            "ordinary-attention K/V source entries are replaced in place"
        ),
        "total_target_cohorts": total_cohorts,
        "complete": bool(arrays["completed"].all()),
        "natural_validation": {
            "max_abs_trusted_logit_error": float(
                np.nanmax(arrays["trusted_max_abs_error"][:, completed_mask])
            ),
            "trusted_choice_agreement": float(
                arrays["trusted_choice_match"][:, completed_mask].mean()
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
        "decision_letter_jlens": (
            {
                "repo": lens_repo,
                "filename": lens_filename,
                "diagnostics": decision_diagnostics,
                "geometry": (
                    "Orthogonally remove the complete centered three-dimensional "
                    "A-D JLens decoder subspace from the first-decision residual, "
                    "then add only the resulting K/V delta at every ordinary-"
                    "attention block."
                ),
            }
            if joint_factorial
            else None
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-plan", type=Path, required=True)
    parser.add_argument("--second-mapping", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--probe-results", type=Path, required=True)
    parser.add_argument("--residual-cache-dir", type=Path, required=True)
    parser.add_argument("--trusted-game", type=Path, required=True)
    parser.add_argument("--trusted-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--joint-factorial", action="store_true")
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default=(
            "qwen3.6-27b/jlens/Salesforce-wikitext/"
            "Qwen3.6-27B_jacobian_lens_n1000.pt"
        ),
    )
    args = parser.parse_args()
    run(
        args.config,
        args.discovery_plan,
        args.second_mapping,
        args.baseline,
        args.probe_results,
        args.residual_cache_dir,
        args.trusted_game,
        args.trusted_neutral,
        args.output_dir,
        args.max_cohorts,
        args.joint_factorial,
        args.lens_repo,
        args.lens_filename,
    )


if __name__ == "__main__":
    main()
