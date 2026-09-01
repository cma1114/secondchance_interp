from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def linear_attention_layers(parts: Any) -> list[int]:
    return [
        index for index, layer in enumerate(parts.layers)
        if getattr(layer, "linear_attn", None) is not None
    ]


@dataclass(frozen=True)
class GDNTarget:
    layer: int
    heads: tuple[int, ...] | None = None


def zero_beta_writes(beta: Any, positions: tuple[int, ...], heads: tuple[int, ...] | None = None) -> Any:
    import torch

    if not positions:
        raise ValueError("At least one source position is required")
    result = beta.clone()
    position_index = torch.as_tensor(positions, device=result.device)
    if heads is None:
        result[:, position_index, :] = 0
    else:
        head_index = torch.as_tensor(heads, device=result.device)
        result[:, position_index[:, None], head_index[None, :]] = 0
    return result


class BetaWriteAblator:
    """Set beta=0 only for selected token writes in selected Gated DeltaNet heads."""

    def __init__(self, parts: Any, targets: list[GDNTarget], positions: list[int]) -> None:
        self.originals: list[tuple[Any, Any]] = []
        source = tuple(int(position) for position in positions)
        if not targets:
            raise ValueError("No GDN targets were selected")
        for target in targets:
            module = getattr(parts.layers[target.layer], "linear_attn", None)
            if module is None:
                self.close()
                raise ValueError(f"Layer {target.layer} is not a Gated DeltaNet block")
            if target.heads is not None and any(
                head < 0 or head >= module.num_v_heads for head in target.heads
            ):
                self.close()
                raise ValueError(f"Invalid value head at layer {target.layer}")
            original = module.chunk_gated_delta_rule
            heads = target.heads

            def wrapped(*args: Any, _original=original, _heads=heads, **kwargs: Any):
                if "beta" not in kwargs:
                    raise RuntimeError("Qwen Gated DeltaNet did not pass beta by keyword")
                kwargs["beta"] = zero_beta_writes(kwargs["beta"], source, _heads)
                return _original(*args, **kwargs)

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def close(self) -> None:
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()

    def __enter__(self) -> "BetaWriteAblator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

