from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class HeadTarget:
    layer: int
    heads: tuple[int, ...]


class FinalHeadContextPatcher:
    """Replace selected final-position attention-head contexts before o_proj."""

    def __init__(
        self,
        parts: Any,
        targets: list[HeadTarget],
        source: dict[int, Any],
        last_indices: list[int],
    ) -> None:
        self.last_indices = last_indices
        self.handles = []
        for target in targets:
            if target.layer not in source:
                raise KeyError(f"No cached head context for layer {target.layer}")
            attention = getattr(parts.layers[target.layer], "self_attn", None)
            projection = getattr(attention, "o_proj", None)
            if projection is None:
                raise ValueError(f"Layer {target.layer} has no ordinary attention o_proj")
            n_heads = getattr(attention, "num_heads", None)
            if n_heads is None:
                q_projection = getattr(attention, "q_proj", None)
                head_dim = getattr(attention, "head_dim", None)
                if q_projection is None or head_dim is None:
                    raise RuntimeError(f"Cannot infer head count at layer {target.layer}")
                # Qwen3.6 concatenates query and per-head gate projections.
                n_heads = int(q_projection.out_features // (2 * head_dim))
            self.handles.append(
                projection.register_forward_pre_hook(
                    self._hook(target, source[target.layer], int(n_heads))
                )
            )

    def _hook(self, target: HeadTarget, source: Any, n_heads: int) -> Callable:
        def patch(_module: Any, inputs: Any) -> tuple[Any, ...]:
            import torch

            hidden = inputs[0]
            if hidden.ndim != 3:
                raise RuntimeError(f"Expected [batch, sequence, heads*dim], got {hidden.shape}")
            batch_size, _, width = hidden.shape
            replacement = source.to(device=hidden.device, dtype=hidden.dtype)
            if replacement.shape != (batch_size, width):
                raise RuntimeError(
                    f"Source context shape {replacement.shape} does not match {(batch_size, width)}"
                )
            attention = getattr(_module, "in_features", width)
            if int(attention) != width:
                raise RuntimeError("Unexpected attention output-projection input width")
            if width % n_heads:
                raise RuntimeError(f"Attention width {width} is not divisible by {n_heads} heads")
            head_dim = width // n_heads
            if any(head < 0 or head >= n_heads for head in target.heads):
                raise ValueError(f"Invalid head in {target.heads}; expected 0..{n_heads - 1}")
            updated = hidden.clone().reshape(batch_size, hidden.shape[1], n_heads, head_dim)
            replacement = replacement.reshape(batch_size, n_heads, head_dim)
            indices = torch.as_tensor(self.last_indices, device=hidden.device)
            rows = torch.arange(batch_size, device=hidden.device)
            heads = torch.as_tensor(target.heads, device=hidden.device)
            updated[rows[:, None], indices[:, None], heads[None, :], :] = replacement[:, heads, :]
            return (updated.reshape_as(hidden), *inputs[1:])

        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "FinalHeadContextPatcher":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class BatchedSingleHeadContextPatcher:
    """Patch a different single head in each batch row at one attention layer."""

    def __init__(
        self,
        parts: Any,
        layer: int,
        heads_by_row: list[int],
        source: Any,
        last_indices: list[int],
    ) -> None:
        import torch

        attention = getattr(parts.layers[layer], "self_attn", None)
        projection = getattr(attention, "o_proj", None)
        if projection is None:
            raise ValueError(f"Layer {layer} has no ordinary attention o_proj")
        n_heads = getattr(attention, "num_heads", None)
        if n_heads is None:
            n_heads = int(attention.q_proj.out_features // (2 * attention.head_dim))
        self.n_heads = int(n_heads)
        if any(head < 0 or head >= self.n_heads for head in heads_by_row):
            raise ValueError(f"Invalid head selection for {self.n_heads} heads")
        if len(heads_by_row) != len(last_indices):
            raise ValueError("Need one selected head and final position per batch row")
        self.heads = torch.as_tensor(heads_by_row)
        self.source = source
        self.last_indices = last_indices
        self.handle = projection.register_forward_pre_hook(self._hook)

    def _hook(self, _module: Any, inputs: Any) -> tuple[Any, ...]:
        import torch

        hidden = inputs[0]
        batch, sequence, width = hidden.shape
        if width % self.n_heads:
            raise RuntimeError("Attention width is not divisible by the head count")
        head_dim = width // self.n_heads
        source = self.source.to(device=hidden.device, dtype=hidden.dtype)
        if source.shape == (1, width):
            source = source.expand(batch, -1)
        if source.shape != (batch, width):
            raise RuntimeError(f"Source shape {source.shape} does not match {(batch, width)}")
        updated = hidden.clone().reshape(batch, sequence, self.n_heads, head_dim)
        source = source.reshape(batch, self.n_heads, head_dim)
        rows = torch.arange(batch, device=hidden.device)
        positions = torch.as_tensor(self.last_indices, device=hidden.device)
        heads = self.heads.to(hidden.device)
        updated[rows, positions, heads, :] = source[rows, heads, :]
        return (updated.reshape_as(hidden), *inputs[1:])

    def close(self) -> None:
        self.handle.remove()


class BatchedScenarioHeadContextPatcher:
    """Patch an arbitrary set of heads in each batch row at one or more layers."""

    def __init__(
        self,
        parts: Any,
        targets_by_row: list[list[HeadTarget]],
        source: dict[int, Any],
        last_indices: list[int],
    ) -> None:
        if len(targets_by_row) != len(last_indices):
            raise ValueError("Need one target set and final position per batch row")
        self.last_indices = last_indices
        self.source = source
        by_layer: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for row, targets in enumerate(targets_by_row):
            for target in targets:
                by_layer.setdefault(target.layer, []).append((row, target.heads))
        self.handles = []
        for layer, selections in by_layer.items():
            attention = getattr(parts.layers[layer], "self_attn", None)
            projection = getattr(attention, "o_proj", None)
            if projection is None:
                raise ValueError(f"Layer {layer} has no ordinary attention o_proj")
            n_heads = getattr(attention, "num_heads", None)
            if n_heads is None:
                n_heads = int(attention.q_proj.out_features // (2 * attention.head_dim))
            if layer not in source:
                raise KeyError(f"No cached head context for layer {layer}")
            self.handles.append(
                projection.register_forward_pre_hook(
                    self._hook(layer, selections, int(n_heads))
                )
            )

    def _hook(
        self,
        layer: int,
        selections: list[tuple[int, tuple[int, ...]]],
        n_heads: int,
    ) -> Callable:
        def patch(_module: Any, inputs: Any) -> tuple[Any, ...]:
            import torch

            hidden = inputs[0]
            batch, sequence, width = hidden.shape
            if width % n_heads:
                raise RuntimeError("Attention width is not divisible by the head count")
            head_dim = width // n_heads
            source = self.source[layer].to(device=hidden.device, dtype=hidden.dtype)
            if source.shape == (1, width):
                source = source.expand(batch, -1)
            if source.shape != (batch, width):
                raise RuntimeError(f"Source shape {source.shape} does not match {(batch, width)}")
            updated = hidden.clone().reshape(batch, sequence, n_heads, head_dim)
            source = source.reshape(batch, n_heads, head_dim)
            for row, heads in selections:
                if any(head < 0 or head >= n_heads for head in heads):
                    raise ValueError(f"Invalid head in {heads}; expected 0..{n_heads - 1}")
                head_indices = torch.as_tensor(heads, device=hidden.device)
                updated[row, self.last_indices[row], head_indices, :] = source[row, head_indices, :]
            return (updated.reshape_as(hidden), *inputs[1:])

        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
