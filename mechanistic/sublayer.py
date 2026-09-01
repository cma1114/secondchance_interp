from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def mixer_module(layer: Any) -> Any:
    for name in ("self_attn", "linear_attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise RuntimeError(f"Layer {type(layer).__name__} has neither self_attn nor linear_attn")


def middle_norm(layer: Any) -> Any:
    for name in ("post_attention_layernorm", "post_attn_layernorm"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise RuntimeError(f"Layer {type(layer).__name__} has no post-attention layer norm")


def component_module(layer: Any, kind: str) -> Any:
    if kind == "mixer":
        return mixer_module(layer)
    if kind == "mlp":
        module = getattr(layer, "mlp", None)
        if module is None:
            raise RuntimeError(f"Layer {type(layer).__name__} has no MLP")
        return module
    raise ValueError(f"Unknown component kind: {kind}")


class SublayerBoundaryCollector:
    """Capture actual residual boundaries before mixer, before MLP, and after MLP."""

    def __init__(self, parts: Any, last_indices: list[int]):
        self.last_indices = last_indices
        self.values: list[list[Any]] = [[None, None, None] for _ in parts.layers]
        self.handles = []
        for index, layer in enumerate(parts.layers):
            self.handles.append(layer.register_forward_pre_hook(self._pre_hook(index)))
            self.handles.append(middle_norm(layer).register_forward_pre_hook(self._mid_hook(index)))
            self.handles.append(layer.register_forward_hook(self._post_hook(index)))

    def _select(self, hidden: Any) -> Any:
        import torch

        idx = torch.as_tensor(self.last_indices, device=hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, idx].detach().to("cpu", dtype=torch.float16)

    def _pre_hook(self, index: int) -> Callable:
        def capture(_module: Any, inputs: Any) -> None:
            self.values[index][0] = self._select(inputs[0])
        return capture

    def _mid_hook(self, index: int) -> Callable:
        def capture(_module: Any, inputs: Any) -> None:
            self.values[index][1] = self._select(inputs[0])
        return capture

    def _post_hook(self, index: int) -> Callable:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            self.values[index][2] = self._select(_hidden(output))
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def stacked(self) -> Any:
        import torch

        missing = [
            (layer, boundary)
            for layer, row in enumerate(self.values)
            for boundary, value in enumerate(row)
            if value is None
        ]
        if missing:
            raise RuntimeError(f"Missing sublayer boundaries: {missing}")
        return torch.stack([torch.stack(row, dim=1) for row in self.values], dim=1)


@dataclass(frozen=True)
class ComponentTarget:
    layer: int
    kind: str

    @property
    def key(self) -> str:
        return f"{self.kind}_l{self.layer}"


class ComponentOutputCollector:
    """Capture final-position output vectors from selected mixer/MLP modules."""

    def __init__(self, parts: Any, targets: list[ComponentTarget], last_indices: list[int]):
        self.last_indices = last_indices
        self.values: dict[str, Any] = {}
        self.handles = []
        for target in targets:
            module = component_module(parts.layers[target.layer], target.kind)
            self.handles.append(module.register_forward_hook(self._hook(target.key)))

    def _hook(self, key: str) -> Callable:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = _hidden(output)
            idx = torch.as_tensor(self.last_indices, device=hidden.device)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            self.values[key] = hidden[batch, idx].detach().to("cpu", dtype=torch.float16)
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class ComponentOutputPatcher:
    """Replace selected modules' final-position outputs with paired source outputs."""

    def __init__(self, parts: Any, targets: list[ComponentTarget], source: dict[str, Any], last_indices: list[int]):
        self.last_indices = last_indices
        self.handles = []
        for target in targets:
            if target.key not in source:
                raise KeyError(f"No cached source for {target.key}")
            module = component_module(parts.layers[target.layer], target.kind)
            self.handles.append(module.register_forward_hook(self._hook(target.key, source[target.key])))

    def _hook(self, key: str, source: Any) -> Callable:
        def patch(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            hidden = _hidden(output)
            replacement = source.to(device=hidden.device, dtype=hidden.dtype)
            updated = hidden.clone()
            idx = torch.as_tensor(self.last_indices, device=hidden.device)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            updated[batch, idx] = replacement
            return _replace_hidden(output, updated)
        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "ComponentOutputPatcher":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@dataclass(frozen=True)
class PositionComponentTarget:
    """A mixer or MLP output at a semantically anchored prompt position."""

    layer: int
    kind: str
    anchor: str

    @property
    def component_key(self) -> str:
        return f"{self.kind}_l{self.layer}"

    @property
    def source_key(self) -> str:
        return f"{self.anchor}__{self.component_key}"


class PositionComponentOutputCollector:
    """Capture selected component outputs at one or more prompt anchors."""

    def __init__(
        self,
        parts: Any,
        targets: list[PositionComponentTarget],
        positions: dict[str, int],
    ) -> None:
        self.positions = positions
        self.values: dict[str, Any] = {}
        grouped: dict[tuple[int, str], list[PositionComponentTarget]] = {}
        for target in targets:
            if target.anchor not in positions:
                raise KeyError(f"No token position for anchor {target.anchor!r}")
            grouped.setdefault((target.layer, target.kind), []).append(target)
        self.handles = []
        for (layer, kind), selections in grouped.items():
            module = component_module(parts.layers[layer], kind)
            self.handles.append(module.register_forward_hook(self._hook(selections)))

    def _hook(self, selections: list[PositionComponentTarget]) -> Callable:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            hidden = _hidden(output)
            if hidden.shape[0] != 1:
                raise ValueError("Natural position-component collection expects batch size one")
            for target in selections:
                self.values[target.source_key] = (
                    hidden[0, self.positions[target.anchor]]
                    .detach()
                    .to("cpu", dtype=torch.float16)
                    .unsqueeze(0)
                )
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class BatchedPositionComponentOutputPatcher:
    """Patch arbitrary anchored component outputs independently by batch row."""

    def __init__(
        self,
        parts: Any,
        targets_by_row: list[list[PositionComponentTarget]],
        source: dict[str, Any],
        positions: dict[str, int],
    ) -> None:
        self.source = source
        self.positions = positions
        grouped: dict[tuple[int, str], list[tuple[int, PositionComponentTarget]]] = {}
        for row, targets in enumerate(targets_by_row):
            for target in targets:
                if target.anchor not in positions:
                    raise KeyError(f"No token position for anchor {target.anchor!r}")
                if target.source_key not in source:
                    raise KeyError(f"No paired source output for {target.source_key}")
                grouped.setdefault((target.layer, target.kind), []).append((row, target))
        self.handles = []
        for (layer, kind), selections in grouped.items():
            module = component_module(parts.layers[layer], kind)
            self.handles.append(module.register_forward_hook(self._hook(selections)))

    def _hook(
        self,
        selections: list[tuple[int, PositionComponentTarget]],
    ) -> Callable:
        def patch(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = _hidden(output)
            updated = hidden.clone()
            for row, target in selections:
                replacement = self.source[target.source_key].to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                if replacement.shape == (1, hidden.shape[-1]):
                    replacement = replacement[0]
                if replacement.shape != (hidden.shape[-1],):
                    raise RuntimeError(
                        f"Source shape {tuple(replacement.shape)} does not match "
                        f"component width {(hidden.shape[-1],)}"
                    )
                updated[row, self.positions[target.anchor]] = replacement
            return _replace_hidden(output, updated)
        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class BatchedRowSourcePositionComponentOutputPatcher:
    """Patch anchored component outputs from a row-specific source cache.

    This is the conditional-route analogue of
    :class:`BatchedPositionComponentOutputPatcher`: every intervention row can
    receive the same set of component targets while differing in one cached
    source vector (for example, a Mixer-56 output with one attention route
    removed).  Keeping the rows in a single fixed-size forward controls Qwen's
    batch-shape numerical drift.
    """

    def __init__(
        self,
        parts: Any,
        targets_by_row: list[list[PositionComponentTarget]],
        sources_by_row: list[dict[str, Any]],
        positions: dict[str, int],
    ) -> None:
        if len(targets_by_row) != len(sources_by_row):
            raise ValueError("targets_by_row and sources_by_row must have equal length")
        self.positions = positions
        grouped: dict[
            tuple[int, str], list[tuple[int, PositionComponentTarget, Any]]
        ] = {}
        for row, (targets, source) in enumerate(
            zip(targets_by_row, sources_by_row)
        ):
            for target in targets:
                if target.anchor not in positions:
                    raise KeyError(f"No token position for anchor {target.anchor!r}")
                if target.source_key not in source:
                    raise KeyError(
                        f"No row-{row} source output for {target.source_key}"
                    )
                grouped.setdefault((target.layer, target.kind), []).append(
                    (row, target, source[target.source_key])
                )
        self.handles = []
        for (layer, kind), selections in grouped.items():
            module = component_module(parts.layers[layer], kind)
            self.handles.append(module.register_forward_hook(self._hook(selections)))

    def _hook(
        self,
        selections: list[tuple[int, PositionComponentTarget, Any]],
    ) -> Callable:
        def patch(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = _hidden(output)
            updated = hidden.clone()
            for row, target, source in selections:
                replacement = source.to(device=hidden.device, dtype=hidden.dtype)
                if replacement.shape == (1, hidden.shape[-1]):
                    replacement = replacement[0]
                if replacement.shape != (hidden.shape[-1],):
                    raise RuntimeError(
                        f"Source shape {tuple(replacement.shape)} does not match "
                        f"component width {(hidden.shape[-1],)}"
                    )
                updated[row, self.positions[target.anchor]] = replacement
            return _replace_hidden(output, updated)

        return patch

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
