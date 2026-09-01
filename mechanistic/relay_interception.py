from __future__ import annotations

import inspect
from typing import Any


class BatchedSDPARelayWriteCache:
    """Record clean ordinary-attention K/V at selected relay positions.

    The cache is collected in an otherwise unmodified forward pass.  Keeping
    the original batch geometry avoids the substantial numerical drift caused
    by replacing the other questions in a canonical batch with duplicates.
    """

    def __init__(
        self,
        parts: Any,
        positions_by_row: dict[int, list[int]],
        layers: list[int],
    ) -> None:
        import torch

        if not positions_by_row or not layers:
            raise ValueError("Relay cache needs rows, positions, and layers")
        self.positions = {
            int(row): tuple(sorted(set(int(value) for value in positions)))
            for row, positions in positions_by_row.items()
        }
        if any(not values for values in self.positions.values()):
            raise ValueError("Every cached row needs relay positions")
        self.layers = sorted(set(int(value) for value in layers))
        self.cache: dict[int, dict[int, tuple[tuple[int, ...], Any, Any]]] = {}
        self.active_layer: int | None = None
        self.handles: list[Any] = []
        for layer_index in self.layers:
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not ordinary attention")
            self.handles.append(attention.register_forward_pre_hook(self._enter(layer_index)))
            self.handles.append(attention.register_forward_hook(self._leave(layer_index)))
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        layer = self.active_layer
        if layer is not None and layer in self.layers:
            if layer in self.cache:
                raise RuntimeError(f"Ordinary relay layer {layer} was cached twice")
            rows: dict[int, tuple[tuple[int, ...], Any, Any]] = {}
            for row, positions in self.positions.items():
                if row >= key.shape[0] or max(positions) >= key.shape[-2]:
                    raise RuntimeError("Ordinary relay cache index exceeds batch or sequence")
                selected = list(positions)
                rows[row] = (
                    positions,
                    key[row, :, selected, :].detach().clone(),
                    value[row, :, selected, :].detach().clone(),
                )
            self.cache[layer] = rows
        return self.original_sdpa(query, key, value, *args, **kwargs)

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None


class BatchedSDPACachedRelayRestorer:
    """Restore ordinary-attention relay K/V from a clean same-row cache.

    This is an identity control for cached relay interception.  Unlike the
    source-crossing interceptor, it performs no donor swap: each row receives
    its own clean cached relay K/V in an otherwise natural forward.
    """

    def __init__(
        self,
        parts: Any,
        positions_by_row: dict[int, list[int]],
        layers: list[int],
        clean_cache: dict[int, dict[int, tuple[tuple[int, ...], Any, Any]]],
    ) -> None:
        import torch

        self.positions = {
            int(row): tuple(sorted(set(int(value) for value in positions)))
            for row, positions in positions_by_row.items()
        }
        self.layers = sorted(set(int(value) for value in layers))
        if not self.positions or any(not values for values in self.positions.values()):
            raise ValueError("Cached relay restoration needs nonempty row positions")
        missing = set(self.layers) - set(clean_cache)
        if missing:
            raise RuntimeError(f"Clean ordinary cache lacks layers: {sorted(missing)}")
        self.clean_cache = clean_cache
        self.active_layer: int | None = None
        self.handles: list[Any] = []
        self.calls = 0
        self.restored_position_count = 0
        self.layers_seen: set[int] = set()
        for layer_index in self.layers:
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not ordinary attention")
            self.handles.append(attention.register_forward_pre_hook(self._enter(layer_index)))
            self.handles.append(attention.register_forward_hook(self._leave(layer_index)))
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        layer = self.active_layer
        if layer is None or layer not in self.layers:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        self.calls += 1
        self.layers_seen.add(layer)
        restored_key = key.clone()
        restored_value = value.clone()
        for row, positions in self.positions.items():
            cached_positions, cached_key, cached_value = self.clean_cache[layer][row]
            indices = [cached_positions.index(value) for value in positions]
            selected = list(positions)
            restored_key[row, :, selected, :] = cached_key[:, indices, :]
            restored_value[row, :, selected, :] = cached_value[:, indices, :]
            self.restored_position_count += len(selected)
        return self.original_sdpa(
            query, restored_key, restored_value, *args, **kwargs
        )

    def assert_fired(self) -> None:
        if self.calls <= 0 or self.restored_position_count <= 0:
            raise RuntimeError("Cached ordinary relay restorer did not restore any position")
        missing = set(self.layers) - self.layers_seen
        if missing:
            raise RuntimeError(f"Cached ordinary relay restorer missed layers: {sorted(missing)}")

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None


class BatchedSDPACachedRelayDownstreamRestorer(BatchedSDPACachedRelayRestorer):
    """Restore cached relay K/V only for downstream readers.

    The ordinary cached restorer is the correct identity control, but in a
    perturbed forward it also lets a relay query attend to its own clean K/V.
    That changes the relay token's local output.  Mediation instead requires
    preserving the relay token's perturbed computation while replacing the K/V
    that it exposes to causally later queries.  This subclass evaluates both
    paths and copies the perturbed output back at every restored relay query.

    It composes with a previously installed SDPA intervention such as
    ``BatchedSDPAQuerySourceAttentionAblator``: construct that source
    intervention first, then this downstream restorer.
    """

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        layer = self.active_layer
        if layer is None or layer not in self.layers:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        self.calls += 1
        self.layers_seen.add(layer)
        perturbed = self.original_sdpa(query, key, value, *args, **kwargs)
        restored_key = key.clone()
        restored_value = value.clone()
        for row, positions in self.positions.items():
            cached_positions, cached_key, cached_value = self.clean_cache[layer][row]
            indices = [cached_positions.index(position) for position in positions]
            selected = list(positions)
            restored_key[row, :, selected, :] = cached_key[:, indices, :]
            restored_value[row, :, selected, :] = cached_value[:, indices, :]
            self.restored_position_count += len(selected)
        downstream = self.original_sdpa(
            query, restored_key, restored_value, *args, **kwargs
        )
        output = downstream.clone()
        for row, positions in self.positions.items():
            selected = list(positions)
            output[row, :, selected, :] = perturbed[row, :, selected, :]
        return output


class BatchedSDPACachedRelayInterceptor:
    """Cross feedback K/V, then restore relay K/V from a clean cache."""

    def __init__(
        self,
        parts: Any,
        specs_by_target_row: dict[int, tuple[int, list[int], list[int], list[int]]],
        clean_cache: dict[int, dict[int, tuple[tuple[int, ...], Any, Any]]],
    ) -> None:
        import torch

        if not specs_by_target_row:
            raise ValueError("No cached source/relay K/V interceptions were specified")
        self.by_layer: dict[int, dict[int, tuple[int, tuple[int, ...], tuple[int, ...]]]] = {}
        for target, (source_donor, source_positions, relay_positions, layers) in specs_by_target_row.items():
            source = tuple(sorted(set(int(value) for value in source_positions)))
            relay = tuple(sorted(set(int(value) for value in relay_positions)))
            if not source or not relay or not layers:
                raise ValueError("Every interception needs source positions, relay positions, and layers")
            if int(target) == int(source_donor):
                raise ValueError("Target and source donor must differ")
            if max(source) >= min(relay):
                raise ValueError("Every relay position must causally follow the feedback source")
            for layer in sorted(set(int(value) for value in layers)):
                self.by_layer.setdefault(layer, {})[int(target)] = (
                    int(source_donor), source, relay
                )
        self.clean_cache = clean_cache
        missing = set(self.by_layer) - set(clean_cache)
        if missing:
            raise RuntimeError(f"Clean ordinary cache lacks layers: {sorted(missing)}")
        self.active_layer: int | None = None
        self.handles: list[Any] = []
        for layer_index in sorted(self.by_layer):
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not ordinary attention")
            self.handles.append(attention.register_forward_pre_hook(self._enter(layer_index)))
            self.handles.append(attention.register_forward_hook(self._leave(layer_index)))
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        layer = self.active_layer
        if layer is None or layer not in self.by_layer:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        natural = self.original_sdpa(query, key, value, *args, **kwargs)
        source_key = key.clone()
        source_value = value.clone()
        for target, (donor, source_positions, _relay) in self.by_layer[layer].items():
            selected = list(source_positions)
            source_key[target, :, selected, :] = key[donor, :, selected, :]
            source_value[target, :, selected, :] = value[donor, :, selected, :]
        source_edited = self.original_sdpa(query, source_key, source_value, *args, **kwargs)
        intercepted_key = source_key.clone()
        intercepted_value = source_value.clone()
        for target, (_donor, _source, relay_positions) in self.by_layer[layer].items():
            selected = list(relay_positions)
            cached_positions, all_cached_key, all_cached_value = self.clean_cache[layer][target]
            cache_indices = [cached_positions.index(value) for value in relay_positions]
            cached_key = all_cached_key[:, cache_indices, :]
            cached_value = all_cached_value[:, cache_indices, :]
            if cached_key.shape != intercepted_key[target, :, selected, :].shape:
                raise RuntimeError("Cached ordinary relay K shape changed")
            intercepted_key[target, :, selected, :] = cached_key
            intercepted_value[target, :, selected, :] = cached_value
        intercepted = self.original_sdpa(query, intercepted_key, intercepted_value, *args, **kwargs)
        output = natural.clone()
        for target, (_donor, source_positions, relay_positions) in self.by_layer[layer].items():
            cutoff = max(source_positions)
            if cutoff + 1 < output.shape[-2]:
                output[target, :, cutoff + 1 :, :] = intercepted[target, :, cutoff + 1 :, :]
            output[target, :, list(relay_positions), :] = source_edited[target, :, list(relay_positions), :]
        return output

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None


def _combine_gla_results(
    natural_result: Any,
    source_result: Any,
    intercepted_result: Any,
    specs: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
) -> Any:
    """Preserve source/relay token outputs while intercepting their later writes."""
    if not all(isinstance(result, (tuple, list)) and len(result) == 2 for result in (
        natural_result, source_result, intercepted_result
    )):
        raise RuntimeError("Expected every Qwen GLA rule call to return output and state")
    natural_output, natural_state = natural_result
    source_output, source_state = source_result
    intercepted_output, intercepted_state = intercepted_result
    output = natural_output.clone()
    for row, (source_positions, relay_positions) in specs.items():
        cutoff = max(source_positions)
        if cutoff + 1 < output.shape[1]:
            output[row, cutoff + 1 :] = intercepted_output[row, cutoff + 1 :]
        output[row, list(relay_positions)] = source_output[row, list(relay_positions)]
    state = natural_state
    if any(value is not None for value in (natural_state, source_state, intercepted_state)):
        if any(value is None for value in (natural_state, source_state, intercepted_state)):
            raise RuntimeError("Natural/source/intercepted GLA calls returned inconsistent states")
        state = natural_state.clone()
        for row in specs:
            state[row] = intercepted_state[row]
    result = (output, state)
    return result if isinstance(natural_result, tuple) else list(result)


class BatchedGLARelayWriteCache:
    """Record clean GLA key/value/gate/write tensors at relay positions."""

    def __init__(
        self,
        parts: Any,
        positions_by_row: dict[int, list[int]],
        layers: list[int],
    ) -> None:
        if not positions_by_row or not layers:
            raise ValueError("GLA relay cache needs rows, positions, and layers")
        self.positions = {
            int(row): tuple(sorted(set(int(value) for value in positions)))
            for row, positions in positions_by_row.items()
        }
        if any(not values for values in self.positions.values()):
            raise ValueError("Every cached row needs relay positions")
        self.layers = sorted(set(int(value) for value in layers))
        self.cache: dict[
            int, dict[int, tuple[tuple[int, ...], Any, Any, Any, Any]]
        ] = {}
        self.originals: list[tuple[Any, Any]] = []
        self.hook_handles: list[Any] = []
        self.modeling_module: Any | None = None
        self.original_global_rule: Any | None = None
        self.active_layer: int | None = None
        selected_modules = {
            layer_index: getattr(parts.layers[layer_index], "linear_attn", None)
            for layer_index in self.layers
        }
        use_global_rule = bool(selected_modules) and all(
            module is not None and not hasattr(module, "chunk_gated_delta_rule")
            for module in selected_modules.values()
        )
        if use_global_rule:
            modeling_modules = {
                inspect.getmodule(type(module)) for module in selected_modules.values()
            }
            if len(modeling_modules) != 1 or None in modeling_modules:
                raise RuntimeError("Could not identify one shared Qwen GLA modeling module")
            self.modeling_module = modeling_modules.pop()
            if not hasattr(self.modeling_module, "torch_chunk_gated_delta_rule"):
                raise RuntimeError("Qwen modeling module has no chunk GLA rule")
            self.original_global_rule = self.modeling_module.torch_chunk_gated_delta_rule
            for layer_index, module in selected_modules.items():
                def mark_active(_module: Any, _args: Any, _layer=layer_index) -> None:
                    if self.active_layer is not None:
                        raise RuntimeError("Nested Qwen GLA forward calls are unsupported")
                    self.active_layer = _layer
                def clear_active(_module: Any, _args: Any, _output: Any) -> None:
                    self.active_layer = None
                self.hook_handles.append(module.register_forward_pre_hook(mark_active))
                self.hook_handles.append(module.register_forward_hook(clear_active))
            original_global_rule = self.original_global_rule

            def wrapped_global_rule(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
                if self.active_layer is not None:
                    self._capture(self.active_layer, key, value, kwargs)
                return original_global_rule(query, key, value, *args, **kwargs)
            self.modeling_module.torch_chunk_gated_delta_rule = wrapped_global_rule
            return

        for layer_index, module in selected_modules.items():
            if module is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not Gated DeltaNet")
            original = module.chunk_gated_delta_rule
            def wrapped(
                query: Any, key: Any, value: Any, *args: Any,
                _original=original, _layer=layer_index, **kwargs: Any,
            ):
                self._capture(_layer, key, value, kwargs)
                return _original(query, key, value, *args, **kwargs)
            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def _capture(self, layer: int, key: Any, value: Any, kwargs: dict[str, Any]) -> None:
        if "g" not in kwargs or "beta" not in kwargs:
            raise RuntimeError("Qwen GLA rule did not pass g and beta by keyword")
        if layer in self.cache:
            raise RuntimeError(f"GLA relay layer {layer} was cached twice")
        rows: dict[int, tuple[tuple[int, ...], Any, Any, Any, Any]] = {}
        for row, positions in self.positions.items():
            if row >= key.shape[0] or max(positions) >= key.shape[1]:
                raise RuntimeError("GLA relay cache index exceeds batch or sequence")
            selected = list(positions)
            rows[row] = (
                positions,
                key[row, selected].detach().clone(),
                value[row, selected].detach().clone(),
                kwargs["g"][row, selected].detach().clone(),
                kwargs["beta"][row, selected].detach().clone(),
            )
        self.cache[layer] = rows

    def close(self) -> None:
        for handle in reversed(getattr(self, "hook_handles", [])):
            handle.remove()
        self.hook_handles.clear()
        if self.modeling_module is not None and self.original_global_rule is not None:
            self.modeling_module.torch_chunk_gated_delta_rule = self.original_global_rule
            self.original_global_rule = None
            self.modeling_module = None
            self.active_layer = None
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()


class BatchedGLACachedRelayRestorer:
    """Restore GLA relay k/v/g/beta from a clean same-row cache."""

    def __init__(
        self,
        parts: Any,
        positions_by_row: dict[int, list[int]],
        layers: list[int],
        clean_cache: dict[
            int, dict[int, tuple[tuple[int, ...], Any, Any, Any, Any]]
        ],
    ) -> None:
        self.positions = {
            int(row): tuple(sorted(set(int(value) for value in positions)))
            for row, positions in positions_by_row.items()
        }
        self.layers = sorted(set(int(value) for value in layers))
        if not self.positions or any(not values for values in self.positions.values()):
            raise ValueError("Cached GLA relay restoration needs nonempty row positions")
        missing = set(self.layers) - set(clean_cache)
        if missing:
            raise RuntimeError(f"Clean GLA cache lacks layers: {sorted(missing)}")
        self.clean_cache = clean_cache
        self.originals: list[tuple[Any, Any]] = []
        self.hook_handles: list[Any] = []
        self.modeling_module: Any | None = None
        self.original_global_rule: Any | None = None
        self.active_layer: int | None = None
        self.calls = 0
        self.restored_position_count = 0
        self.layers_seen: set[int] = set()
        selected_modules = {
            layer_index: getattr(parts.layers[layer_index], "linear_attn", None)
            for layer_index in self.layers
        }
        use_global_rule = bool(selected_modules) and all(
            module is not None and not hasattr(module, "chunk_gated_delta_rule")
            for module in selected_modules.values()
        )
        if use_global_rule:
            modeling_modules = {
                inspect.getmodule(type(module)) for module in selected_modules.values()
            }
            if len(modeling_modules) != 1 or None in modeling_modules:
                raise RuntimeError("Could not identify one shared Qwen GLA modeling module")
            self.modeling_module = modeling_modules.pop()
            if not hasattr(self.modeling_module, "torch_chunk_gated_delta_rule"):
                raise RuntimeError("Qwen modeling module has no chunk GLA rule")
            self.original_global_rule = self.modeling_module.torch_chunk_gated_delta_rule
            for layer_index, module in selected_modules.items():
                def mark_active(_module: Any, _args: Any, _layer=layer_index) -> None:
                    if self.active_layer is not None:
                        raise RuntimeError("Nested Qwen GLA forward calls are unsupported")
                    self.active_layer = _layer
                def clear_active(_module: Any, _args: Any, _output: Any) -> None:
                    self.active_layer = None
                self.hook_handles.append(module.register_forward_pre_hook(mark_active))
                self.hook_handles.append(module.register_forward_hook(clear_active))
            original_global_rule = self.original_global_rule

            def wrapped_global_rule(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
                layer_index = self.active_layer
                if layer_index is None or layer_index not in self.layers:
                    return original_global_rule(query, key, value, *args, **kwargs)
                return self._run_rule(
                    original_global_rule, layer_index, query, key, value, args, kwargs
                )
            self.modeling_module.torch_chunk_gated_delta_rule = wrapped_global_rule
            return

        for layer_index, module in selected_modules.items():
            if module is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not Gated DeltaNet")
            original = module.chunk_gated_delta_rule
            def wrapped(
                query: Any, key: Any, value: Any, *args: Any,
                _original=original, _layer=layer_index, **kwargs: Any,
            ):
                return self._run_rule(
                    _original, _layer, query, key, value, args, kwargs
                )
            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def _run_rule(
        self, rule: Any, layer: int, query: Any, key: Any, value: Any,
        args: tuple[Any, ...], kwargs: dict[str, Any],
    ) -> Any:
        if "g" not in kwargs or "beta" not in kwargs:
            raise RuntimeError("Qwen GLA rule did not pass g and beta by keyword")
        self.calls += 1
        self.layers_seen.add(layer)
        restored_key = key.clone()
        restored_value = value.clone()
        restored_g = kwargs["g"].clone()
        restored_beta = kwargs["beta"].clone()
        for row, positions in self.positions.items():
            (
                cached_positions, cached_key, cached_value, cached_g, cached_beta
            ) = self.clean_cache[layer][row]
            indices = [cached_positions.index(value) for value in positions]
            selected = list(positions)
            restored_key[row, selected] = cached_key[indices]
            restored_value[row, selected] = cached_value[indices]
            restored_g[row, selected] = cached_g[indices]
            restored_beta[row, selected] = cached_beta[indices]
            self.restored_position_count += len(selected)
        restored_kwargs = dict(kwargs)
        restored_kwargs["g"] = restored_g
        restored_kwargs["beta"] = restored_beta
        return rule(
            query, restored_key, restored_value, *args, **restored_kwargs
        )

    def assert_fired(self) -> None:
        if self.calls <= 0 or self.restored_position_count <= 0:
            raise RuntimeError("Cached GLA relay restorer did not restore any position")
        missing = set(self.layers) - self.layers_seen
        if missing:
            raise RuntimeError(f"Cached GLA relay restorer missed layers: {sorted(missing)}")

    def close(self) -> None:
        for handle in reversed(getattr(self, "hook_handles", [])):
            handle.remove()
        self.hook_handles.clear()
        if self.modeling_module is not None and self.original_global_rule is not None:
            self.modeling_module.torch_chunk_gated_delta_rule = self.original_global_rule
            self.original_global_rule = None
            self.modeling_module = None
            self.active_layer = None
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()


class BatchedGLACachedRelayDownstreamRestorer(BatchedGLACachedRelayRestorer):
    """Restore cached GLA writes while preserving relay-token local outputs."""

    def _run_rule(
        self,
        rule: Any,
        layer: int,
        query: Any,
        key: Any,
        value: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if "g" not in kwargs or "beta" not in kwargs:
            raise RuntimeError("Qwen GLA rule did not pass g and beta by keyword")
        self.calls += 1
        self.layers_seen.add(layer)
        perturbed_result = rule(query, key, value, *args, **kwargs)
        if not isinstance(perturbed_result, (tuple, list)) or len(perturbed_result) != 2:
            raise RuntimeError("Expected Qwen GLA rule to return output and state")

        restored_key = key.clone()
        restored_value = value.clone()
        restored_g = kwargs["g"].clone()
        restored_beta = kwargs["beta"].clone()
        for row, positions in self.positions.items():
            (
                cached_positions,
                cached_key,
                cached_value,
                cached_g,
                cached_beta,
            ) = self.clean_cache[layer][row]
            indices = [cached_positions.index(position) for position in positions]
            selected = list(positions)
            restored_key[row, selected] = cached_key[indices]
            restored_value[row, selected] = cached_value[indices]
            restored_g[row, selected] = cached_g[indices]
            restored_beta[row, selected] = cached_beta[indices]
            self.restored_position_count += len(selected)
        restored_kwargs = dict(kwargs)
        restored_kwargs["g"] = restored_g
        restored_kwargs["beta"] = restored_beta
        downstream_result = rule(
            query, restored_key, restored_value, *args, **restored_kwargs
        )
        if not isinstance(downstream_result, (tuple, list)) or len(downstream_result) != 2:
            raise RuntimeError("Expected Qwen GLA rule to return output and state")
        perturbed_output, _perturbed_state = perturbed_result
        downstream_output, downstream_state = downstream_result
        output = downstream_output.clone()
        for row, positions in self.positions.items():
            selected = list(positions)
            output[row, selected] = perturbed_output[row, selected]
        result = (output, downstream_state)
        return result if isinstance(perturbed_result, tuple) else list(result)


class BatchedGLACachedRelayInterceptor:
    """Cross feedback GLA writes, then restore relay writes from a clean cache."""

    def __init__(
        self,
        parts: Any,
        specs_by_target_row: dict[int, tuple[int, list[int], list[int], list[int]]],
        clean_cache: dict[
            int, dict[int, tuple[tuple[int, ...], Any, Any, Any, Any]]
        ],
    ) -> None:
        if not specs_by_target_row:
            raise ValueError("No cached source/relay GLA interceptions were specified")
        by_layer: dict[int, dict[int, tuple[int, tuple[int, ...], tuple[int, ...]]]] = {}
        for target, (source_donor, source_positions, relay_positions, layers) in specs_by_target_row.items():
            source = tuple(sorted(set(int(value) for value in source_positions)))
            relay = tuple(sorted(set(int(value) for value in relay_positions)))
            if not source or not relay or not layers:
                raise ValueError("Every interception needs source positions, relay positions, and layers")
            if int(target) == int(source_donor):
                raise ValueError("Target and source donor must differ")
            if max(source) >= min(relay):
                raise ValueError("Every relay position must causally follow the feedback source")
            for layer in sorted(set(int(value) for value in layers)):
                by_layer.setdefault(layer, {})[int(target)] = (
                    int(source_donor), source, relay
                )
        self.clean_cache = clean_cache
        missing = set(by_layer) - set(clean_cache)
        if missing:
            raise RuntimeError(f"Clean GLA cache lacks layers: {sorted(missing)}")
        self.originals: list[tuple[Any, Any]] = []
        self.hook_handles: list[Any] = []
        self.modeling_module: Any | None = None
        self.original_global_rule: Any | None = None
        self.active_layer: int | None = None
        selected_modules = {
            layer_index: getattr(parts.layers[layer_index], "linear_attn", None)
            for layer_index in by_layer
        }
        use_global_rule = bool(selected_modules) and all(
            module is not None and not hasattr(module, "chunk_gated_delta_rule")
            for module in selected_modules.values()
        )
        if use_global_rule:
            modeling_modules = {
                inspect.getmodule(type(module)) for module in selected_modules.values()
            }
            if len(modeling_modules) != 1 or None in modeling_modules:
                raise RuntimeError("Could not identify one shared Qwen GLA modeling module")
            self.modeling_module = modeling_modules.pop()
            if not hasattr(self.modeling_module, "torch_chunk_gated_delta_rule"):
                raise RuntimeError("Qwen modeling module has no chunk GLA rule")
            self.original_global_rule = self.modeling_module.torch_chunk_gated_delta_rule
            for layer_index, module in selected_modules.items():
                def mark_active(_module: Any, _args: Any, _layer=layer_index) -> None:
                    if self.active_layer is not None:
                        raise RuntimeError("Nested Qwen GLA forward calls are unsupported")
                    self.active_layer = _layer
                def clear_active(_module: Any, _args: Any, _output: Any) -> None:
                    self.active_layer = None
                self.hook_handles.append(module.register_forward_pre_hook(mark_active))
                self.hook_handles.append(module.register_forward_hook(clear_active))
            original_global_rule = self.original_global_rule

            def wrapped_global_rule(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
                layer_index = self.active_layer
                if layer_index is None or layer_index not in by_layer:
                    return original_global_rule(query, key, value, *args, **kwargs)
                return self._run_rule(
                    original_global_rule, layer_index, by_layer[layer_index],
                    query, key, value, args, kwargs,
                )
            self.modeling_module.torch_chunk_gated_delta_rule = wrapped_global_rule
            return

        for layer_index, row_specs in by_layer.items():
            module = getattr(parts.layers[layer_index], "linear_attn", None)
            if module is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not Gated DeltaNet")
            original = module.chunk_gated_delta_rule
            def wrapped(
                query: Any, key: Any, value: Any, *args: Any,
                _original=original, _layer=layer_index, _row_specs=row_specs,
                **kwargs: Any,
            ):
                return self._run_rule(
                    _original, _layer, _row_specs, query, key, value, args, kwargs
                )
            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def _run_rule(
        self,
        rule: Any,
        layer: int,
        row_specs: dict[int, tuple[int, tuple[int, ...], tuple[int, ...]]],
        query: Any,
        key: Any,
        value: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if "g" not in kwargs or "beta" not in kwargs:
            raise RuntimeError("Qwen GLA rule did not pass g and beta by keyword")
        natural = rule(query, key, value, *args, **kwargs)
        source_key = key.clone(); source_value = value.clone()
        source_g = kwargs["g"].clone(); source_beta = kwargs["beta"].clone()
        for target, (donor, source_positions, _relay) in row_specs.items():
            selected = list(source_positions)
            source_key[target, selected] = key[donor, selected]
            source_value[target, selected] = value[donor, selected]
            source_g[target, selected] = kwargs["g"][donor, selected]
            source_beta[target, selected] = kwargs["beta"][donor, selected]
        source_kwargs = dict(kwargs)
        source_kwargs["g"] = source_g; source_kwargs["beta"] = source_beta
        source_result = rule(query, source_key, source_value, *args, **source_kwargs)
        intercepted_key = source_key.clone(); intercepted_value = source_value.clone()
        intercepted_g = source_g.clone(); intercepted_beta = source_beta.clone()
        for target, (_donor, _source, relay_positions) in row_specs.items():
            selected = list(relay_positions)
            (
                cached_positions,
                all_cached_key,
                all_cached_value,
                all_cached_g,
                all_cached_beta,
            ) = self.clean_cache[layer][target]
            cache_indices = [cached_positions.index(value) for value in relay_positions]
            cached_key = all_cached_key[cache_indices]
            cached_value = all_cached_value[cache_indices]
            cached_g = all_cached_g[cache_indices]
            cached_beta = all_cached_beta[cache_indices]
            if cached_key.shape != intercepted_key[target, selected].shape:
                raise RuntimeError("Cached GLA relay K shape changed")
            intercepted_key[target, selected] = cached_key
            intercepted_value[target, selected] = cached_value
            intercepted_g[target, selected] = cached_g
            intercepted_beta[target, selected] = cached_beta
        intercepted_kwargs = dict(kwargs)
        intercepted_kwargs["g"] = intercepted_g; intercepted_kwargs["beta"] = intercepted_beta
        intercepted_result = rule(
            query, intercepted_key, intercepted_value, *args, **intercepted_kwargs
        )
        combine_specs = {
            target: (source_positions, relay_positions)
            for target, (_donor, source_positions, relay_positions) in row_specs.items()
        }
        return _combine_gla_results(natural, source_result, intercepted_result, combine_specs)

    def close(self) -> None:
        for handle in reversed(getattr(self, "hook_handles", [])):
            handle.remove()
        self.hook_handles.clear()
        if self.modeling_module is not None and self.original_global_rule is not None:
            self.modeling_module.torch_chunk_gated_delta_rule = self.original_global_rule
            self.original_global_rule = None
            self.modeling_module = None
            self.active_layer = None
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()


class BatchedSDPASourceRelayInterceptor:
    """Cross feedback K/V, then restore one relay region's downstream K/V.

    For each target row the complete feedback source is copied from the paired
    other-task row.  The selected relay K/V is copied from an exact clean
    duplicate of the target task.  Feedback-source outputs remain natural and
    relay-token outputs retain the source-crossed computation; only queries at
    other causally later positions receive the restored relay writes.
    """

    def __init__(
        self,
        parts: Any,
        specs_by_target_row: dict[
            int, tuple[int, list[int], int, list[int], list[int]]
        ],
    ) -> None:
        import torch

        if not specs_by_target_row:
            raise ValueError("No source/relay K/V interceptions were specified")
        self.by_layer: dict[
            int, dict[int, tuple[int, tuple[int, ...], int, tuple[int, ...]]]
        ] = {}
        for target, (source_donor, source_positions, clean_row, relay_positions, layers) in specs_by_target_row.items():
            source = tuple(sorted(set(int(value) for value in source_positions)))
            relay = tuple(sorted(set(int(value) for value in relay_positions)))
            if not source or not relay or not layers:
                raise ValueError("Every interception needs source positions, relay positions, and layers")
            if int(target) in {int(source_donor), int(clean_row)}:
                raise ValueError("Target, source donor, and clean duplicate must be distinct rows")
            if max(source) >= min(relay):
                raise ValueError("Every relay position must causally follow the feedback source")
            for layer in sorted(set(int(value) for value in layers)):
                self.by_layer.setdefault(layer, {})[int(target)] = (
                    int(source_donor), source, int(clean_row), relay
                )

        self.active_layer: int | None = None
        self.handles: list[Any] = []
        for layer_index in sorted(self.by_layer):
            attention = getattr(parts.layers[layer_index], "self_attn", None)
            if attention is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not ordinary attention")
            self.handles.append(attention.register_forward_pre_hook(self._enter(layer_index)))
            self.handles.append(attention.register_forward_hook(self._leave(layer_index)))
        self.original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = self._wrapped_sdpa

    def _enter(self, layer_index: int):
        def enter(_module: Any, _inputs: Any) -> None:
            if self.active_layer is not None:
                raise RuntimeError("Nested ordinary-attention calls are unsupported")
            self.active_layer = layer_index
        return enter

    def _leave(self, layer_index: int):
        def leave(_module: Any, _inputs: Any, _output: Any) -> None:
            if self.active_layer != layer_index:
                raise RuntimeError("Ordinary-attention layer stack became inconsistent")
            self.active_layer = None
        return leave

    def _wrapped_sdpa(self, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
        layer = self.active_layer
        if layer is None or layer not in self.by_layer:
            return self.original_sdpa(query, key, value, *args, **kwargs)
        natural = self.original_sdpa(query, key, value, *args, **kwargs)
        source_key = key.clone()
        source_value = value.clone()
        for target, (donor, source_positions, _clean, _relay) in self.by_layer[layer].items():
            if max(target, donor) >= key.shape[0] or max(source_positions) >= key.shape[-2]:
                raise RuntimeError("Source K/V interception index exceeds the batch or sequence")
            selected = list(source_positions)
            source_key[target, :, selected, :] = key[donor, :, selected, :]
            source_value[target, :, selected, :] = value[donor, :, selected, :]
        source_edited = self.original_sdpa(
            query, source_key, source_value, *args, **kwargs
        )
        intercepted_key = source_key.clone()
        intercepted_value = source_value.clone()
        for target, (_donor, _source, clean, relay_positions) in self.by_layer[layer].items():
            if max(target, clean) >= key.shape[0] or max(relay_positions) >= key.shape[-2]:
                raise RuntimeError("Relay K/V interception index exceeds the batch or sequence")
            selected = list(relay_positions)
            intercepted_key[target, :, selected, :] = key[clean, :, selected, :]
            intercepted_value[target, :, selected, :] = value[clean, :, selected, :]
        intercepted = self.original_sdpa(
            query, intercepted_key, intercepted_value, *args, **kwargs
        )
        output = natural.clone()
        for target, (_donor, source_positions, _clean, relay_positions) in self.by_layer[layer].items():
            cutoff = max(source_positions)
            if cutoff + 1 < output.shape[-2]:
                output[target, :, cutoff + 1 :, :] = intercepted[target, :, cutoff + 1 :, :]
            output[target, :, list(relay_positions), :] = source_edited[
                target, :, list(relay_positions), :
            ]
        return output

    def close(self) -> None:
        import torch

        if hasattr(self, "original_sdpa"):
            torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
        for handle in reversed(getattr(self, "handles", [])):
            handle.remove()
        self.handles = []
        self.active_layer = None


class BatchedGLASourceRelayInterceptor:
    """GLA counterpart of :class:`BatchedSDPASourceRelayInterceptor`."""

    def __init__(
        self,
        parts: Any,
        specs_by_target_row: dict[
            int, tuple[int, list[int], int, list[int], list[int]]
        ],
    ) -> None:
        if not specs_by_target_row:
            raise ValueError("No source/relay GLA interceptions were specified")
        by_layer: dict[
            int, dict[int, tuple[int, tuple[int, ...], int, tuple[int, ...]]]
        ] = {}
        for target, (source_donor, source_positions, clean_row, relay_positions, layers) in specs_by_target_row.items():
            source = tuple(sorted(set(int(value) for value in source_positions)))
            relay = tuple(sorted(set(int(value) for value in relay_positions)))
            if not source or not relay or not layers:
                raise ValueError("Every interception needs source positions, relay positions, and layers")
            if int(target) in {int(source_donor), int(clean_row)}:
                raise ValueError("Target, source donor, and clean duplicate must be distinct rows")
            if max(source) >= min(relay):
                raise ValueError("Every relay position must causally follow the feedback source")
            for layer in sorted(set(int(value) for value in layers)):
                by_layer.setdefault(layer, {})[int(target)] = (
                    int(source_donor), source, int(clean_row), relay
                )

        self.originals: list[tuple[Any, Any]] = []
        self.hook_handles: list[Any] = []
        self.modeling_module: Any | None = None
        self.original_global_rule: Any | None = None
        self.active_layer: int | None = None
        selected_modules = {
            layer_index: getattr(parts.layers[layer_index], "linear_attn", None)
            for layer_index in by_layer
        }
        use_global_rule = bool(selected_modules) and all(
            module is not None and not hasattr(module, "chunk_gated_delta_rule")
            for module in selected_modules.values()
        )
        if use_global_rule:
            modeling_modules = {
                inspect.getmodule(type(module)) for module in selected_modules.values()
            }
            if len(modeling_modules) != 1 or None in modeling_modules:
                raise RuntimeError("Could not identify one shared Qwen GLA modeling module")
            self.modeling_module = modeling_modules.pop()
            if not hasattr(self.modeling_module, "torch_chunk_gated_delta_rule"):
                raise RuntimeError("Qwen modeling module has no chunk GLA rule")
            self.original_global_rule = self.modeling_module.torch_chunk_gated_delta_rule
            for layer_index, module in selected_modules.items():
                def mark_active(_module: Any, _args: Any, _layer=layer_index) -> None:
                    if self.active_layer is not None:
                        raise RuntimeError("Nested Qwen GLA forward calls are unsupported")
                    self.active_layer = _layer
                def clear_active(_module: Any, _args: Any, _output: Any) -> None:
                    self.active_layer = None
                self.hook_handles.append(module.register_forward_pre_hook(mark_active))
                self.hook_handles.append(module.register_forward_hook(clear_active))
            original_global_rule = self.original_global_rule

            def wrapped_global_rule(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any):
                layer_index = self.active_layer
                if layer_index is None or layer_index not in by_layer:
                    return original_global_rule(query, key, value, *args, **kwargs)
                return self._run_rule(
                    original_global_rule, by_layer[layer_index], query, key, value,
                    args, kwargs,
                )
            self.modeling_module.torch_chunk_gated_delta_rule = wrapped_global_rule
            return

        for layer_index, row_specs in by_layer.items():
            module = getattr(parts.layers[layer_index], "linear_attn", None)
            if module is None:
                self.close()
                raise ValueError(f"Layer {layer_index} is not Gated DeltaNet")
            original = module.chunk_gated_delta_rule
            def wrapped(
                query: Any, key: Any, value: Any, *args: Any,
                _original=original, _row_specs=row_specs, **kwargs: Any,
            ):
                return self._run_rule(
                    _original, _row_specs, query, key, value, args, kwargs
                )
            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    @staticmethod
    def _run_rule(
        rule: Any,
        row_specs: dict[int, tuple[int, tuple[int, ...], int, tuple[int, ...]]],
        query: Any,
        key: Any,
        value: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if "g" not in kwargs or "beta" not in kwargs:
            raise RuntimeError("Qwen GLA rule did not pass g and beta by keyword")
        natural = rule(query, key, value, *args, **kwargs)
        source_key = key.clone()
        source_value = value.clone()
        source_g = kwargs["g"].clone()
        source_beta = kwargs["beta"].clone()
        for target, (donor, source_positions, _clean, _relay) in row_specs.items():
            selected = list(source_positions)
            source_key[target, selected] = key[donor, selected]
            source_value[target, selected] = value[donor, selected]
            source_g[target, selected] = kwargs["g"][donor, selected]
            source_beta[target, selected] = kwargs["beta"][donor, selected]
        source_kwargs = dict(kwargs)
        source_kwargs["g"] = source_g
        source_kwargs["beta"] = source_beta
        source_result = rule(
            query, source_key, source_value, *args, **source_kwargs
        )
        intercepted_key = source_key.clone()
        intercepted_value = source_value.clone()
        intercepted_g = source_g.clone()
        intercepted_beta = source_beta.clone()
        for target, (_donor, _source, clean, relay_positions) in row_specs.items():
            selected = list(relay_positions)
            intercepted_key[target, selected] = key[clean, selected]
            intercepted_value[target, selected] = value[clean, selected]
            intercepted_g[target, selected] = kwargs["g"][clean, selected]
            intercepted_beta[target, selected] = kwargs["beta"][clean, selected]
        intercepted_kwargs = dict(kwargs)
        intercepted_kwargs["g"] = intercepted_g
        intercepted_kwargs["beta"] = intercepted_beta
        intercepted_result = rule(
            query, intercepted_key, intercepted_value, *args, **intercepted_kwargs
        )
        combine_specs = {
            target: (source_positions, relay_positions)
            for target, (_donor, source_positions, _clean, relay_positions)
            in row_specs.items()
        }
        return _combine_gla_results(
            natural, source_result, intercepted_result, combine_specs
        )

    def close(self) -> None:
        for handle in reversed(getattr(self, "hook_handles", [])):
            handle.remove()
        self.hook_handles.clear()
        if self.modeling_module is not None and self.original_global_rule is not None:
            self.modeling_module.torch_chunk_gated_delta_rule = self.original_global_rule
            self.original_global_rule = None
            self.modeling_module = None
            self.active_layer = None
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()
