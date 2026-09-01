from __future__ import annotations

import inspect
from typing import Any


class AttentionSourceRouteCollector:
    """Exact final-query source-edge counterfactuals for one attention mixer."""

    _attribute = "_secondchance_source_route_collector"

    def __init__(
        self,
        parts: Any,
        layer: int,
        source_positions: dict[str, list[int]],
        canonical_ids: list[int],
    ) -> None:
        import torch

        self.layer = int(layer)
        self.sources = list(source_positions)
        self.positions = {
            name: tuple(int(position) for position in values)
            for name, values in source_positions.items()
        }
        attention = getattr(parts.layers[self.layer], "self_attn", None)
        if attention is None:
            raise ValueError(f"Layer {self.layer} is not ordinary attention")
        self.attention = attention
        self.module = inspect.getmodule(type(attention))
        if self.module is None or not hasattr(self.module, "eager_attention_forward"):
            raise RuntimeError("Could not locate Qwen eager attention implementation")
        self.original = self.module.eager_attention_forward
        repeat_kv = self.original.__globals__.get("repeat_kv")
        if repeat_kv is None:
            raise RuntimeError("Qwen eager attention does not expose repeat_kv")
        self.repeat_kv = repeat_kv
        self.gate = None
        self.route_hidden = None
        self.route_direct_ad = None
        self.attention_mass = None
        self.token_attention = None
        self.output_rows = parts.output_head.weight.detach()[canonical_ids].float()
        self.q_hook = attention.q_proj.register_forward_hook(self._capture_gate)
        setattr(attention, self._attribute, True)
        self.module.eager_attention_forward = self._wrapped

    def _capture_gate(self, module: Any, _inputs: Any, output: Any) -> None:
        head_dim = int(self.attention.head_dim)
        shaped = output.view(*output.shape[:-1], -1, head_dim * 2)
        _query, gate = shaped.chunk(2, dim=-1)
        self.gate = gate[:, -1].sigmoid()

    def _wrapped(
        self,
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        import torch

        if not getattr(module, self._attribute, False):
            return self.original(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )
        if self.gate is None:
            raise RuntimeError("Attention gate was not captured before eager attention")
        key_states = self.repeat_kv(key, module.num_key_value_groups)
        value_states = self.repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        weights = torch.nn.functional.dropout(
            weights, p=dropout, training=module.training
        )
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()

        final_weights = weights[:, :, -1]
        normal = output[:, -1]
        deltas = []
        masses = []
        for source in self.sources:
            indices = torch.as_tensor(
                self.positions[source], device=weights.device, dtype=torch.long
            )
            source_weights = final_weights.index_select(-1, indices)
            source_values = value_states.index_select(-2, indices)
            mass = source_weights.sum(dim=-1)
            removed = torch.einsum("bhk,bhkd->bhd", source_weights, source_values)
            denominator = (1.0 - mass).clamp_min(1e-6)
            counterfactual = (normal - removed) / denominator[..., None]
            deltas.append((normal - counterfactual) * self.gate)
            masses.append(mass)
        gated_delta = torch.stack(deltas, dim=1)
        mass = torch.stack(masses, dim=1)
        projection = module.o_proj.weight.detach().float().reshape(
            module.o_proj.out_features,
            gated_delta.shape[2],
            gated_delta.shape[3],
        )
        route_hidden = torch.einsum(
            "bshd,ohd->bsho", gated_delta.float(), projection
        )
        route_direct = torch.einsum(
            "bsho,co->bshc", route_hidden, self.output_rows
        )
        self.route_hidden = route_hidden.detach()
        self.route_direct_ad = route_direct.detach().to("cpu", dtype=torch.float16)
        self.attention_mass = mass.detach().to("cpu", dtype=torch.float16)
        self.token_attention = final_weights.detach().to("cpu", dtype=torch.float16)
        return output, weights

    def arrays(self) -> dict[str, Any]:
        if any(
            value is None
            for value in (
                self.route_direct_ad,
                self.attention_mass,
                self.token_attention,
            )
        ):
            raise RuntimeError("Attention route collector has not observed a forward")
        return {
            "attention_route_direct_ad": self.route_direct_ad[0].numpy(),
            "attention_mass": self.attention_mass[0].numpy(),
            "attention_token_weights": self.token_attention[0].numpy(),
        }

    def hidden_delta(self, source: str, head: int) -> Any:
        if self.route_hidden is None:
            raise RuntimeError("Attention route collector has not observed a forward")
        source_index = self.sources.index(source)
        return self.route_hidden[0, source_index, int(head)]

    def close(self) -> None:
        if hasattr(self.attention, self._attribute):
            delattr(self.attention, self._attribute)
        if self.module.eager_attention_forward is self._wrapped:
            self.module.eager_attention_forward = self.original
        else:
            self.module.eager_attention_forward = self.original
        self.q_hook.remove()

    def __enter__(self) -> "AttentionSourceRouteCollector":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class DeltaNetSourceRouteCollector:
    """Exact source-write counterfactuals for every value head in one GDN mixer."""

    def __init__(
        self,
        parts: Any,
        layer: int,
        source_positions: dict[str, list[int]],
        canonical_ids: list[int],
    ) -> None:
        import torch

        self.layer = int(layer)
        self.sources = list(source_positions)
        self.positions = {
            name: tuple(int(position) for position in values)
            for name, values in source_positions.items()
        }
        self.module = getattr(parts.layers[self.layer], "linear_attn", None)
        if self.module is None:
            raise ValueError(f"Layer {self.layer} is not Gated DeltaNet")
        self.original = self.module.chunk_gated_delta_rule
        self.z_final = None
        self.route_hidden = None
        self.route_direct_ad = None
        self.beta = None
        self.retention = None
        self.output_rows = parts.output_head.weight.detach()[canonical_ids].float()
        self.z_hook = self.module.in_proj_z.register_forward_hook(self._capture_z)
        self.module.chunk_gated_delta_rule = self._wrapped

    def _capture_z(self, module: Any, _inputs: Any, output: Any) -> None:
        self.z_final = output[:, -1].reshape(
            output.shape[0],
            module.out_features // self.module.head_v_dim,
            self.module.head_v_dim,
        )

    def _wrapped(self, *args: Any, **kwargs: Any):
        import torch

        normal, state = self.original(*args, **kwargs)
        if self.z_final is None:
            raise RuntimeError("DeltaNet output gate was not captured")
        beta = kwargs["beta"]
        g = kwargs["g"]
        z_flat = self.z_final.reshape(-1, self.module.head_v_dim)
        normal_final = normal[:, -1].reshape(-1, self.module.head_v_dim)
        normal_normed = self.module.norm(normal_final, z_flat).reshape(
            normal.shape[0], self.module.num_v_heads, self.module.head_v_dim
        )
        hidden_deltas = []
        direct_deltas = []
        beta_values = []
        retention_values = []
        projection = self.module.out_proj.weight.detach().float().reshape(
            self.module.hidden_size,
            self.module.num_v_heads,
            self.module.head_v_dim,
        )
        for source in self.sources:
            positions = list(self.positions[source])
            beta_cf = beta.clone()
            beta_cf[:, positions, :] = 0
            cf_kwargs = dict(kwargs)
            cf_kwargs["beta"] = beta_cf
            cf_kwargs["output_final_state"] = False
            counterfactual, _ = self.original(*args, **cf_kwargs)
            cf_final = counterfactual[:, -1].reshape(-1, self.module.head_v_dim)
            cf_normed = self.module.norm(cf_final, z_flat).reshape(
                normal.shape[0], self.module.num_v_heads, self.module.head_v_dim
            )
            difference = (normal_normed - cf_normed).float()
            hidden = torch.einsum("bhd,ohd->bho", difference, projection)
            direct = torch.einsum("bho,co->bhc", hidden, self.output_rows)
            hidden_deltas.append(hidden)
            direct_deltas.append(direct)
            beta_values.append(beta[:, positions].mean(dim=1))
            retention_values.append(g[:, positions].exp().mean(dim=1))
        self.route_hidden = torch.stack(hidden_deltas, dim=1).detach()
        self.route_direct_ad = torch.stack(direct_deltas, dim=1).detach().to(
            "cpu", dtype=torch.float16
        )
        self.beta = torch.stack(beta_values, dim=1).detach().to(
            "cpu", dtype=torch.float16
        )
        self.retention = torch.stack(retention_values, dim=1).detach().to(
            "cpu", dtype=torch.float16
        )
        return normal, state

    def arrays(self) -> dict[str, Any]:
        if any(
            value is None
            for value in (self.route_direct_ad, self.beta, self.retention)
        ):
            raise RuntimeError("DeltaNet route collector has not observed a forward")
        return {
            "gdn_route_direct_ad": self.route_direct_ad[0].numpy(),
            "gdn_beta": self.beta[0].numpy(),
            "gdn_retention": self.retention[0].numpy(),
        }

    def hidden_delta(self, source: str, head: int) -> Any:
        if self.route_hidden is None:
            raise RuntimeError("DeltaNet route collector has not observed a forward")
        source_index = self.sources.index(source)
        return self.route_hidden[0, source_index, int(head)]

    def close(self) -> None:
        self.module.chunk_gated_delta_rule = self.original
        self.z_hook.remove()

    def __enter__(self) -> "DeltaNetSourceRouteCollector":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
