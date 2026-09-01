from __future__ import annotations

from typing import Any

import numpy as np

from .gdn_intervention import linear_attention_layers
from .modeling import ModelParts


class DeltaNetScreenCollector:
    """Exact within-module counterfactual replay for every GDN layer and value head.

    All value heads recur independently. Zeroing the selected beta write for all
    heads in one replay therefore yields the same per-head counterfactual output
    as 48 separate head ablations, before the linear output projection mixes
    heads.
    """

    def __init__(
        self,
        parts: ModelParts,
        source_positions: dict[str, list[int]],
        canonical_ids: list[int],
    ) -> None:
        import torch

        self.parts = parts
        self.sources = list(source_positions)
        self.positions = {
            name: tuple(int(position) for position in positions)
            for name, positions in source_positions.items()
        }
        self.layers = linear_attention_layers(parts)
        self.layer_to_index = {layer: index for index, layer in enumerate(self.layers)}
        self.originals: list[tuple[Any, Any]] = []
        self.hooks = []
        self.z_final: dict[int, Any] = {}
        first = parts.layers[self.layers[0]].linear_attn
        shape = (len(self.layers), len(self.sources), first.num_v_heads)
        self.direct_ad = torch.empty((*shape, 4), dtype=torch.float16, device="cpu")
        self.output_norm = torch.empty(shape, dtype=torch.float16, device="cpu")
        self.core_difference_norm = torch.empty(shape, dtype=torch.float16, device="cpu")
        self.beta_at_source = torch.empty(shape, dtype=torch.float16, device="cpu")
        self.retention_at_source = torch.empty(shape, dtype=torch.float16, device="cpu")
        rows = parts.output_head.weight.detach()[canonical_ids].float()

        for layer in self.layers:
            module = parts.layers[layer].linear_attn
            layer_index = self.layer_to_index[layer]
            self.hooks.append(module.in_proj_z.register_forward_hook(self._capture_z(layer)))
            projection = module.out_proj.weight.detach().float().T @ rows.T
            projection = projection.reshape(module.num_v_heads, module.head_v_dim, 4)
            original = module.chunk_gated_delta_rule

            def wrapped(*args: Any, _module=module, _layer_index=layer_index,
                        _projection=projection, _original=original, **kwargs: Any):
                normal, state = _original(*args, **kwargs)
                beta = kwargs["beta"]
                g = kwargs["g"]
                z = self.z_final[_layer_index]
                normal_final = normal[:, -1].reshape(-1, _module.head_v_dim)
                z_flat = z.reshape(-1, _module.head_v_dim)
                normal_normed = _module.norm(normal_final, z_flat).reshape(
                    -1, _module.num_v_heads, _module.head_v_dim
                )
                for source_index, source in enumerate(self.sources):
                    beta_cf = beta.clone()
                    positions = self.positions[source]
                    beta_cf[:, list(positions), :] = 0
                    cf_kwargs = dict(kwargs)
                    cf_kwargs["beta"] = beta_cf
                    cf_kwargs["output_final_state"] = False
                    counterfactual, _ = _original(*args, **cf_kwargs)
                    cf_final = counterfactual[:, -1].reshape(-1, _module.head_v_dim)
                    cf_normed = _module.norm(cf_final, z_flat).reshape(
                        -1, _module.num_v_heads, _module.head_v_dim
                    )
                    difference = (normal_normed - cf_normed)[0].float()
                    direct = torch.einsum("hd,hdc->hc", difference, _projection)
                    weights = _module.out_proj.weight.detach().float().reshape(
                        _module.hidden_size, _module.num_v_heads, _module.head_v_dim
                    )
                    head_output = torch.einsum("hd,ohd->ho", difference, weights)
                    core_difference = (normal[:, -1] - counterfactual[:, -1])[0].float()
                    self.direct_ad[_layer_index, source_index] = direct.detach().to("cpu", dtype=torch.float16)
                    self.output_norm[_layer_index, source_index] = torch.linalg.vector_norm(
                        head_output, dim=-1
                    ).detach().to("cpu", dtype=torch.float16)
                    self.core_difference_norm[_layer_index, source_index] = torch.linalg.vector_norm(
                        core_difference, dim=-1
                    ).detach().to("cpu", dtype=torch.float16)
                    self.beta_at_source[_layer_index, source_index] = beta[0, list(positions)].mean(dim=0).detach().to(
                        "cpu", dtype=torch.float16
                    )
                    self.retention_at_source[_layer_index, source_index] = g[0, list(positions)].exp().mean(dim=0).detach().to(
                        "cpu", dtype=torch.float16
                    )
                return normal, state

            self.originals.append((module, original))
            module.chunk_gated_delta_rule = wrapped

    def _capture_z(self, layer: int):
        layer_index = self.layer_to_index[layer]

        def hook(module: Any, _inputs: Any, output: Any) -> None:
            self.z_final[layer_index] = output[:, -1].reshape(
                output.shape[0], module.out_features // self.parts.layers[layer].linear_attn.head_v_dim,
                self.parts.layers[layer].linear_attn.head_v_dim,
            )

        return hook

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "head_direct_ad": self.direct_ad.numpy(),
            "head_output_norm": self.output_norm.numpy(),
            "core_difference_norm": self.core_difference_norm.numpy(),
            "beta_at_source": self.beta_at_source.numpy(),
            "retention_at_source": self.retention_at_source.numpy(),
        }

    def close(self) -> None:
        for module, original in reversed(self.originals):
            module.chunk_gated_delta_rule = original
        self.originals.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self) -> "DeltaNetScreenCollector":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

