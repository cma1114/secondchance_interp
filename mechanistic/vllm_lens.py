from __future__ import annotations

from typing import Any


def _trunk_and_head(model: Any) -> tuple[Any, Any]:
    """Locate the Llama decoder trunk and sharded LM head in a vLLM model."""
    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
    for candidate in candidates:
        trunk = getattr(candidate, "model", None)
        head = getattr(candidate, "lm_head", None)
        if trunk is not None and head is not None and hasattr(trunk, "layers"):
            return trunk, head
    raise RuntimeError(f"Could not locate vLLM Llama trunk/head under {type(model)!r}")


class LensWorkerExtension:
    """vLLM worker RPC extension for a final-position native logit lens.

    vLLM's Llama blocks carry the residual accumulator separately from the
    current block output.  The actual post-block residual stream is their sum.
    Only TP rank zero captures it; residuals are replicated after TP reductions.
    """

    def install_lens_capture(self, answer_token_ids: list[int]) -> dict[str, Any]:
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        self._lens_tp_rank = int(get_tensor_model_parallel_rank())
        self._lens_answer_token_ids = [int(x) for x in answer_token_ids]
        self._lens_active = False
        self._lens_expected_batch = 0
        self._lens_values = []
        self._lens_handles = []

        model = self.model_runner.model
        trunk, head = _trunk_and_head(model)
        self._lens_trunk = trunk
        self._lens_head = head

        if self._lens_tp_rank == 0:
            def final_indices(device: torch.device) -> torch.Tensor:
                n = self._lens_expected_batch
                runner = self.model_runner
                if hasattr(runner, "input_buffers"):
                    # vLLM's V2 GPU model runner.
                    starts = runner.input_buffers.query_start_loc[: n + 1]
                else:
                    # vLLM's V1 GPU model runner.
                    starts = runner.query_start_loc.gpu[: n + 1]
                return (starts[1:] - 1).to(device=device, dtype=torch.long)

            def embedding_hook(_module: Any, _inputs: Any, output: Any) -> None:
                if not self._lens_active or self._lens_values[0] is not None:
                    return
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                self._lens_values[0] = hidden[final_indices(hidden.device)].detach()

            def block_hook(layer_index: int):
                def capture(_module: Any, _inputs: Any, output: Any) -> None:
                    if not self._lens_active or self._lens_values[layer_index + 1] is not None:
                        return
                    if not isinstance(output, (tuple, list)) or len(output) < 2:
                        raise RuntimeError("Expected vLLM Llama block to return hidden_states, residual")
                    hidden, residual = output[0], output[1]
                    stream = hidden if residual is None else hidden + residual
                    self._lens_values[layer_index + 1] = stream[
                        final_indices(stream.device)
                    ].detach()
                return capture

            self._lens_handles.append(trunk.embed_tokens.register_forward_hook(embedding_hook))
            self._lens_handles.extend(
                layer.register_forward_hook(block_hook(i)) for i, layer in enumerate(trunk.layers)
            )

        return {
            "tp_rank": self._lens_tp_rank,
            "n_layers": len(trunk.layers),
            "head_rows": int(head.weight.shape[0]),
            "head_width": int(head.weight.shape[1]),
        }

    def begin_lens_capture(self, expected_batch: int) -> None:
        if self._lens_tp_rank != 0:
            return
        if self._lens_active:
            raise RuntimeError("A lens capture is already active")
        if expected_batch < 1:
            raise ValueError("expected_batch must be positive")
        self._lens_expected_batch = int(expected_batch)
        self._lens_values = [None] * (len(self._lens_trunk.layers) + 1)
        self._lens_active = True

    def finish_lens_capture(self) -> dict[str, Any] | None:
        import torch

        if self._lens_tp_rank != 0:
            return None
        self._lens_active = False
        missing = [i for i, value in enumerate(self._lens_values) if value is None]
        if missing:
            raise RuntimeError(f"Lens hooks did not capture readouts {missing}")

        # readout, batch, hidden -> batch, readout, hidden
        residuals = torch.stack(self._lens_values, dim=0).transpose(0, 1)
        batch, readouts, width = residuals.shape
        flat = residuals.reshape(batch * readouts, width)
        normalized = self._lens_trunk.norm(flat)
        if isinstance(normalized, (tuple, list)):
            normalized = normalized[0]

        # Canonical and leading-space A-D tokens all fall in TP rank zero's
        # vocab shard for Llama 3.1.  Guard this explicitly rather than silently
        # using a wrong local row.
        local_rows = int(self._lens_head.weight.shape[0])
        if max(self._lens_answer_token_ids) >= local_rows:
            raise RuntimeError(
                "Requested answer token lies outside TP rank zero's local vocab shard; "
                "distributed row gathering is required"
            )
        weights = self._lens_head.weight[self._lens_answer_token_ids]
        logits = normalized.float() @ weights.float().T
        logits = logits.reshape(batch, readouts, len(self._lens_answer_token_ids))
        result = {
            # Plain lists survive vLLM's multiprocess RPC serializer without
            # the ndarray metadata wrapper used for tensor data-plane traffic.
            "variant_logits": logits.cpu().tolist(),
            "residual_norms": residuals.float().norm(dim=-1).cpu().tolist(),
        }
        self._lens_values = []
        return result

    def remove_lens_capture(self) -> None:
        for handle in getattr(self, "_lens_handles", []):
            handle.remove()
        self._lens_handles = []
        self._lens_active = False
