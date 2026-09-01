from __future__ import annotations

import copy
from typing import Any


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class ReadoutCapture:
    """Capture the final-position residual at one post-block readout."""

    def __init__(self, parts: Any, readout: int, last_indices: list[int]):
        if readout < 1 or readout > len(parts.layers):
            raise ValueError(f"Invalid post-block readout {readout}")
        self.last_indices = tuple(int(index) for index in last_indices)
        self.value = None
        self.handle = parts.layers[readout - 1].register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        import torch

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        cols = torch.as_tensor(self.last_indices, device=hidden.device)
        self.value = hidden[rows, cols].detach().float().cpu()

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "ReadoutCapture":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class ReadoutAdd:
    """Add a precomputed residual perturbation at final position."""

    def __init__(self, parts: Any, readout: int, last_indices: list[int], delta: Any):
        if len(last_indices) != len(delta):
            raise ValueError("One residual perturbation is required per batch item")
        self.last_indices = tuple(int(index) for index in last_indices)
        self.delta = delta
        self.handle = parts.layers[readout - 1].register_forward_hook(self._add)

    def _add(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        changed = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        cols = torch.as_tensor(self.last_indices, device=hidden.device)
        changed[rows, cols] = changed[rows, cols] + self.delta.to(
            device=hidden.device, dtype=hidden.dtype
        )
        return _replace_hidden(output, changed)

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "ReadoutAdd":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class CpuAnswerLens:
    """Differentiable CPU copy of the native final norm and A-D unembedding."""

    def __init__(self, parts: Any, canonical_ids: list[int]):
        import torch

        self.norm = copy.deepcopy(parts.final_norm)
        if hasattr(self.norm, "_hf_hook"):
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(self.norm, recurse=True)
        self.norm = self.norm.float().cpu().eval()
        for parameter in self.norm.parameters():
            parameter.requires_grad_(False)
        self.rows = parts.output_head.weight.detach()[canonical_ids].float().cpu()
        bias = getattr(parts.output_head, "bias", None)
        self.bias = None if bias is None else bias.detach()[canonical_ids].float().cpu()
        self.torch = torch

    def logits(self, residual: Any) -> Any:
        values = self.norm(residual) @ self.rows.T
        return values if self.bias is None else values + self.bias

    @staticmethod
    def contrast(logits: Any, target_letters: Any) -> Any:
        torch = __import__("torch")
        rows = torch.arange(len(logits))
        selected = logits[rows, target_letters]
        return selected - (logits.sum(dim=-1) - selected) / 3.0

    def calibrated_delta(
        self,
        residual: Any,
        target_letters: Any,
        target_change: Any,
        steps: int,
    ) -> tuple[Any, Any]:
        """Minimum-local-norm update calibrated in native lens-contrast units."""
        torch = self.torch
        # A tensor captured by a hook inside ``torch.inference_mode`` remains an
        # inference tensor after moving to CPU.  Copy through NumPy so the small
        # standalone lens calculation receives an ordinary autograd-capable
        # tensor without enabling gradients through the model forward pass.
        base = torch.from_numpy(residual.detach().float().cpu().numpy().copy())
        letters = torch.as_tensor(target_letters, dtype=torch.long)
        wanted = torch.as_tensor(target_change, dtype=torch.float32)
        start = self.contrast(self.logits(base), letters).detach()
        delta = torch.zeros_like(base)
        for _ in range(steps):
            point = (base + delta).detach().requires_grad_(True)
            current = self.contrast(self.logits(point), letters)
            gradient = torch.autograd.grad(current.sum(), point)[0]
            error = wanted - (current.detach() - start)
            denominator = gradient.square().sum(dim=-1).clamp_min(1e-12)
            delta = delta + (error / denominator)[:, None] * gradient.detach()
        achieved = self.contrast(self.logits(base + delta), letters).detach() - start
        return delta.detach(), achieved

    def answer_orthogonal_control(self, residual: Any, matched_delta: Any, seed: int) -> Any:
        """Equal-norm random updates locally orthogonal to all four A-D scores."""
        torch = self.torch
        point = torch.from_numpy(
            residual.detach().float().cpu().numpy().copy()
        ).requires_grad_(True)
        logits = self.logits(point)
        gradients = []
        for letter in range(4):
            gradient = torch.autograd.grad(
                logits[:, letter].sum(), point, retain_graph=letter < 3
            )[0]
            gradients.append(gradient.detach())
        basis = torch.stack(gradients, dim=-1)  # batch, hidden, four
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        random = torch.randn(point.shape, generator=generator)
        gram = basis.transpose(1, 2) @ basis
        gram = gram + torch.eye(4)[None] * 1e-6
        coefficients = torch.linalg.solve(gram, basis.transpose(1, 2) @ random[..., None])
        orthogonal = random - (basis @ coefficients).squeeze(-1)
        target_norm = matched_delta.float().norm(dim=-1)
        orthogonal_norm = orthogonal.norm(dim=-1).clamp_min(1e-12)
        return orthogonal * (target_norm / orthogonal_norm)[:, None]
