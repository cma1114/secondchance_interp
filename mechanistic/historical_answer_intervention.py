from __future__ import annotations

import hashlib
from typing import Any


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class PositionReadoutCapture:
    """Capture one token's post-block residual from a single-example forward."""

    def __init__(self, parts: Any, readout: int, position: int):
        if readout < 1 or readout > len(parts.layers):
            raise ValueError(f"Invalid post-block readout {readout}")
        self.position = int(position)
        self.value = None
        self.handle = parts.layers[readout - 1].register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = _hidden(output)
        if hidden.shape[0] != 1:
            raise ValueError("PositionReadoutCapture requires a one-example forward")
        self.value = hidden[0, self.position].detach().float().cpu()

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "PositionReadoutCapture":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class BatchedPositionReadoutAdd:
    """Add one precomputed post-block residual perturbation to each batch row."""

    def __init__(self, parts: Any, readout: int, positions: list[int], deltas: Any):
        if readout < 1 or readout > len(parts.layers):
            raise ValueError(f"Invalid post-block readout {readout}")
        if len(positions) != len(deltas):
            raise ValueError("One token position and perturbation are required per batch row")
        self.positions = tuple(int(position) for position in positions)
        self.deltas = deltas
        self.handle = parts.layers[readout - 1].register_forward_hook(self._add)

    def _add(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = _hidden(output)
        if hidden.shape[0] != len(self.positions):
            raise ValueError("Intervention batch size changed unexpectedly")
        changed = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        cols = torch.as_tensor(self.positions, device=hidden.device)
        changed[rows, cols] = changed[rows, cols] + self.deltas.to(
            device=hidden.device, dtype=hidden.dtype
        )
        return _replace_hidden(output, changed)

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "BatchedPositionReadoutAdd":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class JLensAnswerSubspace:
    """The question-independent A-D decoder subspace of one learned JLens map.

    The learned map transports a source residual ``h`` as ``h @ J.T``. Qwen's
    final RMSNorm multiplies every A-D logit numerator by the same positive,
    activation-dependent scale, so the centered answer evidence is linear in
    ``h`` up to that common scale. This class performs minimum-L2 updates in
    that exact three-dimensional centered numerator subspace.
    """

    def __init__(self, jacobian: Any, norm_weight: Any, answer_rows: Any):
        import torch

        J = jacobian.detach().float().cpu()
        gamma = norm_weight.detach().float().cpu()
        rows = answer_rows.detach().float().cpu()
        if rows.shape[0] != 4:
            raise ValueError("Expected exactly four answer-token unembedding rows")
        # Row-vector convention: (h @ J.T) @ (gamma * w_l), hence decoder_l =
        # (gamma * w_l) @ J.
        decoder = (rows * gamma[None, :]) @ J
        self.decoder = decoder
        self.centered = decoder - decoder.mean(dim=0, keepdim=True)
        self.gram_pinv = torch.linalg.pinv(self.centered @ self.centered.T)
        self.jacobian = J

    def numerator_scores(self, residual: Any) -> Any:
        return residual.detach().float().cpu() @ self.decoder.T

    def centered_scores(self, residual: Any) -> Any:
        return residual.detach().float().cpu() @ self.centered.T

    def _minimum_delta(self, score_change: Any) -> Any:
        return score_change @ self.gram_pinv @ self.centered

    def erase_rank(self, residual: Any, answer_index: int) -> Any:
        import torch

        answer_index = int(answer_index)
        others = [index for index in range(4) if index != answer_index]
        contrast = self.decoder[answer_index] - self.decoder[others].mean(dim=0)
        value = torch.dot(residual.detach().float().cpu(), contrast)
        return -(value / torch.dot(contrast, contrast)) * contrast

    def erase_all_answer_evidence(self, residual: Any) -> Any:
        scores = self.centered_scores(residual)
        return self._minimum_delta(-scores)

    def swap(self, residual: Any, first: int, second: int) -> Any:
        scores = self.centered_scores(residual)
        target = scores.clone()
        target[int(first)] = scores[int(second)]
        target[int(second)] = scores[int(first)]
        return self._minimum_delta(target - scores)

    def orthogonal_matched_control(
        self, delta: Any, question_id: str, seed: int
    ) -> Any:
        import numpy as np
        import torch

        digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], "little", signed=False)
        generator = np.random.default_rng(local_seed)
        random = torch.from_numpy(generator.standard_normal(self.centered.shape[1])).float()
        # Remove the entire centered A-D row-space from the random vector.
        random = random - self._minimum_delta(random @ self.centered.T)
        norm = random.norm()
        if norm <= 1e-8:
            raise RuntimeError("Degenerate answer-orthogonal random control")
        return random * (delta.detach().float().cpu().norm() / norm)

    def diagnostics(self) -> dict[str, float]:
        import torch

        singular = torch.linalg.svdvals(self.centered)
        nonzero = singular[singular > singular.max() * 1e-6]
        return {
            "centered_rank": int(nonzero.numel()),
            "largest_singular_value": float(nonzero.max()),
            "smallest_nonzero_singular_value": float(nonzero.min()),
            "nonzero_condition_number": float(nonzero.max() / nonzero.min()),
        }
