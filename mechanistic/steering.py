from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SteeringSpec:
    condition: str
    direction_kind: str
    readout: int | None
    dose: float

    @property
    def scenario_id(self) -> str:
        if self.direction_kind == "none":
            return f"{self.condition}__unsteered"
        dose = f"{abs(self.dose):g}".replace(".", "p")
        sign = "p" if self.dose >= 0 else "m"
        return f"{self.condition}__{self.direction_kind}__l{self.readout}__{sign}{dose}"


def build_schedule(
    conditions: list[str],
    scan_readouts: list[int],
    scan_doses: list[float],
    detailed_readout: int,
    detailed_doses: list[float],
    control_readout: int,
    control_doses: list[float],
) -> list[SteeringSpec]:
    specs = [SteeringSpec(condition, "none", None, 0.0) for condition in conditions]
    for readout in scan_readouts:
        for condition in conditions:
            for dose in scan_doses:
                specs.append(SteeringSpec(condition, "feedback", readout, float(dose)))
    for condition in conditions:
        for dose in detailed_doses:
            specs.append(SteeringSpec(condition, "feedback", detailed_readout, float(dose)))
        for dose in control_doses:
            specs.append(SteeringSpec(condition, "control", control_readout, float(dose)))
    unique: dict[str, SteeringSpec] = {}
    for spec in specs:
        unique[spec.scenario_id] = spec
    return list(unique.values())


class ResidualSteerer:
    """Add one scaled direction to the final prompt position after a readout."""

    def __init__(
        self,
        parts: Any,
        readout: int,
        last_indices: list[int],
        direction: Any,
        scale: float,
        dose: float,
    ):
        if not 0 <= readout <= len(parts.layers):
            raise ValueError(f"readout must be between 0 and {len(parts.layers)}")
        self.last_indices = last_indices
        self.direction = direction
        self.scale = float(scale)
        self.dose = float(dose)
        module = parts.embedding if readout == 0 else parts.layers[readout - 1]
        self.handle = module.register_forward_hook(self._hook())

    def _hook(self) -> Callable:
        def steer(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            modified = hidden.clone()
            direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
            delta = direction * (self.scale * self.dose)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            indices = torch.as_tensor(self.last_indices, device=hidden.device)
            modified[batch, indices] = modified[batch, indices] + delta
            if isinstance(output, tuple):
                return (modified, *output[1:])
            if isinstance(output, list):
                return [modified, *output[1:]]
            return modified

        return steer

    def close(self) -> None:
        self.handle.remove()

