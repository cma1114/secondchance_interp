from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass
class SublayerExperimentConfig(ExperimentConfig):
    bootstrap_samples: int = 10_000
    discovery_fraction: float = 0.5
    candidates_per_kind: int = 4

    def validate(self) -> None:
        super().validate()
        if set(self.conditions) != {"baseline", "incorrect", "neutral"}:
            raise ValueError("Sublayer decomposition requires baseline, incorrect, and neutral")
        if not 0.2 <= self.discovery_fraction <= 0.8:
            raise ValueError("discovery_fraction must be between 0.2 and 0.8")
        if self.candidates_per_kind < 1:
            raise ValueError("candidates_per_kind must be positive")

