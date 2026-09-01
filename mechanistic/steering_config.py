from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SteeringConfig:
    base_config_path: str = "configs/mechinterp_qwen36_simplemc.json"
    directions_path: str = "artifacts/feedback_direction_qwen36_simplemc.npz"
    output_dir: str = "outputs/causal/qwen36_27b_simplemc_feedback"
    conditions: list[str] = field(default_factory=lambda: ["incorrect", "neutral"])
    scan_readouts: list[int] = field(default_factory=lambda: [24, 30, 36])
    scan_doses: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    detailed_readout: int = 30
    detailed_doses: list[float] = field(default_factory=lambda: [-0.5, 0.5, 2.0])
    control_readout: int = 30
    control_doses: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    max_questions: int | None = None
    unsteered_abs_tolerance: float = 0.15
    unsteered_rel_tolerance: float = 0.0075
    seed: int = 42

    @classmethod
    def load(cls, path: str | Path) -> "SteeringConfig":
        data = json.loads(Path(path).read_text())
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown steering config keys: {unknown}")
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        if set(self.conditions) != {"incorrect", "neutral"}:
            raise ValueError("conditions must contain exactly incorrect and neutral")
        if not self.scan_readouts:
            raise ValueError("scan_readouts cannot be empty")
        if 0.0 in self.scan_doses or 0.0 in self.detailed_doses or 0.0 in self.control_doses:
            raise ValueError("Zero-dose conditions are represented by the unsteered scenarios")
        if self.max_questions is not None and self.max_questions < 1:
            raise ValueError("max_questions must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

