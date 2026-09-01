from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass
class RunnerInterventionConfig(ExperimentConfig):
    signal_path: str = (
        "outputs/mechanistic/qwen35_397b_popmc/analysis/runner_causal_signal.npz"
    )
    intervention_readout: int = 49
    early_control_readout: int = 40
    strengths: tuple[float, ...] = (0.5, 1.0)
    calibration_steps: int = 3

    def validate(self) -> None:
        super().validate()
        if self.conditions != ["incorrect", "neutral", "baseline"]:
            raise ValueError(
                "Runner intervention conditions must be ['incorrect', 'neutral', 'baseline']"
            )
        if not 0 < self.intervention_readout:
            raise ValueError("intervention_readout must be a post-block readout")
        if not 0 < self.early_control_readout < self.intervention_readout:
            raise ValueError("early_control_readout must precede the primary readout")
        if not self.strengths or any(float(value) <= 0 for value in self.strengths):
            raise ValueError("strengths must be positive")
        if not 1 <= self.calibration_steps <= 8:
            raise ValueError("calibration_steps must be between 1 and 8")
