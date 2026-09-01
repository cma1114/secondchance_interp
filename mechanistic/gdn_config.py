from __future__ import annotations

from dataclasses import dataclass, field

from .config import ExperimentConfig


@dataclass
class GDNExperimentConfig(ExperimentConfig):
    natural_attention_dir: str = "outputs/causal/qwen36_27b_simplemc_attention_edge"
    mechanistic_dir: str = "outputs/mechanistic/qwen36_27b_simplemc"
    structural_controls: int = 4
    screen_sources: list[str] = field(
        default_factory=lambda: ["user_incorrect", "structural_0", "structural_1", "structural_2", "structural_3"]
    )
    bootstrap_samples: int = 10_000

    def validate(self) -> None:
        super().validate()
        if self.conditions != ["incorrect"]:
            raise ValueError("The GDN experiment runs only the Game (`incorrect`) condition")
        if self.batch_size != 1:
            raise ValueError("The GDN experiment currently requires batch_size=1")
        if self.structural_controls < 2:
            raise ValueError("Use at least two random structural-token controls")
        expected = {"user_incorrect"} | {
            f"structural_{index}" for index in range(self.structural_controls)
        }
        if set(self.screen_sources) != expected:
            raise ValueError(f"screen_sources must be exactly {sorted(expected)}")

