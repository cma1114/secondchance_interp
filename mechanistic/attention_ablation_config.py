from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ExperimentConfig


@dataclass
class AttentionAblationConfig(ExperimentConfig):
    natural_attention_dir: str = "outputs/mechanistic/qwen36_27b_simplemc_attention"
    mechanistic_dir: str = "outputs/mechanistic/qwen36_27b_simplemc"
    source_token: str = "user_incorrect"
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    bootstrap_samples: int = 10_000

    def validate(self) -> None:
        super().validate()
        if self.conditions != ["incorrect"]:
            raise ValueError("The attention-edge experiment runs only the Game (`incorrect`) condition")
        if self.attn_implementation != "eager":
            raise ValueError("Attention-edge ablation requires attn_implementation='eager'")
        if self.batch_size != 1:
            raise ValueError("Attention-edge ablation currently requires batch_size=1")
        if not self.scenarios:
            raise ValueError("At least one intervention scenario is required")
        ids: set[str] = set()
        for scenario in self.scenarios:
            if set(scenario) != {"id", "source", "targets"}:
                raise ValueError("Each scenario needs exactly id, source, and targets")
            if scenario["id"] in ids:
                raise ValueError(f"Duplicate scenario ID: {scenario['id']}")
            ids.add(scenario["id"])
            if scenario["source"] not in {"user_incorrect", "system_incorrect", "none"}:
                raise ValueError(f"Unknown source-token selector: {scenario['source']}")
            if scenario["source"] == "none":
                if scenario["targets"]:
                    raise ValueError("A no-op scenario cannot have intervention targets")
            elif not scenario["targets"]:
                raise ValueError(f"Scenario {scenario['id']} has no targets")
            for target in scenario["targets"]:
                if set(target) != {"layer", "heads"} or not target["heads"]:
                    raise ValueError(f"Malformed target in scenario {scenario['id']}")
