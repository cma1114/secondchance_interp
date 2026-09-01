from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    model_id: str = "Qwen/Qwen3.6-27B"
    model_revision: str | None = None
    manifest_path: str = "outputs/reproduction/simplemc_qwen36_27b/stimulus_manifest.json"
    baseline_results_path: str = "compiled_results_simplemc_qwen36_27b/qwen3.6-27b_phase1_compiled.json"
    output_dir: str = "outputs/mechanistic/qwen36_27b_simplemc"
    conditions: list[str] = field(default_factory=lambda: ["baseline", "incorrect", "neutral"])
    question_ids: list[str] | None = None
    max_questions: int | None = None
    batch_size: int = 1
    trial_major: bool = False
    dtype: str = "bfloat16"
    device_map: str = "auto"
    model_loader: str = "multimodal"
    attn_implementation: str | None = None
    save_residuals: bool = True
    residual_dtype: str = "float16"
    prompt_mode: str = "faithful"
    feedback_variant: str = "standard"
    chat_serialization: str = "hf_template"
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    decision_mode: str = "unrestricted"
    disable_thinking: bool = True
    trust_remote_code: bool = False
    skip_missing_baseline: bool = False
    seed: int = 42
    answer_variants: dict[str, list[str]] = field(
        default_factory=lambda: {letter: [letter, " " + letter] for letter in "ABCD"}
    )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text())
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        valid_conditions = {
            "baseline",
            "incorrect",
            "incorrect_no_system_setup",
            "neutral",
        }
        if not self.conditions or not set(self.conditions) <= valid_conditions:
            raise ValueError(f"conditions must be drawn from {sorted(valid_conditions)}")
        if self.prompt_mode not in {
            "faithful", "clean", "no_system_incorrect", "baseline_matched",
            "baseline_matched_empty_history",
        }:
            raise ValueError(
                "prompt_mode must be 'faithful', 'clean', 'no_system_incorrect', "
                "'baseline_matched', or 'baseline_matched_empty_history'"
            )
        if self.feedback_variant not in {"standard", "token_matched_test"}:
            raise ValueError(
                "feedback_variant must be 'standard' or 'token_matched_test'"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.residual_dtype not in {"float16", "float32"}:
            raise ValueError("residual_dtype must be float16 or float32")
        if self.model_loader not in {"multimodal", "causal_lm"}:
            raise ValueError("model_loader must be 'multimodal' or 'causal_lm'")
        if self.chat_serialization not in {
            "hf_template", "hf_template_direct_assistant",
            "raw_qwen_chatml", "raw_qwen_chatml_bare"
        }:
            raise ValueError(
                "chat_serialization must be 'hf_template', "
                "'hf_template_direct_assistant', 'raw_qwen_chatml', or "
                "'raw_qwen_chatml_bare'"
            )
        if self.decision_mode not in {"unrestricted", "ad_constrained"}:
            raise ValueError("decision_mode must be 'unrestricted' or 'ad_constrained'")
        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a JSON object")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def config_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to an experiment JSON config")
    return parser
