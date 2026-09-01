from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .attention_spans import attention_span_indices
from .gdn_config import GDNExperimentConfig
from .gdn_intervention import BetaWriteAblator, GDNTarget
from .gdn_tokens import source_positions
from .io import atomic_save_npz, json_array, shard_path
from .modeling import get_tokenizer, load_model_and_processor, model_input_device, render_chat, resolve_answer_tokens, tokenize_batch
from .prompts import build_messages, load_trials, prompt_hash


def run(config: GDNExperimentConfig, plan_path: Path) -> None:
    import torch

    plan = json.loads(plan_path.read_text())
    trials = load_trials(config.manifest_path, config.baseline_results_path, plan["question_ids"], None)
    model, processor, parts = load_model_and_processor(config); tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    output_dir = Path(config.output_dir)
    for scenario in plan["scenarios"]:
        pending = [trial for trial in trials if not shard_path(output_dir, scenario["id"], trial.question_id).exists()]
        print(f"{scenario['id']}: {len(pending)} pending / {len(trials)} total", flush=True)
        targets = [GDNTarget(int(target["layer"]), None if target["heads"] is None else tuple(target["heads"])) for target in scenario["targets"]]
        for completed, trial in enumerate(pending, 1):
            messages = build_messages(trial.question, "incorrect", config.prompt_mode)
            prompt = render_chat(processor, messages, config.disable_thinking)
            annotated_ids, spans = attention_span_indices(tokenizer, prompt, "incorrect", trial.question)
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
            positions = source_positions(scenario["source"], tokenizer, annotated_ids, spans, trial.question_id, config.structural_controls, config.seed)
            device = model_input_device(parts)
            with BetaWriteAblator(parts, targets, positions), torch.inference_mode():
                kwargs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device), "use_cache": False, "return_dict": True}
                try: output = model(**kwargs, logits_to_keep=1)
                except TypeError: output = model(**kwargs)
            logits = output.logits.detach().float().cpu()[0, -1, canonical_ids].numpy()
            metadata = {"question_id": trial.question_id, "scenario_id": scenario["id"], "source": scenario["source"], "targets": scenario["targets"], "source_positions": positions, "prompt_hash": prompt_hash(prompt)}
            atomic_save_npz(shard_path(output_dir, scenario["id"], trial.question_id), final_canonical_logits=logits.astype(np.float32), metadata=json_array(metadata))
            del output
            if completed == 1 or completed % 25 == 0 or completed == len(pending): print(f"{scenario['id']}: saved {completed}/{len(pending)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out GDN localization interventions")
    parser.add_argument("--config", required=True); parser.add_argument("--plan", required=True)
    args = parser.parse_args(); run(GDNExperimentConfig.load(args.config), Path(args.plan))


if __name__ == "__main__": main()

