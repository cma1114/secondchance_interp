from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import atomic_save_npz, shard_path
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials
from .sublayer_config import SublayerExperimentConfig


def run(
    config: SublayerExperimentConfig,
    calibration_plan_path: Path,
    patch_root: Path,
) -> None:
    import torch

    plan = json.loads(calibration_plan_path.read_text())
    trials = load_trials(
        config.manifest_path, config.baseline_results_path, plan["question_ids"], None
    )
    by_qid = {trial.question_id: trial for trial in trials}
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]

    def control(qids: list[str]) -> dict[str, np.ndarray]:
        prompts = []
        for qid in qids:
            trial = by_qid[qid]
            prompt = render_chat(
                processor,
                build_messages(trial.question, "incorrect", config.prompt_mode),
                config.disable_thinking,
            )
            prompts.extend([prompt] * 24)
        input_ids, attention_mask, _ = tokenize_batch(tokenizer, prompts)
        with torch.inference_mode():
            device = model_input_device(parts)
            kwargs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "use_cache": False,
                "return_dict": True,
            }
            try:
                result = model(**kwargs, logits_to_keep=1)
            except TypeError:
                result = model(**kwargs)
        logits = result.logits.detach().float().cpu()[:, -1, canonical_ids].numpy()
        values = {}
        for index, qid in enumerate(qids):
            rows = logits[index * 24:(index + 1) * 24]
            if float(np.max(np.abs(rows - rows[:1]))) > 0:
                raise RuntimeError("Identical unpatched batch rows produced different logits")
            values[qid] = rows[0]
        return values

    controls: dict[str, np.ndarray] = {}
    batch24_qids = list(plan["batch24_scenarios_by_qid"])
    for completed, qid in enumerate(batch24_qids, 1):
        controls.update(control([qid]))
        if completed == 1 or completed % 20 == 0 or completed == len(batch24_qids):
            print(f"batch-24 controls: completed {completed}/{len(batch24_qids)}", flush=True)
    for completed, qids in enumerate(plan["grid_batches"], 1):
        controls.update(control(qids))
        if completed == 1 or completed % 10 == 0 or completed == len(plan["grid_batches"]):
            print(f"grid controls: completed {completed}/{len(plan['grid_batches'])}", flush=True)

    for mapping_name in ("batch24_scenarios_by_qid", "grid_scenarios_by_qid"):
        for qid, scenarios in plan[mapping_name].items():
            for scenario in scenarios:
                path = shard_path(patch_root, scenario, qid)
                with np.load(path, allow_pickle=False) as data:
                    arrays = {name: data[name] for name in data.files}
                arrays["matched_natural_logits"] = controls[qid].astype(np.float32)
                atomic_save_npz(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add batch-matched controls to head discovery shards")
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration-plan", required=True)
    parser.add_argument("--patch-root", required=True)
    args = parser.parse_args()
    run(
        SublayerExperimentConfig.load(args.config),
        Path(args.calibration_plan),
        Path(args.patch_root),
    )


if __name__ == "__main__":
    main()
