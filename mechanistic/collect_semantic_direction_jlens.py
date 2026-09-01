from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from mechanistic.config import ExperimentConfig
from mechanistic.modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
)
from mechanistic.run_final_decision_semantic_ablation import (
    LETTERS,
    _load_mapping_plans,
    _semantic_directions,
)


def _display(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def collect(
    config_path: Path,
    baseline_path: Path,
    mapping_plan_paths: list[Path],
    question_ids_path: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
    top_k: int,
) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    baseline = json.loads(baseline_path.read_text())["results"]
    qids = json.loads(question_ids_path.read_text())["question_ids"]
    manifest = json.loads(Path(config.manifest_path).read_text())
    questions = {row["id"]: row for row in manifest["questions"]}
    all_qids = [row["id"] for row in manifest["questions"]]
    mapping_plans = _load_mapping_plans(mapping_plan_paths, all_qids)

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]

    rows: dict[str, dict] = {}
    batch_size = int(config.batch_size)
    for start in range(0, len(qids), batch_size):
        group_qids = qids[start : start + batch_size]
        directions, _audit = _semantic_directions(
            model,
            processor,
            parts,
            tokenizer,
            config,
            questions,
            group_qids,
            mapping_plans,
            baseline,
        )
        # direction: readout, batch, width. JLens layer k transports post-block
        # readout k+1; the final readout is already in output coordinates.
        for row, qid in enumerate(group_qids):
            rows[qid] = {}
        for layer in range(len(parts.layers)):
            positive = torch.from_numpy(directions[layer]).to(device, dtype=torch.float16)
            if layer < len(parts.layers) - 1:
                J = jacobians[layer].to(device, dtype=torch.float16)
                positive = positive @ J.T
                del J
            values = torch.cat((positive, -positive), dim=0)
            with torch.inference_mode():
                normed = parts.final_norm(values.to(parts.final_norm.weight.dtype))
                logits = parts.output_head(normed).float()
                scores, token_ids = torch.topk(logits, k=top_k, dim=-1)
            scores = scores.cpu().numpy()
            token_ids = token_ids.cpu().numpy()
            for row, qid in enumerate(group_qids):
                layer_row = rows[qid].setdefault(str(layer + 1), {})
                for sign, offset in (("positive", 0), ("negative", len(group_qids))):
                    layer_row[sign] = [
                        {
                            "token_id": int(token_id),
                            "token": _display(tokenizer.decode([int(token_id)])),
                            "score": round(float(score), 4),
                        }
                        for score, token_id in zip(
                            scores[offset + row], token_ids[offset + row]
                        )
                    ]
        print(f"semantic-direction JLens: {min(start + batch_size, len(qids))}/{len(qids)}", flush=True)

    audit = {}
    for qid in qids:
        question = questions[qid]
        audit[qid] = {
            "question": question["question"],
            "options": question["options"],
            "correct_answer": question["correct_answer"],
            "w1": baseline[qid]["answer"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "definition": (
            "Exact question- and layer-specific unit W1 semantic direction used by "
            "the causal ablations, decoded under JLens for v_W1 and -v_W1."
        ),
        "question_ids": qids,
        "layers": list(range(1, len(parts.layers) + 1)),
        "top_k": top_k,
        "audit": audit,
        "top_tokens": rows,
    }, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mapping-plans", nargs=3, type=Path, required=True)
    parser.add_argument("--question-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    collect(
        args.config,
        args.baseline,
        args.mapping_plans,
        args.question_ids,
        args.output,
        args.lens_repo,
        args.lens_filename,
        args.top_k,
    )


if __name__ == "__main__":
    main()
