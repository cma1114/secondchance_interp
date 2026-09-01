from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .modeling import get_tokenizer, load_model_and_processor, model_input_device


def _display(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _sample_qids(metadata: dict, baseline_results: Path) -> list[str]:
    rows = json.loads(baseline_results.read_text())["results"]
    selected: list[str] = []
    for letter in "ABCD":
        for qid in metadata["question_ids"]:
            if rows[qid]["answer"] == letter:
                selected.append(qid)
                break
    if len(selected) != 4:
        raise ValueError("Could not select one discovery question per Baseline letter")
    return selected


def collect(
    config_path: Path,
    residual_root: Path,
    baseline_results: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
    top_k: int,
) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    config = ExperimentConfig.load(config_path)
    metadata = json.loads((residual_root / "metadata.json").read_text())
    residuals = np.load(residual_root / "position_residuals.npy", mmap_mode="r")
    qids = _sample_qids(metadata, baseline_results)
    qid_to_index = {qid: index for index, qid in enumerate(metadata["question_ids"])}
    question_indices = np.asarray([qid_to_index[qid] for qid in qids], dtype=np.int64)
    anchors = [
        f"{kind}_{letter}"
        for kind in ("content_end", "line_end")
        for letter in "ABCD"
    ]
    anchor_indices = [metadata["anchors"].index(anchor) for anchor in anchors]

    lens_path = hf_hub_download(
        repo_id=lens_repo,
        filename=lens_filename,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    jacobians = checkpoint["J"]
    if sorted(int(value) for value in jacobians) != list(range(63)):
        raise ValueError("JLens checkpoint does not contain source layers 0--62")

    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    device = model_input_device(parts)
    sample = np.asarray(residuals[question_indices][:, :, anchor_indices]).copy()
    # sample: question, layer, anchor, width

    rows: dict[str, dict] = {qid: {} for qid in qids}
    for layer in range(64):
        natural = torch.from_numpy(sample[:, layer].reshape(-1, sample.shape[-1])).to(
            device, dtype=torch.float16
        )
        methods = {"logit_lens": natural}
        if layer < 63:
            J = jacobians[layer].to(device, dtype=torch.float16)
            methods["jlens"] = natural @ J.T
            del J
        else:
            methods["jlens"] = natural

        for method, values in methods.items():
            with torch.inference_mode():
                normed = parts.final_norm(values.to(parts.final_norm.weight.dtype))
                logits = parts.output_head(normed)
                scores, token_ids = torch.topk(logits.float(), k=top_k, dim=-1)
            scores = scores.cpu().numpy()
            token_ids = token_ids.cpu().numpy()
            for qi, qid in enumerate(qids):
                layer_row = rows[qid].setdefault(str(layer + 1), {})
                method_row = layer_row.setdefault(method, {})
                for ai, anchor in enumerate(anchors):
                    flat = qi * len(anchors) + ai
                    method_row[anchor] = [
                        {
                            "token_id": int(token_id),
                            "token": _display(tokenizer.decode([int(token_id)])),
                            "score": round(float(score), 4),
                        }
                        for token_id, score in zip(token_ids[flat], scores[flat])
                    ]
        if layer == 0 or (layer + 1) % 8 == 0 or layer == 63:
            print(f"contextual option lens diagnostics: {layer + 1}/64", flush=True)

    audit = {}
    for qid in qids:
        source = metadata["audit"][qid]
        audit[qid] = {
            "baseline_answer": json.loads(baseline_results.read_text())["results"][qid]["answer"],
            "options": source["options"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "question_ids": qids,
        "anchors": anchors,
        "methods": ["jlens", "logit_lens"],
        "layers": list(range(1, 65)),
        "top_k": top_k,
        "lens_repo": lens_repo,
        "lens_filename": lens_filename,
        "audit": audit,
        "top_tokens": rows,
    }, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    collect(
        args.config,
        args.residual_root,
        args.baseline_results,
        args.output,
        args.lens_repo,
        args.lens_filename,
        args.top_k,
    )


if __name__ == "__main__":
    main()
