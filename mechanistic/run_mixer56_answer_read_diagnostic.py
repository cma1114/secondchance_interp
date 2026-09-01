from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .attention_spans import attention_span_indices
from .historical_answer_intervention import JLensAnswerSubspace
from .io import atomic_save_npz, json_array, shard_path
from .jlens_collect import ANCHORS, _anchor_positions
from .modeling import (
    get_tokenizer,
    load_model_and_processor,
    model_input_device,
    render_chat,
    resolve_answer_tokens,
    tokenize_batch,
)
from .prompts import build_messages, load_trials, prompt_hash
from .source_partition import prompt_source_partition
from .source_route_collectors import AttentionSourceRouteCollector
from .sublayer_config import SublayerExperimentConfig


MIXER_LAYER = 55  # zero-based model index: Mixer 56
SOURCE_READOUT = 55  # resid_pre[56], i.e. output of one-based layer 55


def _forward(model, parts, input_ids, attention_mask):
    import torch

    with torch.inference_mode():
        kwargs = {
            "input_ids": input_ids.to(model_input_device(parts)),
            "attention_mask": attention_mask.to(model_input_device(parts)),
            "use_cache": False,
            "return_dict": True,
        }
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def _weight_diagnostic(parts, subspace, answer_rows, random_samples: int, seed: int):
    import torch

    attention = parts.layers[MIXER_LAYER].self_attn
    head_dim = int(attention.head_dim)
    v_weight = attention.v_proj.weight.detach().float()
    k_weight = attention.k_proj.weight.detach().float()
    o_weight = attention.o_proj.weight.detach().float()
    hidden = int(v_weight.shape[1])
    num_kv_heads = int(v_weight.shape[0] // head_dim)
    num_heads = int(o_weight.shape[1] // head_dim)
    if num_heads % num_kv_heads:
        raise RuntimeError("Query heads are not an integer multiple of KV heads")
    groups = num_heads // num_kv_heads
    kv_for_query = torch.arange(num_heads, device=v_weight.device) // groups
    v_kv = v_weight.reshape(num_kv_heads, head_dim, hidden)
    k_kv = k_weight.reshape(num_kv_heads, head_dim, hidden)
    v_query = v_kv[kv_for_query]
    k_query = k_kv[kv_for_query]
    o_heads = o_weight.reshape(o_weight.shape[0], num_heads, head_dim)

    decoder = subspace.decoder.to(v_weight.device)
    directions = []
    for letter in range(4):
        others = [index for index in range(4) if index != letter]
        direction = decoder[letter] - decoder[others].mean(dim=0)
        directions.append(direction / direction.norm())
    directions = torch.stack(directions)
    value = torch.einsum("hdx,lx->lhd", v_query, directions)
    key = torch.einsum("hdx,lx->lhd", k_query, directions)
    value_survival = value.norm(dim=-1)
    key_survival = key.norm(dim=-1)

    generator = torch.Generator(device=v_weight.device)
    generator.manual_seed(seed)
    random = torch.randn(random_samples, hidden, generator=generator, device=v_weight.device)
    random = random / random.norm(dim=-1, keepdim=True)
    random_value_kv = torch.einsum("kdx,rx->rkd", v_kv, random).norm(dim=-1)
    random_key_kv = torch.einsum("kdx,rx->rkd", k_kv, random).norm(dim=-1)
    random_value = random_value_kv[:, kv_for_query]
    random_key = random_key_kv[:, kv_for_query]
    value_percentile = (
        random_value[:, None, :] <= value_survival[None, :, :]
    ).float().mean(dim=0)
    key_percentile = (
        random_key[:, None, :] <= key_survival[None, :, :]
    ).float().mean(dim=0)

    # M[h, c, x] maps a source-residual perturbation x through head h's value
    # and output projections into the immediate unembedded logit for answer c.
    rows = answer_rows.detach().float().to(v_weight.device)
    answer_through_o = torch.einsum("co,ohd->chd", rows, o_heads)
    effective_matrix = torch.einsum("chd,hdx->hcx", answer_through_o, v_query)
    direct_ad = torch.einsum("hcx,lx->lhc", effective_matrix, directions)
    direct_contrast = []
    for letter in range(4):
        others = [index for index in range(4) if index != letter]
        direct_contrast.append(
            direct_ad[letter, :, letter]
            - direct_ad[letter][:, others].mean(dim=-1)
        )
    direct_contrast = torch.stack(direct_contrast)
    return {
        "num_heads": np.asarray(num_heads, dtype=np.int32),
        "num_kv_heads": np.asarray(num_kv_heads, dtype=np.int32),
        "head_dim": np.asarray(head_dim, dtype=np.int32),
        "kv_head_for_query": kv_for_query.cpu().numpy().astype(np.int16),
        "answer_directions": directions.cpu().numpy().astype(np.float16),
        "value_survival": value_survival.cpu().numpy().astype(np.float32),
        "key_survival": key_survival.cpu().numpy().astype(np.float32),
        "value_random_percentile": value_percentile.cpu().numpy().astype(np.float32),
        "key_random_percentile": key_percentile.cpu().numpy().astype(np.float32),
        "direct_ad_per_unit_direction": direct_ad.cpu().numpy().astype(np.float32),
        "direct_target_contrast_per_unit_direction": direct_contrast.cpu().numpy().astype(np.float32),
        "value_per_unit_direction": value.cpu().numpy().astype(np.float16),
        "answer_through_output_projection": answer_through_o.permute(1, 0, 2).cpu().numpy().astype(np.float16),
        "value_output_ad_matrix": effective_matrix.cpu().numpy().astype(np.float16),
        "random_value_survival": random_value.cpu().numpy().astype(np.float32),
        "random_key_survival": random_key.cpu().numpy().astype(np.float32),
    }


def run(
    config_path: Path,
    plan_path: Path,
    output: Path,
    lens_repo: str,
    lens_filename: str,
    random_samples: int,
) -> None:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    config = SublayerExperimentConfig.load(config_path)
    if config.attn_implementation != "eager":
        raise ValueError("Exact token attention requires eager attention")
    plan = json.loads(plan_path.read_text())
    qids = plan.get("question_ids", plan.get("confirmation_question_ids"))
    if not qids:
        raise ValueError("Plan has no question IDs")
    trials = load_trials(config.manifest_path, config.baseline_results_path, qids, None)
    lens_path = hf_hub_download(repo_id=lens_repo, filename=lens_filename)
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    model, processor, parts = load_model_and_processor(config)
    tokenizer = get_tokenizer(processor)
    resolved = resolve_answer_tokens(tokenizer, config.answer_variants)
    canonical_ids = [resolved[letter][0][1] for letter in "ABCD"]
    answer_rows = parts.output_head.weight.detach()[canonical_ids]
    subspace = JLensAnswerSubspace(
        checkpoint["J"][SOURCE_READOUT - 1], parts.final_norm.weight, answer_rows
    )
    del checkpoint
    if getattr(parts.layers[MIXER_LAYER], "self_attn", None) is None:
        raise RuntimeError("Mixer 56 is not an ordinary attention layer")

    output.mkdir(parents=True, exist_ok=True)
    weights = _weight_diagnostic(
        parts, subspace, answer_rows, random_samples, config.seed
    )
    atomic_save_npz(output / "weight_diagnostic.npz", **weights)
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": config.as_dict(),
                "plan_path": str(plan_path),
                "question_ids": qids,
                "model_layer_index_zero_based": MIXER_LAYER,
                "component_name": "Mixer 56",
                "source_boundary": "resid_pre[56]",
                "source_readout_in_post_block_figure_convention": SOURCE_READOUT,
                "random_direction_samples": random_samples,
                "lens_repo": lens_repo,
                "lens_filename": lens_filename,
                "software": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "platform": platform.platform(),
            },
            indent=2,
            sort_keys=True,
        )
    )

    audit_path = output / "position_audit.json"
    for completed, trial in enumerate(trials, 1):
        for condition in ("incorrect", "neutral"):
            path = shard_path(output, condition, trial.question_id)
            if path.exists():
                continue
            messages = build_messages(trial.question, condition, config.prompt_mode)
            prompt = render_chat(
                processor, messages, config.disable_thinking, config.chat_serialization
            )
            token_ids, sources = prompt_source_partition(
                tokenizer, prompt, messages, trial.question, condition
            )
            annotated, spans = attention_span_indices(
                tokenizer, prompt, condition, trial.question
            )
            anchors = _anchor_positions(
                tokenizer,
                prompt,
                condition,
                spans,
                messages[0]["content"],
                messages[-1]["content"],
            )
            historical_end = int(anchors[ANCHORS.index("historical_answer_end")])
            input_ids, attention_mask, _ = tokenize_batch(tokenizer, [prompt])
            if token_ids != annotated or token_ids != input_ids[0].tolist():
                raise RuntimeError("Prompt tokenizations disagree")
            collector = AttentionSourceRouteCollector(
                parts,
                MIXER_LAYER,
                {"historical_assistant": sources["historical_assistant"]},
                canonical_ids,
            )
            try:
                result = _forward(model, parts, input_ids, attention_mask)
                arrays = collector.arrays()
                gate = collector.gate[0].detach().float().cpu().numpy()
            finally:
                collector.close()
            final_logits = (
                result.logits[0, -1, canonical_ids].detach().float().cpu().numpy()
            )
            atomic_save_npz(
                path,
                final_canonical_logits=final_logits.astype(np.float32),
                historical_attention_mass=arrays["attention_mass"][0].astype(np.float32),
                token_attention=arrays["attention_token_weights"].astype(np.float16),
                historical_route_direct_ad=arrays["attention_route_direct_ad"][0].astype(np.float32),
                query_output_gate=gate.astype(np.float32),
                token_ids=np.asarray(token_ids, dtype=np.int32),
                historical_positions=np.asarray(sources["historical_assistant"], dtype=np.int32),
                historical_end_position=np.asarray(historical_end, dtype=np.int32),
                metadata=json_array(
                    {
                        "question_id": trial.question_id,
                        "condition": condition,
                        "prompt_hash": prompt_hash(prompt),
                        "historical_tokens": tokenizer.convert_ids_to_tokens(
                            [token_ids[position] for position in sources["historical_assistant"]]
                        ),
                        "historical_decoded_tokens": [
                            tokenizer.decode([token_ids[position]])
                            for position in sources["historical_assistant"]
                        ],
                        "historical_end_token": tokenizer.decode([token_ids[historical_end]]),
                    }
                ),
            )
            if completed == 1 and not audit_path.exists():
                audit_path.write_text(
                    json.dumps(
                        {
                            "question_id": trial.question_id,
                            "condition": condition,
                            "historical_positions": sources["historical_assistant"],
                            "historical_tokens": tokenizer.convert_ids_to_tokens(
                                [token_ids[position] for position in sources["historical_assistant"]]
                            ),
                            "historical_decoded_tokens": [
                                tokenizer.decode([token_ids[position]])
                                for position in sources["historical_assistant"]
                            ],
                            "answer_decodable_endpoint": historical_end,
                            "answer_decodable_endpoint_token": tokenizer.decode(
                                [token_ids[historical_end]]
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
        if completed == 1 or completed % 10 == 0 or completed == len(trials):
            print(f"Mixer 56 answer-read diagnostic: {completed}/{len(trials)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-samples", type=int, default=512)
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument(
        "--lens-filename",
        default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.plan,
        args.output,
        args.lens_repo,
        args.lens_filename,
        args.random_samples,
    )


if __name__ == "__main__":
    main()
