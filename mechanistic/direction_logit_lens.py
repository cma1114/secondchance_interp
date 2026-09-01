from __future__ import annotations

import argparse
import csv
import json
import os
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
import requests


def _resolve_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def _headers(token: str | None, byte_range: tuple[int, int] | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return headers


def _request_range(
    session: requests.Session,
    url: str,
    token: str | None,
    start: int,
    end: int,
) -> bytes:
    response = session.get(url, headers=_headers(token, (start, end)), timeout=120)
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != end - start + 1:
        raise RuntimeError(
            f"Server did not honor byte range {start}-{end}: "
            f"status={response.status_code}, bytes={len(response.content)}"
        )
    return response.content


def _safetensors_header(
    session: requests.Session,
    url: str,
    token: str | None,
) -> tuple[int, dict]:
    header_size = struct.unpack("<Q", _request_range(session, url, token, 0, 7))[0]
    raw = _request_range(session, url, token, 8, 8 + header_size - 1)
    return 8 + header_size, json.loads(raw)


def _tensor_location(data_start: int, header: dict, name: str) -> tuple[int, int, list[int], str]:
    if name not in header:
        raise KeyError(f"Tensor {name!r} is absent from the safetensors shard")
    item = header[name]
    start, end = item["data_offsets"]
    return data_start + int(start), data_start + int(end), list(item["shape"]), item["dtype"]


def _decode_bfloat16(raw: bytes | bytearray, shape: list[int]) -> np.ndarray:
    import torch

    values = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)
    return values.float().numpy()


def rms_norm_directions(
    directions: np.ndarray,
    norm_weight: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float32)
    variance = np.mean(np.square(directions), axis=-1, keepdims=True)
    denominator = np.sqrt(variance + epsilon)
    denominator = np.where(variance == 0.0, 1.0, denominator)
    normalized = directions / denominator
    return normalized * np.asarray(norm_weight, dtype=np.float32)[None, :]


def _stream_tensor_rows(
    session: requests.Session,
    url: str,
    token: str | None,
    start: int,
    end: int,
    row_bytes: int,
    rows_per_chunk: int,
) -> Iterator[tuple[int, bytearray]]:
    response = session.get(
        url,
        headers=_headers(token, (start, end - 1)),
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Server did not honor streamed tensor range: {response.status_code}")
    buffer = bytearray()
    first_row = 0
    request_bytes = row_bytes * rows_per_chunk
    for piece in response.iter_content(chunk_size=request_bytes):
        if not piece:
            continue
        buffer.extend(piece)
        complete_rows = min(len(buffer) // row_bytes, rows_per_chunk)
        while complete_rows:
            byte_count = complete_rows * row_bytes
            block = buffer[:byte_count]
            del buffer[:byte_count]
            yield first_row, block
            first_row += complete_rows
            complete_rows = min(len(buffer) // row_bytes, rows_per_chunk)
    if buffer:
        raise RuntimeError(f"Stream ended with {len(buffer)} incomplete tensor bytes")


def run(
    repo_id: str,
    revision: str,
    model_index_path: str | Path,
    tokenizer_path: str | Path,
    directions_path: str | Path,
    output_dir: str | Path,
    top_k: int,
    rows_per_chunk: int,
    device: str,
    token: str | None,
) -> dict:
    import torch
    from tokenizers import Tokenizer

    index = json.loads(Path(model_index_path).read_text())["weight_map"]
    head_name = "lm_head.weight"
    norm_name = "model.language_model.norm.weight"
    head_file = index[head_name]
    norm_file = index[norm_name]
    session = requests.Session()

    norm_url = _resolve_url(repo_id, revision, norm_file)
    norm_data_start, norm_header = _safetensors_header(session, norm_url, token)
    norm_start, norm_end, norm_shape, norm_dtype = _tensor_location(
        norm_data_start, norm_header, norm_name
    )
    if norm_dtype != "BF16":
        raise ValueError(f"Expected BF16 final norm, found {norm_dtype}")
    norm_weight = _decode_bfloat16(
        _request_range(session, norm_url, token, norm_start, norm_end - 1), norm_shape
    )

    with np.load(directions_path, allow_pickle=False) as data:
        directions = data["directions"].astype(np.float32)
        direction_metadata = json.loads(str(data["metadata"].item()))
    config = json.loads((Path(tokenizer_path).parent / "config.json").read_text())
    epsilon = float(config["text_config"]["rms_norm_eps"])
    normalized = rms_norm_directions(directions, norm_weight, epsilon)

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_device = torch.device(device)
    normalized_t = torch.from_numpy(normalized).to(torch_device)

    head_url = _resolve_url(repo_id, revision, head_file)
    head_data_start, head_header = _safetensors_header(session, head_url, token)
    head_start, head_end, head_shape, head_dtype = _tensor_location(
        head_data_start, head_header, head_name
    )
    if head_dtype != "BF16" or len(head_shape) != 2:
        raise ValueError(f"Expected a 2D BF16 output head, found {head_dtype} {head_shape}")
    vocab_size, hidden_size = head_shape
    if normalized.shape[1] != hidden_size:
        raise ValueError(
            f"Direction hidden size {normalized.shape[1]} != output head {hidden_size}"
        )

    logits = np.empty((len(directions), vocab_size), dtype=np.float16)
    row_bytes = hidden_size * 2
    total_rows = 0
    for first_row, raw in _stream_tensor_rows(
        session, head_url, token, head_start, head_end, row_bytes, rows_per_chunk
    ):
        rows = len(raw) // row_bytes
        weights = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(rows, hidden_size)
        weights = weights.to(device=torch_device, dtype=torch.float32)
        with torch.inference_mode():
            block_logits = weights @ normalized_t.T
        logits[:, first_row:first_row + rows] = block_logits.T.cpu().numpy().astype(np.float16)
        total_rows += rows
        if total_rows % (rows_per_chunk * 16) == 0 or total_rows == vocab_size:
            print(f"Projected {total_rows}/{vocab_size} vocabulary rows", flush=True)
    if total_rows != vocab_size:
        raise RuntimeError(f"Projected {total_rows} rows, expected {vocab_size}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "feedback_direction_vocab_logits.npz",
        logits=logits,
        metadata=np.asarray(json.dumps({
            "repo_id": repo_id,
            "revision": revision,
            "direction_metadata": direction_metadata,
            "lens": "checkpoint final RMSNorm followed by lm_head",
            "device": device,
        }, sort_keys=True)),
    )

    rows = []
    for layer in range(len(directions)):
        layer_logits = logits[layer].astype(np.float32)
        positive_ids = np.argpartition(layer_logits, -top_k)[-top_k:]
        positive_ids = positive_ids[np.argsort(-layer_logits[positive_ids])]
        negative_ids = np.argpartition(layer_logits, top_k)[:top_k]
        negative_ids = negative_ids[np.argsort(layer_logits[negative_ids])]
        for sign, ids in (("positive", positive_ids), ("negative", negative_ids)):
            for rank, token_id in enumerate(ids, start=1):
                rows.append({
                    "readout": layer,
                    "sign": sign,
                    "rank": rank,
                    "token_id": int(token_id),
                    "token": tokenizer.id_to_token(int(token_id)),
                    "decoded": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    "lens_logit": float(layer_logits[token_id]),
                })
    with (output_dir / "feedback_direction_top_tokens.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_layers = [8, 16, 24, 30, 36, 48, 56, 64]
    summary = {
        "repo_id": repo_id,
        "revision": revision,
        "lens": "checkpoint final RMSNorm followed by lm_head",
        "n_readouts": len(directions),
        "vocab_size": vocab_size,
        "top_k": top_k,
        "selected_readouts": {
            str(layer): {
                sign: [
                    {
                        "rank": row["rank"],
                        "token_id": row["token_id"],
                        "token": row["token"],
                        "decoded": row["decoded"],
                        "lens_logit": row["lens_logit"],
                    }
                    for row in rows
                    if row["readout"] == layer and row["sign"] == sign and row["rank"] <= 20
                ]
                for sign in ("positive", "negative")
            }
            for layer in summary_layers
        },
    }
    (output_dir / "feedback_direction_logit_lens.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a full-vocabulary logit lens over residual directions")
    parser.add_argument("--repo", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-index", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    summary = run(
        args.repo,
        args.revision,
        args.model_index,
        args.tokenizer,
        args.directions,
        args.output,
        args.top_k,
        args.rows_per_chunk,
        args.device,
        os.environ.get(args.token_env),
    )
    print(json.dumps({
        "repo_id": summary["repo_id"],
        "revision": summary["revision"],
        "n_readouts": summary["n_readouts"],
        "vocab_size": summary["vocab_size"],
        "top_k": summary["top_k"],
    }, indent=2))


if __name__ == "__main__":
    main()
