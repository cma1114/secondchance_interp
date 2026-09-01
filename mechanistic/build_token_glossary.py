from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI


def selected_non_ascii_tokens(source: Path, top_n: int, exclude_system: bool) -> list[str]:
    document = json.loads(source.read_text())
    tokens: set[str] = set()
    for key, row in document["positions"].items():
        if exclude_system and "/system_" in key:
            continue
        for side in ("top", "bottom"):
            # Rank pseudo-tokens are always displayed in addition to the first
            # `top_n` ordinary vocabulary rows, so they must not consume the
            # glossary selection quota.
            ordinary = [item for item in row[side] if not item.get("tracked")]
            for item in ordinary[:top_n]:
                token = item["token"]
                if any(ord(character) > 127 for character in token):
                    tokens.add(token)
    return sorted(tokens)


def parse_json_object(text: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]!r}")
    return json.loads(match.group(0))


def translate_batch(client: OpenAI, model: str, tokens: list[str]) -> dict[str, str]:
    numbered = [{"id": index, "token": token} for index, token in enumerate(tokens)]
    prompt = (
        "Give each standalone multilingual language-model token a short, literal English gloss. "
        "Translate Chinese, Japanese, Cyrillic, and other non-English text. For a single ambiguous "
        "character, give the principal senses separated by slashes. For punctuation or a letter, "
        "give its English name. Preserve leading-space distinctions only in the source token, not "
        "the gloss. Do not infer anything from experimental context. Return exactly one gloss per "
        "id as JSON of the form {\"items\":[{\"id\":0,\"gloss\":\"...\"}]}.\n\n"
        + json.dumps(numbered, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        parsed = parse_json_object(response.choices[0].message.content or "")
        items = parsed.get("items")
        if not isinstance(items, list):
            raise ValueError("Translation response has no items list")
        by_id = {
            int(item["id"]): str(item["gloss"]).strip()
            for item in items
            if isinstance(item, dict) and "id" in item and "gloss" in item
        }
    except (TypeError, ValueError, json.JSONDecodeError):
        by_id = {}

    translated = {
        token: by_id[index]
        for index, token in enumerate(tokens)
        if index in by_id
    }
    missing_tokens = [token for index, token in enumerate(tokens) if index not in by_id]
    if missing_tokens:
        if len(tokens) == 1:
            raise ValueError(f"Could not translate token {tokens[0]!r}")
        midpoint = max(1, len(missing_tokens) // 2)
        translated.update(translate_batch(client, model, missing_tokens[:midpoint]))
        if midpoint < len(missing_tokens):
            translated.update(translate_batch(client, model, missing_tokens[midpoint:]))
    return translated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--exclude-system", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    tokens = selected_non_ascii_tokens(args.source, args.top_n, args.exclude_system)
    glossary = json.loads(args.output.read_text()) if args.output.exists() else {}
    pending = [token for token in tokens if token not in glossary]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        glossary.update(translate_batch(client, args.model, batch))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(glossary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(f"Translated {min(start + len(batch), len(pending))}/{len(pending)} pending tokens")
    print(f"Glossary contains {len(glossary)} tokens")


if __name__ == "__main__":
    main()
