from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import read_metadata


CONDITION_FILES = {
    "baseline": ("baseline_results.json", "subject_answer"),
    "incorrect": ("game_results.json", "new_answer"),
    "neutral": ("neutral_results.json", "new_answer"),
}


def audit(input_dir: str, behavioral_dir: str, output_path: str) -> dict:
    root = Path(input_dir)
    layout = json.loads((root / "run_metadata.json").read_text())["variant_layout"]
    behavior_root = Path(behavioral_dir)
    summary = {"input_dir": str(root), "conditions": {}}
    for condition, (filename, answer_key) in CONDITION_FILES.items():
        behavior = json.loads((behavior_root / filename).read_text())["results"]
        rows = []
        for path in sorted((root / "shards" / condition).glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                canonical = z["canonical_logits"]
                variants = z["variant_logits"]
                meta = read_metadata(z)
            canonical_choice = "ABCD"[int(np.argmax(canonical[-1]))]
            top_variant = layout[int(np.argmax(variants[-1]))]["letter"]
            recorded = behavior[path.stem][answer_key]
            expected_tokens = behavior[path.stem]["call_metadata"]["prompt_tokens"]
            rows.append({
                "finite": bool(np.isfinite(canonical).all() and np.isfinite(variants).all()),
                "shape_ok": canonical.shape == (127, 4) and variants.shape == (127, 8),
                "prompt_length_match": int(meta["prompt_length"]) == int(expected_tokens),
                "canonical_vs_behavior": canonical_choice == recorded,
                "variant_vs_behavior": top_variant == recorded,
                "canonical_vs_variant": canonical_choice == top_variant,
                "generated_check_present": meta.get("generated_vs_lens_agree") is not None,
                "generated_vs_lens": meta.get("generated_vs_lens_agree"),
            })
        n = len(rows)
        checked = [r for r in rows if r["generated_check_present"]]
        summary["conditions"][condition] = {
            "n_shards": n,
            "all_finite": all(r["finite"] for r in rows),
            "all_shapes_ok": all(r["shape_ok"] for r in rows),
            "prompt_length_agreement": sum(r["prompt_length_match"] for r in rows) / n,
            "canonical_vs_historical_behavior_agreement": sum(r["canonical_vs_behavior"] for r in rows) / n,
            "top_variant_vs_historical_behavior_agreement": sum(r["variant_vs_behavior"] for r in rows) / n,
            "canonical_vs_top_variant_agreement": sum(r["canonical_vs_variant"] for r in rows) / n,
            "same_run_generated_checks_available": len(checked),
            "same_run_top_variant_vs_generated_agreement": (
                sum(bool(r["generated_vs_lens"]) for r in checked) / len(checked) if checked else None
            ),
        }
    summary["complete_questions"] = min(x["n_shards"] for x in summary["conditions"].values())
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit vLLM native-lens shards against behavioral runs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--behavioral", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.behavioral, args.output), indent=2))


if __name__ == "__main__":
    main()
