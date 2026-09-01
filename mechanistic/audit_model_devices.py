from __future__ import annotations

import json
from collections import Counter

from .config import ExperimentConfig, config_arg_parser
from .modeling import load_model_and_processor


def main() -> None:
    args = config_arg_parser("Audit a loaded model's parameter device placement").parse_args()
    config = ExperimentConfig.load(args.config)
    model, _processor, _parts = load_model_and_processor(config)

    device_map = getattr(model, "hf_device_map", None)
    print("hf_device_map=" + json.dumps(device_map, indent=2, sort_keys=True), flush=True)

    counts: Counter[str] = Counter()
    cpu_parameters: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        device = str(parameter.device)
        counts[device] += parameter.numel()
        if parameter.device.type == "cpu":
            cpu_parameters.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
            )
    print("parameter_elements_by_device=" + json.dumps(counts, indent=2, sort_keys=True), flush=True)
    print("cpu_parameters=" + json.dumps(cpu_parameters, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
