#!/usr/bin/env bash
set -euo pipefail

cd /root/secondchance_interp
export HF_TOKEN="$(< /root/.hf_token)"

python -m pip install \
  transformers==5.14.1 \
  accelerate==1.14.0 \
  safetensors==0.8.0 \
  huggingface-hub==1.28.0

python - <<'PY'
import os
import torch
import transformers
from huggingface_hub import snapshot_download

print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, flush=True)

path = snapshot_download(
    repo_id="Qwen/Qwen3.6-27B",
    revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    token=os.environ["HF_TOKEN"],
)
print({"snapshot_path": path}, flush=True)
PY

rm -f /root/.hf_token
echo SETUP_COMPLETE
