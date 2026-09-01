from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def shard_path(output_dir: str | Path, condition: str, question_id: str) -> Path:
    return Path(output_dir) / "shards" / condition / f"{question_id}.npz"


def atomic_save_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def json_array(value: Any) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True))


def read_metadata(npz: Any) -> dict[str, Any]:
    return json.loads(str(npz["metadata"].item()))

