"""Shared I/O functions for loading and saving bf16 raw tensors and dump metadata."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def load_raw_bf16(path: str, shape: tuple) -> torch.Tensor:
    """Load a .raw bf16 file into a float32 torch tensor with given shape."""
    data = open(path, "rb").read()
    if len(data) == 0:
        raise ValueError(f"Empty file: {path}")
    arr = np.frombuffer(data, dtype=np.uint16)
    f32_bits = arr.astype(np.uint32) << 16
    return torch.from_numpy(f32_bits.view(np.float32).reshape(shape))


def save_raw_bf16(path: str, tensor: torch.Tensor) -> None:
    """Save a float32 torch tensor as .raw bf16 file (same format as engine dumps).

    Flattens the tensor and writes little-endian bf16 values.
    """
    t = tensor.float().flatten()
    f32_arr = t.numpy()
    i32_bits = f32_arr.view(np.int32)
    bf16_bits = (i32_bits >> 16).astype(np.uint16)
    with open(path, "wb") as f:
        f.write(bf16_bits.tobytes())


def load_meta(path: str) -> dict:
    """Load a .meta JSON sidecar file.

    Returns dict with name, layer, gpu, shape, dtype, stage.
    """
    with open(path) as f:
        return json.load(f)


def discover_dumps(dump_dir: str) -> Dict[int, List[dict]]:
    """Scan a dump directory for all .meta files, organized by layer number.

    Walks `layer_N/` subdirectories looking for `.meta` JSON sidecar files.

    Returns:
        {layer_idx: [meta_dict, ...]}
    """
    result: Dict[int, List[dict]] = {}
    root = Path(dump_dir)
    if not root.exists():
        return result

    for layer_dir in sorted(root.iterdir()):
        if not layer_dir.is_dir():
            continue
        try:
            layer_idx = int(layer_dir.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue

        metas = []
        for meta_path in sorted(layer_dir.glob("*.meta")):
            metas.append(load_meta(str(meta_path)))
        if metas:
            result[layer_idx] = metas

    return result


def discover_final_dumps(dump_dir: str) -> Dict[str, dict]:
    """Scan for final-layer dumps in the `final/` directory.

    Looks for `.meta` JSON sidecar files under `dump_dir/final/`.

    Returns:
        {stage_name: meta_dict}
    """
    result: Dict[str, dict] = {}
    final_dir = Path(dump_dir) / "final"
    if not final_dir.exists():
        return result

    for meta_path in sorted(final_dir.glob("*.meta")):
        meta = load_meta(str(meta_path))
        stage_name = meta.get("stage", meta_path.stem)
        result[stage_name] = meta

    return result
