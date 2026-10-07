"""Read-only validation of NavGPT3 safetensors and inference metadata."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct


# Tensor names follow the upstream Qwen3-VL layout plus the action head.
TENSOR_PREFIXES = ("model.visual.", "model.language_model.", "lm_head.", "action_head.")
_CONFIGS = (
    "config.json", "tokenizer_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "processor_config.json", "generation_config.json",
)
_REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json",
             "preprocessor_config.json", "video_preprocessor_config.json")
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
                "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def read_header(path):
    """Return a validated safetensors header and the tensor-payload offset."""
    path = Path(path)
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated safetensors file: {path.name}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 100_000_000 or size < 2 or size + 8 > path.stat().st_size:
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        header = json.loads(stream.read(size), object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path.name}")
    intervals = []
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid tensor descriptor: {key}")
        dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
        if (dtype not in _DTYPE_BYTES or not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)):
            raise ValueError(f"Invalid tensor shape/dtype/offsets: {key}")
        start, end = offsets
        if end - start != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise ValueError(f"Tensor byte length does not match shape: {key}")
        intervals.append((start, end, key))
    cursor = 0
    for start, end, key in sorted(intervals):
        if start != cursor:
            raise ValueError(f"Non-contiguous or overlapping tensor payload: {key}")
        cursor = end
    if not intervals or cursor != path.stat().st_size - size - 8:
        raise ValueError(f"Truncated or trailing tensor payload: {path.name}")
    return header, size + 8


def _inventory(directory):
    index_path = directory / "model.safetensors.index.json"
    index = _json(index_path) if index_path.exists() else None
    if index is not None:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Checkpoint index must contain a nonempty weight_map")
        if any(not isinstance(name, str) or Path(name).name != name
               or not name.endswith(".safetensors") for name in weight_map.values()):
            raise ValueError("Checkpoint shard names must be local safetensors basenames")
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ["model.safetensors"]
    actual_shards = {p.name for p in directory.glob("*.safetensors")}
    if actual_shards != set(shard_names):
        raise ValueError("Checkpoint shards do not match the index (or model.safetensors is missing)")
    tensors, headers = {}, {}
    for name in shard_names:
        header, offset = read_header(directory / name)
        headers[name] = (header, offset)
        for key in header:
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"Tensor occurs in multiple shards: {key}")
            tensors[key] = name
    if index is not None and tensors != index["weight_map"]:
        raise ValueError("Checkpoint index and safetensors headers disagree")
    return tensors, headers, index


def _require_assets(directory):
    missing = [name for name in _REQUIRED if not (directory / name).is_file()]
    if missing:
        raise ValueError("Missing inference assets: " + ", ".join(missing))


def validate_checkpoint(checkpoint):
    """Check tensor names, shard integrity, index, and inference metadata, without GPU use."""
    directory = Path(checkpoint)
    _require_assets(directory)
    tensors, headers, _ = _inventory(directory)
    unexpected = sorted(key for key in tensors if not key.startswith(TENSOR_PREFIXES))
    if unexpected:
        raise ValueError(f"Unexpected tensor names (first: {unexpected[0]}); "
                         f"expected names starting with {', '.join(TENSOR_PREFIXES)}")
    if not any(key.startswith("action_head.") for key in tensors):
        raise ValueError("Checkpoint has no action_head tensors")
    config = _json(directory / "config.json")
    if config.get("model_type") != "navgpt3" or config.get("architectures") != ["NavGPT3ForConditionalGeneration"]:
        raise ValueError("config.json must declare the NavGPT3ForConditionalGeneration architecture")
    for key, expected in (("text_config", "qwen3_vl_text"), ("vision_config", "qwen3_vl_vision")):
        if config.get(key, {}).get("model_type") != expected:
            raise ValueError(f"config.json {key}.model_type must be {expected}")
    for name, field, expected in (
        ("preprocessor_config.json", "image_processor_type", "NavGPT3ImageProcessor"),
        ("video_preprocessor_config.json", "video_processor_type", "Qwen3VLVideoProcessor"),
    ):
        if _json(directory / name).get(field) != expected:
            raise ValueError(f"{name} {field} must be {expected}")
    for name in _CONFIGS:
        if (directory / name).exists():
            data = _json(directory / name)
            if "auto_map" in data:
                raise ValueError(f"External auto_map is unsupported in {name}")
            if data.get("processor_class", "NavGPT3Processor") != "NavGPT3Processor":
                raise ValueError(f"{name} must use NavGPT3Processor")
    return {"format": "navgpt3-v2", "tensor_count": len(tensors), "shard_count": len(headers),
            "tensor_bytes": sum((directory / n).stat().st_size - offset for n, (_, offset) in headers.items())}
