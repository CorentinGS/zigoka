"""Import/export of the `million-go-int8` artifact (go-model.json + .bin).

Mirrors the contract enforced by tools/gen_model_layout.zig: format
`million-go-int8`, version 1, no quantization groups, every weight
per-output-channel symmetric INT8 with a scale block, every bias float32,
all offsets 4-byte aligned and in bounds. Manifest conv weights are
[out, kh, kw, in]; torch conv weights are [out, in, kh, kw].
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .model import MOKA_V1, Architecture, MokaNet, manifest_map

ARCHITECTURE = MOKA_V1.to_manifest()
FORMAT = "million-go-int8"
VERSION = 1


@lru_cache(maxsize=None)
def tensor_specs(arch: Architecture = MOKA_V1) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """(manifest name, manifest shape, state_dict key) derived from MokaNet."""
    reference = MokaNet(arch)
    key_of = dict(manifest_map(arch))
    return tuple(
        (name, tuple(_to_manifest(name, tensor).shape), key_of[name])
        for name, tensor in reference.manifest_items()
    )


def _to_manifest(name: str, tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if name.endswith(".weight") and arr.ndim == 4:
        arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 1))
    return arr


def _from_manifest(name: str, arr: np.ndarray) -> torch.Tensor:
    if name.endswith(".weight") and arr.ndim == 4:
        arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
    return torch.from_numpy(arr.astype(np.float32))


def _quantize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = arr.reshape(arr.shape[0], -1)
    scale = np.maximum(np.abs(rows).max(axis=1) / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.round(rows / scale[:, None]), -127, 127).astype(np.int8)
    return q.reshape(arr.shape), scale


def dequantize(arr: np.ndarray) -> np.ndarray:
    q, scale = _quantize(arr)
    return (q.reshape(arr.shape[0], -1) * scale[:, None]).reshape(arr.shape).astype(np.float32)


def quantized_snapshot(model: MokaNet) -> MokaNet:
    """Copy of `model` with every weight replaced by its INT8 round-trip.

    Outputs of this copy are exactly (up to float accumulation order) what
    the shipped artifact computes, so checkpoint selection can score the
    artifact instead of the float weights.
    """
    import copy

    snapshot = copy.deepcopy(model)
    state = {}
    for name, tensor in snapshot.manifest_items():
        arr = _to_manifest(name, tensor)
        if name.endswith(".weight"):
            arr = dequantize(arr)
        state[name] = _from_manifest(name, arr)
    snapshot.load_manifest_items(state)
    return snapshot.eval()


def export_model(model: MokaNet, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = dict(model.manifest_items())
    chunks: list[bytes] = []
    tensors: dict[str, dict] = {}
    offset = 0

    def push(data: bytes) -> int:
        nonlocal offset
        pad = (-offset) % 4
        if pad:
            chunks.append(b"\x00" * pad)
            offset += pad
        here = offset
        chunks.append(data)
        offset += len(data)
        return here

    for name, shape, _key in tensor_specs(model.arch):
        arr = _to_manifest(name, items[name])
        assert tuple(arr.shape) == shape, f"{name}: {arr.shape} != {shape}"
        if name.endswith(".weight"):
            q, scale = _quantize(arr)
            tensors[name] = {"dataOffset": push(q.tobytes()), "dtype": "int8",
                             "scaleOffset": push(scale.astype("<f4").tobytes()), "shape": list(shape)}
        else:
            tensors[name] = {"dataOffset": push(arr.astype("<f4").tobytes()), "dtype": "float32", "shape": list(shape)}
    blob = b"".join(chunks)
    manifest = {"architecture": model.arch.to_manifest(), "format": FORMAT,
                "sha256": hashlib.sha256(blob).hexdigest(), "tensors": tensors,
                "version": VERSION, "weightsBytes": len(blob)}
    (out / "go-model.bin").write_bytes(blob)
    (out / "go-model.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parse_manifest(model_dir: str | Path) -> tuple[dict, bytes, Architecture]:
    root = Path(model_dir)
    manifest = json.loads((root / "go-model.json").read_text())
    blob = (root / "go-model.bin").read_bytes()
    if manifest.get("format") != FORMAT:
        raise ValueError(f"format must be {FORMAT}, found {manifest.get('format')!r}")
    if manifest.get("version") != VERSION:
        raise ValueError(f"version must be {VERSION}, found {manifest.get('version')!r}")
    if not isinstance(manifest.get("architecture"), dict):
        raise ValueError("manifest has no architecture object")
    arch = Architecture.from_manifest(manifest["architecture"])
    if len(blob) != manifest.get("weightsBytes"):
        raise ValueError("go-model.bin size does not match weightsBytes")
    if hashlib.sha256(blob).hexdigest() != manifest.get("sha256"):
        raise ValueError("go-model.bin sha256 mismatch")
    return manifest, blob, arch


def _read_tensors(model_dir: str | Path) -> tuple[dict[str, np.ndarray], Architecture]:
    manifest, blob, arch = _parse_manifest(model_dir)
    tensors: dict[str, np.ndarray] = {}
    for name, entry in manifest["tensors"].items():
        if "quantizationGroupSize" in entry:
            raise ValueError(f"tensor {name} uses quantizationGroupSize, unsupported")
        dtype = entry.get("dtype")
        if dtype not in ("int8", "float32"):
            raise ValueError(f"tensor {name} has unsupported dtype {dtype!r}")
        shape = tuple(entry["shape"])
        count = int(np.prod(shape))
        data_offset = int(entry["dataOffset"])
        element_bytes = 4 if dtype == "float32" else 1
        if data_offset % 4 != 0 or data_offset + count * element_bytes > len(blob):
            raise ValueError(f"tensor {name} data offset {data_offset} is misaligned or out of bounds")
        if dtype == "float32":
            tensors[name] = np.frombuffer(blob, "<f4", count, data_offset).reshape(shape).copy()
            continue
        if "scaleOffset" not in entry:
            raise ValueError(f"tensor {name} is int8 but has no scaleOffset")
        scale_offset = int(entry["scaleOffset"])
        if scale_offset % 4 != 0 or scale_offset + shape[0] * 4 > len(blob):
            raise ValueError(f"tensor {name} scale offset {scale_offset} is misaligned or out of bounds")
        q = np.frombuffer(blob, np.int8, count, data_offset).reshape(shape).astype(np.float32)
        scales = np.frombuffer(blob, "<f4", shape[0], scale_offset)
        tensors[name] = (q.reshape(shape[0], count // shape[0]) * scales[:, None]).reshape(shape)
    return tensors, arch


def import_model(model_dir: str | Path, device: torch.device | None = None) -> MokaNet:
    tensors, arch = _read_tensors(model_dir)
    expected = {name for name, _, _ in tensor_specs(arch)}
    if set(tensors) != expected:
        raise ValueError(f"tensor set mismatch: missing={expected - set(tensors)} extra={set(tensors) - expected}")
    items = {name: _from_manifest(name, tensors[name]) for name, _, _ in tensor_specs(arch)}
    model = MokaNet(arch)
    model.load_manifest_items(items)
    return model.to(device) if device is not None else model


def validate_model_dir(model_dir: str | Path) -> dict:
    """Strict Python mirror of tools/gen_model_layout.zig's checks."""
    manifest, _blob, arch = _parse_manifest(model_dir)
    tensors, _ = _read_tensors(model_dir)
    expected = {name: shape for name, shape, _ in tensor_specs(arch)}
    if set(tensors) != set(expected):
        raise ValueError(f"tensor set mismatch: missing={set(expected) - set(tensors)} extra={set(tensors) - set(expected)}")
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name}: shape {tensors[name].shape} != {shape}")
    return {"tensors": len(tensors), "weightsBytes": manifest["weightsBytes"], "sha256": manifest["sha256"],
            "boardSize": arch.board_size}
