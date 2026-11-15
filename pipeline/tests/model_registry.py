"""The registered models, read from the repository's model registry.

`build.zig` validates `models/registry.json`; this module trusts it.
"""
from __future__ import annotations

import json
import pathlib
from typing import NamedTuple

REGISTRY = pathlib.Path(__file__).resolve().parents[2] / "models" / "registry.json"


class RegisteredModel(NamedTuple):
    id: str
    directory: pathlib.Path
    """The model's source folder, which holds go-model.json and go-model.bin."""
    kind: str
    """"trained" or "synthetic"."""


def registered_models(registry: pathlib.Path = REGISTRY) -> list[RegisteredModel]:
    """The registered models, in the registry's order.

    The repository root is the parent of the registry's folder.
    """
    root = registry.resolve().parents[1]
    entries = json.loads(registry.read_text())["models"]
    return [RegisteredModel(entry["id"], root / entry["directory"], entry["kind"]) for entry in entries]
