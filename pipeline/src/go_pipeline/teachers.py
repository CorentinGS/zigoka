"""Teachers: anything that labels a GameState with (policy probs, value).

- CheckpointTeacher: any MokaNet checkpoint, including the published
  moka-9 artifact imported from its go-model.bin (bootstrap / DAgger).
- KataGoTeacher (katago.py): a KataGo ONNX model on ONNX Runtime
  (CUDAExecutionProvider on NVIDIA), for fresh teacher distillation.

Teachers always return temperature-1 probabilities; rollout temperature
is applied exactly once, at sampling time in generate.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .board import GameState, encode_features
from .model import MokaNet, load_checkpoint


class Teacher(Protocol):
    def label_state(self, state: GameState) -> tuple[np.ndarray, float]: ...


class CheckpointTeacher:
    def __init__(self, model: MokaNet, device: torch.device) -> None:
        self.model = model.eval().to(device)
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "CheckpointTeacher":
        return cls(load_checkpoint(path, device), device)

    @classmethod
    def from_model_dir(cls, path: str | Path, device: torch.device) -> "CheckpointTeacher":
        from .manifest import import_model

        return cls(import_model(path, device), device)

    @torch.no_grad()
    def label_state(self, state: GameState) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(encode_features(state)[None]).to(self.device, non_blocking=True)
        logits, value = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        return probs.cpu().numpy().astype(np.float32), float(value[0])
