"""Dataset: teacher-labelled positions in a single .npz.

Arrays: features [N,12,9,9] f32, policy [N,82] f32 (probabilities),
value [N] f32 in [-1,1], game_id [N] i64 (split unit: whole games).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import AREA


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        out = {k: data[k] for k in data.files}
    if "game_id" not in out:
        out["game_id"] = np.zeros(len(out["value"]), dtype=np.int64)
    return out


def split_by_game(data: dict[str, np.ndarray], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    games = np.unique(data["game_id"])
    if val_fraction <= 0:
        return np.arange(len(data["value"])), np.empty(0, dtype=np.int64)
    if len(games) < 2:
        raise ValueError("validation split needs at least 2 distinct game_id values")
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    val_count = min(len(games) - 1, max(1, int(len(games) * val_fraction)))
    val_games = set(games[:val_count].tolist())
    is_val = np.array([g in val_games for g in data["game_id"]])
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("game split produced an empty train or validation set")
    return train_idx, val_idx


def apply_symmetry(features: torch.Tensor, policy: torch.Tensor, k: int, flip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Dihedral transform; the pass logit (index 81) is invariant."""
    def transform(x: torch.Tensor) -> torch.Tensor:
        if flip:
            x = torch.flip(x, dims=[-1])
        return torch.rot90(x, k, dims=[-2, -1]) if k else x

    spatial = transform(policy[:-1].reshape(9, 9)).reshape(AREA)
    return transform(features), torch.cat([spatial, policy[-1:]])


class GoDataset(Dataset):
    """Symmetry augmentation draws from a per-(epoch, item) seeded RNG,
    so training is reproducible regardless of DataLoader worker count."""

    def __init__(self, data: dict[str, np.ndarray], indices: np.ndarray, augment: bool = False, seed: int = 0) -> None:
        self.features = torch.from_numpy(data["features"][indices])
        self.policy = torch.from_numpy(data["policy"][indices])
        self.value = torch.from_numpy(data["value"][indices])
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, policy, value = self.features[i], self.policy[i], self.value[i]
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + i)
            feats, policy = apply_symmetry(feats, policy, k=int(rng.integers(0, 4)), flip=bool(rng.integers(0, 2)))
        return feats, policy, value
