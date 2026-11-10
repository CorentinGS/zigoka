"""Training data: legacy .npz files and the targets read from shards.

Legacy .npz arrays: features [N,12,S,S] f32, policy [N,S*S+1] f32
(probabilities), value [N] f32 in [-1,1], game_id [N] i64 (split unit:
whole games). Shards (shards.py) carry several labels per position;
`load_training_data` turns either into one set of targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .shards import load_shards


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        out = {k: data[k] for k in data.files}
    if "game_id" not in out:
        out["game_id"] = np.zeros(len(out["value"]), dtype=np.int64)
    return out


def split_by_game(data: dict[str, np.ndarray], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    games = np.unique(data["game_id"])
    if val_fraction <= 0:
        return np.arange(len(data["game_id"])), np.empty(0, dtype=np.int64)
    if len(games) < 2:
        raise ValueError("validation split needs at least 2 distinct game_id values")
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    val_count = min(len(games) - 1, max(1, int(len(games) * val_fraction)))
    is_val = np.isin(data["game_id"], games[:val_count])
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("game split produced an empty train or validation set")
    return train_idx, val_idx


def apply_symmetry(features: torch.Tensor, policy: torch.Tensor, k: int, flip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Dihedral transform; the pass logit (last index) is invariant."""
    def transform(x: torch.Tensor) -> torch.Tensor:
        if flip:
            x = torch.flip(x, dims=[-1])
        return torch.rot90(x, k, dims=[-2, -1]) if k else x

    size = features.shape[-1]
    spatial = transform(policy[:-1].reshape(size, size)).reshape(size * size)
    return transform(features), torch.cat([spatial, policy[-1:]])


def symmetry_permutations(size: int) -> torch.Tensor:
    """[8, S²]: row s maps each point of the transformed board to its source
    point, `out[p] = in[perm[s, p]]`, for the 8 transforms of apply_symmetry."""
    grid = torch.arange(size * size).reshape(1, size, size).float()
    perms = []
    for k in range(4):
        for flip in (False, True):
            moved, _ = apply_symmetry(grid, torch.zeros(size * size + 1), k, flip)
            perms.append(moved.reshape(-1).long())
    return torch.stack(perms)


@dataclass
class TrainingData:
    """Targets of every position, CPU arrays. Inputs are either dense
    features (legacy .npz) or packed planes + komi (shards)."""

    size: int
    policy: np.ndarray                  # [n, M] probabilities
    value: np.ndarray                   # [n] in [-1, 1]
    game_id: np.ndarray                 # [n]
    features: np.ndarray | None = None  # [n, 12, S, S] f32
    planes: np.ndarray | None = None    # [n, P] u8
    komi: np.ndarray | None = None      # [n] f32
    ownership: np.ndarray | None = None  # [n, A] i8, ×127
    has_ownership: np.ndarray | None = None
    score: np.ndarray | None = None     # [n] score lead / scale
    has_score: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.value)


def load_training_data(paths: list[str], visits_mix: float = 0.5, outcome_mix: float = 0.3,
                       score_scale: float = 20.0) -> TrainingData:
    """One .npz, or shards (directories or globs, see shards.shard_dirs).

    Shard targets: policy = visits_mix · visits + (1 - visits_mix) · raw
    where the position was searched, raw elsewhere; value = (1 -
    outcome_mix) · search value (raw value where not searched) +
    outcome_mix · game result where the result is known.
    """
    if len(paths) == 1 and Path(paths[0]).suffix == ".npz":
        data = load_npz(paths[0])
        return TrainingData(size=data["features"].shape[-1], policy=data["policy"].astype(np.float32),
                            value=data["value"].astype(np.float32), game_id=data["game_id"],
                            features=data["features"].astype(np.float32))
    data, size = load_shards(paths)
    searched = np.asarray(data["has_search"])
    raw = np.asarray(data["policy_raw"], dtype=np.float32)
    visits = np.asarray(data["policy_visits"], dtype=np.float32)
    policy = np.where(searched[:, None], visits_mix * visits + (1 - visits_mix) * raw, raw)
    value = np.where(searched, data["value_search"], data["value_raw"]).astype(np.float32)
    outcome = np.asarray(data["outcome"], dtype=np.float32)
    known = np.isfinite(outcome)
    value[known] = (1 - outcome_mix) * value[known] + outcome_mix * outcome[known]
    return TrainingData(size=size, policy=policy.astype(np.float16), value=value, game_id=np.asarray(data["game_id"]),
                        planes=np.asarray(data["planes"]), komi=np.asarray(data["komi_feature"], dtype=np.float32),
                        ownership=np.asarray(data["ownership"]), has_ownership=np.asarray(data["has_ownership"]),
                        score=(np.asarray(data["score_lead"], dtype=np.float32) / score_scale), has_score=searched)
