"""Compact, memory-mappable training shards.

A shard is a directory of `.npy` files, one per field, plus `meta.json`.
At 19×19 a float32 position is 17 KB of features alone; a shard row is
about 2.3 KB: the 11 binary input planes bit-packed, the komi plane as
one scalar, both policies in float16 and the ownership in int8.

Fields (n rows, S = board size, A = S², M = A + 1):

| field            | dtype   | shape        | meaning                                             |
|------------------|---------|--------------|-----------------------------------------------------|
| planes           | uint8   | [n, P]       | planes 0-10 of encode_features, np.packbits (big)   |
| komi_feature     | float32 | [n]          | value of plane 11                                   |
| policy_raw       | float16 | [n, M]       | teacher raw policy, probabilities                   |
| policy_visits    | float16 | [n, M]       | search visit distribution (zeros if no search)      |
| has_search       | bool    | [n]          | policy_visits / value_search / score are present    |
| value_raw        | float32 | [n]          | teacher raw value, side to move, [-1, 1]            |
| value_search     | float32 | [n]          | search value, side to move, [-1, 1]                 |
| score_lead       | float32 | [n]          | search score lead, side to move, points             |
| outcome          | float32 | [n]          | game result, side to move: -1 / 0 / 1, NaN unknown  |
| ownership        | int8    | [n, A]       | ownership ×127, side to move (0 if absent)          |
| has_ownership    | bool    | [n]          |                                                     |
| game_id          | int64   | [n]          | split unit: whole games                             |
| turn             | int16   | [n]          | move number of the position                         |

P = ceil(11 · A / 8). Values are from the side to move's perspective,
like the student's value head.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import INPUT_PLANES

SCHEMA_VERSION = 1
BINARY_PLANES = INPUT_PLANES - 1
KOMI_PLANE = INPUT_PLANES - 1
OWNERSHIP_SCALE = 127.0

FIELDS = ("planes", "komi_feature", "policy_raw", "policy_visits", "has_search", "value_raw", "value_search",
          "score_lead", "outcome", "ownership", "has_ownership", "game_id", "turn")


def packed_bytes(size: int) -> int:
    return (BINARY_PLANES * size * size + 7) // 8


def pack_features(features: np.ndarray) -> tuple[np.ndarray, float]:
    """[12, S, S] float32 features -> (packed planes 0-10, komi plane value)."""
    binary = features[:BINARY_PLANES]
    if not np.isin(binary, (0.0, 1.0)).all():
        raise ValueError("planes 0-10 must be binary")
    komi = features[KOMI_PLANE]
    if not (komi == komi.flat[0]).all():
        raise ValueError("the komi plane must be constant")
    return np.packbits(binary.astype(np.uint8).reshape(-1)), float(komi.flat[0])


def unpack_features(planes: np.ndarray, komi_feature: np.ndarray, size: int) -> np.ndarray:
    """[n, P] uint8 + [n] -> [n, 12, S, S] float32 (CPU reference of the GPU decoder)."""
    n = len(planes)
    area = size * size
    bits = np.unpackbits(planes, axis=1, count=BINARY_PLANES * area).reshape(n, BINARY_PLANES, size, size)
    out = np.empty((n, INPUT_PLANES, size, size), dtype=np.float32)
    out[:, :BINARY_PLANES] = bits
    out[:, KOMI_PLANE] = komi_feature.astype(np.float32)[:, None, None]
    return out


@dataclass
class ShardWriter:
    """Accumulates rows, then writes one shard directory."""

    board_size: int
    rows: dict[str, list] = field(default_factory=lambda: {name: [] for name in FIELDS})

    def __len__(self) -> int:
        return len(self.rows["game_id"])

    def add(self, *, features: np.ndarray, policy_raw: np.ndarray, value_raw: float, game_id: int, turn: int,
            policy_visits: np.ndarray | None = None, value_search: float | None = None,
            score_lead: float | None = None, ownership: np.ndarray | None = None,
            outcome: float | None = None) -> None:
        size = self.board_size
        area, moves = size * size, size * size + 1
        if features.shape != (INPUT_PLANES, size, size):
            raise ValueError(f"features shape {features.shape} is not {(INPUT_PLANES, size, size)}")
        planes, komi = pack_features(features)
        has_search = policy_visits is not None
        if has_search and (value_search is None or score_lead is None):
            raise ValueError("a searched position needs value_search and score_lead")
        self.rows["planes"].append(planes)
        self.rows["komi_feature"].append(komi)
        self.rows["policy_raw"].append(_distribution(policy_raw, moves))
        self.rows["policy_visits"].append(_distribution(policy_visits, moves) if has_search
                                          else np.zeros(moves, dtype=np.float32))
        self.rows["has_search"].append(has_search)
        self.rows["value_raw"].append(float(value_raw))
        self.rows["value_search"].append(float(value_search) if has_search else 0.0)
        self.rows["score_lead"].append(float(score_lead) if has_search else 0.0)
        self.rows["outcome"].append(np.nan if outcome is None else float(outcome))
        if ownership is not None:
            own = np.asarray(ownership, dtype=np.float32).reshape(area)
            self.rows["ownership"].append(np.clip(np.round(own * OWNERSHIP_SCALE), -127, 127).astype(np.int8))
        else:
            self.rows["ownership"].append(np.zeros(area, dtype=np.int8))
        self.rows["has_ownership"].append(ownership is not None)
        self.rows["game_id"].append(int(game_id))
        self.rows["turn"].append(int(turn))

    def set_outcome(self, game_id: int, black_result: float, colors: list[int]) -> None:
        """Fill the outcome of every row of `game_id`: `black_result` is +1 / 0 / -1
        for Black, `colors` the side to move of each of those rows, in order."""
        rows = [i for i, g in enumerate(self.rows["game_id"]) if g == game_id]
        if len(rows) != len(colors):
            raise ValueError(f"game {game_id} has {len(rows)} rows but {len(colors)} colors")
        for i, color in zip(rows, colors):
            self.rows["outcome"][i] = float(black_result * color)

    def write(self, directory: str | Path, extra_meta: dict | None = None) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        dtypes = {"planes": np.uint8, "komi_feature": np.float32, "policy_raw": np.float16,
                  "policy_visits": np.float16, "has_search": np.bool_, "value_raw": np.float32,
                  "value_search": np.float32, "score_lead": np.float32, "outcome": np.float32,
                  "ownership": np.int8, "has_ownership": np.bool_, "game_id": np.int64, "turn": np.int16}
        for name in FIELDS:
            np.save(out / f"{name}.npy", np.asarray(self.rows[name], dtype=dtypes[name]))
        meta = {"schemaVersion": SCHEMA_VERSION, "boardSize": self.board_size, "rows": len(self)}
        meta.update(extra_meta or {})
        (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return out


def _distribution(values: np.ndarray, moves: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) != moves:
        raise ValueError(f"policy has {len(arr)} entries, expected {moves}")
    arr = np.where(arr > 0, arr, 0.0)
    total = arr.sum()
    if not total > 0:
        raise ValueError("policy has no positive mass")
    return (arr / total).astype(np.float32)


def read_shard(directory: str | Path, mmap: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    root = Path(directory)
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"{root}: schema version {meta.get('schemaVersion')} is not {SCHEMA_VERSION}")
    mode = "r" if mmap else None
    data = {name: np.load(root / f"{name}.npy", mmap_mode=mode) for name in FIELDS}
    rows = {name: len(arr) for name, arr in data.items()}
    if set(rows.values()) != {meta["rows"]}:
        raise ValueError(f"{root}: field lengths {rows} differ from meta rows {meta['rows']}")
    size = meta["boardSize"]
    if data["planes"].shape[1] != packed_bytes(size) or data["policy_raw"].shape[1] != size * size + 1:
        raise ValueError(f"{root}: field widths do not match board size {size}")
    return data, meta


def shard_dirs(patterns: list[str]) -> list[Path]:
    """Shard directories: each pattern is a shard, a directory of shards, or a glob."""
    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if (path / "meta.json").exists():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p.parent for p in path.glob("*/meta.json")))
        else:
            found.extend(sorted(p.parent for p in Path().glob(f"{pattern}/meta.json")))
    if not found:
        raise ValueError(f"no shard found in {patterns}")
    return found


def load_shards(patterns: list[str], mmap: bool = False) -> tuple[dict[str, np.ndarray], int]:
    """Concatenate shards. Game ids are offset per shard so they stay unique."""
    parts, size, offset = [], None, 0
    for directory in shard_dirs(patterns):
        data, meta = read_shard(directory, mmap=mmap)
        if size is None:
            size = meta["boardSize"]
        elif meta["boardSize"] != size:
            raise ValueError(f"{directory}: board size {meta['boardSize']} differs from {size}")
        if meta["rows"] == 0:
            continue
        data = dict(data)
        data["game_id"] = np.asarray(data["game_id"]) + offset
        offset = int(data["game_id"].max()) + 1
        parts.append(data)
    if not parts:
        raise ValueError(f"every shard in {patterns} is empty")
    if len(parts) == 1:
        return parts[0], size
    return {name: np.concatenate([p[name] for p in parts]) for name in FIELDS}, size
