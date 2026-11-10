"""Parity against the oracle fixtures of every model, 9×9 and 19×19.

The fixtures come from Moka's own code (`pnpm oracle` in the repo root):
19×19 positions are played by Moka's rules with the repo's 19×19
constants. They are generated data (gitignored): each test skips where
they are absent instead of passing vacuously.

* Rules: every fixture game is replayed through board.py. The moves
  played since the previous position are read off the recent-move planes
  (at most two plies apart). Features must match bit for bit, and so must
  the legal moves.
* Network: the PyTorch replica of each model reproduces Moka's
  normal-mode outputs (f64 accumulation) to float32 precision.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from go_pipeline import board_spec
from go_pipeline.board import GameState, encode_features, legal_moves, play_move
from go_pipeline.manifest import import_model

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "oracle" / "fixtures"
MODEL_DIRS = {
    "moka-9": ROOT / "vendor" / "moka" / "model",
    "moka-9-v2": ROOT / "vendor" / "moka-v2" / "model",
    "synthetic-19": ROOT / "models" / "synthetic-19",
    "synthetic-19-v2": ROOT / "models" / "synthetic-19-v2",
}
CASES = [(model, split) for model in MODEL_DIRS for split in ("calibration", "validation")]


def _load(model: str, split: str) -> tuple[dict, np.ndarray, int] | None:
    directory = FIXTURES / model / split
    if not (directory / "index.json").exists():
        return None
    index = json.loads((directory / "index.json").read_text())
    size = index["boardSize"]
    area = size * size
    feats = np.fromfile(directory / "features.f32", dtype=np.float32)
    # The fixtures are position-major ([position, point, plane]), like Moka's encoder.
    feats = feats.reshape(index["positions"], area, 12).transpose(0, 2, 1).reshape(-1, 12, size, size)
    return index, feats, size


def _recent_move(features: np.ndarray, offset: int, size: int) -> int:
    """Moka marks the last move on plane 7 and the one before on plane 8,
    or fills plane 9 / 10 for a pass."""
    area = size * size
    if features[8 + offset].all():
        return area
    points = np.flatnonzero(features[6 + offset].reshape(area))
    assert len(points) == 1, "a recent-move plane must mark exactly one point"
    return int(points[0])


@pytest.mark.parametrize(("model", "split"), CASES)
def test_rules_replay_matches_moka(model: str, split: str) -> None:
    loaded = _load(model, split)
    if loaded is None:
        pytest.skip(f"oracle fixtures for {model}/{split} not generated")
    index, feats, size = loaded
    spec = board_spec(size)
    legal_ref = np.fromfile(FIXTURES / model / split / "legal.u8", dtype=np.uint8).reshape(-1, spec.move_count)
    move_counts = np.fromfile(FIXTURES / model / split / "move-count.u16", dtype=np.uint16)
    checked = skipped = 0
    for game in index["games"]:
        state = GameState(size=size)
        for position in range(game["start"], game["start"] + game["count"]):
            gap = int(move_counts[position]) - state.move_count
            if gap > 2:
                # Only the last two moves are visible in the features (the smoke probe).
                skipped += game["start"] + game["count"] - position
                break
            for offset in range(gap, 0, -1):
                state = play_move(state, _recent_move(feats[position], offset, size))
                assert state is not None
            np.testing.assert_array_equal(encode_features(state), feats[position])
            legal = np.zeros(spec.move_count, dtype=np.uint8)
            for move in legal_moves(state):
                if move != spec.pass_move or state.move_count >= spec.min_pass_move:
                    legal[move] = 1
            np.testing.assert_array_equal(legal, legal_ref[position])
            checked += 1
    assert checked + skipped == index["positions"]
    assert checked >= 0.95 * index["positions"]


@pytest.mark.parametrize("model", list(MODEL_DIRS))
def test_network_matches_moka_runtime(model: str) -> None:
    loaded = _load(model, "validation")
    if loaded is None:
        pytest.skip(f"oracle fixtures for {model}/validation not generated")
    index, feats, size = loaded
    spec = board_spec(size)
    n = min(len(feats), 256)
    net = import_model(MODEL_DIRS[model]).eval()
    assert net.arch.board_size == size
    with torch.no_grad():
        logits, value = net(torch.from_numpy(np.ascontiguousarray(feats[:n])))
    directory = FIXTURES / model / "validation"
    log_ref = np.fromfile(directory / "logits-normal.f32", dtype=np.float32)[: n * spec.move_count].reshape(n, -1)
    val_ref = np.fromfile(directory / "value-normal.f64", dtype=np.float64)[:n]
    # Relative to the logit range: synthetic-19's generated heads have logits ~5× wider than moka-9's.
    scale = max(1.0, float(np.abs(log_ref).max()))
    assert float(np.abs(logits.numpy() - log_ref).max()) < 1e-5 * scale * 10
    assert float(np.abs(value.numpy() - val_ref).max()) < 1e-5
    assert float((logits.numpy().argmax(1) == log_ref.argmax(1)).mean()) == 1.0
