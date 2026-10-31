import numpy as np
import pytest

from go_pipeline.dataset import split_by_game
from go_pipeline.train import TrainConfig, train


def _tiny_npz(tmp_path):
    rng = np.random.default_rng(0)
    n = 48
    policy = rng.random((n, 82), dtype=np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    data = tmp_path / "tiny.npz"
    np.savez(data, features=rng.random((n, 12, 9, 9), dtype=np.float32),
             policy=policy, value=rng.uniform(-1, 1, n).astype(np.float32),
             game_id=np.repeat(np.arange(6), 8))
    return data


def test_train_one_epoch_cpu(tmp_path):
    out = tmp_path / "student.pt"
    train(TrainConfig(data=str(_tiny_npz(tmp_path)), out=str(out), epochs=1,
                      batch_size=16, val_fraction=0.33, num_workers=0, device="cpu", amp=False))
    assert out.exists()
    history = (tmp_path / "student.pt.history.json").read_text()
    assert "val_int8" in history  # checkpoint selection scores the INT8 artifact


def test_train_no_validation_split(tmp_path):
    out = tmp_path / "student.pt"
    train(TrainConfig(data=str(_tiny_npz(tmp_path)), out=str(out), epochs=1,
                      batch_size=16, val_fraction=0.0, num_workers=0, device="cpu", amp=False))
    assert out.exists()


def test_split_rejects_single_game():
    data = {"value": np.zeros(4, dtype=np.float32), "game_id": np.zeros(4, dtype=np.int64)}
    with pytest.raises(ValueError):
        split_by_game(data, 0.1, seed=1)
    train_idx, val_idx = split_by_game(data, 0.0, seed=1)
    assert len(train_idx) == 4 and len(val_idx) == 0
