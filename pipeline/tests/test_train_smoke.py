import numpy as np
import pytest
import torch

from go_pipeline.board import GameState, encode_features, sample_legal_move
from go_pipeline.dataset import apply_symmetry, load_training_data, split_by_game, symmetry_permutations
from go_pipeline.model import MokaNet, is_trunk_key, load_checkpoint, preset, save_checkpoint
from go_pipeline.shards import ShardWriter, load_shards, unpack_features
from go_pipeline.train import Store, TrainConfig, augment, decode_planes, train


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
                      batch_size=16, val_fraction=0.33, device="cpu", amp=False))
    assert out.exists()
    history = (tmp_path / "student.pt.history.json").read_text()
    assert "val_int8" in history  # checkpoint selection scores the INT8 artifact


def test_train_no_validation_split(tmp_path):
    out = tmp_path / "student.pt"
    train(TrainConfig(data=str(_tiny_npz(tmp_path)), out=str(out), epochs=1,
                      batch_size=16, val_fraction=0.0, device="cpu", amp=False))
    assert out.exists()


def test_split_rejects_single_game():
    data = {"value": np.zeros(4, dtype=np.float32), "game_id": np.zeros(4, dtype=np.int64)}
    with pytest.raises(ValueError):
        split_by_game(data, 0.1, seed=1)
    train_idx, val_idx = split_by_game(data, 0.0, seed=1)
    assert len(train_idx) == 4 and len(val_idx) == 0


def _write_shards(root, size, games=6, moves=12, seed=0):
    """Random legal games with random labels; every third position unsearched,
    the last game without a known result."""
    rng = np.random.default_rng(seed)
    writer = ShardWriter(size)
    area = size * size
    for game in range(games):
        state = GameState(size=size)
        colors = []
        for _ in range(moves):
            policy = rng.random(area + 1)
            searched = state.move_count % 3 != 2
            writer.add(features=encode_features(state), policy_raw=policy, value_raw=float(rng.uniform(-1, 1)),
                       game_id=game, turn=state.move_count, ownership=rng.uniform(-1, 1, area),
                       policy_visits=rng.random(area + 1) if searched else None,
                       value_search=float(rng.uniform(-1, 1)) if searched else None,
                       score_lead=float(rng.normal(0, 10)) if searched else None)
            colors.append(int(state.next_color))
            _, state = sample_legal_move(state, policy, rng)
        if game < games - 1:
            writer.set_outcome(game, 1.0 if game % 2 else -1.0, colors)
    return writer.write(root / f"shard-{size}")


def test_shard_targets_mix_visits_raw_search_and_outcome(tmp_path):
    shard = _write_shards(tmp_path, 9)
    raw, _ = load_shards([str(shard)])
    data = load_training_data([str(shard)], visits_mix=0.25, outcome_mix=0.4, score_scale=10.0)
    searched = raw["has_search"]
    expected = np.where(searched[:, None], 0.25 * raw["policy_visits"].astype(np.float32)
                        + 0.75 * raw["policy_raw"].astype(np.float32), raw["policy_raw"].astype(np.float32))
    np.testing.assert_allclose(data.policy.astype(np.float32), expected, atol=1e-3)
    base = np.where(searched, raw["value_search"], raw["value_raw"])
    known = np.isfinite(raw["outcome"])
    assert 0 < known.sum() < len(known)
    np.testing.assert_allclose(data.value[known], 0.6 * base[known] + 0.4 * raw["outcome"][known], rtol=1e-6)
    np.testing.assert_allclose(data.value[~known], base[~known], rtol=1e-6)
    np.testing.assert_allclose(data.score, raw["score_lead"] / 10.0, rtol=1e-6)


def test_gpu_decoder_and_augmentation_match_the_references(tmp_path):
    shard = _write_shards(tmp_path, 9, games=2)
    data = load_training_data([str(shard)])
    store = Store(data, torch.device("cpu"))
    index = torch.arange(len(data))
    batch = store.batch(index, torch.device("cpu"))
    np.testing.assert_array_equal(batch["features"].numpy(), unpack_features(data.planes, data.komi, 9))
    perms = symmetry_permutations(9)
    for sym in range(8):
        single = {k: v[:1].clone() for k, v in batch.items()}
        k, flip = divmod(sym, 2)
        expected_feats, expected_policy = apply_symmetry(single["features"][0], single["policy"][0], k, bool(flip))
        expected_own, _ = apply_symmetry(single["ownership"][0].reshape(1, 9, 9), torch.zeros(82), k, bool(flip))

        torch_randint = torch.randint
        try:
            torch.randint = lambda low, high, size, generator=None: torch.full(size, sym)
            augment(single, perms, torch.Generator())
        finally:
            torch.randint = torch_randint
        torch.testing.assert_close(single["features"][0], expected_feats)
        torch.testing.assert_close(single["policy"][0], expected_policy)
        torch.testing.assert_close(single["ownership"][0], expected_own.reshape(81))


def test_train_on_shards_with_aux_heads(tmp_path):
    shard = _write_shards(tmp_path, 9)
    out = tmp_path / "student.pt"
    train(TrainConfig(data=[str(shard)], out=str(out), epochs=2, batch_size=16, val_fraction=0.34,
                      device="cpu", amp=False))
    payload = torch.load(out, weights_only=False)
    assert payload["aux_state_dict"] is not None and payload["architecture"]["board_size"] == 9
    history = (tmp_path / "student.pt.history.json").read_text()
    assert '"ownership"' in history and '"score"' in history and "val_int8" in history


def test_warm_start_from_9x9_keeps_the_trunk_at_19x19(tmp_path):
    torch.manual_seed(0)
    source = MokaNet(preset("v1"))
    init = tmp_path / "v1-9.pt"
    save_checkpoint(source, str(init))
    shard = _write_shards(tmp_path, 19, games=3, moves=4)
    out = tmp_path / "student-19.pt"
    train(TrainConfig(data=[str(shard)], out=str(out), init_checkpoint=str(init), epochs=1, batch_size=8,
                      val_fraction=0.0, lr=0.0, device="cpu", amp=False))
    student = load_checkpoint(str(out), torch.device("cpu"))
    assert student.arch == preset("v1", 19)
    before, after = source.state_dict(), student.state_dict()
    for key, value in after.items():
        if is_trunk_key(key):
            torch.testing.assert_close(value, before[key])  # lr 0: the trunk is the 9×9 one
    assert after["policy_linear.weight"].shape == (362, 4 * 361)


def test_fixed_validation_set_and_kl_metrics(tmp_path):
    train_shard = _write_shards(tmp_path / "train", 9, games=4, seed=1)
    val_shard = _write_shards(tmp_path / "val", 9, games=2, seed=2)
    out = tmp_path / "student.pt"
    train(TrainConfig(data=[str(train_shard)], val_data=[str(val_shard)], out=str(out), epochs=1, batch_size=16,
                      device="cpu", amp=False))
    import json
    epoch = json.loads((tmp_path / "student.pt.history.json").read_text())[0]
    assert epoch["val_int8"]["soft"] >= epoch["val_int8"]["target_entropy"] - 1e-4  # KL >= 0
    assert epoch["val_fp"]["target_entropy"] > 0
