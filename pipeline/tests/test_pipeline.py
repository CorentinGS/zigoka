import numpy as np
import torch

from go_pipeline.dataset import apply_symmetry
from go_pipeline.device import resolve_device
from go_pipeline.manifest import export_model, import_model, validate_model_dir
from go_pipeline.model import MokaNet


def test_model_shapes_and_param_count():
    model = MokaNet()
    assert model.parameter_count() == 105_353
    logits, value = model(torch.zeros(2, 12, 9, 9))
    assert logits.shape == (2, 82)
    assert value.shape == (2,)
    assert float(value.abs().max()) <= 1.0


def test_symmetry_keeps_pass_and_distribution():
    feats = torch.zeros(12, 9, 9)
    feats[0, 0, 0] = 1.0
    policy = torch.full((82,), 1.0 / 82)
    out_feats, out_policy = apply_symmetry(feats, policy, k=1, flip=True)
    assert out_feats[0].sum() == 1.0
    assert torch.allclose(out_policy.sum(), torch.tensor(1.0))
    assert out_policy[-1] == policy[-1]


def test_export_import_round_trip(tmp_path):
    torch.manual_seed(0)
    model = MokaNet().eval()
    manifest = export_model(model, tmp_path)
    assert manifest["weightsBytes"] > 100_000
    info = validate_model_dir(tmp_path)
    assert info["tensors"] == 120
    restored = import_model(tmp_path).eval()
    x = torch.randn(3, 12, 9, 9)
    with torch.no_grad():
        logits_a, value_a = model(x)
        logits_b, value_b = restored(x)
    # INT8 per-channel quantization drift should stay small for a random net.
    assert float((logits_a - logits_b).abs().max()) < 0.5
    assert float((value_a - value_b).abs().max()) < 0.2


def test_vendor_teacher_import(tmp_path):
    import pathlib

    vendor = pathlib.Path(__file__).resolve().parents[2] / "vendor" / "moka" / "model"
    if not (vendor / "go-model.bin").exists():
        return
    teacher = import_model(vendor).eval()
    assert teacher.parameter_count() == 105_353
    with torch.no_grad():
        logits, value = teacher(torch.zeros(1, 12, 9, 9))
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    # Re-exporting the published weights must stay close to the original net.
    export_model(teacher, tmp_path)
    restored = import_model(tmp_path).eval()
    x = torch.randn(2, 12, 9, 9)
    with torch.no_grad():
        a, _ = teacher(x)
        b, _ = restored(x)
    assert float((a - b).abs().max()) < 0.1


def test_device_auto_resolves():
    device = resolve_device("auto")
    assert device.type in ("cuda", "cpu")


def test_quantized_snapshot_matches_exported_artifact(tmp_path):
    from go_pipeline.manifest import quantized_snapshot

    torch.manual_seed(3)
    model = MokaNet().eval()
    export_model(model, tmp_path)
    restored = import_model(tmp_path).eval()
    snapshot = quantized_snapshot(model)
    x = torch.randn(4, 12, 9, 9)
    with torch.no_grad():
        logits_a, value_a = restored(x)
        logits_b, value_b = snapshot(x)
    assert float((logits_a - logits_b).abs().max()) < 1e-5
    assert float((value_a - value_b).abs().max()) < 1e-6


def test_validator_rejects_bad_format(tmp_path):
    import json
    import pathlib

    torch.manual_seed(0)
    export_model(MokaNet().eval(), tmp_path)
    manifest_path = pathlib.Path(tmp_path) / "go-model.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = "something-else"
    manifest_path.write_text(json.dumps(manifest))
    with __import__("pytest").raises(ValueError):
        validate_model_dir(tmp_path)


def test_architecture_presets_and_19x19_round_trip(tmp_path):
    import pathlib

    from go_pipeline.model import MOKA_V2, Architecture, preset

    root = pathlib.Path(__file__).resolve().parents[2]
    assert MokaNet(MOKA_V2).parameter_count() == 244_529
    v2 = import_model(root / "vendor" / "moka-v2" / "model")
    assert v2.arch == MOKA_V2
    arch = preset("v2", board_size=19)
    assert Architecture.from_manifest(arch.to_manifest()) == arch
    torch.manual_seed(1)
    model = MokaNet(arch).eval()
    manifest = export_model(model, tmp_path)
    assert manifest["architecture"]["boardSize"] == 19
    assert manifest["architecture"]["policyMoveCount"] == 362
    assert tuple(manifest["tensors"]["policy.linear.weight"]["shape"]) == (362, 4 * 361)
    assert validate_model_dir(tmp_path)["boardSize"] == 19
    restored = import_model(tmp_path).eval()
    assert restored.arch == arch
    logits, value = restored(torch.zeros(2, 12, 19, 19))
    assert logits.shape == (2, 362) and value.shape == (2,)


def test_manifest_architecture_validation_mirrors_zig():
    import pytest

    from go_pipeline.model import MOKA_V1, Architecture

    good = MOKA_V1.to_manifest()
    for key, value in [("policyMoveCount", 81), ("residualBlockKind", "standard"), ("inputPlaneCount", 13),
                       ("globalResidualHiddenChannelCount", 0), ("trunkChannelCount", 0)]:
        bad = dict(good, **{key: value})
        with pytest.raises(ValueError):
            Architecture.from_manifest(bad)
    with pytest.raises(ValueError):
        Architecture.from_manifest({k: v for k, v in good.items() if k != "boardSize"})


def test_legacy_checkpoint_loads_as_v1(tmp_path):
    from go_pipeline.model import MOKA_V1, load_checkpoint

    torch.manual_seed(2)
    model = MokaNet()
    torch.save({"state_dict": model.state_dict(), "meta": {}}, tmp_path / "old.pt")
    assert load_checkpoint(str(tmp_path / "old.pt"), torch.device("cpu")).arch == MOKA_V1


def test_sample_legal_move_rejects_illegal_and_respects_min_pass():
    from go_pipeline.board import GameState, sample_legal_move

    state = GameState(size=19)
    rng = np.random.default_rng(0)
    probs = np.zeros(362)
    probs[361] = 0.9  # pass, not allowed before move 89
    probs[0] = 0.1
    move, nxt = sample_legal_move(state, probs, rng)
    assert move == 0 and nxt.board[0] == 1
    state = GameState(size=19)
    state.board[1] = state.board[19] = -1  # point 0 is suicide for Black
    probs = np.zeros(362)
    probs[0], probs[40] = 0.99, 0.01
    move, _ = sample_legal_move(state, probs, rng)
    assert move == 40
