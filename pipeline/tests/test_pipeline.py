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
