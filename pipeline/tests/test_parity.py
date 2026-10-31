"""Parity vs the vendored Moka runtime, using oracle fixtures when present.

Fixtures are generated data (gitignored); the test skips where they have
not been generated (`pnpm oracle` in the repo root) instead of passing
vacuously everywhere.
"""
import json
import pathlib

import numpy as np
import pytest
import torch

from go_pipeline.manifest import import_model

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "oracle" / "fixtures" / "moka-9" / "validation"


@pytest.mark.skipif(not (FIXTURES / "features.f32").exists(), reason="oracle fixtures not generated")
def test_pytorch_matches_moka_runtime_outputs():
    feats = np.fromfile(FIXTURES / "features.f32", dtype=np.float32)
    n = min(feats.size // (81 * 12), 256)
    x = feats[: n * 81 * 12].reshape(n, 81, 12).transpose(0, 2, 1).reshape(n, 12, 9, 9)
    model = import_model(ROOT / "vendor" / "moka" / "model").eval()
    with torch.no_grad():
        logits, value = model(torch.from_numpy(x))
    log_ref = np.fromfile(FIXTURES / "logits-normal.f32", dtype=np.float32)[: n * 82].reshape(n, 82)
    val_ref = np.fromfile(FIXTURES / "value-normal.f64", dtype=np.float64)[:n]
    assert float(np.abs(logits.numpy() - log_ref).max()) < 1e-4
    assert float(np.abs(value.numpy() - val_ref).max()) < 1e-5
    assert float((logits.numpy().argmax(1) == log_ref.argmax(1)).mean()) == 1.0
