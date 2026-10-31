"""go-export: checkpoint -> million-go-int8 artifact, validated twice:
by the strict Python mirror of the Zig checks, and — when a Zig toolchain
is available — by the repo's own tools/gen_model_layout.zig.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .device import resolve_device
from .manifest import export_model, validate_model_dir
from .model import load_checkpoint


def find_zig() -> str | None:
    found = shutil.which("zig")
    if found:
        return found
    for candidate in sorted(glob.glob(os.path.expanduser("~/.local/share/mise/installs/zig/*/bin/zig"))):
        return candidate
    return None


def validate_with_zig(model_dir: str | Path) -> str:
    zig = find_zig()
    root = Path(__file__).resolve().parents[3]
    tool = root / "tools" / "gen_model_layout.zig"
    if zig is None or not tool.exists():
        return "skipped (no zig toolchain found)"
    with tempfile.NamedTemporaryFile(suffix=".zig", delete=False) as tmp:
        tmp_path = tmp.name
    result = subprocess.run(
        [zig, "run", str(tool), "--", str(Path(model_dir) / "go-model.json"),
         str(Path(model_dir) / "go-model.bin"), tmp_path],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"zig validator refused the artifact: {result.stderr.strip()}")
    return f"accepted by {tool} ({zig})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a student checkpoint to the Moka artifact format.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, resolve_device(args.device))
    export_model(model, args.output_dir)
    info = validate_model_dir(args.output_dir)
    zig_status = validate_with_zig(args.output_dir)
    print(f"wrote {args.output_dir}: {info['tensors']} tensors, {info['weightsBytes']} bytes, sha256 {info['sha256']}")
    print(f"zig validation: {zig_status}")


if __name__ == "__main__":
    main()
