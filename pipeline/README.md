# Distillation pipeline, 9×9 and 19×19

Generate → train → export for a Moka-format student, in PyTorch, with the
same artifact contract as the rest of this repo (`million-go-int8`,
`go-model.json` + `go-model.bin`, validated by `tools/gen_model_layout.zig`).
It mirrors upstream Moka's `go-generate` / `go-train` / `go-export` stages,
which are not vendored in this repo.

## Setup

The environment lives in `pipeline/.venv` (managed by `uv`).

NVIDIA GPU (Linux x86_64 — the default PyPI torch wheel bundles CUDA):

```sh
cd pipeline
uv sync --extra dev
uv run python -m go_pipeline.device   # must print your GPU name
```

AMD GPU with ROCm (checked on a Radeon RX 6800/6900 XT, gfx1030, with
torch 2.14.1+rocm7.2: `torch.cuda` works through HIP, fp16 and bf16
autocast included). The wheel bundles the ROCm runtime; no system ROCm
install is needed beyond the amdgpu kernel driver:

```sh
cd pipeline
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install --python .venv/bin/python -e ".[dev]"
uv run --no-sync python -m go_pipeline.device   # must print the Radeon
```

CPU-only (e.g. this dev machine, which has no NVIDIA GPU):

```sh
cd pipeline
uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Run

```sh
# 1. Teacher-labelled positions. Teachers: --katago <model.onnx> for fresh
#    KataGo distillation (ONNX Runtime, CUDA provider on NVIDIA; ladder
#    planes 14-17 are zero, see katago.py), --teacher-dir for any published
#    Moka artifact (bootstrap), --teacher for a .pt checkpoint.
#    --student switches rollouts to DAgger (upstream Moka's on-policy
#    recipe: student distribution, teacher labels, 0.15 intervention).
uv run go-generate --katago teachers/katago-b6c96.onnx --positions 50000 \
  --output data/distillation.npz --device auto

# 2. Train. --device auto picks CUDA when present (AMP bf16/fp16, TF32,
#    pinned loaders); deterministic by default (--non-deterministic for
#    speed). Checkpoint selection scores the INT8-quantized snapshot, i.e.
#    the shipped artifact, not the float weights. Warm start with
#    --init-dir or --init-checkpoint.
uv run go-train --data data/distillation.npz --out checkpoints/student.pt \
  --epochs 30 --batch-size 256 --device auto

# 3. Export the exact INT8 artifact. Validation is strict (Python mirror
#    of the Zig checks) and, when a Zig toolchain is found on PATH or via
#    mise, go-export also runs tools/gen_model_layout.zig on the result.
uv run go-export --checkpoint checkpoints/student.pt --output-dir ../models/moka-9-v2
```

Dataset format (`.npz`): `features [N,12,9,9] f32`, `policy [N,82] f32`
probabilities, `value [N] f32` in [-1, 1], `game_id [N]` — training splits
by whole game (and refuses single-game or empty splits). Rollout
temperature is applied exactly once, at sampling time.

## 19×19: KataGo self-play, shards, training

The teacher is a native KataGo network driven through `katago analysis`
(`analysis.py`), which computes its own inputs, ladders included. On AMD,
`katago/katago-rocm` runs KataGo's ROCm build with the ROCm libraries of
the torch wheel (unzip the `katago-v1.18.1-rocm7.2.4-linux-x64` release
into `~/.cache/zigoka-katago/rocm/`, or point `KATAGO_ROCM` at it).

```sh
# KataGo plays itself under Moka's rules (no pass before move 89, 540
# moves at most); every position keeps the raw policy and value, a quarter
# of them a 64-visit search. Compact shards: about 2.4 KB per position.
uv run go-selfplay --katago katago/katago-rocm \
  --model ~/.cache/zigoka-katago/nets/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz \
  --out data/selfplay-19 --board-size 19 --positions 1000000

# Train on every shard of the directory. --arch v2 at the shards' board
# size, or a warm start: a 9×9 checkpoint or model keeps its trunk and
# gets new 19×19 heads.
uv run go-train --data data/selfplay-19 --arch v2 --epochs 20 --out checkpoints/moka-19.pt
uv run go-export --checkpoint checkpoints/moka-19.pt --output-dir ../models/moka-19
```

Measured on a 16 GB gfx1030 Radeon (RX 6800 / 6900 XT family): self-play
about 97 positions/s (KataGo ROCm, b18c384nbt; the OpenCL build is 4×
slower), training Moka v2 at 19×19 about 4 000 positions/s in bf16. The
shards' format is in `shards.py`, the targets the trainer derives from
them in `dataset.load_training_data`.

To serve an exported model from this repo, add one entry for its directory
in `MODELS` in the root `build.zig` (next to `moka-9`), then run the usual
`pnpm oracle` / `zig build test` flow for that model id.

## Fidelity

`MokaNet` is a PyTorch replica of `vendor/moka/src/runtime.ts`, including
the position-major flattening of the policy/value heads.
`tests/test_parity.py` checks it against the vendored runtime's reference
outputs (256 validation positions; measured max |Δlogit| 3.1e-05, max
|Δvalue| 9.6e-07, top-move agreement 100%) whenever the generated oracle
fixtures are present, and skips otherwise. Re-exporting the published
weights reproduces a 113,648-byte artifact that `go-export` validates
with `tools/gen_model_layout.zig` itself.

## Tests

```sh
uv run pytest   # board, KataGo encoding, symmetry, artifact selection,
                # strict validation, split edges, train smokes, parity,
                # oracle parity of every model, KataGo analysis responses,
                # self-play against a scripted engine
```

Not verified here: actual CUDA execution on NVIDIA (only the ROCm path
ran on a GPU). The CUDA path is the standard PyTorch one —
`resolve_device("cuda")` fails loudly if torch cannot see a GPU rather
than silently using CPU.
