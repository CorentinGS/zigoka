"""PyTorch replica of the Moka student (see vendor/moka/src/runtime.ts).

12 planes -> stem -> nested-bottleneck blocks (trunk->bottleneck->
bottleneck->trunk, global mean/max adapters every `global_interval`
blocks) -> policy head (1×1 conv + linear to N²+1) and value head
(1×1 conv + hidden + tanh). The dimensions come from an `Architecture`,
the same fields as the manifest's `architecture` object: Moka v1 is
32/16/12 on 9×9, Moka v2 48/24/16. Input is channel-first [B, 12, N, N];
the vendored runtime consumes the same values position-major, the layout
conversion lives in manifest.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from . import INPUT_PLANES

BLOCK_KIND = "nested-bottleneck"


@dataclass(frozen=True)
class Architecture:
    board_size: int
    trunk: int
    bottleneck: int
    blocks: int
    global_interval: int
    global_hidden: int
    policy_channels: int
    value_channels: int
    value_hidden: int
    input_planes: int = INPUT_PLANES

    @property
    def area(self) -> int:
        return self.board_size * self.board_size

    @property
    def policy_moves(self) -> int:
        return self.area + 1

    def has_global(self, block: int) -> bool:
        return self.global_interval > 0 and (block + 1) % self.global_interval == 0

    def with_board_size(self, size: int) -> "Architecture":
        return replace(self, board_size=size)

    def to_manifest(self) -> dict:
        return {
            "boardSize": self.board_size,
            "bottleneckChannelCount": self.bottleneck,
            "globalResidualBlockInterval": self.global_interval,
            "globalResidualHiddenChannelCount": self.global_hidden,
            "inputPlaneCount": self.input_planes,
            "policyChannelCount": self.policy_channels,
            "policyMoveCount": self.policy_moves,
            "residualBlockCount": self.blocks,
            "residualBlockKind": BLOCK_KIND,
            "scoreHiddenChannelCount": self.value_hidden,
            "trunkChannelCount": self.trunk,
            "valueChannelCount": self.value_channels,
        }

    @classmethod
    def from_manifest(cls, architecture: dict) -> "Architecture":
        """Validates like tools/gen_model_layout.zig (`parseConfig`)."""
        expected = set(cls(9, 1, 1, 1, 0, 0, 1, 1, 1).to_manifest())
        if set(architecture) != expected:
            raise ValueError(f"architecture fields differ: missing={expected - set(architecture)} "
                             f"extra={set(architecture) - expected}")
        if architecture["residualBlockKind"] != BLOCK_KIND:
            raise ValueError(f"residualBlockKind must be {BLOCK_KIND}, found {architecture['residualBlockKind']!r}")
        integers = {k: v for k, v in architecture.items() if k != "residualBlockKind"}
        for key, value in integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"architecture.{key} must be a non-negative integer, found {value!r}")
        arch = cls(
            board_size=architecture["boardSize"],
            trunk=architecture["trunkChannelCount"],
            bottleneck=architecture["bottleneckChannelCount"],
            blocks=architecture["residualBlockCount"],
            global_interval=architecture["globalResidualBlockInterval"],
            global_hidden=architecture["globalResidualHiddenChannelCount"],
            policy_channels=architecture["policyChannelCount"],
            value_channels=architecture["valueChannelCount"],
            value_hidden=architecture["scoreHiddenChannelCount"],
            input_planes=architecture["inputPlaneCount"],
        )
        arch.validate()
        if architecture["policyMoveCount"] != arch.policy_moves:
            raise ValueError(f"architecture.policyMoveCount must be boardSize^2 + 1 = {arch.policy_moves}, "
                             f"found {architecture['policyMoveCount']}")
        return arch

    def validate(self) -> None:
        if self.input_planes != INPUT_PLANES:
            raise ValueError(f"inputPlaneCount must be {INPUT_PLANES}, found {self.input_planes}")
        positive = {"boardSize": self.board_size, "trunkChannelCount": self.trunk,
                    "bottleneckChannelCount": self.bottleneck, "residualBlockCount": self.blocks,
                    "policyChannelCount": self.policy_channels, "valueChannelCount": self.value_channels,
                    "scoreHiddenChannelCount": self.value_hidden}
        for key, value in positive.items():
            if value <= 0:
                raise ValueError(f"architecture.{key} must be positive, found {value}")
        if (self.global_interval == 0) != (self.global_hidden == 0):
            raise ValueError("globalResidualBlockInterval and globalResidualHiddenChannelCount "
                             "must both be zero or both be positive")


MOKA_V1 = Architecture(board_size=9, trunk=32, bottleneck=16, blocks=12, global_interval=4,
                       global_hidden=8, policy_channels=4, value_channels=2, value_hidden=32)
MOKA_V2 = Architecture(board_size=9, trunk=48, bottleneck=24, blocks=16, global_interval=4,
                       global_hidden=8, policy_channels=4, value_channels=2, value_hidden=32)
PRESETS = {"v1": MOKA_V1, "v2": MOKA_V2}


def preset(name: str, board_size: int | None = None) -> Architecture:
    if name not in PRESETS:
        raise ValueError(f"unknown architecture preset {name!r}; known: {', '.join(PRESETS)}")
    arch = PRESETS[name]
    return arch.with_board_size(board_size) if board_size is not None else arch


class Block(nn.Module):
    def __init__(self, arch: Architecture, global_adapter: bool) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(arch.trunk, arch.bottleneck, 1)
        self.first = nn.Conv2d(arch.bottleneck, arch.bottleneck, 3, padding=1)
        self.second = nn.Conv2d(arch.bottleneck, arch.bottleneck, 3, padding=1)
        self.expand = nn.Conv2d(arch.bottleneck, arch.trunk, 1)
        self.global_hidden = nn.Linear(2 * arch.bottleneck, arch.global_hidden) if global_adapter else None
        self.global_output = nn.Linear(arch.global_hidden, arch.bottleneck) if global_adapter else None

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.first(torch.relu(self.reduce(trunk))))
        if self.global_hidden is not None and self.global_output is not None:
            pooled = torch.cat([hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))], dim=1)
            bias = self.global_output(torch.relu(self.global_hidden(pooled)))
            hidden = hidden + bias[:, :, None, None]
        residual = self.expand(torch.relu(self.second(hidden)))
        return torch.relu(trunk + residual)


class MokaNet(nn.Module):
    def __init__(self, arch: Architecture = MOKA_V1) -> None:
        super().__init__()
        arch.validate()
        self.arch = arch
        self.stem = nn.Conv2d(arch.input_planes, arch.trunk, 3, padding=1)
        self.blocks = nn.ModuleList(Block(arch, global_adapter=arch.has_global(i)) for i in range(arch.blocks))
        self.policy_convolution = nn.Conv2d(arch.trunk, arch.policy_channels, 1)
        self.policy_linear = nn.Linear(arch.policy_channels * arch.area, arch.policy_moves)
        self.value_convolution = nn.Conv2d(arch.trunk, arch.value_channels, 1)
        self.value_hidden = nn.Linear(arch.value_channels * arch.area, arch.value_hidden)
        self.value_output = nn.Linear(arch.value_hidden, 1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        trunk = torch.relu(self.stem(x))
        for block in self.blocks:
            trunk = block(trunk)
        return trunk

    def heads(self, trunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The vendored runtime flattens position-major (move * channels + ch),
        # so the linear heads must see that order, not torch's channel-major.
        policy = self.policy_linear(torch.relu(self.policy_convolution(trunk)).permute(0, 2, 3, 1).flatten(1))
        value_feats = torch.relu(self.value_convolution(trunk)).permute(0, 2, 3, 1).flatten(1)
        value = torch.tanh(self.value_output(torch.relu(self.value_hidden(value_feats)))).squeeze(1)
        return policy, value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads(self.trunk(x))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def manifest_items(self):
        """(manifest tensor name, torch tensor) pairs, owned by the model."""
        state = self.state_dict()
        for manifest_name, state_key in manifest_map(self.arch):
            yield manifest_name, state[state_key]

    def load_manifest_items(self, items: dict) -> None:
        by_key = {state_key: items[manifest_name] for manifest_name, state_key in manifest_map(self.arch)}
        self.load_state_dict(by_key)


def manifest_map(arch: Architecture) -> list[tuple[str, str]]:
    """(manifest tensor name, state_dict key), in the order of the artifact."""
    pairs: list[tuple[str, str]] = [("stem.weight", "stem.weight"), ("stem.bias", "stem.bias")]
    for i in range(arch.blocks):
        manifest_prefix, torch_prefix = f"residual.{i}", f"blocks.{i}"
        for suffix in ("reduce", "first"):
            pairs += [(f"{manifest_prefix}.{suffix}.weight", f"{torch_prefix}.{suffix}.weight"),
                      (f"{manifest_prefix}.{suffix}.bias", f"{torch_prefix}.{suffix}.bias")]
        if arch.has_global(i):
            pairs += [(f"{manifest_prefix}.global.hidden.weight", f"{torch_prefix}.global_hidden.weight"),
                      (f"{manifest_prefix}.global.hidden.bias", f"{torch_prefix}.global_hidden.bias"),
                      (f"{manifest_prefix}.global.output.weight", f"{torch_prefix}.global_output.weight"),
                      (f"{manifest_prefix}.global.output.bias", f"{torch_prefix}.global_output.bias")]
        for suffix in ("second", "expand"):
            pairs += [(f"{manifest_prefix}.{suffix}.weight", f"{torch_prefix}.{suffix}.weight"),
                      (f"{manifest_prefix}.{suffix}.bias", f"{torch_prefix}.{suffix}.bias")]
    pairs += [
        ("policy.convolution.weight", "policy_convolution.weight"),
        ("policy.convolution.bias", "policy_convolution.bias"),
        ("policy.linear.weight", "policy_linear.weight"),
        ("policy.linear.bias", "policy_linear.bias"),
        ("value.convolution.weight", "value_convolution.weight"),
        ("value.convolution.bias", "value_convolution.bias"),
        ("value.hidden.weight", "value_hidden.weight"),
        ("value.hidden.bias", "value_hidden.bias"),
        ("value.output.weight", "value_output.weight"),
        ("value.output.bias", "value_output.bias"),
    ]
    return pairs


def is_trunk_key(state_key: str) -> bool:
    """Trunk tensors do not depend on the board size; head tensors do."""
    return state_key.startswith(("stem.", "blocks."))


def save_checkpoint(model: MokaNet, path: str, meta: dict | None = None, extra: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "architecture": asdict(model.arch), "meta": meta or {}}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def read_checkpoint(path: str, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(payload, dict) and "state_dict" in payload):
        payload = {"state_dict": payload}
    return payload


def load_checkpoint(path: str, device: torch.device) -> MokaNet:
    """Checkpoints without an architecture predate it and are Moka v1."""
    payload = read_checkpoint(path, device)
    arch = Architecture(**payload["architecture"]) if "architecture" in payload else MOKA_V1
    model = MokaNet(arch)
    model.load_state_dict(payload["state_dict"])
    return model.to(device)
